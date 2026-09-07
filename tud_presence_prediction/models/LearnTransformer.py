import torch
import math
import pytorch_lightning as pl
import torch.nn as nn
from torch.nn import MultiheadAttention
from torchmetrics.classification import BinaryAccuracy, BinaryRecall, BinaryF1Score, BinaryPrecision

from .internal.Time2Vec import Time2Vec
from .internal.model_util import model_util
from flash_attn.modules.mha import MHA as FlashMHA

class LearnTransformerEncoderBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout, use_flash_attention):
        super(LearnTransformerEncoderBlock, self).__init__()

        feature_dim = hidden_dim * 2  

        if use_flash_attention:
            self.mha = FlashMHA(hidden_dim*2, num_heads=num_heads, dropout=dropout, causal=False, cross_attn=False)
            self._use_flash = True
        else:
            self.mha = MultiheadAttention(embed_dim=hidden_dim*2, num_heads=num_heads, dropout=dropout, batch_first=True)
            self._use_flash = False

        self.norm1 = nn.LayerNorm(hidden_dim*2)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim*2, 4 * hidden_dim*2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim*2, hidden_dim*2),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(hidden_dim*2)

    def forward(self, x_embedded_combined):
        # Handle both cases: with and without batch dimension
        if len(x_embedded_combined.shape) == 4:
            batch_size, days, timeslots, features = x_embedded_combined.shape
            x_embedded_combined = x_embedded_combined.reshape(batch_size, days * timeslots, features)
            # restore original shape after processing
            output_shape = (batch_size, days, timeslots, features)
        else:
            days, timeslots, features = x_embedded_combined.shape
            x_embedded_combined = x_embedded_combined.reshape(1, days * timeslots, features)
            # restore original shape after processing
            output_shape = (days, timeslots, features)

        if self._use_flash:
            # First sublayer: MHA then Add & Norm
            x = self.norm1(x_embedded_combined + self.mha(x_embedded_combined))
            # Second sublayer: FFN then Add & Norm
            x = self.norm2(x + self.ff(x))
        else:
            # First sublayer: MHA then Add & Norm
            attn_output, _ = self.mha(x_embedded_combined, x_embedded_combined, x_embedded_combined)
            x = self.norm1(x_embedded_combined + attn_output)
            # Second sublayer: FFN then Add & Norm
            x = self.norm2(x + self.ff(x))
        
        # restore the original shape
        x = x.reshape(output_shape)
        return x


class LearnTransformerDecoderBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout, use_flash_attention):
        super(LearnTransformerDecoderBlock, self).__init__()

        feature_dim = hidden_dim * 2
        
        if use_flash_attention:
            self.self_mha = FlashMHA(feature_dim, num_heads=num_heads, dropout=dropout, causal=True, cross_attn=False)
            self.cross_mha = FlashMHA(feature_dim, num_heads=num_heads, dropout=dropout, causal=False, cross_attn=True)
            self._use_flash = True
        else:
            self.self_mha = MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
            self.cross_mha = MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
            self._use_flash = False
            
        self.norm1 = nn.LayerNorm(hidden_dim*2)
        self.norm2 = nn.LayerNorm(hidden_dim*2)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim*2, 4 * hidden_dim*2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim*2, hidden_dim*2),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(hidden_dim*2)

    def forward(self, target, encoder_output):
        if self._use_flash:            
            # Flash Attention API
            # Self-Attention
            x = self.norm1(target + self.self_mha(target))
            # Cross-Attention
            # print(f"shape of x: {x.shape}, \n shape of encoder_output: {encoder_output.shape}")
            cross_output = self.cross_mha(x, encoder_output)
            x = self.norm2(x + cross_output)
        else:
            # Standard PyTorch Attention API
            # Self-Attention
            T = target.size(1)
            mask = torch.ones(T,T, device=target.device, dtype=torch.bool).triu()
            attn_output, _ = self.self_mha(target, target, target, is_causal=True, attn_mask=mask)
            x = self.norm1(target + attn_output)
            # Cross-Attention
            cross_output, _ = self.cross_mha(x, encoder_output, encoder_output)
            x = self.norm2(x + cross_output)
            
        # Feedforward (gleich fuer beide Implementierungen)
        x = self.norm3(x + self.ff(x))
        return x


class LearnTransformer(pl.LightningModule):
    """
    A Transformer-based model for presence prediction using PyTorch Lightning.
    This model includes an encoder-decoder architecture with multi-head attention and feedforward networks.
    """
    def __init__(
        self, 
        location_dim, 
        calendar_dim, input_dim_3=None, hidden_dim=16, num_heads=4, dropout=0.4, num_of_layers=3,
        use_optimizer="AdamW",
        optimizer_beta1=None, 
        optimizer_beta2=None,
        weight_decay=None,
        fixed_learning_rate=0.001,
        log_LR=True,
        use_LR_scheduler=True,
        LR_scheduler_min=3e-6,
        LR_scheduler_max=3e-4,
        LR_scheduler_convergence=0.000016,
        LR_increase_phase_percentage=0.05,
        step_size=1,
        gradient_clip_val=1.0,
        use_flash_attention=True,
        num_days=None): 

        super(LearnTransformer, self).__init__()

        self.prediction_total_length = None
        self.prediction_step_length = (0, step_size)
        self.step_size = step_size
        self.use_flash_attention = use_flash_attention  # Speichern der Flash-Attention-Einstellung
        self.num_days = num_days

        self.training_step_outputs = []   
        self.training_step_targets = []       
        self.validation_step_targets = [] 

        self.use_optimizer = use_optimizer
        self.optimizer_beta1 = optimizer_beta1 
        self.optimizer_beta2 = optimizer_beta2
        self.weight_decay = weight_decay
        self.fixed_learning_rate = fixed_learning_rate
        self.log_LR = log_LR
        self.use_LR_scheduler = use_LR_scheduler
        self.LR_scheduler_min = LR_scheduler_min
        self.LR_scheduler_max = LR_scheduler_max 
        self.LR_scheduler_convergence = LR_scheduler_convergence
        self.LR_increase_phase_percentage = LR_increase_phase_percentage
        self.gradient_clip_val = gradient_clip_val
        self.visual_metrics = {}

        model_util.configure_logging(self, log_lr=self.log_LR)
        self.save_hyperparameters()
        
        self.location_dim = location_dim
        self.calendar_dim = calendar_dim
        self.hidden_dim = hidden_dim

        if step_size > 1:
            self.step_mode = "step"
        elif step_size == 1:
            self.step_mode = "single"
        else:
            raise ValueError("Ungueltige step_size wurde angegeben: " + str(step_size))

        self.decoder_start_token = nn.Parameter(torch.full((1, step_size, 1), 0.6))  
            
        # Input embeddings
        self.time2vec = Time2Vec(location_dim, hidden_dim) # output of time2vec is hidden_dim*2
        self.calendar_embedding = nn.Linear(calendar_dim, hidden_dim*2)

        self.token_embedding = nn.Linear(1, hidden_dim*2)
        self.target_embedding = nn.Linear(1, hidden_dim*2)  # New embedding for target sequence
        
        # Encoder and decoder blocks
        self.encoder_blocks = nn.ModuleList([
            LearnTransformerEncoderBlock(hidden_dim, num_heads, dropout, use_flash_attention=use_flash_attention)
            for _ in range(num_of_layers)  # encoder layers
        ])

        self.decoder_blocks = nn.ModuleList([
            LearnTransformerDecoderBlock(hidden_dim, num_heads, dropout, use_flash_attention=use_flash_attention)
            for _ in range(num_of_layers)  # decoder layers
        ])
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim*2, 1)       
        self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([4])) # pos_weight should be num_0/num_1, where num_0 is the number of 0s and num_1 is the number of 1s in the target data

        self.val_acc = BinaryAccuracy(threshold=0.5)
        self.val_recall = BinaryRecall(threshold=0.5)
        self.val_f1 = BinaryF1Score(threshold=0.5)
        self.val_precision = BinaryPrecision(threshold=0.5)

        self.val_acc_open = BinaryAccuracy(threshold=0.5)
        self.val_recall_open = BinaryRecall(threshold=0.5)
        self.val_f1_open = BinaryF1Score(threshold=0.5)
        self.val_precision_open = BinaryPrecision(threshold=0.5)


    def forward(self, x_location, x_calendar, target=None):
        """
        Forward pass for both training/validation and inference.

        Expected shapes:
          - x_location: (B, D_hist, T, location_dim)  -> history only (for the encoder)
          - x_calendar: (B, D_hist, T, calendar_dim)  -> history only (for the encoder)
          - target    : (B, D_fut,  T)                -> future (for decoder/loss), optional during inference

        Behavior:
          - If `target` is provided (training/validation), we do teacher forcing over the entire future.
          - If `target` is None (inference), we generate autoregressively for `self.num_days`
            (or fall back to the input length if `self.num_days` is None).
        """
        # --- Embed history (encoder input) ---
        x_loc = self.time2vec(x_location)            # (B, D_hist, T, 2*hidden)
        x_cal = self.calendar_embedding(x_calendar)  # (B, D_hist, T, 2*hidden)
        x_embed = x_loc + x_cal                      # (B, D_hist, T, 2*hidden)

        # Basic safety: history must not be empty when training/validating
        total_days = x_embed.size(1)
        if (self.training or target is not None) and total_days <= 0:
            raise ValueError("History must not be empty (total_days == 0).")

        # --- Encoder over the full history ---
        encoder_output = x_embed
        for block in self.encoder_blocks:
            encoder_output = block(encoder_output)

        if self.training or target is not None:
            # TRAIN/VALIDATION: target already *is* the future we supervise on
            # target: (B, D_fut, T)
            decoder_output = self._decode_step(encoder_output=encoder_output, target=target)  # (B, D_fut, T, 2*hidden)

            # Project to 1 logit per timeslot and return as (B, D_fut, T)
            prediction = self.output_projection(decoder_output).squeeze(-1)  # (B, D_fut, T)
            return prediction

        else:
            # INFERENCE: autoregressively produce `num_days` (or default to history length)
            pred_len_days = int(self.num_days) if (self.num_days is not None) else x_location.size(1)
            self.prediction_total_length = pred_len_days

            decoder_output = self._decode_step(encoder_output=encoder_output, target=None)  # (B, D_pred, T, 2*hidden)
            predictions = self.output_projection(decoder_output).squeeze(-1)               # (B, D_pred, T)
            return predictions

    def _decode_step(self, encoder_output, target=None):
        """Process target sequence through decoder blocks with encoder context.

        Args:
            target: Target sequence of presence data for training
                   Shape: (batch_size, days, timeslots)
            encoder_output: Encoded representation from encoder blocks.
                   Shape: (batch_size, days, timeslots, hidden_dim*2)

        Returns:
           decoder_output: Processed sequence through decoder blocks.
           Shape: (batch_size, days, timeslots, hidden_dim*2)
        """

        # Reshape encoder output to match decoder input
        if len(encoder_output.shape) == 4:
            batch_size, days, timeslots, features = encoder_output.shape
            encoder_output = encoder_output.reshape(batch_size, days * timeslots, features)
        else:
            days, timeslots, features = encoder_output.shape
            batch_size = 1
            encoder_output = encoder_output.reshape(batch_size, days * timeslots, features)
            
        if target is not None:
            # Training mode
            batch_size, days, timeslots = target.shape
            target_length = days * timeslots
            target = target.reshape(batch_size, target_length, 1)
            
            # Korrekte Anwendung des Token-Embeddings und Expansion
            start_token = self.token_embedding(self.decoder_start_token)
            # Erweitern auf die batch_size, jedoch Beibehaltung der step_size-Dimension
            start_token = start_token.expand(batch_size, self.step_size, -1)

            if self.step_size > target_length:
                raise ValueError(f"Step size {self.step_size} cannot be larger than target length {target_length}, increase amount of training data to use step_mode.")

            # Decoder-Eingabe initialisieren
            decoder_input = self.target_embedding(target[:, :-self.step_size, :])  # Shape: (batch_size, target_length - step_size, hidden_dim*2)
            
            # Start-Token hinzufuegen
            decoder_input = torch.cat([start_token, decoder_input], dim=1)
            
            # Durch Decoder-Bloecke verarbeiten
            decoder_output = decoder_input
            for block in self.decoder_blocks:
                decoder_output = block(decoder_output, encoder_output)
            
            decoder_output = decoder_output.reshape(batch_size, days, timeslots, -1)
            return decoder_output

        else:
            # For inference, we start with the decoder start token
            decoder_input = self.token_embedding(self.decoder_start_token)
            decoder_input = decoder_input.expand(batch_size, self.step_size, -1).clone() 

            # Wir verwenden eine separate Variable, um alle generierten Tokens zu speichern
            all_tokens = decoder_input.clone()
            
            total_slots = self.prediction_total_length * timeslots

            if self.step_mode == "single":
                # Generate predictions one step at a time
                for _ in range(total_slots):
                    # Verwende alle bisher generierten Tokens für die Kontextberechnung
                    current_output = all_tokens
                    for block in self.decoder_blocks:
                        current_output = block(current_output, encoder_output)

                    next_token = current_output[:, -1:, :]  # Nur das letzte Token nehmen
                    all_tokens = torch.cat([all_tokens, next_token], dim=1)  # An Sequenz anhaengen
                    
            # Generate with larger step size
            elif self.step_mode == "step":
                # Calculate how many steps we need to take
                num_steps = total_slots // self.step_size if self.step_size < total_slots else 1
                rem_steps = total_slots % self.step_size
                
                # For each step, we process all tokens generated so far
                for _ in range(num_steps):
                    current_output = all_tokens
                    for block in self.decoder_blocks:
                        current_output = block(current_output, encoder_output)
                        
                    next_tokens = current_output[:, -self.step_size:, :]  # Nur die letzten tokens nehmen
                    all_tokens = torch.cat([all_tokens, next_tokens], dim=1)  # An Sequenz anhaengen

                # Handle remaining steps if any
                if rem_steps > 0:
                    current_output = all_tokens
                    for block in self.decoder_blocks:
                        current_output = block(current_output, encoder_output)
                        
                    next_tokens = current_output[:, -rem_steps:, :]  # Nur die verbleibenden tokens
                    all_tokens = torch.cat([all_tokens, next_tokens], dim=1)  # An Sequenz anhaengen

            # Entferne Start-Token und forme um fuer Ausgabe
            decoder_output = all_tokens[:, self.step_size:, :]  # Start-Token entfernen
            decoder_output = decoder_output.reshape(batch_size, self.prediction_total_length, timeslots, -1)
            return decoder_output


    def configure_optimizers(self):
        # Base-LR konsistent zum OneCycle-Start: max_lr / div_factor
        base_max = self.LR_scheduler_max if self.LR_scheduler_max else (self.fixed_learning_rate or 1e-3)
        base_lr  = (base_max / 25.0) if self.use_LR_scheduler else (self.fixed_learning_rate or 1e-3)

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=base_lr,
            weight_decay=(self.weight_decay if self.weight_decay is not None else 1e-2),
        )

        if not self.use_LR_scheduler:
            return optimizer

        # sicheres total_steps bestimmen
        ts = int(getattr(self.trainer, "estimated_stepping_batches", 0))

        # Fallback: zu wenige Schritte -> kein OneCycle (verhindert ZeroDivision)
        if ts < 2:
            print(f"[WARN] OneCycleLR deaktiviert (estimated_stepping_batches={ts})")
            return optimizer

        # pct_start robust machen: mind. 1 Warmup-Schritt, mind. 1 Anneal-Schritt
        pct = self.LR_increase_phase_percentage or 0.05
        warmup_steps = max(1, int(math.ceil(ts * max(pct, 1.0 / ts + 1e-8))))
        if warmup_steps >= ts:
            warmup_steps = ts - 1
        pct_safe = warmup_steps / ts

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.LR_scheduler_max or 0.001,
            total_steps=ts,
            pct_start=pct_safe,
            div_factor=25,
            final_div_factor=1000,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def training_step(self, batch, batch_idx):
        x_location, x_calendar, target = batch         

        B, D, T, _ = x_location.shape
        if self.num_days is None:
            print("num_days konnte nicht verarbeitet werden, letzte 3 Tage werden jeweils vorhergesagt")
            num_days = 3
        else:
            num_days = int(self.num_days)

        # Split Index: Vergangenheit / Zukunft
        split_idx = D - num_days

        # Vergangenheit fuer Encoder
        x_loc_hist = x_location[:, :split_idx, :, :]
        x_cal_hist = x_calendar[:, :split_idx, :, :]

        # Zukunft fuer Decoder/Loss
        target_future = target[:, split_idx:, :]      # Shape: [B, num_days, T]

        # Forward pass (Vergangenheit → Encoder, Zukunft → Decoder)
        pred = self(x_loc_hist, x_cal_hist, target_future)

        # Handle NaN predictions
        if torch.isnan(pred).any():
            print(f"NaN in predictions at batch {batch_idx}")

        # Ensure shapes match
        if pred.shape != target_future.shape:
            pred = pred.view(target_future.shape)

        # Masking
        mask = ~torch.isnan(target_future)
        if not mask.any():
            print(f"No valid samples in batch {batch_idx}")

        filtered_pred = pred[mask]
        filtered_target = target_future[mask]  

        # Loss berechnen
        try:
            loss = self.loss(filtered_pred, filtered_target)
            if not torch.isfinite(loss):
                print(f"Non-finite loss value: {loss}")
                return torch.tensor(float('inf'), requires_grad=True)
        except Exception as e:
            print(f"Error in loss calculation: {e}")
            return torch.tensor(float('inf'), requires_grad=True)

        binary_pred = (torch.sigmoid(filtered_pred) > 0.5).float()
        accuracy = (binary_pred == filtered_target).float().mean()

        self.training_step_outputs.append(loss)

        self.log("train_loss_step", loss, on_step=True, on_epoch=False, prog_bar=True)

        return loss


    def validation_step(self, batch, batch_idx):
        x_location, x_calendar, target = batch
        B, D, T, _ = x_location.shape
        num_days = int(self.num_days) if self.num_days is not None else 3
        split_idx = D - num_days

        x_loc_hist = x_location[:, :split_idx, :, :]
        x_cal_hist = x_calendar[:, :split_idx, :, :]
        target_future = target[:, split_idx:, :]  # [B, num_days, T]

        # ----- (1) Teacher-forced path (as before, used for loss & legacy metrics) -----
        with torch.no_grad():
            pred_tf = self(x_loc_hist, x_cal_hist, target_future)
        if pred_tf.shape != target_future.shape:
            pred_tf = pred_tf.view(target_future.shape)

        mask = ~torch.isnan(target_future)
        filtered_pred_tf = pred_tf[mask]
        filtered_target = target_future[mask]

        loss = self.loss(filtered_pred_tf, filtered_target)  # keep loss on TF path

        probs_tf = torch.sigmoid(filtered_pred_tf).detach()
        t = filtered_target.detach().float()
        self.val_acc.update(probs_tf, t)
        self.val_recall.update(probs_tf, t)
        self.val_f1.update(probs_tf, t)
        self.val_precision.update(probs_tf, t)

        self.log("val_loss_epoch",       loss,               on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_accuracy_epoch",   self.val_acc,       on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_recall_epoch",     self.val_recall,    on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_f1_epoch",         self.val_f1,        on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_precision_epoch",  self.val_precision, on_step=False, on_epoch=True, prog_bar=False)

        # ----- (2) Open-loop path (no teacher forcing) -> realistic metrics -----
        with torch.no_grad():
            pred_open = self(x_loc_hist, x_cal_hist)  # IMPORTANT: no target
        if pred_open.shape != target_future.shape:
            pred_open = pred_open.view(target_future.shape)

        filtered_pred_open = pred_open[mask]
        probs_open = torch.sigmoid(filtered_pred_open).detach()

        self.val_acc_open.update(probs_open, t)
        self.val_recall_open.update(probs_open, t)
        self.val_f1_open.update(probs_open, t)
        self.val_precision_open.update(probs_open, t)

        self.log("val_accuracy_open_epoch",   self.val_acc_open,       on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_recall_open_epoch",     self.val_recall_open,    on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_f1_open_epoch",         self.val_f1_open,        on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_precision_open_epoch",  self.val_precision_open, on_step=False, on_epoch=True, prog_bar=False)

        return loss


    def supports_variable_length(self):
        """Tell presence_prediction that this is a regressive model"""
        return True

    def get_model_prediction_step_length(self):
        """Return the prediction step length used by this model"""
        return self.prediction_step_length  

    def set_model_prediction_length(self, length):
        """Configure the prediction length for the model"""
        self.num_days = int(length)
        self.prediction_total_length = length

    def _debug_outputs(self, batch_idx, x_location, x_calendar, target, pred):
        """Helper method for debug outputs"""
        binary_predictions = (torch.sigmoid(pred) > 0.5).float()
        
        print("\nBatch", batch_idx)
        # print("First 5 timeslots of first batch:")
        # print(f"Raw predictions (pre-sigmoid): {pred[0,0,:5]}")
        # print(f"Sigmoid predictions: {torch.sigmoid(pred[0,0,:5])}")
        # print(f"Binary predictions: {binary_predictions[0,0,:5]}")
        # print(f"Target values: {target[0,0,:5]}")             

        print(f"Input shapes: loc={x_location.shape}, cal={x_calendar.shape}, target={target.shape}")
        # print(f"Contains NaN - location: {torch.isnan(x_location).any()}, calendar: {torch.isnan(x_calendar).any()}, target: {torch.isnan(target).any()}")
        # print(f"Value ranges - location: [{x_location.min()}, {x_location.max()}], calendar: [{x_calendar.min()}, {x_calendar.max()}]")