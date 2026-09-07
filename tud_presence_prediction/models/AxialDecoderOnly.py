from json import decoder, encoder
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torch.nn import MultiheadAttention
from torch.nn.modules import batchnorm
from torchmetrics.classification import negative_predictive_value

from .internal.Time2Vec import Time2Vec
from .internal.model_util import model_util
from ..helpers.profiling.time_profiling import TimeProfiler
from flash_attn.modules.mha import MHA

class LearnTransformerDecoderBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout, use_flash_attention):
        super(LearnTransformerDecoderBlock, self).__init__()

        feature_dim = hidden_dim * 2
        
        if use_flash_attention:
            self.self_mha_timeslot = MHA(feature_dim, num_heads=num_heads, dropout=dropout, causal=True,  cross_attn=False)
            self.self_mha_days     = MHA(feature_dim, num_heads=num_heads, dropout=dropout, causal=True,  cross_attn=False)
            self.cross_mha         = MHA(feature_dim, num_heads=num_heads, dropout=dropout, causal=False, cross_attn=True)
            self._use_flash        = True
        else:
            self.self_mha_timeslot = MultiheadAttention(feature_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
            self.self_mha_days     = MultiheadAttention(feature_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
            self.cross_mha         = MultiheadAttention(feature_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
            self._use_flash        = False
            
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.norm3 = nn.LayerNorm(feature_dim)
        self.norm4 = nn.LayerNorm(feature_dim)
        self.norm_ff = nn.LayerNorm(feature_dim)

        self.ff    = nn.Sequential(
            nn.Linear(feature_dim, 4 * feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * feature_dim, feature_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, encoder_output):
        """
        Args
        x: shape [batch_size, days, timeslots, features] 
        encoder_output: shape [batch_size, days, timeslots, features] if training
                        shape [days, timeslots, features] if inference
        """
        shapes = {
            "x": (x.shape),
            "encoder_output": encoder_output.shape
        }
        b_x, d_x, t_x, f_x = x.shape
        b_enc, d_enc, t_enc, f_enc = encoder_output.shape if len(encoder_output.shape) == 4 else (1, *encoder_output.shape)

        x = x.reshape(b_x * d_x, t_x, f_x)

        # Masked Self-Attention timeslot-wise
        if self._use_flash:
            x = self.norm1(x + self.self_mha_timeslot(x))  
        else:
            attn_output, _ = self.self_mha_timeslot(x, x, x)
            x = self.norm1(x + attn_output)

        # Reshape to 4D tensor for next step
        x = x.reshape(b_x, d_x, t_x, f_x)  

        # Masked Self-Attention day-wise
        x = x.permute(0, 2, 3, 1).reshape(b_x * t_x, d_x, f_x)
        if self._use_flash:
            x = self.norm2(x + self.self_mha_days(x))
        else:
            attn_out, _ = self.self_mha_days(x, x, x, need_weights=False)
            x = self.norm2(x + attn_out)

        x = x.reshape(b_x, t_x, f_x, d_x).permute(0, 3, 1, 2)  # (B, D, T, F)

        # Cross-Attention timeslot-wise
        x = x.reshape(b_x * d_x, t_x, f_x)
        encoder_output = encoder_output.reshape(b_enc*d_enc, t_enc, f_enc)
        if self._use_flash:
            x = self.norm3(x + self.cross_mha(x, encoder_output)[0])
        else:
            attn_out, _ = self.cross_mha(x, encoder_output, encoder_output, need_weights=False)
            x = self.norm3(x + attn_out)
        
        # Cross-Attention day-wise        
        x = x.reshape(b_x, d_x, t_x, f_x).permute(0,2,1,3) # (B, T, D, F)
        encoder_output = encoder_output.reshape(b_enc, d_enc, t_enc, f_enc).permute(0,2,1,3) # (B, T, D, F)

        x = x.reshape(b_x * t_x, d_x, f_x)
        encoder_output = encoder_output.reshape(b_enc * t_enc, d_enc, f_enc)

        if self._use_flash:
            x = self.norm4(x + self.cross_mha(x, encoder_output)[0])
        else:
            attn_out, _ = self.cross_mha(x, encoder_output, encoder_output, need_weights=False)
            x = self.norm4(x + attn_out)
        
        x = x.reshape(b_x, t_x, f_x, d_x).permute(0, 3, 1, 2)  # (B, D, T, F)
        x = self.norm_ff(self.ff(x))

        return x

class LearnTransformer(pl.LightningModule):
    def __init__(self, location_dim, calendar_dim, input_dim_3=None, hidden_dim=8, num_heads=4, dropout=0.5, num_of_layers=3,
             use_optimizer="AdamW",
             optimizer_beta1=None, 
             optimizer_beta2=None,
             weight_decay=None,
             fixed_learning_rate=0.001,
             log_LR=True,
             use_LR_scheduler=True,
             LR_scheduler_min=0.000000175,
             LR_scheduler_max=0.00009,
             LR_scheduler_convergence=0.000016,
             LR_increase_phase_percentage=0.05,
             step_size=1,
             gradient_clip_val=1.0,
             use_flash_attention=True):  # Neuer Parameter fuer Flash Attention

        super(LearnTransformer, self).__init__()

        self.prediction_total_length = None
        self.prediction_step_length = (0, step_size)
        self.step_size = step_size
        self.use_flash_attention = use_flash_attention  # Speichern der Flash-Attention-Einstellung

        self.training_step_outputs = []   
        self.training_step_targets = []   
        self.validation_step_outputs = []        
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

        self.decoder_blocks = nn.ModuleList([
            LearnTransformerDecoderBlock(hidden_dim, num_heads, dropout, use_flash_attention=use_flash_attention)
            for _ in range(num_of_layers)  # decoder layers
        ])
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim*2, 1)       
        self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([4])) # pos_weight should be num_0/num_1, where num_0 is the number of 0s and num_1 is the number of 1s in the target data

    def forward(self, x_location, x_calendar, target=None):
        # Some sort of encoding of the input sequence

        

        
        if self.training or target is not None: # covering training and validation  
            decoder_output = self._decode_step(target = target, encoder_output = encoder_output)  # shape: (batch_size, days*timeslots, hidden_dim*2)
            batch_size, days, timeslots, _ = decoder_output.shape
            prediction = self.output_projection(decoder_output)
            prediction_flat = prediction.reshape(batch_size, days, timeslots)
     
            return prediction_flat

        else:
            # During inference, we'll generate predictions for the same length as input
            if self.prediction_total_length is None:
                print("No prediction window was given, therefore: prediction length = input length")
                self.prediction_total_length = x_location.size(1)
            
            decoder_output = self._decode_step(encoder_output = encoder_output) 
            predictions = self.output_projection(decoder_output)

            # Reshape to remove the last dimension
            predictions = predictions.squeeze(-1)  # Shape: [batch_size, days, timeslots]
            #print(f"This is what the predictions look like: {predictions}")

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
        else:
            days, timeslots, features = encoder_output.shape
            batch_size = 1
            
        device = encoder_output.device

        if target is not None:
            # Training mode
            batch_size, days, timeslots = target.shape
            
            # Korrekte Anwendung des Token-Embeddings und Expansion
            start_token = self.token_embedding(self.decoder_start_token.to(device))
            # Erweitern auf die batch_size, jedoch Beibehaltung der step_size-Dimension
            start_token = start_token.expand(batch_size, days , self.step_size, -1)

            if self.step_size > timeslots:
                raise ValueError(f"Step size {self.step_size} cannot be larger than {timeslots}, decrease step_size to use step_mode.")

            target_reshaped = target.unsqueeze(-1) # Hinzufuegen einer Dimension fuer die Features
            target_embed = self.target_embedding(target_reshaped)  # Shape: (batch_size, days, timeslots, hidden_dim*2)

            # Decoder-Eingabe initialisieren
            decoder_input = target_embed[:, :, :-self.step_size, :]  # Shape: (batch_size, days, timeslots - step_size, hidden_dim*2)
            
            # Start-Token hinzufuegen
            decoder_input = torch.cat([start_token, decoder_input], dim=2)
            
            # Durch Decoder-Bloecke verarbeiten
            decoder_output = decoder_input
            for block in self.decoder_blocks:
                decoder_output = block(decoder_output, encoder_output)
            
            return decoder_output

        else:
            # Inference mode
            start_tok = self.token_embedding(self.decoder_start_token.to(device))  # [1,1,1,f]
            start_tok = start_tok.expand(batch_size, self.prediction_total_length, self.step_size, features) 
            input_tensor = torch.zeros((batch_size, self.prediction_total_length, timeslots + self.step_size, features), device=device)
            input_tensor[:, :, :self.step_size, :] = start_tok # Start token + empty slots, shape: (batch_size, days, ts + self.step_size, features)

            decoder_input = input_tensor[:, 0:1, :self.step_size, :] # Start token shape (1, 1, step_size, features)

            output = torch.zeros((batch_size, self.prediction_total_length, timeslots, features), device=device)  # Output tensor

            # Single step mode: generate one timeslot at a time
            if self.step_mode == "single":
                for day in range(self.prediction_total_length):
                    for ts in range(timeslots):
                        if (day == 0) and (ts == 0):  # Erster Tag und erster Zeitslot
                            pass
                        # Erster Tag außer dem ersten Zeitslot
                        elif day == 0:    
                            decoder_input = input_tensor[:, :, :ts+self.step_size, :]
                        elif day > 0:  # Alle darauf folgenden Tage
                            decoder_input = torch.cat([output[:, :day, :ts+1, :], input_tensor[:, day:day+1, :ts+1, :]], dim=1)  # Vorherige embeddings und neuer Tag mit Start-Token bis zum aktuellen Zeitslot
                    
                        #DEBUG
                        print(f"Decoder input shape: {decoder_input.shape}, Encoder output shape: {encoder_output.shape}")  
    
                        # Durch Decoder-Blöcke leiten
                        decoder_output = decoder_input
                        for block in self.decoder_blocks:
                            decoder_output = block(decoder_output, encoder_output)
                        
                        output[:, day, ts, :] = decoder_output[:, -1, -1, :]  # Speichern des Outputs für diesen Zeitslot
                        input_tensor[:, day, ts+self.step_size, :] = decoder_output[:, -1, -1, :]  # Update input tensor for next step
            
            
            # Step mode: generate multiple timeslots at a time
            elif self.step_mode == "step":
                for day in range(self.prediction_total_length):
                    for ts in range(0, timeslots, self.step_size):
                        # Berechne die tatsächliche Anzahl der Zeitslots in diesem Schritt
                        ts_end = min(ts + self.step_size, timeslots)
                        actual_step_size = ts_end - ts
                        
                        if (day == 0) and (ts == 0):  # Erster Tag und erster Zeitslot-Block
                            pass
                        # Erster Tag außer dem ersten Zeitslot-Block
                        elif day == 0:    
                            decoder_input = input_tensor[:, :, :ts+self.step_size, :]
                        elif day > 0:  # Alle darauf folgenden Tage
                            # Vorherige embeddings und neuer Tag mit Start-Token bis zum aktuellen Zeitslot-Block
                            decoder_input = torch.cat([output[:, :day, :ts+1, :], input_tensor[:, day:day+1, :ts+1, :]], dim=1)
                        
                        # Durch Decoder-Blöcke leiten
                        decoder_output = decoder_input
                        for block in self.decoder_blocks:
                            decoder_output = block(decoder_output, encoder_output)
                        
                        # Speichern des Outputs für diesen Zeitslot-Block
                        for i in range(actual_step_size):
                            output[:, day, ts + i, :] = decoder_output[:, -1, -(actual_step_size) + i, :]  # Speichern des Outputs für diesen Zeitslot
                            
                            # Update input tensor für nächste Schritte (falls innerhalb der Grenzen)
                            if ts + i + self.step_size < timeslots + self.step_size:
                                input_tensor[:, day, ts + i + self.step_size, :] = decoder_output[:, -1, -(actual_step_size) + i, :]
            
            # Return processed output with proper shape
            return output