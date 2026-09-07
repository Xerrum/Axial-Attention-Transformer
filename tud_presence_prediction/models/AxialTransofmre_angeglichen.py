import torch
import math
import pytorch_lightning as pl
import torch.nn as nn
from torch.nn import MultiheadAttention
from torchmetrics.classification import BinaryAccuracy, BinaryRecall, BinaryF1Score, BinaryPrecision

from .internal.Time2Vec import Time2Vec
from .internal.model_util import model_util
from ..helpers.profiling.time_profiling import TimeProfiler
from flash_attn.modules.mha import MHA as FlashMHA


# ----------------------------
# Axial Encoder: Timeslot -> Day
# ----------------------------
class LearnTransformerEncoderBlock(nn.Module):
    """
    Axial encoder block:
      (1) Self-Attn über Timeslots (innerhalb eines Tages)
      (2) Self-Attn über Tage (für jeden Timeslot separat)
      (3) Position-wise FFN
    Interface, Gewichte, Dropout, Normierung analog zur Basisklasse.
    """
    def __init__(self, hidden_dim, num_heads, dropout, use_flash_attention: bool = True):
        super().__init__()
        feature_dim = hidden_dim * 2

        if use_flash_attention:
            self.mha_times = FlashMHA(feature_dim, num_heads=num_heads, dropout=dropout, causal=False, cross_attn=False)
            self.mha_days  = FlashMHA(feature_dim, num_heads=num_heads, dropout=dropout, causal=False, cross_attn=False)
            self._use_flash = True
        else:
            self.mha_times = MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
            self.mha_days  = MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
            self._use_flash = False

        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.ff = nn.Sequential(
            nn.Linear(feature_dim, 4 * feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * feature_dim, feature_dim),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(feature_dim)

    def forward(self, x_embedded_combined):
        # Unterstütze [B, D, T, F] und [D, T, F]
        if len(x_embedded_combined.shape) == 4:
            b, d, t, f = x_embedded_combined.shape
            out_shape = (b, d, t, f)
        else:
            d, t, f = x_embedded_combined.shape
            b = 1
            x_embedded_combined = x_embedded_combined.unsqueeze(0)
            out_shape = (d, t, f)

        # (1) Timeslot-Attention (innerhalb eines Tages)
        x = x_embedded_combined.reshape(b * d, t, f)
        if self._use_flash:
            x = self.norm1(x + self.mha_times(x))
        else:
            attn_out, _ = self.mha_times(x, x, x, need_weights=False)
            x = self.norm1(x + attn_out)

        # (2) Day-Attention (für jeden Timeslot separat)
        x = x.reshape(b, d, t, f).permute(0, 2, 1, 3).reshape(b * t, d, f)
        if self._use_flash:
            x = self.norm2(x + self.mha_days(x))
        else:
            attn_out, _ = self.mha_days(x, x, x, need_weights=False)
            x = self.norm2(x + attn_out)
        x = x.reshape(b, t, d, f).permute(0, 2, 1, 3).contiguous()

        # (3) FFN
        x = self.norm3(x + self.ff(x))
        return x.reshape(out_shape)


# ----------------------------
# Axial Decoder: u-Kontext (T->D, kausal über Tage) + kausale Timeslot-Attn + Cross-Attn
# ----------------------------
class LearnTransformerDecoderBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout, use_flash_attention):
        super().__init__()
        feature_dim = hidden_dim * 2

        if use_flash_attention:
            self.self_mha_times_ctx = FlashMHA(feature_dim, num_heads=num_heads, dropout=dropout, causal=False, cross_attn=False)
            self.self_mha_days      = FlashMHA(feature_dim, num_heads=num_heads, dropout=dropout, causal=True,  cross_attn=False)
            self.self_mha_times     = FlashMHA(feature_dim, num_heads=num_heads, dropout=dropout, causal=True,  cross_attn=False)
            self.cross_mha          = FlashMHA(feature_dim, num_heads=num_heads, dropout=dropout, causal=False, cross_attn=True)
            self._use_flash         = True
        else:
            self.self_mha_times_ctx = MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
            self.self_mha_days      = MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
            self.self_mha_times     = MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
            self.cross_mha          = MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
            self._use_flash         = False

        self.norm_u  = nn.LayerNorm(feature_dim)
        self.norm_ts = nn.LayerNorm(feature_dim)
        self.norm_x  = nn.LayerNorm(feature_dim)
        self.ff = nn.Sequential(
            nn.Linear(feature_dim, 4 * feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * feature_dim, feature_dim),
            nn.Dropout(dropout),
        )
        self.norm_ff = nn.LayerNorm(feature_dim)

    def _shift_down(self, x_bt_d_f):
        # x: [B*T, D, F] -> Shift um 1 nach unten (oben 0), für Kausalität entlang der Tagesachse
        btf, D, F = x_bt_d_f.shape
        z = x_bt_d_f.new_zeros((btf, 1, F))
        return torch.cat([z, x_bt_d_f[:, :-1, :]], dim=1)

    def compute_day_context(self, x):
        # x: [B, D, T, F] -> u: nur frühere Tage sichtbar
        B, D, T, F = x.shape
        # (i) unmaskierte Timeslot-Attn als Kontextaufbau
        ts_in = x.reshape(B * D, T, F)
        if self._use_flash:
            ts_ctx = self.self_mha_times_ctx(ts_in)
        else:
            ts_ctx, _ = self.self_mha_times_ctx(ts_in, ts_in, ts_in, need_weights=False)
        ts_ctx = ts_ctx.reshape(B, D, T, F)

        # (ii) kausale Day-Attn (mit ShiftDown)
        day_in = ts_ctx.permute(0, 2, 1, 3).reshape(B * T, D, F)
        day_in = self._shift_down(day_in)
        if self._use_flash:
            u = self.self_mha_days(day_in)
        else:
            u, _ = self.self_mha_days(day_in, day_in, day_in, need_weights=False)
        u = u.reshape(B, T, D, F).permute(0, 2, 1, 3).contiguous()
        return u  # [B, D, T, F]

    def forward(self, target, encoder_output, u_cache: torch.Tensor = None):
        """
        target: [B, D, T, F], encoder_output: [B, D, T, F]
        u_cache erlaubt semi-paralleles Decoding (T-Schritte eines Tages ohne erneute u-Berechnung).
        """
        B, D, T, F = target.shape
        # (1) u (frühere Tage) berechnen/verwenden
        u = self.compute_day_context(target) if u_cache is None else u_cache
        x = self.norm_u(target + u)

        # (2) kausale Timeslot-Attn
        ts_in = x.reshape(B * D, T, F)
        if self._use_flash:
            ts = self.self_mha_times(ts_in)
        else:
            mask = torch.ones(T, T, device=ts_in.device, dtype=torch.bool).triu(1)
            ts, _ = self.self_mha_times(ts_in, ts_in, ts_in, need_weights=False, is_causal=True, attn_mask=mask)
        x = self.norm_ts(ts_in + ts).reshape(B, D, T, F)

        # (3) Cross-Attn (unkausal)
        q  = x.reshape(B, D * T, F)
        if len(encoder_output.shape) == 4:
            kv = encoder_output.reshape(encoder_output.size(0), -1, encoder_output.size(-1))
        else:
            kv = encoder_output.reshape(1, -1, encoder_output.size(-1))
        if self._use_flash:
            cross = self.cross_mha(q, kv)
        else:
            cross, _ = self.cross_mha(q, kv, kv, need_weights=False)
        x = self.norm_x(q + cross).reshape(B, D, T, F)

        # (4) FFN
        x = self.norm_ff(x + self.ff(x))
        return x


# ----------------------------
# LightningModule (identisch zur Basisklasse, nur Blocks sind axial)
# ----------------------------
class LearnTransformer(pl.LightningModule):
    """
    Gleiche Features, Hyperparameter, Scheduler, Logging wie LearnTransformer,
    aber Encoder/Decoder verwenden die axialen Blöcke.
    """
    def __init__(self, location_dim, calendar_dim, input_dim_3=None, hidden_dim=16, num_heads=4, dropout=0.4, num_of_layers=3,
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
        self.use_flash_attention = use_flash_attention
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

        # Embeddings
        self.time2vec = Time2Vec(location_dim, hidden_dim)         # -> 2*hidden_dim
        self.calendar_embedding = nn.Linear(calendar_dim, hidden_dim * 2)
        self.token_embedding = nn.Linear(1, hidden_dim * 2)
        self.target_embedding = nn.Linear(1, hidden_dim * 2)

        # Axial Encoder/Decoder
        self.encoder_blocks = nn.ModuleList([
            LearnTransformerEncoderBlock(hidden_dim, num_heads, dropout, use_flash_attention=use_flash_attention)
            for _ in range(num_of_layers)
        ])
        self.decoder_blocks = nn.ModuleList([
            LearnTransformerDecoderBlock(hidden_dim, num_heads, dropout, use_flash_attention=use_flash_attention)
            for _ in range(num_of_layers)
        ])

        # Output head & loss/metrics
        self.output_projection = nn.Linear(hidden_dim * 2, 1)
        self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([4]))

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
        Gleiche API wie Basismodell:
          - x_location: (B, D_hist, T, location_dim)
          - x_calendar: (B, D_hist, T, calendar_dim)
          - target    : (B, D_fut,  T)   (nur Training/Val; Inferenz ohne target)
        """
        # Encoder-Einbettung
        x_loc = self.time2vec(x_location)
        x_cal = self.calendar_embedding(x_calendar)
        x_embed = x_loc + x_cal

        # Encoder
        encoder_output = x_embed
        for block in self.encoder_blocks:
            encoder_output = block(encoder_output)

        if self.training or target is not None:
            # Teacher forcing
            decoder_output = self._decode_step(encoder_output=encoder_output, target=target)
            prediction = self.output_projection(decoder_output).squeeze(-1)  # (B, D_fut, T)
            return prediction
        else:
            # Inferenz: autoregressiv für self.num_days (oder Länge der Historie)
            pred_len_days = int(self.num_days) if (self.num_days is not None) else x_location.size(1)
            self.prediction_total_length = pred_len_days
            decoder_output = self._decode_step(encoder_output=encoder_output, target=None)
            predictions = self.output_projection(decoder_output).squeeze(-1)  # (B, D_pred, T)
            return predictions

    def _decode_step(self, encoder_output, target=None):
        """
        Liefert Decoder-Ausgabe mit Form (B, D, T, 2*hidden) für Projektion
        """
        # Encoder zu [B, D, T, F] vereinheitlichen
        if len(encoder_output.shape) == 4:
            B, D_enc, T_enc, F = encoder_output.shape
        else:
            D_enc, T_enc, F = encoder_output.shape
            B = 1
            encoder_output = encoder_output.unsqueeze(0)

        device = encoder_output.device

        if target is not None:
            # target: (B, D_fut, T)
            B, D_tgt, T_tgt = target.shape

            # Start-Token pro Tag
            start_tok = self.token_embedding(self.decoder_start_token.to(device))              # [1, step, F]
            start_tok = start_tok.expand(B, D_tgt, self.step_size, -1)                         # [B, D, step, F]

            if self.step_size > T_tgt:
                raise ValueError(f"Step size {self.step_size} cannot be larger than target length {T_tgt}, increase amount of training data to use step_mode.")

            # Ziel-Embedding und Decoder-Eingang
            tgt_emb = self.target_embedding(target.unsqueeze(-1))                              # [B, D, T, F]
            dec_in  = torch.cat([start_tok, tgt_emb[:, :, :-self.step_size, :]], dim=2)       # [B, D, T, F]

            x = dec_in
            for block in self.decoder_blocks:
                x = block(x, encoder_output)
            return x  # [B, D, T, F]

        else:
            # Inferenz: autoregressiv
            D_pred, T_slots = self.prediction_total_length, T_enc
            start_tok = self.token_embedding(self.decoder_start_token.to(device))              # [1, step, F]
            start_tok = start_tok.expand(B, D_pred, self.step_size, -1)                        # [B, D, step, F]

            # Eingabecontainer mit Start-Token und leeren Slots
            in_tensor = torch.zeros((B, D_pred, T_slots, F), device=device)
            in_tensor[:, :, :self.step_size, :] = start_tok
            out = torch.zeros_like(in_tensor)

            if self.step_mode == "single":
                for d in range(D_pred):
                    # u-Cache pro Layer optional (ein Rechenersparnis, Schnittstelle bleibt gleich)
                    u_caches = [None] * len(self.decoder_blocks)
                    for ts in range(T_slots):
                        if d == 0 and ts == 0:
                            dec_in = in_tensor[:, 0:1, :, :]
                        elif d == 0:
                            dec_in = in_tensor[:, 0:1, :, :]
                        elif ts == 0:
                            dec_in = torch.cat([out[:, :d, :, :], in_tensor[:, d:d+1, :, :]], dim=1)
                        else:
                            dec_in = in_tensor[:, :d+1, :, :]

                        x = dec_in
                        for li, block in enumerate(self.decoder_blocks):
                            if ts == 0:
                                x = block(x, encoder_output, u_cache=None)
                                # optionalen Cache nur für den aktuellen Tag sichern
                                u_full = block.compute_day_context(dec_in if li == 0 else x)
                                u_cache = torch.zeros_like(u_full)
                                u_cache[:, d:d+1, :, :] = u_full[:, d:d+1, :, :]
                                u_caches[li] = u_cache.detach()
                            else:
                                x = block(x, encoder_output, u_cache=u_caches[li])

                        out[:, d, ts, :] = x[:, d, ts, :]
                        if ts != T_slots - 1:
                            in_tensor[:, d, ts + self.step_size, :] = x[:, d, ts, :]

            # step_mode=="step": identisch zur Basisklasse noch nicht vektorisiert genutzt
            return out

    # -------- Optimizer & Scheduler (1:1 wie LearnTransformer) --------
    def configure_optimizers(self):
        base_max = self.LR_scheduler_max if self.LR_scheduler_max else (self.fixed_learning_rate or 1e-3)
        base_lr  = (base_max / 25.0) if self.use_LR_scheduler else (self.fixed_learning_rate or 1e-3)

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=base_lr,
            weight_decay=(self.weight_decay if self.weight_decay is not None else 1e-2),
        )

        if not self.use_LR_scheduler:
            return optimizer

        ts = int(getattr(self.trainer, "estimated_stepping_batches", 0))
        if ts < 2:
            print(f"[WARN] OneCycleLR deaktiviert (estimated_stepping_batches={ts})")
            return optimizer

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
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    # -------- Training / Validation (1:1 wie LearnTransformer) --------
    def training_step(self, batch, batch_idx):
        x_location, x_calendar, target = batch

        B, D, T, _ = x_location.shape
        if self.num_days is None:
            print("num_days konnte nicht verarbeitet werden, letzte 3 Tage werden jeweils vorhergesagt")
            num_days = 3
        else:
            num_days = int(self.num_days)

        split_idx = D - num_days
        x_loc_hist = x_location[:, :split_idx, :, :]
        x_cal_hist = x_calendar[:, :split_idx, :, :]
        target_future = target[:, split_idx:, :]

        pred = self(x_loc_hist, x_cal_hist, target_future)
        if torch.isnan(pred).any():
            print(f"NaN in predictions at batch {batch_idx}")

        if pred.shape != target_future.shape:
            pred = pred.view(target_future.shape)

        mask = ~torch.isnan(target_future)
        if not mask.any():
            print(f"No valid samples in batch {batch_idx}")

        filtered_pred = pred[mask]
        filtered_target = target_future[mask]

        try:
            loss = self.loss(filtered_pred, filtered_target)
            if not torch.isfinite(loss):
                print(f"Non-finite loss value: {loss}")
                return torch.tensor(float('inf'), requires_grad=True)
        except Exception as e:
            print(f"Error in loss calculation: {e}")
            return torch.tensor(float('inf'), requires_grad=True)

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
        target_future = target[:, split_idx:, :]

        # (1) Teacher-forced Pfad
        with torch.no_grad():
            pred_tf = self(x_loc_hist, x_cal_hist, target_future)
        if pred_tf.shape != target_future.shape:
            pred_tf = pred_tf.view(target_future.shape)

        mask = ~torch.isnan(target_future)
        filtered_pred_tf = pred_tf[mask]
        filtered_target = target_future[mask]

        loss = self.loss(filtered_pred_tf, filtered_target)

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

        # (2) Open-loop Pfad
        with torch.no_grad():
            pred_open = self(x_loc_hist, x_cal_hist)
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

    # Präsenz-API kompatibel halten
    def supports_variable_length(self):
        return True

    def get_model_prediction_step_length(self):
        return self.prediction_step_length

    def set_model_prediction_length(self, length):
        # wie in LearnTransformer: beide Felder setzen
        self.num_days = int(length)
        self.prediction_total_length = length

    def _debug_outputs(self, batch_idx, x_location, x_calendar, target, pred):
        binary_predictions = (torch.sigmoid(pred) > 0.5).float()
        print("\nBatch", batch_idx)
        print(f"Input shapes: loc={x_location.shape}, cal={x_calendar.shape}, target={target.shape}")

