import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Optional
from flash_attn.modules.mha import MHA as FlashMHA

TIMESLOTS_DAY = 24


class ConvDownsampler(nn.Module):
    """
    Per-time 2D Conv downsampling mit residualem 1x1 Pfad.
    
    Args:
        C_in (int): Anzahl der Eingangskanaele
        C_out (int): Anzahl der Ausgangskanaele
        stride (int): Faktor fuer raeumliches Downsampling
        kernel_size (int): Groesse des Faltungskernels
    """
    def __init__(self, C_in: int, C_out: int, stride: int = 2, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding)
        self.proj = nn.Conv2d(C_in, C_out, kernel_size=1, stride=stride)  # residual 1x1
        self.norm = nn.BatchNorm2d(C_out)
        self.act = nn.GELU()

    def forward(self, x):
        """
        Fuehrt Downsampling durch Faltung aus.
        
        Args:
            x (torch.Tensor): Eingabetensor [Shape: B, T, C, H, W]
                B: Batch-Groesse
                T: Anzahl der Zeitschritte
                C: Anzahl der Kanaele
                H, W: Raeumliche Dimensionen
        
        Returns:
            torch.Tensor: Downgesampelter Tensor [Shape: B, T, C_out, H2, W2]
                H2 = H/stride, W2 = W/stride
        """
        B, T, C, H, W = x.shape
        x2d = x.reshape(B * T, C, H, W)        # [B*T, C, H, W]  (keine permute noetig)
        y = self.conv(x2d) + self.proj(x2d)    # [B*T, C_out, H2, W2]
        y = self.norm(y)
        y = self.act(y)
        C_out, H2, W2 = y.shape[1], y.shape[2], y.shape[3]
        return y.reshape(B, T, C_out, H2, W2)  # [B, T, C_out, H2, W2]  (channel-first)


class AFNO2DMinimal(nn.Module):
    """
    Eine kompakte AFNO-style spektrale Mischblock fuer 2D-Felder.
    
    Args:
        C (int): Anzahl der Kanaele
        H (int): Hoehe des Eingabetensors
        W (int): Breite des Eingabetensors
        keep_ratio (float): Anteil der beizubehaltenden Frequenzen
        dropout (float): Dropout-Rate
    """
    def __init__(self, C: int, H: int, W: int, keep_ratio: float = 0.25, dropout: float = 0.0):
        super().__init__()
        self.keep_h = max(1, int(H * keep_ratio))
        self.keep_w = max(1, int(W * keep_ratio))
        self.weight_real = nn.Parameter(torch.randn(C, self.keep_h, self.keep_w) * 0.02)
        self.weight_imag = nn.Parameter(torch.randn(C, self.keep_h, self.keep_w) * 0.02)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm2d(C)
        self.act = nn.GELU()

    def _mix_freq(self, x_fft):
        """
        Mischt niederfrequente Komponenten mit lernbaren komplexen Gewichten.
    
        Args:
            x_fft (torch.Tensor): Fourier-transformierter Tensor [Shape: B*T, C, Hf, Wf]
                Hf: Frequenzdimension fuer Hoehe
                Wf: Frequenzdimension fuer Breite
    
        Returns:
            torch.Tensor: Gemischter Tensor im Frequenzbereich [Shape: B*T, C, Hf, Wf]
        """
        # Stellen Sie sicher, dass Parameter auch in float32 sind f�r komplexe Multiplikation
        orig_dtype = x_fft.dtype.real_dtype if hasattr(x_fft.dtype, 'real_dtype') else x_fft.dtype
    
        BxT, C, Hf, Wf = x_fft.shape
        kh = min(self.keep_h, Hf)
        kw = min(self.keep_w, Wf)
        low = x_fft[:, :, :kh, :kw]                # [B*T, C, kh, kw]
    
        # Konvertiere Gewichte zu float32 f�r die komplexe Multiplikation
        wr = self.weight_real[:, :kh, :kw].to(torch.float32).unsqueeze(0).expand(BxT, -1, -1, -1)
        wi = self.weight_imag[:, :kh, :kw].to(torch.float32).unsqueeze(0).expand(BxT, -1, -1, -1)
    
        a, b = low.real, low.imag
        real = a * wr - b * wi
        imag = a * wi + b * wr
        mixed = torch.complex(real, imag)
    
        x_fft = x_fft.clone()
        x_fft[:, :, :kh, :kw] = mixed
    
        return x_fft

    def forward(self, x):
        """
        Fuehrt spektrale Mischung durch FFT, Filterung und IFFT durch.
        
        Args:
            x (torch.Tensor): Eingabetensor [Shape: B, T, C, H, W]
        
        Returns:
            torch.Tensor: Spektral gemischter Tensor [Shape: B, T, C, H, W]
        """
        B, T, C, H, W = x.shape
        x2d = x.reshape(B * T, C, H, W)           # [B*T, C, H, W]

        orig_dtype = x2d.dtype
        x2d = x2d.to(torch.float32)

        x_fft = torch.fft.rfft2(x2d, norm='ortho')
        x_fft = self._mix_freq(x_fft)
        y = torch.fft.irfft2(x_fft, s=(H, W), norm='ortho')   # [B*T, C, H, W]
        y = y.to(orig_dtype)

        y = self.dropout(y)
        y = self.norm(y)
        y = self.act(y)
        return y.reshape(B, T, C, H, W)           # [B, T, C, H, W]


class TimePositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding over time T applied to tokens.
    Input/Output shape for tokens: [Nseq, T, D]
    """
    def __init__(self, d_model: int, max_len: int = 16384):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # [max_len, D]

    def forward(self, x):
        # x: [N, T, D]
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)


class Time2Vec(nn.Module):
    """
    Time2Vec (Kazemi et al., 2019) – einfache Implementierung.
    Eingabe:  time_feats [B, T, F] (hier F=2: hour_frac, doy_frac in [0,1])
    Ausgabe:  [B, T, k]  (k = Embedding-Dimension)
    """
    def __init__(self, in_features: int, k: int):
        super().__init__()
        self.in_features = in_features
        self.k = k
        # lineare Komponente
        self.w0 = nn.Linear(in_features, 1)
        self.b0 = nn.Parameter(torch.zeros(1))
        # periodische Komponenten
        self.W = nn.Linear(in_features, k - 1)
        self.B = nn.Parameter(torch.zeros(k - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,F]
        v0 = self.w0(x) + self.b0              # [B,T,1]
        v1 = torch.sin(self.W(x) + self.B)     # [B,T,k-1]
        return torch.cat([v0, v1], dim=-1)     # [B,T,k]


class AxialTransformerEncoderBlock(nn.Module):
    """
    Hierarchischer Encoder-Block: Attention erst über Zeitslots innerhalb eines Tages,
    dann über Tage, gefolgt von einem Feed-Forward Netzwerk.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float, use_flash_attention: bool = True):
        super().__init__()
        
        # Attention-Implementierungswahl
        if use_flash_attention:
            self.mha_timeslots = FlashMHA(hidden_dim, num_heads=num_heads, dropout=dropout, causal=False, cross_attn=False)
            self.mha_days = FlashMHA(hidden_dim, num_heads=num_heads, dropout=dropout, causal=False, cross_attn=False)
            self._use_flash = True
        else:
            self.mha_timeslots = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
            self.mha_days = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
            self._use_flash = False

        # Normalisierung & FFN
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x_embedded):
        """
        Args:
            x_embedded: Shape [batch_size, days, timeslots, features] 
                       oder [N, days, timeslots, features] wenn N = batch_size * H2 * W2
        Returns:
            Tensor mit gleicher Form wie die Eingabe.
        """
        # Handle Dimensionen mit und ohne Batch
        if len(x_embedded.shape) == 4:
            batch_size, days, timeslots, features = x_embedded.shape
            # Originale Form für Ausgabe speichern
            output_shape = (batch_size, days, timeslots, features)
        else:
            raise ValueError("Erwarte 4D-Tensor [batch, days, timeslots, features]")

        # --- 1) Timeslot-Self-Attention ---------------------------------------------------
        x = x_embedded.contiguous().view(batch_size * days, timeslots, features)
        if self._use_flash:
            x = self.norm1(x + self.mha_timeslots(x))
        else:
            attn_out, _ = self.mha_timeslots(x, x, x, need_weights=False)
            x = self.norm1(x + attn_out)

        # Reshape zu 4D-Tensor für nächsten Schritt
        x_day = x.contiguous().view(batch_size, days, timeslots, features)

        # --- 2) Day-Self-Attention --------------------------------------------------------
        x_day = x_day.permute(0, 2, 1, 3).contiguous().view(batch_size * timeslots, days, features)        
        if self._use_flash:
            x_day = self.norm2(x_day + self.mha_days(x_day))
        else:
            attn_out, _ = self.mha_days(x_day, x_day, x_day, need_weights=False)
            x_day = self.norm2(x_day + attn_out)
        x_day = x_day.contiguous().view(batch_size, timeslots, days, features).permute(0, 2, 1, 3).contiguous()

        # --- 3) Position-wise Feed-Forward -----------------------------------------------
        x_ff = self.norm3(x_day + self.ff(x_day))
        
        return x_ff.reshape(output_shape)


class AxialTransformerDecoderBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout, use_flash_attention):
        super(AxialTransformerDecoderBlock, self).__init__()

        self.hidden_dim = hidden_dim
        
        if use_flash_attention:
            # Non-causal Self-Attention für Kontext, causal Self-Attention für Zielsequenz
            self.self_mha_timeslot_ctx = FlashMHA(hidden_dim, num_heads=num_heads, dropout=dropout, causal=False, cross_attn=False)
            self.self_mha_timeslot = FlashMHA(hidden_dim, num_heads=num_heads, dropout=dropout, causal=True, cross_attn=False)
            self.self_mha_days = FlashMHA(hidden_dim, num_heads=num_heads, dropout=dropout, causal=True, cross_attn=False)
            self.cross_mha = FlashMHA(hidden_dim, num_heads=num_heads, dropout=dropout, causal=False, cross_attn=True)
            self._use_flash = True
        else:
            self.self_mha_timeslot_ctx = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
            self.self_mha_timeslot = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
            self.self_mha_days = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
            self.cross_mha = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
            self._use_flash = False
            
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.norm_cross = nn.LayerNorm(hidden_dim)
        self.norm_ff = nn.LayerNorm(hidden_dim)

        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def compute_day_context(self, x):
        """
        Berechnet u (Kontext aus früheren Tagen) gemäß:
        unmasked Row (Timeslot) -> masked Column (Day) -> ShiftDown.
        x: [B, D, T, F]  ->  u: [B, D, T, F]
        """
        b_x, d_x, t_x, f_x = x.shape

        # (i) unmasked Row / Timeslot (nur Kontextaufbau, kein Residual auf x!)
        ts_in = x.reshape(b_x * d_x, t_x, f_x)
        if self._use_flash:
            ts_ctx = self.self_mha_timeslot_ctx(ts_in)
        else:
            ts_ctx, _ = self.self_mha_timeslot_ctx(ts_in, ts_in, ts_in, need_weights=False)
        ts_ctx = ts_ctx.reshape(b_x, d_x, t_x, f_x)

        # (ii) masked Column / Day + (iii) ShiftDown
        day_in = ts_ctx.permute(0, 2, 1, 3).reshape(b_x * t_x, d_x, f_x)
        day_in = self._shift_down(day_in)
        if self._use_flash:
            u = self.self_mha_days(day_in)   # kausal entlang Day-Achse
        else:
            u, _ = self.self_mha_days(day_in, day_in, day_in, need_weights=False)
        u = u.reshape(b_x, t_x, d_x, f_x).permute(0, 2, 1, 3).contiguous()
        return u

    def _shift_down(self, x_bt_d_f):
        """
        ShiftDown wie in Ho et al.: Token (d,t) sieht nur Tage < d
        x_bt_d_f: [B*T, D, F] -> schiebt um 1 nach unten, füllt oben mit 0
        """
        btf, D, F = x_bt_d_f.shape
        z = x_bt_d_f.new_zeros((btf, 1, F))
        return torch.cat([z, x_bt_d_f[:, :-1, :]], dim=1)

    def forward(self, x, encoder_output, u_cache: torch.Tensor = None):
        """
        Optionales u_cache erlaubt semi-paralleles Decoding:
        - u_cache=None  -> u wird frisch berechnet (Training, Tag-Beginn)
        - u_cache=Tensor -> u-Schritt wird übersprungen (Timeslots desselben Tages)
        
        x: [B, D, T, F] - Decoder Input
        encoder_output: [B, D, T, F] - Encoder Output
        """
        b_x, d_x, t_x, f_x = x.shape

        if len(encoder_output.shape) == 4:
            b_enc, d_enc, t_enc, f_enc = encoder_output.shape
        else:
            encoder_output = encoder_output.unsqueeze(0)
            b_enc, d_enc, t_enc, f_enc = encoder_output.shape

        # (1)+(2) u: entweder frisch oder aus Cache
        if u_cache is None:
            u = self.compute_day_context(x)         # teuer -> 1x pro Tag
        else:
            u = u_cache                              # reuse

        # u addieren (nur frühere Tage) + LN
        x = self.norm2(x + u)

        # (3) masked Row / Timeslot (autoregressiv im Tag)
        ts2_in = x.reshape(b_x * d_x, t_x, f_x)
        if self._use_flash:
            ts2 = self.self_mha_timeslot(ts2_in)    # kausal=True
        else:
            T = ts2_in.size(1)
            mask = torch.ones(T,T,device=ts2_in.device, dtype=torch.bool).triu()
            ts2, _ = self.self_mha_timeslot(ts2_in, ts2_in, ts2_in, need_weights=False, is_causal=True)
        x = self.norm3(ts2_in + ts2).reshape(b_x, d_x, t_x, f_x)

        # (4) Cross-Attention (unkausal) + (5) FFN
        q  = x.reshape(b_x, d_x * t_x, f_x)
        kv = encoder_output.reshape(b_enc, d_enc * t_enc, f_enc)
        if self._use_flash:
            cross = self.cross_mha(q, kv)
            x = self.norm_cross(q + cross)
        else:
            cross, _ = self.cross_mha(q, kv, kv, need_weights=False)
            x = self.norm_cross(q + cross)

        x = x.reshape(b_x, d_x, t_x, f_x)
        x = self.norm_ff(x + self.ff(x))
        return x


class WeatherAxialTransformer(pl.LightningModule):
    """
    Encoder-Decoder Weather Transformer mit AFNO-Bloecken und Axialer Attention.
    Verwendet raeumliche Verarbeitung mit Conv + AFNO, gefolgt von zeitlicher Attention.
    
    Args:
        in_channels (int): Anzahl der Eingangskanaele
        hidden_dim (int): Versteckte Dimension des Modells
        num_heads (int): Anzahl der Attention-Heads
        num_of_layers (int): Anzahl der Transformer-Schichten
        dropout (float): Dropout-Rate
        stride (int): Faktor fuer raeumliches Downsampling
        step_size (int): Schrittgroesse fuer autoregressive Generierung
        use_flash_attention (bool): Ob Flash-Attention verwendet werden soll
        use_optimizer (str): Welcher Optimizer verwendet werden soll
        fixed_learning_rate (float): Lernrate
        weight_decay (float): Gewichtszerfall-Faktor
        use_LR_scheduler (bool): Ob ein LR-Scheduler verwendet werden soll
        context_len (int): Laenge des Kontexts (Vergangenheit)
    """
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_of_layers: int = 4,
        dropout: float = 0.1,
        stride: int = 2,   # spatial downsampling factor
        step_size: int = 1,
        use_flash_attention: bool = True,
        use_optimizer: str = "AdamW",
        fixed_learning_rate: float = 3e-4,
        weight_decay: float = 1e-2,
        use_LR_scheduler: bool = True,
        context_len: int = 24,   # Anzahl der Kontext-Zeitschritte (Vergangenheit)
        use_time2vec: bool = True,
        t2v_dim: int = 8,
        ss_mode = "batch",
        ss_p_start = 0.0,
        ss_p_end = 0.5,
        ss_k = 1000,
        ss_warmup_epochs = 0,
    ):
        super(WeatherAxialTransformer, self).__init__()
        self.save_hyperparameters()
        self.use_flash_attention = use_flash_attention
        self.stride = stride
        self.hidden_dim = hidden_dim
        self.prediction_total_length = None  # in std Format
        self.in_channels = in_channels
        self.step_size = step_size
        self.context_len = int(context_len)
        self.use_time2vec = use_time2vec


        # Zeit-Einbettung
        if self.use_time2vec:
            self.t2v = Time2Vec(in_features=2, k=t2v_dim)
            self.time_proj = nn.Linear(t2v_dim, hidden_dim)

        # Step-Mode wie im LearnTransformer
        if step_size > 1:
            self.step_mode = "step"
        elif step_size == 1:
            self.step_mode = "single"
        else:
            raise ValueError("Ungueltige step_size wurde angegeben: " + str(step_size))
        
        # Learning parameters
        self.fixed_learning_rate = fixed_learning_rate
        self.use_optimizer = use_optimizer
        self.weight_decay = weight_decay
        self.use_LR_scheduler = use_LR_scheduler
        self.loss = nn.MSELoss()
        
        # For decoder start token LEGACY
        # self.decoder_start_token = nn.Parameter(torch.zeros(1, 1, step_size, hidden_dim))
        
        # Encoder components
        self.encoder_down = ConvDownsampler(C_in=in_channels, C_out=hidden_dim, stride=stride, kernel_size=3)
        self.encoder_posenc = TimePositionalEncoding(d_model=hidden_dim)
        self.encoder_afno = nn.Identity()
        
        # Axialer Transformer für den Encoder
        self.encoder_blocks = nn.ModuleList([
            AxialTransformerEncoderBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_flash_attention=self.use_flash_attention
            )
            for _ in range(num_of_layers)
        ])
        
        # Axialer Transformer für den Decoder
        self.decoder_blocks = nn.ModuleList([
            AxialTransformerDecoderBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_flash_attention=self.use_flash_attention
            )
            for _ in range(num_of_layers)
        ])
        
        # Output projection
        self.output_head = nn.Conv2d(hidden_dim, in_channels, kernel_size=1)

    def _ss_prob(self):
        """
        Returns the current probability q that a *whole batch* is trained in free-run (sequence-level SS).
        Inverse-sigmoid schedule after Bengio et al. 2015, with warmup epochs.
        """
        if getattr(self.hparams, 'ss_mode', 'off') != 'batch':
            return 0.0

        epoch = getattr(self.trainer, 'current_epoch', 0)
        if epoch < int(self.hparams.ss_warmup_epochs):
            return 0.0

        t = float(epoch - int(self.hparams.ss_warmup_epochs))
        k = max(1e-6, float(self.hparams.ss_k))
        # inverse-sigmoid core: p_raw in (0,1), small at start, grows with t
        p_raw = k / (k + math.exp(t / k))
        # map to [ss_p_start, ss_p_end], growing with epoch
        p = float(self.hparams.ss_p_start + (self.hparams.ss_p_end - self.hparams.ss_p_start) * (1.0 - p_raw))
        # clamp numeric guard
        return float(max(0.0, min(1.0, p)))
        
    def load_state_dict(self, state_dict, strict: bool = True):
        # Falls AFNO noch nicht gebaut ist, aber Gewichte im Checkpoint vorhanden sind:
        if isinstance(self.encoder_afno, nn.Identity) and "encoder_afno.weight_real" in state_dict:
            C, kh, kw = state_dict["encoder_afno.weight_real"].shape
            # Unsere AFNO2DMinimal nutzt keep_ratio=0.25 -> rekonstruierbare Minimalabmessungen
            keep_ratio = 0.25
            H2_est = max(1, int(math.ceil(kh / keep_ratio)))
            W2_est = max(1, int(math.ceil(kw / keep_ratio)))
            self.encoder_afno = AFNO2DMinimal(
                C=self.hidden_dim, H=H2_est, W=W2_est, keep_ratio=keep_ratio, dropout=0.0
            )
        return super().load_state_dict(state_dict, strict=strict)

    def encode(self, x, apply_pos_encoding=True, encode=False, time_context: Optional[torch.Tensor] = None):
        """
        Kodiert Eingabe [B,T,C,H,W] in axialen Format [N,days,timeslots,D] für die Transformer-Verarbeitung
        
        Args:
            x = [B,T,C,H,W]
            apply_pos_encoding: Optional, standardmäßig True - ob Positionsembeddings angewendet werden sollen
            encode = Is this encoding for the encoder (True) or for the target (False, no start token needed)
            time_context: [B,T,2] (hour_frac, doy_frac) in [0,1]
            
        Returns:
            y: [N,days,timeslots,D] - Kodierte Darstellung der Eingabe
            N: Anzahl der Tokens (B*H2*W2)
            spatial_dims: (H, W, H2, W2)
            start_token: Start-Token fuer den Decoder before positional encoding (nur wenn encoder=True) [N,step_size,D]
        """
        B, T, C, H, W = x.shape

        # Downsample + AFNO
        y = self.encoder_down(x)                      # [B, T, D, H2, W2]
        _, _, D, H2, W2 = y.shape
        if isinstance(self.encoder_afno, nn.Identity):
            self.encoder_afno = AFNO2DMinimal(C=D, H=H2, W=W2, keep_ratio=0.25, dropout=0.0).to(y.device)
        y = self.encoder_afno(y)                      # [B, T, D, H2, W2]

        # Tokens: [B,T,D,H2,W2] -> [N,T,D]
        y = y.reshape(B, T, D, H2 * W2).permute(0, 3, 1, 2).reshape(B * H2 * W2, T, D)

        # >>> Time2Vec additiv hinzufügen (Broadcast über alle Spatial-Tokens)
        if self.use_time2vec and (time_context is not None):
            t_emb = self.time_proj(self.t2v(time_context))          # [B,T,D]
            t_emb = t_emb.unsqueeze(1).expand(B, H2 * W2, T, D).reshape(B * H2 * W2, T, D)
            y = y + t_emb
        
        if encode:
            start_token = y[:, -self.step_size:, :]  # [N, step_size, D] fuer Decoder Start Token
        
        if apply_pos_encoding:
            y = self.encoder_posenc(y)                                  # [N,T,D]                          

        # Reorganisiere zu Tagen und Timeslots für die axiale Attention
        # Teile die Zeit T in Tage und Timeslots auf
        days = T // TIMESLOTS_DAY
        if days == 0:  # Fallback, wenn weniger als ein Tag
            days = 1
            timeslots = T
        else:
            timeslots = TIMESLOTS_DAY

        N = B * H2 * W2  # N definieren
        y = y.unsqueeze(1).reshape(N, days, timeslots, D)

        spatial_dims = (H, W, H2, W2)
        return y, N, spatial_dims, start_token if encode else None

    def _decode_step(self, encoder_output, N, spatial_dims, start_token, time_future, target=None):
        """
        Autoregressive Dekodierung mit axialer Attention.
        
        Args:
            encoder_output: [N, days, timeslots, D] - Encodierte Eingabesequenz
            N: Anzahl der Token (B * H2 * W2)
            spatial_dims: (H, W, H2, W2) 
            start_token: Start-Token fuer den Decoder (nur wenn encoder=True) [N,step_size,D]
            time_future: [B, T_fut, 2] - Zeitfeatures für Vorhersage
            target: Optional [N, days, timeslots, D] - Zielsequenz für Training
            
        Returns:
            Tensor: Dekodierten Vorhersagen [B, T_future, C, H, W]
        """
        H, W, H2, W2 = spatial_dims
        B = N // (H2 * W2)
        D = self.hidden_dim

        if target is not None:  # TRAINING (Teacher Forcing)
            # Target hat Form [N, d_target, timeslots, D]
            days_target = target.shape[1]

            # Start-Token fuer Decoder 
            start0 = start_token.unsqueeze(1)  # [N, 1, step_size, D]
            last_target_tokens = target[:, :-1, -self.step_size:, :] # Letzter Timeslot aller Tage außer letzten [N, d_target-1, step_size, D]
            start_token_days = torch.cat([start0, last_target_tokens], dim=1)  # [N, d_target, step_size, D]

            # Decoder-Eingabe: Start-Token + alle Tokens außer letzten
            decoder_input = torch.cat([start_token_days, target[:, :, :TIMESLOTS_DAY-self.step_size, :]], dim=2) # [N, d_target, 24, D]

            # Durch Decoder-Blöcke
            decoder_output = decoder_input
            for block in self.decoder_blocks:
                decoder_output = block(decoder_output, encoder_output) # [N, d_target, 24, D]

            # Output Head expects [B*T, D, H2, W2]
            output_tokens = decoder_output.reshape(B, H2, W2, days_target, TIMESLOTS_DAY, D).permute(0, 3, 4, 5, 1, 2).reshape(B*days_target*TIMESLOTS_DAY, D, H2, W2)

            y_hat = self.output_head(output_tokens)
            if self.stride > 1:
                y_hat = F.interpolate(y_hat, size=(H, W), mode='bilinear', align_corners=False)
            
            return y_hat.reshape(B, days_target * TIMESLOTS_DAY, self.in_channels, H, W)

        else:  # INFERENCE (autoregressiv)
            # 1) Ziel-Länge und Ziel-Anzahl Tage
            target_len = self.context_len if (self.prediction_total_length is None) else self.prediction_total_length

            if self.prediction_total_length is None:
                # wir erwarten context_len als Vielfaches von timeslots_enc
                if self.context_len % TIMESLOTS_DAY != 0:
                    raise ValueError("context_len muss ein Vielfaches von 24 sein.")
                target_days = self.context_len // TIMESLOTS_DAY
            else:
                if self.prediction_total_length % TIMESLOTS_DAY != 0:
                    raise ValueError("prediction_total_length muss ein Vielfaches von 24 sein.")
                target_days = self.prediction_total_length // TIMESLOTS_DAY

            if self.step_size > TIMESLOTS_DAY:
                raise ValueError(f"step_size={self.step_size} darf nicht größer als TIMESLOTS_DAY={TIMESLOTS_DAY} sein.")

            generated_days = []  # jede: [N, 1, TIMESLOTS_DAY, D]

            for day in range(target_days):
                # -- Tagespuffer in fester Länge --
                day_buf = encoder_output.new_zeros((N, 1, TIMESLOTS_DAY, D))

                # Seed für diesen Tag: Tag0 -> Encoder-Tail; sonst -> letzter step_size-Block des Vortags
                s0 = min(self.step_size, TIMESLOTS_DAY)
                if day == 0:
                    day_seed = start_token[:, :s0, :]                              # [N, s0, D]
                else:
                    day_seed = generated_days[-1][:, :, -s0:, :].reshape(N, s0, D) # [N, s0, D]
                day_buf[:, :, :s0, :] = day_seed.unsqueeze(1)

                # u-Cache je Layer (1× pro Tag befüllen)
                u_caches = [None] * len(self.decoder_blocks)

                # -- innerhalb des Tages in Schritten von step_size generieren --
                g = 0
                while g < TIMESLOTS_DAY:
                    s = min(self.step_size, TIMESLOTS_DAY - g)

                    # Optional: Time2Vec für genau die s neu zu erzeugenden Slots
                    if self.use_time2vec and (time_future is not None):
                        idx_start = min(day * TIMESLOTS_DAY + g, max(0, time_future.size(1) - s))
                        idx_end   = idx_start + s
                        t_emb = self.time_proj(self.t2v(time_future[:, idx_start:idx_end]))      # [B, s, D]
                        t_emb = t_emb.unsqueeze(1).unsqueeze(1).expand(B, H2, W2, 1, s, D).reshape(N, 1, s, D)
                        day_buf[:, :, g:g+s, :] = day_buf[:, :, g:g+s, :] + t_emb

                    # Gesamter Decoder-Input: alle fertigen Vortage + aktueller Tagespuffer
                    if len(generated_days) > 0:
                        context_days = torch.cat(generated_days, dim=1)  # [N, d_ctx, TIMESLOTS_DAY, D]
                        decoder_input = torch.cat([context_days, day_buf], dim=1)
                    else:
                        decoder_input = day_buf  # [N, 1, TIMESLOTS_DAY, D]

                    # Durch Decoder-Blöcke (u nur 1× pro Tag frisch berechnen)
                    x_dec = decoder_input
                    for li, block in enumerate(self.decoder_blocks):
                        if g == 0:
                            # Tag-Start: u frisch
                            x_dec = block(x_dec, encoder_output, u_cache=None)
                            u_full = block.compute_day_context(decoder_input)
                            u_cache = torch.zeros_like(u_full)
                            u_cache[:, -1:, :, :] = u_full[:, -1:, :, :]
                            u_caches[li] = u_cache.detach()
                        else:
                            # innerhalb des Tages: u aus Cache
                            x_dec = block(x_dec, encoder_output, u_cache=u_caches[li])

                    # Vorhersage-Chunk in den Tagespuffer schreiben
                    pred_chunk = x_dec[:, -1:, g:g+s, :]  # [N,1,s,D]
                    day_buf[:, :, g:g+s, :] = pred_chunk
                    g += s

                # Voller Tag fertig
                generated_days.append(day_buf)

            # 2) Decoder-Ausgabe stapeln
            decoder_output = torch.cat(generated_days, dim=1)  # [N, days_out, TIMESLOTS_DAY, D]
            days_out = decoder_output.shape[1]

            # 3) Reshape -> Projektion -> auf target_len begrenzen
            output_tokens = decoder_output.reshape(B, H2, W2, days_out, TIMESLOTS_DAY, D).permute(0, 3, 4, 5, 1, 2).reshape(B * days_out * TIMESLOTS_DAY, D, H2, W2)

            y_hat = self.output_head(output_tokens)
            if self.stride > 1:
                y_hat = F.interpolate(y_hat, size=(H, W), mode='bilinear', align_corners=False)
            return y_hat.reshape(B, days_out * TIMESLOTS_DAY, self.in_channels, H, W)


    def forward(self, x, target=None, time_context: Optional[torch.Tensor] = None, time_future: Optional[torch.Tensor] = None):
        """
        Forward pass mit Training/Inference Mode.
            
        Args:
            x (Tensor): Eingabe-Sequenz [B, T, C, H, W]
            target (Tensor, optional): Zukünftige Zeitschritte [B, T_future, C, H, W]
            time_context: [B,T_ctx,2] - Zeitfeatures für Kontext
            time_future: [B,T_fut,2] - Zeitfeatures für Vorhersage
                                          
        Returns:
            Tensor: Vorhersagen des Modells [B, T_future, C, H, W]
        """
        B, T_seq, C, H, W = x.shape

        # Encoder
        encoder_output, N, spatial_dims, start_token = self.encode(x, apply_pos_encoding=True, encode=True, time_context=time_context) # [N, days, timeslots, D]

        # Durch Encoder-Blöcke
        for block in self.encoder_blocks:
            encoder_output = block(encoder_output)

        if target is not None:  # Training mit Teacher Forcing
            # Target kodieren (ohne Encoder-Blocks)
            target_embedded, _, _, _ = self.encode(x=target,apply_pos_encoding=False, encode=False, time_context=time_future,)
            predictions = self._decode_step(encoder_output, N, spatial_dims, start_token, time_future, target=target_embedded)
        else:  # Inference mit autoregressiver Generierung
            predictions = self._decode_step(encoder_output, N, spatial_dims,start_token, time_future,target=None)           
        return predictions
        
    def set_model_prediction_length(self, length):
        """Configure the prediction length for the model"""
        self.prediction_total_length = length
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                    lr=self.fixed_learning_rate,
                                    weight_decay=self.weight_decay)

        if not self.use_LR_scheduler:
            return optimizer

        # sicheres total_steps bestimmen
        ts = int(getattr(self.trainer, "estimated_stepping_batches", 0))

        # Fallback: zu wenige Schritte -> kein OneCycle (verhindert ZeroDivision)
        if ts < 2:
            return optimizer

        # min. 1 Warmup-Schritt, min. 1 Anneal-Schritt
        warmup_steps = max(1, int(math.ceil(ts * 0.1)))
        if warmup_steps >= ts:
            warmup_steps = ts - 1
        pct_safe = warmup_steps / ts

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.fixed_learning_rate,
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
        # batch: (seq, time_feats)
        seq, time_feats = batch                               # [B,T,C,H,W], [B,T,2]

        x_context  = seq[:, :self.context_len]
        x_target   = seq[:, self.context_len:]  # Teacher Forcing target

        t_ctx = time_feats[:, :self.context_len] if time_feats is not None else None
        t_fut = time_feats[:, self.context_len:] if time_feats is not None else None

        mode = getattr(self.hparams, 'ss_mode', 'off')
        if mode == 'batch':
            q = self._ss_prob()
            use_free_run = torch.rand((), device=x_context.device) < q
            if use_free_run:
                preds = self(x_context, target=None,       time_context=t_ctx, time_future=t_fut)
            else:
                preds = self(x_context, target=x_target,   time_context=t_ctx, time_future=t_fut)
        else:
            preds = self(x_context, target=x_target,       time_context=t_ctx, time_future=t_fut)

        loss = self.loss(preds, x_target)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        if mode == 'batch':
            self.log("ss/q_batch_free_run", float(q), on_step=False, on_epoch=True, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        seq, time_feats = batch
        x_context  = seq[:, :self.context_len]
        x_target = seq[:, self.context_len:]

        t_ctx    = time_feats[:, :self.context_len]
        t_fut    = time_feats[:, self.context_len:]
        with torch.no_grad():
            preds = self(x_context, target=x_target, time_context=t_ctx, time_future=t_fut)
        loss = self.loss(preds, x_target)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        seq, time_feats = batch
        x_context  = seq[:, :self.context_len]
        x_target = seq[:, self.context_len:]

        t_ctx    = time_feats[:, :self.context_len]
        t_fut    = time_feats[:, self.context_len:]
        with torch.no_grad():
            preds = self(x_context, target=x_target, time_context=t_ctx, time_future=t_fut)
        loss = self.loss(preds, x_target)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss