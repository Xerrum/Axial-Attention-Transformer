import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Optional
from flash_attn.modules.mha import MHA as FlashMHA

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


class TransformerBlock(nn.Module):
    """
    Transformer-Block mit optionaler Flash-Attention-Unterstuetzung.
    
    Args:
        d_model (int): Dimension des Modells
        num_heads (int): Anzahl der Attention-Heads
        dropout (float): Dropout-Rate
        use_flash_attention (bool): Ob Flash-Attention verwendet werden soll
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float, use_flash_attention: bool):
        super().__init__()
        self.use_flash_attention = use_flash_attention

        if self.use_flash_attention:
            self.mha = FlashMHA(d_model, num_heads=num_heads, dropout=dropout, causal=False, cross_attn=False)
        else:
            self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4*d_model, d_model), nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Fuehrt Self-Attention und Feed-Forward-Berechnung durch.
        
        Args:
            x (torch.Tensor): Eingabe-Embeddings [Shape: N, T, D]
                N: Anzahl der Token-Sequenzen
                T: Anzahl der Zeitschritte
                D: Embedding-Dimension
        
        Returns:
            torch.Tensor: Transformierte Embeddings [Shape: N, T, D]
        """
        if self.use_flash_attention:
            x = self.norm1(x + self.mha(x))
        else:
            attn_out, _ = self.mha(x, x, x)
            x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


class DecoderBlock(nn.Module):
    """
    Decoder-Block mit Self-Attention, Cross-Attention und optionaler Flash-Attention-Unterstuetzung.
    
    Args:
        d_model (int): Dimension des Modells
        num_heads (int): Anzahl der Attention-Heads
        dropout (float): Dropout-Rate
        use_flash_attention (bool): Ob Flash-Attention verwendet werden soll
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float, use_flash_attention: bool):
        super().__init__()
        self.use_flash_attention = use_flash_attention
        
        # Self-attention
        if self.use_flash_attention:
            self.self_mha = FlashMHA(d_model, num_heads=num_heads, dropout=dropout, causal=True, cross_attn=False)
        else:
            self.self_mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Cross-attention
        if self.use_flash_attention:
            self.cross_mha = FlashMHA(d_model, num_heads=num_heads, dropout=dropout, causal=False, cross_attn=True)
        else:
            self.cross_mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4*d_model, d_model), nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output):
        # Self-attention
        if self.use_flash_attention:
            x = self.norm1(x, self.self_mha(x))
        else:
            # explicit causal mask for PyTorch MHA (batch_first=True → mask is [T,T])
            T = x.size(1)
            causal_mask = torch.ones(T, T, device=x.device, dtype=torch.bool).triu(1)
            attn_output, _ = self.self_mha(x, x, x, attn_mask=causal_mask, need_weights=False)
            x = self.norm1(x + attn_output)
        
        # Cross-attention
        if self.use_flash_attention:
            x = self.norm2(x + self.cross_mha(x, encoder_output))
        else:
            cross_output, _ = self.cross_mha(x, encoder_output, encoder_output)
            x = self.norm2(x + cross_output)
            
        # Feed-forward
        x = self.norm3(x + self.ff(x))
        return x


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


class WeatherTransformer(pl.LightningModule):
    """
    Encoder-Decoder Weather Transformer mit AFNO-Bloecken und optionaler Flash-Attention.
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
        context_days (int): Laenge des Kontexts in Tagen (Vergangenheit)
        hours_per_day (int): Anzahl der Zeitschritte pro Tag
        prediction_days (int): Länge der Vorhersage in Tagen
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
        context_len: int = 24,   # Anzahl der Kontext-timeslots (Vergangenheit)
        hours_per_day: int = 24, # Anzahl der Zeitschritte pro Tag
        use_time2vec: bool = True,
        t2v_dim: int = 8,
        ss_mode = "batch",
        ss_p_start = 0.0,
        ss_p_end = 0.5,
        ss_k = 1000,
        ss_warmup_epochs = 0,
    ):
        super(WeatherTransformer, self).__init__()
        self.save_hyperparameters()
        self.use_flash_attention = use_flash_attention 
        self.stride = stride
        self.hidden_dim = hidden_dim
        self.in_channels = in_channels
        self.step_size = step_size
        self.hours_per_day = hours_per_day
        
        # Umrechnung von Tagen in Zeitschritten
        self.context_len = context_len  # Kontext in Zeitschritten
        self.prediction_total_length = None 
        
        self.use_time2vec = use_time2vec

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
        
        # Encoder components
        self.encoder_down = ConvDownsampler(C_in=in_channels, C_out=hidden_dim, stride=stride, kernel_size=3)
        self.encoder_afno = nn.Identity()
        self.encoder_posenc = TimePositionalEncoding(d_model=hidden_dim)
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(d_model=hidden_dim, num_heads=num_heads, dropout=dropout,
                            use_flash_attention=self.use_flash_attention)
            for _ in range(num_of_layers)
        ])
        
        # Decoder components
        self.decoder_posenc = TimePositionalEncoding(d_model=hidden_dim)
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(d_model=hidden_dim, num_heads=num_heads, dropout=dropout,
                        use_flash_attention=self.use_flash_attention)
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
  

    def set_model_prediction_length(self, length):
        """Configure the prediction length for the model in hours (legacy method)"""
        self.prediction_total_length = length


    def encode(self, x, apply_pos_encoding=True, encoder=False, time_context: Optional[torch.Tensor] = None):
        """Encodes input [B,T,C,H,W] -> Tokens [N,T,D].
           Args:
                x = [B,T,C,H,W]
                apply_pos_encoding: Optional, standardmäßig True - ob Positionsembeddings angewendet werden sollen
                encoder = Is this encoding for the encoder (True) or for the target (False, no start token needed)
                time_context: [B,T,2] (hour_frac, doy_frac) in [0,1]

           Returns:
                - y: [N,T,D] - Kodierte Darstellung der Eingabe
                - N: Anzahl der Tokens (B*H2*W2)
                - spatial_dims: (H, W, H2, W2) 
                - start_token: Start-Token fuer den Decoder (nur wenn encoder=True) [N,step_size,D]
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
        
        if encoder:
            start_token = y[:, -self.step_size:, :]  # [N, step_size, D] fuer Decoder Start Token
        
        if apply_pos_encoding:
            y = self.encoder_posenc(y)                                  # [N,T,D]

        spatial_dims = (H, W, H2, W2)
        return y, (B * H2 * W2), spatial_dims, start_token if encoder else None

    
    def _decode_step(self, encoder_output, N, spatial_dims, start_token, time_future, target=None):
        """Autoregressive Decoding fuer Training und Inferenz
            
           Diese Funktion implementiert die autoregressive Dekodierung der encodierten Eingabe,
           entweder mit Teacher Forcing im Trainingsmodus oder durch schrittweise Generierung im Inferenzmodus.
            
           Args:
               encoder_output (Tensor): Die encodierte Eingabesequenz mit Form [N, T_context, D] nach Encoder down, Encoder AFNO, Time2Vec, Pos_enc und Encoder Blocks
               N (int): Anzahl der Token (B * H2 * W2)
               spatial_dims (tuple): Raeumliche Dimensionen (H, W, H2, W2)
               target (Tensor, optional): Zielsequenz fuer Teacher Forcing mit Form [N, T_target, D] nach Encoder down, Encoder AFNO, Time2Vec
                                          None im Inferenzmodus fuer autoregressive Generierung
               start_token: Last token of encoder output for initializing decoder in inference mode (Nicht durch Encoder Blocks gewandert) [N, step_size, D]
               time_future: [B,T_fut,2] (hour_frac, doy_frac) in [0,1]
                
           Returns:
               Tensor: Die dekodierten Vorhersagen mit Form [B, T_future, C, H, W]
                      Im Trainingsmodus ist T_future gleich der Laenge des Targets
                      Im Inferenzmodus ist T_future gleich prediction_total_length oder context_len
        """
        # encoder_output shape: [N, T_context, D]
        H, W, H2, W2 = spatial_dims
        B = N // (H2 * W2)
        D = self.hidden_dim

        if target is not None:  # TRAINING (Teacher Forcing)
            N, T_future, D = target.shape

            # first token in target_tokens is last token from context and used as start token
            decoder_input = torch.cat([start_token, target[:, :-self.step_size, :]], dim=1)
            decoder_input = self.decoder_posenc(decoder_input)

            for block in self.decoder_blocks:
                decoder_input = block(decoder_input, encoder_output)

            dec_out = decoder_input
            output_tokens = dec_out.reshape(B, H2, W2, T_future, D).permute(0, 3, 4, 1, 2)  # [B,T_f,D,H2,W2]
            y_hat = self.output_head(output_tokens.reshape(B * T_future, D, H2, W2))
            if self.stride > 1:
                y_hat = F.interpolate(y_hat, size=(H, W), mode='bilinear', align_corners=False)
            return y_hat.view(B, T_future, self.in_channels, H, W)

        else:  # INFERENCE (autoregressiv)
            all_tokens = start_token  # [N, step_size, D]

            # Standardmäßig Vorhersage für prediction_days Tage
            target_len = self.prediction_total_length

            if self.step_mode == "single":
                for _ in range(target_len):
                    current_input = self.decoder_posenc(all_tokens)

                    # >>> add Time2Vec for the actually generated length L
                    if self.use_time2vec and (time_future is not None):
                        L = current_input.size(1)
                        t_emb_full = self.time_proj(self.t2v(time_future[:, :L]))   # [B,L,D]
                        t_emb_full = t_emb_full.unsqueeze(1).expand(B, H2 * W2, L, D).reshape(N, L, D)
                        current_input = current_input + t_emb_full

                    current_output = current_input
                    for block in self.decoder_blocks:
                        current_output = block(current_output, encoder_output)

                    next_token = current_output[:, -1:, :]  # [N,1,D]
                    all_tokens = torch.cat([all_tokens, next_token], dim=1)

            decoder_output = all_tokens[:, self.step_size:, :]  # [N, target_len, D]
            decoder_interpol = decoder_output.view(B, H2, W2, target_len, D).permute(0, 3, 4, 1, 2)
            y_hat = self.output_head(decoder_interpol.reshape(B * target_len, D, H2, W2))
            if self.stride > 1:
                y_hat = F.interpolate(y_hat, size=(H, W), mode='bilinear', align_corners=False)
            return y_hat.view(B, target_len, self.in_channels, H, W)


    def forward(self, x, target=None,
            time_context: Optional[torch.Tensor] = None,
            time_future: Optional[torch.Tensor] = None):
        """Forward pass mit Training/Inference Mode wie im LearnTransformer
            
        Diese Methode fuehrt die Vorwaertsberechnung des Encoder-Decoder Modells durch und
        unterscheidet automatisch zwischen Training (mit Teacher Forcing) und 
        Inferenz (mit autoregressiver Generierung).
            
        Args:
            x (Tensor): Eingabe-Sequenz mit Form [B, T, C, H, W], wobei
                        B = Batch-Groesse, T = Zeitschritte, C = Kanaele, H/W = raeumliche Dimensionen
            target (Tensor, optional): Im Trainingsmodus: zukuenftige Zeitschritte mit
                                        Form [B, T_future, C, H, W]. Wird fuer Teacher Forcing verwendet.
                                        Im Inferenzmodus: None (autoregressive Generierung).
            time_context: [B,T_ctx,2]  (hour_frac, doy_frac)
            time_future:  [B,T_fut,2]
                                          
        Returns:
            Tensor: Vorhersagen des Modells mit Form [B, T_future, C, H, W]
                    Im Trainingsmodus: T_future entspricht der Laenge des Targets
                    Im Inferenzmodus: T_future wird durch prediction_total_length bestimmt
        """
        B, T_seq, C, H, W = x.shape

        # Encoder part
        encoder_output, N, spatial_dims, start_token = self.encode(x, apply_pos_encoding=True, encoder=True, time_context=time_context)

        for block in self.encoder_blocks:
            encoder_output = block(encoder_output)

        if target is not None:  # Training
            target_embedded, _, _, _ = self.encode(target, apply_pos_encoding=False, encoder=False, time_context=time_future)
            predictions = self._decode_step(encoder_output, N, spatial_dims, start_token, time_future=None, target=target_embedded)
        else:                   # Inference
            predictions = self._decode_step(encoder_output, N, spatial_dims, start_token, time_future=time_future, target=None)
        return predictions
        
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
        seq, time_feats = batch  # [B,T,H,W,C]-abhängig von deiner Vorverarbeitung
        x_context  = seq[:, :self.context_len]
        x_target   = seq[:, self.context_len:]
        t_context  = time_feats[:, :self.context_len] if time_feats is not None else None
        t_future   = time_feats[:, self.context_len:] if time_feats is not None else None

        mode = getattr(self.hparams, 'ss_mode', 'off')
        if mode == 'batch':
            q = self._ss_prob()
            use_free_run = torch.rand((), device=x_context.device) < q
            if use_free_run:
                preds = self(x_context, target=None, time_context=t_context, time_future=t_future)
            else:
                preds = self(x_context, target=x_target, time_context=t_context, time_future=t_future)
        else:
            preds = self(x_context, target=x_target, time_context=t_context, time_future=t_future)

        loss = self.loss(preds, x_target)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        if mode == 'batch':
            self.log("ss/q_batch_free_run", float(q), on_step=False, on_epoch=True, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        seq, time_feats = batch
        x_context  = seq[:, :self.context_len]
        x_target = seq[:, self.context_len:] # Take last token from context and feed to target for teacher forcing as start token

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
        x_target = seq[:, self.context_len:] # Take last token from context and feed to target for teacher forcing as start token

        t_ctx    = time_feats[:, :self.context_len]
        t_fut    = time_feats[:, self.context_len:]
        with torch.no_grad():
            preds = self(x_context, target=x_target, time_context=t_ctx, time_future=t_fut)
        loss = self.loss(preds, x_target)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss