import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class scoring_head(nn.Module):
    def __init__(self, depth, input_dim, dim, input_len=2, num_scores=1, bidirection=False, noise_std=0.0):
        super().__init__()
        self.video_proj = nn.Linear(4096, input_dim)
        self.fusion_linear = nn.Linear(input_dim * 2, input_dim)
        # cross-modal attention: audio queries attend to video features
        self.cross_attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=8, batch_first=True)
        # gated fusion layers
        self.gate_linear = nn.Linear(input_dim * 2, 1)
        self.gate_fusion = nn.Linear(input_dim, input_dim)
        self.rnn = nn.LSTM(input_dim, input_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        # positional encoding for transformer
        self.pos_encoder = PositionalEncoding(input_dim)
        # residual + layer norm for fused sequence
        self.layer_norm_fused = nn.LayerNorm(input_dim)
        self.output = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(input_dim, num_scores)
        )
        self.noise_std = noise_std

    def _encode_audio(self, audio):
        seq = audio.squeeze(2)
        if self.training and self.noise_std > 0:
            seq = seq + self.noise_std * torch.randn_like(seq)
        return seq

    def _encode_video(self, video):
        seq = video.mean(dim=2)
        seq = self.video_proj(seq)
        if self.training and self.noise_std > 0:
            seq = seq + self.noise_std * torch.randn_like(seq)
        return seq

    def _fuse(self, audio_seq, video_seq, seq_len):
        audio_seq = audio_seq[:, :seq_len, :]
        video_seq = video_seq[:, :seq_len, :]
        attn_seq, _ = self.cross_attn(audio_seq, video_seq, video_seq)
        cat = torch.cat([attn_seq, video_seq], dim=2)
        gate = torch.sigmoid(self.gate_linear(cat))
        fused = gate * attn_seq + (1 - gate) * video_seq
        fused = self.dropout(self.gate_fusion(fused))
        fused_res = self.pos_encoder(fused)
        return self.layer_norm_fused(fused_res + fused)

    def _apply_rnn(self, seq, audio_len, seq_len):
        lengths = torch.clamp(torch.tensor(audio_len), max=seq_len).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(seq, lengths, batch_first=True, enforce_sorted=False)
        _, (hn, _) = self.rnn(packed)
        h = torch.cat([hn[-2], hn[-1]], dim=1)
        return self.dropout(h)

    def forward(self, audio_feature, video_feature, audio_len):
        audio_seq = self._encode_audio(audio_feature)
        video_seq = self._encode_video(video_feature)
        seq_len = min(audio_seq.size(1), video_seq.size(1))
        fused = self._fuse(audio_seq, video_seq, seq_len)
        rnn_out = self._apply_rnn(fused, audio_len, seq_len)
        return self.output(rnn_out).squeeze(1)
