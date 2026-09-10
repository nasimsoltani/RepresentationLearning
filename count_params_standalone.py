"""
Standalone param counter — no project imports, no heavy deps.
Uses exact experiment configs: d2=256, encoder_num_blocks=2,
task_adaptive_encoder=True, slice_len=1024, latent_dim=512.
"""
import torch
import torch.nn as nn


# ─── TaskAdaptiveEncoder (copied from code/rep_lr/models.py) ─────────────────
class TaskAdaptiveEncoder(nn.Module):
    def __init__(self, slice_size, output_dim=128, dropout=0.25, num_blocks=3, input_channels=2):
        super().__init__()
        self.output_dim = output_dim
        self.num_blocks = num_blocks
        self.layers = nn.ModuleList()
        channels = [input_channels, 32, 64, 64]
        for i in range(num_blocks):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            block = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=7, padding=3),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.Conv1d(out_ch, out_ch, kernel_size=5, padding=2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(slice_size // (2 ** (i + 1))) if i < num_blocks - 1 else nn.Identity()
            )
            self.layers.append(block)
        actual_output_channels = channels[num_blocks]
        self.feature_projection = nn.Sequential(
            nn.AdaptiveAvgPool1d(32),
            nn.Flatten(),
            nn.Linear(actual_output_channels * 32, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2 * output_dim)
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.feature_projection(x).view(x.size(0), 2, self.output_dim)


# ─── Decoder / Adversary (copied from code/dra_1/models.py) ──────────────────
class Decoder(nn.Module):
    def __init__(self, latent_dim, dropout=0.1):
        super().__init__()
        self.shared_decoder = nn.Sequential(
            nn.Linear(latent_dim, 512), nn.LeakyReLU(0.2), nn.Dropout(dropout),
            nn.Linear(512, 1024),       nn.LeakyReLU(0.2), nn.Dropout(dropout),
            nn.Linear(1024, 1024),      nn.LeakyReLU(0.2), nn.Dropout(dropout),
            nn.Unflatten(1, (64, 16)),
        )
        self.shared_conv = nn.Sequential(
            nn.ConvTranspose1d(64, 128, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(128, 64, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(64,  32, 4, stride=2, padding=1), nn.ReLU(),
        )
        self.rf_head = nn.Sequential(
            nn.ConvTranspose1d(32, 16, 3, padding=1), nn.LeakyReLU(0.2), nn.BatchNorm1d(16),
            nn.ConvTranspose1d(16,  8, 3, padding=1), nn.LeakyReLU(0.2), nn.BatchNorm1d(8),
            nn.ConvTranspose1d(8,   2, 3, padding=0),
            nn.Flatten(),
            nn.Linear(260, 1024),  nn.LeakyReLU(0.2), nn.BatchNorm1d(1024),
            nn.Linear(1024, 1024), nn.LeakyReLU(0.2), nn.BatchNorm1d(1024),
            nn.Linear(1024, 2048),
            nn.Unflatten(1, (2, 1024)),
        )
        self.cfo_head = nn.Sequential(
            nn.ConvTranspose1d(32, 16, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(16,  8, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(8,   2, 3, padding=0),
            nn.Upsample(size=160, mode='linear', align_corners=False),
        )
        self.channel_head = nn.Sequential(
            nn.ConvTranspose1d(32, 16, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(16,  8, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(8,   2, 3, padding=0),
            nn.Upsample(size=160, mode='linear', align_corners=False),
        )


def count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def breakdown(model, name):
    print(f"\n{'='*62}")
    print(f"  {name}")
    print(f"{'='*62}")
    for n, m in model.named_modules():
        if not list(m.children()):
            p = sum(x.numel() for x in m.parameters())
            if p:
                print(f"  {n:<52s}  {p:>8,}")
    print(f"  {'─'*62}")
    print(f"  {'TOTAL trainable':<52s}  {count(model):>8,}")


# ── exact experiment config ───────────────────────────────────────────────────
SLICE_LEN  = 1024
D2         = 256
NUM_BLOCKS = 2
LATENT_DIM = 2 * D2   # 512
DROPOUT    = 0.1

print("\n" + "="*62)
print("  Experiment config")
print("="*62)
print(f"  slice_len={SLICE_LEN}  d2={D2}  num_blocks={NUM_BLOCKS}  latent_dim={LATENT_DIM}")

enc = TaskAdaptiveEncoder(SLICE_LEN, output_dim=D2, dropout=DROPOUT, num_blocks=NUM_BLOCKS)
dec = Decoder(latent_dim=LATENT_DIM, dropout=DROPOUT)

breakdown(enc, "Encoder  (TaskAdaptiveEncoder, num_blocks=2)")
breakdown(dec, "Adversary / Decoder  (latent_dim=512)")

print("\n" + "="*62)
print("  SUMMARY")
print("="*62)
enc_p = count(enc)
dec_p = count(dec)
print(f"  Encoder (TaskAdaptiveEncoder)         {enc_p:>10,}")
print(f"  Adversary / Decoder  (total)          {dec_p:>10,}")
print(f"    shared_decoder (MLP)                {count(dec.shared_decoder):>10,}")
print(f"    shared_conv  (ConvTranspose)        {count(dec.shared_conv):>10,}")
print(f"    rf_head                             {count(dec.rf_head):>10,}")
print(f"    cfo_head                            {count(dec.cfo_head):>10,}")
print(f"    channel_head                        {count(dec.channel_head):>10,}")
print()
