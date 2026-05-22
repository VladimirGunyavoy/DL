import torch
import torch.nn as nn


class DecisionTransformer(nn.Module):
    def __init__(self, obs_dim=3, act_dim=2, embed_dim=64, n_heads=4,
                 n_layers=3, dropout=0.1, context_len=20, max_timestep=1000):
        super().__init__()
        self.context_len = context_len
        self.embed_dim   = embed_dim

        self.obs_embed = nn.Linear(obs_dim, embed_dim)
        self.act_embed = nn.Linear(act_dim, embed_dim)
        self.rtg_embed = nn.Linear(1, embed_dim)
        self.pos_embed = nn.Embedding(max_timestep + 1, embed_dim)

        self.embed_ln = nn.LayerNorm(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=4 * embed_dim, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.action_head = nn.Linear(embed_dim, act_dim)

    def forward(self, obs, actions, rtg, timesteps):
        # obs:       (B, K, obs_dim)
        # actions:   (B, K, act_dim)
        # rtg:       (B, K, 1)
        # timesteps: (B, K)
        B, K, _ = obs.shape

        pos     = self.pos_embed(timesteps)
        obs_emb = self.obs_embed(obs)     + pos
        act_emb = self.act_embed(actions) + pos
        rtg_emb = self.rtg_embed(rtg)     + pos

        # interleave: [RTG_0, obs_0, act_0, RTG_1, obs_1, act_1, ...]
        tokens = torch.stack([rtg_emb, obs_emb, act_emb], dim=2)   # (B, K, 3, D)
        tokens = self.embed_ln(tokens.reshape(B, 3 * K, self.embed_dim))

        seq  = 3 * K
        mask = torch.triu(torch.full((seq, seq), float('-inf'), device=obs.device), diagonal=1)

        out     = self.transformer(tokens, mask=mask)   # (B, 3K, D)
        obs_out = out[:, 1::3, :]                       # obs tokens: 1, 4, 7, ...
        return self.action_head(obs_out)                # (B, K, act_dim)
