# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from torch.nn.utils import weight_norm
import torch.nn.functional as F


class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        use_bn=False,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
        mlp_bias=True,
    ):
        super().__init__()
        nlayers = max(nlayers, 1)
        self.mlp = _build_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim=hidden_dim, use_bn=use_bn, bias=mlp_bias)
        self.apply(self._init_weights)
        self.last_layer = weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        eps = 1e-6 if x.dtype == torch.float16 else 1e-12
        x = nn.functional.normalize(x, dim=-1, p=2, eps=eps)
        x = self.last_layer(x)
        return x


class DINOHeadWithVerifier(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        use_bn=False,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
        mlp_bias=True,
        verifier_hidden_mult=4,
        use_pos_in_verifier=False,
    ):
        super().__init__()
        self.use_pos_in_verifier = use_pos_in_verifier

        nlayers = max(nlayers, 1)
        self.mlp = _build_mlp(nlayers, in_dim, bottleneck_dim,
                              hidden_dim=hidden_dim, use_bn=use_bn, bias=mlp_bias)

        self.last_layer = weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1.0)

        v_in = bottleneck_dim * (2 if use_pos_in_verifier else 1)
        self.verifier = nn.Sequential(
            nn.LayerNorm(v_in),
            nn.Linear(v_in, verifier_hidden_mult * bottleneck_dim), nn.GELU(),
            nn.Linear(verifier_hidden_mult * bottleneck_dim, bottleneck_dim), nn.GELU(),
            nn.Linear(bottleneck_dim, 1),
        )

        self.apply(self._init_weights)

        # key stability trick: make verifier start near p=0.5
        nn.init.zeros_(self.verifier[-1].weight)
        nn.init.zeros_(self.verifier[-1].bias)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos_emb=None):
        # bottleneck (unnormalized)
        bottleneck = self.mlp(x)  # [..., bottleneck_dim]

        # dino/ibot logits
        eps = 1e-6 if bottleneck.dtype == torch.float16 else 1e-12
        z = F.normalize(bottleneck, dim=-1, p=2, eps=eps)
        logits = self.last_layer(z)

        # verifier logits + probabilities
        if self.use_pos_in_verifier:
            if pos_emb is None:
                raise ValueError("use_pos_in_verifier=True but pos_emb=None")
            pos_emb = pos_emb.reshape_as(bottleneck)
            v_in = torch.cat([bottleneck, pos_emb], dim=-1)
        else:
            v_in = bottleneck

        v_logits = self.verifier(v_in)

        return logits, v_logits, bottleneck



def _build_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim=None, use_bn=False, bias=True):
    if nlayers == 1:
        return nn.Linear(in_dim, bottleneck_dim, bias=bias)
    else:
        layers = [nn.Linear(in_dim, hidden_dim, bias=bias)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim, bias=bias))
        return nn.Sequential(*layers)

