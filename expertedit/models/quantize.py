# Adapted from PoseGPT's quantizer (NAVER Corp., CC BY-NC-SA 4.0).
# The original implementation credits CompVis/taming-transformers; see NOTICE.
"""Inference-only product quantization with the original checkpoint keys."""

import torch
from torch import nn


class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta=0.25, nbooks=1, balance=False):
        super().__init__()
        if n_e % nbooks or e_dim % nbooks:
            raise ValueError("Codebook count must divide vocabulary and latent dimensions.")
        if balance:
            raise ValueError("Balanced codebook updates are outside the inference release.")
        self.n_e = n_e
        self.e_dim = e_dim
        self.nbooks = nbooks
        self.n_e_i = n_e // nbooks
        self.embeddings = nn.ModuleDict({
            str(i): nn.Embedding(self.n_e_i, e_dim // nbooks)
            for i in range(nbooks)
        })
        for embedding in self.embeddings.values():
            nn.init.uniform_(embedding.weight, -1 / self.n_e_i, 1 / self.n_e_i)

    def forward(self, z):
        if z.shape[-1] != self.e_dim:
            raise ValueError("Latent dimension does not match the quantizer.")
        indices = []
        quantized = []
        for i, part in enumerate(z.split(self.e_dim // self.nbooks, dim=-1)):
            table = self.embeddings[str(i)]
            flat = part.reshape(-1, part.shape[-1])
            distances = (
                flat.square().sum(1, keepdim=True)
                - 2 * flat @ table.weight.t()
                + table.weight.square().sum(1).unsqueeze(0)
            )
            index = distances.argmin(1).reshape(part.shape[:-1])
            indices.append(index.unsqueeze(-1))
            quantized.append(table(index))
        return torch.cat(quantized, dim=-1), torch.cat(indices, dim=-1)

    def get_codebook_entry(self, indices, eos_mask=None):
        # Invalid padding tokens are zeroed by the tokenizer's validity mask.
        indices = indices.clamp(0, self.n_e_i - 1)
        return torch.cat([
            self.embeddings[str(i)](indices[..., i])
            for i in range(self.nbooks)
        ], dim=-1)
