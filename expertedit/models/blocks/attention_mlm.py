import torch
import torch.nn as nn
import math
from einops import rearrange

""" Extended attention module supporting both causal and non-causal (MLM) modes """

class FlexibleSelfAttention(nn.Module):
    """
    Self-attention that supports both causal and non-causal modes.
    For MLM training, we use bidirectional attention.
    """

    def __init__(self, config, in_dim=None):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.causal = getattr(config, 'causal', True)
        self.mlm_mode = getattr(config, 'mlm_mode', False)

        # key, query, value projections for all heads
        if in_dim is None:
            in_dim = config.n_embd
        self.kdim = config.n_embd
        self.key = nn.Linear(in_dim, config.n_embd)
        self.query = nn.Linear(in_dim, config.n_embd)
        self.value = nn.Linear(in_dim, config.n_embd)

        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)

        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        # causal mask - only used when causal=True and not in MLM mode
        if self.causal and not self.mlm_mode:
            mask = torch.tril(torch.ones(config.block_size, config.block_size))
            if hasattr(config, "n_unmasked"):
                mask[:config.n_unmasked, :config.n_unmasked] = 1
            self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))
        else:
            self.register_buffer("mask", None)

        self.n_head = config.n_head

    def forward(self, x, layer_past=None, valid_mask=None, attention_mask=None):
        """
        Forward pass with flexible attention masking.

        Args:
            x: input tensor [B, T, C]
            layer_past: past key-value pairs for generation
            valid_mask: mask for valid tokens [B, T]
            attention_mask: custom attention mask [B, T, T] or [B, 1, T, T]
        """
        B, T, C = x.size()
        C = self.kdim

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        present = torch.stack((k, v))
        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # Apply masking based on mode
        if self.mlm_mode:
            # In MLM mode, use bidirectional attention (no causal mask)
            # Only apply custom attention mask if explicitly provided
            if attention_mask is not None:
                # Ensure attention_mask has correct shape [B, nh, T, T] or [B, T, T]
                if attention_mask.dim() == 3:  # [B, T, T]
                    attention_mask = attention_mask.unsqueeze(1)  # [B, 1, T, T]
                elif attention_mask.dim() == 4:  # [B, nh, T, T]
                    pass  # Already correct shape
                else:
                    raise ValueError("attention_mask must have 3 or 4 dimensions.")

                if attention_mask is not None:
                    att = att.masked_fill(attention_mask == 0, float('-inf'))
            # For MLM without explicit mask, use full bidirectional attention (no masking)
        else:
            # Standard causal masking for autoregressive mode
            if layer_past is None and self.causal and self.mask is not None:
                att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))

        # Apply valid mask if provided
        if valid_mask is not None:
            valid_mask = rearrange(valid_mask, 'b j -> b () () j')
            att = att.masked_fill_(~valid_mask, float('-inf'))

        att = nn.functional.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, present


class CausalSelfAttention(FlexibleSelfAttention):
    """
    Backward compatibility wrapper for the original CausalSelfAttention.
    """
    def __init__(self, config, in_dim=None):
        # Force causal mode and disable MLM mode
        config.causal = True
        config.mlm_mode = False
        super().__init__(config, in_dim)

    def forward(self, x, layer_past=None, valid_mask=None):
        # Maintain original interface
        return super().forward(x, layer_past, valid_mask, attention_mask=None)
