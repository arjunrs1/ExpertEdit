# Adapted from the ExpertEdit PoseGPT extension; see LICENSE and NOTICE.
"""
Extended PoseGPT model with MLM (Masked Language Modeling) support.
Bidirectional motion token prediction and infilling.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

from .blocks.mingpt import GPTConfig
from .blocks.attention_mlm import FlexibleSelfAttention


def freeze(model):
    for param in model.parameters():
        param.requires_grad = False


def pad(z):
    """ Pad by one for auto-regression: first value is predicted from nothing. """
    b, _, c = z.shape
    pad = torch.zeros((b, 1, c)).long().to(z.device)
    return torch.cat((pad, z), dim=1)


class MLMBlock(nn.Module):
    """Transformer block with flexible attention for MLM support"""

    def __init__(self, config, in_factor=None, out_factor=None):
        super().__init__()
        in_dim = config.n_embd * (in_factor if in_factor is not None else 1)
        out_dim = config.n_embd * (out_factor if out_factor is not None else 1)
        self.ln1 = nn.LayerNorm(in_dim)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = FlexibleSelfAttention(config, in_dim=in_dim if in_factor is not None else None)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, out_dim),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, layer_past=None, return_present=False, valid_mask=None,
                attention_mask=None, in_residual=True, out_residual=True):
        """Forward pass with optional attention mask for MLM"""
        if return_present:
            assert not self.training

        # layer past: tuple of length two with B, nh, T, hs
        attn, present = self.attn(self.ln1(x), layer_past=layer_past,
                                  valid_mask=valid_mask, attention_mask=attention_mask)
        x = x + attn if in_residual else attn
        x = x + self.mlp(self.ln2(x)) if out_residual else self.mlp(self.ln2(x))

        if layer_past is not None or return_present:
            return x, present
        return x


class MLMGPT(nn.Module):
    """
    GPT-like model with MLM support for human motion.
    Includes the masked transformer architecture used by ExpertEdit.
    """

    def __init__(self, vocab_size, block_size, n_layer=12, n_head=8, n_embd=256,
                 embd_pdrop=0., resid_pdrop=0., attn_pdrop=0., n_unmasked=0,
                 n_codebook=1, embed_every_step=False, embed_seqlen=False, concat_emb=False,
                 head_type='fc_wo_bias', pos_emb='scratch', class_conditional=True,
                 causal=True, gen_eos=False, mlm_mode=False, mask_token_id=-1):
        super().__init__()

        self.class_conditional = class_conditional
        self.gen_eos = gen_eos
        self.mlm_mode = mlm_mode
        self.mask_token_id = mask_token_id
        self.n_codebook = n_codebook

        # Create config with only standard GPT parameters
        config = GPTConfig(vocab_size=vocab_size, block_size=block_size,
                           embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop,
                           n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                           n_unmasked=n_unmasked, causal=causal and not mlm_mode)

        # Store MLM-specific settings separately
        self.mlm_mode = mlm_mode
        self.mask_token_id = mask_token_id

        # Vocabulary management: Input vocab includes MASK token, output vocab excludes it
        if mlm_mode:
            # Input vocabulary: VQ codebook + MASK_ID (one per codebook)
            self.input_vocab_size = vocab_size + n_codebook  # MASK_ID for each codebook
            # Output vocabulary: VQ codebook only (model can never generate MASK)
            self.output_vocab_size = vocab_size
            # Actual MASK token IDs (one per codebook)
            self.actual_mask_ids = [vocab_size // n_codebook + i for i in range(n_codebook)]
        else:
            self.input_vocab_size = vocab_size + n_codebook + (1 if gen_eos else 0)
            self.output_vocab_size = vocab_size + n_codebook + (1 if gen_eos else 0)
            self.actual_mask_ids = []

        # Input embeddings (include MASK token if MLM)
        self.tok_emb = nn.Embedding(self.input_vocab_size, config.n_embd // n_codebook)

        # positional embedding
        self.set_pos_embedding(pos_emb, config)

        self.drop = nn.Dropout(config.embd_pdrop)

        # Transformer torso - use ModuleList instead of Sequential to pass attention masks
        # Use MLMBlock which supports attention_mask parameter for proper masking
        self.blocks = nn.ModuleList([MLMBlock(config) for _ in range(config.n_layer)])

        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.set_output_head(n_codebook, head_type, config)

        self.block_size = config.block_size
        self.apply(self._init_weights)
        self.config = config
        self.embed_every_step = embed_every_step
        self.embed_seqlen = embed_seqlen
        self.concat_emb = concat_emb


        if self.concat_emb:
            emb_count = 2 + int(class_conditional) + int(embed_seqlen)
        else:
            emb_count = 3 + int(embed_seqlen)
        self.fc_emb = nn.Linear((emb_count) * config.n_embd, config.n_embd) if embed_every_step else None

    def set_pos_embedding(self, pos_emb, config):
        if 'scratch' in pos_emb:
            self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        elif 'sine' in pos_emb:
            n_seqlens, gpt_nembd = config.block_size, config.n_embd
            pe = torch.zeros(n_seqlens, gpt_nembd)
            position = torch.arange(0, n_seqlens, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, gpt_nembd, 2).float() * (-math.log(10000.0) / gpt_nembd))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.pos_emb = nn.Parameter(pe.unsqueeze(0))
        else:
            raise NameError("Unknown embedding")
        if 'frozen' in pos_emb:
            self.pos_emb.requires_grad = False

    def set_output_head(self, n_codebook, head_type, config):
        self.list_head = nn.ModuleList()
        for cb in range(n_codebook):
            if self.mlm_mode:
                # Output head excludes MASK token - only VQ codebook tokens
                nb_out = self.output_vocab_size // n_codebook
            else:
                nb_out = config.vocab_size // n_codebook + (1 if self.gen_eos else 0)

            if head_type == 'fc_wo_bias':
                head = nn.Linear(config.n_embd, nb_out, bias=False)
            elif head_type == 'mlp':
                head = nn.Sequential(
                    nn.Linear(config.n_embd, config.n_embd),
                    nn.ReLU(),
                    nn.Linear(config.n_embd, nb_out)
                )
            else:
                raise ValueError(f"Unknown head type: {head_type}")
            self.list_head.append(head)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_token_embedding_shape(self, batch_size, n_codebook, idx):
        token_embeddings = self.tok_emb(torch.ones((batch_size * n_codebook), requires_grad=False).to(idx.device).long())
        token_embeddings = token_embeddings.reshape(batch_size, 1, n_codebook, -1)
        token_embeddings = token_embeddings.flatten(2)[:, :0, :]
        return token_embeddings

    def cat_indices(self, idx):
        idx = torch.cat([x + (i * self.config.vocab_size // idx.shape[-1])
                          for i, x in enumerate(idx.split(1, dim=-1))], dim=-1)
        return idx

    def forward(self, idx, actions_emb=None, seqlens_emb=None, attention_mask=None):
        """
        Forward pass supporting both autoregressive and MLM modes.

        Args:
            idx: [batch_size, seq_len, n_codebook]
            actions_emb: [batch_size, 1, emb_dim]
            seqlens_emb: [batch_size, 1, emb_dim]
            attention_mask: [batch_size, seq_len, seq_len] for MLM mode
            mask_positions: [batch_size, seq_len] boolean mask indicating which positions are masked
        """
        assert not (self.embed_seqlen and not self.embed_every_step), "Seqlength embedding is only coded with embed_every_step"
        assert not (self.embed_seqlen and seqlens_emb is None), "This model needs to be conditioned on sequence length"

        idx = idx.clone()  # Mapping MASK IDs must not mutate caller-owned tokens.
        batch_size, seq_len, n_codebook = idx.size()

        # Handle mask tokens for MLM
        if self.mlm_mode:
            # Map the generic mask_token_id to actual mask token indices for each codebook
            for cb in range(n_codebook):
                cb_mask_positions = (idx[:, :, cb] == self.mask_token_id)
                if cb_mask_positions.any():
                    # Use the actual mask token ID for this codebook
                    actual_mask_id = self.actual_mask_ids[cb]
                    idx[:, :, cb] = torch.where(cb_mask_positions, actual_mask_id, idx[:, :, cb])

        # Ensure all indices are in valid range for input embeddings
        max_valid_idx = self.tok_emb.num_embeddings - 1
        idx = torch.clamp(idx, 0, max_valid_idx)

        # forward the GPT model
        idx = self.cat_indices(idx)

        if seq_len != 0:
            token_embeddings = self.tok_emb(idx.flatten(1))
            token_embeddings = token_embeddings.reshape(batch_size, seq_len, n_codebook, -1)
            token_embeddings = token_embeddings.flatten(2)
        else:
            token_embeddings = self.get_token_embedding_shape(batch_size, n_codebook, idx)

        # prepend explicit embeddings (different for MLM vs autoregressive)
        if self.mlm_mode:
            # For MLM, we don't need causal padding
            pass
        else:
            # For autoregressive, prepend action embeddings or pad
            if actions_emb is not None and not self.embed_every_step:
                token_embeddings = torch.cat((actions_emb, token_embeddings), dim=1)
            else:
                token_embeddings = pad(token_embeddings)

        t = token_embeddings.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        position_embeddings = self.pos_emb[:, :t, :]

        list_emb = [token_embeddings, position_embeddings.repeat(batch_size, 1, 1)]
        if actions_emb is not None:
            if self.mlm_mode:
                list_emb += [actions_emb.repeat(1, t, 1)] if self.embed_every_step else []
            else:
                list_emb += [actions_emb.repeat(1, t, 1) if actions_emb.shape[1] == 1 else pad(actions_emb)[:,:t]] if self.embed_every_step else []
        list_emb += [seqlens_emb.repeat(1, t, 1)] if self.embed_seqlen and self.embed_every_step else []

        if self.concat_emb:
            x = self.fc_emb(torch.cat(list_emb, -1))
        else:
            x = torch.stack(list_emb, -1).sum(-1)

        x = self.drop(x)

        # Forward through transformer blocks - manually iterate to pass attention_mask
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)

        x = self.ln_f(x)
        logits = torch.stack([head(x) for head in self.list_head], 2)

        # In MLM mode, ensure we never predict mask tokens by setting their logits to -inf
        if self.mlm_mode:
            for cb in range(self.n_codebook):
                # The output head doesn't include mask tokens, so no masking needed
                pass

        return logits

    @torch.no_grad()
    def one_shot_infill(self, idx, mask_positions, actions_emb=None, seqlens_emb=None, temperature=1.0):
        """
        One-shot infill: predict all masked positions in a single forward pass.

        Args:
            idx: [batch_size, seq_len, n_codebook] with mask tokens at masked positions
            mask_positions: [batch_size, seq_len] boolean mask
            temperature: sampling temperature

        Returns:
            completed_idx: [batch_size, seq_len, n_codebook] with predictions filled in
        """
        assert self.mlm_mode, "One-shot infill only works in MLM mode"

        # Forward pass
        logits = self.forward(idx, actions_emb, seqlens_emb)

        # Sample predictions
        if temperature == 0.0:
            predictions = torch.argmax(logits, dim=-1)
        else:
            # logits shape: [batch_size, seq_len, n_codebook, vocab_size]
            # Need to sample for each codebook position independently
            batch_size, seq_len, n_codebook, vocab_size = logits.shape
            logits_scaled = logits / temperature

            # Reshape to [batch*seq*codebook, vocab_size] for multinomial sampling
            logits_flat = logits_scaled.reshape(-1, vocab_size)
            probs = F.softmax(logits_flat, dim=-1)
            sampled_flat = torch.multinomial(probs, 1).squeeze(-1)  # [batch*seq*codebook]

            # Reshape back to [batch_size, seq_len, n_codebook]
            predictions = sampled_flat.view(batch_size, seq_len, n_codebook)

        # Ensure predictions are in valid VQVAE vocabulary range
        # For each codebook, clamp to [0, vqvae_vocab_size-1]
        vqvae_vocab_size = self.output_vocab_size // self.n_codebook
        predictions = torch.clamp(predictions, 0, vqvae_vocab_size - 1)

        # Fill in masked positions with predictions
        completed_idx = idx.clone()
        mask_expanded = mask_positions.unsqueeze(-1).expand(-1, -1, self.n_codebook)
        completed_idx = torch.where(mask_expanded, predictions, completed_idx)

        return completed_idx

    @torch.no_grad()
    def iterative_infill(self, idx, mask_positions, actions_emb=None, seqlens_emb=None,
                        temperature=1.0, confidence_thresholds=[0.3, 0.6, 1.0], max_iters=10):
        """
        Iterative infill (MaskGIT style): Fill in most confident predictions first.

        Args:
            idx: [batch_size, seq_len, n_codebook] with mask tokens at masked positions
            mask_positions: [batch_size, seq_len] boolean mask
            confidence_thresholds: thresholds for progressive unmasking
            max_iters: maximum number of iterations

        Returns:
            completed_idx: [batch_size, seq_len, n_codebook] with predictions filled in
        """
        assert self.mlm_mode, "Iterative infill only works in MLM mode"

        current_idx = idx.clone()
        current_mask = mask_positions.clone()

        for iter_idx in range(max_iters):
            if not current_mask.any():
                break

            # Forward pass
            logits = self.forward(current_idx, actions_emb, seqlens_emb)

            # Get confidence scores (max probability)
            probs = F.softmax(logits, dim=-1)
            max_probs, predictions = torch.max(probs, dim=-1)

            # Ensure predictions are in valid VQVAE vocabulary range
            vqvae_vocab_size = self.output_vocab_size // self.n_codebook
            predictions = torch.clamp(predictions, 0, vqvae_vocab_size - 1)

            # Average confidence across codebooks for each position
            confidence = max_probs.mean(dim=-1)  # [batch_size, seq_len]

            # Determine threshold for this iteration
            progress = iter_idx / max_iters
            threshold_idx = min(int(progress * len(confidence_thresholds)), len(confidence_thresholds) - 1)
            threshold = confidence_thresholds[threshold_idx]

            # Select positions to unmask (high confidence + currently masked)
            unmask_positions = (confidence >= threshold) & current_mask

            if not unmask_positions.any():
                # If no positions meet threshold, unmask the most confident ones
                n_unmask = max(1, int(0.1 * current_mask.sum().item()))  # Unmask at least 10%
                _, top_indices = torch.topk(confidence[current_mask], min(n_unmask, current_mask.sum().item()))
                # Convert back to full indices
                mask_indices = torch.where(current_mask)
                unmask_positions = torch.zeros_like(current_mask)
                for i in range(len(top_indices)):
                    batch_idx = mask_indices[0][top_indices[i]]
                    pos_idx = mask_indices[1][top_indices[i]]
                    unmask_positions[batch_idx, pos_idx] = True

            # Fill in selected positions
            unmask_expanded = unmask_positions.unsqueeze(-1).expand(-1, -1, self.n_codebook)
            current_idx = torch.where(unmask_expanded, predictions, current_idx)
            current_mask = current_mask & ~unmask_positions

        # Complete any tokens still masked after the confidence schedule.
        if current_mask.any():
            current_idx = self.one_shot_infill(
                current_idx, current_mask, actions_emb, seqlens_emb, temperature
            )
        return current_idx


class poseGPTMLM(nn.Module):
    """
    Extended PoseGPT with MLM capability.
    Wraps the frozen motion tokenizer and masked transformer for infilling.
    """

    def __init__(self, *, n_e, gpt_nembd, gpt_blocksize, gpt_nhead, gpt_nlayer, vqvae,
                 causal_gpt, n_actions=85, clip_dim=512, mlm_mode=False, **kwargs):
        super(poseGPTMLM, self).__init__()

        self.mlm_mode = mlm_mode
        self.mask_token_id = kwargs.get('mask_token_id', -1)

        gpt_config = {'vocab_size': n_e,
                      'block_size': gpt_blocksize,
                      'n_layer': gpt_nlayer,
                      'n_head': gpt_nhead,
                      'n_embd': gpt_nembd,
                      'n_codebook': kwargs['n_codebook'],
                      'embd_pdrop': kwargs['gpt_embd_pdrop'],
                      'resid_pdrop': kwargs['gpt_resid_pdrop'],
                      'attn_pdrop': kwargs['gpt_attn_pdrop'],
                      'embed_every_step': kwargs['embed_every_step'],
                      'embed_seqlen': kwargs['seqlen_conditional'],
                      'concat_emb': kwargs['concat_emb'],
                      'head_type': kwargs['head_type'],
                      'pos_emb': kwargs['pos_emb'],
                      'causal': causal_gpt and not mlm_mode,  # Non-causal for MLM
                      'class_conditional': kwargs['class_conditional'],
                      'gen_eos': kwargs['gen_eos'],
                      'mlm_mode': mlm_mode,
                      'mask_token_id': self.mask_token_id}

        self.causal = causal_gpt and not mlm_mode
        self.gen_eos = kwargs['gen_eos']
        self.gpt = MLMGPT(**gpt_config)
        self.class_conditional = kwargs['class_conditional']
        self.seqlen_conditional = kwargs['seqlen_conditional']

        # The auto-encoder is frozen
        self.vqvae = vqvae
        self.vqvae.eval()
        freeze(self.vqvae)

        self.vocab_size = n_e
        self.autoreg_pq = kwargs['autoreg_pq']

        if self.autoreg_pq:
            raise ValueError("Autoregressive product quantization is outside this release.")

        self.set_action_embedding(n_actions, gpt_nembd, kwargs)
        n_seqlens = 1024
        self.set_seqlen_embedding(n_seqlens, gpt_nembd, kwargs)
        self.factor = kwargs['factor']

        if kwargs.get('use_clip_conditioning', False):
            raise ValueError('CLIP conditioning is outside this release.')


    def set_action_embedding(self, n_actions, gpt_nembd, kwargs):
        self.n_actions = n_actions
        self.action_emb_type = kwargs['action_emb']
        if self.action_emb_type != 'scratch':
            raise ValueError('The inference release supports learned action embeddings.')
        self.action_emb = nn.Embedding(n_actions, gpt_nembd)

    def set_seqlen_embedding(self, n_seqlens, gpt_nembd, kwargs):
        self.sample_eos_force = kwargs['eos_force']
        self.seqlen_emb_type = kwargs['seqlen_emb']
        if kwargs['seqlen_emb'] == 'scratch':
            self.seqlen_emb = nn.Embedding(n_seqlens, gpt_nembd)
        elif 'sine' in kwargs['seqlen_emb']:
            pe = torch.zeros(n_seqlens, gpt_nembd)
            position = torch.arange(0, n_seqlens, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, gpt_nembd, 2).float() * (-math.log(10000.0) / gpt_nembd))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.seqlen_emb = nn.Parameter(pe)
            if 'frozen' in kwargs['seqlen_emb']:
                self.seqlen_emb.requires_grad = False


    def actions_to_embeddings(self, actions, factor, use_clip=False):
        if use_clip or isinstance(actions, dict):
            raise ValueError('Use integer action IDs in this release.')
        return self.action_emb(actions.to(self.action_emb.weight.device)).unsqueeze(1)

    def seqlens_to_embeddings(self, seqlens):
        batch_size = seqlens.shape[0]
        list_emb = []
        for i in range(batch_size):
            idx = seqlens[i].long()
            if 'sine' in self.seqlen_emb_type:
                emb = self.seqlen_emb[idx]
            elif 'scratch' in self.seqlen_emb_type:
                emb = self.seqlen_emb(idx)
            else:
                raise NameError
            list_emb.append(emb)
        emb = torch.stack(list_emb).unsqueeze(1)
        return emb


    @torch.no_grad()
    def infill_motion(self, x, valid, mask_positions, actions=None, seqlens=None,
                     method='one_shot', temperature=1.0, **kwargs):
        """
        Infill motion at masked positions using MLM.

        Args:
            x: Input motion sequences [batch_size, seq_len, pose_dim]
            valid: Valid token mask [batch_size, seq_len]
            mask_positions: Positions to infill [batch_size, compressed_seq_len]
            actions: Action conditioning
            seqlens: Sequence length conditioning
            method: 'one_shot' or 'iterative'
            temperature: Sampling temperature

        Returns:
            result: Infilled motion sequences
            valid: Valid token mask
        """
        assert self.mlm_mode, "Infill only works in MLM mode"

        # Encode input motion to VQ indices
        with torch.no_grad():
            self.vqvae.eval()
            _, zidx, zvalid = self.vqvae.forward_latents(x, valid, return_indices=True, return_mask=True)

        # Create masked input by replacing masked positions with mask tokens
        masked_zidx = zidx.clone()
        mask_expanded = mask_positions.unsqueeze(-1).expand(-1, -1, zidx.shape[-1])

        # Use the generic mask_token_id - the model will map it to actual mask IDs
        masked_zidx = torch.where(mask_expanded, self.mask_token_id, masked_zidx)

        # Get embeddings
        actions_emb = self.actions_to_embeddings(actions, self.factor) if actions is not None else None
        seqlens_emb = self.seqlens_to_embeddings(seqlens) if seqlens is not None else None

        # Perform infilling
        if method == 'one_shot':
            completed_zidx = self.gpt.one_shot_infill(
                masked_zidx, mask_positions, actions_emb, seqlens_emb, temperature
            )
        elif method == 'iterative':
            completed_zidx = self.gpt.iterative_infill(
                masked_zidx, mask_positions, actions_emb, seqlens_emb, temperature, **kwargs
            )
        else:
            raise ValueError(f"Unknown infill method: {method}")

        # Decode back to motion
        result = self.vqvae.forward_from_indices(completed_zidx, valid=zvalid)

        return result, zvalid
