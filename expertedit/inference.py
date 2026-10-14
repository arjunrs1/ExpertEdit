"""Standalone inference on prepared motion tensors; no research pipeline imports."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import roma
import torch

from .models.posegpt import poseGPTMLM
from .models.tokenizer import CausalVQVAE


TOKENIZER_DEFAULTS = {
    "in_dim": 147,
    "n_e": 512,
    "e_dim": 256,
    "n_codebook": 2,
    "hid_dim": 384,
    "n_layers": [0, 6],
    "heads": 4,
    "dropout": 0.0,
    "block_size": 2048,
    "pos_type": "sine_frozen",
    "pos_all": True,
    "sample_method": "conv",
    "seq_len": 120,
    "beta": 0.25,
}
EDITOR_DEFAULTS = {
    "gpt_nembd": 256,
    "gpt_blocksize": 2048,
    "gpt_nhead": 8,
    "gpt_nlayer": 12,
    "gpt_embd_pdrop": 0.0,
    "gpt_resid_pdrop": 0.0,
    "gpt_attn_pdrop": 0.0,
    "n_actions": 85,
    "class_conditional": True,
    "embed_every_step": False,
    "seqlen_conditional": False,
    "concat_emb": False,
    "head_type": "fc_wo_bias",
    "pos_emb": "scratch",
    "action_emb": "scratch",
    "seqlen_emb": "scratch",
    "mask_token_id": -1,
}


def _settings(config):
    if not isinstance(config, dict) or set(config) - {"tokenizer", "editor"}:
        raise ValueError("Config must contain only 'tokenizer' and 'editor' objects.")
    sections = []
    for name, defaults in (("tokenizer", TOKENIZER_DEFAULTS), ("editor", EDITOR_DEFAULTS)):
        supplied = config.get(name, {})
        if not isinstance(supplied, dict):
            raise ValueError(f"Config section '{name}' must be an object.")
        unknown = set(supplied) - set(defaults)
        if unknown:
            raise ValueError(f"Unknown {name} settings: {', '.join(sorted(unknown))}")
        sections.append({**defaults, **supplied})
    tokenizer, editor = sections
    for key in ("in_dim", "n_e", "e_dim", "n_codebook", "hid_dim", "heads", "block_size", "seq_len"):
        if type(tokenizer[key]) is not int or tokenizer[key] <= 0:
            raise ValueError(f"tokenizer.{key} must be a positive integer.")
    for key in ("gpt_nembd", "gpt_blocksize", "gpt_nhead", "gpt_nlayer", "n_actions"):
        if type(editor[key]) is not int or editor[key] <= 0:
            raise ValueError(f"editor.{key} must be a positive integer.")
    layers = tokenizer["n_layers"]
    if not isinstance(layers, list) or len(layers) != 2 or layers[0] != 0 or type(layers[1]) is not int or layers[1] <= 0:
        raise ValueError("This release supports n_layers=[0, N] with N > 0 (2x temporal compression).")
    if tokenizer["in_dim"] < 9 or (tokenizer["in_dim"] - 3) % 6:
        raise ValueError("in_dim must be 6 * number_of_joints + 3.")
    if tokenizer["n_e"] % tokenizer["n_codebook"] or tokenizer["e_dim"] % tokenizer["n_codebook"]:
        raise ValueError("n_codebook must divide n_e and e_dim.")
    if tokenizer["hid_dim"] % tokenizer["heads"] or editor["gpt_nembd"] % editor["gpt_nhead"] or editor["gpt_nembd"] % tokenizer["n_codebook"]:
        raise ValueError("Hidden dimensions must be divisible by their head/codebook counts.")
    if tokenizer["sample_method"] != "conv" or editor["action_emb"] != "scratch":
        raise ValueError("This release supports convolutional resampling and learned action embeddings.")
    if editor["seqlen_conditional"] and not editor["embed_every_step"]:
        raise ValueError("Sequence length conditioning requires embed_every_step=true.")
    if editor["concat_emb"] and not editor["embed_every_step"]:
        raise ValueError("Concatenated embeddings require embed_every_step=true.")
    if editor["mask_token_id"] != -1:
        raise ValueError("This release uses mask_token_id=-1.")
    return tokenizer, editor


def build_model(config):
    """Construct the architecture. Weights remain random until explicitly loaded."""
    tokenizer, editor = _settings(config)
    vqvae = CausalVQVAE(**tokenizer)
    return poseGPTMLM(
        vqvae=vqvae,
        n_e=tokenizer["n_e"],
        n_codebook=tokenizer["n_codebook"],
        causal_gpt=False,
        mlm_mode=True,
        gen_eos=False,
        eos_force=True,
        autoreg_pq=False,
        factor=2,
        use_clip_conditioning=False,
        **editor,
    ).eval()


def axis_angle_to_motion(poses, translation):
    """Convert [B,T,J,3] axis-angle rotations and [B,T,3] translation to model input.

    Inputs must already use the coordinate frame, normalization, units and frame
    rate expected by the supplied tokenizer. This function only changes rotation
    representation, using PoseGPT's row-major flattening of the first two columns.
    """
    if poses.ndim != 4 or poses.shape[-1] != 3 or translation.shape != poses.shape[:2] + (3,):
        raise ValueError("Expected poses [B,T,J,3] and translation [B,T,3].")
    rotations = roma.rotvec_to_rotmat(poses)
    return torch.cat((rotations[..., :2].flatten(2), translation), dim=-1)


def _load_state(path, allow_legacy_checkpoint):
    checkpoint = torch.load(path, map_location="cpu", weights_only=not allow_legacy_checkpoint)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    if not isinstance(checkpoint, dict) or not checkpoint or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in checkpoint.items()
    ):
        raise ValueError("Checkpoint must be a state_dict or contain 'model_state_dict'.")
    return {key.removeprefix("module."): value for key, value in checkpoint.items()}


class ExpertEdit:
    def __init__(self, model):
        self.model = model.eval()

    @classmethod
    def from_checkpoints(cls, config, tokenizer_checkpoint, editor_checkpoint, device="cpu", *, allow_legacy_checkpoint=False):
        """Load caller-supplied weights, rejecting missing or incompatible tensors.

        The editor checkpoint includes the frozen tokenizer under 'vqvae.*', as
        in the research model. It must agree with the separate tokenizer weights.
        """
        model = build_model(config)
        tokenizer_state = _load_state(tokenizer_checkpoint, allow_legacy_checkpoint)
        editor_state = _load_state(editor_checkpoint, allow_legacy_checkpoint)
        model.vqvae.load_state_dict(tokenizer_state, strict=True)
        for key, value in tokenizer_state.items():
            embedded = editor_state.get("vqvae." + key)
            if embedded is None or not torch.equal(value, embedded):
                raise ValueError("Editor and tokenizer checkpoints do not contain the same tokenizer weights.")
        model.load_state_dict(editor_state, strict=True)
        return cls(model.to(device))

    @torch.inference_mode()
    def edit(self, motion, edit_mask, actions=None, *, method="one_shot", temperature=0.0):
        """Infill tokens selected by a boolean [B,T/2] edit mask.

        All batch entries must have the same even frame count, without padding.
        Returns decoded rotation matrices [B,T,J,3,3] and translation [B,T,3].
        The decoder can change frames outside the edited token span.
        """
        if method not in {"one_shot", "iterative"}:
            raise ValueError("method must be 'one_shot' or 'iterative'.")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and nonnegative.")
        if method == "iterative" and temperature != 0:
            raise ValueError("Iterative inference uses greedy prediction; set temperature=0.")
        device = next(self.model.parameters()).device
        motion = torch.as_tensor(motion, dtype=torch.float32, device=device)
        if motion.ndim != 3 or motion.shape[0] < 1 or motion.shape[-1] != self.model.vqvae.in_dim:
            raise ValueError(f"Expected motion [B,T,{self.model.vqvae.in_dim}].")
        batch, frames, _ = motion.shape
        if frames < 2 or frames % 2:
            raise ValueError("Motion must have an even frame count of at least 2.")
        limit = min(self.model.vqvae.encoder_stacks[0].block_size, 2 * self.model.gpt.block_size)
        if frames > limit or not torch.isfinite(motion).all():
            raise ValueError(f"Motion must be finite and contain at most {limit} frames.")
        edit_mask = torch.as_tensor(edit_mask, device=device)
        if edit_mask.dtype != torch.bool or edit_mask.shape != (batch, frames // 2):
            raise ValueError("edit_mask must be boolean with shape [B,T/2].")
        if self.model.class_conditional:
            if actions is None:
                raise ValueError("An action ID per motion is required for this checkpoint.")
            actions = torch.as_tensor(actions, device=device)
            if actions.shape != (batch,) or actions.dtype not in (torch.int32, torch.int64):
                raise ValueError("actions must be integer IDs with shape [B].")
            if (actions < 0).any() or (actions >= self.model.n_actions).any():
                raise ValueError("Action IDs are outside the checkpoint's vocabulary.")
            actions = actions.long()
        else:
            if actions is not None:
                raise ValueError("This checkpoint does not use action conditioning.")
        if self.model.seqlen_conditional and frames // 2 >= 1024:
            raise ValueError("Sequence length conditioning supports fewer than 1024 tokens.")
        valid = torch.ones((batch, frames), dtype=torch.long, device=device)
        lengths = torch.full((batch,), frames // 2, dtype=torch.long, device=device)
        self.model.eval()
        ((rotations, translation), _), _ = self.model.infill_motion(
            motion, valid, edit_mask, actions=actions if self.model.class_conditional else None,
            seqlens=lengths if self.model.seqlen_conditional else None,
            method=method, temperature=temperature,
        )
        if not torch.isfinite(rotations).all() or not torch.isfinite(translation).all():
            raise ValueError("Inference produced non-finite motion; check inputs and checkpoints.")
        return {"rotations": rotations, "translation": translation}


def main():
    parser = argparse.ArgumentParser(description="ExpertEdit inference on prepared motion (weights required).")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tokenizer-checkpoint", type=Path, required=True)
    parser.add_argument("--editor-checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="NPZ containing motion, edit_mask and optional actions.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--method", choices=("one_shot", "iterative"), default="one_shot")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-legacy-checkpoint", action="store_true", help="Allow pickle-based training containers; use only with trusted checkpoints.")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve() or args.output.exists():
        parser.error("Choose an output path that does not already exist.")
    if args.output.suffix != ".npz":
        parser.error("Output must have the .npz extension.")
    torch.manual_seed(args.seed)
    model = ExpertEdit.from_checkpoints(
        json.loads(args.config.read_text()), args.tokenizer_checkpoint, args.editor_checkpoint,
        args.device, allow_legacy_checkpoint=args.allow_legacy_checkpoint,
    )
    with np.load(args.input, allow_pickle=False) as arrays:
        result = model.edit(
            arrays["motion"], arrays["edit_mask"], arrays["actions"] if "actions" in arrays else None,
            method=args.method, temperature=args.temperature,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **{key: value.cpu().numpy() for key, value in result.items()})


if __name__ == "__main__":
    main()
