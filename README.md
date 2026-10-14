# ExpertEdit: Learning Skill-Aware Motion Editing from Expert Videos

ExpertEdit studies how to edit a novice's 3D motion using motion patterns learned
from expert demonstrations. A vector-quantized motion tokenizer represents a
sequence as discrete tokens, and a bidirectional masked transformer predicts
replacement tokens for selected parts of the motion.

## Release status

This initial source release includes the motion tokenizer, masked transformer,
and an inference API and command-line tool for **prepared motion with an explicit
edit mask**. Training code, evaluation code, checkpoints, datasets, and body-model
assets are withheld from this release.

**Pretrained checkpoints are not currently available.** Inference requires
compatible tokenizer and editor weights supplied separately. Constructing the
architecture without loading weights produces a randomly initialized model,
not useful motion edits. Paper and project-page links will be added when available.

## Installation

Requires Python 3.10 or newer. From the initial release directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
expertedit-infer --help
```

CPU inference is supported. For GPU inference, use a PyTorch installation matching
your CUDA environment and pass `--device cuda`. The inference package does not
require SMPL model files, a renderer, or the training environment.

## Model configuration and checkpoints

Copy `configs/model.example.json` and set its architecture values to match your
checkpoints. The example is an architecture template, not a verified configuration
for a released checkpoint. Joint count, vocabulary, codebooks, dimensions,
positional embeddings, and conditioning settings must match the weights.

The tokenizer checkpoint contains a PyTorch state dictionary for `CausalVQVAE`.
The editor checkpoint contains the full `poseGPTMLM` state dictionary, including
its frozen tokenizer under `vqvae.*`. Both a plain state dictionary and a dictionary
containing `model_state_dict` are supported; distributed-training `module.` prefixes
are removed. Loading is strict, and the two checkpoints must contain identical
tokenizer weights.

Only convolutional 2x temporal compression (`n_layers=[0, N]`), bidirectional MLM,
and learned action embeddings are supported by this entry point. When the checkpoint
uses action conditioning, supply the action IDs from its original vocabulary.

The default loader uses PyTorch's weights-only mode. Legacy checkpoint containers
with pickled metadata require `--allow-legacy-checkpoint`; enable this option only
for checkpoints you trust.

## Input and output

Prepare an `.npz` file with these arrays:

| Array | Shape | Type | Meaning |
| --- | --- | --- | --- |
| `motion` | `[B, T, 6J + 3]` | float32 | Joint rotations in 6D representation followed by translation. |
| `edit_mask` | `[B, T / 2]` | bool | `true` selects a compressed motion token to replace. |
| `actions` | `[B]` | int64 | Action ID per sequence; required for action-conditioned weights. |

`B` is batch size, `T` is an even frame count, and `J` includes the root joint.
Batch entries must have the same length without padding. For 24-joint SMPL motion,
the input dimension is **147** (`24 * 6 + 3`). The 6D layout uses the first two
columns of each rotation matrix, flattened in row-major order, matching PoseGPT.

Motion must already have the coordinate frame, normalization, translation units,
and frame rate expected by the checkpoint. Video pose extraction, automatic
kinematic mask selection, alignment, and mesh rendering are outside this initial
release. A token mask selects time spans across all codebooks, rather than individual
joints. The decoder reconstructs the entire sequence, so unmasked frames may also
change.

The output `.npz` contains `rotations` (`[B, T, J, 3, 3]`) and `translation`
(`[B, T, 3]`).

## Inference

Once compatible checkpoints and prepared input are available:

```bash
expertedit-infer \
  --config configs/model.example.json \
  --tokenizer-checkpoint /path/to/tokenizer.pt \
  --editor-checkpoint /path/to/editor.pt \
  --input /path/to/motion.npz \
  --output /path/to/edited_motion.npz \
  --device cpu \
  --method one_shot \
  --temperature 0
```

`one_shot` fills the selected tokens in a single pass. Temperature `0` uses greedy
prediction; a positive temperature samples predictions. `iterative` progressively
fills tokens by confidence and completes any remaining tokens at the end. Use
temperature `0` for iterative inference. `--seed` controls sampling (default `0`).
Choose a new output path; the CLI refuses to overwrite an existing file.

The Python API accepts tensors or NumPy arrays:

```python
import json
import numpy as np
from expertedit import ExpertEdit

with open("configs/model.example.json") as file:
    config = json.load(file)

editor = ExpertEdit.from_checkpoints(
    config,
    tokenizer_checkpoint="/path/to/tokenizer.pt",
    editor_checkpoint="/path/to/editor.pt",
    device="cpu",
)
with np.load("/path/to/motion.npz", allow_pickle=False) as data:
    result = editor.edit(data["motion"], data["edit_mask"], data["actions"])
```

For prepared axis-angle motion, `expertedit.axis_angle_to_motion(poses, translation)`
converts rotations `[B, T, J, 3]` and translation `[B, T, 3]` to the required 6D
layout. It does not normalize or align the motion.

## Source layout

```text
expertedit/
  inference.py       # Checkpoint loading, motion conversion, inference API and CLI
  models/
    tokenizer.py     # Motion tokenizer encoder and decoder
    posegpt.py       # Masked transformer and token infilling
    quantize.py      # Inference-only product quantization
    blocks/          # Attention, transformer and temporal resampling components
configs/
  model.example.json
```

## Acknowledgments and license

The model source is adapted from PoseGPT and retains its existing
**CC BY-NC-SA 4.0** license and upstream notices. See [LICENSE](LICENSE),
[NOTICE](NOTICE), and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution
and component terms. This source release contains no dataset or body-model assets.

## Citation

Paper links and the final citation will be added here when available.
