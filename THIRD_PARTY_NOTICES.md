# Third-party source notices

ExpertEdit's public model package is adapted from the PoseGPT source in this
research repository. The upstream source credits NAVER Corp. and uses
CC BY-NC-SA 4.0. The accompanying `LICENSE` and `NOTICE` are preserved verbatim
from `models/PoseGPT/`.

The transformer blocks credit Andrej Karpathy's minGPT. The vector quantizer
credits CompVis/taming-transformers. Their upstream component notices and license
texts are retained in `NOTICE`.

The initial release adapts package imports, removes training entry points,
loss computation, CUDA-only codebook usage tracking, and research-only branches,
and adds a standalone inference interface. Parameter names and the motion/token
layout are retained for compatible research checkpoints. Mask-token mapping no
longer mutates caller-owned inputs, and iterative infilling completes any tokens
remaining after its confidence schedule.

SMPL, SMPL-H, MANO, third-party pretrained weights, source datasets, and vendored
baseline repositories are not included. The Python inference package operates on
prepared motion tensors and does not load body-model files.
