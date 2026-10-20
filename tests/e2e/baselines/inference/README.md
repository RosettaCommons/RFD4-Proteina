The CUDA 13 reference covers `rfd4-proteina-flow-release-2026-10-19.ckpt` paired
with `rfd4-proteina-ae-release-2026-10-19.ckpt`. CUDA 13 is the CI runtime.
The reference uses H200, Torch 2.14.1, AtomWorks 3.0.3, CUDA 13.2, and cuDNN 9.24.
The all-atom RMSD tolerance is 0.20 Å.

These references run eager (`compile_lengths=null`), which takes the torch triangle
path, so they do not exercise the cuEquivariance default that this branch sets. That
path runs under training and `torch.compile` and is covered by the GPU training
regression and `tests/nn/test_triangle_cueq.py`.

To regenerate the current runtime's reference, run
`python src/script_utils/generate_e2e_baselines.py --tests inference` in its
configured GPU environment. The generator checks two seeded runs for
reproducibility before writing the reference.
