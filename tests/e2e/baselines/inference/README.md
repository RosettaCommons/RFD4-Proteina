Inference references are specific to the CUDA runtime on H200. Both use Torch
2.14.1 and AtomWorks 3.0.0: `cuda12` uses CUDA 12.6/cuDNN 9.10.2, and `cuda13`
uses CUDA 13.2/cuDNN 9.24. The all-atom RMSD tolerance remains 0.20 Å.

To regenerate the current runtime's reference, run
`python src/script_utils/generate_e2e_baselines.py --tests inference` in its
configured GPU environment. The generator checks two seeded runs for
reproducibility before writing the reference.
