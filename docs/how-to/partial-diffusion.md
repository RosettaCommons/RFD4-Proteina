# Partial diffusion and inverse folding

How to start a run from a structure you already have instead of from noise. See [Partial diffusion
and inverse folding](../concepts/partial-diffusion.md) first if you haven't: it explains what these
do and what `partial_t` means, which you'll want before picking a value below.

## Partial diffusion

Pass your structure with `inputs=` like any other run, then tell the model how much of it to keep
versus redesign with a Hydra override under `generation.args`. There are two ways to set that,
and you only need one.

- **Simple — pick this unless you have a reason not to.**
  `generation.args.partial_trajectory_fraction_steps=<f>` runs that fraction of the total sampling
  steps (`nsteps`). The count is rounded to the nearest integer (at least one step is
  required). `0.3` runs the last 30% of the trajectory, so the model does less work and stays closer
  to your input; `0.8` runs most of it, so the output drifts further. Corruption and denoising use
  the same tail of the configured full schedule, including its late refinement. For example,
  `nsteps=60` with `f=0.3333` runs the final 20 intervals, starting at index 40 of that schedule. A
  restart index is not a diffusion time: changing the schedule or its step count can change the
  corruption level at a given fraction.
- **Advanced — reach for this when the backbone and sequence need to change at different rates.**
  Set `generation.args.partial_t_backbone=<t>` and `generation.args.partial_t_latent=<t>`
  separately, each a float in `(0, 1)` (not a fraction of steps — see the concept page). Only
  setting one and adding `generation.args.couple_partial_t=true` derives the other from the
  default schedule instead of requiring both.

For example, to keep both the fold and the sequence close to the input, regenerating only the last
20% of the trajectory:

```bash
rfd4-generate \
    inputs=my_structure.cif \
    generation.args.partial_t_backbone=0.8 generation.args.partial_t_latent=0.8 \
    num_replicates=8 out_dir=./runs/partial
```

## Inverse folding

Use this when you want to keep a backbone exactly as given and only get new sequences for it — no
backbone noise at all, just resampled sequence and side chains.

- **Simple — start here.** Add the flag `generation=inverse_folding` and pass your backbone with
  `inputs=`. That's the whole setup.
- **Advanced.** Tune sequence temperature with
  `generation.model.local_latents.simulation_step_params.tsr_k` (default `4`).
  Lower `k` raises temperature; `tsr_sigma` (default `12`) provides secondary control over
  when sharpening takes effect.

```bash
rfd4-generate \
    inputs=my_backbone.cif \
    generation=inverse_folding recycle_strategy=medium/flat \
    generation.model.local_latents.simulation_step_params.tsr_k=2 \
    num_replicates=8 out_dir=./runs/inverse_folding
```
