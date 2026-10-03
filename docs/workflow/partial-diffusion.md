# Partial diffusion and inverse folding

Both of these start from a structure you already have instead of from noise. Inverse folding is the
special case where you keep the backbone and redesign the sequence.

## Partial diffusion

Partial diffusion starts denoising from a noisy copy of an existing structure, so the model refines
that structure rather than building one from scratch. Pass the structure with `inputs=` and set the
noise level with Hydra overrides on the command line, under `generation.args`. You get two
interfaces and you pick one.

- **Simple.** `generation.args.partial_trajectory_fraction_steps=<f>` runs that fraction of
  `nsteps`, so `0.3` runs 30% of them. The per modality `partial_t` values come from the schedule.
- **Advanced.** `generation.args.partial_t_backbone=<t>` and `generation.args.partial_t_latent=<t>`
  take floats in `(0, 1)`. Set `generation.args.couple_partial_t=true` to derive one from the other.

`partial_t` is a corruption level, **not** Angstroms. Near `1` the model runs very little of the
trajectory and your output stays close to the input. Near `0` it runs most of the trajectory and you
get something closer to full generation. `nsteps` does not depend on the level.

```bash
rfd4-generate \
    inputs=my_structure.cif \
    generation.args.partial_t_backbone=0.8 generation.args.partial_t_latent=0.8 \
    num_replicates=8 out_dir=./runs/partial
```

## Inverse folding

Inverse folding is partial diffusion with the **backbone** held fixed. The model resamples only the
sequence and side chains, the `local_latents`, and you get back new sequences that fold to the same
shape. Pass the backbone with `inputs=` and turn it on either way.

- **Simple.** `generation=inverse_folding` does it in one flag.
- **Advanced.** Tune sequence diversity with the sidechain noise scale
  `generation.model.local_latents.simulation_step_params.sc_scale_noise`, which defaults to `0.1`.
  Raise it for more varied sequences.

```bash
rfd4-generate \
    inputs=my_backbone.cif \
    generation=inverse_folding \
    generation.model.local_latents.simulation_step_params.sc_scale_noise=1.0 \
    num_replicates=8 out_dir=./runs/inverse_folding
```
