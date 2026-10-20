# Partial diffusion and inverse folding

Standard generation starts from pure noise, so nothing about the output is tied to anything you
provide. Partial diffusion and inverse folding start from a structure you already have instead, and
ask the model to refine or redesign it rather than invent one from nothing. Reach for one of these
when you already have a design, a real structure, or a backbone you like, and want the model to
build on it rather than start over. Inverse folding is the special case where you keep the backbone
exactly as given and only redesign the sequence.

## Partial diffusion

Generation works by denoising: the model starts from noise and removes it step by step until a
structure falls out. Partial diffusion skips the early part of that process. Instead of starting
from pure noise, it starts from your input structure with a measured amount of noise added back in,
so the model only has to clean up what remains. The less noise you add, the closer the output
stays to your input; the more you add, the more room the model has to change it.

That noise amount is set by `partial_t`, and it is best thought of as "how much do I trust the
model to change what I gave it," **not** a physical distance like Angstroms:

- Near `1`: almost no noise was added. The model runs very little of the trajectory and the output
  stays close to the input.
- Near `0`: heavy noise was added. The model runs nearly the whole trajectory, and the output looks
  more like it was generated from scratch.

`nsteps`, the number of sampling steps, does not change with `partial_t`. You are choosing *where
on the trajectory* to start, not how many steps the model takes to get to the end.

Use partial diffusion to generate variations on a design you like, refine one region without
redoing the whole structure, or nudge an existing fold in a new direction while keeping it
recognizable.

See [Partial diffusion](../how-to/partial-diffusion.md#partial-diffusion) for the commands and flags.

## Inverse folding

Inverse folding is partial diffusion pushed to one extreme: instead of partially noising the
backbone, it holds the backbone **completely fixed** and resamples everything else, the sequence
and the side chains (`local_latents`), from scratch. You give the model a shape you want to keep,
and it finds sequences that are predicted to fold into exactly that shape.

Reach for this when you already have a backbone, from a solved structure, a previous design, or a
hand-built scaffold, and you want new sequences for it rather than a new fold.

See [Inverse folding](../how-to/partial-diffusion.md#inverse-folding) for the commands and flags.
