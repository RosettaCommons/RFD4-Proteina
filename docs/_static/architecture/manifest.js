window.RFD4_ARCHITECTURE = {
 "root": "overview",
 "diagrams": {
  "overview": {
   "title": "RFD4-Proteina at a glance",
   "short": "Overview",
   "blurb": "The whole system: inputs, the flow-matching sampler, the recycled denoiser and the decoder.",
   "summary": "<p>RFD4-Proteina generates a design by flow matching in a product space: the CA position of every token (atom37 slot 1: C1′ for a nucleotide), in nm, and 32 latent channels per residue or nucleotide learned by a partial autoencoder. A generation starts from Gaussian noise, runs 60 steps of a stochastic sampler whose velocity comes from the <em>denoiser</em>, a 14-layer pair-biased transformer, and hands the final CA positions and latents to the autoencoder's decoder, which writes every atom of the residues and nucleotides and, by default, takes the sequence the denoiser predicted wherever the input leaves it open; atomized ligand atoms keep the positions the flow produced.</p><p>Click a box to open its diagram; the annotated CIF and design boxes open the workflow pages on inputs and outputs. Every diagram below this map is a neural circuit diagram generated from an algebraic expression of the code, drawn at the dimensions of the reference model. In those diagrams, hover a box to see its formula, description and code, click a box whose details offer <em>Open diagram</em> to go one level deeper (the pointer changes over it), and use the breadcrumbs or the tree to go back up.</p>",
   "highlights": [
    {
     "kind": "loop",
     "label": "Recycling",
     "html": "Every network evaluation runs the trunk <em>R</em> times, feeding the previous single and pair states back through zero-initialised projections. By default (<code>thinking_medium</code>) <em>R</em> starts at 6 and steps down to 1 across the 60 steps: 180 trunk passes per design."
    },
    {
     "kind": "loop",
     "label": "Sampling",
     "html": "60 integration steps from noise at <em>t</em> = 0 to data at <em>t</em> = 1, one time per modality."
    },
    {
     "kind": "open",
     "label": "Drill down",
     "html": "Overview → Generation → Denoiser → Trunk → Layer → Attention is the longest path: six levels."
    }
   ],
   "facts": [
    [
     "Denoiser",
     "178,448,304 parameters"
    ],
    [
     "Autoencoder",
     "257,174,161 parameters (encoder 129,013,832, decoder 128,160,329), frozen during flow training"
    ],
    [
     "Token / pair width",
     "768 / 384"
    ],
    [
     "Layers / heads",
     "14 / 12 of width 64"
    ],
    [
     "Latents",
     "32 channels per residue or nucleotide"
    ],
    [
     "Release bundle",
     "435,622,465 stored parameters: <code>nvidia/RFD4-Proteina</code> at revision <code>a06e250</code>, with the fine-tuning LoRA merged into the flow weights"
    ],
    [
     "Recipes",
     "flow pre-training <code>releases/flow/2026_10_19_hero_pretrain</code>, fine-tuning <code>releases/flow/2026_10_19_hero_lora_c384</code>; autoencoder <code>releases/ae/2026_10_15_unified_biomolecule</code>, checkpoint dated 2026-10-19"
    ]
   ],
   "parent": null,
   "children": [
    "generation",
    "training",
    "autoencoder"
   ],
   "kind": "map"
  },
  "generation": {
   "title": "Generation",
   "short": "Generation",
   "blurb": "Noise to design: 60 sampling steps around the denoiser, then the decoder.",
   "summary": "<p>The design features of the annotated CIF are computed and put on the tape; the denoiser reshuffles their coordinate-group labels at every network evaluation. The CA positions and latents start as Gaussian noise, and the loop under the <em>T</em> bracket runs 60 times: the schedule gives both modalities their time and step size, the <strong>Denoiser</strong> predicts both velocities (with the recycle depth of that step), and one <strong>Step</strong> per modality integrates them. The <strong>Decoder</strong> then fills every position the input leaves without a sequence from the denoiser's last sequence prediction and turns the final CA positions and latents into residues, atom37 coordinates and atom masks.</p>",
   "highlights": [
    {
     "kind": "loop",
     "label": "Sampling loop",
     "html": "The outer loop, <em>T</em> = 60. Each iteration is one network evaluation."
    },
    {
     "kind": "loop",
     "label": "Recycling",
     "html": "Inside the <strong>Denoiser</strong> box: its recycle depth <em>R</em>(<em>i</em>) changes with the step <em>i</em>."
    },
    {
     "kind": "open",
     "label": "Try",
     "html": "Click <strong>Denoiser</strong>, <strong>Step<sub>CA</sub></strong> or <strong>Decoder</strong>."
    }
   ],
   "facts": [
    [
     "Steps",
     "60 (<code>generation.args.nsteps</code>)"
    ],
    [
     "Sampler",
     "<code>vf_tsr</code>: temperature-scaled score (temporal score rescaling) with a Langevin term between t = 0.02 and 0.95"
    ],
    [
     "Network evaluations",
     "60, with 180 trunk passes under the default <code>thinking_medium</code> schedule (<code>medium/front_loaded</code>: R = 6, 5, 4, 3, 2, 1 for 5, 10, 5, 10, 20 and 10 steps)"
    ],
    [
     "Outputs",
     "sequence (the flow model's prediction where the input leaves it open), atom37 coordinates and atom mask, insertion mapping"
    ],
    [
     "Backend",
     "native by default (<code>inference_backend: native</code>); BioIR is optional"
    ]
   ],
   "parent": "overview",
   "children": [
    "denoiser",
    "integrator"
   ],
   "kind": "ncd",
   "page": "ncd/generation.html",
   "code": [
    {
     "label": "vanilla.py L195-L324",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/search/vanilla.py#L195-L324"
    },
    {
     "label": "proteina.py L1948-L2106",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/proteina.py#L1948-L2106"
    },
    {
     "label": "generate.py L610-L744",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/generate.py#L610-L744"
    }
   ]
  },
  "denoiser": {
   "title": "Denoiser and its recycling loop",
   "short": "Denoiser",
   "blurb": "LocalLatentsTransformer: embeddings, the recycled trunk and the output heads.",
   "summary": "<p>One evaluation of the flow network. The <em>feature factories</em> build the conditioning <em>c</em> and the initial single and pair representations <em>s</em><sub>0</sub>, <em>z</em><sub>0</sub> once, and drop them on the tape. The <strong>Recycling</strong> block, under the <em>R</em> bracket, then runs <em>R</em> times: each cycle adds an RMS-normalised projection of the previous cycle's states to <em>s</em><sub>0</sub> and <em>z</em><sub>0</sub>, reads <em>c</em> from the tape (a later cycle can rebuild it with the recycle flag set) and runs the whole <strong>Trunk</strong>, which passes <em>c</em> through its layers. The first cycle starts from zeros. The heads read the final single representation.</p>",
   "highlights": [
    {
     "kind": "loop",
     "label": "Recycling",
     "html": "The projections <strong>W</strong><sup>s</sup><sub>rec</sub> and <strong>W</strong><sup>z</sup><sub>rec</sub> start at zero, so initially every cycle receives the same <em>s</em><sub>0</sub> and <em>z</em><sub>0</sub>. Later cycles can still produce different results because the recycle flag can change their conditioning. Only the last cycle keeps gradients in training. State does not carry across sampling steps."
    },
    {
     "kind": "note",
     "label": "Tape",
     "html": "Coloured labels are tape slots: values written once (<em>s</em><sub>0</sub>, <em>z</em><sub>0</sub>, <em>c</em>) and read in many places, and the design features <em>f</em><sub>s</sub>, <em>f</em><sub>z</sub> and the CATH labels κ read from the generation."
    },
    {
     "kind": "open",
     "label": "Try",
     "html": "Click <strong>Trunk</strong> to see the 14 layers, or <strong>Cond</strong>, <strong>Embed<sub>s</sub></strong>, <strong>Embed<sub>z</sub></strong>, <strong>Heads</strong>."
    }
   ],
   "facts": [
    [
     "Parameters",
     "178,448,304"
    ],
    [
     "Recycle depth R",
     "6 down to 1 per step at inference (<code>thinking_medium</code>); Poisson(3) clipped to [1, 9] in training"
    ],
    [
     "Inputs",
     "noisy CA (x, 3), noisy latents (x, 32), two times"
    ],
    [
     "Outputs",
     "CA velocity, latent velocity, insertion logits (x, x), sequence logits (x, 147)"
    ]
   ],
   "parent": "generation",
   "children": [
    "embeddings",
    "conditioning",
    "trunk",
    "heads"
   ],
   "kind": "ncd",
   "page": "ncd/denoiser.html",
   "code": [
    {
     "label": "local_latents_transformer.py L37-L208",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/nn/local_latents_transformer.py#L37-L208"
    },
    {
     "label": "local_latents_transformer.py L281-L316",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/nn/local_latents_transformer.py#L281-L316"
    },
    {
     "label": "local_latents_score_nn_160M_atom37.yaml L1-L132",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/configs/nn/local_latents_score_nn_160M_atom37.yaml#L1-L132"
    },
    {
     "label": "2026_10_19_hero_pretrain.yaml L390-L423",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/configs/releases/flow/2026_10_19_hero_pretrain.yaml#L390-L423"
    }
   ]
  },
  "embeddings": {
   "title": "Single and pair embeddings",
   "short": "Embeddings",
   "blurb": "How the noisy state and the design features become s₀ and z₀.",
   "summary": "<p>The single embedding is one linear map of 897 concatenated token channels, drawn as three maps by where the inputs come from: the noisy CA coordinates (3), the noisy latents (32), and the 862 channels of design features. The design features are not computed from the noisy state, with two qualifications: their 37 coordinate-group channels are relabelled at random on every evaluation, and their 18 chirality channels are rotated toward the noisy CA positions (<code>noise_align_chirality</code>). The pair embedding bins the pairwise distances of the noisy CA positions into 30 classes, adds 77 channels of pair design features, normalises and applies an AdaLN conditioned on the two times.</p>",
   "highlights": [
    {
     "kind": "note",
     "label": "No positions",
     "html": "There is no residue-index embedding. Chain order enters through the bond-path-length and polymer-ordering pair features."
    }
   ],
   "facts": [
    [
     "Single features",
     "897 → 768: 3 CA + 32 latent + 862 design channels"
    ],
    [
     "Pair features",
     "107 → 384: 30 distance bins + 77 design channels (the nucleic-acid pair feature is 2 wide)"
    ],
    [
     "Distance bins",
     "30, from 0.1 to 3 nm"
    ]
   ],
   "parent": "denoiser",
   "children": [],
   "kind": "ncd",
   "page": "ncd/embeddings.html",
   "code": [
    {
     "label": "local_latents_transformer.py L68-L75",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/nn/local_latents_transformer.py#L68-L75"
    },
    {
     "label": "local_latents_transformer.py L289-L291",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/nn/local_latents_transformer.py#L289-L291"
    },
    {
     "label": "feature_factory.py L1397-L1423",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/nn/feature_factory.py#L1397-L1423"
    },
    {
     "label": "2026_10_19_hero_pretrain.yaml L395-L416",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/configs/releases/flow/2026_10_19_hero_pretrain.yaml#L395-L416"
    },
    {
     "label": "pair_rep_initial.py L7-L47",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/nn/modules/pair_rep_initial.py#L7-L47"
    },
    {
     "label": "feature_factory.py L103-L142",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/nn/feature_factory.py#L103-L142"
    },
    {
     "label": "feature_factory.py L1098-L1116",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/nn/feature_factory.py#L1098-L1116"
    },
    {
     "label": "2026_10_19_hero_pretrain.yaml L417-L423",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/configs/releases/flow/2026_10_19_hero_pretrain.yaml#L417-L423"
    },
    {
     "label": "local_latents_score_nn_160M_atom37.yaml L70-L93",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/configs/nn/local_latents_score_nn_160M_atom37.yaml#L70-L93"
    }
   ]
  },
  "conditioning": {
   "title": "Conditioning",
   "short": "Conditioning",
   "blurb": "Time embeddings, the recycle flag and CATH labels become the vector c read by every AdaLN and output gate of the trunk.",
   "summary": "<p>Both flow times are embedded with 128 sinusoid pairs, a one-channel recycle flag ρ marks recycling cycles after the first, the three CATH levels of each token are looked up in three tables, and the parts are projected to 256 channels and summed. Two SwiGLU transitions without a residual finish the conditioning vector <em>c</em>, which every recycling cycle reads from the tape and the trunk hands to every AdaLN and every output gate. The feature factories build <em>c</em> with ρ = 0; when recycle conditioning applies, every cycle after the first rebuilds it with ρ = 1.</p>",
   "highlights": [
    {
     "kind": "note",
     "label": "Per token",
     "html": "The times are per design but CATH is per token, so <em>c</em> is a (<em>x</em>, 256) array."
    }
   ],
   "facts": [
    [
     "Inputs",
     "609 channels → 256: 2 × 256 time, 1 recycle flag, 3 × 32 CATH"
    ],
    [
     "Time embedding",
     "256 channels, sin and cos of 2000 t ω<sub>k</sub>"
    ],
    [
     "CATH tables",
     "6, 44 and 1473 rows of width 32"
    ],
    [
     "Recycle flag",
     "<code>recycle_remaining</code>, 1 channel under <code>recycle.grad: last</code>: it marks a later cycle and does not count the cycles left"
    ]
   ],
   "parent": "denoiser",
   "children": [],
   "kind": "ncd",
   "page": "ncd/conditioning.html",
   "code": [
    {
     "label": "local_latents_transformer.py L77-L88",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/nn/local_latents_transformer.py#L77-L88"
    },
    {
     "label": "local_latents_transformer.py L285-L287",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/nn/local_latents_transformer.py#L285-L287"
    },
    {
     "label": "feature_factory.py L228-L266",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/nn/feature_factory.py#L228-L266"
    },
    {
     "label": "feature_factory.py L280-L312",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/nn/feature_factory.py#L280-L312"
    },
    {
     "label": "local_latents_transformer.py L386-L407",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/nn/local_latents_transformer.py#L386-L407"
    },
    {
     "label": "local_latents_score_nn_160M_atom37.yaml L59-L68",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/configs/nn/local_latents_score_nn_160M_atom37.yaml#L59-L68"
    },
    {
     "label": "2026_10_19_hero_pretrain.yaml L417-L421",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/configs/releases/flow/2026_10_19_hero_pretrain.yaml#L417-L421"
    }
   ]
  },
  "trunk": {
   "title": "The trunk",
   "short": "Trunk",
   "blurb": "14 layers with a pair update after layers 0, 2, …, 12.",
   "summary": "<p>The trunk is seven repetitions of [layer, pair update, layer], each with its own weights, which is exactly the schedule the code runs: a pair update after each even-indexed layer (0, 2, …, 12) and none after the final layer 13. The trunk receives the conditioning <em>c</em> from its recycling cycle with the single representation (768) and the pair representation (384), and passes it through every layer. A pair update reads <em>s</em> and rewrites <em>z</em>; <em>s</em> and <em>c</em> pass around it.</p>",
   "highlights": [
    {
     "kind": "loop",
     "label": "Repetition",
     "html": "The bracket with the count <em>P</em> = 7 marks the repetitions: 14 layers and 7 pair updates."
    },
    {
     "kind": "open",
     "label": "Try",
     "html": "Click <strong>Layer</strong> or <strong>PairUpdate</strong>."
    }
   ],
   "facts": [
    [
     "Per layer",
     "11,221,376 parameters"
    ],
    [
     "Per pair update",
     "2,462,336 parameters"
    ],
    [
     "Whole trunk",
     "174.3 M of the denoiser's 178.4 M"
    ]
   ],
   "parent": "denoiser",
   "children": [
    "layer",
    "pair-update"
   ],
   "kind": "ncd",
   "page": "ncd/trunk.html",
   "code": [
    {
     "label": "local_latents_transformer.py L318-L334",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/nn/local_latents_transformer.py#L318-L334"
    },
    {
     "label": "local_latents_transformer.py L107-L147",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/nn/local_latents_transformer.py#L107-L147"
    }
   ]
  },
  "layer": {
   "title": "One trunk layer",
   "short": "Layer",
   "blurb": "Pair-biased attention and a SwiGLU transition, each in AdaLN, a gate and a residual.",
   "summary": "<p>Each sublayer normalises the single representation under the conditioning (<strong>AdaLN</strong>), runs its body, multiplies the result by a gate read off the conditioning (<strong>Gate</strong>) and adds it to the residual. The attention reads the pair representation; the layer never writes it, and passes it and the conditioning through unchanged.</p>",
   "highlights": [
    {
     "kind": "note",
     "label": "Near identity",
     "html": "Every gate starts at σ(−2) ≈ 0.12 with zero weights, so a fresh layer barely changes its input."
    },
    {
     "kind": "open",
     "label": "Try",
     "html": "Click <strong>Attn</strong>, <strong>AdaLN</strong> or <strong>SwiGLU</strong>."
    }
   ],
   "facts": [
    [
     "Attention",
     "12 heads × 64, pair bias from 384 channels"
    ],
    [
     "Transition",
     "SwiGLU, hidden width 3072"
    ],
    [
     "Conditioning",
     "256 channels, per token"
    ]
   ],
   "parent": "trunk",
   "children": [
    "attention",
    "adaptive",
    "transition"
   ],
   "kind": "ncd",
   "page": "ncd/layer.html",
   "code": [
    {
     "label": "attn_n_transition.py L96-L113",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/nn/modules/attn_n_transition.py#L96-L113"
    },
    {
     "label": "local_latents_transformer.py L108-L124",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/nn/local_latents_transformer.py#L108-L124"
    }
   ]
  },
  "attention": {
   "title": "Pair-biased attention",
   "short": "Attention",
   "blurb": "Gated multi-head attention with a per-head bias read off the pair representation.",
   "summary": "<p>Queries, keys, values and a gate are projected from the normalised token, with biases. Queries and keys are RMS-normalised across all 768 channels jointly before the split into 12 heads. Each head adds a bias <strong>W</strong><sup>B</sup> RMS(<em>z</em><sub>ij</sub>) to its scaled logits, the softmax runs over keys, and the sigmoid gate multiplies the result before the output projection.</p>",
   "highlights": [
    {
     "kind": "note",
     "label": "Kernel",
     "html": "With <code>attention_type: torch</code> the attention runs through PyTorch SDPA with the pair bias as <code>attn_mask</code>, which selects the memory-efficient kernel; the pair bias of pairs that touch padding is replaced by −10⁴. Inference runs the same native path by default (<code>inference_backend: native</code>); the optional BioIR backend (<code>inference_backend: bioir</code>) runs the workflow through BioIR's fused kernels, in bf16 by default."
    }
   ],
   "facts": [
    [
     "Heads × width",
     "12 × 64"
    ],
    [
     "Pair bias",
     "Linear 384 → 12, no bias"
    ],
    [
     "Norms",
     "RMSNorm on the input, q, k and z"
    ]
   ],
   "parent": "layer",
   "children": [],
   "kind": "ncd",
   "page": "ncd/attention.html",
   "code": [
    {
     "label": "pair_bias_attn.py L69-L193",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/nn/modules/pair_bias_attn.py#L69-L193"
    },
    {
     "label": "pair_bias_attn.py L296-L330",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/nn/modules/pair_bias_attn.py#L296-L330"
    }
   ]
  },
  "adaptive": {
   "title": "AdaLN and the output gate",
   "short": "AdaLN & gate",
   "blurb": "How the conditioning scales, shifts and gates every sublayer.",
   "summary": "<p><strong>AdaLN</strong> divides each token by its root mean square and applies a scale σ(<strong>W</strong><sup>γ</sup> RMS(<em>c</em>) + <em>b</em>) and a shift <strong>W</strong><sup>β</sup> RMS(<em>c</em>) read off the token's conditioning. The <strong>output gate</strong> multiplies the sublayer's result by σ(<strong>W</strong><sup>gate</sup><em>c</em> + <em>b</em><sup>gate</sup>), with the weight initialised to zero and the bias to −2.</p>",
   "highlights": [],
   "facts": [
    [
     "AdaLN parameters",
     "394,240 per use"
    ],
    [
     "Gate parameters",
     "197,376 per use"
    ],
    [
     "Uses",
     "2 AdaLNs and 2 gates per layer, 28 of each in the trunk"
    ]
   ],
   "parent": "layer",
   "children": [],
   "kind": "ncd",
   "page": "ncd/adaptive.html",
   "code": [
    {
     "label": "adaptive_ln_scale.py L11-L38",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/nn/modules/adaptive_ln_scale.py#L11-L38"
    },
    {
     "label": "adaptive_ln_scale.py L41-L64",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/nn/modules/adaptive_ln_scale.py#L41-L64"
    }
   ]
  },
  "transition": {
   "title": "SwiGLU transition",
   "short": "SwiGLU",
   "blurb": "The per-token feed-forward block of every layer.",
   "summary": "<p><strong>W</strong><sup>D</sup>(SiLU(<strong>W</strong><sup>G</sup><em>s</em>) ⊙ <strong>W</strong><sup>U</sup><em>s</em>) with hidden width 3072 and no biases. The code computes <strong>W</strong><sup>G</sup> and <strong>W</strong><sup>U</sup> as one 768 → 6144 map and splits it.</p>",
   "highlights": [],
   "facts": [
    [
     "Parameters",
     "7,077,888 per layer"
    ]
   ],
   "parent": "layer",
   "children": [],
   "kind": "ncd",
   "page": "ncd/transition.html",
   "code": [
    {
     "label": "seq_transition_af3.py L12-L42",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/nn/modules/seq_transition_af3.py#L12-L42"
    },
    {
     "label": "seq_transition_af3.py L45-L68",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/nn/modules/seq_transition_af3.py#L45-L68"
    }
   ]
  },
  "pair-update": {
   "title": "Pair update",
   "short": "Pair update",
   "blurb": "Single-to-pair write, two triangle multiplications and a pair transition.",
   "summary": "<p>The single track writes into the pair track with a rank-one Hadamard product of two projections, then the outgoing and incoming <strong>triangle multiplicative updates</strong> of AlphaFold 2 and a <strong>pair transition</strong> rewrite it, each added to the residual. The block reads the single representation and returns only the pair representation; in the trunk, <em>s</em> passes around it.</p>",
   "highlights": [
    {
     "kind": "note",
     "label": "At initialisation",
     "html": "The triangle and transition outputs start at zero, so a fresh pair update adds only the rank-one write."
    },
    {
     "kind": "open",
     "label": "Try",
     "html": "Click <strong>TriOut</strong>, <strong>TriIn</strong> or <strong>PairTrans</strong>."
    }
   ],
   "facts": [
    [
     "Parameters",
     "2,462,336 per pair update"
    ],
    [
     "Triangle hidden width",
     "256"
    ],
    [
     "Runs",
     "after layers 0, 2, 4, 6, 8, 10, 12"
    ]
   ],
   "parent": "trunk",
   "children": [
    "triangle",
    "pair-transition"
   ],
   "kind": "ncd",
   "page": "ncd/pair-update.html",
   "code": [
    {
     "label": "pair_update.py L60-L91",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/nn/modules/pair_update.py#L60-L91"
    },
    {
     "label": "local_latents_transformer.py L126-L147",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/nn/local_latents_transformer.py#L126-L147"
    }
   ]
  },
  "triangle": {
   "title": "Triangle multiplication",
   "short": "Triangles",
   "blurb": "Outgoing and incoming triangle multiplicative updates.",
   "summary": "<p>Each pair (<em>i</em>, <em>j</em>) sums, over every token <em>k</em>, the product of two gated projections of the edges that close the triangle <em>i</em>, <em>j</em>, <em>k</em>: <em>i</em>–<em>k</em> and <em>j</em>–<em>k</em> (outgoing) or <em>k</em>–<em>i</em> and <em>k</em>–<em>j</em> (incoming). The contraction over <em>k</em> is the trunk's only step whose cost grows with the cube of the number of tokens, yet at design sizes it is not where most of the work goes: it is about 6% of a trunk pass's multiply-adds at 256 tokens and 21% at 1024, and it overtakes the per-pair projections only near 3,700 tokens.</p>",
   "highlights": [],
   "facts": [
    [
     "Parameters",
     "639,616 per triangle update, without biases (<code>tri_mult_bias: false</code>)"
    ],
    [
     "Hidden width",
     "256"
    ]
   ],
   "parent": "pair-update",
   "children": [],
   "kind": "ncd",
   "page": "ncd/triangle.html",
   "code": [
    {
     "label": "triangular_multiplicative_update.py L178-L226",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/nn/modules/openfold_modules/triangular_multiplicative_update.py#L178-L226"
    },
    {
     "label": "triangular_multiplicative_update.py L229-L246",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/nn/modules/openfold_modules/triangular_multiplicative_update.py#L229-L246"
    },
    {
     "label": "local_latents_score_nn_160M_atom37.yaml L100-L102",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/configs/nn/local_latents_score_nn_160M_atom37.yaml#L100-L102"
    },
    {
     "label": "triangular_multiplicative_update.py L249-L266",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/nn/modules/openfold_modules/triangular_multiplicative_update.py#L249-L266"
    }
   ]
  },
  "pair-transition": {
   "title": "Pair transition",
   "short": "Pair transition",
   "blurb": "A two-layer ReLU network on every pair.",
   "summary": "<p>LayerNorm, a linear map from 384 to 768 channels, ReLU and a linear map back, on every pair independently: the AlphaFold 2 pair transition with a 2× rather than 4× expansion.</p>",
   "highlights": [],
   "facts": [
    [
     "Parameters",
     "591,744 per pair update"
    ]
   ],
   "parent": "pair-update",
   "children": [],
   "kind": "ncd",
   "page": "ncd/pair-transition.html",
   "code": [
    {
     "label": "pair_transition.py L23-L97",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/nn/modules/openfold_modules/pair_transition.py#L23-L97"
    }
   ]
  },
  "heads": {
   "title": "Output heads",
   "short": "Heads",
   "blurb": "CA velocity, latent velocity, insertion logits and the sequence readout.",
   "summary": "<p>Two RMSNorm and linear heads read the final single representation and give the velocity of the CA coordinates (3) and of the latents (32). Two probes read a stop-gradient copy: the insertion head scores every pair of tokens bilinearly, to match unindexed motif residues to generated tokens, and the sequence readout predicts each token's class among the 147 of the unified vocabulary. Generation hands the last step's sequence prediction to the decoder (<code>condition_decoder_on_sequence_prediction: true</code>), which keeps any sequence the input specifies.</p>",
   "highlights": [
    {
     "kind": "note",
     "label": "Parameterisation",
     "html": "The heads predict velocities (<code>v</code>); the clean sample is x̂<sub>1</sub> = x<sub>t</sub> + (1 − t) v̂."
    }
   ],
   "facts": [
    [
     "Velocity heads",
     "768 → 3 and 768 → 32"
    ],
    [
     "Insertion head",
     "two 768 × 768 maps, scale 768<sup>−1/2</sup>"
    ],
    [
     "Sequence readout",
     "768 → 147 on a stop-gradient copy, trained by its own cross-entropy"
    ]
   ],
   "parent": "denoiser",
   "children": [],
   "kind": "ncd",
   "page": "ncd/heads.html",
   "code": [
    {
     "label": "local_latents_transformer.py L169-L197",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/nn/local_latents_transformer.py#L169-L197"
    },
    {
     "label": "local_latents_transformer.py L298-L316",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/nn/local_latents_transformer.py#L298-L316"
    }
   ]
  },
  "integrator": {
   "title": "Integration step",
   "short": "Step",
   "blurb": "One vf_tsr step: score, temperature scaling, Euler–Maruyama.",
   "summary": "<p>The predicted velocity is converted to a score, the score is scaled by a ratio that grows from 1 to <em>k</em> as <em>t</em> goes from 0 to 1, and the state moves by an Euler–Maruyama step with Langevin noise between <em>t</em> = 0.02 and 0.95 and a deterministic Euler step outside that window. The latent step is the same expression with its own constants and a tangent noise schedule in place of 1/(<em>t</em> + 0.01).</p>",
   "highlights": [],
   "facts": [
    [
     "CA",
     "k = 16, σ<sub>tsr</sub> = 3, η = 1, noise schedule 1/(t + 0.01); g<sub>t</sub> = schedule / r(t)²"
    ],
    [
     "Latents",
     "k = 4, σ<sub>tsr</sub> = 12, η = 1, tangent noise schedule; g<sub>t</sub> = schedule / r(t)²"
    ]
   ],
   "parent": "generation",
   "children": [],
   "kind": "ncd",
   "page": "ncd/integrator.html",
   "code": [
    {
     "label": "rdn_flow_matcher.py L525-L550",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/flow_matching/rdn_flow_matcher.py#L525-L550"
    },
    {
     "label": "rdn_flow_matcher.py L553-L569",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/flow_matching/rdn_flow_matcher.py#L553-L569"
    },
    {
     "label": "product_space_flow_matcher.py L809-L855",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/flow_matching/product_space_flow_matcher.py#L809-L855"
    },
    {
     "label": "inference_base.yaml L17-L98",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/configs/generation/inference_base.yaml#L17-L98"
    }
   ]
  },
  "training": {
   "title": "Training step",
   "short": "Training",
   "blurb": "Encode, interpolate, denoise, regress the velocity.",
   "summary": "<p>The frozen encoder samples the clean latents from the structure. Both modalities are interpolated with Gaussian noise at independently drawn times, the denoiser runs with a random recycle depth, and the loss is the weighted squared velocity error of each modality plus the insertion cross-entropy against <em>src</em>, the token each unindexed motif copy was taken from, and the cross-entropy of the detached sequence readout against the true residues. Every design is tried four times with fresh noise at the same times, and only the attempt with the lowest summed CA and latent loss is kept. A signed-volume loss on ligand stereocentres (weight 2.5) is added but not drawn.</p>",
   "highlights": [
    {
     "kind": "loop",
     "label": "Recycling",
     "html": "The recycle depth is drawn from Poisson(3), clipped to [1, 9] and shared by all ranks; gradients flow through the last cycle only."
    },
    {
     "kind": "note",
     "label": "Exploration",
     "html": "<code>exploration_k: 4</code>: the four attempts of a design run in one batch; the figure draws one attempt."
    }
   ],
   "facts": [
    [
     "Target",
     "v = x<sub>1</sub> − x<sub>0</sub>"
    ],
    [
     "Loss weights",
     "1, +2.5 for tokens outside the standard amino acids and nucleotides (modified residues, ligand atoms, ions) and for unindexed sources"
    ],
    [
     "Auxiliary losses",
     "insertion cross-entropy, sequence-readout cross-entropy, signed volume (weight 2.5)"
    ],
    [
     "Recipe",
     "pre-training <code>releases/flow/2026_10_19_hero_pretrain</code>; the release is then fine-tuned with <code>releases/flow/2026_10_19_hero_lora_c384</code>"
    ]
   ],
   "parent": "overview",
   "children": [],
   "kind": "ncd",
   "page": "ncd/training.html",
   "code": [
    {
     "label": "proteina.py L593-L738",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/proteina.py#L593-L738"
    },
    {
     "label": "torch.py L33-L68",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/utils/torch.py#L33-L68"
    },
    {
     "label": "product_space_flow_matcher.py L222-L273",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/flow_matching/product_space_flow_matcher.py#L222-L273"
    },
    {
     "label": "proteina.py L789-L815",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/proteina.py#L789-L815"
    },
    {
     "label": "rdn_flow_matcher.py L356-L402",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/flow_matching/rdn_flow_matcher.py#L356-L402"
    },
    {
     "label": "product_space_flow_matcher.py L307-L504",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/flow_matching/product_space_flow_matcher.py#L307-L504"
    },
    {
     "label": "2026_10_19_hero_pretrain.yaml L448-L454",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/configs/releases/flow/2026_10_19_hero_pretrain.yaml#L448-L454"
    }
   ]
  },
  "autoencoder": {
   "title": "Partial autoencoder",
   "short": "Autoencoder",
   "blurb": "The encoder that defines the latents and the decoder that writes atoms.",
   "summary": "<p>The autoencoder is trained first, on its own: the encoder maps a structure to 32 latent channels per residue or nucleotide, and the decoder maps the CA positions and latents back to residue types, atom37 coordinates and atom presence. Neither half has a pair track: attention pair bias, pair updates and triangle multiplication are off. The flow model then learns to generate CA positions and latents, and reuses the decoder at the end of every generation.</p>",
   "highlights": [
    {
     "kind": "note",
     "label": "No conditioning",
     "html": "Both halves are built with LayerNorms and an empty conditioning list, so each AdaLN is a LayerNorm with a constant gain and shift, and each output gate a constant."
    }
   ],
   "facts": [
    [
     "Encoder",
     "129,013,832 parameters, 12 layers"
    ],
    [
     "Decoder",
     "128,160,329 parameters, 12 layers"
    ],
    [
     "Recipe",
     "<code>releases/ae/2026_10_15_unified_biomolecule</code> with <code>nn_ae/nn_130m_unified</code>; released checkpoint dated 2026-10-19"
    ]
   ],
   "parent": "overview",
   "children": [
    "ae-encoder",
    "ae-decoder",
    "ae-layer"
   ],
   "kind": "ncd",
   "page": "ncd/autoencoder.html",
   "code": [
    {
     "label": "autoencoder.py L1-L171",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/partial_autoencoder/autoencoder.py#L1-L171"
    },
    {
     "label": "autoencoder.py L227-L289",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/partial_autoencoder/autoencoder.py#L227-L289"
    },
    {
     "label": "training_ae.yaml L35-L54",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/configs/training_ae.yaml#L35-L54"
    },
    {
     "label": "2026_10_15_unified_biomolecule.yaml L1-L115",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/configs/releases/ae/2026_10_15_unified_biomolecule.yaml#L1-L115"
    }
   ]
  },
  "ae-encoder": {
   "title": "Encoder",
   "short": "Encoder",
   "blurb": "Structure to a Gaussian posterior over 32 latents per residue or nucleotide.",
   "summary": "<p>1587 structure channels per token (token type 32, absolute 148 and CA-relative 148 atom37 coordinates with masks, chain index 1, incident bond counts 370 and element and charge embeddings 888) are projected to 768, run through 12 layers, and mapped to the mean and log scale of a Gaussian, from which one sample is drawn.</p>",
   "highlights": [],
   "facts": [
    [
     "Inputs",
     "1587 → 768"
    ],
    [
     "Output",
     "μ, log σ ∈ ℝ<sup>32</sup>; x<sup>lat</sup> = μ + σ ε"
    ]
   ],
   "parent": "autoencoder",
   "children": [],
   "kind": "ncd",
   "page": "ncd/ae-encoder.html",
   "code": [
    {
     "label": "encoder.py L111-L158",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/partial_autoencoder/encoder.py#L111-L158"
    },
    {
     "label": "feature_factory.py L510-L546",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/nn/feature_factory.py#L510-L546"
    },
    {
     "label": "nn_130m_unified.yaml L3-L51",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/configs/nn_ae/nn_130m_unified.yaml#L3-L51"
    }
   ]
  },
  "ae-decoder": {
   "title": "Decoder",
   "short": "Decoder",
   "blurb": "CA positions and latents to residues, atom37 coordinates and atom presence.",
   "summary": "<p>The CA position, the 32 latents, the token type and the conditioned residue type are embedded, run through 12 layers, and read by three heads, each behind its own LayerNorm: 147 residue-class logits restricted to the classes the token type allows, 37 atom positions relative to the CA, to which the CA position is added, and 37 atom-presence logits. The decoded residue is the conditioned class where one is given and the most likely allowed class elsewhere; its atom mask retains legal slots whose presence logit is positive, the CA slot, and any legal atoms required by coordinate conditioning. Tokens the decoder does not emit receive no decoded atoms. In a generation the conditioned residue type of every position the input leaves open is the denoiser's sequence prediction.</p>",
   "highlights": [],
   "facts": [
    [
     "Inputs",
     "214 → 768: CA 3, latents 32, token type 32, conditioned residue 147"
    ],
    [
     "Heads",
     "768 → 147 logits, 768 → 111 (37 × 3) coordinates, 768 → 37 presence logits"
    ]
   ],
   "parent": "autoencoder",
   "children": [],
   "kind": "ncd",
   "page": "ncd/ae-decoder.html",
   "code": [
    {
     "label": "decoder.py L160-L251",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/partial_autoencoder/decoder.py#L160-L251"
    },
    {
     "label": "autoencoder.py L140-L171",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/partial_autoencoder/autoencoder.py#L140-L171"
    },
    {
     "label": "proteina.py L2041-L2085",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/proteina.py#L2041-L2085"
    },
    {
     "label": "nn_130m_unified.yaml L53-L88",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/configs/nn_ae/nn_130m_unified.yaml#L53-L88"
    }
   ]
  },
  "ae-layer": {
   "title": "Autoencoder layer",
   "short": "AE layer",
   "blurb": "Attention without a pair bias and a SwiGLU transition, with LayerNorms.",
   "summary": "<p>The module of a flow-model trunk layer, built with LayerNorms instead of RMSNorms and without a pair bias: LayerNorm, gated multi-head attention, a constant gate and a residual; then LayerNorm, SwiGLU, a constant gate and a residual. Its conditioning is identically zero and LayerNorm(0) returns the norm's bias, so each AdaLN is a LayerNorm with a constant gain and shift and each output gate a constant σ(<em>b</em><sup>gate</sup>).</p>",
   "highlights": [],
   "facts": [
    [
     "Width",
     "768, 12 heads × 64"
    ],
    [
     "Transition",
     "SwiGLU, hidden width 3072"
    ]
   ],
   "parent": "autoencoder",
   "children": [],
   "kind": "ncd",
   "page": "ncd/ae-layer.html",
   "code": [
    {
     "label": "encoder.py L69-L85",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/partial_autoencoder/encoder.py#L69-L85"
    },
    {
     "label": "attn_n_transition.py L96-L113",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/nn/modules/attn_n_transition.py#L96-L113"
    }
   ]
  }
 }
};
