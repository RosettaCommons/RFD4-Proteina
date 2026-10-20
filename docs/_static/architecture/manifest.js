window.RFD4_ARCHITECTURE = {
 "root": "overview",
 "diagrams": {
  "overview": {
   "title": "RFD4-Proteína at a glance",
   "short": "Overview",
   "blurb": "The whole system: inputs, the flow-matching sampler, the recycled denoiser and the decoder.",
   "summary": "<p>RFD4-Proteína generates a design by flow matching in a product space: the CA position of every token (atom37 slot 1: C1′ for a nucleotide), in nm, and 16 latent channels per residue or nucleotide learned by a partial autoencoder. A generation starts from Gaussian noise, runs 60 steps of a stochastic sampler whose velocity comes from the <em>denoiser</em>, a 14-layer pair-biased transformer, and hands the final CA positions and latents to the autoencoder's decoder, which writes the sequence and every atom of the residues and nucleotides; atomized ligand atoms keep the positions the flow produced.</p><p>Click a box to open its diagram; the annotated CIF and design boxes open the workflow pages on inputs and outputs. Every diagram below this map is a neural circuit diagram generated from an algebraic expression of the code on <code>main</code> at commit <code>19540b0</code>, drawn at the dimensions of the 380k checkpoint. In those diagrams, hover a box to see its formula, description and code, click a box whose details offer <em>Open diagram</em> to go one level deeper (the pointer changes over it), and use the breadcrumbs or the tree to go back up.</p>",
   "highlights": [
    {
     "kind": "loop",
     "label": "Recycling",
     "html": "Every network evaluation runs the trunk <em>R</em> times, feeding the previous single and pair states back through zero-initialised projections. <em>R</em> follows the schedule 1, 2, 3, 5, 6, 5, 3, 2, 1 across the 60 steps: 185 trunk passes per design."
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
     "178.2 M parameters (main, 380k dimensions)"
    ],
    [
     "Autoencoder",
     "255.9 M parameters, frozen during flow training"
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
     "16 channels per residue or nucleotide"
    ],
    [
     "Checkpoint",
     "<code>chk_epoch_00000189_step_000000380000-ema.ckpt</code>"
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
   "summary": "<p>The design features of the annotated CIF are computed and put on the tape; the denoiser reshuffles their coordinate-group labels at every network evaluation. The CA positions and latents start as Gaussian noise, and the loop under the <em>T</em> bracket runs 60 times: the schedule gives both modalities their time and step size, the <strong>Denoiser</strong> predicts both velocities (with the recycle depth of that step), and one <strong>Step</strong> per modality integrates them. The <strong>Decoder</strong> then turns the final CA positions and latents into residue logits and atom37 coordinates.</p>",
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
     "<code>vf_tsr</code>: temperature-scaled score (temporal score rescaling) with a Langevin term between t = 0.02 and 0.98"
    ],
    [
     "Network evaluations",
     "60, with 185 trunk passes under <code>midpeak6_60</code>"
    ],
    [
     "Outputs",
     "sequence, atom37 coordinates, insertion mapping"
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
     "label": "vanilla.py L172-L300",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/search/vanilla.py#L172-L300"
    },
    {
     "label": "proteina.py L1819-L1930",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/proteina.py#L1819-L1930"
    },
    {
     "label": "generate.py L630-L757",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/generate.py#L630-L757"
    }
   ]
  },
  "denoiser": {
   "title": "Denoiser and its recycling loop",
   "short": "Denoiser",
   "blurb": "LocalLatentsTransformer: embeddings, the recycled trunk and the output heads.",
   "summary": "<p>One evaluation of the flow network. The <em>feature factories</em> build the conditioning <em>c</em> and the initial single and pair representations <em>s</em><sub>0</sub>, <em>z</em><sub>0</sub> once, and drop them on the tape. The <strong>Recycling</strong> block, under the <em>R</em> bracket, then runs <em>R</em> times: each cycle adds an RMS-normalised projection of the previous cycle's states to <em>s</em><sub>0</sub> and <em>z</em><sub>0</sub>, reads <em>c</em> from the tape and runs the whole <strong>Trunk</strong>, which passes <em>c</em> through its layers. The first cycle starts from zeros. The heads read the final single representation.</p>",
   "highlights": [
    {
     "kind": "loop",
     "label": "Recycling",
     "html": "The projections <strong>W</strong><sup>s</sup><sub>rec</sub> and <strong>W</strong><sup>z</sup><sub>rec</sub> start at zero, so at first every cycle starts from <em>s</em><sub>0</sub> and <em>z</em><sub>0</sub> alone and <em>R</em> cycles compute the same as one pass. Only the last cycle keeps gradients in training. State does not carry across sampling steps."
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
     "178,242,737 on main at these dimensions (checkpoint: 178,336,624)"
    ],
    [
     "Recycle depth R",
     "1 to 6 per step at inference; Poisson(3) clipped to [1, 6] in training"
    ],
    [
     "Inputs",
     "noisy CA (x, 3), noisy latents (x, 16), two times"
    ],
    [
     "Outputs",
     "CA velocity, latent velocity, insertion logits (x, x)"
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
     "label": "local_latents_transformer.py L33-L169",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/nn/local_latents_transformer.py#L33-L169"
    },
    {
     "label": "local_latents_transformer.py L207-L238",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/nn/local_latents_transformer.py#L207-L238"
    },
    {
     "label": "local_latents_score_nn_160M_atom37.yaml L1-L112",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/configs/nn/local_latents_score_nn_160M_atom37.yaml#L1-L112"
    }
   ]
  },
  "embeddings": {
   "title": "Single and pair embeddings",
   "short": "Embeddings",
   "blurb": "How the noisy state and the design features become s₀ and z₀.",
   "summary": "<p>The single embedding is one linear map of 748 concatenated token features, drawn as three maps by where the inputs come from: the noisy CA coordinates, the noisy latents, and the 729 channels of design features, which do not depend on the noisy state (their 37 coordinate-group channels are relabelled at random on every evaluation). The pair embedding bins the pairwise distances of the noisy CA positions into 30 classes, adds 77 channels of pair design features, normalises and applies an AdaLN conditioned on the two times.</p>",
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
     "748 → 768 (the released checkpoint: 882)"
    ],
    [
     "Pair features",
     "107 → 384"
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
     "label": "local_latents_transformer.py L59-L66",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/nn/local_latents_transformer.py#L59-L66"
    },
    {
     "label": "local_latents_transformer.py L215-L217",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/nn/local_latents_transformer.py#L215-L217"
    },
    {
     "label": "feature_factory.py L1317-L1343",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/nn/feature_factory.py#L1317-L1343"
    },
    {
     "label": "2026_09_15_release.yaml L107-L127",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/configs/releases/flow/2026_09_15_release.yaml#L107-L127"
    },
    {
     "label": "pair_rep_initial.py L7-L47",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/nn/modules/pair_rep_initial.py#L7-L47"
    },
    {
     "label": "feature_factory.py L97-L136",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/nn/feature_factory.py#L97-L136"
    },
    {
     "label": "feature_factory.py L1018-L1036",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/nn/feature_factory.py#L1018-L1036"
    },
    {
     "label": "2026_09_15_release.yaml L129-L140",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/configs/releases/flow/2026_09_15_release.yaml#L129-L140"
    }
   ]
  },
  "conditioning": {
   "title": "Conditioning",
   "short": "Conditioning",
   "blurb": "Time embeddings and CATH labels become the vector c read by every AdaLN and output gate of the trunk.",
   "summary": "<p>Both flow times are embedded with 128 sinusoid pairs, the three CATH levels of each token are looked up in three tables, and the parts are projected to 256 channels and summed. Two SwiGLU transitions without a residual finish the conditioning vector <em>c</em>, which every recycling cycle reads from the tape and the trunk hands to every AdaLN and every output gate.</p>",
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
     "608 channels → 256"
    ],
    [
     "Time embedding",
     "256 channels, sin and cos of 2000 t ω<sub>k</sub>"
    ],
    [
     "CATH tables",
     "6, 44 and 1473 rows of width 32"
    ]
   ],
   "parent": "denoiser",
   "children": [],
   "kind": "ncd",
   "page": "ncd/conditioning.html",
   "code": [
    {
     "label": "local_latents_transformer.py L68-L78",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/nn/local_latents_transformer.py#L68-L78"
    },
    {
     "label": "local_latents_transformer.py L211-L213",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/nn/local_latents_transformer.py#L211-L213"
    },
    {
     "label": "feature_factory.py L222-L260",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/nn/feature_factory.py#L222-L260"
    },
    {
     "label": "local_latents_score_nn_160M_atom37.yaml L47-L56",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/configs/nn/local_latents_score_nn_160M_atom37.yaml#L47-L56"
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
     "2,467,200 parameters (main)"
    ],
    [
     "Whole trunk",
     "174.4 M of the denoiser's 178.2 M"
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
     "label": "local_latents_transformer.py L240-L256",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/nn/local_latents_transformer.py#L240-L256"
    },
    {
     "label": "local_latents_transformer.py L97-L135",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/nn/local_latents_transformer.py#L97-L135"
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
     "label": "attn_n_transition.py L94-L111",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/nn/modules/attn_n_transition.py#L94-L111"
    },
    {
     "label": "local_latents_transformer.py L98-L114",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/nn/local_latents_transformer.py#L98-L114"
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
     "html": "With the training backend (<code>execution.training</code> = torch) the attention runs through PyTorch SDPA with the pair bias as <code>attn_mask</code>, which selects the memory-efficient kernel; the pair bias of pairs that touch padding is replaced by −10⁴. Default inference (<code>execution.inference</code> = bioir) swaps each layer for a fused bionemo_ir block in bf16 that masks padding with −10⁶."
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
     "label": "pair_bias_attn.py L51-L172",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/nn/modules/pair_bias_attn.py#L51-L172"
    },
    {
     "label": "pair_bias_attn.py L275-L309",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/nn/modules/pair_bias_attn.py#L275-L309"
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
     "2,467,200 per pair update (main)"
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
     "label": "pair_update.py L57-L88",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/nn/modules/pair_update.py#L57-L88"
    },
    {
     "label": "local_latents_transformer.py L116-L135",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/nn/local_latents_transformer.py#L116-L135"
    }
   ]
  },
  "triangle": {
   "title": "Triangle multiplication",
   "short": "Triangles",
   "blurb": "Outgoing and incoming triangle multiplicative updates.",
   "summary": "<p>Each pair (<em>i</em>, <em>j</em>) sums, over every token <em>k</em>, the product of two gated projections of the edges that close the triangle <em>i</em>, <em>j</em>, <em>k</em>: <em>i</em>–<em>k</em> and <em>j</em>–<em>k</em> (outgoing) or <em>k</em>–<em>i</em> and <em>k</em>–<em>j</em> (incoming). The contraction over <em>k</em> is the trunk's only step whose cost grows with the cube of the number of tokens, yet at design sizes it is not where most of the work goes: it is about 6% of a trunk pass's multiply-adds at 256 tokens and 21% at 1024, and it overtakes the per-pair projections only near 3,700 tokens.</p>",
   "highlights": [
    {
     "kind": "note",
     "label": "Checkpoint",
     "html": "The 380k checkpoint trains these without biases (<code>tri_mult_bias: false</code>); main always has them."
    }
   ],
   "facts": [
    [
     "Parameters",
     "642,048 per triangle update (main)"
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
     "label": "triangular_multiplicative_update.py L61-L87",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/nn/modules/openfold_modules/triangular_multiplicative_update.py#L61-L87"
    },
    {
     "label": "triangular_multiplicative_update.py L90-L105",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/nn/modules/openfold_modules/triangular_multiplicative_update.py#L90-L105"
    },
    {
     "label": "triangular_multiplicative_update.py L108-L123",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/nn/modules/openfold_modules/triangular_multiplicative_update.py#L108-L123"
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
   "blurb": "CA velocity, latent velocity and insertion logits.",
   "summary": "<p>Two RMSNorm and linear heads read the final single representation and give the velocity of the CA coordinates (3) and of the latents (16). The insertion head scores every pair of tokens bilinearly on a stop-gradient copy, to match unindexed motif residues to generated tokens.</p>",
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
     "768 → 3 and 768 → 16"
    ],
    [
     "Insertion head",
     "two 768 × 768 maps, scale 768<sup>−1/2</sup>"
    ]
   ],
   "parent": "denoiser",
   "children": [],
   "kind": "ncd",
   "page": "ncd/heads.html",
   "code": [
    {
     "label": "local_latents_transformer.py L154-L167",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/nn/local_latents_transformer.py#L154-L167"
    },
    {
     "label": "local_latents_transformer.py L224-L238",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/nn/local_latents_transformer.py#L224-L238"
    }
   ]
  },
  "integrator": {
   "title": "Integration step",
   "short": "Step",
   "blurb": "One vf_tsr step: score, temperature scaling, Euler–Maruyama.",
   "summary": "<p>The predicted velocity is converted to a score, the score is scaled by a ratio that grows from 1 to <em>k</em> as <em>t</em> goes from 0 to 1, and the state moves by an Euler–Maruyama step with Langevin noise between <em>t</em> = 0.02 and 0.98 and a deterministic Euler step outside that window. The latent step is the same expression with its own constants and a tangent noise schedule in place of 1/(<em>t</em> + 0.01).</p>",
   "highlights": [],
   "facts": [
    [
     "CA",
     "k = 16, σ<sub>tsr</sub> = 3, η = 0.3, noise schedule 1/(t + 0.01); g<sub>t</sub> = schedule / r(t)"
    ],
    [
     "Latents",
     "k = 4, σ<sub>tsr</sub> = 8, η = 0.1, tangent noise schedule; g<sub>t</sub> = schedule / r(t)"
    ]
   ],
   "parent": "generation",
   "children": [],
   "kind": "ncd",
   "page": "ncd/integrator.html",
   "code": [
    {
     "label": "rdn_flow_matcher.py L561-L586",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/flow_matching/rdn_flow_matcher.py#L561-L586"
    },
    {
     "label": "rdn_flow_matcher.py L589-L605",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/flow_matching/rdn_flow_matcher.py#L589-L605"
    },
    {
     "label": "product_space_flow_matcher.py L730-L775",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/flow_matching/product_space_flow_matcher.py#L730-L775"
    },
    {
     "label": "inference_base.yaml L16-L99",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/configs/generation/inference_base.yaml#L16-L99"
    }
   ]
  },
  "training": {
   "title": "Training step",
   "short": "Training",
   "blurb": "Encode, interpolate, denoise, regress the velocity.",
   "summary": "<p>The frozen encoder samples the clean latents from the structure. Both modalities are interpolated with Gaussian noise at independently drawn times, the denoiser runs with a random recycle depth, and the loss is the weighted squared velocity error of each modality plus the insertion cross-entropy against <em>src</em>, the token each unindexed motif copy was taken from.</p>",
   "highlights": [
    {
     "kind": "loop",
     "label": "Recycling",
     "html": "The recycle depth is drawn from Poisson(3), clipped to [1, 6] and shared by all ranks; gradients flow through the last cycle only."
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
    ]
   ],
   "parent": "overview",
   "children": [],
   "kind": "ncd",
   "page": "ncd/training.html",
   "code": [
    {
     "label": "proteina.py L504-L626",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/proteina.py#L504-L626"
    },
    {
     "label": "product_space_flow_matcher.py L215-L266",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/flow_matching/product_space_flow_matcher.py#L215-L266"
    },
    {
     "label": "proteina.py L677-L703",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/proteina.py#L677-L703"
    },
    {
     "label": "rdn_flow_matcher.py L356-L402",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/flow_matching/rdn_flow_matcher.py#L356-L402"
    },
    {
     "label": "product_space_flow_matcher.py L300-L425",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/flow_matching/product_space_flow_matcher.py#L300-L425"
    },
    {
     "label": "2026_09_15_release.yaml L142-L148",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/configs/releases/flow/2026_09_15_release.yaml#L142-L148"
    }
   ]
  },
  "autoencoder": {
   "title": "Partial autoencoder",
   "short": "Autoencoder",
   "blurb": "The encoder that defines the latents and the decoder that writes atoms.",
   "summary": "<p>The autoencoder is trained first, on its own: the encoder maps a structure to 16 latent channels per residue or nucleotide, and the decoder maps the CA positions and latents back to residue types and atom37 coordinates. The flow model then learns to generate CA positions and latents, and reuses the decoder at the end of every generation.</p>",
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
     "128.0 M parameters, 12 layers"
    ],
    [
     "Decoder",
     "127.9 M parameters, 12 layers"
    ],
    [
     "Checkpoint",
     "<code>2026_09_04_release_ae_z16_400k</code>"
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
     "label": "autoencoder.py L1-L138",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/partial_autoencoder/autoencoder.py#L1-L138"
    },
    {
     "label": "autoencoder.py L236-L280",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/partial_autoencoder/autoencoder.py#L236-L280"
    },
    {
     "label": "training_ae.yaml L36-L52",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/configs/training_ae.yaml#L36-L52"
    },
    {
     "label": "2026_09_04_release.yaml L1-L60",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/configs/releases/ae/2026_09_04_release.yaml#L1-L60"
    }
   ]
  },
  "ae-encoder": {
   "title": "Encoder",
   "short": "Encoder",
   "blurb": "Structure to a Gaussian posterior over 16 latents per residue or nucleotide.",
   "summary": "<p>329 structure channels per token (token type, absolute and CA-relative atom37 coordinates with masks, chain index) are projected to 768, run through 12 layers, and mapped to the mean and log scale of a Gaussian, from which one sample is drawn.</p>",
   "highlights": [],
   "facts": [
    [
     "Inputs",
     "329 → 768"
    ],
    [
     "Output",
     "μ, log σ ∈ ℝ<sup>16</sup>; x<sup>lat</sup> = μ + σ ε"
    ]
   ],
   "parent": "autoencoder",
   "children": [],
   "kind": "ncd",
   "page": "ncd/ae-encoder.html",
   "code": [
    {
     "label": "encoder.py L113-L162",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/partial_autoencoder/encoder.py#L113-L162"
    },
    {
     "label": "feature_factory.py L434-L470",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/nn/feature_factory.py#L434-L470"
    },
    {
     "label": "2026_09_04_release.yaml L45-L57",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/configs/releases/ae/2026_09_04_release.yaml#L45-L57"
    }
   ]
  },
  "ae-decoder": {
   "title": "Decoder",
   "short": "Decoder",
   "blurb": "CA positions and latents to residue logits and atom37 coordinates.",
   "summary": "<p>The CA position, the 16 latents, the token type and the conditioned residue type are embedded, run through 12 layers, and read by two heads, each behind its own LayerNorm: 33 residue-class logits restricted to the classes the token type allows, and 37 atom positions relative to the CA, to which the CA position is added.</p>",
   "highlights": [
    {
     "kind": "note",
     "label": "Checkpoint",
     "html": "On the 380k checkpoint's branch the decoder's residue-type input is filled from the flow model's sequence readout; main has no readout."
    }
   ],
   "facts": [
    [
     "Inputs",
     "84 → 768"
    ],
    [
     "Heads",
     "768 → 33 logits, 768 → 37 × 3 coordinates"
    ]
   ],
   "parent": "autoencoder",
   "children": [],
   "kind": "ncd",
   "page": "ncd/ae-decoder.html",
   "code": [
    {
     "label": "decoder.py L148-L236",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/partial_autoencoder/decoder.py#L148-L236"
    },
    {
     "label": "autoencoder.py L118-L138",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/partial_autoencoder/autoencoder.py#L118-L138"
    },
    {
     "label": "nn_130m_unified.yaml L48-L83",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/configs/nn_ae/nn_130m_unified.yaml#L48-L83"
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
     "label": "encoder.py L72-L88",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/partial_autoencoder/encoder.py#L72-L88"
    },
    {
     "label": "attn_n_transition.py L94-L111",
     "url": "https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/nn/modules/attn_n_transition.py#L94-L111"
    }
   ]
  }
 }
};
