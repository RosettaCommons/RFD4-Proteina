# How-to guides

Task-oriented instructions for running RFD4-Proteina: the config overrides and CLI flags for a
specific job, not why the underlying mechanism behaves the way it does. Each page links back to
its concept page for that background. Start at [Quickstart](../quickstart.rst) if you haven't run
the model at all yet.

<div class="wf-cards">
  <div class="wf-card wf-card-group">
    <a class="wf-card-link" href="input_prep/index.html">
      <span class="wf-card-title">Preparing inputs</span>
      <span class="wf-card-body">Building an annotated CIF: the agent, YAML, and Python API
      routes.</span>
    </a>
    <div class="wf-card-sublinks">
      <a class="wf-card-subbutton" href="input_prep/agent.html">Agent</a>
      <a class="wf-card-subbutton" href="input_prep/yaml-json.html">YAML/JSON spec</a>
      <a class="wf-card-subbutton" href="input_prep/python-api.html">Python API</a>
    </div>
  </div>
  <div class="wf-card wf-card-group">
    <a class="wf-card-link" href="run-the-model.html">
      <span class="wf-card-title">Running the model</span>
      <span class="wf-card-body">The core generation command: sampling workflows, inference noise,
      TSR, and troubleshooting a run.</span>
    </a>
    <div class="wf-card-sublinks">
      <a class="wf-card-subbutton" href="advanced/multi-job-runs.html">Multi-job runs</a>
      <a class="wf-card-subbutton" href="advanced/foldseek.html">FoldSeek clustering</a>
    </div>
  </div>
  <a class="wf-card" href="partial-diffusion.html">
    <span class="wf-card-title">Partial diffusion and inverse folding</span>
    <span class="wf-card-body">Starting a run from a structure you already have instead of from
    noise.</span>
  </a>
  <a class="wf-card" href="advanced/metric-guidance.html">
    <span class="wf-card-title">Metric guidance</span>
    <span class="wf-card-body">Configuring the smoke test, metric and output keys, schedules, and a
    real guidance objective.</span>
  </a>
  <a class="wf-card" href="advanced/backends.html">
    <span class="wf-card-title">Inference backends</span>
    <span class="wf-card-body">Running the workflow through the native model or BioIR, and what each
    backend supports.</span>
  </a>
  <a class="wf-card" href="search.html">
    <span class="wf-card-title">Reward guided search</span>
    <span class="wf-card-body">Configuring beam search checkpoints, rewards, continuation criteria,
    and novelty.</span>
  </a>
  <div class="wf-card wf-card-group">
    <a class="wf-card-link" href="evaluate-with-metrics.html">
      <span class="wf-card-title">Evaluate with metrics</span>
      <span class="wf-card-body">Turning on metrics inline or offline, running refolding, and
      aggregating results across jobs.</span>
    </a>
    <div class="wf-card-sublinks">
      <a class="wf-card-subbutton" href="advanced/custom-metrics.html">Custom metrics</a>
      <a class="wf-card-subbutton" href="advanced/offline-metrics.html">Offline metrics</a>
    </div>
  </div>
  <div class="wf-card wf-card-group">
    <a class="wf-card-link" href="refolding.html">
      <span class="wf-card-title">Refolding</span>
      <span class="wf-card-body">Turning on the AF3/OF3 oracle, what it costs, and running a design
      through more than one refold.</span>
    </a>
    <div class="wf-card-sublinks">
      <a class="wf-card-subbutton" href="advanced/pre-oracle-transforms.html">Pre-oracle transforms</a>
    </div>
  </div>
  <a class="wf-card" href="read-outputs.html">
    <span class="wf-card-title">Reading outputs</span>
    <span class="wf-card-body">Finding a region in a generated structure by its standard
    annotations, without hand-parsing CIFs.</span>
  </a>
  <a class="wf-card" href="faq.html">
    <span class="wf-card-title">FAQ</span>
    <span class="wf-card-body">Common questions and gotchas across condition specs, generation, and
    metrics.</span>
  </a>
</div>

More guides are coming; this page will grow with them.

```{toctree}
:hidden:
:maxdepth: 1

input_prep/index
run-the-model
partial-diffusion
advanced/metric-guidance
advanced/backends
search
evaluate-with-metrics
refolding
read-outputs
faq
```
