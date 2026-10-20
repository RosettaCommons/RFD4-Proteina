# Contributing a metric

If a metric you wrote under [custom metrics](../how-to/advanced/custom-metrics.md) measures
something generally useful, contribute it back: move the class to a tracked file under
`src/rfproteina/metrics/`, add a YAML entry to a tracked config, and open a pull request.

```bash
# 1. Create a branch before making any changes
git switch -c feat/my-metric-name

# 2. Add your class to a tracked metrics file under src/rfproteina/metrics/

# 3. Add a YAML entry to an existing tracked config

# 4. Stage all your changes and commit
git add .
git commit -m "feat: describe your metric here"
git push -u origin HEAD

# 5. Open a pull request on the foundry-dev GitHub page
```

See the {doc}`contributor guide <index>` for the current commit and pull-request guidance.
