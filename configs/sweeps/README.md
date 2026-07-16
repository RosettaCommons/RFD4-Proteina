# Sweep Configs

YAML config files that define parameter sweeps for inference/evaluation.

Each sweep config specifies a base Hydra config, optional overrides, and a set of parameters to sweep over. The sweep generates a Cartesian product of all parameter combinations.

## Schema

```yaml
# Required
base_config: inference          # Hydra config name to compose on
sweep:                          # Parameters to sweep (Cartesian product)
  key.subkey: [val1, val2]      # Dot-notation for nested keys

# Optional
base_config_overrides: []       # Extra Hydra overrides applied before sweep
output_dir: ./configs/sweep_configs  # Where to write generated configs
run_name_prefix: my_sweep       # Prefix for auto-generated run names
```

## Usage

```bash
# Generate all sweep configs
python src/script_utils/sweep.py --sweep-config configs/sweeps/ppi_benchmark.yaml

# Dry run (show combinations without writing files)
python src/script_utils/sweep.py --sweep-config configs/sweeps/ppi_benchmark.yaml --dry-run

# Generate + submit via launch_sweep.sh
./scripts/launch_sweep.sh digs --sweep-config configs/sweeps/ppi_benchmark.yaml
```
