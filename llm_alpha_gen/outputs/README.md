# Alpha Combiner Outputs

This directory contains simulation results from `combine_and_simulate.py`.

## Files

- `results.jsonl` - Main output file with all simulation results (JSON Lines format)
- `checkpoint.json` - Checkpoint file for resume capability

## Output Schema (results.jsonl)

Each line is a JSON object with these fields:

| Field | Type | Description |
|-------|------|-------------|
| source_mdl25_id | string | Original alpha ID from mdl25.txt |
| source_mdl25_block | int | Block number in mdl25.txt |
| source_mdl138_id | string | Original alpha ID from mdl138.txt |
| source_mdl138_block | int | Block number in mdl138.txt |
| expr25 | string | Expression from mdl25 |
| expr138 | string | Expression from mdl138 |
| operator | string | Combination operator (add/min/max) |
| applied_group_zscore | bool | Whether group_zscore was applied |
| final_expression | string | Final combined expression |
| sim_status | string | SUCCESS or FAILED |
| sharpe | float | Sharpe ratio (null if failed) |
| fitness | float | Fitness score (null if failed) |
| turnover | float | Turnover (null if failed) |
| fail_reasons | string | Comma-separated fail reasons or null |
| alpha_id | string | Brain API alpha ID (null if failed) |
| created_at | string | ISO timestamp |

## Example Query (Python)

```python
import pandas as pd

# Load results
results = pd.read_json("results.jsonl", lines=True)

# Filter successful alphas with Sharpe >= 2.0
good_alphas = results[
    (results['sim_status'] == 'SUCCESS') &
    (results['sharpe'] >= 2.0)
]

# Show top performers
print(good_alphas.nlargest(10, 'sharpe')[['operator', 'sharpe', 'fitness', 'alpha_id']])
```
