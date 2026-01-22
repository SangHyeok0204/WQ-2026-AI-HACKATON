# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLM Alpha Generation Pipeline for WorldQuant Brain - an automated system that uses OpenAI GPT models to generate quantitative trading alpha formulas (FASTEXPR syntax), validates them via sanity checking, and simulates their performance on the WorldQuant Brain platform.

## Key Commands

```bash
# Start Jupyter notebook (main entry point)
jupyter notebook llm_alpha_gen/llm_alpha_guide.ipynb

# Run automated pipeline (from llm_alpha_gen directory)
python my_research.py
```

## Architecture

### 5-Stage Pipeline Flow

```
[1. Initialize]        [2. LLM Generate]     [3. Sanity Check]    [4. Simulate]      [5. Analyze]
AAF.initiate_       →  generate_expressions  →  sanity_checker()  →  multi_simulate   →  results/*.csv
datafield()            _from_dataset()          parser.tree_node()    _alphas_map()
```

### Core Modules

| Module | Purpose |
|--------|---------|
| `llm_alpha_guide.ipynb` | Main executable notebook with resume support |
| `ace_lib.py` | WorldQuant Brain API wrapper (auth, datafields, simulation) |
| `llm_functions.py` | OpenAI API calls, prompt construction, JSON parsing |
| `parser.py` | FASTEXPR expression parsing to TreeNode, operator I/O type validation |
| `AAF.py` | Datafield initialization, whitelist enforcement (ALLOWED_DATASETS) |
| `my_research.py` | Sanity checker implementation, pipeline orchestration |

### Data Flow

- **Input**: `datafield/{delay}/{region}/{universe}/*.json` - cached metadata from Brain API
- **LLM Output**: `gen_json/{dataset}_{batch}.json` - generated alpha candidates with idea/description/implementation/confidence
- **Final Output**: `results/{region}_{universe}_{category}.csv` - simulation results

## Critical Validation Rules

### Sanity Check (parser.py + my_research.py)
- Expression must parse to valid TreeNode structure
- Operator input/output types must match (MATRIX, VECTOR, NUMBER, SPECIAL_ARGUMENT)
- Final output type must be MATRIX
- Uses `operators_list.json` for type definitions

### Submission Requirements (all must PASS)
- `LOW_SHARPE`: Sharpe >= 1.58
- `LOW_FITNESS`: Fitness >= 1.0
- `LOW_TURNOVER`: Turnover >= 0.01
- `HIGH_TURNOVER`: Turnover <= 0.7
- `SELF_CORRELATION`: < 0.7
- `PROD_CORRELATION`: <= 0.7

### Soft Criteria (for prioritization)
- Fitness >= 1.5, Sharpe >= 2.0
- Turnover in 0.1-0.5 range

## Key Design Patterns

- **Whitelist Enforcement**: Only datasets in `AAF.ALLOWED_DATASETS` are processed (competition requirement)
- **Resume Support**: Checks `gen_json/{dataset}_*.json` before regenerating
- **Session Singleton**: Thread-safe WorldQuant Brain connection via `ace.start_session()`
- **Parallel Simulation**: 8-alpha concurrent ThreadPool execution

## Important Files to Understand

1. `PIPELINE_GUIDE.md` - Comprehensive technical documentation (Korean)
2. `for_gpt.txt` - Context guide for LLM prompt construction
3. `operators_list.json` - Operator definitions with I/O types (72KB)
4. `operator_inputs.json` - Additional operator type info

## Known Limitations

1. Sanity Checker validates types only - may miss syntax errors or runtime failures
2. `rank` operator excluded from LLM suggestions (overuse tendency)
3. Coverage < 0.6 fields may fail simulation despite passing sanity check
4. Some operators may be missing from `operators_list.json`

## Session Management

```python
# Check session timeout before batch operations
if ace.check_session_timeout(s) < 1000:
    s = ace.start_session()  # Re-authenticate
```

WorldQuant Brain authentication may require biometrics - follow URL prompts when session starts.
