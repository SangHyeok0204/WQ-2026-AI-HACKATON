# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLM Alpha Generator is a Python tool that uses LLMs (OpenAI GPT models) to generate and backtest quantitative trading alpha expressions for the WorldQuant BRAIN platform. The system automates the process of discovering, validating, and simulating trading signals.

## Architecture

```
llm_alpha_gen/
├── ace_lib.py         # Core API client for WorldQuant BRAIN platform
├── llm_functions.py   # OpenAI LLM integration for alpha generation
├── parser.py          # Expression parser and AST tree builder
├── helpful_functions.py # Result formatting and file I/O utilities
├── AAF.py             # Datafield downloading with whitelist filtering
├── my_research.py     # Main orchestration script (entry point)
```

### Key Components

**ace_lib.py** - WorldQuant BRAIN API wrapper:
- `SingleSession`: Thread-safe singleton session with auto-relogin
- `generate_alpha()`: Creates alpha configuration dict for simulation
- `simulate_alpha_list()` / `simulate_alpha_list_multi()`: Concurrent simulation runners
- `get_datafields()` / `get_datasets()`: Data discovery APIs

**parser.py** - Expression tree parser:
- `parse_expression()`: Converts infix notation to function calls (e.g., `a + b` → `add(a,b)`)
- `build_tree()` / `tree_node()`: Builds AST from alpha expressions
- `TreeNode`: Node class with depth calculation, type classification, graph visualization

**llm_functions.py** - LLM integration:
- `call_llm_stream()`: Streaming OpenAI API call with JSON schema enforcement
- `generate_expressions_from_dataset()`: Generates alpha ideas from datafield metadata

**my_research.py** - Main workflow:
- Iterates through datasets, generates alphas via LLM, validates with `sanity_checker()`, simulates via BRAIN API
- Results saved to `./results/{region}_{universe}_{category}.csv`

## Environment Setup

Required environment variables (in `.env.local`):
- `OPENAI_API_KEY`: For LLM alpha generation
- `BRAIN_CREDENTIAL_EMAIL` / `BRAIN_CREDENTIAL_PASSWORD`: Optional, otherwise prompts for login

Credentials are cached in `~/secrets/platform-brain.json` after first login.

## Common Commands

```bash
# Run main alpha generation pipeline
python llm_alpha_gen/my_research.py

# Initialize datafield cache for a region
python -c "import ace_lib as ace; import AAF; s = ace.start_session(); AAF.initiate_datafield(s, {'region': 'USA', 'universe': 'TOP3000', 'delay': 1})"
```

## Key Data Flows

1. **Alpha Generation**: Dataset → `get_datafields()` → LLM prompt → JSON response → `sanity_checker()` validation
2. **Simulation**: `generate_alpha()` dict → `simulate_alpha_list()` → BRAIN API → results DataFrame
3. **Expression Parsing**: Infix string → `parse_expression()` → `build_tree()` → AST for validation

## Important Constraints

- **Whitelist Filtering**: `AAF.ALLOWED_DATASETS` restricts which datasets can be downloaded per region (competition rules)
- **Rate Limiting**: Session timeout check (`check_session_and_relogin`) runs before API calls; concurrent simulation limited to 3-8 threads
- **Expression Validation**: `sanity_checker()` validates operator input/output types against datafield types before simulation

## File Output Locations

- `./datafield/{delay}/{region}/{universe}/`: Cached datafield JSON files
- `./gen_json/`: Raw LLM-generated alpha JSONs
- `./results/`: Simulation results CSVs
- `./simulation_results/`: Individual alpha result JSONs
- `ace.log`: Debug/info logging output
