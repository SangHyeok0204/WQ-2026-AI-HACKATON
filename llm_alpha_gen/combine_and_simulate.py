#!/usr/bin/env python3
"""
Alpha Combiner and Simulator

Combines alphas from mdl25 and mdl138 using allowed operators and simulates on Brain API.
Allowed operators: add, min, max, group_zscore(x, sector)

Usage:
    python combine_and_simulate.py [--dry-run] [--skip-zscore]
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add parent directory to path for ace_lib import
sys.path.insert(0, str(Path(__file__).parent))

import ace_lib as ace

# =============================================================================
# Configuration
# =============================================================================

# File paths
MDL25_FILE = Path(__file__).parent / "mdl25.txt"
MDL138_FILE = Path(__file__).parent / "mdl138.txt"
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "results.jsonl"
CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint.json"

# Simulation settings
REGION = "EUR"
UNIVERSE = "TOP2500"
DELAY = 1
DECAY = 0
NEUTRALIZATION = "INDUSTRY"

# Concurrency settings
MAX_CONCURRENT_SIMULATIONS = 3
BATCH_SIZE = 10  # Number of alphas per multi-simulation batch

# Allowed operators
ALLOWED_OPERATORS = ["add", "min", "max"]
GROUP_ZSCORE_GROUP = "sector"

# Retry settings
MAX_RETRIES = 3
RETRY_DELAY_BASE = 5  # Base delay in seconds for exponential backoff


# =============================================================================
# Parsing Functions
# =============================================================================

def parse_fail0_expressions(file_path: Path) -> list[dict]:
    """
    Parse a txt file and extract expressions from FAIL: 0 blocks.

    Returns a list of dicts with keys: id, expression, sharpe, fitness, turnover
    """
    content = file_path.read_text(encoding="utf-8")

    # Pattern to match each block
    block_pattern = re.compile(
        r"--- #(\d+) \| FAIL: (\d+) ---\n"
        r"ID: ([A-Za-z0-9]+)\n"
        r"Region: ([A-Z]+), Universe: ([A-Z0-9]+)\n"
        r"Sharpe: ([\d.]+), Fitness: ([\d.]+), Turnover: ([\d.]+)\n"
        r"Expression: (.+?)\n"
        r"Failed: (.+?)(?=\n\n|$)",
        re.DOTALL
    )

    results = []
    for match in block_pattern.finditer(content):
        block_num = int(match.group(1))
        fail_count = int(match.group(2))
        alpha_id = match.group(3)
        region = match.group(4)
        universe = match.group(5)
        sharpe = float(match.group(6))
        fitness = float(match.group(7))
        turnover = float(match.group(8))
        expression = match.group(9).strip()
        failed_reasons = match.group(10).strip()

        if fail_count == 0:
            results.append({
                "block_num": block_num,
                "id": alpha_id,
                "region": region,
                "universe": universe,
                "sharpe": sharpe,
                "fitness": fitness,
                "turnover": turnover,
                "expression": expression,
            })

    return results


# =============================================================================
# Combination Generation
# =============================================================================

def generate_combinations(
    mdl25_alphas: list[dict],
    mdl138_alphas: list[dict],
    apply_group_zscore: bool = True
) -> list[dict]:
    """
    Generate all combinations of alphas using allowed operators.

    For each (mdl25_expr, mdl138_expr) pair:
    - Creates op(expr25, expr138) for op in [add, min, max]
    - Optionally wraps with group_zscore(base, sector)

    Returns a list of dicts with combination metadata.
    """
    combinations = []
    seen_expressions = set()

    for alpha25 in mdl25_alphas:
        for alpha138 in mdl138_alphas:
            expr25 = alpha25["expression"]
            expr138 = alpha138["expression"]

            for op in ALLOWED_OPERATORS:
                # Wrap expressions in parentheses for safety
                base_expr = f"{op}(({expr25}),({expr138}))"

                # Add base expression
                if base_expr not in seen_expressions:
                    seen_expressions.add(base_expr)
                    combinations.append({
                        "source_mdl25_id": alpha25["id"],
                        "source_mdl25_block": alpha25["block_num"],
                        "source_mdl138_id": alpha138["id"],
                        "source_mdl138_block": alpha138["block_num"],
                        "expr25": expr25,
                        "expr138": expr138,
                        "operator": op,
                        "applied_group_zscore": False,
                        "final_expression": base_expr,
                    })

                # Add group_zscore variant if enabled
                if apply_group_zscore:
                    zscore_expr = f"group_zscore(({base_expr}),{GROUP_ZSCORE_GROUP})"
                    if zscore_expr not in seen_expressions:
                        seen_expressions.add(zscore_expr)
                        combinations.append({
                            "source_mdl25_id": alpha25["id"],
                            "source_mdl25_block": alpha25["block_num"],
                            "source_mdl138_id": alpha138["id"],
                            "source_mdl138_block": alpha138["block_num"],
                            "expr25": expr25,
                            "expr138": expr138,
                            "operator": op,
                            "applied_group_zscore": True,
                            "final_expression": zscore_expr,
                        })

    return combinations


# =============================================================================
# Checkpoint Management
# =============================================================================

def load_checkpoint() -> set[str]:
    """Load set of already simulated expressions from checkpoint file."""
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return set(data.get("completed_expressions", []))
        except (json.JSONDecodeError, KeyError):
            return set()
    return set()


def save_checkpoint(completed_expressions: set[str]):
    """Save set of completed expressions to checkpoint file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "completed_expressions": list(completed_expressions),
            "last_updated": datetime.now().isoformat()
        }, f, indent=2)


def append_result(result: dict):
    """Append a single result to the JSONL output file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


# =============================================================================
# Simulation Functions
# =============================================================================

def simulate_with_retry(
    session,
    expression: str,
    max_retries: int = MAX_RETRIES
) -> Optional[dict]:
    """
    Simulate a single alpha expression with exponential backoff retry.

    Returns simulation result dict or None if all retries fail.
    """
    simulate_data = ace.generate_alpha(
        regular=expression,
        alpha_type="REGULAR",
        region=REGION,
        universe=UNIVERSE,
        delay=DELAY,
        decay=DECAY,
        neutralization=NEUTRALIZATION,
    )

    for attempt in range(max_retries):
        try:
            session = ace.check_session_and_relogin(session)
            simulate_response = ace.start_simulation(session, simulate_data)

            if simulate_response.status_code == 429:
                # Rate limited - wait and retry
                wait_time = RETRY_DELAY_BASE * (2 ** attempt)
                print(f"  Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue

            if simulate_response.status_code >= 500:
                # Server error - retry
                wait_time = RETRY_DELAY_BASE * (2 ** attempt)
                print(f"  Server error {simulate_response.status_code}, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue

            # Process simulation
            result = ace.simulation_progress(session, simulate_response)

            if result["completed"]:
                return result["result"]
            else:
                # Simulation failed but not due to server error
                return None

        except Exception as e:
            wait_time = RETRY_DELAY_BASE * (2 ** attempt)
            print(f"  Error: {e}, retrying in {wait_time}s...")
            time.sleep(wait_time)

    return None


def process_combination(
    session,
    combo: dict,
    completed: set[str]
) -> Optional[dict]:
    """
    Process a single combination: simulate and return result.

    Returns None if already completed or simulation fails.
    """
    expr = combo["final_expression"]

    if expr in completed:
        return None

    print(f"Simulating: {combo['operator']}{'(group_zscore)' if combo['applied_group_zscore'] else ''} "
          f"mdl25#{combo['source_mdl25_block']} x mdl138#{combo['source_mdl138_block']}")

    sim_result = simulate_with_retry(session, expr)

    result = {
        **combo,
        "created_at": datetime.now().isoformat(),
    }

    if sim_result is None:
        result.update({
            "sim_status": "FAILED",
            "sharpe": None,
            "fitness": None,
            "turnover": None,
            "fail_reasons": "Simulation failed",
            "alpha_id": None,
        })
    else:
        # Extract metrics from simulation result
        is_data = sim_result.get("is", {})
        checks = is_data.get("checks", [])

        # Get failed checks
        failed_checks = [c["test"] for c in checks if c.get("result") == "FAIL"]

        result.update({
            "sim_status": "SUCCESS",
            "sharpe": is_data.get("sharpe"),
            "fitness": is_data.get("fitness"),
            "turnover": is_data.get("turnover"),
            "fail_reasons": ", ".join(failed_checks) if failed_checks else None,
            "alpha_id": sim_result.get("id"),
        })

    return result


def run_batch_simulation(
    session,
    combinations: list[dict],
    completed: set[str]
) -> list[dict]:
    """
    Run batch simulation using multi-simulation API.

    Returns list of result dicts.
    """
    # Filter out already completed
    pending = [c for c in combinations if c["final_expression"] not in completed]

    if not pending:
        return []

    # Prepare simulation data
    sim_data_list = []
    for combo in pending:
        sim_data = ace.generate_alpha(
            regular=combo["final_expression"],
            alpha_type="REGULAR",
            region=REGION,
            universe=UNIVERSE,
            delay=DELAY,
            decay=DECAY,
            neutralization=NEUTRALIZATION,
        )
        sim_data_list.append(sim_data)

    results = []

    # Use multi-simulation
    try:
        session = ace.check_session_and_relogin(session)
        sim_results = ace.simulate_multi_alpha(session, sim_data_list)

        for i, (combo, sim_res) in enumerate(zip(pending, sim_results)):
            result = {
                **combo,
                "created_at": datetime.now().isoformat(),
            }

            alpha_id = sim_res.get("alpha_id")

            if alpha_id is None:
                result.update({
                    "sim_status": "FAILED",
                    "sharpe": None,
                    "fitness": None,
                    "turnover": None,
                    "fail_reasons": "Simulation failed",
                    "alpha_id": None,
                })
            else:
                # Get full result
                full_result = ace.get_simulation_result_json(session, alpha_id)
                is_data = full_result.get("is", {})
                checks = is_data.get("checks", [])
                failed_checks = [c["test"] for c in checks if c.get("result") == "FAIL"]

                result.update({
                    "sim_status": "SUCCESS",
                    "sharpe": is_data.get("sharpe"),
                    "fitness": is_data.get("fitness"),
                    "turnover": is_data.get("turnover"),
                    "fail_reasons": ", ".join(failed_checks) if failed_checks else None,
                    "alpha_id": alpha_id,
                })

            results.append(result)

    except Exception as e:
        print(f"Batch simulation error: {e}")
        # Fallback to individual simulations
        for combo in pending:
            result = process_combination(session, combo, completed)
            if result:
                results.append(result)

    return results


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Combine and simulate alphas")
    parser.add_argument("--dry-run", action="store_true",
                       help="Only show combinations without simulating")
    parser.add_argument("--skip-zscore", action="store_true",
                       help="Skip group_zscore variants")
    parser.add_argument("--limit", type=int, default=0,
                       help="Limit number of combinations to process (0=all)")
    args = parser.parse_args()

    print("=" * 60)
    print("Alpha Combiner and Simulator")
    print("=" * 60)

    # Parse input files
    print(f"\nParsing {MDL25_FILE.name}...")
    mdl25_alphas = parse_fail0_expressions(MDL25_FILE)
    print(f"  Found {len(mdl25_alphas)} FAIL:0 alphas")

    print(f"\nParsing {MDL138_FILE.name}...")
    mdl138_alphas = parse_fail0_expressions(MDL138_FILE)
    print(f"  Found {len(mdl138_alphas)} FAIL:0 alphas")

    # Generate combinations
    print(f"\nGenerating combinations...")
    apply_zscore = not args.skip_zscore
    combinations = generate_combinations(mdl25_alphas, mdl138_alphas, apply_zscore)
    print(f"  Generated {len(combinations)} unique combinations")
    print(f"    - Base operators: {len(mdl25_alphas)} x {len(mdl138_alphas)} x 3 = {len(mdl25_alphas) * len(mdl138_alphas) * 3}")
    if apply_zscore:
        print(f"    - With group_zscore variants: x2")

    if args.dry_run:
        print("\n[DRY RUN] Showing first 10 combinations:")
        for i, combo in enumerate(combinations[:10]):
            print(f"  {i+1}. {combo['operator']} | zscore={combo['applied_group_zscore']}")
            print(f"      mdl25#{combo['source_mdl25_block']}: {combo['expr25'][:50]}...")
            print(f"      mdl138#{combo['source_mdl138_block']}: {combo['expr138'][:50]}...")
        print(f"\n[DRY RUN] Total: {len(combinations)} combinations would be simulated")
        return

    # Load checkpoint
    completed = load_checkpoint()
    pending_count = sum(1 for c in combinations if c["final_expression"] not in completed)
    print(f"\nCheckpoint loaded: {len(completed)} already completed")
    print(f"  Pending: {pending_count} combinations")

    if pending_count == 0:
        print("\nAll combinations already simulated!")
        return

    # Apply limit if specified
    if args.limit > 0:
        combinations = combinations[:args.limit]
        print(f"\nLimited to first {args.limit} combinations")

    # Start session
    print("\nStarting Brain API session...")
    try:
        session = ace.start_session()
        print("  Session started successfully")
    except Exception as e:
        print(f"  Failed to start session: {e}")
        return

    # Process in batches
    print(f"\nStarting simulation (batch size: {BATCH_SIZE}, concurrency: {MAX_CONCURRENT_SIMULATIONS})...")
    print("-" * 60)

    total_processed = 0
    total_success = 0
    total_failed = 0

    for batch_start in range(0, len(combinations), BATCH_SIZE):
        batch = combinations[batch_start:batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(combinations) + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"\nBatch {batch_num}/{total_batches} ({len(batch)} combinations)")

        results = run_batch_simulation(session, batch, completed)

        for result in results:
            # Save result
            append_result(result)
            completed.add(result["final_expression"])

            total_processed += 1
            if result["sim_status"] == "SUCCESS":
                total_success += 1
                sharpe = result.get("sharpe", 0) or 0
                status_str = f"OK (Sharpe: {sharpe:.2f})"
            else:
                total_failed += 1
                status_str = "FAILED"

            print(f"  [{total_processed}] {result['operator']}"
                  f"{'(zscore)' if result['applied_group_zscore'] else ''}: {status_str}")

        # Save checkpoint after each batch
        save_checkpoint(completed)

        # Small delay between batches
        time.sleep(1)

    # Summary
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    print(f"Total processed: {total_processed}")
    print(f"  Success: {total_success}")
    print(f"  Failed: {total_failed}")
    print(f"\nResults saved to: {OUTPUT_FILE}")
    print(f"Checkpoint saved to: {CHECKPOINT_FILE}")


if __name__ == "__main__":
    main()
