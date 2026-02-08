"""
Combinatorial Alpha Pipeline for WorldQuant Brain

전수 조합 기반 알파 생성 파이프라인:
1. 각 dataset의 0-fail seed 추출
2. Cartesian product로 전수 조합 생성
3. operator_list.json 기반으로 다양한 variant 생성
4. Sanity check로 invalid expression 제거
5. Brain API 시뮬레이션 및 결과 저장
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from itertools import product
import random

import sys
sys.path.insert(0, str(Path('.').resolve()))

import ace_lib as ace
import llm_functions as llm
from parser import tree_node


# ==================== Configuration ====================

# Regex patterns
ZERO_FAIL_BLOCK_PATTERN = re.compile(
    r"--- #(\d+) \| FAIL: 0[^-]*---\n"
    r"ID: ([A-Za-z0-9]+)\n"
    r"Region: ([A-Z]+), Universe: ([A-Z0-9]+)\n"
    r"Sharpe: ([\d.-]+), Fitness: ([\d.-]+), Turnover: ([\d.-]+)\n"
    r"Expression: (.+?)\n",
    re.DOTALL
)

# Operator categories for variant generation
COMBINATION_OPERATORS = {
    'arithmetic': ['add', 'subtract', 'multiply', 'divide', 'min', 'max'],
    'normalization': ['rank', 'zscore', 'quantile', 'normalize', 'scale'],
    'group': ['group_zscore', 'group_rank', 'group_neutralize', 'group_scale'],
    'tail_handling': ['winsorize', 'tail', 'pasteurize'],
}


# ==================== Sanity Check (from my_research.py) ====================

def return_type(node, operators, datafields):
    """Determine the return type of a tree node"""
    if node.node_type == 'operator':
        return operators[node.value]['output']

    if node.node_type == 'datafield':
        return datafields[node.value]['type']

    if node.node_type in ['number', 'string']:
        return "NUMBER"

    if node.node_type == 'special_argument':
        return "SPECIAL_ARGUMENT"


def check_input(operator_inputs, children_types, _debug=False):
    """Check if operator inputs match children types"""
    if _debug:
        print(operator_inputs)

    while len(operator_inputs) != 0:
        if operator_inputs[0] == []:
            operator_inputs.pop(0)
            children_types.pop(0)
        elif children_types[0] in operator_inputs[0]:
            operator_inputs.pop(0)
            children_types.pop(0)
        else:
            return False

    return True


def sanity_checker(exp: str, operators: dict, datafields: dict, _debug=False) -> Tuple[bool, Optional[str]]:
    """
    Validate expression using tree parsing and type checking

    Returns:
        (is_valid, error_message)
    """
    try:
        exp_tree = tree_node(exp)

        for node in [node for node in exp_tree.collect_all_nodes() if node.node_type == "operator"]:
            operator_name = node.value

            if operator_name not in operators:
                return False, f"Unknown operator: {operator_name}"

            operator_input_types = eval(operators[operator_name]['input'])
            children_return_types = [return_type(x, operators, datafields) for x in node.children]

            if not check_input(operator_input_types, children_return_types):
                if _debug:
                    print(f"Type mismatch for {operator_name}: expected {operator_input_types}, got {children_return_types}")
                return False, f"Type mismatch in operator: {operator_name}"

        final_type = return_type(exp_tree, operators, datafields)
        if final_type != "MATRIX":
            return False, f"Final output type is {final_type}, not MATRIX"

        return True, None

    except Exception as e:
        return False, f"Parsing error: {str(e)}"


# ==================== Dataset Parsing ====================

def parse_zero_fail_alphas_from_file(filepath: Path) -> List[Dict]:
    """
    Parse dataset file and extract zero-fail alphas

    Returns:
        List of dicts with keys: block_num, id, sharpe, fitness, turnover, expression
    """
    if not filepath.exists():
        print(f"[WARNING] File not found: {filepath}")
        return []

    content = filepath.read_text(encoding="utf-8")
    results = []

    for match in ZERO_FAIL_BLOCK_PATTERN.finditer(content):
        results.append({
            "block_num": int(match.group(1)),
            "id": match.group(2),
            "sharpe": float(match.group(5)),
            "fitness": float(match.group(6)),
            "turnover": float(match.group(7)),
            "expression": match.group(8).strip(),
        })

    return results


def load_zero_fail_alphas(dataset_files: Dict[str, Path]) -> Dict[str, List[Dict]]:
    """
    Load zero-fail alphas from dataset files

    Args:
        dataset_files: Dict mapping dataset name to file path

    Returns:
        Dict mapping dataset name to list of zero-fail alphas
    """
    all_alphas = {}

    for dataset_name, filepath in dataset_files.items():
        alphas = parse_zero_fail_alphas_from_file(filepath)
        all_alphas[dataset_name] = alphas
        print(f"[INFO] {dataset_name}: {len(alphas)} zero-fail alphas loaded")

    return all_alphas


# ==================== Cartesian Product Combination ====================

def generate_all_combinations(
    all_alphas: Dict[str, List[Dict]],
    dataset_names: List[str],
    max_combinations: Optional[int] = None,
    random_seed: Optional[int] = None
) -> List[Dict]:
    """
    Generate Cartesian product of zero-fail seeds from specified datasets

    Args:
        all_alphas: Dict mapping dataset name to list of alphas
        dataset_names: List of dataset names to combine
        max_combinations: Maximum number of combinations (None = unlimited)
        random_seed: Seed for random sampling if max_combinations is set

    Returns:
        List of combination dicts with structure:
        {
            'datasets': ['mdl25', 'mdl30', ...],
            'seeds': [alpha1_dict, alpha2_dict, ...],
            'expressions': [expr1, expr2, ...],
            'metadata': {...}
        }
    """
    if random_seed is not None:
        random.seed(random_seed)

    # Extract seed lists for each dataset
    seed_lists = []
    valid_dataset_names = []

    for dataset_name in dataset_names:
        if dataset_name not in all_alphas:
            print(f"[WARNING] Dataset {dataset_name} not found, skipping")
            continue

        seeds = all_alphas[dataset_name]
        if not seeds:
            print(f"[WARNING] Dataset {dataset_name} has no seeds, skipping")
            continue

        seed_lists.append(seeds)
        valid_dataset_names.append(dataset_name)

    if len(seed_lists) < 2:
        print(f"[ERROR] Need at least 2 datasets with seeds, got {len(seed_lists)}")
        return []

    # Calculate total combinations
    total_combinations = 1
    for seeds in seed_lists:
        total_combinations *= len(seeds)

    print(f"[INFO] Generating Cartesian product from {len(valid_dataset_names)} datasets:")
    for i, dataset_name in enumerate(valid_dataset_names):
        print(f"  - {dataset_name}: {len(seed_lists[i])} seeds")
    print(f"[INFO] Total combinations: {total_combinations:,}")

    # Generate combinations
    all_combinations = []

    if max_combinations is not None and total_combinations > max_combinations:
        print(f"[INFO] Sampling {max_combinations:,} combinations from {total_combinations:,}")

        # Random sampling
        sampled_count = 0
        for combination_tuple in product(*seed_lists):
            if random.random() < (max_combinations / total_combinations):
                all_combinations.append({
                    'datasets': valid_dataset_names,
                    'seeds': list(combination_tuple),
                    'expressions': [seed['expression'] for seed in combination_tuple],
                    'metadata': {
                        'seed_ids': [seed.get('id', 'N/A') for seed in combination_tuple],
                        'sharpes': [seed.get('sharpe', 0) for seed in combination_tuple],
                        'fitnesses': [seed.get('fitness', 0) for seed in combination_tuple],
                    }
                })
                sampled_count += 1
                if sampled_count >= max_combinations:
                    break
    else:
        # Full Cartesian product
        for combination_tuple in product(*seed_lists):
            all_combinations.append({
                'datasets': valid_dataset_names,
                'seeds': list(combination_tuple),
                'expressions': [seed['expression'] for seed in combination_tuple],
                'metadata': {
                    'seed_ids': [seed.get('id', 'N/A') for seed in combination_tuple],
                    'sharpes': [seed.get('sharpe', 0) for seed in combination_tuple],
                    'fitnesses': [seed.get('fitness', 0) for seed in combination_tuple],
                }
            })

    print(f"[INFO] Generated {len(all_combinations):,} seed combinations")
    return all_combinations


# ==================== Variant Generation using operators_list.json ====================

def load_operators(operators_file: Path = Path('operators_list.json')) -> dict:
    """Load operators from operators_list.json"""
    return llm.import_json(str(operators_file))


def generate_combination_variants(
    seed_combination: Dict,
    operators: dict,
    max_variants_per_combo: int = 5,
    random_seed: Optional[int] = None
) -> List[Dict]:
    """
    Generate expression variants from a seed combination using operators

    Args:
        seed_combination: Single combination from generate_all_combinations()
        operators: operators_list.json content
        max_variants_per_combo: Number of variants to generate per combination
        random_seed: Random seed for reproducibility

    Returns:
        List of variant dicts with structure:
        {
            'expression': 'combined_expression',
            'variant_type': 'description',
            'combination_info': original seed_combination,
            'operators_used': [list of operator names]
        }
    """
    if random_seed is not None:
        random.seed(random_seed)

    variants = []
    seed_exprs = seed_combination['expressions']
    num_seeds = len(seed_exprs)

    # Variant 1: Simple addition
    variants.append({
        'expression': 'add(' + ', '.join(seed_exprs) + ')',
        'variant_type': 'simple_add',
        'combination_info': seed_combination,
        'operators_used': ['add']
    })

    # Variant 2: Addition + group normalization
    group_ops = [op for op in COMBINATION_OPERATORS['group'] if op in operators]
    if group_ops:
        group_op = random.choice(group_ops)
        variants.append({
            'expression': f"{group_op}(add({', '.join(seed_exprs)}), subindustry)",
            'variant_type': f'add_with_{group_op}',
            'combination_info': seed_combination,
            'operators_used': ['add', group_op]
        })

    # Variant 3: Weighted combination with normalization
    norm_ops = [op for op in COMBINATION_OPERATORS['normalization'] if op in operators]
    if norm_ops:
        norm_op = random.choice(norm_ops)
        weighted_sum = ' + '.join([f"{norm_op}({expr})" for expr in seed_exprs])
        variants.append({
            'expression': f"scale({weighted_sum})",
            'variant_type': f'{norm_op}_weighted_sum',
            'combination_info': seed_combination,
            'operators_used': [norm_op, 'add', 'scale']
        })

    # Variant 4: Min/Max ensemble
    if 'min' in operators and 'max' in operators:
        variants.append({
            'expression': f"add(min({', '.join(seed_exprs)}), max({', '.join(seed_exprs)}))",
            'variant_type': 'min_max_ensemble',
            'combination_info': seed_combination,
            'operators_used': ['min', 'max', 'add']
        })

    # Variant 5: Tail-clipped combination
    tail_ops = [op for op in COMBINATION_OPERATORS['tail_handling'] if op in operators]
    if tail_ops and norm_ops:
        tail_op = random.choice(tail_ops)
        norm_op = random.choice(norm_ops)
        base_expr = 'add(' + ', '.join(seed_exprs) + ')'
        variants.append({
            'expression': f"{tail_op}({norm_op}({base_expr}))",
            'variant_type': f'{norm_op}_with_{tail_op}',
            'combination_info': seed_combination,
            'operators_used': ['add', norm_op, tail_op]
        })

    # Limit to max_variants_per_combo
    return variants[:max_variants_per_combo]


# ==================== Main Pipeline ====================

def run_combinatorial_pipeline(
    dataset_files: Dict[str, Path],
    dataset_names: List[str],
    operators_file: Path,
    datafields_file: Path,
    output_dir: Path,
    max_combinations: Optional[int] = None,
    max_variants_per_combo: int = 5,
    concurrency: int = 3,
    random_seed: Optional[int] = None,
    check_sanity: bool = True
) -> Dict:
    """
    Run the full combinatorial alpha pipeline

    Args:
        dataset_files: Dict of dataset_name -> file_path
        dataset_names: List of datasets to combine
        operators_file: Path to operators_list.json
        datafields_file: Path to datafields JSON
        output_dir: Directory for output files
        max_combinations: Maximum seed combinations (None = all)
        max_variants_per_combo: Variants per seed combination
        concurrency: Number of concurrent simulations
        random_seed: Random seed for reproducibility
        check_sanity: Whether to run sanity check

    Returns:
        Summary dict with pipeline statistics
    """
    print("=" * 80)
    print("COMBINATORIAL ALPHA PIPELINE")
    print("=" * 80)

    # Load resources
    print("\n[Step 1] Loading resources...")
    all_alphas = load_zero_fail_alphas(dataset_files)
    operators = load_operators(operators_file)
    datafields = llm.import_json(str(datafields_file))

    # Generate combinations
    print("\n[Step 2] Generating seed combinations...")
    seed_combinations = generate_all_combinations(
        all_alphas,
        dataset_names,
        max_combinations=max_combinations,
        random_seed=random_seed
    )

    if not seed_combinations:
        print("[ERROR] No combinations generated, exiting")
        return {'status': 'failed', 'reason': 'no_combinations'}

    # Generate variants
    print("\n[Step 3] Generating expression variants...")
    all_variants = []
    for i, seed_combo in enumerate(seed_combinations):
        variants = generate_combination_variants(
            seed_combo,
            operators,
            max_variants_per_combo=max_variants_per_combo,
            random_seed=random_seed
        )
        all_variants.extend(variants)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(seed_combinations)} combinations...")

    print(f"[INFO] Generated {len(all_variants):,} total variants")

    # Sanity check
    if check_sanity:
        print("\n[Step 4] Running sanity check...")
        valid_variants = []
        sanity_failures = {}

        for variant in all_variants:
            is_valid, error_msg = sanity_checker(
                variant['expression'],
                operators,
                datafields
            )

            if is_valid:
                valid_variants.append(variant)
            else:
                reason = error_msg or "unknown"
                sanity_failures[reason] = sanity_failures.get(reason, 0) + 1

        print(f"[INFO] Sanity check results:")
        print(f"  - Valid: {len(valid_variants):,} / {len(all_variants):,}")
        print(f"  - Failed: {len(all_variants) - len(valid_variants):,}")

        if sanity_failures:
            print(f"\n  Failure breakdown:")
            for reason, count in sorted(sanity_failures.items(), key=lambda x: -x[1])[:10]:
                print(f"    - {reason}: {count}")

        all_variants = valid_variants

    # Save results
    print("\n[Step 5] Saving results...")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_combo_name = '_'.join(sorted(dataset_names))
    output_file = output_dir / f"{dataset_combo_name}_comb.txt"
    json_file = output_dir / f"{dataset_combo_name}_comb.json"

    # Save text format
    save_variants_to_txt(all_variants, output_file, dataset_names)

    # Save JSON format
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'datasets': dataset_names,
            'total_seed_combinations': len(seed_combinations),
            'total_variants': len(all_variants),
            'variants': all_variants
        }, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Saved results to:")
    print(f"  - {output_file}")
    print(f"  - {json_file}")

    summary = {
        'status': 'success',
        'datasets': dataset_names,
        'seed_combinations': len(seed_combinations),
        'total_variants_generated': len(all_variants) if check_sanity else len(all_variants),
        'valid_variants': len(all_variants),
        'output_files': {
            'txt': str(output_file),
            'json': str(json_file)
        }
    }

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)

    return summary


def save_variants_to_txt(variants: List[Dict], filepath: Path, dataset_names: List[str]):
    """Save variants to human-readable text file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"COMBINATORIAL ALPHAS: {' x '.join(dataset_names)}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total variants: {len(variants)}\n")
        f.write("=" * 80 + "\n\n")

        for i, variant in enumerate(variants, 1):
            combo_info = variant.get('combination_info', {})
            metadata = combo_info.get('metadata', {})

            f.write(f"--- #{i} | {variant.get('variant_type', 'N/A')} ---\n")
            f.write(f"Expression: {variant['expression']}\n")
            f.write(f"Operators Used: {', '.join(variant.get('operators_used', []))}\n")
            f.write(f"Source Datasets: {', '.join(combo_info.get('datasets', []))}\n")
            f.write(f"Seed IDs: {', '.join(metadata.get('seed_ids', []))}\n")
            f.write(f"Source Sharpes: {', '.join(map(str, metadata.get('sharpes', [])))}\n")
            f.write(f"Source Fitnesses: {', '.join(map(str, metadata.get('fitnesses', [])))}\n")
            f.write("\n")

        f.write("=" * 80 + "\n")
        f.write(f"Total: {len(variants)} expression variants\n")
        f.write("=" * 80 + "\n")
