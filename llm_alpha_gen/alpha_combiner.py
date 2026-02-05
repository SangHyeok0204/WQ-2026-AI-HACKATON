"""
Alpha Combiner Pipeline
========================
fail_alpha_categorized.txt에서 실패한 알파들을 분석하여
서로 다른 fail 카테고리의 알파를 조합 → Brain API 시뮬레이션 → good_alpha_list.json 저장

디렉토리 구조:
    llm_alpha_gen/
    ├── alpha_combiner.py          (이 파일)
    ├── fail_alpha_categorized.txt (입력: fail 알파 목록)
    ├── good_alpha_list.json       (출력: 성공 알파 저장)
    ├── combine_logs/              (출력: 배치별 로그)
    ├── ace_lib.py                 (Brain API 래퍼)
    ├── parser.py                  (FASTEXPR 파서)
    └── operators_list.json        (연산자 정의)

실행 방법:
    cd llm_alpha_gen
    python alpha_combiner.py

또는 Jupyter에서:
    from alpha_combiner import run_pipeline
    run_pipeline(total_batches=5)
"""

import json
import os
import random
import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from functools import partial
from multiprocessing.pool import ThreadPool

import ace_lib as ace
from parser import tree_node

# pandas import (결과 처리용)
try:
    import pandas as pd
except ImportError:
    pd = None

# ============================================================================
# 설정 상수
# ============================================================================
REGION = "EUR"
UNIVERSE = "TOP2500"
DELAY = 1

# 시뮬레이션 기준값
CUT_SHARPE = 1.58
CUT_FITNESS = 1.0
CUT_TURNOVER_LOW = 0.01
CUT_TURNOVER_HIGH = 0.7
CUT_SUB_SHARPE = 1.23

# 배치 설정
ALPHAS_PER_BATCH = 10
TOP_ALPHAS_TO_SAVE = 3
TOTAL_BATCHES = 5

# 파일 경로 (스크립트 위치 기준)
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Jupyter notebook에서 실행 시
    SCRIPT_DIR = os.getcwd()

FAIL_ALPHA_FILE = os.path.join(SCRIPT_DIR, "fail_alpha_categorized.txt")
GOOD_ALPHA_FILE = os.path.join(SCRIPT_DIR, "good_alpha_list.json")
LOG_DIR = os.path.join(SCRIPT_DIR, "combine_logs")

# ============================================================================
# 유틸리티 함수
# ============================================================================

def load_json(path: str) -> list:
    """JSON 파일 로드"""
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data: list, path: str):
    """JSON 파일 저장"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def append_to_json(new_items: list, path: str):
    """JSON 파일에 항목 추가"""
    existing = load_json(path)
    existing.extend(new_items)
    save_json(existing, path)

def log_batch_result(batch_num: int, results: list, log_dir: str):
    """배치 결과를 로그 파일로 저장"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"batch_{batch_num}_{timestamp}.json")
    save_json(results, log_path)
    print(f"[LOG] Batch {batch_num} results saved to {log_path}")

# ============================================================================
# Step 1: fail_alpha_categorized.txt 파싱
# ============================================================================

def parse_fail_alpha_file(filepath: str) -> Dict[str, List[Dict]]:
    """
    fail_alpha_categorized.txt를 파싱하여 카테고리별 알파 목록 반환

    Returns:
        {
            "LOW_SHARPE": [{"expression": "...", "sharpe": 0.75, ...}, ...],
            "LOW_FITNESS": [...],
            ...
        }
    """
    categories = defaultdict(list)
    current_category = None

    if not os.path.exists(filepath):
        print(f"[ERROR] File not found: {filepath}")
        return categories

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        raw_line = lines[i]
        line = raw_line.strip()

        # 빈 줄이나 구분선 스킵
        if not line or line.startswith("===") or line.startswith("---"):
            i += 1
            continue

        # 카테고리 헤더 감지: "Fail 사유: CATEGORY_NAME"
        if line.startswith("Fail 사유:"):
            match = re.search(r'Fail 사유:\s*(\S+)', line)
            if match:
                current_category = match.group(1)
            i += 1
            continue

        # 스펙 라인 스킵 (-> 로 시작)
        if line.startswith("->"):
            i += 1
            continue

        # 알파 표현식 라인 감지
        # 조건: 현재 카테고리가 있고, 라인이 공백으로 시작하고, 알파 표현식처럼 보이는 경우
        if current_category and raw_line.startswith("  ") and not line.startswith("->"):
            # 알파 표현식인지 확인 (괄호가 있거나 알파벳으로 시작)
            if "(" in line or line[0].isalpha():
                expression = line

                # 다음 줄에서 스펙 정보 파싱
                specs = {}
                if i + 1 < len(lines):
                    spec_line = lines[i + 1].strip()
                    if spec_line.startswith("->"):
                        # 스펙 파싱: "-> Sharpe: 0.75, Fitness: 0.54, Turnover: 0.0917"
                        spec_parts = spec_line[2:].strip()
                        for part in spec_parts.split(","):
                            part = part.strip()
                            if ":" in part:
                                key, val = part.split(":", 1)
                                key = key.strip().lower().replace(" ", "_").replace("-", "_")
                                val = val.strip()
                                # 괄호 안 내용 제거 (예: "(IS Sharpe: 2.51)")
                                if "(" in val:
                                    val = val.split("(")[0].strip()
                                try:
                                    specs[key] = float(val) if val != "N/A" else None
                                except:
                                    specs[key] = val
                        i += 1  # 스펙 라인 스킵

                categories[current_category].append({
                    "expression": expression,
                    "category": current_category,
                    **specs
                })

        i += 1

    # 요약 출력
    print("\n[PARSE] Fail alpha categories loaded:")
    for cat, alphas in categories.items():
        print(f"  - {cat}: {len(alphas)} alphas")

    if not categories:
        print("[WARNING] No categories found! Check file format.")

    return dict(categories)

# ============================================================================
# Step 2: 알파 조합 생성
# ============================================================================

# 조합 방식 정의 (다양한 템플릿)
COMBINATION_TEMPLATES = [
    # ===== 기본 조합 =====
    ("add", "add({a1}, {a2})"),
    ("subtract", "subtract({a1}, {a2})"),
    ("multiply", "multiply({a1}, {a2})"),

    # ===== 가중 조합 =====
    ("weighted_add_60_40", "add(multiply({a1}, 0.6), multiply({a2}, 0.4))"),
    ("weighted_add_70_30", "add(multiply({a1}, 0.7), multiply({a2}, 0.3))"),

    # ===== 시계열 결합 =====
    ("ts_zscore_add", "add(ts_zscore({a1}, 63), ts_zscore({a2}, 63))"),
    ("ts_rank_add", "add(ts_rank({a1}, 21), ts_rank({a2}, 21))"),
    ("ts_ir_add", "add(ts_ir({a1}, 63), ts_ir({a2}, 63))"),

    # ===== Decay 결합 =====
    ("decay_add", "ts_decay_linear(add({a1}, {a2}), 21)"),
    ("decay_weighted", "ts_decay_linear(add(multiply({a1}, 0.6), multiply({a2}, 0.4)), 42)"),

    # ===== Delta/Momentum =====
    ("delta_combine", "add(ts_delta({a1}, 21), ts_delta({a2}, 21))"),
    ("momentum_zscore", "ts_zscore(add({a1}, {a2}), 63)"),

    # ===== 평균/스무딩 =====
    ("mean_combine", "ts_mean(add({a1}, {a2}), 21)"),
    ("ewma_combine", "ts_decay_exp_window(add({a1}, {a2}), 21, factor=0.5)"),
]

# 최종 래퍼 (concentrated weight / zero coverage 방지)
# rank()를 제외하고 다양한 래퍼 사용 (rank 과다 사용 방지)
FINAL_WRAPPERS = [
    "normalize({expr})",
    "zscore({expr})",
    "quantile({expr})",
    "signed_power({expr}, 0.5)",
    "winsorize({expr}, std=3)",
]

def select_alpha_pair(categories: Dict[str, List[Dict]]) -> Tuple[Dict, Dict, str, str]:
    """
    서로 다른 fail 카테고리에서 알파 2개 선택

    Returns:
        (alpha1_info, alpha2_info, category1, category2)
    """
    available_cats = [cat for cat, alphas in categories.items() if len(alphas) > 0]

    if len(available_cats) < 2:
        # 같은 카테고리에서 2개 선택
        cat1 = cat2 = random.choice(available_cats)
        alphas = categories[cat1]
        if len(alphas) >= 2:
            a1, a2 = random.sample(alphas, 2)
        else:
            a1 = a2 = alphas[0]
    else:
        # 서로 다른 카테고리에서 1개씩 선택
        cat1, cat2 = random.sample(available_cats, 2)
        a1 = random.choice(categories[cat1])
        a2 = random.choice(categories[cat2])

    return a1, a2, cat1, cat2

def generate_combined_alpha(alpha1: Dict, alpha2: Dict, template_name: str = None) -> str:
    """
    두 알파를 조합하여 새 알파 생성
    """
    expr1 = alpha1["expression"]
    expr2 = alpha2["expression"]

    # 템플릿 선택
    if template_name:
        template = next((t for n, t in COMBINATION_TEMPLATES if n == template_name), None)
    else:
        _, template = random.choice(COMBINATION_TEMPLATES)

    # 조합 생성
    combined = template.format(a1=expr1, a2=expr2)

    return combined

def apply_final_wrapper(expression: str) -> str:
    """
    제출 안정성을 위한 최종 래퍼 적용
    """
    wrapper = random.choice(FINAL_WRAPPERS)
    return wrapper.format(expr=expression)

def generate_combination_batch(
    categories: Dict[str, List[Dict]],
    count: int = ALPHAS_PER_BATCH
) -> List[Dict]:
    """
    count개의 조합 알파 후보 생성
    """
    candidates = []

    for i in range(count):
        # 알파 페어 선택
        alpha1, alpha2, cat1, cat2 = select_alpha_pair(categories)

        # 조합 방식 선택 (다양성 확보)
        template_idx = i % len(COMBINATION_TEMPLATES)
        template_name, _ = COMBINATION_TEMPLATES[template_idx]

        # 조합 생성
        raw_combined = generate_combined_alpha(alpha1, alpha2, template_name)

        # 최종 래퍼 적용
        final_expr = apply_final_wrapper(raw_combined)

        candidates.append({
            "expression": final_expr,
            "raw_expression": raw_combined,
            "parent1": alpha1["expression"],
            "parent2": alpha2["expression"],
            "parent1_category": cat1,
            "parent2_category": cat2,
            "combination_type": template_name,
        })

    return candidates

# ============================================================================
# Step 3: 프리체크 (시뮬레이션 전 필터링)
# ============================================================================

# 위험 패턴 정의
HIGH_TURNOVER_PATTERNS = [
    r"ts_arg_max.*if_else",
    r"if_else.*ts_arg_max",
    r"equal\s*\(\s*ts_arg_max",
    r"ts_arg_min.*if_else",
]

INVALID_PATTERNS = [
    r"bucket\s*=",  # bucket 인자 misuse
    r"\(\s*\)",     # 빈 괄호
    r",,",          # 연속 콤마
]

def precheck_syntax(expression: str) -> Tuple[bool, str]:
    """
    문법 오류 사전 체크

    Returns:
        (통과 여부, 실패 사유)
    """
    # 기본 괄호 균형 체크
    open_count = expression.count("(")
    close_count = expression.count(")")
    if open_count != close_count:
        return False, "Unbalanced parentheses"

    # 위험 패턴 체크
    for pattern in INVALID_PATTERNS:
        if re.search(pattern, expression):
            return False, f"Invalid pattern: {pattern}"

    return True, ""

def precheck_high_turnover(expression: str) -> Tuple[bool, str]:
    """
    High turnover 위험 패턴 체크
    """
    for pattern in HIGH_TURNOVER_PATTERNS:
        if re.search(pattern, expression, re.IGNORECASE):
            return False, f"High turnover pattern: {pattern}"
    return True, ""

def precheck_parser(expression: str) -> Tuple[bool, str]:
    """
    parser.py를 이용한 구문 분석 체크
    """
    try:
        node = tree_node(expression)
        if node is None:
            return False, "Parser returned None"
        return True, ""
    except Exception as e:
        return False, f"Parser error: {str(e)}"

def precheck_alpha(candidate: Dict) -> Tuple[bool, str]:
    """
    알파 후보에 대한 종합 프리체크

    Returns:
        (통과 여부, 실패 사유)
    """
    expr = candidate["expression"]

    # 1. 문법 체크
    ok, reason = precheck_syntax(expr)
    if not ok:
        return False, reason

    # 2. High turnover 패턴 체크
    ok, reason = precheck_high_turnover(expr)
    if not ok:
        return False, reason

    # 3. Parser 체크
    ok, reason = precheck_parser(expr)
    if not ok:
        return False, reason

    return True, ""

def filter_by_precheck(candidates: List[Dict]) -> List[Dict]:
    """
    프리체크 통과한 후보만 필터링
    """
    passed = []
    for c in candidates:
        ok, reason = precheck_alpha(c)
        if ok:
            passed.append(c)
            print(f"  [PASS] {c['expression'][:60]}...")
        else:
            print(f"  [FAIL] {reason}: {c['expression'][:40]}...")

    print(f"[PRECHECK] {len(passed)}/{len(candidates)} passed")
    return passed

# ============================================================================
# Step 4: Brain API 시뮬레이션
# ============================================================================

def simulate_single_with_check(session, simulate_data: dict, candidate: dict) -> dict:
    """
    단일 알파 시뮬레이션 및 체크 결과 수집 (ace_lib 패턴 따름)
    """
    try:
        # 세션 체크
        session = ace.check_session_and_relogin(session)

        # 시뮬레이션 실행
        simulate_response = ace.start_simulation(session, simulate_data)
        simulation_result = ace.simulation_progress(session, simulate_response)

        if not simulation_result["completed"]:
            return {
                **candidate,
                "alpha_id": None,
                "simulation_success": False,
                "error": "Simulation not completed",
            }

        alpha_id = simulation_result["result"]["id"]

        # 알파 속성 설정
        ace.set_alpha_properties(session, alpha_id, tags=["combiner_tag"])

        # 체크 결과 가져오기
        result_json = ace.get_simulation_result_json(session, alpha_id)

        # is 섹션에서 메트릭 추출
        is_data = result_json.get("is", {})
        checks = is_data.get("checks", [])

        return {
            **candidate,
            "alpha_id": alpha_id,
            "simulation_success": True,
            "is_sharpe": is_data.get("sharpe"),
            "is_fitness": is_data.get("fitness"),
            "is_turnover": is_data.get("turnover"),
            "is_margin": is_data.get("margin"),
            "checks": checks,
        }

    except Exception as e:
        return {
            **candidate,
            "alpha_id": None,
            "simulation_success": False,
            "error": str(e),
        }

def simulate_alphas(session, candidates: List[Dict]) -> List[Dict]:
    """
    Brain API로 알파 시뮬레이션 수행 (병렬 처리)
    """
    if not candidates:
        return []

    print(f"\n[SIMULATE] Running {len(candidates)} alphas...")

    # generate_alpha 형식으로 변환
    simulate_data_list = [
        ace.generate_alpha(
            regular=c["expression"],
            region=REGION,
            universe=UNIVERSE,
            delay=DELAY,
            neutralization="INDUSTRY",
        )
        for c in candidates
    ]

    results = []

    # 8개 단위로 배치 처리
    for batch_start in range(0, len(candidates), 8):
        batch_end = min(batch_start + 8, len(candidates))
        batch_data = simulate_data_list[batch_start:batch_end]
        batch_candidates = candidates[batch_start:batch_end]

        print(f"  Simulating batch {batch_start//8 + 1}: {batch_start} to {batch_end}")

        # 세션 체크
        if ace.check_session_timeout(session) < 1000:
            print("  [SESSION] Re-authenticating...")
            session = ace.start_session()

        # ThreadPool로 병렬 실행
        try:
            with ThreadPool(min(8, len(batch_data))) as pool:
                batch_results = pool.starmap(
                    simulate_single_with_check,
                    [(session, data, cand) for data, cand in zip(batch_data, batch_candidates)]
                )
                results.extend(batch_results)

        except Exception as e:
            print(f"  [ERROR] Batch simulation failed: {e}")
            # 개별 재시도
            for data, cand in zip(batch_data, batch_candidates):
                try:
                    result = simulate_single_with_check(session, data, cand)
                    results.append(result)
                except Exception as inner_e:
                    results.append({
                        **cand,
                        "alpha_id": None,
                        "simulation_success": False,
                        "error": str(inner_e),
                    })

        # 배치 간 휴식 (API rate limit 방지)
        time.sleep(2)

    # 결과 요약 출력
    success_count = sum(1 for r in results if r.get("simulation_success"))
    print(f"  [SIMULATE] Completed: {success_count}/{len(candidates)} successful")

    return results

# ============================================================================
# Step 5: 결과 선별 및 저장
# ============================================================================

def evaluate_alpha(result: Dict) -> Tuple[bool, float, List[str]]:
    """
    알파 결과 평가

    Returns:
        (통과 여부, 점수, fail 사유 리스트)
    """
    if not result.get("simulation_success"):
        return False, 0, ["Simulation failed"]

    fail_reasons = []

    sharpe = result.get("is_sharpe", 0) or 0
    fitness = result.get("is_fitness", 0) or 0
    turnover = result.get("is_turnover", 0) or 0

    # 기본 컷 체크
    if sharpe < CUT_SHARPE:
        fail_reasons.append(f"LOW_SHARPE ({sharpe:.2f} < {CUT_SHARPE})")
    if fitness < CUT_FITNESS:
        fail_reasons.append(f"LOW_FITNESS ({fitness:.2f} < {CUT_FITNESS})")
    if turnover < CUT_TURNOVER_LOW:
        fail_reasons.append(f"LOW_TURNOVER ({turnover:.4f} < {CUT_TURNOVER_LOW})")
    if turnover > CUT_TURNOVER_HIGH:
        fail_reasons.append(f"HIGH_TURNOVER ({turnover:.4f} > {CUT_TURNOVER_HIGH})")

    # checks에서 fail 항목 확인
    checks = result.get("checks", [])
    for check in checks:
        if check.get("result") == "FAIL":
            check_name = check.get("name", "UNKNOWN")
            if check_name not in [r.split("(")[0] for r in fail_reasons]:
                fail_reasons.append(check_name)

    # 점수 계산 (Sharpe + Fitness 가중)
    score = sharpe * 0.6 + fitness * 0.4

    passed = len(fail_reasons) == 0
    return passed, score, fail_reasons

def select_top_alphas(results: List[Dict], top_n: int = TOP_ALPHAS_TO_SAVE) -> List[Dict]:
    """
    상위 N개 알파 선택
    """
    # 평가 및 점수 계산
    evaluated = []
    for r in results:
        passed, score, fail_reasons = evaluate_alpha(r)
        evaluated.append({
            **r,
            "evaluation_passed": passed,
            "evaluation_score": score,
            "evaluation_fail_reasons": fail_reasons,
        })

    # 통과한 것 중 상위 선택
    passed_alphas = [e for e in evaluated if e["evaluation_passed"]]

    if passed_alphas:
        # 점수순 정렬
        passed_alphas.sort(key=lambda x: x["evaluation_score"], reverse=True)
        selected = passed_alphas[:top_n]
        print(f"\n[SELECT] {len(passed_alphas)} passed, selecting top {len(selected)}")
    else:
        # 통과한 것이 없으면 점수 높은 순으로 선택
        evaluated.sort(key=lambda x: x["evaluation_score"], reverse=True)
        selected = evaluated[:top_n]
        print(f"\n[SELECT] 0 passed, selecting top {len(selected)} by score")

    for s in selected:
        print(f"  - Score: {s['evaluation_score']:.2f}, "
              f"Sharpe: {s.get('is_sharpe', 'N/A')}, "
              f"Fitness: {s.get('is_fitness', 'N/A')}")

    return selected

def format_for_good_alpha_list(alpha: Dict) -> Dict:
    """
    good_alpha_list.json 형식으로 변환
    """
    return {
        "timestamp": datetime.now().isoformat(),
        "alpha_id": alpha.get("alpha_id"),
        "expression": alpha.get("expression"),
        "is_sharpe": alpha.get("is_sharpe"),
        "is_fitness": alpha.get("is_fitness"),
        "is_turnover": alpha.get("is_turnover"),
        "settings": {
            "region": REGION,
            "universe": UNIVERSE,
            "delay": DELAY,
            "decay": 0,
            "truncation": 0.08,
            "neutralization": "INDUSTRY",
            "pasteurization": "ON",
            "nanHandling": "OFF",
            "unitHandling": "VERIFY",
            "language": "FASTEXPR",
            "testPeriod": "P0Y0M0D",
            "visualization": False,
        },
        "checks": alpha.get("checks", []),
        "combination_info": {
            "parent1": alpha.get("parent1"),
            "parent2": alpha.get("parent2"),
            "parent1_category": alpha.get("parent1_category"),
            "parent2_category": alpha.get("parent2_category"),
            "combination_type": alpha.get("combination_type"),
        },
        "source": "alpha_combiner",
    }

# ============================================================================
# Step 6: 배치 간 학습 (이전 결과 참고)
# ============================================================================

def load_previous_batch_results(log_dir: str) -> List[Dict]:
    """
    이전 배치 결과 로드 (학습용)
    """
    results = []
    if not os.path.exists(log_dir):
        return results

    for filename in sorted(os.listdir(log_dir)):
        if filename.endswith(".json"):
            filepath = os.path.join(log_dir, filename)
            try:
                batch_data = load_json(filepath)
                results.extend(batch_data)
            except:
                pass

    return results

def analyze_previous_results(results: List[Dict]) -> Dict:
    """
    이전 결과 분석하여 성공적인 조합 패턴 추출
    """
    analysis = {
        "successful_templates": defaultdict(int),
        "successful_category_pairs": defaultdict(int),
        "failed_expressions": set(),
        "best_sharpe": 0,
        "avg_sharpe": 0,
    }

    successful = [r for r in results if r.get("simulation_success") and r.get("is_sharpe", 0) >= CUT_SHARPE]

    if successful:
        for r in successful:
            template = r.get("combination_type", "unknown")
            analysis["successful_templates"][template] += 1

            cat_pair = (r.get("parent1_category", ""), r.get("parent2_category", ""))
            analysis["successful_category_pairs"][cat_pair] += 1

        sharpes = [r.get("is_sharpe", 0) for r in successful]
        analysis["best_sharpe"] = max(sharpes)
        analysis["avg_sharpe"] = sum(sharpes) / len(sharpes)

    # 실패한 expression 기록 (중복 방지)
    failed = [r for r in results if not r.get("simulation_success")]
    for r in failed:
        analysis["failed_expressions"].add(r.get("expression", ""))

    return analysis

def generate_combination_batch_smart(
    categories: Dict[str, List[Dict]],
    count: int,
    previous_analysis: Dict = None
) -> List[Dict]:
    """
    이전 결과를 참고하여 스마트하게 조합 생성
    """
    candidates = []

    # 성공적인 템플릿 가중치 적용
    template_weights = []
    for name, template in COMBINATION_TEMPLATES:
        base_weight = 1.0
        if previous_analysis and name in previous_analysis.get("successful_templates", {}):
            # 성공한 템플릿에 가중치 부여
            base_weight += previous_analysis["successful_templates"][name] * 0.5
        template_weights.append((name, template, base_weight))

    total_weight = sum(w for _, _, w in template_weights)
    normalized_weights = [(n, t, w/total_weight) for n, t, w in template_weights]

    failed_exprs = previous_analysis.get("failed_expressions", set()) if previous_analysis else set()

    attempts = 0
    max_attempts = count * 3  # 재시도 한도

    while len(candidates) < count and attempts < max_attempts:
        attempts += 1

        # 알파 페어 선택
        alpha1, alpha2, cat1, cat2 = select_alpha_pair(categories)

        # 가중치 기반 템플릿 선택
        rand_val = random.random()
        cumulative = 0
        selected_template = COMBINATION_TEMPLATES[0]
        for name, template, weight in normalized_weights:
            cumulative += weight
            if rand_val <= cumulative:
                selected_template = (name, template)
                break

        template_name, template_str = selected_template

        # 조합 생성
        raw_combined = template_str.format(a1=alpha1["expression"], a2=alpha2["expression"])

        # 최종 래퍼 적용
        final_expr = apply_final_wrapper(raw_combined)

        # 중복/실패 expression 체크
        if final_expr in failed_exprs:
            continue

        # 이미 생성된 것과 중복 체크
        if any(c["expression"] == final_expr for c in candidates):
            continue

        candidates.append({
            "expression": final_expr,
            "raw_expression": raw_combined,
            "parent1": alpha1["expression"],
            "parent2": alpha2["expression"],
            "parent1_category": cat1,
            "parent2_category": cat2,
            "combination_type": template_name,
        })

    return candidates

# ============================================================================
# Step 7: 메인 배치 파이프라인
# ============================================================================

def run_single_batch(
    session,
    categories: Dict[str, List[Dict]],
    batch_num: int,
    previous_analysis: Dict = None
) -> Tuple[List[Dict], Dict]:
    """
    단일 배치 실행

    Returns:
        (저장된 알파 리스트, 배치 결과 분석)
    """
    print(f"\n{'='*70}")
    print(f"  BATCH {batch_num} START")
    print(f"{'='*70}")

    # 1. 알파 조합 생성 (이전 결과 참고)
    print("\n[STEP 1] Generating alpha combinations...")
    if previous_analysis:
        print(f"  Using insights from previous batches (best sharpe: {previous_analysis.get('best_sharpe', 0):.2f})")
        candidates = generate_combination_batch_smart(categories, ALPHAS_PER_BATCH, previous_analysis)
    else:
        candidates = generate_combination_batch(categories, ALPHAS_PER_BATCH)
    print(f"Generated {len(candidates)} candidates")

    # 2. 프리체크
    print("\n[STEP 2] Running precheck...")
    valid_candidates = filter_by_precheck(candidates)

    if not valid_candidates:
        print("[WARNING] No candidates passed precheck")
        return [], {}

    # 3. 시뮬레이션
    print("\n[STEP 3] Running Brain API simulation...")
    results = simulate_alphas(session, valid_candidates)

    # 4. 결과 선별
    print("\n[STEP 4] Selecting top alphas...")
    top_alphas = select_top_alphas(results, TOP_ALPHAS_TO_SAVE)

    # 5. 로그 저장
    log_batch_result(batch_num, results, LOG_DIR)

    # 6. good_alpha_list.json에 저장
    if top_alphas:
        formatted = [format_for_good_alpha_list(a) for a in top_alphas]
        append_to_json(formatted, GOOD_ALPHA_FILE)
        print(f"\n[SAVE] Added {len(formatted)} alphas to {GOOD_ALPHA_FILE}")

    # 7. 이 배치 결과 분석 (다음 배치용)
    batch_analysis = analyze_previous_results(results)

    print(f"\n{'='*70}")
    print(f"  BATCH {batch_num} COMPLETE")
    print(f"  - Candidates: {len(candidates)}")
    print(f"  - Precheck passed: {len(valid_candidates)}")
    print(f"  - Simulated: {len(results)}")
    print(f"  - Saved: {len(top_alphas)}")
    if batch_analysis.get("best_sharpe", 0) > 0:
        print(f"  - Best Sharpe this batch: {batch_analysis['best_sharpe']:.2f}")
    print(f"{'='*70}")

    return top_alphas, batch_analysis

def merge_analysis(existing: Dict, new: Dict) -> Dict:
    """
    분석 결과 병합
    """
    if not existing:
        return new
    if not new:
        return existing

    merged = {
        "successful_templates": defaultdict(int),
        "successful_category_pairs": defaultdict(int),
        "failed_expressions": set(),
        "best_sharpe": max(existing.get("best_sharpe", 0), new.get("best_sharpe", 0)),
        "avg_sharpe": 0,
    }

    # 템플릿 카운트 병합
    for template, count in existing.get("successful_templates", {}).items():
        merged["successful_templates"][template] += count
    for template, count in new.get("successful_templates", {}).items():
        merged["successful_templates"][template] += count

    # 카테고리 페어 카운트 병합
    for pair, count in existing.get("successful_category_pairs", {}).items():
        merged["successful_category_pairs"][pair] += count
    for pair, count in new.get("successful_category_pairs", {}).items():
        merged["successful_category_pairs"][pair] += count

    # 실패 expression 병합
    merged["failed_expressions"] = existing.get("failed_expressions", set()) | new.get("failed_expressions", set())

    return merged

def run_pipeline(total_batches: int = TOTAL_BATCHES):
    """
    전체 파이프라인 실행 (total_batches회 반복)

    각 배치는 이전 배치의 결과를 참고하여 더 나은 조합을 생성합니다.
    """
    print(f"\n{'#'*70}")
    print(f"#  ALPHA COMBINER PIPELINE")
    print(f"#  Region: {REGION}, Universe: {UNIVERSE}")
    print(f"#  Batches: {total_batches}, Alphas/batch: {ALPHAS_PER_BATCH}")
    print(f"{'#'*70}")

    # 1. fail_alpha_categorized.txt 파싱
    print("\n[INIT] Parsing fail alpha file...")
    categories = parse_fail_alpha_file(FAIL_ALPHA_FILE)

    if not categories:
        print("[ERROR] No fail alphas found. Exiting.")
        return

    # 2. 이전 로그에서 학습 (있을 경우)
    print("\n[INIT] Loading previous batch results...")
    previous_results = load_previous_batch_results(LOG_DIR)
    cumulative_analysis = analyze_previous_results(previous_results) if previous_results else None
    if previous_results:
        print(f"  Loaded {len(previous_results)} previous results")
    else:
        print("  No previous results found")

    # 3. Brain 세션 시작
    print("\n[INIT] Starting Brain session...")
    session = ace.start_session()

    # 4. 배치 반복 실행
    all_saved = []
    for batch_num in range(1, total_batches + 1):
        try:
            saved, batch_analysis = run_single_batch(
                session, categories, batch_num, cumulative_analysis
            )
            all_saved.extend(saved)

            # 분석 결과 누적
            cumulative_analysis = merge_analysis(cumulative_analysis, batch_analysis)

            # 세션 갱신
            if ace.check_session_timeout(session) < 2000:
                print("\n[SESSION] Refreshing session...")
                session = ace.start_session()

            # 배치 간 휴식
            if batch_num < total_batches:
                print(f"\n[WAIT] Sleeping 5 seconds before next batch...")
                time.sleep(5)

        except Exception as e:
            print(f"\n[ERROR] Batch {batch_num} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 5. 최종 요약
    print(f"\n{'#'*70}")
    print(f"#  PIPELINE COMPLETE")
    print(f"#  Total batches: {total_batches}")
    print(f"#  Total alphas saved: {len(all_saved)}")
    if cumulative_analysis:
        print(f"#  Best Sharpe achieved: {cumulative_analysis.get('best_sharpe', 0):.2f}")
        if cumulative_analysis.get("successful_templates"):
            top_template = max(cumulative_analysis["successful_templates"].items(), key=lambda x: x[1])
            print(f"#  Most successful template: {top_template[0]} ({top_template[1]} successes)")
    print(f"{'#'*70}")

    return all_saved

# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    run_pipeline(TOTAL_BATCHES)
