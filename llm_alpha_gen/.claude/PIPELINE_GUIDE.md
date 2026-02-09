# WorldQuant Brain + LLM 알파 생성 파이프라인 가이드

## 목차
1. [개요](#개요)
2. [A. Seed-Alpha 생성 단계](#a-seed-alpha-생성-단계)
3. [B. Sanity Check 단계](#b-sanity-check-단계)
4. [C. Brain Simulation 단계](#c-brain-simulation-단계)
5. [D. 제출 가능성 판단 단계](#d-제출-가능성-판단-단계)
6. [E. 사람이 눈으로 확인하는 포인트](#e-사람이-눈으로-확인하는-포인트)
7. [F. 진화 알고리즘 기반 알파 개선 (refine.ipynb)](#f-진화-알고리즘-기반-알파-개선-refineipynb)
8. [G. Cross-Dataset Alpha Combination (combine_and_simulate.ipynb)](#g-cross-dataset-alpha-combination-combine_and_simulateipynb)
9. [H. LLM-Based Combinatorial Pipeline (combine_and_simulate_v2.ipynb)](#h-llm-based-combinatorial-pipeline-combine_and_simulate_v2ipynb)

---

## 개요

이 파이프라인은 LLM(Large Language Model)을 활용하여 WorldQuant Brain 플랫폼에서 사용할 수 있는 알파(Alpha) 아이디어를 대량으로 생성하고, 검증하여 제출 가능한 후보를 선별하는 자동화 시스템입니다.

**네 가지 주요 파이프라인:**
| 파이프라인 | 노트북 | 목적 | 특징 (2026-02) |
|-----------|--------|------|------|
| **Seed Generation** | `llm_alpha_guide.ipynb` | 단순 seed-alpha 대량 탐색 | 70-80% 패턴 + 20-30% 자유 조합, 구조 필터링, .txt 출력 |
| **Combinatorial v1** | `combine_and_simulate.ipynb` | 다중 데이터셋 seed 전수 조합 | **전수 조합(Cartesian Product)**, operators_list.json 전체 사용, Sanity Check 통합 |
| **Combinatorial v2** | `combine_and_simulate_v2.ipynb` | LLM 기반 seed 조합 | **LLM이 2~4개 seed 자유 조합**, 10 batch × 100개, PASS만 저장 |
| **Refinement** | `refine.ipynb` | 기존 알파 개선 | 진화 알고리즘, 토너먼트 선택 |

### 전체 흐름도 (Seed → Combine → Refine) [2026-02-09]

```
[Pipeline A: Seed Generation]
  llm_alpha_guide.ipynb
  데이터셋별 단순 seed-alpha 대량 생성
  → {dataset}.txt (per-dataset 결과)
          │
          ├─────────────────────────────────────────┐
          ▼                                         ▼
[Pipeline G: Combinatorial v1]           [Pipeline H: Combinatorial v2] ★ NEW
  combine_and_simulate.ipynb               combine_and_simulate_v2.ipynb
  Cartesian Product 전수 조합              LLM 기반 자유 조합
  → {ds1}_{ds2}_{ds3}_comb.txt             → results/combinatorial/combined_alphas.txt
                                           (10 batch × 100개, PASS만 저장)
          │                                         │
          └─────────────────┬───────────────────────┘
                            ▼
[Pipeline F: Refinement]
  refine.ipynb
  유망 알파를 진화 알고리즘으로 개선
  → TARGET_SHARPE 도달까지 반복
```

### Seed Generation Pipeline 상세 흐름

```
[Phase 1: 초기화]      [Phase 2: 데이터 탐색]     [Phase 3: Sanity Checker]
REGION/UNIVERSE 설정 → Brain API 인증           → datafields/operators JSON 로드
                    → datasets_df 조회           → sanity_checker() 정의
                    → TARGET_DATASET_IDS 설정
                    → datasets_dict 구성

[Phase 4: Seed 생성]              [Phase 5: 구조 필터링]        [Phase 6: 시뮬 & 저장]
LLM 호출                        → is_valid_seed_alpha()      → 8개씩 배치 시뮬레이션
(70-80% 패턴 + 20-30% 자유)       depth ≤ 3?                 → extract_sim_record()
gen_json/{dataset}_{idx}.json     operators ≤ 3?             → write_dataset_txt()
                                  signal datafield 1개?       → {dataset}.txt
```

### Combination Pipeline 상세 흐름 (2026-02)

```
[Step 1: 로드]         [Step 2: 전수조합]      [Step 3: Variant]     [Step 4: Sanity]   [Step 5: 시뮬 & 저장]
{dataset}.txt       → Cartesian Product  → generate_          → sanity_        → {datasets}_comb.txt
0-fail 알파 추출       (30×25×5 예시)        combination_         checker()        {datasets}_comb.json
parse_zero_fail_     generate_all_        variants()           (parser.py       simulate_alpha_
alphas_from_file()   combinations()       operators_list.       기반)             list_multi()
                     → 3,750 조합          json 기반                             (선택적)
```

**GPT 제외**: Variant 생성은 `operators_list.json`에서 직접 샘플링하므로 GPT API를 사용하지 않음

### Refinement Pipeline 상세 흐름

```
[베이스 알파 입력]     [LLM 변형 생성]      [전체 시뮬레이션]    [토너먼트 선택]     [반복]
      │                    │                    │                   │              │
  BASE_EXPR         →  build_prompt()  →  simulate_alpha_  →  tournament_  →  TARGET_SHARPE
  (사용자 입력)         50개 변형 생성       list_multi()        select()        도달까지
                                            (전부 시뮬)      parent_pool 유지
```

---

## A. Seed-Alpha 생성 단계

### A.0 Seed-Alpha 컨셉

Seed-alpha는 **단일 신호(signal) 데이터필드 하나만 사용하는 단순한 알파 수식**입니다.
나중에 `combine_and_simulate.ipynb`에서 여러 seed를 조합하여 복합 알파를 만드는 데 사용됩니다.

**구조 제약 (is_valid_seed_alpha로 검증):**
- signal datafield 1개만 사용 (cap, industry 등 helper는 허용)
- 최대 3개 operator
- 트리 depth ≤ 3

**생성 비율:**
- **~70-80%**: 4가지 ALLOWED_PATTERNS 중 하나를 따름 (검증된 구조)
- **~20-30%**: FREE_PATTERN - `operators_list.json`에서 operator를 자유롭게 조합 (구조 제한은 동일)

### A.1 노트북 셀별 실행 흐름

| 순서 | Cell ID | 역할 |
|------|---------|------|
| 1 | `93b3d3ce` | 글로벌 설정: REGION, UNIVERSE, DELAY, settings |
| 2 | `793c3dca` | Brain API 세션 인증 (`ace.start_session()`) |
| 3 | `9f6e98dc` | 전체 datasets 조회 |
| 4 | `4fbd2558` | dataset_ids 하드코딩 + datafields 미리보기 |
| 5 | `b2e6df49` | 선택된 datasets 필터링/정렬 |
| 6 | `1168f589` | TARGET_DATASET_IDS 설정 + datafield 스펙 확인 |
| 7 | `db070843` | sanity_checker() 정의 (타입 검증) |
| 8 | `ceeb8178` | datasets_dict, datasets_list 확정 |
| 9 | `7949f887` | **유틸리티 함수** (seed 검증, txt I/O 등) |
| 10 | `0693ee9e` | **메인 파이프라인** (LLM 생성 → 필터링 → 시뮬 → .txt 저장) |
| 11 | `0d62c25c` | LLM 프롬프트 데모/미리보기 |

### A.2 생성 패턴

#### ALLOWED_PATTERNS (~70-80%)

LLM에게 아래 4가지 패턴을 우선적으로 사용하도록 지시합니다:

| 패턴 | 구조 | 용도 | depth |
|------|------|------|-------|
| **P1 Backfill** | `ts_backfill(DATA, window)` | coverage < 0.6 데이터 채우기 | 1 |
| **P2 TS Z-score** | `ts_zscore(DATA, window)` | 시계열 정규화 | 1 |
| **P2+vec** | `ts_zscore(vec_avg(DATA), window)` | VECTOR 타입 시계열 정규화 | 2 |
| **P3 Ratio Z-score** | `ts_zscore(divide(DATA, cap), window)` | 시가총액 대비 비율 | 2 |
| **P4 Group Z-score** | `group_zscore(rank(DATA), industry)` | 산업 내 상대 순위 | 2 |
| **P4+ratio** | `group_zscore(rank(divide(DATA, cap)), industry)` | 산업 내 시가총액 대비 순위 | 3 |

#### FREE_PATTERN (~20-30%)

나머지 20-30%는 `operators_list.json`에 정의된 모든 operator를 자유롭게 조합할 수 있습니다.
구조 제한(depth ≤ 3, operators ≤ 3, signal 1개)은 동일하게 적용됩니다.

**Free pattern 예시:**
```
ts_rank(DATA, 63)
quantile(ts_delta(DATA, 21))
ts_decay_linear(DATA, 126)
zscore(ts_mean(DATA, 63))
group_neutralize(rank(DATA), industry)
ts_ir(DATA, 126)
winsorize(ts_zscore(DATA, 63))
rank(ts_kurtosis(DATA, 126))
group_rank(ts_av_diff(DATA, 63), industry)
```

**Window 파라미터**: 5 ~ 252 사이 다양한 값 (5, 10, 21, 42, 63, 126, 189, 252 등)

### A.3 프롬프트 구조

```
<MISSION>
  - {alpha_num}개의 분산된 seed-alpha 수식 생성
  - idea, description, implementation, confidence_level 4가지 파트
</MISSION>

<GENERATION_RATIO>
  - 70-80%: ALLOWED_PATTERNS 사용
  - 20-30%: FREE_PATTERN (OPERATORS_LIST에서 자유 조합)
</GENERATION_RATIO>

<ALLOWED_PATTERNS>
  Pattern 1 - Backfill: ts_backfill(DATA, window)
  Pattern 2 - Time-series Z-score: ts_zscore(DATA, window)
  Pattern 3 - Ratio Z-score: ts_zscore(divide(DATA, cap), window)
  Pattern 4 - Group Z-score: group_zscore(rank(DATA), industry)
</ALLOWED_PATTERNS>

<FREE_PATTERN>
  - operators_list.json의 모든 operator 사용 가능
  - 구조 제한은 동일 (operators ≤ 3, signal 1개, vec_* 필수)
  - 9가지 예시 제공
</FREE_PATTERN>

<OPERATORS_LIST>
  [Arithmetic] add(x,y) | abs(x) | log(x) | subtract(x,y) | ...
  [Time Series] ts_delta(x,d) | ts_zscore(x,d) | ts_rank(x,d) | ...
  [Cross Sectional] rank(x) | quantile(x) | zscore(x) | ...
  [Vector] vec_avg(x) | vec_sum(x) | vec_max(x) | ...
  [Group] group_zscore(x,group) | group_rank(x,group) | ...
  [Transformational] trade_when(x,y,z) | bucket(...) | ...
  [Logical] if_else(cond,then,else) | and(x,y) | ...
</OPERATORS_LIST>

<STRICT_RULES>
  RULE1: ALLOWED_DATAFIELDS의 datafield ID만 사용
  RULE2: signal datafield 1개만 (cap, industry 등 helper 허용)
  RULE3: VECTOR 타입은 vec_* operator로 감싸기
  RULE4: window 파라미터 다양성 (5~252)
  RULE5: 최대 3개 operator
  RULE6: GROUP 타입은 단독 사용 불가
</STRICT_RULES>

<DIVERSITY_GUIDANCE>
  - 모든 datafield 골고루 사용
  - ALLOWED_PATTERNS + FREE_PATTERN 혼합
  - coverage 낮은 필드는 Pattern 1 우선
</DIVERSITY_GUIDANCE>

<ALLOWED_DATAFIELDS>
  [top N datafields by alphaCount]
</ALLOWED_DATAFIELDS>
```

### A.4 유틸리티 함수 (Cell `7949f887`)

| 함수 | 역할 |
|------|------|
| `datset_to_simnum(dataset)` | dataset 규모 → 배치 횟수 (1~20) |
| `get_json_num(dataset)` | gen_json 인덱스 헬퍼 |
| `is_valid_seed_alpha(expr)` | 구조 검증 → `(bool, reason)` 반환 |
| `extract_sim_record(sim_result)` | Brain API 응답 → dict 추출 |
| `format_alpha_entry(idx, record)` | record → .txt 출력 포맷 |
| `load_existing_txt_records(dataset, dir)` | 기존 .txt 파싱 (incremental append) |
| `write_dataset_txt(dataset, records, ...)` | 정렬 + 중복제거 + .txt 저장 |

#### Operator 포맷 헬퍼 (Cell `0693ee9e`)

| 함수 | 역할 |
|------|------|
| `format_operators_compact(operators_dict)` | operators_list.json → 카테고리별 compact string 변환 |

#### `is_valid_seed_alpha()` 검증 규칙

```python
def is_valid_seed_alpha(expression):
    # 1. sanity_checker() 타입 검증 통과?
    # 2. tree_node() 파싱 성공?
    # 3. depth ≤ 3?
    # 4. operator 수 ≤ 3?
    # 5. signal datafield 1개만? (cap, industry, sector, subindustry는 helper로 허용)
    return (True/False, "reason")
```

### A.5 메인 파이프라인 흐름 (Cell `0693ee9e`)

```python
for dataset in datasets_list:
    # 1. Resume 체크 (gen_json 파일 수 vs 목표)
    #    이미 완료 → SKIP

    # 2. 기존 {dataset}.txt 로드 (incremental append용)
    all_records = load_existing_txt_records(dataset)

    for batch in remaining_batches:
        # 3. LLM 호출: generate_seed_expressions()
        #    - dataset의 top N datafield 조회
        #    - operators_list.json 로드 → compact string
        #    - 70-80% 패턴 + 20-30% 자유 프롬프트

        # 4. JSON 파싱 + gen_json/ 저장

        # 5. is_valid_seed_alpha() 필터링 ← 시뮬 전 필터링 (API 절약)

        # 6. 8개씩 배치 시뮬레이션
        #    태그: ['seed_alpha', 'confidence_X']

        # 7. extract_sim_record()로 결과 수집

        # 8. write_dataset_txt() → {dataset}.txt 저장
        #    (fail_count ASC, sharpe DESC 정렬, 중복 제거)
```

### A.6 출력 형식 ({dataset}.txt)

```
================================================================================
RISK60 SEED ALPHA LIST (Sorted by FAIL count, ascending)
Region: EUR, Universe: TOP2500
Total: 187 alphas
================================================================================

--- #1 | FAIL: 0 (PASS) ---
ID: npkoeVnl
Region: EUR, Universe: TOP2500
Sharpe: 2.04, Fitness: 1.34, Turnover: 0.1017
Expression: ts_zscore(vec_avg(rsk60_offer), 252)
Status: PASS

--- #2 | FAIL: 3 ---
ID: ZYw2qWJ3
Region: EUR, Universe: TOP2500
Sharpe: 0.95, Fitness: 0.4, Turnover: 0.2619
Expression: ts_zscore(vec_avg(rsk60_last), 126)
Failed: LOW_SHARPE, LOW_FITNESS, LOW_SUB_UNIVERSE_SHARPE

================================================================================
Summary:
- Total: 187 alphas
- PASS (no FAIL): 8 alphas
- With FAILs: 179 alphas
- Best Sharpe: 2.04
================================================================================
```

### A.7 저장 경로 규칙

```
llm_alpha_gen/
├── gen_json/                    # LLM 생성 raw JSON (resume용)
│   ├── {dataset}_{batch}.json
│   ├── risk60_0.json
│   └── model25_1.json
│
├── {dataset}.txt                # 시뮬 결과 (per-dataset, 정렬됨)
│   ├── risk60.txt
│   ├── model25.txt
│   └── model138.txt
```

### A.8 배치 수 결정 로직 (`datset_to_simnum`)

```python
score = int(np.log10(userCount + 1) + fieldCount)

if score <= 3:    return 1   # 100개 알파
elif score <= 10: return 2   # 200개 알파
elif score <= 50: return 3   # 300개 알파
elif score <= 100: return 10 # 1,000개 알파
else:             return 20  # 2,000개 알파
```

### A.9 LLM 출력 JSON 구조

```json
{
  "results": [
    {
      "idea": "Securities lending offer rate normalized over 1-year window",
      "description": "Pattern 2 with vec_avg wrapper for VECTOR type, 252-day window for annual normalization",
      "implementation": "ts_zscore(vec_avg(rsk60_offer), 252)",
      "confidence_level": 0.8
    },
    {
      "idea": "Kurtosis-based regime signal for model field",
      "description": "FREE_PATTERN using ts_kurtosis with rank for cross-sectional normalization",
      "implementation": "rank(ts_kurtosis(mdl138_4idpc, 126))",
      "confidence_level": 0.6
    }
  ]
}
```

### A.10 CONFIG 파라미터

```python
TOP_N_DATAFIELDS = 10    # 각 dataset에서 alphaCount 상위 N개만 LLM에 전달
SEED_ALPHA_NUM = 100     # LLM에게 요청할 알파 개수 per batch
OUTPUT_DIR = '.'         # .txt 파일 저장 디렉토리
```

---

## B. Sanity Check 단계

### B.1 파싱 흐름

```
원본 Expression
    │
    ▼
parse_expression()  ─→  중위 표기법을 함수 호출 형태로 변환
    │                   예: "a + b" → "add(a, b)"
    ▼
build_tree()        ─→  함수 호출 문자열을 트리 구조로 변환
    │                   TreeNode 객체 생성
    ▼
classify_node()     ─→  각 노드 타입 분류
    │                   (operator / datafield / number / special_argument)
    ▼
tree_node()         ─→  위 과정을 통합한 최종 함수
```

### B.2 노드 타입 분류 규칙 (`classify_node`)

```python
# parser.py:435-443
def classify_node(node):
    if node.children:
        node.node_type = "operator"           # 자식이 있으면 연산자
    elif '=' in node.value:
        node.node_type = "special_argument"   # '='가 있으면 특수 인자
    elif is_numeric(node.value):
        node.node_type = "number"             # 숫자
    else:
        node.node_type = "datafield"          # 나머지는 데이터필드
```

### B.3 Sanity Checker 핵심 로직

노트북의 `sanity_checker` 함수가 검증하는 사항:

```python
def sanity_checker(exp, _debug=False):
    try:
        exp_tree = tree_node(exp)

        for node in [n for n in exp_tree.collect_all_nodes() if n.node_type == "operator"]:
            # 1) 각 operator의 input 타입과 children 타입 매칭 확인
            if not check_input(
                eval(operators[node.value]['input']),  # 예: ['MATRIX', 'MATRIX']
                [return_type(x, operators, datafields) for x in node.children]
            ):
                return False

        # 2) 최종 출력이 MATRIX여야 함
        if return_type(exp_tree, operators, datafields) == "MATRIX":
            return True
        else:
            return False

    except Exception as e:
        return False  # 파싱 자체가 실패하면 False
```

### B.4 Seed-Alpha 추가 검증 (`is_valid_seed_alpha`)

sanity_checker()를 통과한 후 추가로 seed-alpha 구조 규칙을 검증합니다:

| 검증 항목 | 기준 | 실패 시 |
|----------|------|---------|
| `sanity_checker()` | 타입 매칭 + 최종 MATRIX | 즉시 reject |
| tree depth | ≤ 3 | reject (복잡한 중첩) |
| operator 수 | ≤ 3 | reject (과도한 연산) |
| signal datafield 수 | = 1 | reject (다중 신호) |

**Helper datafields** (제한에서 제외): `cap`, `industry`, `sector`, `subindustry`

### B.5 Sanity 통과했지만 시뮬레이션 실패하는 케이스

| 케이스 | 원인 | 예시 |
|--------|------|------|
| **데이터필드 미존재** | `datafield/total.json`에는 있지만 해당 region/universe에서 실제로 제공되지 않음 | `star_si_shortsqueeze_rank` |
| **Region 불일치** | USA 전용 필드를 EUR에서 사용 | `rsk70_mfm2_usfast_*` 필드를 EUR에서 사용 |
| **Coverage 부족** | 데이터가 너무 적어서 시뮬레이션 불가 | coverage 0.1 이하 필드 |
| **Syntax 오류** | Sanity는 타입만 체크, 문법 오류는 못 잡음 | `ts_zscore(field, "abc")` (윈도우가 문자열) |
| **연산 결과 NaN** | 0으로 나누기, 빈 데이터 등 | `divide(a, 0)` |
| **결과가 상수** | 모든 종목에서 동일한 값 | `subtract(field, field)` → 항상 0 |

---

## C. Brain Simulation 단계

### C.1 세션 인증 흐름

```python
# ace_lib.py
def start_session() -> SingleSession:
    s = SingleSession()                              # 싱글톤 세션
    s.auth = get_credentials()                       # ~/secrets/platform-brain.json
    r = s.post(brain_api_url + "/authentication")

    if r.status_code == 401:
        if r.headers["WWW-Authenticate"] == "persona":
            # Biometrics 인증 필요 → URL 출력 후 사용자 입력 대기
            print("Complete biometrics authentication...")
            input()
    return s
```

**세션 타임아웃 체크:**
```python
# 노트북에서 시뮬레이션 루프 내
if ace.check_session_timeout(s) < 500:  # 500초 미만 남으면
    s = ace.start_session()              # 재로그인
```

### C.2 알파 Payload 구조 (`generate_alpha`)

```python
simulation_data = {
    "type": "REGULAR",
    "settings": {
        "instrumentType": "EQUITY",
        "region": "EUR",             # 🔧 글로벌 변수
        "universe": "TOP2500",       # 🔧 글로벌 변수
        "delay": 1,
        "decay": 0,
        "neutralization": "INDUSTRY",
        "truncation": 0.08,
        "pasteurization": "ON",
        "unitHandling": "VERIFY",
        "nanHandling": "OFF",
        "language": "FASTEXPR",
        "visualization": False
    },
    "regular": "ts_zscore(vec_avg(rsk60_offer), 252)"  # seed-alpha expression
}
```

### C.3 병렬 시뮬레이션

```python
# 8개씩 배치로 시뮬레이션
for rep in range(0, len_alpha_list, 8):
    batch_end = min(rep + 8, len_alpha_list)
    sim_results = list(ace.multi_simulate_alphas_map(
        s,
        alpha_list[rep:batch_end],
        tags_list[rep:batch_end],     # ['seed_alpha', 'confidence_0.8']
        descs_list[rep:batch_end],
        batch_end - rep
    ))
```

### C.4 결과 추출 (`extract_sim_record`)

```python
def extract_sim_record(sim_result):
    return {
        'alpha_id': sim_result['id'],
        'region': settings.get('region', ''),
        'universe': settings.get('universe', ''),
        'sharpe': is_data.get('sharpe', 0),
        'fitness': is_data.get('fitness', 0),
        'turnover': is_data.get('turnover', 0),
        'expression': sim_result.get('regular', {}).get('code', ''),
        'fail_count': len(failed_checks),
        'failed_checks': ['LOW_SHARPE', ...],
    }
```

---

## D. 제출 가능성 판단 단계

> **중요**: 현재 코드는 자동 제출을 하지 않습니다. .txt 결과를 기반으로 사람이 판단합니다.

### D.1 필수 통과 조건 (Hard Filter)

Brain 제출을 위해 **모든 항목이 PASS**여야 합니다:

| Check 이름 | 조건 | 의미 |
|-----------|------|------|
| `LOW_SHARPE` | Sharpe ≥ 1.58 | 최소 수익/위험 비율 |
| `LOW_FITNESS` | Fitness ≥ 1.0 | 알파 적합도 |
| `LOW_TURNOVER` | Turnover ≥ 0.01 | 너무 거래가 없으면 안 됨 |
| `HIGH_TURNOVER` | Turnover ≤ 0.7 | 너무 잦은 거래 방지 |
| `CONCENTRATED_WEIGHT` | - | 특정 종목 집중 방지 |
| `LOW_SUB_UNIVERSE_SHARPE` | - | 하위 유니버스 성과 |

### D.2 추가 확인 조건 (상관관계)

| 상관관계 | 의미 | 기준 |
|---------|------|------|
| `SELF_CORRELATION` | 내가 이전에 제출한 알파와 얼마나 비슷한가 | < 0.7 |
| `PROD_CORRELATION` | 현재 운용 중인 알파와 얼마나 비슷한가 | ≤ 0.7 |

### D.3 선호 조건 (Soft Filter)

| 지표 | 좋은 값 | 설명 |
|------|---------|------|
| `fitness` | 높을수록 좋음 (1.5+) | Sharpe와 Turnover를 종합한 지표 |
| `sharpe` | 높을수록 좋음 (2.0+) | 위험 대비 수익 |
| `turnover` | 0.1~0.5 범위 | 너무 낮거나 높지 않은 적정 거래 |

---

## E. 사람이 눈으로 확인하는 포인트

### E.1 {dataset}.txt 확인 방법

.txt 파일은 fail_count ASC → sharpe DESC 순으로 정렬되어 있으므로:

```
1. 맨 위 → FAIL: 0 (PASS) 알파가 먼저 나옴 → 제출 후보
2. Summary 섹션 → 전체 PASS 개수 확인
3. PASS 알파의 Expression → 수식 복잡도/해석 가능성 검토
4. 동일 datafield를 사용하는 PASS 알파 간 유사도 체크 (중복 제거)
```

### E.2 gen_json/*.json 확인 포인트

| 항목 | 확인 방법 | 권장 기준 |
|------|----------|----------|
| **implementation 길이** | `len(impl)` | 100자 이내 (seed이므로 짧아야 함) |
| **datafield 사용 개수** | expression 내 datafield 카운트 | **1개** (seed 규칙) |
| **operator 개수** | tree_node 분석 | **3개 이하** |
| **confidence_level** | JSON 필드 직접 확인 | 0.6 이상 우선 검토 |
| **VECTOR 타입 처리** | `vec_avg()` 사용 여부 | VECTOR 필드는 반드시 감싸기 |

### E.3 결과 검토 권장 순서

```
1. {dataset}.txt 상단의 PASS 알파 확인
   → FAIL: 0 (PASS) 인 것만 제출 후보

2. PASS 알파의 Sharpe / Fitness 확인
   → Sharpe ≥ 2.0, Fitness ≥ 1.5 우선

3. Turnover 범위 확인
   → 0.1 ~ 0.5 적정

4. Expression 해석
   → seed-alpha이므로 단순해야 함
   → 같은 datafield의 다른 패턴 결과 비교

5. combine_and_simulate.ipynb로 넘길 seed 선별
   → PASS seed끼리 조합 시뮬레이션
```

---

## F. 진화 알고리즘 기반 알파 개선 (refine.ipynb)

### F.1 개요

`refine.ipynb`는 기존에 발견한 알파를 시작점으로 하여 LLM을 통해 변형을 생성하고,
시뮬레이션 결과 기반의 토너먼트 선택을 통해 점진적으로 성능을 개선하는 진화 알고리즘 파이프라인입니다.

**3개 파이프라인 비교:**

| 항목 | Seed Generation (llm_alpha_guide) | Combination (combine_and_simulate) | Refinement (refine) |
|------|-----------------------------------|-------------------------------------|---------------------|
| 목적 | 단순 seed-alpha 대량 탐색 | cross-dataset seed 조합 | 기존 알파 개선 |
| 입력 | 데이터셋 + 데이터필드 | {dataset}.txt (0-fail 알파) | 베이스 알파 수식 |
| LLM 역할 | 70-80% 패턴 + 20-30% 자유 seed 생성 | cross-dataset 조합 생성 | 기존 알파 변형 생성 |
| 배치 크기 | 100개/배치 | NUM_COMBINATIONS (기본 100) | 50개/세대 |
| 구조 검증 | `is_valid_seed_alpha()` + `sanity_checker()` | `validate_combination()` (permissive) | 없음 (전부 시뮬) |
| 필터 시점 | **시뮬 전** (API 절약) | 시뮬 전 (minimal) | 없음 |
| 선택 방식 | 없음 (전부 저장) | 없음 (전부 저장) | 토너먼트 선택 |
| 반복 | 데이터셋별 고정 횟수 | 1회 실행 | 목표 도달까지 반복 |
| 결과 저장 | `{dataset}.txt` | `combined_alpha.txt` + `good_alpha_list.json` | 콘솔 출력 |
| 태그 | `seed_alpha` | - | - |
| 시뮬 API | `multi_simulate_alphas_map()` | `simulate_alpha_list_multi()` | `simulate_alpha_list_multi()` |

### F.2 핵심 파라미터

```python
MODEL_NAME = "gpt-4.1-mini"
GENERATION_SIZE = 50        # 세대당 생성할 변형 수
MAX_GENERATIONS = 100       # 최대 세대 수
TARGET_SHARPE = 2.5         # 목표 Sharpe (도달 시 종료)
PARENT_POOL_SIZE = 30       # 부모 풀 최대 크기
```

### F.3 진화 루프 흐름

```python
for gen in range(1, MAX_GENERATIONS + 1):
    # 1. 부모 선택 (토너먼트 선택 또는 BASE_EXPR)
    parent_a = tournament_select(parent_pool) if pool else BASE_EXPR
    parent_b = tournament_select(parent_pool) if random < 0.4 else None

    # 2. LLM으로 50개 변형 생성
    prompt = build_prompt(BASE_EXPR, parent_a, parent_b)
    expression_list = parse_expression_list(response)

    # 3. 전체 시뮬레이션 (Sanity Check 없음)
    result = ace.simulate_alpha_list_multi(s, alpha_list)

    # 4. 평가 및 부모 풀 업데이트 (상위 30개만 유지)
    parent_pool = clip_parent_pool(parent_pool, PARENT_POOL_SIZE)

    # 5. TARGET_SHARPE 도달 시 종료
```

### F.4 토너먼트 선택

```python
def tournament_select(pool):
    """풀에서 무작위 2개를 뽑아 Sharpe가 높은 쪽을 선택"""
    a, b = random.sample(pool, 2)
    return a if a["sharpe"] > b["sharpe"] else b
```

### F.5 종료 조건

| 조건 | 의미 |
|------|------|
| `sharpe >= TARGET_SHARPE` | 목표 달성, 성공 종료 |
| `gen >= MAX_GENERATIONS` | 최대 세대 도달, 실패 종료 |

---

## G. Combinatorial Alpha Pipeline (combine_and_simulate.ipynb)

### G.0 개요 (2026-02 업데이트)

`combine_and_simulate.ipynb`는 Seed Generation 파이프라인에서 생성된 **0-fail seed 알파**들을
**전수 조합(Cartesian Product)** 방식으로 결합하여 복합 알파를 만들고, Brain API로 시뮬레이션합니다.

**핵심 변경사항 (2026-02):**
1. **Operator 제한 제거**: ~~add/subtract/min/max/group_zscore만 허용~~ → `operators_list.json`의 모든 operator 사용 가능
2. **전수 조합 생성**: 지정된 dataset들의 0-fail seed 전부를 Cartesian product로 조합
3. **Sanity Check 통합**: `my_research.py`의 sanity checker를 통해 invalid expression 자동 제거
4. **성능 제어 옵션**: `max_combinations`, `max_variants_per_combo`, `concurrency` 파라미터
5. **결과 저장 형식**: `{dataset1}_{dataset2}_{dataset3}_comb.txt` 규칙

**핵심 아이디어:**
- 단일 dataset의 seed 알파는 단순하지만, 서로 다른 dataset의 seed를 결합하면 다양한 신호를 합성
- Operator 선택은 `operators_list.json`에서 자동으로 샘플링 (GPT 제외)
- Sanity check로 타입 mismatch, parsing error 제거
- 조합 후에도 원본 seed expression 내부는 수정하지 않음 (외부에서만 결합)

### G.1 노트북 셀별 실행 흐름

| 순서 | Cell ID | 역할 |
|------|---------|------|
| 1 | `cell-2` | Imports (json, os, re, OpenAI, ace_lib 등) |
| 2 | `cell-3` | Configuration (DATASET_FILES, REGION, UNIVERSE, ALLOWED_OPERATORS) |
| 3 | `cell-5` | 유틸리티 (load_json, save_json, append_to_json) |
| 4 | `cell-7` | `parse_dataset_file()` - .txt에서 0-fail 알파 파싱 |
| 5 | `cell-9` | GPT API 호출 (프롬프트 생성 + 스트리밍 조합 생성) |
| 6 | `cell-11` | 조합 검증 (count_operators, detect_datasets_used) |
| 7 | `cell-13` | combined_alpha.txt 저장 |
| 8 | `cell-15` | Brain API 시뮬레이션 + good_alpha_list.json 저장 |
| 9 | `cell-17` | 데이터셋 파일 관리 (런타임 추가/목록) |
| 10 | `cell-19` | 파이프라인 파라미터 설정 |
| 11 | `cell-20~25` | 메인 파이프라인 5-Step 실행 |

### G.2 입력: 데이터셋 .txt 파일

Seed Generation 파이프라인의 출력인 `{dataset}.txt`를 입력으로 사용합니다.

```python
DATASET_FILES = {
    'mdl25': SCRIPT_DIR / 'mdl25.txt',
    'mdl30': SCRIPT_DIR / 'mdl30.txt',
    'mdl138': SCRIPT_DIR / 'mdl138.txt',
    # 런타임에 add_dataset_file()로 추가 가능
}
```

**파싱 규칙 (`parse_dataset_file`)**:
- `FAIL: 0` 블록만 추출 (0-fail = PASS 알파만)
- 정규식으로 ID, Sharpe, Fitness, Turnover, Expression 추출
- 각 dataset별 최대 10개의 alpha를 GPT 프롬프트에 포함

### G.3 전수 조합 (Cartesian Product) 생성

`generate_all_combinations()`가 Cartesian product로 모든 seed 조합을 생성합니다:

**예시: 3개 dataset 조합**
```python
DATASETS_TO_COMBINE = ['mdl25', 'mdl30', 'mdl138']

# 각 dataset의 0-fail seed 개수
mdl25: 33 seeds
mdl30: 25 seeds
mdl138: 29 seeds

# Cartesian Product
total_combinations = 33 × 25 × 29 = 23,925 combinations
```

**Sampling 옵션**:
```python
MAX_COMBINATIONS = 10000  # 23,925개 중 10,000개만 랜덤 샘플링
# 또는
MAX_COMBINATIONS = None   # 전수 조합 (23,925개 전부 생성)
```

**조합 데이터 구조**:
```python
{
    'datasets': ['mdl25', 'mdl30', 'mdl138'],
    'seeds': [alpha1_dict, alpha2_dict, alpha3_dict],
    'expressions': [
        'quantile(mdl25_smartestimate_f12m_eps)',
        'quantile(mdl30_new_psprise_pct_fy1_eps)',
        'zscore(mdl138_ai_analyst_accuracy)'
    ],
    'metadata': {
        'seed_ids': ['A1B2C3', 'D4E5F6', 'G7H8I9'],
        'sharpes': [1.83, 1.71, 1.65],
        'fitnesses': [1.14, 1.11, 1.09]
    }
}
```

### G.4 Variant 생성 (`operators_list.json` 기반)

각 seed 조합에 대해 `generate_combination_variants()`가 여러 variant를 생성합니다:

**Operator 카테고리 (from `operators_list.json`)**:
```python
COMBINATION_OPERATORS = {
    'arithmetic': ['add', 'subtract', 'multiply', 'divide', 'min', 'max'],
    'normalization': ['rank', 'zscore', 'quantile', 'normalize', 'scale'],
    'group': ['group_zscore', 'group_rank', 'group_neutralize', 'group_scale'],
    'tail_handling': ['winsorize', 'tail', 'pasteurize'],
}
```

**Variant 유형 (max_variants_per_combo = 5)**:
1. **simple_add**: `add(seed1, seed2, seed3)`
2. **add_with_group**: `group_zscore(add(seed1, seed2, seed3), subindustry)`
3. **normalized_weighted_sum**: `scale(zscore(seed1) + zscore(seed2) + zscore(seed3))`
4. **min_max_ensemble**: `add(min(seed1, seed2, seed3), max(seed1, seed2, seed3))`
5. **tail_clipped**: `winsorize(rank(add(seed1, seed2, seed3)))`

**GPT 제외**: Variant 생성은 programmatic 샘플링이므로 GPT API 호출 없음

### G.5 Sanity Check (Type Validation)

`sanity_checker()`가 expression을 tree parsing하여 타입 검증을 수행합니다:

**검증 로직 (from `my_research.py`)**:
```python
def sanity_checker(exp, operators, datafields):
    try:
        exp_tree = tree_node(exp)  # parser.py의 tree 파싱

        # 각 operator node의 input/output 타입 검증
        for node in exp_tree.collect_all_nodes():
            if node.node_type == "operator":
                operator_inputs = eval(operators[node.value]['input'])
                children_types = [return_type(child, operators, datafields)
                                  for child in node.children]

                if not check_input(operator_inputs, children_types):
                    return False, f"Type mismatch in {node.value}"

        # 최종 output type이 MATRIX인지 확인
        if return_type(exp_tree, operators, datafields) != "MATRIX":
            return False, "Final output type is not MATRIX"

        return True, None

    except Exception as e:
        return False, f"Parsing error: {str(e)}"
```

**Sanity Check 실패 예시**:
- Unknown operator
- Type mismatch (e.g., `ts_sum(industry)` - ts_sum expects MATRIX, got NUMBER)
- Parsing error (괄호 불일치, 문법 오류)
- Final output != MATRIX

**통과율**: 일반적으로 60-80%의 variant가 sanity check를 통과합니다.

### G.6 시뮬레이션 방식

Combination 파이프라인은 Seed와 다른 시뮬레이션 API를 사용합니다:

```python
def simulate_combinations(session, combinations):
    # build_alpha_config()로 payload 생성
    # ace.simulate_alpha_list_multi() 호출 (Seed의 multi_simulate_alphas_map과 다름)
    # limit_of_concurrent_simulations=3
    # limit_of_multi_simulations=3
    # check_submission=True
```

**Alpha Config:**
```python
{
    "type": "REGULAR",
    "settings": {
        "instrumentType": "EQUITY",
        "region": REGION,       # EUR
        "universe": UNIVERSE,   # TOP2500
        "delay": DELAY,         # 1
        "decay": 0,
        "neutralization": "INDUSTRY",
        "truncation": 0.08,
        "pasteurization": "ON",
        "unitHandling": "VERIFY",
        "nanHandling": "OFF",
        "language": "FASTEXPR",
        "visualization": False,
    },
    "regular": "add(quantile(mdl25_smartestimate_f12m_eps), quantile(mdl30_new_psprise_pct_fy1_eps))"
}
```

### G.7 출력 파일 (2026-02)

#### {dataset1}_{dataset2}_{dataset3}_comb.txt

파일명 규칙: dataset 이름을 알파벳 순으로 정렬하여 연결

```
================================================================================
COMBINATORIAL ALPHAS: mdl138 x mdl25 x mdl30
Generated: 2026-02-08 10:30:00
Total variants: 5,824
================================================================================

--- #1 | simple_add ---
Expression: add(quantile(mdl25_smartestimate_f12m_eps), quantile(mdl30_new_psprise_pct_fy1_eps), zscore(mdl138_ai_analyst_accuracy))
Operators Used: add
Source Datasets: mdl25, mdl30, mdl138
Seed IDs: A1B2C3, D4E5F6, G7H8I9
Source Sharpes: 1.83, 1.71, 1.65
Source Fitnesses: 1.14, 1.11, 1.09

--- #2 | add_with_group_zscore ---
Expression: group_zscore(add(quantile(mdl25_...), quantile(mdl30_...), zscore(mdl138_...)), subindustry)
Operators Used: add, group_zscore
Source Datasets: mdl25, mdl30, mdl138
Seed IDs: A1B2C3, D4E5F6, G7H8I9
Source Sharpes: 1.83, 1.71, 1.65
Source Fitnesses: 1.14, 1.11, 1.09

================================================================================
Total: 5,824 expression variants
================================================================================
```

#### {dataset1}_{dataset2}_{dataset3}_comb.json

```json
{
  "timestamp": "2026-02-08T10:30:00",
  "datasets": ["mdl25", "mdl30", "mdl138"],
  "total_seed_combinations": 23925,
  "total_variants": 5824,
  "variants": [
    {
      "expression": "add(quantile(mdl25_...), quantile(mdl30_...), zscore(mdl138_...))",
      "variant_type": "simple_add",
      "combination_info": {
        "datasets": ["mdl25", "mdl30", "mdl138"],
        "seeds": [...],
        "expressions": [...],
        "metadata": {
          "seed_ids": ["A1B2C3", "D4E5F6", "G7H8I9"],
          "sharpes": [1.83, 1.71, 1.65],
          "fitnesses": [1.14, 1.11, 1.09]
        }
      },
      "operators_used": ["add"]
    }
  ]
}
```

#### (선택) {dataset1}_{dataset2}_{dataset3}_simulation_results.json

Brain API 시뮬레이션 실행 시 별도 저장 (good_alpha_list.json 제외)

### G.8 메인 파이프라인 (2026-02)

```python
from combinatorial_alpha_pipeline import run_combinatorial_pipeline

# 파이프라인 실행 (한 번에 모든 단계)
summary = run_combinatorial_pipeline(
    dataset_files=DATASET_FILES,
    dataset_names=['mdl25', 'mdl30', 'mdl138'],
    operators_file=Path('operators_list.json'),
    datafields_file=Path(f'datafield/1/{REGION}/{UNIVERSE}/1_{REGION}_{UNIVERSE}_total.json'),
    output_dir=Path('results/combinatorial'),
    max_combinations=None,  # None = 전수 조합
    max_variants_per_combo=5,
    concurrency=3,
    random_seed=42,
    check_sanity=True
)
```

**단계별 흐름**:
```python
# Step 1: 0-fail seed 로드
all_alphas = load_zero_fail_alphas(DATASET_FILES)
# → {'mdl25': [33 alphas], 'mdl30': [25 alphas], 'mdl138': [29 alphas]}

# Step 2: Cartesian product 생성
seed_combinations = generate_all_combinations(
    all_alphas,
    ['mdl25', 'mdl30', 'mdl138'],
    max_combinations=None  # 33 × 25 × 29 = 23,925 조합
)

# Step 3: Variant 생성 (operators_list.json 기반)
all_variants = []
for seed_combo in seed_combinations:
    variants = generate_combination_variants(
        seed_combo,
        operators,
        max_variants_per_combo=5  # 각 조합당 5개 variant
    )
    all_variants.extend(variants)
# → 23,925 × 5 = 119,625 variants

# Step 4: Sanity check
valid_variants = []
for variant in all_variants:
    is_valid, error_msg = sanity_checker(
        variant['expression'],
        operators,
        datafields
    )
    if is_valid:
        valid_variants.append(variant)
# → 통과율 60-80% (약 70,000~90,000 variants)

# Step 5: 결과 저장
save_variants_to_txt(valid_variants, 'mdl138_mdl25_mdl30_comb.txt')
# + JSON 파일 저장

# Step 6: (선택) Brain API 시뮬레이션
# RUN_SIMULATION = True 설정 시 실행
```

### G.9 CONFIG 파라미터 (2026-02)

```python
# ========== 경로 설정 ==========
SCRIPT_DIR = Path('.').resolve()
OUTPUT_DIR = SCRIPT_DIR / "results" / "combinatorial"

# 데이터셋 파일 (확장 가능)
DATASET_FILES = {
    'mdl25': SCRIPT_DIR / 'mdl25.txt',
    'mdl30': SCRIPT_DIR / 'mdl30.txt',
    'mdl138': SCRIPT_DIR / 'mdl138.txt',
    # 'nws17': SCRIPT_DIR / 'nws17.txt',  # 추가 가능
}

# Resource 파일
OPERATORS_FILE = SCRIPT_DIR / 'operators_list.json'
REGION = "EUR"
UNIVERSE = "TOP2500"
DELAY = 1
DATAFIELDS_FILE = SCRIPT_DIR / f'datafield/{DELAY}/{REGION}/{UNIVERSE}/{DELAY}_{REGION}_{UNIVERSE}_total.json'

# ========== 조합할 데이터셋 선택 ==========
DATASETS_TO_COMBINE = ['mdl25', 'mdl30', 'mdl138']  # 2~N개 선택

# ========== 성능 제어 옵션 ==========
MAX_COMBINATIONS = None       # None = 전수 조합, 숫자 = 샘플링 개수 (예: 10000)
MAX_VARIANTS_PER_COMBO = 5    # 각 seed 조합당 생성할 variant 개수
RANDOM_SEED = 42              # 재현성을 위한 시드
CONCURRENCY = 3               # Brain API 동시 시뮬레이션 수
CHECK_SANITY = True           # Sanity check 활성화 여부

# ========== Brain 시뮬레이션 설정 ==========
RUN_SIMULATION = False        # True로 설정 시 Brain API 시뮬레이션 실행
```

**성능 제어 가이드**:
| Seed 개수 | MAX_COMBINATIONS | MAX_VARIANTS_PER_COMBO | 예상 variant 수 |
|-----------|------------------|------------------------|-----------------|
| 30×25×5 = 3,750 | None | 5 | 18,750 |
| 30×25×5 = 3,750 | 1,000 | 5 | 5,000 |
| 33×25×29 = 23,925 | None | 5 | 119,625 |
| 33×25×29 = 23,925 | 10,000 | 5 | 50,000 |

### G.10 핵심 변경사항 요약 (2026-02 vs 구버전)

| 항목 | 구버전 (GPT 기반) | 2026-02 (Combinatorial) |
|------|-------------------|------------------------|
| **Operator 제한** | add/subtract/min/max/group_zscore만 허용 | `operators_list.json` 전체 사용 가능 |
| **조합 방식** | GPT가 랜덤 조합 생성 (NUM_COMBINATIONS개) | Cartesian product 전수 조합 |
| **Variant 생성** | GPT 프롬프트 기반 | Programmatic 샘플링 (operators_list.json) |
| **Sanity Check** | 없음 (GPT 신뢰) | 엄격한 type validation (parser.py) |
| **출력 파일명** | `combined_alpha.txt` (고정) | `{dataset1}_{dataset2}_{dataset3}_comb.txt` |
| **성능 제어** | NUM_COMBINATIONS 하나만 | max_combinations, max_variants_per_combo, concurrency |
| **재현성** | GPT temperature로 제어 | random_seed 파라미터 |
| **모듈화** | 노트북 내부 함수 | `combinatorial_alpha_pipeline.py` 별도 모듈 |

### G.11 Seed Pipeline → Combination Pipeline 연결 (2026-02)

```
[Seed Pipeline Output]                [Combination Pipeline Input]
{dataset}.txt                      →  DATASET_FILES dict에 등록
  ├── mdl25.txt (33 PASS)         →  'mdl25': Path('mdl25.txt')
  ├── mdl30.txt (25 PASS)         →  'mdl30': Path('mdl30.txt')
  └── mdl138.txt (29 PASS)        →  'mdl138': Path('mdl138.txt')

parse_zero_fail_alphas_from_file() 가 0-fail 블록만 추출
→ Cartesian product 생성 (33 × 25 × 29 = 23,925 조합)
→ 각 조합당 5개 variant 생성 (→ 119,625 variants)
→ Sanity check (통과율 60-80%)
→ {mdl138_mdl25_mdl30}_comb.txt 저장
```

**완료 기준 (Acceptance Criteria)**:
1. ✅ Operator 제한(4개 강제)이 코드/프롬프트 어디에도 남아있지 않다
2. ✅ `operators_list.json`을 operator 후보 풀로 사용한다
3. ✅ 지정 dataset들의 0-fail seed 개수대로 Cartesian product 조합 수가 계산/로그에 표시된다
4. ✅ Sanity check 탈락은 simulate 대상에서 제외된다
5. ✅ 시뮬레이션 결과가 `{dataset1}_{dataset2}_{dataset3}_comb.txt`로 저장된다
6. ✅ 예시(30,25,5) 같은 경우 조합 수(=3,750) 계산이 맞게 출력된다

---

## H. LLM-Based Combinatorial Pipeline (combine_and_simulate_v2.ipynb)

### H.0 개요 (2026-02-09)

`combine_and_simulate_v2.ipynb`는 `llm_alpha_guide.ipynb`의 구조를 차용하여 LLM 기반으로
여러 dataset의 0-fail seed 알파를 조합하는 파이프라인입니다.

**기존 `combine_and_simulate.ipynb` (Cartesian Product)와의 차이:**

| 항목 | v1 (Cartesian Product) | v2 (LLM-Based) |
|------|------------------------|----------------|
| **조합 방식** | 전수 조합 (N×M×K) | LLM이 자유롭게 2~4개 seed 조합 |
| **Dataset 선택** | 모든 dataset 전수 | 랜덤 2개 선택 (batch당) |
| **Seed 추출** | 전체 0-fail seed 사용 | 랜덤 10개씩 (batch당 20개) |
| **조합 생성** | Programmatic variant | GPT-4o가 100개 조합 생성 |
| **Batch 수** | 1회 실행 | 10 batches |
| **저장 조건** | 전체 저장 | PASS (0-fail)만 저장 |

### H.1 파이프라인 흐름

```
10번의 Dataset_combine Batch
    │
    ▼
[1. Dataset 선택] 4개 중 랜덤 2개 선택
    │
    ▼
[2. Seed 추출] 각 dataset에서 0-fail 알파 랜덤 10개 (총 20개)
    │
    ▼
[3. LLM 생성] GPT-4o가 20개 seed → 100개 조합 생성 (2~4개 seed 자유 조합)
    │
    ▼
[4. Sanity Check] parser.py 기반 타입 검증
    │
    ▼
[5. Simulation] 10개씩 배치 시뮬레이션 (1-10, 11-20, ..., 91-100)
    │
    ▼
[6. Save] PASS (0-fail)만 누적 저장 (warning 포함)
```

### H.2 핵심 파라미터

```python
# Configuration
NUM_BATCHES = 10                    # 총 batch 수
SEEDS_PER_DATASET = 10              # 각 dataset에서 추출할 seed 개수
COMBINATIONS_PER_BATCH = 100        # LLM이 생성할 조합 수
SIMULATION_BATCH_SIZE = 10          # 한 번에 시뮬레이션할 알파 수
RANDOM_SEED = 42                    # 재현성을 위한 시드
CONCURRENCY = 3                     # Brain API 동시 시뮬레이션 수
GPT_MODEL = 'gpt-4o'                # LLM 모델

# Dataset 파일
DATASET_FILES = {
    'model25': 'model25.txt',
    'model30': 'model30.txt',
    'model138': 'model138.txt',
    'analyst39': 'analyst39.txt',
}
```

### H.3 LLM 프롬프트 개선 사항

v2에서는 sanity check 통과율을 높이기 위해 프롬프트를 크게 개선했습니다:

#### 1. 전체 Operator 시그니처 제공

```
[Arithmetic]
  - add(x, y, filter = false), x + y
  - abs(x)
  - log(x)
  - subtract(x, y, filter=false), x - y
  ...
```

#### 2. SYNTAX_RULES 섹션

```
1. Binary operators (EXACTLY 2 args): subtract(a, b), divide(a, b), min(a, b), max(a, b)
2. Unary operators (EXACTLY 1 arg): rank(x), zscore(x), abs(x), log(x), sign(x)
3. Variadic operators (2+ args): add(a, b, ...) can take multiple inputs
4. Time-series (2 args): ts_mean(x, days), ts_std(x, days), ts_delta(x, days)
5. group_zscore(x, group) - EXACTLY 2 args: expression and grouping
```

#### 3. COMMON MISTAKES 섹션

```
- rank(a, b) is WRONG - rank takes only 1 argument
- zscore(a, b) is WRONG - zscore takes only 1 argument
- subtract(a, b, c) is WRONG - subtract takes exactly 2 arguments
- winsorize(x, lower, upper) needs 3 args, not 1
```

### H.4 결과 매칭 로직

v2는 시뮬레이션 결과를 expression으로 매칭합니다 (인덱스 매칭 대신):

```python
# Build expression → combo mapping
expr_to_combo = {combo['expression']: combo for combo in sim_batch}

# Match results by original expression from simulate_data
for result in results:
    original_expr = result.get('simulate_data', {}).get('regular', '')
    combo = expr_to_combo.get(original_expr)
```

**이유**: `ace.simulate_alpha_list_multi`가 `imap_unordered`를 사용하여 결과 순서가 입력 순서와 다를 수 있음.

### H.5 디버그 로깅

각 시뮬레이션 배치마다 상세 로그 출력:

```
[DEBUG] Results: 10 total, 7 valid
[DEBUG] abc123: PASS=5, FAIL=2 ['LOW_SHARPE', 'LOW_FITNESS']
[DEBUG] def456: PASS=7, FAIL=0 []
[PASS] #1 Sharpe=2.15 Fitness=1.52
```

### H.6 Resume 지원

- `gen_json/combinatorial_batch_{idx}.json`: 각 batch의 LLM 생성 결과 저장
- 재실행 시 이미 완료된 batch는 자동 스킵

### H.7 출력 파일

#### results/combinatorial/combined_alphas.txt

PASS (0-fail) 알파만 누적 저장:

```
--- #1 | FAIL: 0 (PASS) ---
ID: abc123xyz
Region: EUR, Universe: TOP2500
Sharpe: 2.15, Fitness: 1.52, Turnover: 0.1234
Expression: rank(add(ts_zscore(mdl25_field1, 63), quantile(mdl138_field2)))
Source Datasets: model25, model138
Batch: 3
Status: PASS
```

### H.8 알려진 이슈 및 해결책

| 이슈 | 원인 | 해결책 |
|------|------|--------|
| Sanity check 실패율 높음 | LLM이 잘못된 operator 시그니처 사용 | SYNTAX_RULES, COMMON MISTAKES 프롬프트 추가 |
| 0개 PASS 알파 저장 | is_checks 대신 is_tests 사용해야 함 | `result.get('is_tests', ...)` 사용 |
| 결과-표현식 불일치 | 인덱스 매칭 오류 | `expr_to_combo` 딕셔너리로 expression 매칭 |
| operators_list.json input이 `[]` | 타입 정보 누락 | Brain API에서 최신 operator 정보 sync 필요 |

### H.9 v1 vs v2 선택 가이드

| 상황 | 권장 파이프라인 |
|------|---------------|
| 소수 dataset (2~3개)의 전수 조합 필요 | v1 (Cartesian Product) |
| 대량 dataset에서 다양한 조합 탐색 | v2 (LLM-Based) |
| API 비용 절감 (GPT 호출 없음) | v1 |
| 창의적인 조합 패턴 필요 | v2 |
| 재현성이 중요한 경우 | v1 (deterministic) |

---

## 부록: 파일 구조 요약 (2026-02-09)

```
llm_alpha_gen/
├── llm_functions.py                # LLM 호출, seed-alpha 프롬프트, JSON 파싱
├── parser.py                       # Expression → Tree 변환, 타입 체크
├── ace_lib.py                      # Brain API 연동, 시뮬레이션
├── AAF.py                          # 데이터필드 초기화 (whitelist 관리)
├── my_research.py                  # Sanity checker 구현 (return_type, check_input)
├── combinatorial_alpha_pipeline.py # 전수 조합 파이프라인 핵심 로직 (v1용)
│
├── llm_alpha_guide.ipynb           # [Pipeline A] Seed-alpha 생성
├── combine_and_simulate.ipynb      # [Pipeline G] Combinatorial v1 (Cartesian Product)
├── combine_and_simulate_v2.ipynb   # [Pipeline H] Combinatorial v2 (LLM-Based) ★ NEW
├── refine.ipynb                    # [Pipeline F] 진화 알고리즘 (알파 개선)
├── .claude/PIPELINE_GUIDE.md       # 이 문서
│
├── datafield/                      # 데이터필드 메타정보 캐시
│   └── 1/{REGION}/{UNIVERSE}/
│       ├── 1_{REGION}_{UNIVERSE}_{dataset}.json
│       └── 1_{REGION}_{UNIVERSE}_total.json   # sanity_checker용
│
├── gen_json/                       # LLM 생성 raw JSON (resume용)
│   ├── risk60_0.json               # Seed Pipeline resume
│   ├── model25_1.json
│   └── combinatorial_batch_{idx}.json  # v2 Pipeline resume ★ NEW
│
├── {dataset}.txt                   # Seed-alpha 시뮬 결과 (per-dataset, 정렬)
│   ├── model25.txt                 #   → Combination Pipeline 입력
│   ├── model30.txt
│   ├── model138.txt
│   └── analyst39.txt
│
├── results/                        # 파이프라인 출력 디렉토리
│   └── combinatorial/
│       ├── mdl138_mdl25_mdl30_comb.txt           # v1: Variant 목록
│       ├── mdl138_mdl25_mdl30_comb.json          # v1: Variant 목록 (JSON)
│       └── combined_alphas.txt                   # v2: PASS 알파 누적 ★ NEW
│
├── operators_list.json             # Operator I/O 타입 정의
└── operator_inputs.json            # 추가 operator 타입 정보
```
