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

---

## 개요

이 파이프라인은 LLM(Large Language Model)을 활용하여 WorldQuant Brain 플랫폼에서 사용할 수 있는 알파(Alpha) 아이디어를 대량으로 생성하고, 검증하여 제출 가능한 후보를 선별하는 자동화 시스템입니다.

**세 가지 주요 파이프라인:**
| 파이프라인 | 노트북 | 목적 | 특징 |
|-----------|--------|------|------|
| **Seed Generation** | `llm_alpha_guide.ipynb` | 단순 seed-alpha 대량 탐색 | 70-80% 패턴 + 20-30% 자유 조합, 구조 필터링, .txt 출력 |
| **Combination** | `combine_and_simulate.ipynb` | 다중 데이터셋 seed 조합 | 0-fail seed만 사용, GPT로 cross-dataset 조합, Anti-PowerPool |
| **Refinement** | `refine.ipynb` | 기존 알파 개선 | 진화 알고리즘, 토너먼트 선택 |

### 전체 흐름도 (Seed → Combine → Refine)

```
[Pipeline A: Seed Generation]
  llm_alpha_guide.ipynb
  데이터셋별 단순 seed-alpha 대량 생성
  → {dataset}.txt (per-dataset 결과)
          │
          ▼
[Pipeline G: Combination]
  combine_and_simulate.ipynb
  0-fail seed들을 cross-dataset 조합
  → combined_alpha.txt + good_alpha_list.json
          │
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

### Combination Pipeline 상세 흐름

```
[Step 1: 로드]        [Step 2: GPT 조합]      [Step 3: 검증]     [Step 4: 저장]     [Step 5: 시뮬]
{dataset}.txt      → create_combination   → validate_       → combined_       → Brain API
0-fail 알파 추출      _prompt()              combination()     alpha.txt          simulate
(parse_dataset_     call_gpt_for_          detect_datasets   save_combined_     simulate_
 file)              combinations_           _used()           alphas_to_txt()    combinations()
                    _streaming()                                                → good_alpha_
                                                                                 list.json
```

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

## G. Cross-Dataset Alpha Combination (combine_and_simulate.ipynb)

### G.0 개요

`combine_and_simulate.ipynb`는 Seed Generation 파이프라인에서 생성된 **0-fail seed 알파**들을
서로 다른 데이터셋 간에 GPT를 통해 조합하여 복합 알파를 만들고, Brain API로 시뮬레이션합니다.

**핵심 아이디어:**
- 단일 dataset의 seed 알파는 단순하지만, 서로 다른 dataset의 seed를 결합하면 다양한 신호를 합성
- Anti-PowerPool 구조를 갖도록 농도 분산(concentration mitigation) 레이어 필수
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

### G.3 GPT 조합 프롬프트 구조

`create_combination_prompt()`가 생성하는 프롬프트의 핵심 구조:

```
[역할] Quantitative alpha researcher (Non-PowerPool 알파 생성)

[데이터셋별 0-fail 알파 목록]
=== MDL25 Dataset ===
  1. quantile(mdl25_smartestimate_f12m_eps)  (Sharpe: 1.83, Fitness: 1.14)
  2. ...
=== MDL30 Dataset ===
  1. quantile(mdl30_new_psprise_pct_fy1_eps) (Sharpe: 1.71, Fitness: 1.11)
  2. ...

[MANDATORY STRUCTURAL GUARDRAILS]
최종 Expression 형태:
  [ Normalization ] → [ Random Combination ] → [ Concentration Mitigation ]

A) 원본 alpha는 이미 rank()/zscore()/quantile() 정규화 포함
B) 최종 expression에 concentration-mitigation 레이어 필수:
   group_zscore(), winsorize(), tail() 등
C) 안정성: 앙상블 스타일 조합 선호 (add, min/max gating)

[COMBINATION RULES]
1. 허용 Operator: add, subtract, min, max, group_rank, group_zscore
2. 각 dataset에서 정확히 1개의 alpha expression 선택
3. Operator 자유 중첩 가능: add(alpha1, subtract(alpha2, alpha3))
4. 원본 alpha 내부 구조 수정 금지 (그대로 사용)
5. 유효한 FASTEXPR 문법
6. PowerPool / Pure PowerPool 금지

[ANTI-FAIL HEURISTICS]
- 극단 tail 및 랭킹 집중 감소
- 소수 종목 집중 방지
- IS ladder 안정성 향상을 위한 다중 신호 결합
- 부드럽고 분산된 expression 선호

[OUTPUT FORMAT]
{
  "combinations": [
    {
      "expression": "FINAL_FASTEXPR_EXPRESSION",
      "datasets_used": ["mdl25", "mdl30", "mdl138"],
      "idea": "조합 직관",
      "rationale_data": "데이터셋 보완 이유",
      "rationale_operators": "Operator 선택 이유"
    }
  ]
}
```

### G.4 GPT 호출 방식

```python
def call_gpt_for_combinations_streaming(prompt, model="gpt-4o-mini"):
    # OpenAI API 스트리밍 호출
    # temperature=0.7 (다양성)
    # max_tokens=8000
    # ```json ... ``` 블록 자동 처리
    # 결과: List[Dict] (combinations)
```

**Seed Pipeline과의 차이점:**
| 항목 | Seed (llm_functions.py) | Combination (combine_and_simulate) |
|------|------------------------|-------------------------------------|
| API | `call_llm_stream()` | `call_gpt_for_combinations_streaming()` |
| System prompt | JSON 전용 | 역할 + JSON 전용 |
| temperature | 기본값 | 0.7 |
| max_tokens | 기본값 | 8000 |
| JSON 파싱 | `cut_first_to_last_brace()` | `re.search` + `json.loads` |

### G.5 조합 검증

```python
def validate_combination(combination):
    """모든 조합 허용 (현재 permissive)"""
    expression = combination.get('expression', '')
    datasets_used = detect_datasets_used(expression)
    operator_count = count_operators(expression)
    return True, f"Uses {len(datasets_used)} datasets, {operator_count} operators"

def detect_datasets_used(expression):
    """expression에서 사용된 dataset prefix 감지"""
    # mdl25, mdl30, mdl138, nws17, star_eps 등 패턴 매칭
```

> **참고**: 현재 `validate_combination()`은 항상 True를 반환합니다 (모든 조합 허용).
> 필요 시 최소 2개 dataset 사용, operator 수 제한 등의 필터를 추가할 수 있습니다.

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

### G.7 출력 파일

#### combined_alpha.txt

```
================================================================================
COMBINED ALPHAS (Cross-Dataset, Non-PowerPool)
Generated: 2026-02-07 15:30:00
Total: 50 combinations
================================================================================

--- #1 | VALID | Datasets: 3, Operators: 5 ---
Expression: group_zscore(add(quantile(mdl25_...), quantile(mdl30_...)), industry)
Datasets Used: mdl25, mdl30, mdl138
Validation: Uses 3 datasets (mdl30, mdl138, mdl25), 5 operators
Idea: Diversified earnings quality signal with industry normalization
Data Rationale: mdl25 provides quality, mdl30 adds momentum, mdl138 adds AI factors
Operator Rationale: group_zscore reduces concentration across industries

================================================================================
Summary: 49/50 valid combinations
================================================================================
```

#### good_alpha_list.json

시뮬레이션 결과가 `good_alpha_list.json`에 append됩니다:

```json
[
  {
    "timestamp": "2026-02-07T15:45:00",
    "alpha_id": "AbCdEfGh",
    "expression": "group_zscore(add(quantile(mdl25_...), quantile(mdl30_...)), industry)",
    "is_sharpe": 1.85,
    "is_fitness": 1.12,
    "is_turnover": 0.15,
    "settings": { "region": "EUR", "universe": "TOP2500", ... },
    "checks": [
      { "name": "LOW_SHARPE", "result": "PASS" },
      { "name": "CONCENTRATED_WEIGHT", "result": "WARNING" }
    ],
    "combination_info": {
      "expression": "...",
      "datasets_used": ["mdl25", "mdl30"],
      "idea": "...",
      "rationale_data": "...",
      "rationale_operators": "...",
      "description": "Idea: ... \nRationale for data used: ..."
    },
    "source": "combine_and_simulate"
  }
]
```

### G.8 메인 파이프라인 5-Step

```python
# Step 1: 데이터셋 로드
all_alphas = load_all_zero_fail_alphas()
# → {'mdl25': [33 alphas], 'mdl30': [20 alphas], 'mdl138': [29 alphas]}

# Step 2: GPT로 조합 생성
combinations = generate_combined_alphas(all_alphas, NUM_COMBINATIONS, GPT_MODEL)
# → 100개 조합 생성

# Step 3: 조합 검증
for combo in combinations:
    is_valid, reason = validate_combination(combo)
# → valid / invalid 분류

# Step 4: txt 파일로 저장
save_combined_alphas_to_txt(combinations, OUTPUT_FILE)
# → combined_alpha.txt

# Step 5: Brain API 시뮬레이션
session = ace.start_session()
simulation_results = simulate_combinations(session, combinations)
saved_entries = save_simulation_results(simulation_results, GOOD_ALPHA_FILE)
# → good_alpha_list.json
```

### G.9 CONFIG 파라미터

```python
# 경로 설정
OUTPUT_FILE = SCRIPT_DIR / "combined_alpha.txt"
GOOD_ALPHA_FILE = SCRIPT_DIR / "good_alpha_list.json"

# 데이터셋 파일 (런타임 추가 가능)
DATASET_FILES = {
    'mdl25': SCRIPT_DIR / 'mdl25.txt',
    'mdl30': SCRIPT_DIR / 'mdl30.txt',
    'mdl138': SCRIPT_DIR / 'mdl138.txt',
}

# 시뮬레이션 설정
REGION = "EUR"
UNIVERSE = "TOP2500"
DELAY = 1

# 파이프라인 파라미터
NUM_COMBINATIONS = 100       # 생성할 조합 개수
GPT_MODEL = "gpt-4o-mini"   # GPT 모델
RUN_SIMULATION = True        # Brain API 시뮬레이션 실행 여부
```

### G.10 런타임 데이터셋 추가

```python
# 새 dataset.txt 추가
add_dataset_file('risk60', './risk60.txt')
add_dataset_file('nws17', './nws17.txt')

# 현재 등록된 파일 확인
list_dataset_files()
```

### G.11 Seed Pipeline → Combination Pipeline 연결

```
[Seed Pipeline Output]              [Combination Pipeline Input]
{dataset}.txt                    →   DATASET_FILES dict에 등록
  ├── mdl25.txt (33 PASS)       →   'mdl25': Path('mdl25.txt')
  ├── mdl30.txt (20 PASS)       →   'mdl30': Path('mdl30.txt')
  └── mdl138.txt (29 PASS)      →   'mdl138': Path('mdl138.txt')

parse_dataset_file() 가 0-fail 블록만 추출
→ GPT에 각 dataset별 최대 10개 알파 전달
→ cross-dataset 조합 생성 + 시뮬레이션
```

---

## 부록: 파일 구조 요약

```
llm_alpha_gen/
├── llm_functions.py         # LLM 호출, seed-alpha 프롬프트, JSON 파싱
├── parser.py                # Expression → Tree 변환, 타입 체크
├── ace_lib.py               # Brain API 연동, 시뮬레이션
├── AAF.py                   # 데이터필드 초기화 (whitelist 관리)
├── llm_alpha_guide.ipynb    # [Pipeline A] Seed-alpha 생성
├── combine_and_simulate.ipynb # [Pipeline G] Cross-dataset 조합
├── refine.ipynb             # [Pipeline F] 진화 알고리즘 (알파 개선)
├── PIPELINE_GUIDE.md        # 이 문서
│
├── datafield/               # 데이터필드 메타정보 캐시
│   └── 1/{REGION}/{UNIVERSE}/
│       ├── 1_{REGION}_{UNIVERSE}_{dataset}.json
│       └── 1_{REGION}_{UNIVERSE}_total.json   # sanity_checker용
│
├── gen_json/                # LLM 생성 raw JSON (Seed Pipeline resume용)
│   ├── risk60_0.json
│   └── model25_1.json
│
├── {dataset}.txt            # Seed-alpha 시뮬 결과 (per-dataset, 정렬)
│   ├── risk60.txt           #   → Combination Pipeline 입력
│   ├── mdl25.txt
│   ├── mdl30.txt
│   └── mdl138.txt
│
├── combined_alpha.txt       # Combination Pipeline 조합 결과 (txt)
├── good_alpha_list.json     # Combination Pipeline 시뮬 결과 (json)
│
├── operators_list.json      # operator I/O 타입 정의 (FREE_PATTERN용으로도 사용)
└── operator_inputs.json     # 추가 operator 타입 정보
```
