# WorldQuant Brain + LLM 알파 생성 파이프라인 가이드

## 목차
1. [개요](#개요)
2. [A. 알파 생성 단계](#a-알파-생성-단계)
3. [B. Sanity Check 단계](#b-sanity-check-단계)
4. [C. Brain Simulation 단계](#c-brain-simulation-단계)
5. [D. 제출 가능성 판단 단계](#d-제출-가능성-판단-단계)
6. [E. 사람이 눈으로 확인하는 포인트](#e-사람이-눈으로-확인하는-포인트)

---

## 개요

이 파이프라인은 LLM(Large Language Model)을 활용하여 WorldQuant Brain 플랫폼에서 사용할 수 있는 알파(Alpha) 아이디어를 대량으로 생성하고, 검증하여 제출 가능한 후보를 선별하는 자동화 시스템입니다.

### 전체 흐름도

```
[데이터필드 초기화]          [알파 생성]              [검증]                [결과 분석]
      │                        │                      │                      │
AAF.initiate_datafield() → LLM 호출 → gen_json/*.json → sanity_checker() → Brain API 시뮬레이션 → results/*.csv
      │                        │                      │                      │
datafield/{delay}/        generate_expressions     parser.tree_node()   multi_simulate_        사람이 판단
{region}/{universe}/      _from_dataset()          check_input()        alphas_map()
*.json + total.json
```

---

## A. 알파 생성 단계

### A.1 입력 데이터 구성

알파 생성 시 LLM에게 전달되는 입력은 크게 세 가지입니다:

#### 1) 데이터필드 목록 (`data_fields_inputs`)
```python
# llm_functions.py:66-74
data_fields = ace.get_datafields(s, region=alpha_region, universe=alpha_universe,
                                  dataset_id=dataset_id, data_type='ALL')
data_fields = data_fields[data_fields['type'] != "GROUP"]

keys = ['id','description','subcategory','coverage','userCount','alphaCount']
data_fields_inputs = [{k:d[k] for k in keys if k in d} for d in data_fields_list]
```

| 필드 | 의미 | 예시 |
|------|------|------|
| `id` | 데이터필드 고유 ID | `rsk70_mfm2_usfast_momentum` |
| `description` | 필드 설명 | "Momentum factor loading" |
| `subcategory` | 하위 카테고리 | "Risk Model Factors" |
| `coverage` | 커버리지 (0~1) | 0.95 |
| `userCount` | 사용 유저 수 | 1500 |
| `alphaCount` | 제출된 알파 수 | 200 |

#### 2) 오퍼레이터 목록 (`operators_list`)
```python
# llm_functions.py:80-83
operators = ace.get_operators(s)
operators_list = [row.to_dict() for row in operators[operators['scope']=='REGULAR'].iloc]
operator_exclude = ['rank']  # rank는 LLM이 과도하게 사용하므로 제외
```

#### 3) 프롬프트 구조
```
<MISSION>
  - {alpha_num}개의 분산된 알파 아이디어 제안
  - idea, description, implementation, confidence_level 4가지 파트로 구분
</MISSION>

<SUGGESTIONS>
  - SUGGESTION1: 여러 데이터필드를 적극 활용
  - SUGGESTION2: VECTOR 타입은 vec_avg() 또는 vec_sum()으로 감싸기
  - SUGGESTION3: coverage < 0.6이면 ts_backfill() 사용 고려
  - SUGGESTION4: userCount/alphaCount 높은 필드 우선 사용
</SUGGESTIONS>

<KEEP_IN_MIND>
  - implementation이 너무 길면 안 됨
  - operator 7개, datafield 2개 이하
  - 데이터필드 목록 외 필드 사용 불가
  - GROUP 타입은 단독 사용 불가
</KEEP_IN_MIND>

<OPERATORS>...</OPERATORS>
<DATA>...</DATA>
<ANSWER_FORMAT>JSON 형식 지정</ANSWER_FORMAT>
```

### A.2 LLM 출력 JSON 구조

```json
{
  "results": [
    {
      "idea": "Momentum x Value divergence",
      "description": "SUGGESTION1 & S4 applied. Combine momentum and value factors...",
      "implementation": "multiply(zscore(rsk70_mfm2_usfast_momentum), zscore(rsk70_mfm2_usfast_value))",
      "confidence_level": 0.72
    },
    ...
  ]
}
```

| 필드 | 의미 | 활용 |
|------|------|------|
| `idea` | 알파의 핵심 아이디어 (1줄 요약) | 결과 검토 시 빠른 이해용 |
| `description` | SUGGESTIONS/KEEP_IN_MIND 적용 근거 설명 | 알파 로직의 이론적 배경 파악 |
| `implementation` | 실제 실행 가능한 FASTEXPR 수식 | Brain 시뮬레이션에 직접 입력 |
| `confidence_level` | LLM이 판단한 신뢰도 (0~1) | 우선순위 정렬, 필터링 기준 |

### A.3 저장 경로 규칙

```
gen_json/
├── {dataset_id}_{batch_index}.json
├── risk70_0.json      # risk70 데이터셋, 첫 번째 배치
├── risk70_1.json      # risk70 데이터셋, 두 번째 배치
├── fundamental6_0.json
└── ...
```

**배치 수 결정 로직** (`datset_to_simnum` 함수):
```python
score = int(np.log10(userCount + 1) + fieldCount)

if score <= 3:    return 1   # 100개 알파
elif score <= 10: return 2   # 200개 알파
elif score <= 50: return 3   # 300개 알파
elif score <= 100: return 10 # 1,000개 알파
else:             return 20  # 2,000개 알파
```

### A.4 LLM 출력이 깨졌을 때 (`cut_first_to_last_brace`)

```python
# llm_functions.py:147-152
def cut_first_to_last_brace(s: str) -> str | None:
    start = s.find('{')
    end = s.rfind('}')
    if start == -1 or end == -1 or start > end:
        return None
    return s[start:end+1]
```

**발생할 수 있는 현상:**
- LLM이 JSON 앞뒤에 불필요한 텍스트를 붙인 경우 → 정상 파싱
- JSON이 중간에 끊긴 경우 → `json.loads()` 실패 → 해당 배치 스킵
- `{` 또는 `}` 자체가 없는 경우 → `None` 반환 → 해당 배치 스킵

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

#### `return_type` 함수의 타입 결정:

| 조건 | 반환 타입 |
|------|----------|
| `node_type == 'operator'` | `operators[node.value]['output']` (연산자 정의에서 가져옴) |
| `node_type == 'number'` | `"NUMBER"` |
| `node_type == 'datafield'` (값이 'nan') | `"MATRIX"` |
| `node_type == 'datafield'` | `datafields[node.value]['type']` (MATRIX/VECTOR) |
| `node_type == 'special_argument'` | `"SPECIAL_ARGUMENT"` (무시됨) |

#### `check_input` 함수의 매칭 규칙:

```python
def check_input(operator_inputs, children_types):
    # operator_inputs: operator가 요구하는 타입 리스트
    # children_types: 실제 자식 노드들의 타입 리스트

    # 특수 케이스: MATRIX 자리에 NUMBER가 들어와도 허용
    if operator_input == "MATRIX" and children_type == "NUMBER":
        pass  # OK
    # 그 외에는 정확히 일치해야 함
```

### B.4 Sanity 통과했지만 시뮬레이션 실패하는 케이스

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
# ace_lib.py:153-196
def start_session() -> SingleSession:
    s = SingleSession()                              # 싱글톤 세션
    s.auth = get_credentials()                       # ~/secrets/platform-brain.json
    r = s.post(brain_api_url + "/authentication")

    if r.status_code == 401:
        if r.headers["WWW-Authenticate"] == "persona":
            # Biometrics 인증 필요 → URL 출력 후 사용자 입력 대기
            print("Complete biometrics authentication...")
            input()
            # 인증 완료될 때까지 반복
        else:
            # 비밀번호 오류 → credentials 파일 초기화 후 재시도
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
# ace_lib.py:219-307
simulation_data = {
    "type": "REGULAR",
    "settings": {
        "instrumentType": "EQUITY",
        "region": "USA",
        "universe": "TOP3000",
        "delay": 1,
        "decay": 0,
        "neutralization": "INDUSTRY",
        "truncation": 0.08,
        "pasteurization": "ON",
        "testPeriod": "P0Y0M0D",
        "unitHandling": "VERIFY",
        "nanHandling": "OFF",
        "maxTrade": "OFF",
        "language": "FASTEXPR",
        "visualization": False
    },
    "regular": "multiply(zscore(field1), zscore(field2))"  # implementation
}
```

| 설정 | 의미 | 기본값 |
|------|------|--------|
| `neutralization` | 중립화 방식 | INDUSTRY (섹터 중립) |
| `truncation` | 극단값 절사 비율 | 0.08 (8%) |
| `pasteurization` | 미래 데이터 사용 방지 | ON |
| `decay` | 알파 감쇠 기간 | 0 (감쇠 없음) |

### C.3 병렬 시뮬레이션 (`multi_simulate_alphas_map`)

```python
# ace_lib.py:1450-1465
def multi_simulate_alphas_map(s, alpha_list, tags_list, descs_list, num_threads):
    session_list = [s] * num_threads  # 동일 세션 공유

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = executor.map(
            simulate_single_alpha_result,
            session_list,
            alpha_list,
            tags_list,
            descs_list
        )
    return results
```

**노트북에서의 배치 처리:**
```python
# 8개씩 배치로 시뮬레이션
for rep in range(0, len_alpha_list, 8):
    batch_end = min(rep + 8, len_alpha_list)
    sim_results = list(ace.multi_simulate_alphas_map(
        s,
        alpha_list[rep:batch_end],
        tags_list[rep:batch_end],
        descs_list[rep:batch_end],
        batch_end - rep
    ))
```

### C.4 시뮬레이션 상태 진행

```python
# ace_lib.py:358-418 (simulation_progress)
1. start_simulation() 호출 → simulate_response
2. simulate_response.headers["Location"]에서 progress URL 획득
3. 반복:
   - GET progress URL
   - headers["Retry-After"] > 0 이면 대기
   - "status" == "ERROR" 이면 실패
   - "alpha" 필드가 있으면 완료
4. get_simulation_result_json(s, alpha_id) 호출
5. 최종 결과 반환
```

### C.5 결과 JSON 핵심 필드

Brain API가 반환하는 결과에서 중요한 필드:

```json
{
  "id": "rKQWRbQ3",
  "type": "REGULAR",
  "settings": {...},
  "regular": {"code": "multiply(...)"},
  "is": {
    "pnl": 0.15,
    "sharpe": 1.82,
    "fitness": 1.25,
    "turnover": 0.45,
    "returns": 0.12,
    "drawdown": -0.08,
    "margin": 0.02,
    "checks": [
      {"name": "LOW_SHARPE", "result": "PASS", "limit": 1.58, "value": 1.82},
      {"name": "LOW_FITNESS", "result": "PASS", "limit": 1.0, "value": 1.25},
      ...
    ]
  },
  "train": {...},
  "test": {...}
}
```

| 섹션 | 의미 |
|------|------|
| `is` | In-Sample 기간 (2014~2023) 성과 |
| `train` | 학습 기간 성과 |
| `test` | 테스트 기간 성과 |
| `checks` | 제출 요건 충족 여부 |

---

## D. 제출 가능성 판단 단계

> **중요**: 현재 코드는 자동 제출을 하지 않습니다. 결과 지표를 기반으로 사람이 판단합니다.

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

```python
# ace_lib.py:624-662 - 자기상관 체크
def check_self_corr_test(s, alpha_id, threshold=0.7):
    # 자신이 제출한 다른 알파와의 상관관계
    # value < 0.7 이어야 PASS

# ace_lib.py:545-583 - 프로덕션 상관 체크
def check_prod_corr_test(s, alpha_id, threshold=0.7):
    # 이미 프로덕션에 있는 알파와의 상관관계
    # value <= 0.7 이어야 PASS
```

| 상관관계 | 의미 | 기준 |
|---------|------|------|
| `SELF_CORRELATION` | 내가 이전에 제출한 알파와 얼마나 비슷한가 | < 0.7 |
| `PROD_CORRELATION` | 현재 운용 중인 알파와 얼마나 비슷한가 | ≤ 0.7 |

### D.3 선호 조건 (Soft Filter)

모든 필수 조건을 통과한 후, 다음 기준으로 우선순위 결정:

| 지표 | 좋은 값 | 설명 |
|------|---------|------|
| `fitness` | 높을수록 좋음 (1.5+) | Sharpe와 Turnover를 종합한 지표 |
| `sharpe` | 높을수록 좋음 (2.0+) | 위험 대비 수익 |
| `turnover` | 0.1~0.5 범위 | 너무 낮거나 높지 않은 적정 거래 |
| `returns` | 양수 | 절대 수익률 |
| `drawdown` | 0에 가까울수록 좋음 | 최대 손실폭 |

### D.4 추천 판단 루브릭

**1순위 (강력 추천):**
- 모든 checks PASS
- fitness ≥ 1.5
- sharpe ≥ 2.0
- 0.1 ≤ turnover ≤ 0.5
- self_corr < 0.5

**2순위 (검토 대상):**
- 모든 checks PASS
- fitness ≥ 1.2
- sharpe ≥ 1.7
- 0.05 ≤ turnover ≤ 0.6

---

## E. 사람이 눈으로 확인하는 포인트

### E.1 results/*.csv 핵심 컬럼 (상위 15개)

`flatten_dict` 함수로 평탄화된 결과에서 확인해야 할 컬럼:

| # | 컬럼명 | 의미 | 좋은 값 | 나쁜 값 | 주의사항 |
|---|--------|------|---------|---------|----------|
| 1 | `id` | 알파 고유 ID | 존재함 | None/빈값 | None이면 시뮬 실패 |
| 2 | `regular-code` | 실제 수식 | 유효한 expression | - | Brain에서 확인 가능 |
| 3 | `is-sharpe` | IS Sharpe Ratio | ≥ 1.58 (PASS) | < 1.58 | **가장 중요한 지표** |
| 4 | `is-fitness` | IS Fitness | ≥ 1.0 (PASS) | < 1.0 | Sharpe×√(Turnover) 기반 |
| 5 | `is-turnover` | IS 회전율 | 0.01~0.7 | 범위 밖 | 너무 낮으면 거래 없음 |
| 6 | `is-returns` | IS 수익률 | 양수 | 음수 | 절대 성과 |
| 7 | `is-drawdown` | IS 최대낙폭 | 0에 가까움 | 큰 음수 | 리스크 지표 |
| 8 | `is-checks-0-result` | LOW_SHARPE 결과 | PASS | FAIL | 제출 필수 조건 |
| 9 | `is-checks-1-result` | LOW_FITNESS 결과 | PASS | FAIL | 제출 필수 조건 |
| 10 | `is-checks-2-result` | LOW_TURNOVER 결과 | PASS | FAIL | 제출 필수 조건 |
| 11 | `is-checks-3-result` | HIGH_TURNOVER 결과 | PASS | FAIL | 제출 필수 조건 |
| 12 | `is-checks-4-result` | CONCENTRATED_WEIGHT | PASS | FAIL | 제출 필수 조건 |
| 13 | `settings-region` | 지역 | USA/EUR/ASI/GLB | - | 데이터셋 호환성 확인 |
| 14 | `settings-universe` | 유니버스 | TOP3000 등 | - | 데이터셋 호환성 확인 |
| 15 | `tags-1` | confidence 태그 | confidence_0.7+ | confidence_0.3- | LLM 신뢰도 참고용 |

### E.2 gen_json/*.json 확인 포인트

| 항목 | 확인 방법 | 권장 기준 |
|------|----------|----------|
| **implementation 길이** | `len(impl)` | 200자 이내 |
| **datafield 사용 개수** | expression 내 datafield 카운트 | 1~2개 |
| **operator 개수** | `tree_node(impl).collect_all_nodes()` 중 operator 수 | 7개 이하 |
| **confidence_level** | JSON 필드 직접 확인 | 0.6 이상 우선 검토 |
| **VECTOR 타입 처리** | `vec_avg()` 또는 `vec_sum()` 사용 여부 | VECTOR 필드는 반드시 감싸기 |

### E.3 결과 검토 권장 순서

```
1. id 컬럼 확인 → None이면 시뮬 실패, 스킵

2. is-checks-*-result 전체 확인
   → 하나라도 FAIL이면 제출 불가

3. is-sharpe / is-fitness 정렬
   → 높은 순으로 검토

4. is-turnover 범위 확인
   → 극단값 필터링 (< 0.01 또는 > 0.7)

5. regular-code 복잡도 확인
   → 너무 복잡하면 오버피팅 의심

6. 동일 데이터셋 내 유사 알파 비교
   → 중복 제거
```

---

## 부록: 파일 구조 요약

```
llm_alpha_gen/
├── llm_functions.py      # LLM 호출, JSON 파싱
├── parser.py             # Expression → Tree 변환, 타입 체크
├── ace_lib.py            # Brain API 연동, 시뮬레이션
├── AAF.py                # 데이터필드 초기화 (whitelist 관리)
├── helpful_functions.py  # 결과 정리, prettify
├── llm_alpha_guide.ipynb # 메인 실행 노트북
│
├── datafield/            # 데이터필드 메타정보 캐시
│   └── 1/USA/TOP3000/
│       ├── 1_USA_TOP3000_risk70.json
│       └── 1_USA_TOP3000_total.json  # sanity_checker용
│
├── gen_json/             # LLM 생성 알파 후보
│   ├── risk70_0.json
│   └── fundamental6_19.json
│
└── results/              # 시뮬레이션 결과
    ├── USA_TOP3000_risk.csv
    └── USA_TOP3000_fundamental.csv
```
