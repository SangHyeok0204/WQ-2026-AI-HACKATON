# Alpha Combiner and Simulator

mdl25와 mdl138의 FAIL:0 알파들을 결합하여 새로운 알파를 생성하고 Brain API로 시뮬레이션합니다.

## 개요

- **입력**: `mdl25.txt` (FAIL:0 알파 32개), `mdl138.txt` (FAIL:0 알파 29개)
- **조합**: 32 x 29 x 3 operators x 2 (with/without group_zscore) = 최대 5,568개
- **출력**: `outputs/results.jsonl`

## 허용 연산자

1. `add(a, b)` - 두 알파의 합
2. `min(a, b)` - 두 알파 중 작은 값
3. `max(a, b)` - 두 알파 중 큰 값
4. `group_zscore(x, sector)` - 섹터별 z-score 정규화 (선택적)

## 환경 설정

### 1. 환경변수 (.env 또는 시스템 환경변수)

```bash
# Brain API 인증 정보 (선택)
BRAIN_CREDENTIAL_EMAIL=your@email.com
BRAIN_CREDENTIAL_PASSWORD=your_password

# 또는 ~/secrets/platform-brain.json 파일에 저장됨
```

### 2. 의존성

```bash
pip install requests pandas tqdm
```

## 실행 방법

### 기본 실행 (전체 조합)

```bash
cd llm_alpha_gen
python combine_and_simulate.py
```

### 옵션

```bash
# 시뮬레이션 없이 조합만 확인 (dry-run)
python combine_and_simulate.py --dry-run

# group_zscore 변형 제외 (조합 수 절반)
python combine_and_simulate.py --skip-zscore

# 처음 100개만 실행
python combine_and_simulate.py --limit 100

# 옵션 조합
python combine_and_simulate.py --skip-zscore --limit 50
```

## 체크포인트 & 재시작

- 실행 중 중단되어도 `outputs/checkpoint.json`에 진행 상황 저장
- 재실행 시 이미 완료된 조합은 자동으로 스킵
- 완전히 처음부터 시작하려면 `outputs/` 폴더 삭제

## 출력 파일

### outputs/results.jsonl

각 줄이 하나의 시뮬레이션 결과 (JSON Lines 형식):

```json
{
  "source_mdl25_id": "O0wvX3Zv",
  "source_mdl138_id": "78qgqRbQ",
  "expr25": "ts_zscore(mdl25_vrv421_71v, 252)",
  "expr138": "subtract(ts_backfill(vec_avg(mdl138_5idpqc),60),...)",
  "operator": "add",
  "applied_group_zscore": false,
  "final_expression": "add((ts_zscore(...)),(subtract(...)))",
  "sim_status": "SUCCESS",
  "sharpe": 2.15,
  "fitness": 1.45,
  "turnover": 0.12,
  "fail_reasons": null,
  "alpha_id": "abc123xyz",
  "created_at": "2026-02-05T14:30:00"
}
```

## 결과 분석 예시

```python
import pandas as pd

# 결과 로드
df = pd.read_json("outputs/results.jsonl", lines=True)

# 성공한 알파만 필터
success = df[df['sim_status'] == 'SUCCESS']

# Sharpe >= 2.0 & Fitness >= 1.5 필터
good = success[(success['sharpe'] >= 2.0) & (success['fitness'] >= 1.5)]

# 상위 10개 출력
print(good.nlargest(10, 'sharpe')[['operator', 'applied_group_zscore', 'sharpe', 'fitness', 'alpha_id']])

# 연산자별 통계
print(success.groupby('operator')['sharpe'].agg(['mean', 'max', 'count']))
```

## 설정 변경

`combine_and_simulate.py` 상단의 설정 섹션에서 수정:

```python
# Simulation settings
REGION = "EUR"           # 지역
UNIVERSE = "TOP2500"     # 유니버스
DELAY = 1                # 딜레이
DECAY = 0                # 디케이
NEUTRALIZATION = "INDUSTRY"  # 중립화

# Concurrency settings
MAX_CONCURRENT_SIMULATIONS = 3  # 동시 시뮬레이션 수
BATCH_SIZE = 10                 # 배치당 알파 수
```

## 주의사항

1. Brain API 세션은 자동으로 갱신되지만, 바이오메트릭 인증이 필요할 수 있음
2. Rate limit (429) 및 서버 에러 (5xx)는 자동으로 exponential backoff 재시도
3. 대량 시뮬레이션은 API 제한으로 인해 시간이 걸릴 수 있음
