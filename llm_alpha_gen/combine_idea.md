# Brain Alpha – Weight Concentration / Power Pool 완화 가이드

이 문서는 Brain 플랫폼에서  
**`Weight is too strongly concentrated or too few instruments are assigned weight`**  
에러를 해결하기 위해 실제로 검증된 아이디어와 패턴을 정리한 문서이다.

---

## 1. 문제 정의

Brain에서 해당 에러는 다음 두 경우 중 하나(또는 둘 다)일 때 발생한다.

### 1.1 too few instruments
- 대부분의 종목이 **0 또는 상수값**
- 실제로 weight가 배정되는 종목 수가 너무 적음

### 1.2 too strongly concentrated
- 종목 수는 충분하지만
- **상위 소수 종목에 weight가 과도하게 집중**

---

## 2. 주요 원인 분류

### A. 단면 분포가 이산(discrete)해질 때
- `quantile`, `rank`, `group_rank`가 **최종단**에 위치
- 동점(tie)이 대량 발생

### B. 종목을 죽이는 구조
- `tail(..., newval=0)`
- `max(x, 0)`, `min(x, 0)`
- 조건 필터(`trade_when`, hard if)

### C. 극단값 증폭
- `signed_power(x, >1)`
- 분모가 작은 비율식
- 캡이 너무 타이트(±1)

### D. 같은 테마 중첩
- EPS × EPS × Rank
- Value × Value × Momentum

---

## 3. Brain에서 가장 자주 쓰이는 해결 아이디어 TOP 10

### ① 최종 출력은 반드시 연속형으로
```text
zscore(alpha)
zscore(quantile(x))   // 랭크 성질 유지 절충안

② quantile / rank는 중간 feature로만 사용
feature = quantile(x)
alpha   = zscore(feature)

③ tail / 필터에서 newval=0 금지
tail(x, newval=0.2 ~ 0.4)   // OK
tail(x, newval=0)           // 거의 확정 탈락

④ min / max는 “조합”이 아니라 “캡” 용도
max(min(zscore(x), 2), -2)


❌ max(alpha1, alpha2) 직접 결합


⑤ signed_power는 1보다 작게
signed_power(alpha, 0.6 ~ 0.9)


1 : 극단 확대 → concentration 증가


⑥ 캡은 ±1보다 넓게
±1      → 동점 과다
±2 ~ 3  → 분포 유지 + 안정성↑


⑦ 반드시 그룹 중립화
group_rank(alpha, industry)
group_scale(alpha, subindustry)

⑧ 레짐 선택(min/max)은 연속화 후 사용
zscore(max(min(x,2),-2))

⑨ Analyst / Short 데이터는 느리게 변환
zscore(ts_decay_linear(ts_backfill(analyst_x, 60), 20))


이벤트성 → 연속형 신호

⑩ 최종 점검 체크리스트

값이 0인 종목 비율 > 30% ❌

같은 값(동점) 비율 > 40% ❌

±cap에 붙은 종목이 과도하게 많음 ❌