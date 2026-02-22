# Alpha Generation Ideas

## 성공한 알파 4개에서 추출한 패턴

### 원본 알파

```
# Alpha 1
signal = group_rank(min(ts_decay_linear(rank(subtract(mdl30_numnly_fq2_eps, vec_avg(shrt3_utilizationpercent_units))), 40), add(divide(rank(ts_rank(global_value_momentum_rank_float, 60)), add(rank(industry_value_momentum_rank_float), 1)), rank(short_term_price_momentum_score_2))),industry);
signed_power(signal, 0.8)

# Alpha 2
signal = group_rank(max(multiply(rank(ts_delta(vec_avg(shrt3_bar), 15)), rank(region_value_momentum_rank)), divide(rank(mdl25_21v), add(rank(global_price_momentum_percentile_2), 1))),subindustry);
signed_power(signal, 0.5)

# Alpha 3
group_scale(min(add(multiply(rank(vec_avg(mdl138_ofc_5idp)), 0.4), multiply(rank(star_eps_surprise_prediction_fy2), 0.6)), multiply(rank(ts_delta(vec_avg(shrt3_bar), 10)), rank(industry_value_momentum_rank))),subindustry)

# Alpha 4
max(add(multiply(rank(vec_avg(mdl138_ofc_5idp)), 0.7), multiply(rank(star_new_eps_surprise_prediction_12m), 0.3)), divide(rank(ts_delta(global_value_momentum_rank_float, 40)), add(rank(vec_avg(shrt3_utilizationpercent_units)), 1)))
```

---

## 1. 구조적 패턴

### 1.1 기본 구조
```
group_rank/group_scale(
    min/max(Signal_A, Signal_B),
    industry/subindustry/sector/country/region
)
```

### 1.2 신호 억제
```
signed_power(signal, 0.5~0.8)
```
- 극단적 신호를 완화하여 안정성 확보

### 1.3 min/max 선택
- `min`: 보수적 접근, 두 신호 중 약한 쪽 선택
- `max`: 공격적 접근, 두 신호 중 강한 쪽 선택

---

## 2. 데이터 소스 조합

### 2.1 핵심 데이터 카테고리

| 카테고리 | 데이터필드 | 역할 |
|----------|-----------|------|
| **News Sentiment** | nws17_comp_d1_event_sentiment_score, nws17_comp_d1_event_relevance | 뉴스 감성/센티먼트 |
| **Risk/Volatility** | default_risk_global_percentile_d1, annualized_asset_volatility_pct | 위험/변동성 |
| **Model Predictions** | mdl25_21v, mdl30_numnly_fq2_eps, mdl138_ofc_5idp | 퀀트 모델 예측 |
| **Analyst Estimates** | star_eps_surprise_prediction_fy2, star_new_eps_surprise_prediction_12m | 애널리스트 서프라이즈 |
| **Momentum Ranks** | global_value_momentum_rank_float, industry_value_momentum_rank, region_value_momentum_rank | 모멘텀 순위 |

### 2.2 조합 공식
```
[Model/Analyst 신호] + [News/Risk 신호] + [Momentum Rank]
```

---

## 3. 핵심 연산 패턴

### 3.1 가중 결합 패턴
```
add(
    multiply(rank(X), 0.4),
    multiply(rank(Y), 0.6)
)
```
- 두 신호에 가중치를 부여하여 결합
- 가중치 합이 1.0이 되도록 설정

### 3.2 비율 신호 패턴
```
divide(
    rank(X),
    add(rank(Y), 1)
)
```
- 분모에 +1을 추가하여 0 나눗셈 방지
- X/Y의 상대적 비율을 계산

### 3.3 시계열 변환 패턴
```
ts_delta(vec_avg(...), 10~40)      # 변화량
ts_decay_linear(..., 40)            # 선형 감쇠
ts_rank(..., 60)                    # 시계열 순위
```

### 3.4 Rank 중첩 패턴
```
rank(ts_delta(vec_avg(X), N))
rank(ts_rank(X, N))
```
- 거의 모든 데이터필드에 rank() 적용
- 극단값 완화 및 상대 순위화

---

## 4. Group 정규화

### 4.1 사용 가능한 Group Identifiers
| Group | 설명 |
|-------|------|
| `industry` | 산업 분류 (가장 자주 사용) |
| `subindustry` | 세부 산업 분류 |
| `sector` | 섹터 분류 (industry보다 넓은 범위) |
| `country` | 국가별 그룹 |
| `region` | 지역별 그룹 (GLB/ASI 등 멀티리전에서 유용) |

### 4.2 group_rank
```
group_rank(signal, industry)
group_rank(signal, subindustry)
group_rank(signal, sector)
group_rank(signal, country)
group_rank(signal, region)
```
- 그룹 내에서 순위화 → 그룹 간 편향 제거
- `industry`/`subindustry`가 가장 일반적, `sector`는 더 넓은 범위
- `country`/`region`은 글로벌/아시아 유니버스에서 국가/지역 편향 제거에 효과적

### 4.3 group_scale
```
group_scale(signal, subindustry)
group_scale(signal, sector)
```
- 그룹 내 스케일 정규화

---

## 5. 추천 템플릿

### Template A: Conservative (min 사용)
```
signal = group_rank(
    min(
        ts_decay_linear(rank(subtract(MODEL_FIELD, NEWS_SENTIMENT)), 40),
        add(
            divide(rank(ts_rank(MOMENTUM_FIELD, 60)), add(rank(INDUSTRY_MOMENTUM), 1)),
            rank(PRICE_MOMENTUM)
        )
    ),
    industry
);
signed_power(signal, 0.8)
```

### Template B: Aggressive (max 사용)
```
signal = group_rank(
    max(
        multiply(rank(ts_delta(NEWS_SENTIMENT, 15)), rank(MOMENTUM_FIELD)),
        divide(rank(MODEL_FIELD), add(rank(PRICE_MOMENTUM), 1))
    ),
    subindustry
);
signed_power(signal, 0.5)
```

### Template C: Weighted Combination
```
group_scale(
    min(
        add(multiply(rank(vec_avg(MODEL_FIELD)), 0.4), multiply(rank(ANALYST_FIELD), 0.6)),
        multiply(rank(ts_delta(RISK_FIELD, 10)), rank(INDUSTRY_MOMENTUM))
    ),
    subindustry
)
```

---

## 6. 권장 데이터필드 조합

### News Sentiment (센티먼트 신호)
- `nws17_comp_d1_event_sentiment_score` - 뉴스 감성 점수
- `nws17_comp_d1_event_relevance` - 이벤트 관련성

### Risk/Volatility (위험 신호)
- `default_risk_global_percentile_d1` - 글로벌 부도위험
- `default_risk_industry_percentile_d1` - 산업 내 부도위험
- `annualized_asset_volatility_pct` - 연환산 변동성

### Model Predictions
- `mdl25_21v`
- `mdl30_numnly_fq2_eps`
- `mdl138_ofc_5idp`

### Analyst Estimates
- `star_eps_surprise_prediction_fy2`
- `star_new_eps_surprise_prediction_12m`

### Momentum Ranks
- `global_value_momentum_rank_float`
- `industry_value_momentum_rank`
- `region_value_momentum_rank`
- `global_price_momentum_percentile_2`
- `short_term_price_momentum_score_2`

---

## 7. 핵심 인사이트

1. **Multi-factor 조합**: 단일 신호가 아닌 2개 이상의 독립적 신호를 min/max로 결합

2. **News/Risk 활용**: 센티먼트(nws17) 또는 위험지표(default_risk)로 contrarian 신호 생성

3. **Rank 중첩**: 거의 모든 데이터필드에 rank() 적용하여 극단값 완화

4. **Group 정규화**: group_rank/group_scale로 industry/subindustry/sector/country/region 내 상대 위치 계산

5. **신호 억제**: signed_power(signal, 0.5~0.8)로 극단적 신호 완화

6. **시계열 윈도우**: 10~60일 범위의 다양한 lookback period 활용
