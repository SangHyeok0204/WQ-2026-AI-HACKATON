import random
random.seed(42)

# Top 35 alphas (excluding bottom 20: #36-#55)
exprs = {
    1: "signed_power(group_scale(min(add(multiply(rank(ts_delta(ts_backfill(vec_avg(mdl138_4idpc),20),5)),0.5), multiply(divide(rank(ts_zscore(ts_backfill(vec_avg(mdl138_5idpqc),60),60)), add(rank(mdl30_psprise_pct_fy1_eps),1)),0.5)), multiply(rank(vec_avg(mdl138_4idpc)), rank(industry_value_momentum_rank))), sector), 0.7)",
    2: "signed_power(group_rank(min(quantile(ts_zscore(mdl25_vrv421_91v,63)), zscore(ts_backfill(vec_avg(mdl138_4idpc),30))), industry), 0.65)",
    3: "signed_power(group_rank(min(add(multiply(reverse(ts_rank(mdl25_01v,20)),0.6), multiply(add(zscore(mdl25_vrv421_71v), zscore(mdl25_21v)),0.4)), add(ts_delta(ts_backfill(vec_avg(mdl138_4idpc),5),10), vec_avg(mdl138_5idpc))), subindustry), 0.7)",
    4: "signed_power(group_rank(max(multiply(rank(ts_delta(vec_avg(mdl138_5idpqc), 15)), rank(region_value_momentum_rank)), divide(rank(mdl25_21v), add(rank(global_price_momentum_percentile_2), 1))), subindustry), 0.5)",
    5: "signed_power(group_rank(max(multiply(rank(ts_av_diff(ts_backfill(vec_avg(mdl138_3idpc), 7), 10)), rank(region_value_momentum_rank)), divide(rank(mdl25_21v), add(rank(global_price_momentum_percentile_2), 1))), subindustry), 0.6)",
    6: "signed_power(group_rank(min(ts_delta(ts_backfill(vec_avg(mdl138_4idpc),5),10), ts_zscore(mdl25_vrv421_81v,252)), sector), 0.7)",
    7: "signed_power(group_rank(add(rank(ts_arg_min(mdl25_vrv421_81v, 10)), ts_rank(anl39_qtanbvps, 126)), industry), 0.6)",
    8: "signed_power(group_rank(max(multiply(rank(ts_delta(ts_backfill(vec_avg(mdl138_4idpc), 20), 15)), rank(region_value_momentum_rank)), divide(rank(mdl25_21v), add(rank(global_price_momentum_percentile_2), 1))), subindustry), 0.6)",
    9: "signed_power(group_rank(max(multiply(rank(ts_delta(vec_avg(mdl138_4idpc),15)), rank(region_value_momentum_rank)), divide(rank(mdl25_21v), add(rank(mdl25_vrv421_71v),1))), subindustry), 0.5)",
    10: "signed_power(group_rank(add(ts_ir(mdl25_vrv421_81v, 126), ts_backfill(mdl39_d1_price_mo_short_term_component, 21)), subindustry), 0.6)",
    11: "signed_power(group_rank(min(vec_avg(mdl138_4idpc), ts_delta(ts_backfill(vec_avg(mdl138_5idpqc), 20), 5)), industry), 0.8)",
    12: "signed_power(group_rank(min(winsorize(ts_backfill(vec_avg(mdl138_4idpc), 60), std=4), ts_zscore(mdl25_vrv421_91v, 15)), industry), 0.75)",
    13: "signed_power(group_rank(min(normalize(ts_backfill(vec_avg(mdl138_4idpc), 10)), zscore(ts_av_diff(ts_backfill(vec_avg(mdl138_3idpc), 7), 30))), industry), 0.69)",
    14: "signed_power(group_rank(max(multiply(rank(ts_delta(ts_backfill(vec_avg(mdl138_5idpqc), 20), 14)), rank(region_value_momentum_rank)), divide(rank(mdl25_vrv421_71v), add(rank(global_price_momentum_percentile_2), 1))), subindustry), 0.55)",
    15: "signed_power(group_rank(min(ts_decay_linear(rank(subtract(mdl25_vrv421_71v, ts_mean(mdl25_vrv421_71v, 30))), 40), add(divide(rank(ts_rank(global_value_momentum_rank_float, 60)), add(rank(industry_value_momentum_rank), 1)), rank(short_term_price_momentum_score_2))), industry), 0.8)",
    16: "signed_power(group_rank(max(multiply(rank(ts_delta(vec_avg(mdl138_5idpqc), 15)), rank(region_value_momentum_rank)), divide(rank(mdl25_21v), add(rank(industry_value_momentum_rank), 1))), subindustry), 0.6)",
    17: "signed_power(group_rank(zscore(winsorize(ts_backfill(vec_avg(mdl138_4idpc),20), std=4)), industry), 0.65)",
    18: "signed_power(group_rank(scale(winsorize(ts_backfill(vec_avg(mdl138_4idpc),20), std=4)), industry), 0.68)",
    19: "signed_power(group_rank(normalize(add(ts_backfill(vec_avg(mdl138_4idpc), 10), ts_backfill(vec_avg(mdl138_3idpc), 10))), industry), 0.6)",
    20: "signed_power(group_rank(ts_backfill(add(vec_avg(mdl138_3idpc), vec_avg(mdl138_pdi5_cash)),30), industry), 0.6)",
    21: "group_scale(max(ts_delta(ts_backfill(vec_avg(mdl138_5idpqc), 20), 5), ts_delta(ts_backfill(vec_avg(mdl138_4idpqc), 20), 5)), sector)",
    22: "signed_power(group_rank(ts_backfill(add(vec_avg(mdl138_3idpc), vec_avg(mdl138_5idpc)), 30), industry), 0.7)",
    23: "signed_power(group_rank(max(log(ts_zscore(mdl25_vrv421_71v, 100)), ts_rank(anl39_qtanbvps, 126)), subindustry), 0.5)",
    24: "group_scale(ts_target_tvr_decay(ts_backfill(vec_avg(mdl138_4idpc), 20), lambda_min=0, lambda_max=1, target_tvr=0.1), subindustry)",
    25: "group_scale(add(ts_zscore(ts_backfill(vec_avg(mdl138_4idpc), 20), 40), ts_zscore(ts_backfill(vec_avg(mdl138_qpdi4_op_margin), 20), 40)), sector)",
    26: "group_scale(scale(winsorize(ts_backfill((vec_avg(mdl138_4idpc)+vec_avg(mdl138_5idpc))/2, 20), std=4)), subindustry)",
    27: "signed_power(group_rank(subtract(ts_backfill(vec_avg(mdl138_5idpqc), 60), ts_mean(ts_backfill(vec_avg(mdl138_5idpqc), 60), 60)), industry), 0.7)",
    28: "signed_power(group_rank(subtract(ts_backfill(vec_avg(mdl138_5idpqc), 60), ts_mean(ts_backfill(vec_avg(mdl138_5idpqc), 60), 60)), industry), 0.8)",
    29: "group_scale(add(ts_zscore(ts_backfill(vec_avg(mdl138_4idpc), 20), 40), ts_zscore(ts_backfill(vec_avg(mdl138_qpdi4_op_margin), 20), 40)), industry)",
    30: "signed_power(group_scale(min(rank(vec_avg(mdl138_4idpc)), rank(mdl30_psprise_pct_fy1_eps)), sector), 0.8)",
    31: "signed_power(group_rank(ts_backfill(add(vec_avg(mdl138_3idpqc), vec_avg(mdl138_3idpc)), 30), industry), 0.6)",
    32: "signed_power(group_rank(zscore(ts_av_diff(ts_backfill(vec_avg(mdl138_4idpqc), 5), 10)), country), 0.65)",
    33: "group_rank(min(ts_zscore(ts_returns(mdl25_vrv421_71v,60),60), zscore(ts_av_diff(ts_backfill(vec_avg(mdl138_5idpc),15),60))), industry)",
    34: "signed_power(group_rank(min(winsorize(ts_backfill(vec_avg(mdl138_4idpc), 20), std=4), ts_zscore(ts_returns(mdl25_vrv421_91v, 21), 21)), sector), 0.8)",
    35: "signed_power(group_rank(min(ts_av_diff(ts_backfill(vec_avg(mdl138_5idpc), 15), 60), quantile(mdl30_psprise_pct_fy1_eps)), industry), 0.8)",
}

sharpe = {
    1:2.93, 2:2.84, 3:2.72, 4:2.71, 5:2.48, 6:2.47, 7:2.47, 8:2.43,
    9:2.41, 10:2.37, 11:2.36, 12:2.34, 13:2.32, 14:2.24, 15:2.21, 16:2.20,
    17:2.19, 18:2.19, 19:2.18, 20:2.13, 21:2.07, 22:2.06, 23:2.06, 24:1.99,
    25:1.93, 26:1.91, 27:1.90, 28:1.87, 29:1.86, 30:1.85, 31:1.85, 32:1.76,
    33:1.72, 34:1.70, 35:1.68,
}

# Categories designed for maximum diversity
# A: Pure fundamental level (mdl138 level values, low turnover)
cat_A = [17, 18, 19, 20, 22, 24, 26, 31]
# B: Fundamental change/delta (mdl138 with ts_delta/ts_av_diff)
cat_B = [4, 5, 8, 9, 11, 13, 14, 16, 21, 27, 28, 32]
# C: Volatility model dominant (mdl25_vrv421 prominent)
cat_C = [2, 3, 6, 12, 33, 34]
# D: Non-mdl138 diverse signals (analyst, momentum, price-mo)
cat_D = [7, 10, 15, 23]
# E: Complex multi-signal (mdl138 + mdl30 + momentum combos)
cat_E = [1, 25, 29, 30, 35]

def make_add2(a, b):
    return f"add({exprs[a]}, {exprs[b]})"

def make_add3(a, b, c):
    return f"add(add({exprs[a]}, {exprs[b]}), {exprs[c]})"

combos = []

# ============================================================
# 2-WAY COMBINATIONS (cross-category for low correlation)
# ============================================================

# D x A: Non-fundamental + Pure fundamental (most diverse pairs)
da_pairs = [
    (7,17), (7,19), (7,20), (7,22),
    (10,17), (10,19), (10,20), (10,24),
    (15,17), (15,18), (15,22), (15,26),
    (23,19), (23,20), (23,24), (23,31),
]
for a, b in da_pairs:
    combos.append(('2way', 'DxA', a, b, make_add2(a, b)))

# D x B: Non-fundamental + Fundamental change
db_pairs = [
    (7,4), (7,5), (7,13), (7,21),
    (10,4), (10,8), (10,11), (10,16),
    (15,4), (15,9), (15,13), (15,27),
    (23,5), (23,8), (23,14),
]
for a, b in db_pairs:
    combos.append(('2way', 'DxB', a, b, make_add2(a, b)))

# D x C: Non-fundamental + Volatility
dc_pairs = [
    (7,2), (7,3), (7,6),
    (10,2), (10,3), (10,12),
    (15,2), (15,6), (15,34),
    (23,2), (23,3), (23,6),
]
for a, b in dc_pairs:
    combos.append(('2way', 'DxC', a, b, make_add2(a, b)))

# D x E: Non-fundamental + Complex multi
de_pairs = [
    (7,1), (7,30), (7,35),
    (10,1), (10,25), (10,30),
    (15,1), (15,25),
    (23,1), (23,35),
]
for a, b in de_pairs:
    combos.append(('2way', 'DxE', a, b, make_add2(a, b)))

# A x C: Pure fundamental + Volatility
ac_pairs = [
    (17,2), (17,3), (19,6),
    (20,2), (20,3), (22,6),
    (24,12), (26,34),
]
for a, b in ac_pairs:
    combos.append(('2way', 'AxC', a, b, make_add2(a, b)))

# B x D (additional, D side): ensure no dupes with DxB above - flip order
# Already covered above via DxB

# E x C: Complex + Volatility
ec_pairs = [
    (1,2), (1,3), (1,6),
    (25,2), (30,3), (35,6),
]
for a, b in ec_pairs:
    combos.append(('2way', 'ExC', a, b, make_add2(a, b)))

print(f"2-way combos: {len([c for c in combos if c[0]=='2way'])}")

# ============================================================
# 3-WAY COMBINATIONS (3 different categories)
# ============================================================

# D + A + C (non-fundamental + pure fundamental + volatility)
dac_triples = [
    (7, 17, 2), (7, 19, 3), (7, 20, 6), (7, 22, 12),
    (10, 17, 2), (10, 19, 6), (10, 20, 3), (10, 24, 34),
    (15, 17, 2), (15, 19, 3), (15, 22, 6),
    (23, 17, 2), (23, 20, 3), (23, 24, 6),
]
for a, b, c in dac_triples:
    combos.append(('3way', 'D+A+C', a, b, make_add3(a, b, c)))

# D + B + C (non-fundamental + fundamental change + volatility)
dbc_triples = [
    (7, 4, 2), (7, 13, 3), (7, 21, 6),
    (10, 4, 2), (10, 8, 3), (10, 11, 6),
    (15, 4, 2), (15, 9, 3), (15, 14, 6),
    (23, 5, 2), (23, 8, 6),
]
for a, b, c in dbc_triples:
    combos.append(('3way', 'D+B+C', a, b, make_add3(a, b, c)))

# D + A + E (non-fundamental + pure fundamental + complex)
dae_triples = [
    (7, 17, 1), (7, 20, 30), (7, 22, 35),
    (10, 17, 1), (10, 19, 25), (10, 24, 30),
    (15, 17, 1), (15, 20, 25),
    (23, 19, 1), (23, 22, 35),
]
for a, b, c in dae_triples:
    combos.append(('3way', 'D+A+E', a, b, make_add3(a, b, c)))

# A + C + E (pure fundamental + volatility + complex)
ace_triples = [
    (17, 2, 1), (19, 3, 1), (20, 6, 1),
    (22, 2, 25), (24, 3, 30), (26, 6, 35),
    (17, 12, 30),
]
for a, b, c in ace_triples:
    combos.append(('3way', 'A+C+E', a, b, make_add3(a, b, c)))

# D + B + E (non-fundamental + fundamental change + complex)
dbe_triples = [
    (7, 4, 1), (7, 11, 25),
    (10, 4, 1), (10, 9, 30),
    (15, 4, 1), (15, 13, 35),
    (23, 8, 1), (23, 16, 30),
]
for a, b, c in dbe_triples:
    combos.append(('3way', 'D+B+E', a, b, make_add3(a, b, c)))

print(f"3-way combos: {len([c for c in combos if c[0]=='3way'])}")
print(f"Total combos: {len(combos)}")

# ============================================================
# Write output
# ============================================================
lines = []
lines.append("=" * 80)
lines.append("TEST ALPHA COMBINATIONS - add() merges for PROD_CORRELATION reduction")
lines.append(f"Region: EUR, Universe: TOP2500")
lines.append(f"Total: {len(combos)} alpha combinations")
lines.append(f"Source: Top 35 alphas from submit_ready.txt (Sharpe >= 1.68)")
lines.append("Strategy: Cross-category combinations to minimize signal overlap")
lines.append("=" * 80)
lines.append("")
lines.append("Categories used for diversification:")
lines.append("  A = Pure fundamental level (mdl138 level): #17,18,19,20,22,24,26,31")
lines.append("  B = Fundamental change/delta (mdl138 delta): #4,5,8,9,11,13,14,16,21,27,28,32")
lines.append("  C = Volatility model (mdl25_vrv421): #2,3,6,12,33,34")
lines.append("  D = Non-mdl138 diverse (analyst/momentum): #7,10,15,23")
lines.append("  E = Complex multi-signal (mdl138+mdl30+momentum): #1,25,29,30,35")
lines.append("")

for i, combo in enumerate(combos, 1):
    way = combo[0]
    cross = combo[1]
    if way == '2way':
        src_a, src_b = combo[2], combo[3]
        expr = combo[4]
        avg_sharpe = (sharpe[src_a] + sharpe[src_b]) / 2
        lines.append(f"--- #{i} | {way} | {cross} | src: #{src_a}+#{src_b} | avg_sharpe: {avg_sharpe:.2f} ---")
    else:
        src_a, src_b = combo[2], combo[3]
        # For 3way, we embedded 3rd in the expression but only stored 2 src IDs
        # Need to extract 3rd from the triple
        expr = combo[4]
        lines.append(f"--- #{i} | {way} | {cross} | src: #{src_a}+#{src_b}+... | ---")

    lines.append(expr)
    lines.append("")

lines.append("=" * 80)
lines.append(f"Total: {len(combos)} test alpha combinations")
lines.append("=" * 80)

with open("test_alpha.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"\nDone! Wrote {len(combos)} combos to test_alpha.txt")
