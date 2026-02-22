import random
random.seed(42)

# Top 17 alphas from no_2year_sharpe.txt (by orig # from submit_ready.txt)
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
    12: "signed_power(group_rank(min(winsorize(ts_backfill(vec_avg(mdl138_4idpc), 60), std=4), ts_zscore(mdl25_vrv421_91v, 15)), industry), 0.75)",
    13: "signed_power(group_rank(min(normalize(ts_backfill(vec_avg(mdl138_4idpc), 10)), zscore(ts_av_diff(ts_backfill(vec_avg(mdl138_3idpc), 7), 30))), industry), 0.69)",
    14: "signed_power(group_rank(max(multiply(rank(ts_delta(ts_backfill(vec_avg(mdl138_5idpqc), 20), 14)), rank(region_value_momentum_rank)), divide(rank(mdl25_vrv421_71v), add(rank(global_price_momentum_percentile_2), 1))), subindustry), 0.55)",
    15: "signed_power(group_rank(min(ts_decay_linear(rank(subtract(mdl25_vrv421_71v, ts_mean(mdl25_vrv421_71v, 30))), 40), add(divide(rank(ts_rank(global_value_momentum_rank_float, 60)), add(rank(industry_value_momentum_rank), 1)), rank(short_term_price_momentum_score_2))), industry), 0.8)",
    16: "signed_power(group_rank(max(multiply(rank(ts_delta(vec_avg(mdl138_5idpqc), 15)), rank(region_value_momentum_rank)), divide(rank(mdl25_21v), add(rank(industry_value_momentum_rank), 1))), subindustry), 0.6)",
    23: "signed_power(group_rank(max(log(ts_zscore(mdl25_vrv421_71v, 100)), ts_rank(anl39_qtanbvps, 126)), subindustry), 0.5)",
    30: "signed_power(group_scale(min(rank(vec_avg(mdl138_4idpc)), rank(mdl30_psprise_pct_fy1_eps)), sector), 0.8)",
}

sharpe = {
    1:2.93, 2:2.84, 3:2.72, 4:2.71, 5:2.48, 6:2.47, 7:2.47, 8:2.43,
    9:2.41, 10:2.37, 12:2.34, 13:2.32, 14:2.24, 15:2.21, 16:2.20, 23:2.06, 30:1.85,
}

# ============================================================
# Category classification for diversity
# ============================================================
# A: Pure vol/analyst/momentum (NO mdl138 at all - most unique signals)
#    #7  = mdl25_vrv421_81v arg_min + anl39_qtanbvps
#    #10 = mdl25_vrv421_81v IR + mdl39_d1_price_mo (short term momentum)
#    #15 = mdl25_vrv421_71v decay + global/industry momentum ranks
#    #23 = mdl25_vrv421_71v zscore + anl39_qtanbvps
cat_A = [7, 10, 15, 23]

# B: Vol+Fundamental blend (mdl25_vrv421 prominently combined with mdl138)
#    #2  = mdl25_vrv421_91v zscore + mdl138_4idpc (industry)
#    #3  = mdl25_01v/71v/21v + mdl138_4idpc/5idpc (subindustry)
#    #6  = mdl138_4idpc delta + mdl25_vrv421_81v zscore (sector)
#    #12 = mdl138_4idpc winsorize + mdl25_vrv421_91v zscore (industry)
cat_B = [2, 3, 6, 12]

# C: Fundamental delta + Momentum (mdl138 delta * momentum_rank)
#    #4  = mdl138_5idpqc delta + region_momentum + mdl25_21v (subindustry)
#    #5  = mdl138_3idpc av_diff + region_momentum + mdl25_21v (subindustry)
#    #8  = mdl138_4idpc delta + region_momentum + mdl25_21v (subindustry)
#    #9  = mdl138_4idpc delta + region_momentum + mdl25_vrv421_71v (subindustry)
#    #14 = mdl138_5idpqc delta + region_momentum + mdl25_vrv421_71v (subindustry)
#    #16 = mdl138_5idpqc delta + region_momentum + industry_momentum (subindustry)
cat_C = [4, 5, 8, 9, 14, 16]

# D: Fundamental-focused (mdl138 level/change without momentum rank)
#    #13 = mdl138_4idpc normalize + mdl138_3idpc av_diff zscore (industry)
#    #30 = mdl138_4idpc rank + mdl30_psprise_pct_fy1_eps rank (sector)
cat_D = [13, 30]

# E: Complex multi-signal
#    #1  = mdl138_4idpc/5idpqc + mdl30 + industry_momentum (sector)
cat_E = [1]

def add2(a, b):
    return f"add({exprs[a]}, {exprs[b]})"

def add3(a, b, c):
    return f"add(add({exprs[a]}, {exprs[b]}), {exprs[c]})"

combos = []

# ============================================================
# 2-WAY COMBINATIONS
# ============================================================

# A x C: Pure vol/analyst + Fundamental delta+momentum (MOST DIVERSE)
for a in cat_A:
    for c in cat_C:
        combos.append(('2way', 'AxC', a, c, add2(a, c)))
# 4 * 6 = 24

# A x D: Pure vol/analyst + Pure fundamental
for a in cat_A:
    for d in cat_D:
        combos.append(('2way', 'AxD', a, d, add2(a, d)))
# 4 * 2 = 8

# A x E: Pure vol/analyst + Complex
for a in cat_A:
    combos.append(('2way', 'AxE', a, 1, add2(a, 1)))
# 4

# A x B: Pure vol/analyst + Vol+Fundamental blend
for a in cat_A:
    # pick 2 most different B members for each A
    if a in [7, 10]:  # these use mdl25_vrv421_81v, pair with #2(91v) and #3(01v/71v/21v)
        for b in [2, 3]:
            combos.append(('2way', 'AxB', a, b, add2(a, b)))
    else:  # #15,#23 use mdl25_vrv421_71v, pair with #6(81v) and #12(91v)
        for b in [6, 12]:
            combos.append(('2way', 'AxB', a, b, add2(a, b)))
# 8

# B x C: Vol+Fund blend + Fund delta+momentum (different mdl138 variants)
bc_pairs = [
    (2, 4), (2, 5), (2, 16),   # vrv421_91v blend + 5idpqc/3idpc/5idpqc delta
    (3, 4), (3, 8), (3, 14),   # multi-vol blend + 5idpqc/4idpc/5idpqc delta
    (6, 4), (6, 5),            # vrv421_81v blend + 5idpqc/3idpc delta
    (12, 4), (12, 16),         # vrv421_91v blend + 5idpqc delta
]
for a, b in bc_pairs:
    combos.append(('2way', 'BxC', a, b, add2(a, b)))
# 10

# D x B: Pure fundamental + Vol blend
for d in cat_D:
    for b in cat_B:
        combos.append(('2way', 'DxB', d, b, add2(d, b)))
# 2 * 4 = 8

# E x B: Complex + Vol blend (different grouping: sector vs industry/subindustry)
for b in cat_B:
    combos.append(('2way', 'ExB', 1, b, add2(1, b)))
# 4

print(f"2-way combos: {len([c for c in combos if c[0]=='2way'])}")

# ============================================================
# 3-WAY COMBINATIONS (always 3 different categories)
# ============================================================

# A + C + D: vol/analyst + fund delta+momentum + pure fundamental
acd_triples = [
    (7, 4, 13), (7, 5, 30), (7, 8, 13), (7, 9, 30), (7, 14, 13), (7, 16, 30),
    (10, 4, 13), (10, 5, 30), (10, 8, 30), (10, 14, 13),
    (15, 4, 13), (15, 5, 30), (15, 9, 13), (15, 16, 30),
    (23, 4, 30), (23, 8, 13), (23, 14, 30), (23, 16, 13),
]
for a, c, d in acd_triples:
    combos.append(('3way', 'A+C+D', a, c, d, add3(a, c, d)))

# A + B + C: vol/analyst + vol+fund blend + fund delta+momentum
abc_triples = [
    (7, 2, 4), (7, 3, 5), (7, 12, 14), (7, 6, 16),
    (10, 2, 5), (10, 3, 4), (10, 12, 8), (10, 6, 9),
    (15, 2, 4), (15, 3, 8), (15, 6, 5), (15, 12, 16),
    (23, 2, 9), (23, 3, 14), (23, 6, 4), (23, 12, 5),
]
for a, b, c in abc_triples:
    combos.append(('3way', 'A+B+C', a, b, c, add3(a, b, c)))

# A + C + E: vol/analyst + fund delta+momentum + complex
ace_triples = [
    (7, 4, 1), (7, 5, 1), (7, 9, 1),
    (10, 4, 1), (10, 8, 1), (10, 14, 1),
    (15, 4, 1), (15, 5, 1),
    (23, 4, 1), (23, 16, 1),
]
for a, c, e in ace_triples:
    combos.append(('3way', 'A+C+E', a, c, e, add3(a, c, e)))

# A + D + E: vol/analyst + pure fundamental + complex
ade_triples = [
    (7, 13, 1), (7, 30, 1),
    (10, 13, 1), (10, 30, 1),
    (15, 13, 1), (15, 30, 1),
    (23, 13, 1), (23, 30, 1),
]
for a, d, e in ade_triples:
    combos.append(('3way', 'A+D+E', a, d, e, add3(a, d, e)))

# A + B + D: vol/analyst + vol+fund blend + pure fundamental
abd_triples = [
    (7, 2, 13), (7, 3, 30),
    (10, 2, 30), (10, 3, 13),
    (15, 6, 13), (15, 12, 30),
    (23, 6, 30), (23, 12, 13),
]
for a, b, d in abd_triples:
    combos.append(('3way', 'A+B+D', a, b, d, add3(a, b, d)))

print(f"3-way combos: {len([c for c in combos if c[0]=='3way'])}")
print(f"Total combos: {len(combos)}")

# ============================================================
# Write output
# ============================================================
lines = []
lines.append("=" * 80)
lines.append("TEST ALPHA COMBINATIONS (no LOW_2Y_SHARPE version)")
lines.append("add() merges for PROD_CORRELATION reduction")
lines.append("Region: EUR, Universe: TOP2500")
lines.append(f"Total: {len(combos)} alpha combinations")
lines.append("Source: Top 17 alphas from no_2year_sharpe.txt")
lines.append("Strategy: Cross-category combinations to minimize signal overlap")
lines.append("=" * 80)
lines.append("")
lines.append("Categories used for diversification:")
lines.append("  A = Pure vol/analyst/momentum (no mdl138): #7,10,15,23")
lines.append("  B = Vol+Fundamental blend (mdl25+mdl138): #2,3,6,12")
lines.append("  C = Fundamental delta+Momentum: #4,5,8,9,14,16")
lines.append("  D = Pure fundamental/analyst: #13,30")
lines.append("  E = Complex multi-signal: #1")
lines.append("")

for i, combo in enumerate(combos, 1):
    way = combo[0]
    cross = combo[1]
    if way == '2way':
        src_a, src_b = combo[2], combo[3]
        expr = combo[4]
        avg_sh = (sharpe[src_a] + sharpe[src_b]) / 2
        lines.append(f"--- #{i} | {way} | {cross} | src: #{src_a}+#{src_b} | avg_sharpe: {avg_sh:.2f} ---")
        lines.append(expr)
    else:
        src_a, src_b, src_c = combo[2], combo[3], combo[4]
        expr = combo[5]
        avg_sh = (sharpe[src_a] + sharpe[src_b] + sharpe[src_c]) / 3
        lines.append(f"--- #{i} | {way} | {cross} | src: #{src_a}+#{src_b}+#{src_c} | avg_sharpe: {avg_sh:.2f} ---")
        lines.append(expr)
    lines.append("")

lines.append("=" * 80)
lines.append(f"Total: {len(combos)} test alpha combinations")
lines.append("=" * 80)

with open("test_alpha(no_2year_ver).txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"\nDone! Wrote {len(combos)} combos to test_alpha(no_2year_ver).txt")
