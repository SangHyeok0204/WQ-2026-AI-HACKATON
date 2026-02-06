import json
import re
from collections import defaultdict
from pathlib import Path

# 경로 설정
SCRIPT_DIR = Path(__file__).parent
GOOD_ALPHA_FILE = SCRIPT_DIR / "good_alpha_list.json"
FAIL_ALPHA_FILE = SCRIPT_DIR / "fail_alpha_categorized.txt"

# 데이터셋 감지 함수
def detect_datasets(expression):
    """표현식에서 사용된 데이터셋들을 감지"""
    datasets = set()

    # 데이터셋 패턴 매칭
    if re.search(r'mdl25[_\w]*', expression, re.IGNORECASE):
        datasets.add('mdl25')
    if re.search(r'mdl30[_\w]*', expression, re.IGNORECASE):
        datasets.add('mdl30')
    if re.search(r'mdl138[_\w]*', expression, re.IGNORECASE):
        datasets.add('mdl138')
    if re.search(r'nws17[_\w]*', expression, re.IGNORECASE):
        datasets.add('nws17')
    if re.search(r'star[_\w]*', expression, re.IGNORECASE):
        datasets.add('star')

    return list(datasets)

# good_alpha_list.json 로드
print(f"Loading {GOOD_ALPHA_FILE}...")
with open(GOOD_ALPHA_FILE, 'r', encoding='utf-8') as f:
    alphas = json.load(f)

print(f"Total alphas loaded: {len(alphas)}")

# 데이터셋별로 알파 분류
dataset_alphas = defaultdict(list)
fail_alphas_by_reason = defaultdict(list)

for alpha in alphas:
    expression = alpha.get('expression', '')
    alpha_id = alpha.get('alpha_id', 'unknown')
    sharpe = alpha.get('is_sharpe', 0)
    fitness = alpha.get('is_fitness', 0)
    turnover = alpha.get('is_turnover', 0)
    checks = alpha.get('checks', [])

    # 데이터셋 감지
    datasets = detect_datasets(expression)

    # fail 사유 확인
    fail_reasons = []
    for check in checks:
        if check.get('result') == 'FAIL':
            fail_reasons.append(check.get('name'))

    # 0-fail 알파는 데이터셋별로 분류
    if len(fail_reasons) == 0:
        for dataset in datasets:
            dataset_alphas[dataset].append({
                'id': alpha_id,
                'expression': expression,
                'sharpe': sharpe,
                'fitness': fitness,
                'turnover': turnover,
                'datasets': datasets
            })
    else:
        # fail 알파는 사유별로 분류
        for reason in fail_reasons:
            fail_alphas_by_reason[reason].append({
                'expression': expression,
                'sharpe': sharpe,
                'fitness': fitness,
                'turnover': turnover,
                'fail_reason': reason,
                'checks': checks
            })

print(f"\n=== Dataset Analysis ===")
for dataset, alphas_list in sorted(dataset_alphas.items()):
    print(f"{dataset}: {len(alphas_list)} zero-fail alphas")

print(f"\n=== Fail Reason Analysis ===")
for reason, alphas_list in sorted(fail_alphas_by_reason.items()):
    print(f"{reason}: {len(alphas_list)} alphas")

# 데이터셋별 txt 파일 생성/업데이트
print(f"\n=== Creating/Updating Dataset Files ===")
for dataset, alphas_list in sorted(dataset_alphas.items()):
    output_file = SCRIPT_DIR / f"{dataset}.txt"

    # 기존 파일 읽기 (이미 있는 알파 ID 추적)
    existing_ids = set()
    if output_file.exists():
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # ID 패턴 추출
            id_matches = re.findall(r'ID: ([A-Za-z0-9]+)', content)
            existing_ids.update(id_matches)

    # 새로운 알파만 추가
    new_alphas = [a for a in alphas_list if a['id'] not in existing_ids]

    if new_alphas:
        print(f"Adding {len(new_alphas)} new alphas to {output_file.name}")

        with open(output_file, 'a', encoding='utf-8') as f:
            for i, alpha in enumerate(new_alphas):
                # 기존 파일이 비어있지 않으면 구분선 추가
                if output_file.stat().st_size > 0 or i > 0:
                    f.write("\n")

                f.write(f"--- #{len(existing_ids) + i + 1} | FAIL: 0 | Datasets: {', '.join(alpha['datasets'])} ---\n")
                f.write(f"ID: {alpha['id']}\n")
                f.write(f"Region: EUR, Universe: TOP2500\n")
                f.write(f"Sharpe: {alpha['sharpe']:.2f}, Fitness: {alpha['fitness']:.2f}, Turnover: {alpha['turnover']:.4f}\n")
                f.write(f"Expression: {alpha['expression']}\n")
    else:
        print(f"No new alphas for {output_file.name}")

# fail_alpha_categorized.txt 업데이트
print(f"\n=== Updating fail_alpha_categorized.txt ===")

# 기존 fail 파일 읽기
existing_fail_expressions = set()
if FAIL_ALPHA_FILE.exists():
    with open(FAIL_ALPHA_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
        # 표현식 패턴 추출 (2칸 들여쓰기로 시작하는 줄)
        expr_matches = re.findall(r'^  (.+?)$', content, re.MULTILINE)
        # "-> "로 시작하지 않는 것만 (실제 표현식)
        for expr in expr_matches:
            if not expr.strip().startswith('->'):
                existing_fail_expressions.add(expr.strip())

print(f"Found {len(existing_fail_expressions)} existing fail expressions")

# 새로운 fail 알파 찾기
new_fail_alphas = defaultdict(list)
for reason, alphas_list in fail_alphas_by_reason.items():
    for alpha in alphas_list:
        if alpha['expression'] not in existing_fail_expressions:
            new_fail_alphas[reason].append(alpha)

total_new_fails = sum(len(alphas) for alphas in new_fail_alphas.values())
print(f"Found {total_new_fails} new fail alphas to add")

if total_new_fails > 0:
    # fail_alpha_categorized.txt 업데이트
    # 기존 파일을 읽어서 각 섹션에 추가
    with open(FAIL_ALPHA_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 새로운 내용 작성
    with open(FAIL_ALPHA_FILE, 'w', encoding='utf-8') as f:
        i = 0
        while i < len(lines):
            line = lines[i]
            f.write(line)

            # 각 fail 사유 섹션 찾기
            if line.startswith("Fail 사유:"):
                # 사유 추출
                match = re.search(r'Fail 사유: ([A-Z_]+)', line)
                if match:
                    reason = match.group(1)

                    # 다음 섹션까지 복사
                    i += 1
                    while i < len(lines) and not lines[i].startswith("---"):
                        f.write(lines[i])
                        i += 1

                    # 이 사유에 대한 새로운 알파가 있으면 추가
                    if reason in new_fail_alphas:
                        for alpha in new_fail_alphas[reason]:
                            f.write(f"  {alpha['expression']}\n")

                            # 사유별로 다른 정보 출력
                            if reason == "HIGH_TURNOVER":
                                f.write(f"    -> Turnover: {alpha['turnover']:.4f}, cut: 0.7\n")
                            elif reason == "CONCENTRATED_WEIGHT":
                                f.write(f"    -> Sharpe: {alpha['sharpe']:.2f}, Fitness: {alpha['fitness']:.2f}, Turnover: {alpha['turnover']:.4f}\n")
                            elif reason == "LOW_SHARPE":
                                f.write(f"    -> Sharpe: {alpha['sharpe']:.2f}, cut: 1.58\n")
                            elif reason == "LOW_FITNESS":
                                f.write(f"    -> Fitness: {alpha['fitness']:.2f}, cut: 1.0\n")
                            elif reason == "LOW_2Y_SHARPE":
                                f.write(f"    -> 2Y Sharpe: N/A, cut: 1.58, (IS Sharpe: {alpha['sharpe']:.2f})\n")
                            elif reason == "LOW_SUB_UNIVERSE_SHARPE":
                                f.write(f"    -> Sub-Universe Sharpe: N/A, cut: 1.23, (IS Sharpe: {alpha['sharpe']:.2f})\n")
                            else:
                                f.write(f"    -> Sharpe: {alpha['sharpe']:.2f}, Fitness: {alpha['fitness']:.2f}, Turnover: {alpha['turnover']:.4f}\n")

                        print(f"  Added {len(new_fail_alphas[reason])} alphas to {reason}")

                    continue

            i += 1

    print(f"\nUpdated {FAIL_ALPHA_FILE.name} with {total_new_fails} new fail alphas")
else:
    print("No new fail alphas to add")

print("\n=== Done ===")
