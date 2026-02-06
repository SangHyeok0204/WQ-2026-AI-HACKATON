import re
from pathlib import Path
from collections import defaultdict

# 경로 설정
FAIL_ALPHA_FILE = Path(__file__).parent / "fail_alpha_categorized.txt"

# 파일 읽기
with open(FAIL_ALPHA_FILE, 'r', encoding='utf-8') as f:
    content = f.read()

# 각 섹션 파싱
sections = defaultdict(list)
current_section = None
current_expr = None

for line in content.split('\n'):
    # 섹션 헤더 찾기
    if line.startswith("Fail 사유:"):
        match = re.search(r'Fail 사유: ([A-Z_0-9]+)', line)
        if match:
            current_section = match.group(1)
            current_expr = None

    # 표현식 라인 (2칸 들여쓰기, "-> "로 시작하지 않음)
    elif line.startswith("  ") and not line.strip().startswith("->") and current_section:
        if not line.strip().startswith("---") and line.strip():
            current_expr = line.strip()
            sections[current_section].append({
                'expression': current_expr,
                'details': []
            })

    # 세부 정보 라인 ("-> "로 시작)
    elif line.strip().startswith("->") and current_section and sections[current_section]:
        sections[current_section][-1]['details'].append(line.strip())

# 각 섹션별 개수 출력
print("Section counts:")
for reason in sorted(sections.keys()):
    print(f"  {reason}: {len(sections[reason])}개")

# 새로운 파일 작성
output_lines = []

# 헤더
output_lines.append("=" * 80)
output_lines.append("FAIL ALPHA LIST - CATEGORIZED BY FAIL REASON")
output_lines.append("=" * 80)
output_lines.append("")
output_lines.append("")

# 각 섹션 작성
for reason in sorted(sections.keys()):
    alphas = sections[reason]

    output_lines.append("-" * 80)
    output_lines.append(f"Fail 사유: {reason} (총 {len(alphas)}개)")
    output_lines.append("-" * 80)

    for alpha in alphas:
        output_lines.append(f"  {alpha['expression']}")
        for detail in alpha['details']:
            output_lines.append(f"    {detail}")

    output_lines.append("")
    output_lines.append("")

# SUMMARY
output_lines.append("=" * 80)
output_lines.append("SUMMARY")
output_lines.append("=" * 80)
for reason in sorted(sections.keys()):
    output_lines.append(f"  {reason}: {len(sections[reason])}개")

# 파일 저장
with open(FAIL_ALPHA_FILE, 'w', encoding='utf-8') as f:
    f.write('\n'.join(output_lines))

print(f"\nRebuilt {FAIL_ALPHA_FILE.name}")
print(f"Total fail reasons: {len(sections)}")
print(f"Total fail expressions: {sum(len(alphas) for alphas in sections.values())}")
