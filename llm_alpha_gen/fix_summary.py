import re
from pathlib import Path
from collections import Counter

# 경로 설정
FAIL_ALPHA_FILE = Path(__file__).parent / "fail_alpha_categorized.txt"

# 파일 읽기
with open(FAIL_ALPHA_FILE, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 각 섹션별 개수 세기
section_counts = Counter()
current_section = None

for line in lines:
    # 섹션 헤더 찾기
    if line.startswith("Fail 사유:"):
        match = re.search(r'Fail 사유: ([A-Z_]+)', line)
        if match:
            current_section = match.group(1)

    # 표현식 라인 찾기 (2칸 들여쓰기, "-> "로 시작하지 않음)
    elif line.startswith("  ") and not line.strip().startswith("->") and current_section:
        # 구분선이 아닌 경우만
        if not line.strip().startswith("---"):
            section_counts[current_section] += 1

print("Section counts:")
for reason, count in sorted(section_counts.items()):
    print(f"  {reason}: {count}개")

# SUMMARY 섹션 전까지만 유지
new_lines = []
found_summary = False
for line in lines:
    if line.strip().startswith("=") and not found_summary:
        # 다음 줄이 SUMMARY인지 확인
        idx = lines.index(line)
        if idx + 1 < len(lines) and "SUMMARY" in lines[idx + 1]:
            found_summary = True
            break
    new_lines.append(line)

# 새로운 SUMMARY 추가
new_lines.append("\n")
new_lines.append("=" * 80 + "\n")
new_lines.append("SUMMARY\n")
new_lines.append("=" * 80 + "\n")
for reason, count in sorted(section_counts.items()):
    new_lines.append(f"  {reason}: {count}개\n")

# 파일 저장
with open(FAIL_ALPHA_FILE, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print(f"\nFixed SUMMARY section in {FAIL_ALPHA_FILE.name}")
print(f"Total fail reasons: {len(section_counts)}")
print(f"Total fail expressions: {sum(section_counts.values())}")
