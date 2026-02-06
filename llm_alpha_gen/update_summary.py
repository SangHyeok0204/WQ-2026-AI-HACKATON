import re
from pathlib import Path

# 경로 설정
FAIL_ALPHA_FILE = Path(__file__).parent / "fail_alpha_categorized.txt"

# 파일 읽기
with open(FAIL_ALPHA_FILE, 'r', encoding='utf-8') as f:
    content = f.read()

# 각 섹션별 개수 세기
sections = {}
pattern = r'Fail 사유: ([A-Z_]+) \(총 (\d+)개\)'
matches = re.findall(pattern, content)

for reason, count in matches:
    # 실제 표현식 개수 세기
    section_pattern = rf'Fail 사유: {reason}.*?(?=(?:Fail 사유:|SUMMARY|$))'
    section_match = re.search(section_pattern, content, re.DOTALL)

    if section_match:
        section_text = section_match.group(0)
        # "  " (2칸 들여쓰기)로 시작하고 "-> "를 포함하지 않는 줄 개수 세기
        expressions = re.findall(r'^  (.+?)$', section_text, re.MULTILINE)
        actual_count = sum(1 for expr in expressions if not expr.strip().startswith('->'))
        sections[reason] = actual_count
    else:
        sections[reason] = int(count)

print("Section counts:")
for reason, count in sorted(sections.items()):
    print(f"  {reason}: {count}개")

# SUMMARY 섹션 업데이트
summary_pattern = r'================================================================================\nSUMMARY\n================================================================================\n(.*?)(?:\n|$)'
new_summary = "=" * 80 + "\n"
new_summary += "SUMMARY\n"
new_summary += "=" * 80 + "\n"
for reason, count in sorted(sections.items()):
    new_summary += f"  {reason}: {count}개\n"

# 기존 SUMMARY 섹션 찾기
summary_match = re.search(summary_pattern, content, re.DOTALL)
if summary_match:
    # SUMMARY 섹션 교체
    new_content = content[:summary_match.start()] + new_summary
    # SUMMARY 이후 내용이 있으면 추가 (보통은 없음)
    remaining = content[summary_match.end():]
    if remaining.strip():
        new_content += remaining
else:
    # SUMMARY 섹션이 없으면 끝에 추가
    new_content = content.rstrip() + "\n\n" + new_summary

# 파일 저장
with open(FAIL_ALPHA_FILE, 'w', encoding='utf-8') as f:
    f.write(new_content)

print(f"\nUpdated SUMMARY section in {FAIL_ALPHA_FILE.name}")
