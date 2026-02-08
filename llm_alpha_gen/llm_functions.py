import os
import random
import ace_lib as ace
import asyncio
from openai import OpenAI
from dotenv import load_dotenv

import json

load_dotenv('.env.local')

_llm_instance = None


## LLM의 답변을 실시간으로 스트리밍하면서 출력해주는 함수
def call_llm_stream(prompt, json_schema,
                    model="gpt-5-2025-08-07",
                    ):
    """
    스트리밍으로 중간 토큰을 실시간 출력하고, 최종 응답 전체 문자열을 반환.
    """
    load_dotenv(".env.local")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    full_text = ""
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a quantitative finance expert. Your task is to aid users step-by-step, "
                        "based on the provided context. Respond ONLY with a valid JSON object that conforms "
                        "to the provided schema. Do not add any other text or explanations. "
                        f"Follow this json schema: {json_schema}"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            stream=True,
        )

        # 스트림은 sync iterator
        for chunk in stream:
            if hasattr(chunk, "choices") and chunk.choices:
                delta = getattr(chunk.choices[0], "delta", None)
                if delta is not None:
                    piece = getattr(delta, "content", None)
                    if piece:
                        print(piece, end="", flush=True)
                        full_text += piece

        print()
        return full_text

    except Exception as e:
        print(f"Error calling LLM (stream): {e}")
        return None


## 데이터셋을 가져와서 프롬프트로 만들어주고 LLM까지 넣어주는 함수
## Seed-alpha 전용 프롬프트로 LLM 호출
## 70-80% 4가지 패턴 + 20-30% 자유 operator 조합
def generate_expressions_from_dataset(s, alpha_region, alpha_universe, dataset_id, model = 'gpt-5-mini-2025-08-07', datafields_num_cap = 500, alpha_num = 100, top_n_datafields = 3):

    # 데이터셋에서 데이터필드 불러오기
    data_fields = ace.get_datafields(s, region=alpha_region, universe=alpha_universe, dataset_id=dataset_id, data_type='ALL')
    data_fields = data_fields[data_fields['type'] != "GROUP"]

    # alphaCount 기준 상위 N개 datafield만 선택
    top_fields_df = data_fields.nlargest(top_n_datafields, "alphaCount")

    # LLM 입력용 compact datafield spec
    spec_keys = ['id', 'description', 'coverage', 'userCount', 'alphaCount', 'type']
    available_keys = [k for k in spec_keys if k in top_fields_df.columns]
    selected_datafields_spec = top_fields_df[available_keys].to_dict("records")

    print(f"[INFO] Selected top {len(selected_datafields_spec)} datafields for dataset '{dataset_id}':")
    for df_spec in selected_datafields_spec:
        print(f"  - {df_spec.get('id', 'N/A')} (alphaCount: {df_spec.get('alphaCount', 'N/A')}, type: {df_spec.get('type', 'N/A')})")

    # Load operators list for FREE_PATTERN section
    operators_dict = import_json('operators_list.json')
    ops_by_category = {}
    for op_name, op_info in operators_dict.items():
        cat = op_info.get('category', 'Other')
        if cat not in ops_by_category:
            ops_by_category[cat] = []
        ops_by_category[cat].append(op_info.get('definition', op_name))
    operators_compact = "\n".join(f"[{cat}] " + " | ".join(defs) for cat, defs in ops_by_category.items())

    answer_form = '''{
"results":[
    {
        "idea":...,
        "description":...,
        "implementation":...,
        "confidence_level":...
    }
]
}'''

    prompt = f"""
<MISSION>
Based on the ALLOWED_PATTERNS, FREE_PATTERN, and ALLOWED_DATAFIELDS below,
generate {alpha_num} SEPARATED and DIVERSIFIED simple seed-alpha expressions.

Each result must have 4 parts:
- "idea": core idea of the alpha
- "description": which pattern you used, which datafield, which window parameter, and why
- "implementation": the FASTEXPR expression
- "confidence_level": 0-1 scale, higher = more confident
</MISSION>

<GENERATION_RATIO>
- About 70-80% of generated alphas MUST follow one of the 4 ALLOWED_PATTERNS exactly.
- About 20-30% of generated alphas may use FREE_PATTERN: creative combinations of operators from OPERATORS_LIST.
</GENERATION_RATIO>

<ALLOWED_PATTERNS>
These 4 patterns are the primary building blocks (use for ~70-80% of alphas):

Pattern 1 - Backfill: ts_backfill(DATA, window)
  Example: ts_backfill(rsk60_offer, 252)
  Use when: coverage < 0.6, to fill missing data

Pattern 2 - Time-series Z-score: ts_zscore(DATA, window)
  Example: ts_zscore(mdl138_4idpc, 252)
  If DATA is VECTOR type, wrap with vec_avg(): ts_zscore(vec_avg(rsk60_offer), 252)
  Use when: normalize signal over time

Pattern 3 - Ratio Z-score: ts_zscore(divide(DATA, cap), window)
  Example: ts_zscore(divide(fnd6_revenue, cap), 252)
  Use when: fundamental data should be scaled by market cap

Pattern 4 - Group Z-score with rank: group_zscore(rank(DATA), industry)
  Or with ratio: group_zscore(rank(divide(DATA, cap)), industry)
  Use when: cross-sectional normalization within industry groups
</ALLOWED_PATTERNS>

<FREE_PATTERN>
For about 20-30% of generated alphas, you may create expressions using any combination
of operators from OPERATORS_LIST below. Be creative and explore different operator combos.

These free-form expressions must still follow STRUCTURAL LIMITS:
- Maximum 3 operators per expression
- Only 1 signal datafield (cap, industry, sector, subindustry are helpers only)
- VECTOR type fields must use vec_* operators (vec_avg, vec_sum, vec_max, etc.)
- Time-series window parameters: use values between 5 and 252

Examples of free patterns:
- ts_rank(DATA, 63)
- quantile(ts_delta(DATA, 21))
- ts_decay_linear(DATA, 126)
- zscore(ts_mean(DATA, 63))
- group_neutralize(rank(DATA), industry)
- ts_ir(DATA, 126)
- winsorize(ts_zscore(DATA, 63))
- rank(ts_kurtosis(DATA, 126))
- group_rank(ts_av_diff(DATA, 63), industry)
</FREE_PATTERN>

<OPERATORS_LIST>
{operators_compact}
</OPERATORS_LIST>

<STRICT_RULES>
RULE1: You MUST ONLY use datafield IDs from ALLOWED_DATAFIELDS. No other datafields.
RULE2: Each expression must use exactly ONE signal datafield (cap, industry, sector, subindustry are allowed as helpers only).
RULE3: If a datafield has type=VECTOR, you MUST wrap it with vec_avg() or other vec_* operators before using.
RULE4: Window parameters should be diverse: use various values between 5 and 252.
RULE5: Maximum 3 operators per expression. Keep it simple.
RULE6: You CANNOT use type=GROUP field by itself. Use it only as group parameter (e.g., industry in group_zscore).
</STRICT_RULES>

<DIVERSITY_GUIDANCE>
- Use ALL available datafields, not just the top one
- Vary window parameters (5, 10, 21, 42, 63, 126, 189, 252, etc.)
- Mix ALLOWED_PATTERNS and FREE_PATTERN
- For VECTOR fields, always wrap with vec_* operators
- For low-coverage fields (coverage < 0.6), prefer Pattern 1 (ts_backfill)
</DIVERSITY_GUIDANCE>

<ALLOWED_DATAFIELDS>
{selected_datafields_spec}
</ALLOWED_DATAFIELDS>

<ANSWER_FORMAT>
You must answer in this form:
{answer_form}
</ANSWER_FORMAT>
""".strip()

    result_json = call_llm_stream(prompt, answer_form, model)
    return result_json

def save_json(json_dict, path):
    with open(path,"w") as file:
        json.dump(json_dict, file, indent=4)

def import_json(path):
    return json.loads(open(path, 'r', encoding='utf-8').read())

def cut_first_to_last_brace(s:str) -> str | None:
    start = s.find('{')
    end = s.rfind('}')
    if start == -1 or end == -1 or start > end:
        return None
    return s[start:end+1]