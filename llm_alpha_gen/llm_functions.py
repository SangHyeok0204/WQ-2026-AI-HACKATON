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
## 🔧 MODIFIED: top_n_datafields 파라미터 추가 - 각 dataset에서 alphaCount 상위 N개만 사용
def generate_expressions_from_dataset(s, alpha_region, alpha_universe, dataset_id, model = 'gpt-5-mini-2025-08-07', datafields_num_cap = 500, alpha_num = 100, top_n_datafields = 3):

    # 데이터셋에서 데이터필드 불러오기
    data_fields = ace.get_datafields(s, region=alpha_region, universe=alpha_universe, dataset_id=dataset_id, data_type='ALL')
    data_fields = data_fields[data_fields['type'] != "GROUP"]

    # 🔧 NEW: alphaCount 기준 상위 N개 datafield만 선택
    top_fields_df = data_fields.nlargest(top_n_datafields, "alphaCount")

    # 🔧 NEW: LLM 입력용 compact datafield spec 생성
    spec_keys = ['id', 'description', 'subcategory', 'coverage', 'userCount', 'alphaCount', 'type']
    available_keys = [k for k in spec_keys if k in top_fields_df.columns]
    selected_datafields_spec = top_fields_df[available_keys].to_dict("records")

    print(f"[INFO] Selected top {len(selected_datafields_spec)} datafields for dataset '{dataset_id}':")
    for df_spec in selected_datafields_spec:
        print(f"  - {df_spec.get('id', 'N/A')} (alphaCount: {df_spec.get('alphaCount', 'N/A')}, type: {df_spec.get('type', 'N/A')})")

    # If needed get operators or other data
    operators = ace.get_operators(s)
    operators_list = [row.to_dict() for row in operators[operators['scope']=='REGULAR'].iloc]
    operator_exclude = []
    operators_list = [x for x in operators_list if x['name'] not in operator_exclude]

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
    # 🔧 MODIFIED: 프롬프트에 ALLOWED_DATAFIELDS 섹션 추가 및 KEEP_IN_MIND 업데이트
    prompt = f"""
    <MISSION>
    Based on the OPERATORS and ALLOWED_DATAFIELDS below,
    1. SUGGEST {alpha_num} SEPARATED and DIVERSIFIED Alpha ideas which can create excess return in the market
    2. Divide your idea into 4 parts, "idea", "description", "implementation", and "target".
    3. "idea" must contain the core idea of the alpha.
    4. "description" must include how you considered SUGGESTIONS and KEEP_IN_MINDs, and how much you set the confidence of this alpha. Consider SUGGESTIONS and write the reason you applied or didn't apply each of the SUGGESTIONS.
    5. "implementation" must contain implementation of the variation, including specific template (in OPERATOR(MATRIX) form) or exact name of datafield.
    6. "confidence_level" should be numerical value set in description, in 0-1 scale. Bigger value means higher confidence of this alpha idea.
    Very new user can understand the idea and how to implement this idea.
    </MISSION>

    <SUGGESTIONS>
    SUGGESTION1: Actively use MULTIPLE datafields from ALLOWED_DATAFIELDS. Your main mission is finding great combinations of the allowed datafields.
    SUGGESTION2: If a datafield has type=VECTOR, wrap it with vec_avg() or vec_sum() operator.
    SUGGESTION3: If the datafield's coverage is lower than 0.6, try using ts_backfill() to preprocess the data.
    SUGGESTION4: datafields' userCount and alphaCount are the count of users and alphas submitted. Try to use high user and alphaCount so that you can catch the signal easily.
    </SUGGESTIONS>

    <KEEP_IN_MIND>
    KEEP_IN_MIND1: Final implementation MUST be simple. recommend to use operator "ts_backfill" , "ts_zscore" more often
    KEEP_IN_MIND2: Final implementation MUST NOT contain over 4 operators and over 2 datafields.
    KEEP_IN_MIND3: You MUST ONLY use datafield IDs from ALLOWED_DATAFIELDS. Using any datafield outside this list is INVALID.
    KEEP_IN_MIND4: You CANNOT use type=GROUP field by itself. You need to use it as "group" parameter in Group operator.
    KEEP_IN_MIND5: If a datafield has type=VECTOR, it MUST be wrapped with vec_avg() or vec_sum() operator.
    </KEEP_IN_MIND>

    <OPERATORS>
    You can use those operators: {operators_list}
    </OPERATORS>

    <ALLOWED_DATAFIELDS>
    Available datafields = {selected_datafields_spec}
    </ALLOWED_DATAFIELDS>

    <ANSWER_FORMAT>
    You must answer in this form
    {answer_form}
    </ANSWER_FORMAT>
    """.strip()

    #description = await call_llm(prompt)

    result_json = call_llm_stream(prompt,answer_form, model)
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