import llm_functions as llm
from parser import *
import ace_lib as ace
import AAF
import numpy as np
import pandas as pd
import os
import json
import asyncio


# 전역 변수는 run_pipeline에서 동적으로 로드
datafields = None
operators = llm.import_json('operators_list.json')

def return_type(node, operators, datafields):
    if node.node_type == 'operator':
        return operators[node.value]['output']

    elif node.node_type == 'number':
        return "NUMBER"

    elif node.node_type == 'datafield':
        if node.value.lower() == 'nan':
            return "MATRIX"
        return datafields[node.value]['type']

    else:
        return "SPECIAL_ARGUMENT"


def check_input(operator_inputs, children_types, _debug = False):
    if _debug: print(operator_inputs)
    while len(operator_inputs) != 0:

        operator_input = operator_inputs.pop(0)
        if _debug: print(f"operator input: {operator_input}, remain inputs: {len(operator_inputs)}" )

        if operator_input == '_SPECIAL_ARGUMENT':
            if len(children_types) == 0:
                pass
            elif children_types[0] == '_SPECIAL_ARGUMENT':
                if _debug: print(operator_input)
                if _debug: print(children_types.pop(0))
            else:
                pass

        else:
            if len(children_types) == 0:
                return False

            children_type = children_types.pop(0)
            if operator_input == "MATRIX" and children_type == "NUMBER":
                pass
            elif operator_input != children_type:
                if _debug: print(operator_input ,children_type, operator_input == children_type)
                return False

    return True


def sanity_checker(exp, _debug=False):

    try:
        exp_tree = tree_node(exp)

        for node in [node for node in exp_tree.collect_all_nodes() if node.node_type == "operator"]:

            if check_input(eval(operators[node.value]['input']), [return_type(x,operators, datafields) for x in node.children]):
                if _debug: print(check_input(eval(operators[node.value]['input']), [return_type(x) for x in node.children]))
            else:
                return False
        if return_type(exp_tree,operators, datafields)=="MATRIX":
            return True

        else:
            return False
    except Exception as e:
        print(f"err: {e}")
        return False


def main():
    s = ace.start_session()

    region = "USA"
    universe = "TOP3000"

    datafields = llm.import_json(f'datafield/1/{region}/{universe}/1_{region}_{universe}_total.json')
    operators = llm.import_json('operators_list.json')


    datasets = ace.get_datasets(s, region =region, universe = universe)
    datasets = datasets.sort_values(['category_id','userCount'],ascending=False)

    userCounts = []
    DatasetNames = []
    datasets = datasets.sort_values(['category_id','userCount'],ascending=False).drop_duplicates(['id'])

    for category in datasets.category_id.unique():
        category_datasets = datasets[datasets.category_id == category]
        userCounts.append(category_datasets.userCount.reset_index(drop=True).map(lambda x: int(x) if x>0 else x))
        DatasetNames.append(category_datasets.id.reset_index(drop=True))

    DatasetNamesDF = pd.DataFrame(DatasetNames).T
    DatasetNamesDF.columns = datasets.category_id.unique()
    DatasetNamesDF.to_csv(f'./datasets/1/{region}/{universe}/1_{region}_{universe}_dataset.csv')
    datasets_list = [ x for x in sum([list(x) for x in DatasetNamesDF.values],start=[]) if not pd.isna(x)]
    datasets_list.remove('pv1')
    datasets_list = [d for d in datasets_list if d in AAF.ALLOWED_DATASETS.get(region, [])]
    datasets_dict = {row['id']: row.to_dict() for _, row in datasets.iterrows()}



    def datset_to_simnum(dataset):
        score = int(np.log10(datasets_dict[dataset]['userCount']+1) + datasets_dict[dataset]['fieldCount'])

        if score <= 3:
            return 1
        elif score <= 10:
            return 2
        elif score <= 50:
            return 3
        elif score <= 100:
            return 10
        else:
            return 20



    def get_json_num(dataset):

        json_list = [x for x in os.listdir('./gen_json/') if dataset in x]
        if len(json_list) == 0:
            return 0
        nums = [int(x.split('_')[-1].split('.')[0]) for x in json_list]

        return max(nums)+1

    def load_csv(path: str) -> pd.DataFrame:
        if not os.path.exists(path):
            return pd.DataFrame()
        df = pd.read_csv(path) # index_col=None(기본값): 인덱스 컬럼으로 읽지 않음
        # 과거에 인덱스가 저장되어 있으면 'Unnamed: 0' 같은 컬럼 제거
        df = df.loc[:, ~df.columns.str.match(r'^Unnamed: \d+$')]
        return df

    def save_csv(df: pd.DataFrame, path: str) -> None:
        df.to_csv(path, index=False)

    def append_and_save_csv(path: str, new_rows: pd.DataFrame) -> pd.DataFrame:
        base = load_csv(path)
        out = pd.concat([base, new_rows], ignore_index=True) # 인덱스 재부여
        save_csv(out, path) # index=False
        return out


    for dataset in datasets_list[1:]:

        simnum = datset_to_simnum(dataset)

        for i in range(simnum):

            print(f"---working on {dataset}---")
            dataset_category = datasets_dict[dataset]['category_id']

            batch_name = f"{region}_{universe}_{dataset_category}"

            # 리스트로 받아오기
            print("---generating alphas with llm---")
            try:
                response = llm.generate_expressions_from_dataset(s, region, universe, dataset, model = 'gpt-5-mini-2025-08-07', datafields_num_cap = 500 )
            except:
                continue

            print("---received response---")
            json_response = llm.cut_first_to_last_brace(response)
            alphas = json.loads(json_response)
            print(f"---alpha gened well---")

            # 일단 json 저장
            llm.save_json(alphas, f"./gen_json/{dataset}_{get_json_num(dataset)}.json")

            # 리스트로 돌리기 (데이터셋의 score에 따른) 횟수만큼


            print(f"---gened alpha batch {i}---")

            alpha_expressions = [x for x in alphas["results"] if sanity_checker(x['implementation'], _debug=False)]

            alpha_list = [ace.generate_alpha(regular =x['implementation'], region = region, universe = universe, neutralization="INDUSTRY") for x in alpha_expressions]

            len_alpha_list = len(alpha_list)
            
            print(f"eligible_alpha_count: {len_alpha_list}")

            tags_list = [['llm_test', f"confidence_{x['confidence_level']}"] for x in alpha_expressions]

            descs_list = [x['description'] for x in alpha_expressions]



            for rep in list(range(0,len_alpha_list,8)):
                if ace.check_session_timeout(s) < 1000:
                    s = ace.start_session()
                try:
                    print(f"---simulating alpha batch {rep} to {min(rep+8,len_alpha_list)}---")
                    sim_results = list(
                            ace.multi_simulate_alphas_map(s,    alpha_list[rep:min(rep+8,len_alpha_list)],
                                                                tags_list[rep:min(rep+8,len_alpha_list)],
                                                                descs_list[rep:min(rep+8,len_alpha_list)],   min(rep+8,len_alpha_list)-rep )

                            )

                    df = pd.DataFrame([ace.flatten_dict(x) for x in sim_results if 'id' in x.keys()])

                    append_and_save_csv(f"./results/{batch_name}.csv", df)
                except Exception as e:
                    print(e)



def run_pipeline(allowed_datasets, region="USA", universe="TOP3000", delay=1,
                  skip_existing_datafields=True, skip_existing_alphas=True, llm_model='gpt-5-mini-2025-08-07'):
    """
    허용된 dataset에 대해서만 전체 파이프라인을 실행:
    (A) datafields json 생성
    (B) datasets csv 생성
    (C) LLM 알파 생성 → sanity check → Brain simulation → results csv 저장

    Args:
        allowed_datasets: 처리할 dataset_id 리스트
        region: 지역 (USA, EUR, ASI, GLB)
        universe: 유니버스 (TOP3000 등)
        delay: 딜레이 설정 (기본 1)
        skip_existing_datafields: True면 이미 존재하는 datafield json 스킵
        skip_existing_alphas: True면 이미 알파 생성된 dataset 스킵 (gen_json에 파일 존재 여부로 판단)
        llm_model: LLM 모델명

    Returns:
        dict: 파이프라인 실행 결과
    """
    global datafields

    print(f"\n{'='*70}")
    print(f"  PIPELINE START: {region}/{universe}")
    print(f"  Allowed datasets: {allowed_datasets}")
    print(f"{'='*70}\n")

    # 세션 시작
    s = ace.start_session()

    # settings 구성
    settings = {
        'region': region,
        'universe': universe,
        'delay': delay
    }

    # =========================================================================
    # (A) datafields json 생성 (허용된 dataset만)
    # =========================================================================
    print("\n[STEP A] Generating datafields JSON for allowed datasets...")
    datafield_result = AAF.initiate_datafield_allowed(
        s, settings, allowed_datasets,
        skip_existing=skip_existing_datafields
    )

    # total.json 로드
    total_json_path = f'datafield/{delay}/{region}/{universe}/{delay}_{region}_{universe}_total.json'
    if os.path.exists(total_json_path):
        datafields = llm.import_json(total_json_path)
        print(f"Loaded total.json with {len(datafields)} datafields")
    else:
        print(f"[WARNING] total.json not found at {total_json_path}")
        datafields = {}

    # =========================================================================
    # (B) datasets csv 생성 (허용된 dataset만)
    # =========================================================================
    print("\n[STEP B] Generating datasets CSV for allowed datasets...")

    # Brain에서 전체 datasets 가져오기
    datasets = ace.get_datasets(s, region=region, universe=universe, delay=delay)
    datasets = datasets.sort_values(['category_id', 'userCount'], ascending=False).drop_duplicates(['id'])

    # allowed_datasets로 필터링
    all_dataset_ids = set(datasets.id)
    valid_allowed = [d for d in allowed_datasets if d in all_dataset_ids]
    skipped_datasets = [d for d in allowed_datasets if d not in all_dataset_ids]

    if skipped_datasets:
        print(f"[WARNING] Datasets not in Brain, skipped for CSV: {skipped_datasets}")

    # 필터된 datasets만 사용
    filtered_datasets = datasets[datasets['id'].isin(valid_allowed)]

    # datasets csv 저장
    dataset_dir = f'./datasets/{delay}/{region}/{universe}'
    os.makedirs(dataset_dir, exist_ok=True)
    dataset_csv_path = f'{dataset_dir}/{delay}_{region}_{universe}_dataset.csv'

    # 기존 형식 유지: category별로 나누어 저장
    userCounts = []
    DatasetNames = []
    for category in filtered_datasets.category_id.unique():
        category_datasets = filtered_datasets[filtered_datasets.category_id == category]
        userCounts.append(category_datasets.userCount.reset_index(drop=True).map(lambda x: int(x) if x > 0 else x))
        DatasetNames.append(category_datasets.id.reset_index(drop=True))

    if DatasetNames:
        DatasetNamesDF = pd.DataFrame(DatasetNames).T
        DatasetNamesDF.columns = filtered_datasets.category_id.unique()
        DatasetNamesDF.to_csv(dataset_csv_path)
        print(f"Saved datasets CSV: {dataset_csv_path} ({len(valid_allowed)} datasets)")
    else:
        print(f"[WARNING] No valid datasets to save in CSV")
        DatasetNamesDF = pd.DataFrame()

    # =========================================================================
    # (C) LLM 알파 생성 → sanity check → simulate → results csv 저장
    # =========================================================================
    print("\n[STEP C] Running LLM alpha generation pipeline...")

    # datasets_list 생성 (pv1 제외)
    if not DatasetNamesDF.empty:
        datasets_list = [x for x in sum([list(x) for x in DatasetNamesDF.values], start=[]) if not pd.isna(x)]
        if 'pv1' in datasets_list:
            datasets_list.remove('pv1')
    else:
        datasets_list = valid_allowed

    datasets_dict = {row['id']: row.to_dict() for _, row in filtered_datasets.iterrows()}

    def dataset_to_simnum(dataset):
        if dataset not in datasets_dict:
            return 1
        score = int(np.log10(datasets_dict[dataset]['userCount'] + 1) + datasets_dict[dataset]['fieldCount'])
        if score <= 3:
            return 1
        elif score <= 10:
            return 2
        elif score <= 50:
            return 3
        elif score <= 100:
            return 10
        else:
            return 20

    def get_json_num(dataset):
        if not os.path.exists('./gen_json/'):
            os.makedirs('./gen_json/')
            return 0
        json_list = [x for x in os.listdir('./gen_json/') if dataset in x]
        if len(json_list) == 0:
            return 0
        nums = [int(x.split('_')[-1].split('.')[0]) for x in json_list]
        return max(nums) + 1

    def has_existing_alpha(dataset):
        """dataset에 대해 이미 알파 JSON이 생성되었는지 확인"""
        if not os.path.exists('./gen_json/'):
            return False
        # dataset_0.json, dataset_1.json 등 패턴으로 확인
        json_list = [x for x in os.listdir('./gen_json/') if x.startswith(f"{dataset}_") and x.endswith('.json')]
        return len(json_list) > 0

    def load_csv(path: str) -> pd.DataFrame:
        if not os.path.exists(path):
            return pd.DataFrame()
        df = pd.read_csv(path)
        df = df.loc[:, ~df.columns.str.match(r'^Unnamed: \d+$')]
        return df

    def save_csv(df: pd.DataFrame, path: str) -> None:
        df.to_csv(path, index=False)

    def append_and_save_csv(path: str, new_rows: pd.DataFrame) -> pd.DataFrame:
        base = load_csv(path)
        out = pd.concat([base, new_rows], ignore_index=True)
        save_csv(out, path)
        return out

    # results 디렉토리 생성
    os.makedirs('./results', exist_ok=True)

    total_alphas_generated = 0
    total_alphas_simulated = 0
    skipped_existing_alpha_datasets = []

    for dataset in datasets_list:
        if dataset not in datasets_dict:
            print(f"[SKIP] {dataset}: not in datasets_dict")
            continue

        # 이미 알파 생성된 dataset 스킵
        if skip_existing_alphas and has_existing_alpha(dataset):
            print(f"[SKIP] {dataset}: already has generated alphas in ./gen_json/")
            skipped_existing_alpha_datasets.append(dataset)
            continue

        simnum = dataset_to_simnum(dataset)

        for i in range(simnum):
            print(f"\n---working on {dataset} (batch {i+1}/{simnum})---")
            dataset_category = datasets_dict[dataset]['category_id']
            batch_name = f"{region}_{universe}_{dataset_category}"

            # LLM으로 알파 생성
            print("---generating alphas with llm---")
            try:
                response = llm.generate_expressions_from_dataset(
                    s, region, universe, dataset,
                    model=llm_model,
                    datafields_num_cap=500
                )
            except Exception as e:
                print(f"[ERROR] LLM generation failed for {dataset}: {e}")
                continue

            print("---received response---")
            try:
                json_response = llm.cut_first_to_last_brace(response)
                if json_response is None:
                    print(f"[ERROR] Invalid JSON response for {dataset}")
                    continue
                alphas = json.loads(json_response)
            except json.JSONDecodeError as e:
                print(f"[ERROR] JSON parse failed for {dataset}: {e}")
                continue

            print(f"---alpha gened well---")

            # JSON 저장
            llm.save_json(alphas, f"./gen_json/{dataset}_{get_json_num(dataset)}.json")
            print(f"---gened alpha batch {i}---")

            # sanity check
            alpha_expressions = [
                x for x in alphas.get("results", [])
                if sanity_checker(x['implementation'], _debug=False)
            ]

            alpha_list = [
                ace.generate_alpha(
                    regular=x['implementation'],
                    region=region,
                    universe=universe,
                    neutralization="INDUSTRY"
                ) for x in alpha_expressions
            ]

            len_alpha_list = len(alpha_list)
            total_alphas_generated += len(alphas.get("results", []))
            print(f"eligible_alpha_count: {len_alpha_list}")

            tags_list = [['llm_test', f"confidence_{x['confidence_level']}"] for x in alpha_expressions]
            descs_list = [x['description'] for x in alpha_expressions]

            # 시뮬레이션
            for rep in list(range(0, len_alpha_list, 8)):
                if ace.check_session_timeout(s) < 1000:
                    s = ace.start_session()
                try:
                    print(f"---simulating alpha batch {rep} to {min(rep+8, len_alpha_list)}---")
                    sim_results = list(
                        ace.multi_simulate_alphas_map(
                            s,
                            alpha_list[rep:min(rep+8, len_alpha_list)],
                            tags_list[rep:min(rep+8, len_alpha_list)],
                            descs_list[rep:min(rep+8, len_alpha_list)],
                            min(rep+8, len_alpha_list) - rep
                        )
                    )

                    df = pd.DataFrame([ace.flatten_dict(x) for x in sim_results if 'id' in x.keys()])
                    total_alphas_simulated += len(df)
                    append_and_save_csv(f"./results/{batch_name}.csv", df)
                except Exception as e:
                    print(f"[ERROR] Simulation failed: {e}")

    # 최종 요약
    print(f"\n{'='*70}")
    print(f"  PIPELINE COMPLETE: {region}/{universe}")
    print(f"{'='*70}")
    print(f"  Datafields processed: {len(datafield_result.get('processed', []))}")
    print(f"  Datasets in CSV: {len(valid_allowed)}")
    print(f"  Datasets skipped (already has alphas): {len(skipped_existing_alpha_datasets)}")
    print(f"  Total alphas generated: {total_alphas_generated}")
    print(f"  Total alphas simulated: {total_alphas_simulated}")
    print(f"{'='*70}\n")

    return {
        'datafield_result': datafield_result,
        'datasets_processed': valid_allowed,
        'datasets_skipped_existing': skipped_existing_alpha_datasets,
        'alphas_generated': total_alphas_generated,
        'alphas_simulated': total_alphas_simulated
    }


if __name__ == "__main__":
    # 기본 실행: 기존 main() 또는 run_pipeline() 사용
    # main()

    # 예시: 허용된 dataset만 파이프라인 실행
    ALLOWED = [
        "analyst12", "fundamental85", "news31", "option10", "other455"
    ]
    run_pipeline(ALLOWED, region="USA", universe="TOP3000")
