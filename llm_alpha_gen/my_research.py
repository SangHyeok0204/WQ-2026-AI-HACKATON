import llm_functions as llm
from parser import *
import ace_lib as ace
import numpy as np
import pandas as pd
import os 
import json
import asyncio



datafields = llm.import_json('datafield/1/USA/TOP3000/1_USA_TOP3000_total.json')
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



if __name__ == "__main__":
    main()
