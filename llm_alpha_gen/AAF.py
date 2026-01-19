import helpful_functions as hf
import ace_lib as ace
import json
import os
import time
import traceback
from datetime import datetime
from tqdm import tqdm

# ============================================================================
# WHITELIST: 대회 규정상 허용된 dataset만 다운로드
# ============================================================================
ALLOWED_DATASETS = {
    "USA": [
        'socialmedia8', 'option10', 'option22', 'shortinterest36', 'fundamental85',
        'option13', 'news5', 'news79', 'fundamental91', 'model57', 'sentiment18',
        'analyst9', 'news31', 'model240', 'analyst48', 'news50', 'macro10',
        'analyst12', 'fundamental21', 'other323', 'model54', 'other492',
        'shortinterest3', 'model239', 'analyst2', 'socialmedia3', 'other696',
        'macro52', 'model46', 'news54'
    ],
    "EUR": [
        'risk60', 'risk88', 'other696', 'analyst39', 'news79', 'model39',
        'earnings3', 'news104', 'model28', 'news23', 'risk59', 'shortinterest3',
        'model139', 'model30', 'news17', 'model138', 'model109', 'news46',
        'model25', 'other455', 'model176', 'shortinterest37', 'socialmedia39',
        'other197'
    ],
    "ASI": [
        'shortinterest37', 'pv13', 'fundamental72', 'other455', 'fundamental23',
        'model109', 'analyst69', 'fundamental28', 'risk74', 'risk88', 'other463',
        'other675', 'fundamental6', 'model106', 'model77', 'analyst44', 'analyst9',
        'analyst48', 'model262', 'fundamental17', 'fundamental44', 'model16',
        'other128', 'fundamental21', 'model138', 'model53'
    ],
    "GLB": [
        'shortinterest37', 'fundamental46', 'pv53', 'other463', 'pv13',
        'fundamental6', 'analyst82', 'model16', 'risk74', 'model106', 'analyst9',
        'news76', 'risk88', 'analyst11', 'fundamental72', 'other675', 'analyst44',
        'news52', 'fundamental45', 'fundamental7', 'analyst48', 'model262',
        'other169', 'model162', 'other128', 'pv30', 'fundamental44', 'analyst35'
    ]
}


def datafields_dict(s, dataset_id, settings):
    """get datafields in the dataset and return dict of it"""
    datafields_df = ace.get_datafields(
        s,
        region=settings['region'],
        delay=settings['delay'],
        universe=settings['universe'],
        dataset_id=dataset_id,
        data_type="ALL"
    )
    datafields_df['id_ind'] = datafields_df['id']
    datafields_dict = datafields_df.set_index('id_ind').T.to_dict()
    return datafields_dict


def save_dict_to_json(data, filename):
    """get dict and save it in form of json"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)


def mkdir(directory):
    """checking if the directory exists or not, create if not"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def settings_to_datafield_folder(settings):
    return f"./datafield/{settings['delay']}/{settings['region']}/{settings['universe']}"


def check_initialized(settings):
    """v1.3 initialize 확인"""
    file_folder = settings_to_datafield_folder(settings)
    return os.path.exists(file_folder)


def settings_to_datafield_file_name(settings, dataset_name):
    return f"{settings['delay']}_{settings['region']}_{settings['universe']}_{dataset_name}.json"


def get_datafield_filepath(settings, dataset_id):
    """dataset_id에 해당하는 json 파일 경로 반환"""
    dataset_name = settings_to_datafield_file_name(settings, dataset_id)
    file_folder = settings_to_datafield_folder(settings)
    return f"{file_folder}/{dataset_name}"


def log_error(settings, dataset_id, error, log_type="error"):
    """에러 또는 이벤트를 로그 파일에 기록"""
    file_folder = settings_to_datafield_folder(settings)
    mkdir(file_folder)
    log_file = f"{file_folder}/errors.log"

    timestamp = datetime.now().isoformat()
    region = settings.get('region', 'UNKNOWN')
    universe = settings.get('universe', 'UNKNOWN')
    delay = settings.get('delay', 'UNKNOWN')

    if log_type == "error":
        exc_type = type(error).__name__
        exc_message = str(error)
        tb_summary = ''.join(traceback.format_tb(error.__traceback__)[-2:]).strip() if error.__traceback__ else "N/A"

        log_entry = {
            "timestamp": timestamp,
            "log_type": "error",
            "region": region,
            "universe": universe,
            "delay": delay,
            "dataset_id": dataset_id,
            "exception_type": exc_type,
            "exception_message": exc_message,
            "traceback_summary": tb_summary
        }
    elif log_type == "skipped_existing":
        log_entry = {
            "timestamp": timestamp,
            "log_type": "skipped_existing",
            "region": region,
            "universe": universe,
            "delay": delay,
            "dataset_id": dataset_id
        }
    else:
        log_entry = {
            "timestamp": timestamp,
            "log_type": log_type,
            "region": region,
            "universe": universe,
            "delay": delay,
            "dataset_id": dataset_id,
            "message": str(error)
        }

    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


def datafields_to_json(s, dataset_id, settings, skip_existing=True):
    """주로 쓰게 될 함수 - datafield를 json으로 저장"""
    filepath = get_datafield_filepath(settings, dataset_id)
    file_folder = settings_to_datafield_folder(settings)

    mkdir(file_folder)

    # 이미 존재하는 파일 스킵
    if skip_existing and os.path.exists(filepath):
        log_error(settings, dataset_id, None, log_type="skipped_existing")
        return None

    datafields = datafields_dict(s, dataset_id, settings)
    save_dict_to_json(datafields, filepath)

    return datafields


def get_all_file_paths(directory):
    """모든 파일 경로를 반환"""
    for root, dirs, files in os.walk(directory):
        for file in files:
            yield os.path.join(root, file)


def read_json(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print("JSON read failed:", filepath)
        raise


def merge_dicts(dict_list):
    merged_dict = {}
    for dict_item in dict_list:
        merged_dict.update(dict_item)
    return merged_dict


def make_total_dicts(settings):
    """모든 datafield json을 합쳐서 total.json 생성"""
    total_dataset_name = settings_to_datafield_file_name(settings, 'total')
    file_folder = settings_to_datafield_folder(settings)
    filename = f"{file_folder}/{total_dataset_name}"

    all_files = list(get_all_file_paths(settings_to_datafield_folder(settings)))
    files_except_total = [x for x in all_files if 'total' not in x and x.endswith('.json')]

    if not files_except_total:
        print("No datafield files found to merge.")
        return

    total_json = merge_dicts([read_json(x) for x in files_except_total])
    save_dict_to_json(total_json, filename)


def find_datafield(s, datafield_name, settings):
    total_dataset_name = settings_to_datafield_file_name(settings, 'total')
    file_folder = settings_to_datafield_folder(settings)
    filename = f"{file_folder}/{total_dataset_name}"

    total_dict = read_json(filename)
    if datafield_name in total_dict.keys():
        return total_dict[datafield_name]
    else:
        datafields = datafields_to_json(s, datafield_dataset(datafield_name), settings)
        return datafields[datafield_name]


def datafield_dataset(datafield_name):
    category_short = datafield_name.split('_')[0].translate(str.maketrans('', '', '0123456789'))
    category_full = category_short_to_full[category_short]
    dataset_id = datafield_name.split('_')[0].replace(category_short, category_full)
    return dataset_id


def initiate_datafield(s, settings, delay_seconds=2, max_retries=5, skip_existing=True):
    """
    whitelist에 있는 dataset만 다운로드하고, 에러 발생 시 로그에 기록

    Args:
        s: ace session
        settings: 설정 dict (region, delay, universe 등)
        delay_seconds: API 호출 간 딜레이 (rate limit 방지)
        max_retries: 최대 재시도 횟수
        skip_existing: True면 이미 존재하는 파일 스킵
    """
    region = settings['region']
    universe = settings.get('universe', 'TOP3000')
    delay = settings.get('delay', 1)

    # whitelist 가져오기
    whitelist = ALLOWED_DATASETS.get(region, [])
    if not whitelist:
        print(f"[WARNING] No whitelist found for region '{region}'. Skipping all datasets.")
        return

    # 전체 dataset 가져오기
    datasets = ace.get_datasets(s, region=region, delay=delay, universe=universe)
    all_dataset_ids = list(datasets.id)

    # whitelist에 있는 것만 필터링
    target_datasets = [d for d in all_dataset_ids if d in whitelist]
    skipped_datasets = [d for d in all_dataset_ids if d not in whitelist]

    print(f"\n{'='*60}")
    print(f"Region: {region} | Universe: {universe} | Delay: {delay}")
    print(f"Total datasets available: {len(all_dataset_ids)}")
    print(f"Whitelisted datasets to process: {len(target_datasets)}")
    print(f"Skipped (not in whitelist): {len(skipped_datasets)}")
    print(f"{'='*60}\n")

    success_count = 0
    skip_count = 0
    error_count = 0

    # tqdm으로 진행 상황 표시
    pbar = tqdm(target_datasets, desc="Processing datasets", unit="dataset")

    for dataset_id in pbar:
        pbar.set_postfix_str(f"Current: {dataset_id}")

        # 이미 존재하는 파일 체크
        filepath = get_datafield_filepath(settings, dataset_id)
        if skip_existing and os.path.exists(filepath):
            log_error(settings, dataset_id, None, log_type="skipped_existing")
            skip_count += 1
            pbar.set_postfix_str(f"Skipped (exists): {dataset_id}")
            continue

        # 재시도 로직
        for attempt in range(max_retries):
            try:
                datafields = datafields_dict(s, dataset_id, settings)
                save_dict_to_json(datafields, filepath)
                success_count += 1
                time.sleep(delay_seconds)
                break

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = delay_seconds * (2 ** attempt)
                    pbar.write(f"[RETRY] {dataset_id}: {type(e).__name__} - waiting {wait_time}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    # 모든 재시도 실패 - 로그에 기록하고 스킵
                    log_error(settings, dataset_id, e, log_type="error")
                    error_count += 1
                    pbar.write(f"[ERROR] {dataset_id}: {type(e).__name__}: {str(e)[:50]}...")

    # 모든 dataset 처리 후 total.json 생성
    print(f"\n{'='*60}")
    print("Creating total.json...")
    try:
        make_total_dicts(settings)
        print("total.json created successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to create total.json: {e}")
        log_error(settings, "total", e, log_type="error")

    # 최종 요약
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully processed: {success_count}")
    print(f"Skipped (already exists): {skip_count}")
    print(f"Failed (logged to errors.log): {error_count}")
    print(f"{'='*60}\n")


category_short_to_full = {
    'anl': 'analyst',
    'brk': 'broker',
    'ern': 'earnings',
    'fnd': 'fundamental',
    'imb': 'imbalance',
    'insd': 'insiders',
    'inst': 'institutions',
    'mcr': 'macro',
    'mdl': 'model',
    "nws": 'news',
    'mws': 'news',
    'opt': 'option',
    'oth': "other",
    'rsk': 'risk',
    'snt': 'sentiment',
    'shrt': 'shortinterest',
    'scl': 'socialmedia',
    'pv': 'pv'
}
