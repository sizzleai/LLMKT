import os
import json
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
from utils import get_openai_key, read_jsonl, convert_ndarrays

client = OpenAI(api_key=get_openai_key())
content_path = Path('./')


def _process_item(item, embedding_model, is_single):
    """Process a single item. Returns {'response': res, 'kcs': kcs} or None."""
    try:
        res = json.loads(item['response']['body']['choices'][0]['message']['content'])
    except Exception:
        return None

    if 'knowledge_components' not in res:
        print('exception_no_kc')
        return None

    raw_kcs = []
    if is_single:
        names = []
        descriptions = []
        for kc in res['knowledge_components']:
            if 'description' not in kc or 'name' not in kc:
                continue
            names.append(kc['name'])
            descriptions.append(kc['description'])

        combined_kc = {
            'name': '~~'.join(names),
            'description': '\n- '.join(['Knowledge required to solve the problem:'] + descriptions)
        }
        raw_kcs = [combined_kc]
    else:
        raw_kcs = res['knowledge_components']

    valid_kcs = [kc for kc in raw_kcs if kc.get('description') and kc.get('name')]
    for kc in raw_kcs:
        if kc not in valid_kcs:
            print(kc)

    if embedding_model == 't5':
        for kc in valid_kcs:
            kc['embedding'] = embedding_model.encode(kc['description'])
        kcs = valid_kcs
    elif embedding_model == 'openai_3':
        if not valid_kcs:
            kcs = []
        else:
            descriptions = [kc['description'] for kc in valid_kcs]
            try:
                r = client.embeddings.create(
                    model="text-embedding-3-large",
                    input=descriptions,
                    encoding_format="float",
                )
                for i, kc in enumerate(valid_kcs):
                    kc['embedding'] = r.data[i].embedding
                kcs = valid_kcs
            except (KeyError, IndexError) as e:
                print(f"Unexpected response from openai: {e}")
                kcs = []
    else:
        raise ValueError('Invalid model name')

    return {'response': res, 'kcs': kcs}


def get_embedding(dset, embedding_model='t5', inference_model='gpt-4-turbo', is_single=False, num_workers=10):
    jsonl = read_jsonl(content_path / 'resources' / dset/ f'{inference_model}_success.jsonl')

    print(f"{dset} : total {len(jsonl)} steps.")
    processed = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(
                lambda item: _process_item(item, embedding_model, is_single),
                jsonl,
            ),
            total=len(jsonl),
        ))
    processed = [r for r in results if r is not None]
    try:
        single_postfix = '_single' if is_single else ''
        json.dump(convert_ndarrays(processed), open(content_path / 'resources'/ dset/ f'processed_{embedding_model}{single_postfix}_embeddings.json','w'))
    except:
        return processed
    return processed


if __name__ == '__main__':
    dset_config = json.load(open(os.path.join(content_path,'config.json')))
    parser = argparse.ArgumentParser(description='Get dataset name')
    dataset_choices = list(dset_config.keys())
    parser.add_argument(
        'dataset', 
        type=str, 
        choices=dataset_choices + ['all'],
        default='oli_statics',
        nargs='?',
        help='The dataset string to be processed. Choices: ' + ', '.join(dataset_choices)
    )
    parser.add_argument(
        'embedding_model', 
        type=str, 
        choices=['t5', 'openai_3'],
        default='openai_3',
        nargs='?',
        help='select embedding model. t5 or openai_3 '
    )
    parser.add_argument(
        '--inference_model',
        type=str,
        default='gpt-4o',
        help='OpenAI inference model name (default: gpt-4o).',
    )
    parser.add_argument(
        '--single_kc',
        action='store_true',
        help='Disable multiple KCs'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=10,
        help='Number of workers for parallel embedding requests (default: 10)',
    )
    args = parser.parse_args()
    target_dsets = [args.dataset] if args.dataset != 'all' else dataset_choices
    for dset in target_dsets:
        get_embedding(dset, args.embedding_model, args.inference_model, args.single_kc, args.num_workers)