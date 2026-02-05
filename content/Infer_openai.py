import argparse
import os
import json
from pathlib import Path
from openai import OpenAI
from utils import get_openai_key

if __name__ == '__main__':
    content_path = Path('./')
    dset_config = json.load(open(content_path / 'config.json'))

    parser = argparse.ArgumentParser(description='Submit OpenAI batch jobs for KC inference')
    parser.add_argument(
        'dataset',
        type=str,
        nargs='?',
        default='oli_computing',
        choices=list(dset_config.keys()) + ['all'],
        help='Dataset key to process, or "all" for every dataset. Choices: '
        + ', '.join(list(dset_config.keys()) + ['all']),
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4-turbo',
        dest='model_name',
        help='OpenAI model name (default: gpt-4-turbo).',
    )
    args = parser.parse_args()
    target_dsets = list(dset_config.keys()) if args.dataset == 'all' else [args.dataset]

    client = OpenAI(api_key=get_openai_key())

    jobs = []
    for dset in target_dsets:
        batch_path = content_path / 'resources' / dset / f'{args.model_name}_batch.jsonl'
        if not batch_path.is_file():
            print(f"Skipping {dset}: {batch_path} not found")
            continue
        batch_input_file = client.files.create(
            file=open(batch_path, "rb"),
            purpose="batch",
        )
        batch_input_file_id = batch_input_file.id
        created_job = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": f"{dset} full inference"},
        )
        jobs.append(created_job.to_dict())

    print(f"Submitted {len(jobs)} jobs")
    batch_info_path = content_path / f'{"-".join(target_dsets)}_{args.model_name}_batch_info.json'
    print(f"Saving batch info to {batch_info_path}")
    json.dump(jobs, open(batch_info_path, 'w'))