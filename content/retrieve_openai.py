import argparse
import json
import os
from pathlib import Path
from openai import OpenAI
from utils import get_openai_key

def parse_jsonl(file):
    data_list = [json.loads(line) for line in file.split('\n') if line.strip()]
    return data_list

if __name__ == '__main__':
    content_path = Path('./')
    dset_config = json.load(open(content_path / 'config.json'))

    parser = argparse.ArgumentParser(description='Retrieve OpenAI batch job results')
    parser.add_argument(
        'dataset',
        type=str,
        nargs='?',
        default='oli_computing',
        choices=list(dset_config.keys()) + ['all'],
        help='Dataset key to retrieve, or "all". Must match what was used for Infer_openai. Choices: '
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

    batch_info_path = content_path / f'{"-".join(target_dsets)}_{args.model_name}_batch_info.json'
    jobs = json.load(open(batch_info_path))
    job_ids = [job['id'] for job in jobs]
    if len(job_ids) != len(target_dsets):
        raise SystemExit(
            f"{batch_info_path} has {len(job_ids)} job(s) but dataset '{args.dataset}' implies {len(target_dsets)}. "
            "Use the same dataset argument as for Infer_openai."
        )

    client = OpenAI(api_key=get_openai_key())

    for dset, job_id in zip(target_dsets, job_ids):
        batch = client.batches.retrieve(job_id)
        print(dset, batch.status)
        if batch.status == 'completed':
            success = client.files.content(batch.output_file_id)
            success_list = parse_jsonl(success.text)
            with open(content_path / 'resources' / dset / f'{args.model_name}_success.jsonl', 'w') as file:
                for l in success_list:
                    json_l = json.dumps(l)
                    file.write(json_l + '\n')
        else:
            success_list = []
        if batch.error_file_id:
            failure = client.files.content(batch.error_file_id)
            failure_list = parse_jsonl(failure.text)
            with open(content_path / 'resources' / dset / f'{args.model_name}_failure.jsonl', 'w') as file:
                for l in failure_list:
                    json_l = json.dumps(l)
                    file.write(json_l + '\n')
        else:
            failure_list = []
        print(f'For {dset}, total success: {len(success_list)}, failure: {len(failure_list)}')
