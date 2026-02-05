import base64
import requests
import json
import os
from PIL import Image
import numpy as np
from utils import *

def load_and_save_image(image_path, output_path):
    image = Image.open(image_path)
    data = np.array(image)
    modified_image = Image.fromarray(data)
    modified_image.save(output_path, 'PNG')

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def change_option_text(text):
    return text.split('(value:')[0]

def step_part2text(step, part):
    text= 'Question:\n'
    step_type2explanation = {
        'oli-fill-in-the-blank': f"Fill the blank on {step['key_str']}",
        'oli-multiple-choice': f"Choose correct option on {step['key_str']}",
        'oli-image-hotspot': f"Click right hotspot from image on {step['key_str']}",
        'oli-short-answer': f"Write your answer on {step['key_str']}",
        'oli-numeric': f"Write your numeric answer on {step['key_str']}",
        'oli-text': f"Text response on {step['key_str']}",
    }
    text += step_type2explanation.get(
        step['step_type'],
        f"Answer the question on {step['key_str']}",
    )
    if 'options' in step:
        if type(step['options'][0]) == str:
            text += '\nOptions:\n' + '\n'.join(step['options'])
        else:
            text += '\nOptions:\n' + '\n'.join([change_option_text(o['text']) for o in step['options']])
    answer = None
    for res in part['responses']:
        try:
            if int(res['score']) > 0:
                answer_val = res.get('match')
                answer= None
                for o in step['options']:
                    if o['value'] == answer_val:
                        answer = change_option_text(o['text'])
        except:
            answer = None
    
    if answer:        
        text += '\nAnswer:\n' + answer
    text += '\nFeedbacks:\n' + '\n'.join([f"{r.get('name')}: {r.get('text')}" for r in part['responses'] if r['class'][0] !='oli-no-response'])
    return text

def get_image_resources_base(config: dict, dataset_key: str, file_path: str | None = None) -> str:
    """Build image resources directory from config: content_data_path/subject/resources.
    If file_path is given, use the subject from that path (so images are looked up in the
    same subject as the question); otherwise use the first subject (single-subject datasets).
    """
    entry = config[dataset_key]
    content_path = entry["content_data_path"].replace("/", os.sep)
    if file_path:
        normalized_fp = os.path.normpath(file_path).replace("/", os.sep)
        if normalized_fp.startswith("." + os.sep):
            normalized_fp = normalized_fp[2:]
        if normalized_fp.startswith(content_path):
            remainder = normalized_fp[len(content_path) :].lstrip(os.sep)
            if remainder:
                subject = remainder.split(os.sep)[0]
                return os.path.join(entry["content_data_path"], subject, "resources")
    subject = entry["subjects"][0]
    return os.path.join(entry["content_data_path"], subject, "resources")


def recursive_decompose(images, parsed_list, image_resources_base: str):
    for idx, content in enumerate(parsed_list):
        if content['type'] == 'text':
            for img_obj in images:
                if img_obj['key_str'] in content['text']:
                    basename = os.path.basename(img_obj['text'])
                    image_path = os.path.join(image_resources_base, basename)
                    if not os.path.isfile(image_path):
                        return recursive_decompose(images, [
                            ({'type':'text', 'text':c['text'].replace(img_obj['key_str'],'')} if c['type']=='text' else c )
                            for c in parsed_list
                        ], image_resources_base)
                    base64_image = encode_image(image_path)
                    if not base64_image:
                        print(f"{os.path.basename(image_path)} is empty.")
                        return recursive_decompose(images, [
                            ({'type':'text', 'text':c['text'].replace(img_obj['key_str'],'')} if c['type']=='text' else c )
                            for c in parsed_list
                        ], image_resources_base)
                    new_parsed_list = parsed_list[:idx] + [
                        {
                            'type': 'text',
                            'text': content['text'].split(img_obj['key_str'])[0]
                        },
                        {
                            'type': 'image_url',
                            "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        },
                        {
                            'type': 'text',
                            'text': content['text'].split(img_obj['key_str'])[1]
                        },
                    ]+ parsed_list[idx+1:]
                    return recursive_decompose(images, new_parsed_list, image_resources_base)
    return parsed_list
            

def question_to_prompt(question_obj, image_resources_base: str):
    prompts = []
    prompt_contents = [{
            "type": "text",
            "text": "Content:\n\n"+question_obj['question']
    }]
    prompt_contents = recursive_decompose(question_obj['images'], prompt_contents, image_resources_base)
    for step in question_obj['steps']:
        step_prompt = [_ for _ in prompt_contents]
        matched_part = None
        for part in question_obj['parts']:
            if part['step_id'] == step['step_id']:
                matched_part = part
        if not matched_part:
            raise ValueError('No part matched')
        step_prompt.append(
            {
                'type':'text',
                'text':step_part2text(step, matched_part)
            }
        )
        prompts.append(step_prompt)
    return prompts

def parsed_data2batch_list(parsed_data, config: dict, dataset_key: str):
    batch_list = []
    for idx, question_content in enumerate(parsed_data):
        image_resources_base = get_image_resources_base(
            config, dataset_key, question_content.get("file_path")
        )
        user_conts = question_to_prompt(question_content, image_resources_base)
        for step_idx, user_cont in enumerate(user_conts):
            payload = {
                "model": "gpt-4-turbo",
                'response_format':{ "type": "json_object" },
                'seed': 42,
                "messages": [
                    {
                        "role": 'system',
                        'content': [
                            {
                                'type': 'text',
                                'text': "Extract the knowledge components required to solve this question." \
                                "Each knowledge component has two fields, name of 2 to 4 words and description of 1 sentence."\
                                "Output is in json format, like\n ```json\n{'knowledge_components':[{'name':'knowledge component 1', 'description': 'understand how to apply kc 1'}]}```"
                            }
                        ]
                    },
                    {
                        "role": "user",
                        "content": user_cont
                    }
                ],
                "max_tokens": 500
            }
            batch_list.append({
                "custom_id": f"request-{idx}-{step_idx}", 
                "method": "POST", 
                "url": "/v1/chat/completions",
                "body": payload,
            })

    return batch_list

if __name__ == '__main__':
    import argparse
    with open('config.json') as f:
        config = json.load(f)
    parser = argparse.ArgumentParser(description='Build batch list from parsed steps')
    parser.add_argument(
        'dataset',
        type=str,
        choices=list(config.keys()),
        nargs='?',
        default='oli_statics',
        help='Dataset key (must match parse_data.py argument). Choices: ' + ', '.join(config.keys()),
    )
    args = parser.parse_args()
    parsed_data = json.load(open('parsed_steps.json'))
    batch_list = parsed_data2batch_list(parsed_data, config, args.dataset)
    with open('gpt4_batch.jsonl', 'w') as file:
        for b in batch_list:
            json_b = json.dumps(b)
            file.write(json_b + '\n')
