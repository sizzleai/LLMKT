import pandas as pd
import json
from utils import *
import os
from pathlib import Path

content_path = Path('./')
dset_config = json.load(open(os.path.join(content_path,'config.json')))

# I mapped the name of the KC instead of KC_id, to provide more insight directly.
def number_duplicate_values(input_dict):
    from collections import Counter
    value_counts = Counter(input_dict.values())
    value_numbering = {}
    new_dict = {}

    for key, value in input_dict.items():
        if value_counts[value] == 1:
            new_dict[key] = value
        else:
            count = value_numbering.get(value, 0) + 1
            value_numbering[value] = count
            new_value = f"{value} {count}"
            new_dict[key] = new_value

    return new_dict

# Match and convert our KC into Datashop form, following the instruction https://pslcdatashop.web.cmu.edu/help?page=kcm
# Define the KC model by filling in the cells in the column KC (model_name), replacing "model_name" with a name for your new model.
# Assign multiple KCs to a step by adding additional KC (model_name) columns, placing one KC in each column. Replace "model_name" with the same model name you used for your new model; you will have multiple columns with the same header.
# Add additional KC models by creating a new KC (model_name) column for each KC model, replacing "model_name" with the name of your new model.
# Delete any KC model columns that duplicate existing KC models already in the dataset (unless you want to overwrite these).
# Do not change the values or headers of any other columns.
if __name__ == '__main__':
    for dset in dset_config.keys():
        print(f"Dset: {dset}")
        kc_template = pd.read_csv(os.path.join(content_path,'resources', dset, 'kc_template.txt'), sep='\t')
        
        kc_template = kc_template.iloc[:, :-1] # there is a empty column.
        content_data = json.load(open(os.path.join(content_path,'resources', dset, 'openai_3_content_data.json')))
        cluster_id2name = json.load(open(os.path.join(content_path,'resources', dset, 'openai_3_cluster_id2name.json')))
        cluster_id2datashop_ver_name = number_duplicate_values(cluster_id2name)
        max_tag_len = 0
        valid_cnt = 0
        for c in content_data:
            if 'db_step_name' not in c:
                continue
            # Create a boolean condition for filtering
            condition = (kc_template['Problem Name'] == c['problem_id']) & (kc_template['Step Name'] == c['db_step_name'])
            # Get the index of the rows that match the condition
            idx = kc_template.index[condition]
            if len(idx) == 0:
                continue
            valid_cnt += 1
            unique_tags = list(set(c['tags']))
            if max_tag_len < len(unique_tags):
                for i in range(max_tag_len, len(unique_tags)):
                    kc_template[f"KC (Moon)_{i}"] = None  # Define new column name
                max_tag_len = len(unique_tags)
            for i, tag in enumerate(unique_tags):
                # Use .loc to assign values directly to the DataFrame
                kc_template.loc[idx, f"KC (Moon)_{i}"] = cluster_id2datashop_ver_name[str(tag)]

        print(f"{valid_cnt} / {len(kc_template)} is matched.")

        for i in range(max_tag_len):
            kc_template.rename(columns={f"KC (Moon)_{i}":"KC (Moon)"}, inplace=True)
        kc_template.drop(columns=["KC (new KC model name)"],inplace=True)
        kc_template.to_csv(os.path.join(content_path,'resources', dset, 'openai_3_kc_model_datashop_form.txt'), sep='\t')
