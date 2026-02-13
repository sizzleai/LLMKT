# Updated instructions

## 1. Get data

1. Download zip files from `s3://sizzle-data/CMU_datashop/` and unzip.
2. Move `tx` and `problem_content` folders into `content/resources/[subject]` folder 
3. Update `config.json` with the new path names

## 2. Process data

1. The original README below explains how to extract images from swf files. I skipped this step.
2. Run the parsing script. This script creates a standardized JSON of problem data from the HTML files in the dataset.
    ```bash
    python parse_data.py [subject]
    ```
    `subject` is `oli_statics`, `oli_computing`, etc.
    
    The original script is designed to work with the `all` argument, but I did not try this.

    The script should generate a file such as `resources/oli_statics/parsed_steps.json` and print something like

    ```
    Processing oli_statics..
    Unmatched_problems: 14 / Original problems: 300
    ```
3. Run the batching script for the OpenAI call. 
    ```bash
    python openai_batch.py [subject] --model [openai model]  # default gpt-4o to match the paper
    ```
    `subject` is `oli_statics`, `oli_computing`, etc. `openai_model` is `gpt-4o` to match the paper. This will create a JSON lines file: `resources/[subject]/[openai_model]_batch.jsonl`. Each line contains all the info for the OpenAI request.


## 3. Generate KCs

1. Run inference on the batched file. Make sure you have an env var `OPENAI_API_KEY` set.
    ```bash
    python Infer_openai.py [subject] --model [openai model]  # default gpt-4o to match the paper
    ```
    `subject` is `oli_statics`, `oli_computing`, etc. `openai_model` should match what you used for openai_batch. It should output how many batches it sent.

    It will also generate a batch_info file like `oli_statics_gpt-4o_batch_info.json` with information about the submitted request.

2. Retrieve inference results.
    ```bash
    python retrieve_openai.py [subject] --model [openai model]  # default gpt-4o to match the paper
    ```
    Use the same `subject` and `openai_model` as for Infer_openai. It should print a status (`in_progress`, `completed`) and create two files after completing: `resources/[subject]/[openai_model]_success.jsonl` and `resources/[subject]/[openai_model]_failure.jsonl`. I think failures are meant to be retried separately, but I've been ignoring them since there's just a couple.

## Cluster KCs

1. Generate KC embeddings:
    ```bash
    python calc_embedding.py [subject] [embedding_model]
    ```
    `subject` is `oli_statics`, `oli_computing`, etc. `embedding_model` is `t5` or `openai_3` (default `openai_3`). Use `--inference_model` to match the model used for inference (default `gpt-4o`). Use `--num_workers` to control parallel embedding requests (default 10).

    This will call OpenAI and create a file with the embeddings: `resources/[subject]/processed_[embedding_model]_embeddings.json`

2. Run clustering:
    ```bash
    python kc_clustering.py [subject] [embedding_model]
    ```
    `subject` and `embedding_model` should match what you used for calc_embedding. This runs K means clustering with different numbers of clusters, then evaluates them with the elbow method and silhouette score. A chart will be generated:

    ![Clustering scores for KC clustering (elbow & silhouette)](figures/openai_3_oli_statics_clustering_score.png)

## Run AFM

1. Run the datashop form script. This maps the clustered KCs onto the transaction data and outputs CMU datashop format files, which you need for AFM.
    ```bash
    python generate_datashop_form.py [subject] [embedding_model]
    ```
    `subject` is `oli_statics`, `oli_computing`, etc. `embedding_model` is `t5` or `openai_3` (default `openai_3`). Use `--inference_model` to match the model you used for inference (default `gpt-4o`).

    It creates `resources/[subject]/[inference_model]_datashop_form.txt` (transaction file with KC tags), `[inference_model]_datashop_form-rollup.txt` (roll-up format), and `[inference_model]_content_data.json`. Requires pyAFM installed for the roll-up step.

2. Clone and set up the repo: https://github.com/cmaclell/pyAFM. The author's [blog post](https://chrismaclellan.com/blog/modeling-student-learning-in-python) was kind of helpful.

3. Run AFM on the datashop files. Two scripts share the same data loading (transaction roll-up and parsing) but do different things:
    - **process_datashop**: Fits AFM or AFM+S, runs cross-validation, prints CV scores and KC coefficients.
    - **plot_datashop**: Fits the same models and generates learning curve plots (actual vs predicted error by opportunity).

    Example (run from the pyAFM directory). Use the rollup file, not the transaction file:
    ```bash
    python pyafm/process_datashop.py -m AFM ../LLMKT/content/resources/oli_statics/gpt-4o_datashop_form-rollup.txt
    ```
    The script prompts you to choose a KC model, select "Ours".

    Process datashop will print a bunch of stuff including the RMSE scores:

    | Unstratified CV | Stratified CV | Student CV | Item CV |
    |----------------|---------------|------------|---------|
    | 0.412          | 0.412         | 0.491      | 0.423   |

    (This is slightly worse than what they report in the paper for oli_statics, it could be the lack of image processing)

    For plot_datashop (learning curve plots):
    ```bash
    python pyafm/plot_datashop.py ../LLMKT/content/resources/oli_statics/gpt-4o_datashop_form-rollup.txt -m AFM
    ```
    Use the rollup file. Args: `-m` = which model to plot (`AFM`, `AFM+S`, or `both`); `-p` = graph type (`overall`, `individual_kcs`, or `both`); `-ft` = file type (`student_step` for rollup, `transaction` for raw datashop form). Prompts for KC model, select "Ours".

### Comments

An interesting thing I noticed is that even though the average RMSE for "ours" is worse than for human generated KCs (oli_computing with KCs poc_1_10), the learning curve for the actual data looks much smoother. This could mean that the LLM-generated KCs are more coherent and represent meaningful skills. It could also just be that there are a lot more LLM-KCs. Something to keep an eye on! We could potentially explore measures of learning curve smoothness as an additional metric for KC quality.

## Run KT

Didn't get to this yet


# Original instructions

# Preparing the dataset

To reproduce the experiment, there are two options.
1. Download preprocessed data from CMU datashop (recommanded)
2. Preprocess data on your own

We recommand the first option since step 2 requires extracting flash files.

You need to request access to these datasets.

[OLI Engineering Statics](https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=507)
[OLI Principles of Computing](https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=1806)
[OLI French](https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=918)
[OLI Biology](https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=1148)
[OLI Psychology](https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=863)

## 1. Download preprocessed data



## 2. Reproduce preprocessing procedure

1. Download content data and transaction data from the CMU datashop. After you get dataset access, download datas and modify the 'config.json' to their path. Placing the unzipped files into /resources/{Name of Dataset} is recommanded and is the defaut option.


2. extract image files from swf files. Make sure that all content data are located under resources folder.

```bash
./resources/all_parse.cmd
```

To run this, an swf extractor, which can parse flash files into image files, is required. In our case, we used swfextract.exe obtained from http://www.swftools.org/swftools-0.9.2.tar.gz, which requires Window OS. 

3. run preprocessing 

```bash
python parse_data.py all
```
This will produce 'resources/{dataset_name}/parsed_data.json'. This file is a parsed content data, which allows you to use them to your further exploration.

4. Inference GPT by openai API.

```bash
python openai_batch.py
```
This will generate files for openai batch inference. The jsonl file is a format that supports OpenAI batch inference. 

```bash
python infer_openai.py
```

You can modify infer_openai.py and use your preferred multimodal LLMs. In our case, we used GPT-4o with batch inference. Which costs about 40$ in total.
Note that some of the resources violate OPENAI content policy, with unknown reason. For these failure cases, we will load all images from the respective step using the PIL library and save them again for retry. In such cases, the alpha channel of the images gets lost, resulting in some degradation. However, it has been observed that GPT can still recognize the images. 

```bash
python reterive_openai.py
```

This will recieve infernece result.

5. Obtain text encoding
```
python calc_embedding.py
```

For clustering, run
```python
python kc_clustering.py all openai_3
```

# Reproduce experiments

```bash
python create_sil_test_dset.py
```
this will generate test file for AFM clustering test.

```bash
python generate_datashop_form.py
```
this will generate cmu-datashop form transaction files, and also roll-up file.

```bash
python generate_kt_tsv.py Ours
```
this will convert cmu datashop form into the format of kt_benchmark repo. 