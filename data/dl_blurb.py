import argparse
import datasets

_DATASETS_TASKS_CONFIGS = {
    "ner" : [
        ("bigbio/blurb","bc5chem"),
        ("bigbio/blurb","bc5disease"),
        ("bigbio/blurb","bc2gm"),
        ("bigbio/blurb","jnlpba"),
        ("bigbio/blurb","ncbi_disease"),
    ],

    "pico":[
        ("bigbio/ebm_pico", None),
    ],

    "relation_extraction" : [
        ("bigbio/chemprot", "chemprot_bigbio_kb"),
        ("bigbio/ddi_corpus", "ddi_corpus_bigbio_kb"),
        ("bigbio/gad", "gad_blurb_bigbio_text"),
    ],

    "sentence_similarity" : [
        ("bigbio/biosses", "biosses_bigbio_pairs"),
    ],

    "document_classification" : [
        ("bigbio/hallmarks_of_cancer", None),
    ],

    "qa" : [
        ("bigbio/bioasq_task_b","bioasq_blurb_bigbio_qa"),
        ("bigbio/pubmed_qa","pubmed_qa_labeled_fold0_bigbio_qa"),
    ]
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Path to cache directory where datasets will be stored"
    )
    parser.add_argument(
        "--task",
        default=None,
        type=str,
        help="download only the datasets for the specified task"
    )
    parser.add_argument(
        "--num_proc",
        default=None,
        type=int,
        help="number of processors to load dataset"
    )
    parser.add_argument(
        "--bioasq_data_dir",
        default=None,
        type=str,
        help=("Path to BioASQ data directory, because must be downloaded yourself with login on bioasq site," 
        "must be absolute path, or it will try to search on the HuggingFace Hub")
    )
    args = parser.parse_args()
    # Task sanity check
    if args.task is not None and args.task not in _DATASETS_TASKS_CONFIGS:
        raise ValueError(f"Task {args.task} not found in available tasks, available tasks are: {_DATASETS_TASKS_CONFIGS.keys()}")
    return args

def _cast_label(examples):
            examples["answer"] = [ ans_list[0] for ans_list in examples["answer"]]
            return examples

def _dummyfication(examples):
    per_relation_id = []
    per_relation_texts = []
    per_relation_labels = []
    for i, rel_list in enumerate(examples["relations"]):
        text = examples["passages"][i][0]["text"][0]
        for rel in rel_list :
            e1 = [e for e in examples['entities'][i] if e["id"] == rel["arg1_id"]][0]
            e2 = [e for e in examples['entities'][i] if e["id"] == rel["arg2_id"]][0]
            start1,end1= e1["offsets"][0]
            start2,end2 = e2["offsets"][0]
            # find closest points to extract the sentence
            begin = 0
            for k in range(min(start1,start2),0,-1):
                if text[k] in ['.','!','?']:
                    begin = k+1 # to not have the point in the sentence we take next index
                    break
            finish = len(text)-1
            for k in range(max(end1,end2),len(text)):
                if text[k] in ['.','!','?']:
                    finish = k
                    break
            # Dummify the text
            dummy1,dummy2 = '@'+e1["type"]+'$','@'+e2["type"]+'$'
            if start1 < start2:
                dummy_text = text[begin:start1] + dummy1 + text[end1:start2] + dummy2 + text[end2:finish]
            else:
                dummy_text = text[begin:start2] + dummy2 + text[end2:start1] + dummy1 + text[end1:finish]
            
            per_relation_id.append(examples["document_id"][i] + '_' + rel["id"])
            per_relation_texts.append(dummy_text)
            per_relation_labels.append(rel["type"])
    return {"id":per_relation_id, "text":per_relation_texts, "label":per_relation_labels}


def _preprocess(ds:datasets.DatasetDict,task, num_proc:int):
    if task == "sentence_similarity": 
        # In BLURB paper (Gu et al.) they say they adopt the splits of the Peng et al. paper, 
        # which uses 80% train and 20% test , so we must concatenate train and validation : 
        ds_train = datasets.concatenate_datasets((ds["train"],ds["validation"]),split="train")
        ds = datasets.DatasetDict({"train":ds_train,"test":ds["test"]})
    elif task == "qa":
        ds = ds.map(_cast_label,batched=True,num_proc=num_proc)
        unique_answers = sorted(ds.unique('answer')["train"])
        ds = ds.cast_column("answer", datasets.ClassLabel(num_classes=len(unique_answers), names=unique_answers))
        ds = ds.remove_columns(["question_id","document_id","choices","type"])
    elif task == "relation_extraction" and "relations" in ds["train"].column_names:
        ds = ds.map(
            _dummyfication,
            batched=True,
            remove_columns=ds["train"].column_names
        )
    return ds

def main():
    args = parse_args()
    # download NER datasets
    for task, datasets_configs in _DATASETS_TASKS_CONFIGS.items():
        if args.task is not None and task != args.task:
            continue
        for dataset,config in datasets_configs:
            # Download dataset
            print("Downloading dataset", dataset, "with config(s)", config)
            ds = datasets.load_dataset(
                dataset,
                name=config,
                cache_dir=args.cache_dir,
                num_proc=args.num_proc,
                data_dir=args.bioasq_data_dir if dataset.startswith("bigbio/bioasq") else None,
                trust_remote_code=True if not dataset == "bigbio/ebm_pico" else None,
            )
            # Preprocess dataset
            preproc_ds = _preprocess(ds, task, args.num_proc)
            # Save on disk
            out_dir = dataset.split('/')[-1]
            if config is not None and args.task == "ner":
                out_dir += f"_{config}"
            preproc_ds.save_to_disk(out_dir)
            

if __name__ == "__main__":
    main()