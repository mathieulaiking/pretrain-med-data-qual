import argparse
import datasets
import numpy as np

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
        ("bigbio/chemprot", None),
        ("bigbio/ddi_corpus", None),
        ("bigbio/gad", None),
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
        help="Path to BioASQ data directory, because must be downloaded yourself with login on bioasq site"
    )
    args = parser.parse_args()
    # Task sanity check
    if args.task is not None and args.task not in _DATASETS_TASKS_CONFIGS:
        raise ValueError(f"Task {args.task} not found in available tasks, available tasks are: {_DATASETS_TASKS_CONFIGS.keys()}")
    return args

def _preprocess(ds:datasets.DatasetDict,task, num_proc:int):
    if task == "sentence_similarity": 
        # In BLURB paper (Gu et al.) they say they adopt the splits of the Peng et al. paper, 
        # which uses 80% train and 20% test , so we must concatenate train and validation : 
        ds_train = datasets.concatenate_datasets((ds["train"],ds["validation"]),split="train")
        ds = datasets.DatasetDict({"train":ds_train,"test":ds["test"]})
    elif task == "qa":
        def cast_label(examples):
            examples["answer"] = [ ans_list[0] for ans_list in examples["answer"]]
            return examples
        ds = ds.map(cast_label,batched=True,num_proc=num_proc)
        unique_answers = sorted(ds.unique('answer')["train"])
        ds = ds.cast_column("answer", datasets.ClassLabel(num_classes=len(unique_answers), names=unique_answers))
        ds = ds.remove_columns(["question_id","document_id","choices","type"])
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
            )
            # Preprocess dataset
            preproc_ds = _preprocess(ds, task, args.num_proc)
            # Save on disk
            out_dir = dataset.split('/')[-1]
            if config is not None and args.task == "token_classification":
                out_dir += f"_{config}"
            preproc_ds.save_to_disk(out_dir)
            

if __name__ == "__main__":
    main()