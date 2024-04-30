import argparse
import datasets
import numpy as np

_DATASETS_TASKS_CONFIGS = {
    "token_classification" : [
        ("bigbio/blurb","bc5chem"),
        ("bigbio/blurb","bc5disease"),
        ("bigbio/blurb","bc2gm"),
        ("bigbio/blurb","jnlpba"),
        ("bigbio/blurb","ncbi_disease"),
        ("bigbio/ebm_pico", None),
    ],

    "relation_extraction" : [
        ("bigbio/chemprot", None),
        ("bigbio/ddi_corpus", None),
        ("bigbio/gad", None),
    ],

    "sentence_similarity" : [
        ("bigbio/biosses", None),
    ],

    "document_classification" : [
        ("bigbio/hallmarks_of_cancer", None),
    ],

    "question_answering" : [
        ("bigbio/bioasq_task_b",None),
        ("bigbio/pubmed_qa",None),
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
    args = parser.parse_args()
    # Task sanity check
    if args.task is not None and args.task not in _DATASETS_TASKS_CONFIGS:
        raise ValueError(f"Task {args.task} not found in available tasks, available tasks are: {_DATASETS_TASKS_CONFIGS.keys()}")
    return args

def _preprocess(ds:datasets.DatasetDict, num_proc:int):
    if ds["train"].info.dataset_name == "biosses":
        ds=ds.map(
            lambda examples: examples | {"annotator_avg": np.mean([examples[k] for k in examples if "annotator" in k],axis=0)},
            batched=True,
            remove_columns=[f"annotator_{x}" for x in "abcde"],
            num_proc=num_proc
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
            print("Downloading dataset", dataset, "with config", config)
            ds = datasets.load_dataset(
                dataset,
                name=config,
                cache_dir=args.cache_dir,
                num_proc=args.num_proc,
            )
            # Preprocess dataset
            preproc_ds = _preprocess(ds,args.num_proc)
            # Save on disk
            out_dir = dataset.split('/')[-1]
            if config is not None:
                out_dir += f"_{config}"
            preproc_ds.save_to_disk(out_dir)
            

if __name__ == "__main__":
    main()