import argparse
import datasets

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "cache_dir",
        type=str,
        help="Path to cache directory where datasets will be stored"
    )
    parser.add_argument(
        "--num_proc",
        default=None,
        type=str,
        help="number of processors to load dataset"
    )
    return parser.parse_args()


_DATASETS_CONFIGS = [
    # ner
    ("bigbio/blurb","bc5chem"),
    ("bigbio/blurb","bc5disease"),
    ("bigbio/blurb","bc2gm"),
    ("bigbio/blurb","jnlpba"),
    ("bigbio/blurb","ncbi_disease"),
    # pico
    ("bigbio/ebm_pico", None),
    # relation extraction
    ("bigbio/chemprot", None),
    ("bigbio/ddi_corpus", None),
    ("bigbio/gad", None),
    # sentence similarity
    ("bigbio/biosses", None),
    # document classification
    ("bigbio/hallmarks_of_cancer", None),
    # question answering
    ("bigbio/bioasq_task_b",None),
    ("bigbio/pubmed_qa",None),
]


def main():
    args = parse_args()
    # download NER datasets
    for dataset, config in _DATASETS_CONFIGS:
        datasets.load_dataset(
            dataset,
            name=config,
            cache_dir=args.cache_dir,
            num_proc=args.num_proc,
        )