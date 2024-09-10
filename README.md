# Pre-training data selection for biomedical domain adaptation using journal impact metrics

Here is the code repository for our [article published in the BioNLP 2024 Workshop (ACL 2024)](https://aclanthology.org/2024.bionlp-1.27.pdf).

## Installation
We performed our experiments with Python 3.11 and CUDA 12.1; the required packages are listed in [requirements.txt](./requirements.txt). To install the packages with pip run `pip install -r requirements.txt`, and with conda run `conda env create -f environment.yml`

## Get data
To download the data, from the current projet dir run : 
```bash
cd data
mkdir scimago pubmed
python dl_scimago.py
python dl_and_parse_pubmed.py
```
The PubMed data is downloaded from the PubMed Baseline data that was last Updated January 12, 2024. The complete baseline consists of files pubmed24n0001.xml through pubmed24n1219.xml.
ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline.

## Prepare data
Then to prepare the data for training with the HuggingFace `datasets` format :
```bash
cd data
python prepare_data.py
```
This script will :
- tokenize all abstracts
- remove unused columns for training
- shuffle the whole dataset abstracts
- split the data in train/validation splits (by default: 5% of data for validation)

Note : it requires to load the whole dataset in memory, so approximately 70GB of RAM is needed.

## Continual pretraining of the model

To run pretraining, we adapted the HuggingFace Transformers `run_mlm_offline.py` script , so that we can filter the dataset before training. We also removed the parts that requires internet communication so that it can run offline (for distributed running on a SLURM cluster for example).

## BibTeX citation

```
@inproceedings{lai-king-paroubek-2024-pre,
    title = "Pre-training data selection for biomedical domain adaptation using journal impact metrics",
    author = "Lai-king, Mathieu  and
      Paroubek, Patrick",
    editor = "Demner-Fushman, Dina  and
      Ananiadou, Sophia  and
      Miwa, Makoto  and
      Roberts, Kirk  and
      Tsujii, Junichi",
    booktitle = "Proceedings of the 23rd Workshop on Biomedical Natural Language Processing",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.bionlp-1.27",
    pages = "363--369",
    abstract = "Domain adaptation is a widely used method in natural language processing (NLP) to improve the performance of a language model within a specific domain. This method is particularly common in the biomedical domain, which sees regular publication of numerous scientific articles. PubMed, a significant corpus of text, is frequently used in the biomedical domain. The primary objective of this study is to explore whether refining a pre-training dataset using specific quality metrics for scientific papers can enhance the performance of the resulting model. To accomplish this, we employ two straightforward journal impact metrics and conduct experiments by continually pre-training BERT on various subsets of the complete PubMed training set, we then evaluate the resulting models on biomedical language understanding tasks from the BLURB benchmark. Our results show that pruning using journal impact metrics is not efficient. But we also show that pre-training using fewer abstracts (but with the same number of training steps) does not necessarily decrease the resulting model{'}s performance.",
}
```
