# Data pruning for biomedical pretraining

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
ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline. It contains ~61M abstracts, the full parsed dataset weighs ~69.1GB.

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

To run pretraining, we adapted the HuggingFace Transformers `run_mlm.py` script , so that we can filter the dataset before training. We also removed the parts that requires internet communication so that it can run offline (for distributed running on a SLURM cluster for example).
