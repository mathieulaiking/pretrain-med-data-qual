import os
import copy
import gzip
import json
import random
import datasets
import requests
import argparse
import jsonlines
import xml.etree.ElementTree as ET

from tqdm import tqdm
from io import BytesIO
from bisect import bisect_left

logger = datasets.logging.get_logger(__name__)

# PUBMED BASELINE 2024 CONSTANTS
_BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/pubmed24n"
_END_INDEX = 1219

# FILLED automatically from features
SIMPLE_KEYS = {"PubmedArticleSet"}
LIST_KEYS = {"PubmedArticle"}
IGNORE_KEYS = set()

# features 
Date = {
    "Year": datasets.Value("int32"),
    "Month": datasets.Value("int32"),
    "Day": datasets.Value("int32"),
}
Journal = {
    'ISSN': datasets.Value('string'),
}
Article = {
    "Journal": Journal,
    "Abstract": {"AbstractText": datasets.Value("string")},
    "ArticleTitle": datasets.Value("string"),
    "Language": datasets.Value("string"),
}
features = datasets.Features({
            "MedlineCitation": {
                "PMID": datasets.Value("int32"),
                "DateCompleted": Date,
                "Article": Article,
            },
})

def parse_args():
    parser = argparse.ArgumentParser(
        description="""Download PubMed Baseline 2024 , 
        filter abstracts that do not have text, not in english, or do not have a journal ISSN,
        then parse the xml files and save them in jsonl format.
        The shards will not have the same number of articles because of the filtering, 
        we did not change this to be able to resume the download more easily if it was stopped 
        or crashed for any reason 
        """
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./pubmed',
        help='Output directory where shards will be stored (same name as pubmed_baseline urls)'
    )
    parser.add_argument(
        '--scimago_dir',
        type=str,
        default="./scimago",
        help='Directory containing scimago data (downloaded using dl_scimago.py)'
    )
    return parser.parse_args()
    
def download_and_extract(url):
    r = requests.get(url, allow_redirects=True)
    with gzip.open(BytesIO(r.content), 'rb') as f:
        file_content = f.read()
    return file_content

def default_date():
    return {"Year": 0, "Month": 0, "Day": 0}
            
def default_inline_article():
    return {
        'Journal': {'ISSN': ""},
        "Abstract": {"AbstractText": ""},
        "ArticleTitle": "",
        "Language": "",
    }

def default_article():
    return {
        "MedlineCitation": {
            "PMID": 0,
            "DateCompleted": default_date(),
            "Article": default_inline_article(),
        }
    }

# Copyright Ferry Boender, released under the MIT license.
# Modified by @Narsil to handle more oddities
def deepupdate(target, src):
    """Deep update target dict with src
    For each k,v in src: if k doesn't exist in target, it is deep copied from
    src to target. Otherwise, if v is a list, target[k] is extended with
    src[k]. If v is a set, target[k] is updated with v, If v is a dict,
    recursively deep-update it.

    Examples:
    >>> t = {'name': 'Ferry', 'hobbies': ['programming', 'sci-fi']}
    >>> deepupdate(t, {'hobbies': ['gaming']})
    >>> print(t)
    {'name': 'Ferry', 'hobbies': ['programming', 'sci-fi', 'gaming']}
    """
    for k, v in src.items():
        if k in target and isinstance(target[k], int) and isinstance(v, str):
            try:
                v = int(v)
            except Exception:
                pass
        if k in target and type(target[k]) != type(v):
            logger.warning(f"Ignoring field {k} it's a {type(v)} and we expect a {type(target[k])}")
            continue

        if isinstance(v, list):
            if k not in target:
                target[k] = copy.deepcopy(v)
            elif isinstance(target[k], list):
                target[k].extend(v)
            elif isinstance(target[k], str):
                # Very special case to handle `AbstractText` which sometimes end up
                # being a list.
                new_v = " ".join(el for el in v if isinstance(el, str))
                target[k] = new_v
            else:
                logger.warning(f"Ignoring field {k} it's a {type(v)} and we expect a {type(target[k])}")
        elif isinstance(v, dict):
            if k not in target:
                target[k] = copy.deepcopy(v)
            elif isinstance(target[k], dict):
                deepupdate(target[k], v)
            else:
                logger.warning(f"Ignoring field {k} it's a {type(v)} and we expect a {type(target[k])}")
        elif isinstance(v, set):
            if k not in target:
                target[k] = v.copy()
            elif isinstance(target[k], set):
                target[k].update(v.copy())
            else:
                logger.warning(f"Ignoring field {k} it's a {type(v)} and we expect a {type(target[k])}")
        else:
            if isinstance(target[k], (list, tuple, dict)):
                logger.warning(f"Ignoring field {k} it's a {type(v)} and we expect a {type(target[k])}")
                continue

            target[k] = copy.copy(v)

def fill_keys_from_features(features):
    if isinstance(features, dict):
        for key, value in features.items():
            if isinstance(value, datasets.Sequence):
                LIST_KEYS.add(key)
                fill_keys_from_features(value.feature)
            else:
                SIMPLE_KEYS.add(key)
                fill_keys_from_features(value)

def xml_to_dictionnary(parentElement):
    data = {}
    if parentElement.tag in {"AbstractText", "ArticleTitle"}:
        # XXX
        # Very special case, it will contain html leading to having very odd structure
        tag = parentElement.tag
        string = ET.tostring(parentElement).decode("utf-8").strip()
        inner_string = string[len(f"<{tag}>") : -len(f"</{tag}>")]
        return {parentElement.tag: inner_string}

    for child in list(parentElement):
        child.text = child.text if (child.text is not None) else " "
        key = child.tag
        if len(child) == 0:
            value = child.text.strip()
        else:
            value = xml_to_dictionnary(child)
            if isinstance(value, dict) and set(value.keys()) == {key}:
                value = value[key]
        if key in data:
            old_value = data[key]
            if isinstance(old_value, dict):
                data[key] = [old_value, value]
            elif isinstance(old_value, list):
                data[key].append(value)
        elif key in LIST_KEYS:
            data[key] = [value]
        elif key in SIMPLE_KEYS:
            data[key] = value
        elif key in IGNORE_KEYS:
            continue
        else:
            logger.info(f"Ignoring key {key} from {parentElement.tag}")
            IGNORE_KEYS.add(key)
    return {parentElement.tag: data}

def parse_single_xml(file):
    """Yields examples."""
    try:
        tree = ET.parse(file)
        root = tree.getroot()
        xmldict = xml_to_dictionnary(root)
    except ET.ParseError:
        logger.warning(f"Ignoring file {file}, it is malformed")
        yield

    for article in xmldict["PubmedArticleSet"]["PubmedArticle"]:
        new_article = default_article()
        try:
            deepupdate(new_article, article)
        except Exception:
            logger.warning(f"Ignoring article {article}, it is malformed")
            continue
        try:
            _ = features.encode_example(new_article)
        except Exception as e:
            logger.warning(f"Ignore example because {e}")
            continue
        yield new_article

def is_valid_article(article):
    valid = True
    if not article["MedlineCitation"]["Article"]["Abstract"]["AbstractText"]:
        valid=False
    if not article["MedlineCitation"]["Article"]["Journal"]["ISSN"]:
        valid=False
    if not article["MedlineCitation"]["Article"]["Language"] == "eng":
        valid=False
    return valid    

# to get closest element to a value in a list 
def take_closest(sorted_list, value):
    pos = bisect_left(sorted_list, value)
    if pos == 0:
        return sorted_list[0]
    if pos == len(sorted_list):
        return sorted_list[-1]
    before = sorted_list[pos - 1]
    after = sorted_list[pos]
    if after - value < value - before:
        return after
    else:
        return before

def main():
    args = parse_args()
    urls = [f"{_BASE_URL}{str(i).zfill(len(str(_END_INDEX)))}.xml.gz"
            for i in range(1, _END_INDEX + 1)]

    # load journal quality metrics and init counters
    sjr_dict = json.load(open(os.path.join(args.scimago_dir,"issn_sjr.json")))
    hind_dict = json.load(open(os.path.join(args.scimago_dir,"issn_h-index.json")))
    # set seed for random generation
    random.seed(0)
    # start download and parsing
    fill_keys_from_features(features)
    output_buffer = []
    for url in tqdm(urls):
        out_filepath = os.path.join(args.output_dir, f"{url.split('/')[-1]}".replace(".xml.gz", ".jsonl"))
        if os.path.exists(out_filepath):
            continue
        xml_content = BytesIO(download_and_extract(url))
        for article in parse_single_xml(xml_content):
            if not is_valid_article(article):
                continue
            ## get journal metrics
            issn = article["MedlineCitation"]["Article"]["Journal"]["ISSN"].replace("-","")
            if issn not in hind_dict:
                h_index = 0
                sjr = 0
            else : 
                # h_index same whatever the year
                h_index = hind_dict[issn]
                # sjr depends on year
                years = [int(y) for y in sjr_dict[issn]]
                closest_year = take_closest(years, article["MedlineCitation"]["DateCompleted"]["Year"])
                sjr = sjr_dict[issn][str(closest_year)]
            ## random assignment (for baseline comparison) using uniform distribution between 0 and 1
            random_seed0 = random.random()

            ## write output file and save metrics in list
            output_buffer.append({
                "id":article["MedlineCitation"]["PMID"],
                "text":article["MedlineCitation"]["Article"]["ArticleTitle"]+ "\n" + article["MedlineCitation"]["Article"]["Abstract"]["AbstractText"],
                "h-index":h_index,
                "sjr":sjr,
                "random_0":random_seed0,
            })

        # saving abstracts
        with jsonlines.open(out_filepath, mode="w") as outfile:
            outfile.write_all(output_buffer)

if __name__ == "__main__":
    main()