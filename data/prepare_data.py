import os
import argparse
import jsonlines
from transformers import AutoTokenizer
from datasets import Dataset, Features, Value, disable_progress_bars, disable_caching

def parse_args():
    parser = argparse.ArgumentParser(description="Tokenize the whole Pubmed 2024 dataset, need at least 69.1GB of free space ")
    parser.add_argument("--data_dir", type=str, default="pubmed")
    parser.add_argument("--cache_dir", type=str, default=".datasets_cache")
    parser.add_argument("--output_dir", type=str, default="pubmed_preprocessed")
    parser.add_argument("--do_validation_split", action="store_true", help="if flag is set will split data in train/validation")
    parser.add_argument("--validation_size", type=float, default=0.01, help="Fraction of the data that will be in validation set (0<size<1)")
    parser.add_argument("--num_proc", type=int, default=8) # 8 cores is optimal number tested for loading at max speed
    parser.add_argument("--seed", type=int, default=42, help="Seed for shuffling")
    args = parser.parse_args()
    # arguments checks
    if not os.path.exists(args.data_dir):
        raise ValueError(f"Data directory {args.data_dir} does not exist")
    if os.path.isdir(args.output_dir):
        raise ValueError(f"Output directory {args.output_dir} already exists")
    if args.num_proc < 1:
        raise ValueError("num_proc must be at least 1")
    elif args.num_proc > os.cpu_count():
        raise ValueError(f"you only have {os.cpu_count()} cores, so num_proc must be at most {os.cpu_count()}")
    if args.do_validation_split and (args.validation_size >= 1 or args.validation_size <= 0):
        raise ValueError("validation_size should be a float between 1.0 and 0.0 (both excluded)")
    return args

def shard_jsonl_gen(shards):
    for shard in shards :
        with jsonlines.open(shard) as reader:
            for article in reader :
                yield article

_PARSED_PUBMED_FEATURES = Features({
    "id":Value("uint32"), # https://www.nlm.nih.gov/bsd/mms/medlineelements.html#pmid
    "title":Value("string"),
    "text":Value("string"),
    "h-index":Value("uint16"),
    "sjr":Value("float32")
})

def main():
    args = parse_args()
    disable_progress_bars()
    disable_caching()
    shards = [os.path.join(args.data_dir, f"pubmed24n{str(i).zfill(4)}.jsonl") 
              for i in range(1,1219+1)]
    # Loading 
    print("Loading dataset")
    ds = Dataset.from_generator(
        shard_jsonl_gen, 
        features=_PARSED_PUBMED_FEATURES,
        cache_dir=args.cache_dir, # loading always cache , even if disabled_caching is called
        gen_kwargs={"shards": shards},
        num_proc=args.num_proc,
    )
    # Tokenization
    print("Tokenizing dataset")
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased", cache_dir=".cache")
    def tokenize_and_count(examples):  
        if "title" in examples:
            text = [title + "\n" + text for title, text in zip(examples["title"], examples["text"])]
        else:
            text = examples["text"]
        return tokenizer(
            text, 
            return_special_tokens_mask=True,
        )
    ds = ds.map(
        tokenize_and_count,
        batched=True,
        num_proc=8,
        remove_columns=["id","title","text"],
    )
    # Shuffling 
    print("Shuffling whole dataset")
    ds = ds.shuffle(seed=args.seed)
    # (Optional) Validation split
    if args.do_validation_split :
        ds = ds.train_test_split(
            test_size=args.validation_size,
            shuffle=False # data is already shuffled 
        )
        ds["validation"] = ds.pop("test")
    # Saving
    print("saving dataset")
    ds.save_to_disk(args.output_dir)

if __name__ == "__main__":
    main()