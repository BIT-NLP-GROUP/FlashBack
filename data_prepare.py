import random
import json
import os
import jsonlines
import inspect
import hashlib
import multiprocessing

from datasets import load_from_disk
from dataclasses import dataclass, field
from transformers import HfArgumentParser, set_seed, AutoTokenizer, AutoConfig
from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm

@dataclass
class PrepareArguments:
    dataset_dir: str = field(default=None)
    model_dir: str = field(default=None)
    index_dir: str = field(default=None)
    search_max_width: int = field(default=32,metadata={"help":("The maximum number of tokens used to search."),})
    recall_max_width: int = field(default=256,metadata={"help":("Used to truncate retrieved content."),})
    retrieve_k: int = field(default=1,metadata={"help":("Concatenate `k` retrieved content."),})
    retrieve_stride: int = field(default=32)
    save_dir: str=field(default=None)
    forbid_titles_dir: str=field(default=None, metadata={"help":("Used to filter out documents by titles"),})
    train_split: str=field(default=None)
    validation_split: str=field(default=None)
    max_pos_embeddings: int=field(default=None)
    split_batch_size: int=field(default=1000)
    retrieve_batch_size: int=field(default=1000)
    is_pile: bool=field(default=False)

    seed: int=field(default=42)

    def __post_init__(self):
        if self.dataset_dir is None:
            raise ValueError('--dataset_dir error.')
        
        if self.forbid_titles_dir is not None:
            with open(self.forbid_titles_dir, "r") as fp:
                self.forbid_titles = set([line.strip() for line in fp])
        else:
            self.forbid_titles = []

        if self.save_dir is None:
            raise ValueError("--save_dir error.")
        self.dict = {
            "dataset_dir": self.dataset_dir,
            "model_dir": self.model_dir,
            "index_dir": self.index_dir,
            "search_max_width": self.search_max_width,
            "recall_max_width": self.recall_max_width,
            "retrieve_k": self.retrieve_k,
            "retrieve_stride": self.retrieve_stride,
            "forbid_titles_dir": self.forbid_titles_dir,
            "train_split": self.train_split,
            "validation_split": self.validation_split,
            "max_pos_embeddings": self.max_pos_embeddings,
            "seed": self.seed,
            "is_pile": self.is_pile
        }

def main():
    
    parser = HfArgumentParser(PrepareArguments)

    args = parser.parse_args_into_dataclasses()[0]
    args_dir = args.save_dir + '/args.json'
    if os.path.exists(args_dir):
        with open(args_dir) as fp:
            history_args = json.load(fp)

        #check whether main() function has been changed.
        md5 = hashlib.md5()
        md5.update(inspect.getsource(main).encode('utf-8'))
        args.dict['function_hash'] = md5.hexdigest()
        
        if history_args == args.dict:
            print("No change.")
            return
    print(args)
    set_seed(args.seed)
    dataset = load_from_disk(args.dataset_dir)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    
    config = AutoConfig.from_pretrained(args.model_dir)
    if args.max_pos_embeddings is not None:
        max_pos_embeddings = args.max_pos_embeddings
    else:
        if "opt" in args.model_dir:
            max_pos_embeddings = config.max_position_embeddings
        elif "gpt2" in args.model_dir:
            max_pos_embeddings = config.n_positions
        else:
            raise ValueError("No implementation.")

    searcher = LuceneSearcher(args.index_dir)
    if not os.path.exists(args.save_dir + '/dataset'):
        os.makedirs(args.save_dir + '/dataset')
    for split in ["train", "validation"]:
        if split == "train":
            if args.train_split is None: continue
            else: new_split = args.train_split
        if split == "validation":
            if args.validation_split is None: continue
            else: new_split = args.validation_split
        print(f"process `{split}` spilt...")

        split_dir = args.save_dir + '/dataset/' + split + '.jsonl'
        with jsonlines.open(split_dir, 'w') as fp:
            start_position, end_position = 0, len(dataset[new_split])
            if split == "train" and args.is_pile:
                start_position = int(len(dataset[new_split]) * 0.8)
                end_position = int(len(dataset[new_split]) * 0.9)
            elif split == "validation" and args.is_pile:
                start_position = int(len(dataset[new_split]) * 0.9)
            for batch_pos in tqdm(range(start_position, end_position, args.split_batch_size)):
                raw_data = dataset[new_split][batch_pos: batch_pos + args.split_batch_size]["text"]
                raw_data = "".join([x if x else " \n" for x in raw_data])
                tokens = tokenizer(raw_data).input_ids 
                queries, pos = [], []

                for i in range(max_pos_embeddings, len(tokens), args.retrieve_stride):
                    if i + args.retrieve_stride >= len(tokens): break
                    query = tokenizer.decode(tokens[i - args.search_max_width: i])
                    queries.append(query)
                    pos.append(i)
                
                for i in range(0, len(queries), args.retrieve_batch_size):
                    queries_batch = queries[i: i+args.retrieve_batch_size]

                    recalls = searcher.batch_search(
                        queries_batch,
                        qids=[str(j) for j in range(len(queries_batch))],
                        k=args.retrieve_k * 2, # times 2 avoiding the number of recall is less than `k` due to retrieval
                        threads=multiprocessing.cpu_count()
                    )
                    for qid, res in recalls.items():
                        j = i + int(qid)
                        retrieved_docs_tokens = []
                        num_retrieved_docs = 0
                        for hit in res:
                            res_dict = json.loads(hit.raw)
                            docs = res_dict['contents']
                            if args.forbid_titles is not None:
                                title = docs.split("\n")[0]
                                if title.startswith('"') and title.endswith('"'): title = title[1:-1]
                                if title in args.forbid_titles: continue
                            retrieved_docs_tokens = retrieved_docs_tokens + tokenizer(docs).input_ids[:args.recall_max_width]
                            num_retrieved_docs = num_retrieved_docs + 1
                            if num_retrieved_docs >= args.retrieve_k:
                                break
                        target = tokens[pos[j]: pos[j] + args.retrieve_stride]
                        center_pos = random.randint(max_pos_embeddings // 2, max_pos_embeddings - args.retrieve_stride - 1)
                        source = tokens[pos[j] - center_pos + len(retrieved_docs_tokens): pos[j]]
                        if len(source) == 0:
                            raise ValueError(f"{len(retrieved_docs_tokens), pos[j], center_pos, max_pos_embeddings, num_retrieved_docs }")
                        fp.write({
                            "e": retrieved_docs_tokens,
                            "src": source,
                            "tgt": target
                        })
    with open(args.save_dir + '/args.json', 'w') as fp:
        json.dump(args.dict, fp=fp)
    print("Finish.")

if __name__  == '__main__':
    import time
    time_st = time.time()
    main()
    time_ed = time.time()
    print(f"Spend {time_ed - time_st}s.")
