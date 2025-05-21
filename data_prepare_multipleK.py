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
    context_type: str = field(default=None)
    save_dir: str=field(default=None)
    forbid_titles_dir: str=field(default=None, metadata={"help":("Used to filter out documents by titles"),})
    splits: list[str]=field(default=None)
    construct_type: str=field(
        default=None,
        metadata={
            "help": (
                "There are two construction of data, one is `random`, another is `default`."
                "`random` will sample from a distribution as the positon of target"
                "While `default` use the suffix of max sequence length as the position of target"
            )
        }
    )
    max_pos_embeddings: int=field(default=None)
    max_train_example: int=field(
        default=None,
        metadata={
            "help": (
                "Using how many examples when construct train dataset."
                "Being Set to `None` refers to use full data."
                "(Just for accelerate data preparing."
            )
        }
    )

    def __post_init__(self):

        if self.dataset_dir is None:
            raise ValueError('--dataset_dir error.')
        for _ in self.splits: 
            if _ not in ["train","validation","test"]:
                raise ValueError(f"{_} is not a legal split.")
        if self.context_type not in ['vanilla_incontext', 'marking_incontext']:
            raise ValueError('--context_type error.')
        if self.construct_type not in ["random", "default"]:
            raise ValueError("--construct_type error.")
        
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
            "context_type": self.context_type,
            "forbid_titles_dir":self.forbid_titles_dir,
            "splits":self.splits,
            "construct_type": self.construct_type,
            "max_pos_embeddings": self.max_pos_embeddings,
            "max_train_example": self.max_train_example
        }

MARK_L = '<MARK_L>'
MARK_R = '<MARK_R>'

def main():
    set_seed(42)
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
    dataset = load_from_disk(args.dataset_dir)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if args.context_type == "marking_incontext":
        tokenizer.add_tokens([MARK_L, MARK_R], special_tokens=False)
        MARK_L_TOKEN, MARK_R_TOKEN = tokenizer.convert_tokens_to_ids([MARK_L, MARK_R])
    
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

    splits = args.splits
    for split in splits:
        if not (split in dataset.keys()): continue
        print(f"process `{split}` spilt...")

        split_dir = args.save_dir + '/dataset/' + split + '.jsonl'
        raw_data = "".join([x["text"] if x["text"] else " \n" for x in dataset[split]])
        tokens = tokenizer(raw_data).input_ids  

        with jsonlines.open(split_dir, 'w') as fp:            
            queries, pos = [], []
            for i in range(max_pos_embeddings, len(tokens), args.retrieve_stride):
                if i + args.retrieve_stride >= len(tokens): break
                query = tokenizer.decode(tokens[i - args.search_max_width: i])
                queries.append(query)
                pos.append(i)
                if split == "train" and args.max_train_example is not None:
                    if len(queries) >= args.max_train_example:
                        break

            recalls = searcher.batch_search(
                queries,
                qids=[str(i) for i in range(len(queries))],
                k=args.retrieve_k * 2, # times 2 avoiding the number of recall is less than `k` due to retrieval
                threads=multiprocessing.cpu_count()
            )
            for qid, res in tqdm(recalls.items()):
                i = int(qid)
                allowed_docs = []
                for hit in res:
                    res_dict = json.loads(hit.raw)
                    docs = res_dict['contents']

                    title = docs.split("\n")[0]
                    if title.startswith('"') and title.endswith('"'): title = title[1:-1]
                    if title in args.forbid_titles: continue

                    allowed_docs.append(docs)
                    if len(allowed_docs) >= args.retrieve_k:
                        break
                retrieved = []
                for docs in allowed_docs:
                    if args.context_type == "marking_incontext":
                        retrieved.append([MARK_L_TOKEN] + tokenizer(docs).input_ids[:args.recall_max_width] + [MARK_R_TOKEN])
                        # retrieved = retrieved + [MARK_L_TOKEN] + tokenizer(docs).input_ids[:args.recall_max_width] + [MARK_R_TOKEN]
                    else:
                        retrieved.append(tokenizer(docs).input_ids[:args.recall_max_width])
                        # retrieved = retrieved + tokenizer(docs).input_ids[:args.recall_max_width]
                target = tokens[pos[i]: pos[i] + args.retrieve_stride]
                source = []
                if len(retrieved) == 0:
                    retrieved.append([])
                    if args.construct_type == "random":
                        center_pos = random.randint(max_pos_embeddings // 2, max_pos_embeddings - args.retrieve_stride - 1)
                        source.append(tokens[pos[i] - center_pos: pos[i]])
                    else: # args.construct_type = "default"
                        source.append(tokens[pos[i] - max_pos_embeddings + args.retrieve_stride: pos[i]])
                else:
                    for doc in retrieved:
                        if args.construct_type == "random":
                            center_pos = random.randint(max_pos_embeddings // 2, max_pos_embeddings - args.retrieve_stride - 1)
                            source.append(tokens[pos[i] - center_pos + len(doc): pos[i]])
                        else: # args.construct_type = "default"
                            source.append(tokens[pos[i] - max_pos_embeddings + len(doc) + args.retrieve_stride: pos[i]])
                            if len(doc) + len(source) + len(target) != max_pos_embeddings:
                                raise ValueError(f"defalut construct error: {len(doc) + len(source) + len(target)} != {max_pos_embeddings}.")
                fp.write({
                    "e": retrieved,
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
