from datasets import load_from_disk
from transformers import HfArgumentParser, AutoTokenizer
from tqdm import tqdm
import jsonlines
from dataclasses import dataclass, field
import os
@dataclass
class Arguments:
    dataset_dir: str = field(default=None)
    output_dir: str= field(default=None)
    chunk_size: int= field(default=128)
    tokenizer_dir: str= field(default=None)
    divide_train_dataset: bool=field(default=False)
    def __post_init__(self):
        if not os.path.exists(self.dataset_dir):
            raise ValueError('--dataset_dir error.')
        if not os.path.exists(self.output_dir):
            raise ValueError('--output_dir error.')
        if not os.path.exists(self.tokenizer_dir):
            raise ValueError('--tokenizer_dir error.')
        if self.chunk_size <= 0:
            raise  ValueError('--chunk_size error.')
def main():
    parser = HfArgumentParser(Arguments)
    args = parser.parse_args_into_dataclasses()[0]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    dataset_dir = args.dataset_dir
    output_dir = args.output_dir + "/data.jsonl"
    chunk_size = args.chunk_size

    chunk_count = 0
    data = load_from_disk(dataset_dir)
    validation = data["validation"]
    test = data["test"]
    end_position = int(len(test) * 0.8)
    test = test.select(range(end_position))
    with jsonlines.open(output_dir, "w") as fp:
        for dataset in [validation, test]:
            for data in tqdm(dataset):
                text = data["text"]
                tokens = tokenizer.tokenize(text)
                for i in range(0,len(tokens),chunk_size):
                    chunk = tokenizer.convert_tokens_to_string(tokens[i:i+chunk_size])
                    fp.write({
                        "id": chunk_count,
                        "contents": chunk
                    })
                    chunk_count = chunk_count + 1
    print(f"Totle chunk_count: {chunk_count}.")
if __name__ == "__main__":
    main()
