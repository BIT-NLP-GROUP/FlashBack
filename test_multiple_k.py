import torch

from dataclasses import dataclass, field
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, GPT2LMHeadModel, AutoConfig
from peft import get_peft_model, LoraConfig, PeftModel
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
import math
from tqdm import tqdm

@dataclass
class CustomArguments:
    model_dir: str=field(default=None)
    adapter_dir: str=field(default=None)
    data_dir: str=field(default=None)
    cache_dir: str=field(default=None)
    add_position: str= field(default=None)
    retrieve_k: int=field(default=None)
    def __post_init__(self):
        if self.add_position not in ["back", "front"]:
            raise ValueError("push_position error!")

def main():
    parser = HfArgumentParser((CustomArguments))
    custom_args = parser.parse_args_into_dataclasses()[0]
    # tokenizer = AutoTokenizer.from_pretrained(custom_args.adapter_dir)
    tokenizer = AutoTokenizer.from_pretrained(custom_args.model_dir)
    # print(custom_args)
    if "6.7" in custom_args.model_dir:
        model = AutoModelForCausalLM.from_pretrained(custom_args.model_dir, torch_dtype=torch.bfloat16)
    else:
        model = AutoModelForCausalLM.from_pretrained(custom_args.model_dir)

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    model = PeftModel.from_pretrained(model, custom_args.adapter_dir)
    
    dataset = load_dataset(custom_args.data_dir, split='validation',cache_dir=custom_args.cache_dir)
    loss_fnt = CrossEntropyLoss(reduction='none')
    losses = []
    model.to("cuda")

    model.eval()
    with torch.no_grad():
        for data in tqdm(dataset):
            #[e], src, tgt, 
            e, src, tgt = data['e'], data['src'], data['tgt']

            logits = []
            for count in range(custom_args.retrieve_k):
                if count + 1 > len(e): break
                tgt_st = len(e[count]) + len(src[count])
                tgt_ed = len(e[count]) + len(src[count]) + len(tgt)
                if custom_args.add_position == "front":
                    input_ids = e[count] + src[count] + tgt
                else:
                    input_ids = src[count] + e[count] + tgt
                input_ids = torch.tensor(input_ids).unsqueeze(0).to("cuda")
                local_logits = model(input_ids=input_ids)["logits"] 
                local_logits = local_logits[0, tgt_st-1:tgt_ed-1, :-2] #need test
                logits.append(local_logits)
            logits = torch.stack(logits,dim=0).mean(dim=0)
            loss = loss_fnt(input=logits, target=torch.tensor(tgt).to(input_ids))
            losses.append(loss.mean().cpu())
        losses = torch.tensor(losses).mean().item()
        print(f"loss: {losses}\n ppl: {math.exp(losses)}")


if __name__ == "__main__":
    main()