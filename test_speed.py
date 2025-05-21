import time
import torch
from line_profiler import profile
from peft import get_peft_model, LoraConfig
from transformers import HfArgumentParser, AutoModelForCausalLM, set_seed
from dataclasses import dataclass, field

@dataclass
class TestArguments:
    
    model_dir: str=field(default=None)
    T: int=field(default=2048)
    test_mode: str=field(default='front')
    stride: int=field(default=16)
    init_length: int=field(default=512)
    lora: bool=field(default=False)
    
    search_k: int=field(default=1)
    def __post_init__(self):
        if self.model_dir is None:
            raise ValueError("no model_dir.")
        if self.test_mode not in ['front', 'back']:
            raise ValueError(f"illegal test_mode {self.test_mode}")
    
@profile
def push_front_simulate(model, input_ids, args):
    generate_num = 0
    doc = None
    kv_cache = None
    past_token = None
    t1 = time.process_time()
    while(input_ids.size(-1) + 128 < args.T):
        if generate_num == 0:
            doc = torch.randint(500,1000,(1,128)).to(input_ids)
            outputs = model(input_ids=torch.cat((doc,input_ids),dim=-1).to(input_ids))
        else:
            outputs = model(input_ids=past_token, past_key_values=kv_cache)
        kv_cache = outputs["past_key_values"]
        logits = outputs["logits"][:,-1, :]
        out_token = torch.argmax(logits, dim=-1, keepdim = True)
        past_token = out_token
        input_ids = torch.cat((input_ids, out_token), dim=-1)
        generate_num = (generate_num + 1) % args.stride

    t2 = time.process_time()
    print(f"{input_ids.size(-1)}: {round(t2 - t1, 2)}sec.")

@profile
def push_back_simulate(model, input_ids, args):
    generate_num = 0
    doc = None
    kv_cache = None
    past_token = None
    t1 = time.process_time()
    while(input_ids.size(-1) + 128 < args.T):
        if generate_num == 0:
            doc = torch.randint(500,1000,(1,128)).to(input_ids)
            if kv_cache is None:
                outputs = model(input_ids=torch.cat((input_ids, doc), dim=-1).to(input_ids))
            else:             
                kv_cache = [[arr[:,:,:input_ids.size(-1)-args.stride,:] for arr in _] for _ in kv_cache]
                outputs = model(input_ids=torch.cat((input_ids[:,-args.stride:], doc), dim=-1).to(input_ids), past_key_values=kv_cache)
                kv_cache = outputs["past_key_values"]
        else:
            outputs = model(input_ids=past_token, past_key_values=kv_cache)
        kv_cache = outputs["past_key_values"]
        logits = outputs["logits"][:,-1, :]
        out_token = torch.argmax(logits, dim=-1, keepdim = True)
        past_token = out_token
        input_ids = torch.cat((input_ids, out_token),-1)
        generate_num = (generate_num + 1) % args.stride
    t2 = time.process_time()
    print(f"{input_ids.size(-1)}: {round(t2 - t1, 2)}sec.")

def main():
    set_seed(42)
    parser = HfArgumentParser(TestArguments)
    args = parser.parse_args_into_dataclasses()[0]
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, device_map=0)
    if args.lora:
        if "gpt" in args.model_dir:
            target_modules = ["q_attn", "c_attn", "c_proj"]
        elif "opt" in args.model_dir:
            target_modules = ["q_proj", "v_proj"]
        peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=target_modules,
                bias="none",
                task_type="CAUSAL_LM",
                fan_in_fan_out=True,
            )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    model.eval()
    input = torch.randint(500,1000,(1,args.init_length))
    input_ids = input.to("cuda")
    with torch.no_grad():
        if args.test_mode == 'front':
            push_front_simulate(model, input_ids, args)
        else:
            push_back_simulate(model, input_ids, args)

if __name__ == "__main__":
    main()
    