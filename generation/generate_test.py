"""This python script aims to generate the response data for the target model.
It is modified from: https://github.com/gauss5930/AlpaGasus2-QLoRA/blob/main/evaluation/AlpaGasus-Evaluation/response_data/generate.py
The original version loads additional PEFT module to merge into the base model, while this version loads the fine-tuned model directly.

usage: 
python AlpaGasus-Evaluation/generate.py --model_name_or_path ckpts/llama-2-alpaca-9k-with-score --test_path data/alpagasus-eval/prompt_data/ --save_path data/alpagasus-eval/response_data/
"""

import json

import fire
import jsonlines
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def main(
  model_name_or_path: str = "ckpts/Llama-2-alpaca-52k",
  test_path: str = "data/alpagasus-eval/prompt_data/",
  save_path: str = "data/alpagasus-eval/response_data/",
):

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
  tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)  
  model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)

  test_data = ['koala_test_set.jsonl', 'sinstruct_test_set.jsonl', 'wizardlm_test_set.jsonl', 'vicuna_test_set.jsonl', 'lima_test.jsonl']
  col = ['prompt', 'instruction', 'Instruction', 'text', 'conversations']   # Columns of each test dataset
  model_name = model_name_or_path.split('/')[-1]

  for i in range(len(test_data)):
    result = []
    path = test_path + test_data[i]
    count = 0
    name = test_data[i].split('_')[0]
    sv_path = save_path + model_name + "_" + name + ".json"
    print(f"Generating response data for {model_name} on {name}.\n")
    num_lines = sum(1 for line in open(path,'r'))
    with jsonlines.open(path) as f:
      for line in tqdm(f, total=num_lines, desc="Generating response data"):
        if "sinstruct" in test_data[i]:
            instances = line['instances']
            assert len(instances) == 1
            if instances[0]['input'] != "":
                input_data = (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\r\n\r\n"
                f"### Instruction:\r\n{line[col[i]]}\r\n\r\n### Input:\n{instances[0]['input']}\r\n\r\n### Response:"
                )
            else:
                input_data = (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\r\n\r\n"
                f"### Instruction:\r\n{line[col[i]]}\r\n\r\n### Response:"
                )
        else:
           input_data = (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\r\n\r\n"
                f"### Instruction:\r\n{line[col[i]]}\r\n\r\n### Response:"
                )
        model_inputs = tokenizer(input_data, return_tensors='pt').to(device)
        num_tokens = len(tokenizer.tokenize(input_data))
        # max_length = 512
        max_length = 512 + num_tokens if 512 + num_tokens <= 4096 else 4096
        model_output = model.generate(**model_inputs, max_length=max_length, pad_token_id=tokenizer.eos_token_id) 
        model_output = tokenizer.batch_decode(model_output, skip_special_tokens=True)[0]
        model_output = model_output[len(input_data) :]
        count += 1
        output = {}
        index = name + '_' + str(count)
        output['question_id'] = index
        output[col[i]] = line[col[i]]
        if "sinstruct" in test_data[i]:
            output['instances'] = [{'input': line['instances'][0]['input']}]
        output['prompt'] = input_data
        output[model_name] = model_output
        result.append(output)
  
    with open(sv_path, "x") as json_file:
      json.dump(result, json_file, indent=4)

if __name__ == "__main__":
  fire.Fire(main)
