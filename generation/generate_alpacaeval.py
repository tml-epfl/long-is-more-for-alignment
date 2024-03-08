"""
usage: 
python AlpaGasus-Evaluation/generate_alapcaeval.py --model_name_or_path ckpts/llama-2-alpaca-9k-with-score --save_path data/alpaca-eval/
"""

import json

import fire
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets


def main(
  model_name_or_path: str = "ckpts/Llama-2-alpaca-52k",
  save_path: str = "data/alpaca-eval/",
  save_name: str = None
):

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
  tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)  
  model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)
  model_name = model_name_or_path.split('/')[-1]
  data = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]

  outputs = []
  if save_name:
    sv_path = save_path + save_name + "_outputs.json"
  else:
    sv_path = save_path + model_name + "_outputs.json"
  print(f"Generating response data for {model_name} on AlpacaEval.\n")
  for idx in tqdm(range(len(data)), desc="Generating response data"):
    instruction = data[idx]["instruction"]
    input_data = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\r\n\r\n"
        f"### Instruction:\r\n{instruction}\r\n\r\n### Response:"
        )
    model_inputs = tokenizer(input_data, return_tensors='pt').to(device)
    max_length = 2048  # for fair comparison with GPT-4-Turbo in terms of response length
    model_output = model.generate(**model_inputs, max_length=max_length, pad_token_id=tokenizer.eos_token_id) 
    model_output = tokenizer.batch_decode(model_output, skip_special_tokens=True)[0]
    model_output = model_output[len(input_data) :]
    output = {}
    index = 'AlpacaEval_' + str(idx)
    output['question_id'] = index
    output["instruction"] = instruction
    output["output"] = model_output
    output["generator"] = save_name if save_name else model_name
    output["dataset"] = "AlpacaEval"
    output["datasplit"] = "eval"
    outputs.append(output)
  
  with open(sv_path, "x") as json_file:
    json.dump(outputs, json_file, indent=4)

if __name__ == "__main__":
  fire.Fire(main)
