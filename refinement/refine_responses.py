import argparse
import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List

import openai
import tiktoken
from tqdm import tqdm

gpt_encoder = tiktoken.get_encoding("cl100k_base")
from transformers import LlamaForCausalLM, LlamaTokenizer

llama_tokenizer = LlamaTokenizer.from_pretrained('ckpts/llama-2-7b-hf')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def dispatch_openai_requests(
    messages_list: List[List[Dict[str,Any]]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
) -> List[str]:
    """Dispatches requests to OpenAI API asynchronously.
    
    Args:
        messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        model: OpenAI model to use.
        temperature: Temperature to use for the model.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use for the model.

    Returns:
        List of responses from OpenAI API.
    """
    async_responses = [
        openai.ChatCompletion.acreate(
            model=model,
            messages=x,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)

def gen_prompt_no_input(ins, outp):

    sys_prompt = "You are a helpful, precise but picky assistant for checking the quality of the answer to a given instruction."
    prompt_template = "[Instruction]\n{ins}\n\n[The Start of Answer]\n{outp}\n\n[The End of Answer]\n\n[System]\n{criteria}\n\n"
    criteria = "We would like you to answer several questions related to the quality of the answer to the given instruction. \n" + \
                "1. Why this answer is not good for the given instruction? Analyse based on the Helpfulness, Relevance, Accuracy, Level of Details, and Structure. \n" + \
                "2. Based on the reason you provided, please generate a better answer while preserving the same content. To achieve that, you may want to adjust the level of details, add bullet points, or use comprehensive words, etc. The answer should be in the format of [Better Answer] your answer [End] \n"
    prompt = prompt_template.format(
        ins=ins, outp=outp, criteria=criteria
    )
    return sys_prompt, prompt

def gen_prompt_input(ins, inp, outp):

    sys_prompt = "You are a helpful and precise assistant for checking the quality of the answer to a given instruction and its input."
    prompt_template = "[Instruction]\n{ins}\n\n[The Start of Input]\n{inp}\n\n[The End of Input]\n\n[The Start of Answer]\n{outp}\n\n[The End of Answer]\n\n[System]\n{criteria}\n\n"
    criteria = "We would like you to answer several questions related to the quality of the answer to the given instruction and corresponding input. \n" + \
                "1. Why this answer is not good for the given instruction and corresponding input? Analyse based on the Helpfulness, Relevance, Accuracy, Level of Details, and Structure. \n" + \
                "2. Based on the reason you provided, please generate a better answer while preserving the same content. To achieve that, you may want to adjust the level of details, add bullet points, or use comprehensive words, etc. The answer should be in the format of [Better Answer] your answer [End] \n"
    prompt = prompt_template.format(
        ins=ins, inp=inp, outp=outp, criteria=criteria
    )
    return sys_prompt, prompt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='')
    parser.add_argument("--save_path", type=str, default='')
    parser.add_argument("--api_key", type=str, default='')
    parser.add_argument("--api_model",type=str,default='gpt-3.5-turbo-1106')
    parser.add_argument("--api_base",type=str,default='')
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Batch size to call OpenAI GPT",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="maximum number of tokens produced in the output",
    )
    args = parser.parse_args()
    if args.api_base != '':
        openai.api_base = args.api_base
    openai.api_key = args.api_key

    with open(args.data_path, "r") as f:
        data = json.load(f)

    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx != -1 else len(data)
    sampled_data = data[start_idx:end_idx]
    
    message_list = []
    token_len_list = []
    for i, data_i in enumerate(sampled_data):
        instruct_i = data_i['instruction'].strip()
        output_i = data_i['output'].strip()
        if 'input' in data_i.keys():
            input_i = data_i['input'].strip()
        else:
            input_i = ''

        whole_text = instruct_i + input_i + output_i
        inputs = llama_tokenizer(whole_text, return_tensors="pt")
        input_ids = inputs["input_ids"]
        if input_ids.shape[1] > 2048:
            gap = input_ids.shape[1] - 2048
            output_i = output_i[:-gap]

        if input_i == '':
            sys_prompt, prompt = gen_prompt_no_input(instruct_i, output_i)
        else:
            sys_prompt, prompt = gen_prompt_input(instruct_i, input_i, output_i)

        message =[
                    {"role": "system", "content": sys_prompt},
                    {
                        "role": "user",
                        "content": prompt,
                    },
        ]
        message_list.append(message)
        token_len_list.append(len(gpt_encoder.encode(prompt)))

    predictions = []
    i = 0
    wait_base = 10
    retry = 0
    error = 0
    pbar = tqdm(total=len(message_list))
    batch_size = args.batch_size
    while(i<len(message_list)):
        token_limit_in_current_batch = min(args.max_tokens,4050-max(token_len_list[i:i+batch_size]))
        try:
            batch_predictions = asyncio.run(
                dispatch_openai_requests(
                    messages_list=message_list[i:i+batch_size],
                    model=args.api_model,
                    temperature=0.0,
                    max_tokens=token_limit_in_current_batch,
                    top_p=1.0,
                )
            )
            predictions += batch_predictions
            retry = 0
            i += batch_size
            wait_base = 10
            pbar.update(batch_size)
        except:
            retry += 1
            error += 1
            print("Batch error: ",i, i+batch_size)
            print("retry number: ", retry)
            print("error number: ", error)
            time.sleep(wait_base)
            wait_base = wait_base*2
    pbar.close()

    new_data = []
    for idx, prediction in enumerate(predictions):
        review = prediction['choices'][0]['message']['content']
        new_data.append(review)
        pass

    print('Len New Data', len(new_data))
    with open(args.save_path,'w') as f:
        json.dump(new_data,f,indent=4)

    pass
