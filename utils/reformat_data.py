"""This script aims to process the original data and reformat it to what FastChat needs.

Usage:
python3 -m utils.reformat_data --dataset alpaca --repo_type fastchat
"""
import argparse
import json
import logging
from pathlib import Path

from datasets import load_dataset


def alpaca(format="fastchat"):
    output_file = f"data/alpaca/{format}_alpaca.json"
    if Path(output_file).exists():
        logging.info("Data for stanford alpaca has been processed!")
        return

    if not Path("data/alpaca").exists():
        Path("data/alpaca").mkdir(parents=True)

    logging.info("Start reading data for stanford alpaca")
    dataset = load_dataset("tatsu-lab/alpaca")
    dataset = dataset['train']
    # dataset = load_dataset("umd-zhou-lab/Reflect_Alpaca_All")
    # dataset = dataset['reflect_both']
    logging.info(f"Finish reading {len(dataset)} instances")
    new_data = []
    prompt_normal_prefix = "Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction: "
    prompt_input_prefix = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. ### Instruction: "
    for i, instance in enumerate(dataset):
        new_item = {}
        new_item["id"] = f"identity_{i}"
        new_item["conversations"] = list()
        # prompt
        prompt = dict()
        prompt["from"] = "human"
        prompt["value"] = prompt_normal_prefix + instance["instruction"]
        if instance["input"] != "":
            prompt["value"] = prompt_input_prefix + instance["instruction"] + " ### Input: " + instance["input"]
        new_item["conversations"].append(prompt)
        # response
        response = dict()
        response["from"] = "gpt"
        response["value"] = instance["output"]
        new_item["conversations"].append(response)
        new_data.append(new_item)
    logging.info(f"Finish processing {len(new_data)} instances")
    with open(output_file, "w") as f:
        json.dump(new_data, f, indent=4)
    logging.info(f"Finish writing data to {output_file}")

def lima(format="fastchat"):
    output_file = f"data/lima/{format}_lima.json"
    if Path(output_file).exists():
        logging.info("Data for lima has been processed!")
        return

    if not Path("data/lima").exists():
        Path("data/lima").mkdir(parents=True)

    logging.info("Start reading data for stanford alpaca")
    file_path = "/Users/hzhao/Desktop/research/learning-order/data/lima/train_single_turn.jsonl"
    with open(file_path) as f:
        raw_data = [json.loads(line) for line in f]
    logging.info(f"Finish reading {len(raw_data)} instances")
    new_data = []
    prompt_normal_prefix = "Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction: "
    for i, instance in enumerate(raw_data):
        new_item = {}
        new_item["id"] = f"identity_{i}"
        new_item["score"] = 0.0 # for compatibility, automatically set to 0.0
        new_item["conversations"] = list()
        # prompt
        prompt = dict()
        prompt["from"] = "human"
        prompt["value"] = prompt_normal_prefix + instance["conversations"][0]
        new_item["conversations"].append(prompt)
        # response
        response = dict()
        response["from"] = "gpt"
        response["value"] = instance["conversations"][1]
        new_item["conversations"].append(response)
        new_data.append(new_item)
    logging.info(f"Finish processing {len(new_data)} instances")
    with open(output_file, "w") as f:
        json.dump(new_data, f, indent=4)
    logging.info(f"Finish writing data to {output_file}")

def evol_instruct(format="fastchat"):
    output_file = f"data/evol-instruct/{format}_evol_instruct_70k.json"
    if Path(output_file).exists():
        logging.info("Data for evol-instruct-70k has been processed!")
        return

    if not Path("data/evol-instruct").exists():
        Path("data/evol-instruct").mkdir(parents=True)

    logging.info("Start reading data for evol-instruct-70k")
    dataset = load_dataset("WizardLM/WizardLM_evol_instruct_70k")
    dataset = dataset['train']
    logging.info(f"Finish reading {len(dataset)} instances")
    new_data = []
    prompt_normal_prefix = "Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction: "
    for i, instance in enumerate(dataset):
        new_item = {}
        new_item["id"] = f"identity_{i}"
        new_item["conversations"] = list()
        # prompt
        prompt = dict()
        prompt["from"] = "human"
        prompt["value"] = prompt_normal_prefix + instance["instruction"]
        new_item["conversations"].append(prompt)
        # response
        response = dict()
        response["from"] = "gpt"
        response["value"] = instance["output"]
        new_item["conversations"].append(response)
        new_data.append(new_item)
    logging.info(f"Finish processing {len(new_data)} instances")
    with open(output_file, "w") as f:
        json.dump(new_data, f, indent=4)
    logging.info(f"Finish writing data to {output_file}")

def main(args):
    if args.dataset == "alpaca":
        alpaca(format=args.repo_type)
    elif args.dataset == "lima":
        lima(format=args.repo_type)
    elif args.dataset == "evol-instruct":
        evol_instruct(format=args.repo_type)
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="alpaca")
    parser.add_argument("--repo_type", type=str, default="fastchat")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    main(args)