import argparse
import json
from typing import List

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 20

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str)
    parser.add_argument("--key_target", type=str)
    parser.add_argument("--key_base", type=str)
    parser.add_argument("--plot_target", type=str)
    parser.add_argument("--plot_base", type=str)
    parser.add_argument(
        "--output_path",
        type=str,
        help="The output path."
    )

    return parser.parse_args()

def result_process(args):
    nums = [252, 218, 180, 80, 300]   # length of each test set(koala, sinstruct, vicuna)
    test_set = ['sinstruct', 'wizardlm', 'koala', 'vicuna', 'lima']

    models = [f'{args.key_base}-{args.key_target}-', f'{args.key_target}-{args.key_base}-']
    result_judge = []
    for t in range(len(test_set)):
        number = nums[t]
        res = [0] * number
        result = {'win': 0, 'tie': 0, 'lose': 0}
        for m in range(len(models)):
            path = args.file_path + models[m] + test_set[t] + '.json'
            with open(path, 'r') as f:
                data = json.load(f)

            for i in range(len(data)):
                if m == 0:
                    alp, alpg = data[i]['score']
                    if int(alpg) > int(alp):
                        res[i] += 1
                    elif int(alpg) < int(alp):
                        res[i] -= 1
                else:
                    alpg, alp = data[i]['score']
                    if int(alpg) > int(alp):
                        res[i] += 1
                    elif int(alpg) < int(alp):
                        res[i] -= 1

        for n in range(len(res)):
            if res[n] >= 1:
                result['win'] += 1
            elif res[n] <= -1:
                result['lose'] += 1
            elif res[n] == 0:
                result['tie'] += 1

        upload = {'test_name': test_set[t], 'num_data': number}
        upload.update(result)
        result_judge.append(upload)

    return result_judge

def graph(args, result_list: List[dict], watch: bool):
    dataset = ('Self-Instruct', 'Wizardlm', 'Koala', 'Vicuna', 'LIMA')
    # dataset = ('Self-Instruct', 'Wizardlm', 'Koala', 'Vicuna')
    win_draw_lose = {
        f'{args.plot_target} wins': np.array([0, 0, 0, 0, 0]),
        'Tie': np.array([0, 0, 0, 0, 0]),
        f'{args.plot_base} wins': np.array([0, 0, 0, 0, 0]),
    }
    for i in range(len(result_list)):

        win_draw_lose[f'{args.plot_target} wins'][i] = int(result_list[i]['win'] / result_list[i]['num_data'] * 100)
        win_draw_lose['Tie'][i] = int(result_list[i]['tie'] / result_list[i]['num_data'] * 100)
        win_draw_lose[f'{args.plot_base} wins'][i] = 100 - win_draw_lose[f'{args.plot_target} wins'][i] - win_draw_lose['Tie'][i]
    
    height = 0.8  # the height of the bars: can also be len(x) sequence

    fig, ax = plt.subplots(figsize=(18,12))
    left = np.zeros(5)

    # colors
    category_colors = plt.colormaps['tab20c'](
            np.linspace(0., 0.15, 3))
    idx = 0
    label_colors = ['white', 'black', 'black']

    for judge, judge_count in win_draw_lose.items():
        p = ax.barh(dataset, judge_count, height, label=judge, left=left, color=category_colors[idx, :], )
        left += judge_count

        ax.bar_label(p, label_type='center', fontsize=56, fmt="%g", color=label_colors[idx])
        idx += 1

    plt.xticks([])
    plt.yticks(fontsize=56)
    # ax.set_title(f'7B: {args.base_model} on {args.base_data}')
    ax.legend(loc='upper center', ncol=3, bbox_to_anchor=(.5, 1.22), fontsize=52, columnspacing=0.8)

    if watch:
        plt.show()  # watch=True, if you wanted to watch a result graph.
    else:
        plt.savefig(args.output_path, bbox_inches='tight')   # watch=False, if you wanted to save a result graph.

def main():
    args = parse_args()

    graph(args, result_process(args), watch=False)

if __name__ == "__main__":
    main()
