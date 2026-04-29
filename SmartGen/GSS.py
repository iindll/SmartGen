import argparse
import pickle
import ast
import os
import json
from openai import OpenAI
from dictionary import dayofweek_dict, hour_dict, fr_devices_dict, fr_actions, sp_devices_dict, sp_actions, us_devices_dict, us_actions
from split import Split
from dayse import Dayse
from transtext import Transtext
from transnumber import Transnum
from sppc import SPPC_select, similarity_select
from baseline1 import Anomaly_detection
from baseline2 import Train
from Text_trans2 import ATM
from extract import Extract
from find_categories import Find_categories
from security_check import security_check
from SAS_main import SASRec_behavior_prediction

vocab_dic = {"an": 141, "fr": 223, "us": 269, "sp": 235}
device_dic = {"us": us_devices_dict, "fr": fr_devices_dict, "sp": sp_devices_dict}
act_dic = {"us": us_actions, "fr": fr_actions, "sp": sp_actions}
def get_args_parser():
    parser = argparse.ArgumentParser('LLM generation', add_help=False)
    parser.add_argument('--model', default='gpt-4o', type=str,
                        help='The used LLM: Llama_405B/70B/gpt-4o/deepseek-v3')
    parser.add_argument('--dataset', default='fr', type=str,
                        help='Name of dataset to train: an/fr/us/sp')
    parser.add_argument('--ori_env', default='winter', type=str,
                        help='The original home environment: winter/daytime')
    parser.add_argument('--new_env', default='spring', type=str,
                        help='The new home environment: spring/night')
    parser.add_argument('--method', default='SPPC', type=str,
                        help='The compression method: SPPC/similarity/instance')
    parser.add_argument('--threshold', default=0.918, type=float,
                        help="The compression threshold")
    parser.add_argument('--percentage', default=95.5, type=float,
                        help='The anomaly detection threshold percentage')
    parser.add_argument('--need_test', default=True, type=bool,
                        help='The experimental setup: True/False')
    parser.add_argument('--need_generate', default=False, type=bool,
                        help='The experimental setup: True/False')
    return parser


def study_sequence_distribution(dataset, ori_env, actions):
    from collections import Counter
    import statistics

    with open(f'IoT_data/{dataset}/{ori_env}/split_trn.pkl', 'rb') as f:
        data = pickle.load(f)

    index_to_action = {v: k for k, v in actions.items()}
    indices_to_extract = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]

    raw_lengths = [len(seq) for seq in data]
    action_counts_per_seq = [
        sum(1 for i in indices_to_extract if i < len(seq)) for seq in data
    ]
    all_actions = [
        seq[i] for seq in data for i in indices_to_extract if i < len(seq)
    ]
    action_freq = Counter(all_actions)
    top_count = action_freq.most_common(1)[0][1]

    print(f"\n{'='*60}")
    print(f"  Sequence Distribution: dataset={dataset}  env={ori_env}")
    print(f"{'='*60}")
    print(f"  Total sequences     : {len(data)}")
    print(f"  Total action tokens : {len(all_actions)}")
    print(f"  Unique actions seen : {len(action_freq)}")

    print(f"\n--- Raw sublist length distribution ---")
    print(f"  min={min(raw_lengths)}  max={max(raw_lengths)}"
          f"  mean={statistics.mean(raw_lengths):.1f}"
          f"  median={statistics.median(raw_lengths)}"
          f"  stdev={statistics.stdev(raw_lengths):.2f}")
    length_dist = Counter(raw_lengths)
    max_lcount = max(length_dist.values())
    for ln in sorted(length_dist):
        bar = '#' * (length_dist[ln] * 40 // max_lcount)
        print(f"  length {ln:3d}: {length_dist[ln]:5d} seqs  {bar}")

    print(f"\n--- Actions per sequence ---")
    print(f"  min={min(action_counts_per_seq)}  max={max(action_counts_per_seq)}"
          f"  mean={statistics.mean(action_counts_per_seq):.1f}"
          f"  median={statistics.median(action_counts_per_seq)}")
    count_dist = Counter(action_counts_per_seq)
    max_ccount = max(count_dist.values())
    for c in sorted(count_dist):
        bar = '#' * (count_dist[c] * 40 // max_ccount)
        print(f"  {c:2d} actions: {count_dist[c]:5d} seqs  {bar}")

    print(f"\n--- Action frequency (top 20) ---")
    for action_id, cnt in action_freq.most_common(20):
        name = index_to_action.get(action_id, str(action_id))
        bar = '#' * (cnt * 40 // top_count)
        print(f"  {name:<45s} {cnt:5d}  {bar}")

    print(f"\n--- Action frequency (bottom 10) ---")
    for action_id, cnt in action_freq.most_common()[:-11:-1]:
        name = index_to_action.get(action_id, str(action_id))
        print(f"  {name:<45s} {cnt:5d}")

    print(f"{'='*60}\n")


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    study_sequence_distribution(args.dataset, args.ori_env, act_dic[args.dataset])
    ATM(args.dataset, args.ori_env, act_dic[args.dataset])
