import pickle
import numpy as np
from dictionary import (fr_actions_off, us_actions_off, sp_actions_off,
                         fr_actions, sp_actions, us_actions)

DAY_IDX    = 0
HOUR_IDX   = 1
DEVICE_IDX = 2
ACTION_IDX = 3

off_action_ids = {
    "fr": set(fr_actions_off.values()),
    "us": set(us_actions_off.values()),
    "sp": set(sp_actions_off.values()),
}

def build_on_action_ids(actions_dict):
    on_keywords = [
        "switch on", "valve open", "windowShade open", "doorControl open",
        "lock unlock", "alarm both", "setMachineState run", "start",
        "mediaPlayback play", "setRobotCleanerMovement cleaning"
    ]
    on_ids = set()
    for action_str, action_id in actions_dict.items():
        if any(kw in action_str for kw in on_keywords):
            on_ids.add(action_id)
    return on_ids

on_action_ids = {
    "fr": build_on_action_ids(fr_actions),
    "us": build_on_action_ids(us_actions),
    "sp": build_on_action_ids(sp_actions),
}

def build_device_pairing(actions_dict, off_dict):
    device_to_actions = {}
    for action_str, action_id in actions_dict.items():
        device = action_str.split(":")[0]
        device_to_actions.setdefault(device, []).append((action_str, action_id))
    pairing = {}
    for action_str, off_id in off_dict.items():
        device = action_str.split(":")[0]
        if device in device_to_actions:
            on_ids_for_device = set()
            for a_str, a_id in device_to_actions[device]:
                if any(kw in a_str for kw in [
                    "switch on", "valve open", "windowShade open",
                    "doorControl open", "lock unlock", "setMachineState run",
                    "mediaPlayback play", "setRobotCleanerMovement cleaning"
                ]):
                    on_ids_for_device.add(a_id)
            pairing[off_id] = on_ids_for_device
    return pairing

device_pairing = {
    "fr": build_device_pairing(fr_actions, fr_actions_off),
    "us": build_device_pairing(us_actions, us_actions_off),
    "sp": build_device_pairing(sp_actions, sp_actions_off),
}

DATASET_THRESHOLDS = {
    "fr": {"interval": 9,  "total": 24},
    "sp": {"interval": 9,  "total": 24},
    "us": {"interval": 6,  "total": 18},
    "an": {"interval": 12, "total": 36},
}

def calculate_hours(day1, hour_slot1, day2, hour_slot2):
    total1 = day1 * 24 + hour_slot1 * 3
    total2 = day2 * 24 + hour_slot2 * 3
    if total2 < total1:
        weeks_needed = (total1 - total2) // 168 + 1
        total2 += weeks_needed * 168
    return total2 - total1

def extract_interval(sequence):
    n = len(sequence[0])
    intervals = [0]
    for i in range(n - 1):
        gap = calculate_hours(
            sequence[DAY_IDX][i], sequence[HOUR_IDX][i],
            sequence[DAY_IDX][i+1], sequence[HOUR_IDX][i+1]
        )
        intervals.append(gap)
    return intervals

def extract_total(sequence):
    intervals = extract_interval(sequence)
    totals = [0]
    running = 0
    for gap in intervals[1:]:
        running += gap
        totals.append(running)
    return totals

def is_off_action(action_id, data_name):
    return action_id in off_action_ids[data_name]

def is_on_action(action_id, data_name):
    return action_id in on_action_ids[data_name]

def has_unpaired_on(sublist_cols, next_action_id, data_name):
    pairing = device_pairing[data_name]
    if is_off_action(next_action_id, data_name):
        return False
    action_ids_in_sublist = sublist_cols[ACTION_IDX]
    opened_devices = set()
    for act_id in action_ids_in_sublist:
        if is_on_action(act_id, data_name):
            opened_devices.add(act_id)
    for act_id in action_ids_in_sublist:
        if act_id in pairing:
            matched_ons = pairing[act_id]
            opened_devices -= matched_ons
    if opened_devices:
        return False
    return True

def split_sequence(sequence, interval_threshold, total_threshold, data_name):
    n = len(sequence[0])
    if n == 1:
        return [sequence]
    intervals = extract_interval(sequence)
    pass1_result = []
    current = [sequence[:, 0]]
    for i in range(1, n):
        event_col = sequence[:, i]
        action_id = int(event_col[ACTION_IDX])
        if intervals[i] > interval_threshold:
            current_arr = np.array(current).T
            if has_unpaired_on(current_arr, action_id, data_name):
                pass1_result.append(current_arr)
                current = [event_col]
            else:
                current.append(event_col)
        else:
            current.append(event_col)
    if current:
        pass1_result.append(np.array(current).T)

    pass2_result = []
    for subseq in pass1_result:
        m = len(subseq[0])
        if m == 1:
            pass2_result.append(subseq)
            continue
        totals  = extract_total(subseq)
        current = [subseq[:, 0]]
        t_start = 0
        for i in range(1, m):
            event_col = subseq[:, i]
            action_id = int(event_col[ACTION_IDX])
            elapsed   = totals[i] - t_start
            if elapsed > total_threshold:
                current_arr = np.array(current).T
                if has_unpaired_on(current_arr, action_id, data_name):
                    pass2_result.append(current_arr)
                    current = [event_col]
                    t_start = totals[i]
                else:
                    current.append(event_col)
            else:
                current.append(event_col)
        if current:
            pass2_result.append(np.array(current).T)
    return pass2_result

def filter_single_events(sequences):
    return [seq for seq in sequences if len(seq[0]) > 1]

def validate_alignment(sequences):
    valid = []
    for i, seq in enumerate(sequences):
        flat_len = seq.shape[0] * seq.shape[1]
        if flat_len % 4 == 0:
            valid.append(seq)
        else:
            print(f"[split] WARNING: sequence {i} dropped — length {flat_len} not divisible by 4")
    return valid

def split(file_path, interval_threshold, total_threshold, data_name):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    shaped = []
    for row in data:
        arr = np.reshape(np.array(row), (-1, 4)).T
        shaped.append(arr)
    all_subsequences = []
    for arr in shaped:
        parts = split_sequence(arr, interval_threshold, total_threshold, data_name)
        all_subsequences.extend(parts)
    all_subsequences = filter_single_events(all_subsequences)
    all_subsequences = validate_alignment(all_subsequences)
    result = []
    for arr in all_subsequences:
        flat = arr.reshape(1, -1).tolist()[0]
        result.append(flat)
    return result

def Split(dataset, ori_env, need_split):
    if need_split == 1:
        thresholds = DATASET_THRESHOLDS.get(dataset, {"interval": 9, "total": 24})
        new_groups = split(
            file_path=f"IoT_data/{dataset}/{ori_env}/trn.pkl",
            interval_threshold=thresholds["interval"],
            total_threshold=thresholds["total"],
            data_name=dataset
        )
        with open(f"IoT_data/{dataset}/{ori_env}/split_trn.pkl", "wb") as f:
            pickle.dump(new_groups, f)
        print(f"[Split] {dataset}/{ori_env}: {len(new_groups)} subsequences saved.")
    else:
        print("[Split] Skipping — copying trn.pkl directly.")
        with open(f"IoT_data/{dataset}/{ori_env}/trn.pkl", "rb") as f:
            data = pickle.load(f)
        with open(f"IoT_data/{dataset}/{ori_env}/split_trn.pkl", "wb") as f:
            pickle.dump(data, f)

def Split_test(dataset, new_env):
    thresholds = DATASET_THRESHOLDS.get(dataset, {"interval": 9, "total": 24})
    new_groups = split(
        file_path=f"IoT_data/{dataset}/{new_env}/test.pkl",
        interval_threshold=thresholds["interval"],
        total_threshold=thresholds["total"],
        data_name=dataset
    )
    print(f"[Split_test] {len(new_groups)} subsequences")
    with open(f"IoT_data/{dataset}/{new_env}/split_test.pkl", "wb") as f:
        pickle.dump(new_groups, f)
