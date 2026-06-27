import pickle
import numpy as np
from dictionary import (fr_actions_off, us_actions_off, sp_actions_off,
                        fr_actions, sp_actions, us_actions)

# ── Constants ─────────────────────────────────────────────────────────────
DAY_IDX    = 0
HOUR_IDX   = 1
DEVICE_IDX = 2
ACTION_IDX = 3

# ── OFF action sets ───────────────────────────────────────────────────────
off_action_ids = {
    "fr": set(fr_actions_off.values()),
    "us": set(us_actions_off.values()),
    "sp": set(sp_actions_off.values()),
}

# ── Build ON action sets ──────────────────────────────────────────────────
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

# ── Build device pairing ──────────────────────────────────────────────────
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

# ── FIX 1: عرّف device_pairing و action_to_device الأول
#           عشان semantic_similarity تلاقيهم ──────────────────────────────
device_pairing = {
    "fr": build_device_pairing(fr_actions, fr_actions_off),
    "us": build_device_pairing(us_actions, us_actions_off),
    "sp": build_device_pairing(sp_actions, sp_actions_off),
}

action_to_device = {}
for actions_dict in [fr_actions, us_actions, sp_actions]:
    for action_str, action_id in actions_dict.items():
        device = action_str.split(":")[0]
        action_to_device[action_id] = device

# ── Semantic similarity (دلوقتي بعد ما device_pairing اتعرف) ─────────────
def semantic_similarity(prev_action_id, curr_action_id, data_name):
    if prev_action_id == curr_action_id:
        return 1.0
    pairing = device_pairing[data_name]
    for off_id, on_ids in pairing.items():
        if prev_action_id == off_id and curr_action_id in on_ids:
            return 0.9
        if curr_action_id == off_id and prev_action_id in on_ids:
            return 0.9
    prev_device = action_to_device.get(prev_action_id)
    curr_device = action_to_device.get(curr_action_id)
    if prev_device is not None and curr_device is not None and prev_device == curr_device:
        return 0.6
    return 0.0

# ── Per-dataset thresholds ────────────────────────────────────────────────
DATASET_THRESHOLDS = {
    "fr": {"interval": 9,  "total": 24},
    "sp": {"interval": 9,  "total": 24},
    "us": {"interval": 6,  "total": 18},
    "an": {"interval": 12, "total": 36},
}

# ── Time helpers ──────────────────────────────────────────────────────────
def calculate_hours(day1, hour_slot1, day2, hour_slot2):
    total1 = day1 * 24 + hour_slot1 * 3
    total2 = day2 * 24 + hour_slot2 * 3
    if total2 < total1:
        weeks_needed = (total1 - total2) // 168 + 1
        total2 += weeks_needed * 168
    return total2 - total1

# ── FIX 2: extract_interval معرّفة قبل adaptive_interval_threshold ────────
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

def adaptive_interval_threshold(sequence):
    intervals = extract_interval(sequence)
    if len(intervals) < 5:
        return 6
    return max(3, np.percentile(intervals, 90))

def dynamic_threshold(intervals, idx, window_size=5):
    start = max(1, idx - window_size)
    end   = min(len(intervals), idx + window_size)
    local_intervals = intervals[start:end]
    if len(local_intervals) < 3:
        return np.mean(intervals)
    return np.mean(local_intervals) + np.std(local_intervals)

# ── Semantic helpers ──────────────────────────────────────────────────────
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
            opened_devices -= pairing[act_id]
    return len(opened_devices) == 0

# ── Core split ────────────────────────────────────────────────────────────
def split_sequence(sequence, interval_threshold, total_threshold, data_name):
    n = len(sequence[0])
    if n == 1:
        return [sequence]

    intervals = extract_interval(sequence)

    # FIX 3: استخدم adaptive_threshold في المقارنة فعلاً
    adaptive_threshold = adaptive_interval_threshold(sequence)

    # Pass 1 — interval + semantic
    pass1_result = []
    current = [sequence[:, 0]]

    for i in range(1, n):
        event_col      = sequence[:, i]
        action_id      = int(event_col[ACTION_IDX])
        prev_action_id = int(sequence[ACTION_IDX][i - 1])

        similarity     = semantic_similarity(prev_action_id, action_id, data_name)
        semantic_break = similarity < 0.5

        local_threshold = dynamic_threshold(intervals, i)

        # FIX 4: استخدم AND مع semantic_break مش OR
        # يعني: قطع بس لو الـ gap كبير AND مفيش ترابط semantically
        should_split = (intervals[i] > local_threshold) and semantic_break

        if should_split:
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

    # Pass 2 — total duration
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

# ── Post-processing ───────────────────────────────────────────────────────
def filter_single_events(sequences):
    return [seq for seq in sequences if len(seq[0]) > 1]

def validate_alignment(sequences):
    valid = []
    for i, seq in enumerate(sequences):
        flat_len = seq.shape[0] * seq.shape[1]
        if flat_len % 4 == 0:
            valid.append(seq)
        else:
            print(f"[split] WARNING: seq {i} dropped — length {flat_len} not div by 4")
    return valid

# ── Public API ────────────────────────────────────────────────────────────
def split(file_path, interval_threshold, total_threshold, data_name):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    shaped = [np.reshape(np.array(row), (-1, 4)).T for row in data]
    all_subsequences = []
    for arr in shaped:
        parts = split_sequence(arr, interval_threshold, total_threshold, data_name)
        all_subsequences.extend(parts)
    all_subsequences = filter_single_events(all_subsequences)
    all_subsequences = validate_alignment(all_subsequences)
    return [arr.reshape(1, -1).tolist()[0] for arr in all_subsequences]

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
