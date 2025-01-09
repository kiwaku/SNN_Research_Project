import os
import numpy as np
import torch
from torch.utils.data import Dataset

def parse_aedat(file_path):
    """
    Parse .aedat (AEDAT 3.1) file to extract events with proper timestamp handling.
    Returns np.array of shape [N, 4], columns: x, y, polarity, timestamp
    """
    events = []
    with open(file_path, "rb") as f:
        # Skip text header lines starting with '#'
        while True:
            pos = f.tell()
            line = f.readline()
            if not line.startswith(b"#"):
                f.seek(pos)  # rewind to the last valid position
                break

        # Parse binary blocks
        while True:
            header = f.read(28)  # 28-byte block header
            if len(header) < 28:
                break  # end of file

            (eventType, eventSource, eventSize, eventTSOffset,
             eventTSOverflow, eventCapacity, eventNumber,
             eventValid) = np.frombuffer(header, dtype=np.uint16, count=2).tolist() + \
                           np.frombuffer(header, dtype=np.uint32, count=6, offset=4).tolist()

            # Skip blocks that are not Polarity events
            if eventType != 1:
                f.seek(eventSize * eventNumber, 1)
                continue

            # Read polarity events
            event_data = f.read(eventSize * eventNumber)
            if len(event_data) < eventSize * eventNumber:
                break  # incomplete block

            raw_events = np.frombuffer(event_data, dtype=np.uint32).reshape(-1, 2)
            block_timestamp_base = (eventTSOverflow * (2 ** 32)) + eventTSOffset

            for data, ts in raw_events:
                x = (data >> 17) & 0x00001FFF
                y = (data >> 2)  & 0x00001FFF
                polarity = (data >> 1) & 0x1
                abs_ts = block_timestamp_base + ts
                events.append([x, y, polarity, abs_ts])

    return np.array(events)


def bin_events(events, resolution=(128, 128), time_bin=10000, max_bins=100, start_time=0):
    """
    Bin events into spike frames with uniform time bins,
    using (timestamp - start_time) to find bin index.
    Returns shape [max_bins, H, W].
    """
    if len(events) == 0:
        return np.zeros((max_bins, *resolution), dtype=np.float32)

    spike_frames = np.zeros((max_bins, *resolution), dtype=np.float32)
    max_time = events[:, 3].max() - start_time

    for (x, y, polarity, timestamp) in events:
        relative_ts = timestamp - start_time
        bin_idx = min(int(relative_ts // time_bin), max_bins - 1)
        if 0 <= x < resolution[1] and 0 <= y < resolution[0]:
            if polarity in {0, 1}:
                spike_frames[bin_idx, y, x] += polarity

    return spike_frames


def load_labels(label_file):
    """
    Loads up to 11 gestures from CSV, ensuring classes in [1..11].
    Returns list of dicts: [{"class": c, "start_time": st, "end_time": et}, ...]
    """
    labels = []
    with open(label_file, "r") as f:
        for line in f:
            if line.startswith("class"):
                continue
            parts = line.strip().split(",")
            if len(parts) == 3:
                g_class = int(parts[0])
                if 1 <= g_class <= 11:
                    labels.append({
                        "class": g_class,
                        "start_time": int(parts[1]),
                        "end_time": int(parts[2])
                    })
                    if len(labels) == 11:
                        break
    return labels


class DVS346Sign(Dataset):
    """
    Automatically bins events from AEDAT files, computes a global mean & std across ALL gestures,
    then saves normalized spike frames in .pt files. Each .pt includes:
       - "spike_frames": Tensor [T, H, W]
       - "label": int in [0..10]
       - "mean": global_mean
       - "std": global_std
    """
    def __init__(self, root_dir, labels_path, max_bins=100, precomputed_dir=None):
        self.root_dir = root_dir
        self.labels_path = labels_path
        self.max_bins = max_bins
        self.precomputed_dir = precomputed_dir
        self.samples = []  # will store paths to the .pt files

        if self.precomputed_dir:
            os.makedirs(self.precomputed_dir, exist_ok=True)

        print("[DEBUG] Starting DVS346Sign initialization...")

        # 1) Collect all (frames, label) in memory (UN-NORMALIZED)
        all_frames = []  # will be list of (frames, label)
        for file_name in os.listdir(self.root_dir):
            if file_name.endswith(".aedat"):
                aedat_file = os.path.join(self.root_dir, file_name)
                label_file = os.path.join(
                    self.labels_path, file_name.replace(".aedat", "_labels.csv")
                )

                if not os.path.exists(label_file):
                    print(f"[WARNING] Label file missing for {aedat_file}. Skipping.")
                    continue

                events = parse_aedat(aedat_file)
                gestures = load_labels(label_file)

                for i, gesture in enumerate(gestures):
                    start_t = gesture["start_time"]
                    end_t = gesture["end_time"]
                    duration = end_t - start_t
                    if duration <= 0:
                        print(f"[ERROR] Invalid gesture duration: {gesture}")
                        continue

                    gesture_events = events[
                        (events[:, 3] >= start_t) & (events[:, 3] <= end_t)
                    ]
                    if len(gesture_events) == 0:
                        print(f"[WARNING] No events found for gesture {gesture['class']} in {file_name}.")
                        continue

                    # dynamic time_bin
                    time_bin = max(1, duration // self.max_bins)
                    frames = bin_events(
                        gesture_events,
                        resolution=(128, 128),
                        time_bin=time_bin,
                        max_bins=self.max_bins,
                        start_time=start_t
                    )
                    label = gesture["class"] - 1
                    all_frames.append((frames, label, file_name, i))

        # 2) Compute global mean & std across ALL frames
        if len(all_frames) == 0:
            print("[ERROR] No valid gestures found. Dataset is empty.")
            return

        # Flatten all frames for computing global mean/std
        print("[DEBUG] Computing global mean/std over all binned gestures...")
        cat_array = np.concatenate([
            f[0].reshape(-1)  # f[0] = frames
            for f in all_frames
        ], axis=0)
        global_mean = cat_array.mean()
        global_std = cat_array.std()
        if global_std < 1e-9:
            print(f"[WARNING] Global std is very small ({global_std})."
                  " Setting it to 1.0 to avoid zero division.")
            global_std = 1.0

        print(f"[DEBUG] Global Mean: {global_mean:.6f}, Global Std: {global_std:.6f}")

        # 3) Re-iterate all frames, normalize with global stats, save .pt
        for frames, label, file_name, gesture_idx in all_frames:
            # Apply the global normalization
            frames = (frames - global_mean) / global_std

            if self.precomputed_dir:
                save_path = os.path.join(
                    self.precomputed_dir,
                    f"{file_name}_gesture_{gesture_idx}.pt"
                )
                if not os.path.exists(save_path):
                    # Save .pt file
                    torch.save({
                        "spike_frames": frames,  # still a NumPy array
                        "label": label,
                        "mean": global_mean,
                        "std": global_std
                    }, save_path)
                    print(f"[DEBUG] Saved normalized gesture {gesture_idx} to {save_path}")
                else:
                    print(f"[DEBUG] File already exists: {save_path}")

                # Append the .pt path to samples
                self.samples.append(save_path)
            else:
                # If no precomputed_dir, we store the unrolled frames in-memory (rare case)
                self.samples.append((frames, label))

        print(f"[DEBUG] DVS346Sign initialized with {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # If we're using precomputed_dir, each entry in self.samples is a .pt file path
        item = self.samples[idx]
        if isinstance(item, str) and item.endswith(".pt"):
            data = torch.load(item)
            spike_frames = data["spike_frames"]
            label = data["label"]
            # Convert to float32 Tensor and add channel dim => [1, T, H, W]
            spike_frames = torch.tensor(spike_frames, dtype=torch.float32).unsqueeze(0)
            return spike_frames, label
        else:
            # Fallback if not using .pt
            frames, label = item
            frames = torch.tensor(frames, dtype=torch.float32).unsqueeze(0)
            return frames, label