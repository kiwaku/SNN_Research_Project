import os
import numpy as np
from torch.utils.data import Dataset


def parse_aedat(file_path):
    """
    Parse .aedat (AEDAT 3.1) file to extract events.
    Args:
        file_path (str): Path to the .aedat file.
    Returns:
        events (np.ndarray): Array of events [x, y, polarity, timestamp].
    """
    events = []
    with open(file_path, "rb") as f:
        # Read and skip the text header
        for line in f:
            if line.startswith(b"#"):  # Skip comment lines
                continue
            else:
                break  # End of header; move to binary data

        # Read binary event data
        while True:
            event_block = f.read(8)  # Each event is 8 bytes
            if len(event_block) < 8:
                break  # End of file
            
            # Decode event
            try:
                data, timestamp = np.frombuffer(event_block, dtype=np.uint32)
                x = (data >> 17) & 0x00001FFF
                y = (data >> 2) & 0x00001FFF
                polarity = (data >> 1) & 0x00000001
                events.append([x, y, polarity, timestamp])
            except Exception as e:
                print(f"Error decoding event in file {file_path}: {e}")
    
    events = np.array(events)
    print(f"File: {file_path}, Total Events Parsed: {len(events)}")
    return events


def bin_events(events, resolution=(128, 128), time_bin=10000, max_bins=100):
    """
    Bin events into spike frames with uniform time bins.
    Args:
        events (np.ndarray): Array of events [x, y, polarity, timestamp].
        resolution (tuple): Resolution of the output frames (width, height).
        time_bin (int): Temporal bin size in microseconds.
        max_bins (int): Fixed number of time bins for uniformity.
    Returns:
        spike_frames (np.ndarray): Padded or truncated spike frames.
    """
    max_time = events[:, 3].max() if len(events) > 0 else 0
    bins = int(np.ceil(max_time / time_bin))
    spike_frames = np.zeros((min(bins, max_bins), *resolution))

    for x, y, polarity, timestamp in events:
        if 0 <= x < resolution[1] and 0 <= y < resolution[0]:
            bin_idx = int(timestamp // time_bin)
            if bin_idx < max_bins:
                spike_frames[bin_idx, y, x] += polarity

    print(f"Binned Events Shape: {spike_frames.shape}")
    return spike_frames


class DVS346Sign(Dataset):
    def __init__(self, root_dir, labels_path, time_bin=10000):
        self.root_dir = root_dir
        self.labels_path = labels_path
        self.time_bin = time_bin
        self.samples = self._load_metadata()

    def _load_metadata(self):
        samples = []
        for file_name in os.listdir(self.root_dir):
            if file_name.endswith(".aedat"):
                aedat_file = os.path.join(self.root_dir, file_name)
                label_file = os.path.join(
                    self.labels_path,
                    file_name.replace(".aedat", "_labels.csv")
                )

                print(f"Processing AEDAT file: {aedat_file}")
                print(f"Looking for Label CSV file: {label_file}")

                if not os.path.exists(label_file):
                    print(f"Label file missing for {aedat_file}. Skipping.")
                    continue

                gestures = load_labels(label_file)
                if len(gestures) == 0:
                    print(f"No gestures found in {label_file}. Skipping.")
                    continue
                samples.append((aedat_file, gestures))
        print(f"Total samples loaded: {len(samples)}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        aedat_file, gestures = self.samples[index]
        events = parse_aedat(aedat_file)

        if len(events) == 0:
            print(f"No events found in {aedat_file}. Returning default.")
            return np.zeros((1, 100, 128, 128)), [-1]

        spike_frames = []
        labels = []

        for gesture in gestures:
            start_time = gesture["start_time"]
            end_time = gesture["end_time"]
            gesture_class = gesture["class"]

            # Filter events for this gesture
            gesture_events = events[
                (events[:, 3] >= start_time) & (events[:, 3] <= end_time)
            ]

            if len(gesture_events) == 0:
                print(f"No events found for gesture {gesture_class} in {aedat_file}.")
                continue

            # Bin events into spike frames
            frames = bin_events(gesture_events, resolution=(128, 128), time_bin=self.time_bin)
            spike_frames.append(frames)
            labels.append(gesture_class)

        # Ensure consistent size for spike frames and labels
        max_gestures = 12  # Set maximum number of gestures per file
        if len(spike_frames) < max_gestures:
            padding = np.zeros((max_gestures - len(spike_frames), 100, 128, 128))
            spike_frames = np.vstack((spike_frames, padding))
            labels.extend([-1] * (max_gestures - len(labels)))
        elif len(spike_frames) > max_gestures:
            spike_frames = np.array(spike_frames[:max_gestures])
            labels = labels[:max_gestures]

        return np.array(spike_frames), np.array(labels)


def load_labels(label_file):
    labels = []
    with open(label_file, "r") as f:
        for line in f:
            if line.startswith("class"):
                continue
            parts = line.strip().split(",")
            labels.append({
                "class": int(parts[0]),
                "start_time": int(parts[1]),
                "end_time": int(parts[2])
            })
    print(f"Loaded {len(labels)} gestures from {label_file}")
    return labels