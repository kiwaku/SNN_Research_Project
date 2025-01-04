import torch
from dvs_gesture_dataset import DVS346Sign
from torch.utils.data import DataLoader

def fs_mem_update(ops, x, mem, decay):
    mem = mem + ops(x) * decay
    return mem

def fs_coding(mem, decay, threshold):
    spike = torch.where(mem >= threshold, torch.ones_like(mem), torch.zeros_like(mem))
    mem = mem - decay * spike
    return mem, spike


if __name__ == "__main__":
    root_dir = "/Users/kayraozturk/Downloads/DvsGesture"
    labels_path = "/Users/kayraozturk/Downloads/DvsGesture"
    time_bin = 10000

    dataset = DVS346Sign(
        root_dir=root_dir,
        labels_path=labels_path,
        time_bin=time_bin
    )

    data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

    for spike_frames, labels in data_loader:
        print(f"Spike Frames Shape: {spike_frames.shape}, Labels: {labels}")