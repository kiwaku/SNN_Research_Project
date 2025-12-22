#!/usr/bin/env python
# coding: utf-8

# In[14]:


import os
import shutil

# ==========================
# Clear Cache and Dataset
# ==========================
cache_dir = "voxel_cache"
output_dir = "crest_data"  # This is where processed dataset is stored

"""def clear_cache():
    # Clear model cache
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        os.makedirs(cache_dir)
        print("Cache directory cleared and recreated.")
    else:
        os.makedirs(cache_dir)
        print("Cache directory did not exist; created a new one.")

    # Clear dataset if it exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        print("Processed dataset cleared and recreated.")
        
        
        
    else:
        os.makedirs(output_dir)
        print("Dataset directory did not exist; created a new one.")

# Run the clear function
clear_cache()"""


# In[ ]:





# In[ ]:





# In[1]:


def run_inference(mestor, model, voxel_data, confidence_threshold=0.5, top_n=5):
    """
    Run inference on precomputed voxel data using trained CREST model.
    
    Args:
        mestor: Trained MESTOR model
        model: Trained SpikingYOLO model
        voxel_data: Tensor of shape [5, H, W] (precomputed voxel grid)
        confidence_threshold: Threshold for detection confidence
        top_n: Maximum number of detections to return
    Returns:
        List of detections with bounding boxes, class probabilities, and confidence
    """
    mestor.eval()
    model.eval()
    device = next(mestor.parameters()).device
    
    with torch.no_grad():
        voxel_batch = voxel_data.unsqueeze(0).to(device)  # [1, 5, H, W]
        features = mestor.process_voxels(voxel_batch)  # [1, 48, H/4, W/4]
        
        if torch.isnan(features).any():
            print("[ERROR] NaNs in MESTOR features!")
            return []
        
        outputs = model(features)
        
        confidence = torch.sigmoid(outputs['confidence']).squeeze(0)  # [anchor*grid, 1]
        boxes = torch.sigmoid(outputs['boxes']).squeeze(0)  # [anchor*grid, 4]
        class_probs = outputs['class_probs'].squeeze(0)  # [anchor*grid, num_classes]
        
        mask = confidence.squeeze(-1) > confidence_threshold
        if mask.sum() == 0:
            return []
        
        filtered_boxes = boxes[mask].cpu().numpy()
        filtered_confidence = confidence[mask].cpu().numpy()
        filtered_class_probs = class_probs[mask].cpu().numpy()
        predicted_class = np.argmax(filtered_class_probs, axis=1)
        
        if len(filtered_confidence) > top_n:
            top_indices = np.argsort(filtered_confidence[:, 0])[::-1][:top_n]
            filtered_boxes = filtered_boxes[top_indices]
            filtered_confidence = filtered_confidence[top_indices]
            filtered_class_probs = filtered_class_probs[top_indices]
            predicted_class = predicted_class[top_indices]
        
        pixel_boxes = []
        for box in filtered_boxes:
            x_center, y_center, width, height = box
            x_center = np.clip(x_center, 0, 1) * 640
            y_center = np.clip(y_center, 0, 1) * 480
            width = np.clip(width, 0, 1) * 640
            height = np.clip(height, 0, 1) * 480
            x1 = max(0, int(x_center - width / 2))
            y1 = max(0, int(y_center - height / 2))
            x2 = min(640, int(x_center + width / 2))
            y2 = min(480, int(y_center + height / 2))
            pixel_boxes.append([x1, y1, x2, y2])
        
        detections = []
        class_names = ['pedestrian', 'rider', 'car', 'bus', 'truck', 'bicycle', 'motorcycle', 'train']
        for i in range(len(pixel_boxes)):
            detections.append({
                'box': pixel_boxes[i],
                'class_id': int(predicted_class[i]),
                'class_name': class_names[int(predicted_class[i])],
                'confidence': float(filtered_confidence[i][0])
            })
        
        return detections
    
def run_sample_inference(preprocessed_dir, model_path=None, pretrained_path=None):
    """
    Run sample inference using preprocessed voxel data.
    
    Args:
        preprocessed_dir: Directory with preprocessed .pt files
        model_path: Path to trained SNN model (if None, trains a new model)
        pretrained_path: Path to pre-trained ANN weights (optional)
    Returns:
        Sample and detections
    """
    dataset = PreprocessedEventDataset(preprocessed_dir)
    sample_idx = np.random.randint(0, len(dataset))
    sample = dataset[sample_idx]
    
    print(f"Running inference on sample {sample_idx}")
    print(f"Approximate number of events: {sample['num_events']}")
    print(f"Number of labels: {len(sample['labels']['boxes'])}")
    
    mestor = MESTOR(image_size=(480, 640)).to(device)
    model = SpikingYOLO(num_classes=8).to(device)
    
    if model_path and os.path.exists(model_path):
        print(f"Loading SNN model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        mestor.load_state_dict(checkpoint['mestor_state_dict'])
        model.load_state_dict(checkpoint['model_state_dict'])
    elif pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading pre-trained ANN weights from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=device)
        mestor.load_state_dict(checkpoint['mestor_state_dict'])
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)  # Partial load for SNN
    else:
        print("No model found, training a new model...")
        mestor, model, _, _ = train_crest_sample(
            event_file, label_file, epochs=3, batch_size=2, max_samples=20,
            pretrained_path=pretrained_path
        )
    
    detections = run_inference(mestor, model, sample['voxels'], confidence_threshold=0.3)
    print("Detections:", detections)
    
    return sample, detections

# --------------------------
# Visualization Functions
# --------------------------
import numpy as np
import matplotlib.pyplot as plt

def visualize_event_detections(event_data, detections, figsize=(12, 10), max_events=100000, max_detections=50):
    """
    Visualize event data with detections, with limits to prevent performance issues.
    
    Args:
        event_data: Dictionary with event data ('t', 'x', 'y', 'p')
        detections: List of detection dictionaries ('box', 'class_name', 'confidence')
        figsize: Figure size
        max_events: Maximum number of events to process
        max_detections: Maximum number of detections to plot
    """
    # Subsample events if too many
    if len(event_data['x']) > max_events:
        indices = np.random.choice(len(event_data['x']), max_events, replace=False)
        x = event_data['x'][indices]
        y = event_data['y'][indices]
        p = event_data['p'][indices]
    else:
        x = event_data['x']
        y = event_data['y']
        p = event_data['p']
    
    # Create histograms for visualization
    pos_events = (p == 1)
    neg_events = (p == 0)
    H_pos, _, _ = np.histogram2d(y[pos_events], x[pos_events], 
                                 bins=[480, 640], range=[[0, 480], [0, 640]])
    H_neg, _, _ = np.histogram2d(y[neg_events], x[neg_events], 
                                 bins=[480, 640], range=[[0, 480], [0, 640]])
    
    # Normalize histograms
    if H_pos.max() > 0:
        H_pos = H_pos / H_pos.max()
    if H_neg.max() > 0:
        H_neg = H_neg / H_neg.max()
    
    # Create RGB image
    rgb_img = np.zeros((480, 640, 3))
    rgb_img[:, :, 0] = H_pos
    rgb_img[:, :, 1] = H_neg
    
    # Plot the image
    plt.figure(figsize=figsize)
    plt.imshow(rgb_img)
    
    # Limit the number of detections
    if len(detections) > max_detections:
        detections = detections[:max_detections]  # Take the top max_detections (assumes sorted by confidence)
    
    # Draw bounding boxes and labels
    for det in detections:
        x1, y1, x2, y2 = det['box']
        class_name = det['class_name']
        conf = det['confidence']
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                             fill=False, edgecolor='yellow', linewidth=2)
        plt.gca().add_patch(rect)
        label = f"{class_name} {conf:.2f}"
        plt.text(x1, y1 - 5, label, color='white', 
                 fontsize=10, backgroundcolor='black')
    
    plt.title(f"Event Visualization with {len(detections)} Detections")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
# --------------------------
# Run Sample Inference
# --------------------------


# --------------------------
# Main Execution
# --------------------------
# Uncomment to train a model
# mestor, model, train_losses, val_losses = train_crest_sample(
#     event_file, label_file, epochs=5, batch_size=2, max_samples=50
# )

# Uncomment to run inference
# sample, detections = run_sample_inference(
#     event_file, label_file, model_path=os.path.join(cache_dir, 'crest_best_model.pth')
# )
# Run this in a Jupyter notebook cell

import os
import numpy as np
import h5py
import hdf5plugin
import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# Configuration
event_file = "../../Datasets/DSEC_Detection/dsec-det/train_events/train/zurich_city_18_a/events/left/events.h5"
label_file = "../../Datasets/DSEC_Detection/dsec-det/train_object_detections/train/zurich_city_18_a/object_detections/left/tracks.npy"
cache_dir = "voxel_cache"
epochs = 10  # Reduced for demonstration
batch_size = 8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --------------------------
# MESTOR Implementation
# --------------------------
class MESTOR(nn.Module):
    def __init__(self, scales=[1, 2, 4], time_window=10e-3, image_size=(480, 640)):
        super().__init__()
        self.scales = scales
        self.time_window = time_window
        self.image_size = image_size  # (H, W)

        # Multi-scale convolution layers - but now taking 5 input channels
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(5, 16, 3, stride=s, padding=1),  # 5 input channels
                nn.BatchNorm2d(16),
                nn.LeakyReLU(0.1)
            ) for s in scales
        ])

    def temporal_binning(self, events, num_bins=5):
        t = events['t'].clone()
        t_min, t_max = t.min(), t.max()
        t = (t - t_min) / (t_max - t_min + 1e-6)
        bins = torch.zeros(num_bins, *self.image_size, device=t.device)
        bin_width = 1.0 / num_bins
        for i in range(num_bins):
            t_start = i * bin_width
            t_end = (i + 1) * bin_width
            mask = (t >= t_start) & (t < t_end)
            for polarity in [0, 1]:
                pol_mask = mask & (events['p'] == polarity)
                if pol_mask.any():
                    x_pol = events['x'][pol_mask]
                    y_pol = events['y'][pol_mask]
                    bins[i, y_pol, x_pol] += 1
            bin_max = bins[i].max()
            if bin_max > 0:
                bins[i] = bins[i] / bin_max
        return bins
    
              
    def _event2frame_torch(self, ev_xyt, time_window, k, use_torch=True):
        """
        PyTorch version of _event2frame that maintains gradient flow.
        Args:
            ev_xyt: Event data array [x, y, t]
            time_window: Time window in microseconds
            k: Index for which time chunk to process (0=latest, 1=second latest, etc.)
            use_torch: If True, return PyTorch tensor with gradients
        Returns:
            Processed bin as tensor or numpy array
        """
        # Initial processing in numpy for efficiency
        height, width = self.image_size

        # Find time range indices
        latest_time = ev_xyt[-1, 2]
        if k == 0:
            indices_end = len(ev_xyt)  # Use all events to the end
        else:
            # Find first index where time > (latest - k*window)
            indices_end = np.searchsorted(ev_xyt[:, 2], latest_time - (k-1) * time_window, side='left')

        # Find first index where time > (latest - (k+1)*window)
        indices_start = np.searchsorted(ev_xyt[:, 2], latest_time - (k+1) * time_window, side='left')

        # If we have no events in this window, return zeros
        if indices_start >= indices_end or indices_start >= len(ev_xyt):
            if use_torch:
                return torch.zeros(height, width, dtype=torch.float32)
            else:
                return np.zeros((height, width), dtype=np.float32)

        # Extract events for this time window
        window_events = ev_xyt[indices_start:indices_end]

        # Count events - create histogram using torch.histogramdd if using torch
        if use_torch:
            # Create empty bin
            bin_tensor = torch.zeros(height, width, dtype=torch.float32)

            # Efficient scatter_add_ operation (differentiable)
            if len(window_events) > 0:
                # Extract coordinates and ensure they're within bounds
                valid_y = np.clip(window_events[:, 1].astype(int), 0, height-1)
                valid_x = np.clip(window_events[:, 0].astype(int), 0, width-1)

                # Convert to torch tensors
                y_indices = torch.from_numpy(valid_y)
                x_indices = torch.from_numpy(valid_x)

                # Use scatter_add for gradient-preserving accumulation
                ones = torch.ones(len(y_indices), dtype=torch.float32)
                bin_tensor = bin_tensor.scatter_add_(0, y_indices.unsqueeze(1), 
                                               torch.ones_like(y_indices.float().unsqueeze(1)))

            # Normalize as in original code
            bin_max = bin_tensor.max()
            if bin_max > 0:
                if k == 0:  # Long time bin gets the 1500 scaling
                    bin_tensor = bin_tensor / bin_max * 1500
                    bin_tensor = torch.clamp(bin_tensor, 0, 255)
                    bin_tensor = bin_tensor / 255
                else:
                    bin_tensor = bin_tensor / bin_max

            bin_tensor.requires_grad_(True)
            return bin_tensor
        else:
            # Numpy version for comparison
            bin_array = np.zeros((height, width), dtype=np.float32)

            for event in window_events:
                y, x = int(event[1]), int(event[0])
                if 0 <= y < height and 0 <= x < width:
                    bin_array[y, x] += 1

            # Normalize as in original
            bin_max = bin_array.max()
            if bin_max > 0:
                if k == 0:
                    bin_array = bin_array / bin_max * 1500
                    bin_array[bin_array > 255] = 255
                    bin_array = bin_array / 255
                else:
                    bin_array = bin_array / bin_max

            return bin_array

    def forward(self, events):
        """
        Process events through MESTOR, following original CREST approach.
        """
        batch_size = len(events['t'])
        features_list = []

        for i in range(batch_size):
            sample_events = {
                't': events['t'][i],
                'x': events['x'][i],
                'y': events['y'][i],
                'p': events['p'][i]
            }

            if sample_events['t'].numel() == 0:
                dummy_features = torch.zeros(48, self.image_size[0] // 4, self.image_size[1] // 4,
                                           device=sample_events['t'].device)
                features_list.append(dummy_features)
                continue

            try:
                # Get 5-channel temporal bins (using the exact original approach)
                bins = self.temporal_binning(sample_events)  # Shape [5, H, W]

                # Create multi-scale features through convolution
                sample_features = []
                for conv in self.conv_layers:
                    # Process 5-channel input directly
                    conv_feat = conv(bins.unsqueeze(0))  # Add batch dim [1, 5, H, W]
                    sample_features.append(conv_feat)

                # Ensure all features have the same spatial dimensions
                target_size = (self.image_size[0] // 4, self.image_size[1] // 4)
                for j in range(len(sample_features)):
                    if sample_features[j].shape[2:] != target_size:
                        sample_features[j] = F.interpolate(
                            sample_features[j],
                            size=target_size,
                            mode='bilinear',
                            align_corners=False
                        )

                # Concatenate features from different scales
                features = torch.cat(sample_features, dim=1).squeeze(0)  # Shape [C, H/4, W/4]
                features_list.append(features)

            except Exception as e:
                print(f"[ERROR] Failed to process sample {i}: {e}")
                dummy_features = torch.zeros(48, self.image_size[0] // 4, self.image_size[1] // 4,
                                           device=sample_events['t'].device)
                features_list.append(dummy_features)

        # Stack batch together
        features = torch.stack(features_list, dim=0)

        # Debug output
        print(f"[MESTOR] Features shape: {features.shape}")
        print(f"[MESTOR] Features requires_grad: {features.requires_grad}")
        print(f"[MESTOR] Features grad_fn: {features.grad_fn}")

        return features
          
    def process_voxels(self, voxels):
        """
        Process precomputed voxel grids through convolutional layers.
        Args:
            voxels: Tensor of shape [batch, 5, H, W]
        Returns:
            Features of shape [batch, 48, H/4, W/4]
        """
        batch_size = voxels.size(0)
        sample_features = []
        for conv in self.conv_layers:
            conv_feat = conv(voxels)  # [batch, 16, H/s, W/s]
            sample_features.append(conv_feat)
        target_size = (self.image_size[0] // 4, self.image_size[1] // 4)
        for j in range(len(sample_features)):
            sample_features[j] = F.interpolate(
                sample_features[j], size=target_size, mode='bilinear', align_corners=False
            )
        features = torch.cat(sample_features, dim=1)  # [batch, 48, H/4, W/4]
        return features
# --------------------------
# Box IoU helper function
# --------------------------
def box_iou(box1, box2):
    """Calculate IoU between two sets of boxes"""
    # box format: [x_center, y_center, width, height]
    # Convert to [x1, y1, x2, y2]
    b1_x1, b1_y1 = box1[:, 0] - box1[:, 2] / 2, box1[:, 1] - box1[:, 3] / 2
    b1_x2, b1_y2 = box1[:, 0] + box1[:, 2] / 2, box1[:, 1] + box1[:, 3] / 2
    b2_x1, b2_y1 = box2[:, 0] - box2[:, 2] / 2, box2[:, 1] - box2[:, 3] / 2
    b2_x2, b2_y2 = box2[:, 0] + box2[:, 2] / 2, box2[:, 1] + box2[:, 3] / 2
    
    # Intersection area
    x1 = torch.max(b1_x1.unsqueeze(1), b2_x1)
    y1 = torch.max(b1_y1.unsqueeze(1), b2_y1)
    x2 = torch.min(b1_x2.unsqueeze(1), b2_x2)
    y2 = torch.min(b1_y2.unsqueeze(1), b2_y2)
    
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    
    # Union area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union = b1_area.unsqueeze(1) + b2_area - intersection
    
    # IoU
    iou = intersection / (union + 1e-6)
    return iou

# --------------------------
# Conjoint Learning Rule
# --------------------------
class ConjointLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, spike_reg=0.01):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.spike_reg = spike_reg  # Regularization coefficient for spike activity
        self.mse = nn.MSELoss()
        self.l1 = nn.SmoothL1Loss()
        
    def forward(self, outputs, targets):
        batch_size = outputs['boxes'].size(0)
        total_loss = 0.0

        for b in range(batch_size):
            pred_boxes = outputs['boxes'][b]
            target_boxes = targets['boxes'][b]

            if target_boxes.size(0) == 0:
                conf_loss = self.l1(outputs['confidence'][b], torch.zeros_like(outputs['confidence'][b]))
                total_loss += conf_loss
                continue

            iou_matrix = box_iou(pred_boxes, target_boxes)
            max_iou, max_idx = iou_matrix.max(dim=1)
            mask = max_iou > 0.1
            num_matches = mask.sum().item()

            if num_matches > 0:
                # Print diagnostics
                avg_iou = max_iou[mask].mean().item()
                # Use IoU-based confidence weighting (lower IoU = higher weight for confidence)
                conf_weight = 2.0 - avg_iou  # Dynamic weight based on IoU
                print(f"[DEBUG] Batch {b}: Avg IoU = {avg_iou:.4f}, Conf weight = {conf_weight:.4f}")

                matched_pred_boxes = pred_boxes[mask]
                matched_target_boxes = target_boxes[max_idx[mask]]

                # Box regression loss (use L1)
                reg_loss = self.l1(matched_pred_boxes, matched_target_boxes)

                # Dynamically weighted confidence loss
                target_conf = torch.zeros_like(outputs['confidence'][b])
                target_conf[mask] = 1.0
                conf_loss = self.l1(outputs['confidence'][b], target_conf) * conf_weight

                # Temporal loss if available
                if 'temporal_map' in outputs and 'temporal_mask' in targets:
                    temp_loss = self.mse(outputs['temporal_map'][b], targets['temporal_mask'][b])
                else:
                    temp_loss = torch.tensor(0.0, device=pred_boxes.device)

                # Spatiotemporal IoU loss
                matched_outputs = {
                    'boxes': matched_pred_boxes,
                    't_mask': outputs['t_mask'][b].unsqueeze(0).expand(matched_pred_boxes.size(0), -1)
                }
                matched_targets = {
                    'boxes': matched_target_boxes,
                    't_mask': targets['t_mask'][b].unsqueeze(0).expand(matched_target_boxes.size(0), -1)
                }
                st_loss = 1.0 - self.st_iou(matched_outputs, matched_targets)

                # Combine losses with appropriate weights
                sample_loss = self.alpha * reg_loss + self.beta * temp_loss + conf_loss + st_loss
                total_loss += sample_loss

                # Debug output
                print(f"[DEBUG] Batch {b}: Reg loss = {reg_loss.item():.4f}, Conf loss = {conf_loss.item():.4f}, Temp loss = {temp_loss.item():.4f}, ST loss = {st_loss.item():.4f}")
            else:
                conf_loss = self.l1(outputs['confidence'][b], torch.zeros_like(outputs['confidence'][b]))
                total_loss += conf_loss

        total_loss /= batch_size
    
        # Add spike-rate regularization if available
        if 'spike_map' in outputs:
            total_loss += self.spike_reg * outputs['spike_map']

        return total_loss
        
    def st_iou(self, pred, target):
        if 't_mask' in pred and 't_mask' in target:
            intersection_t = torch.logical_and(pred['t_mask'], target['t_mask']).sum(dim=1).float()
            union_t = torch.logical_or(pred['t_mask'], target['t_mask']).sum(dim=1).float()
            temporal_iou = intersection_t / (union_t + 1e-6)
        else:
            temporal_iou = torch.ones(pred['boxes'].size(0), device=pred['boxes'].device)
            
        iou_matrix = box_iou(pred['boxes'], target['boxes'])
        spatial_iou = torch.diag(iou_matrix)
        
        st_iou_val = (spatial_iou * temporal_iou).mean()
        return st_iou_val

# --------------------------
# Spiking Neural Network Backbone
# --------------------------
class SpikingYOLO(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        # Input: Features from MESTOR (48 channels: 16 * 3 scales)
        
        # Feature extractor backbone
        self.backbone = nn.Sequential(
            # Layer 1
            nn.Conv2d(48, 64, 3, padding=1),
            snn.Leaky(beta=0.9, spike_grad=self.surrogate_grad()),
            nn.MaxPool2d(2),
            
            # Layer 2
            nn.Conv2d(64, 128, 3, padding=1),
            snn.Leaky(beta=0.85, spike_grad=self.surrogate_grad()),
            nn.MaxPool2d(2),
            
            # Layer 3
            nn.Conv2d(128, 256, 3, padding=1),
            snn.Leaky(beta=0.8, spike_grad=self.surrogate_grad())
        )
        
        # Detection head
        self.num_anchors = 3  # Number of anchors per grid cell
        self.num_classes = num_classes
        self.head = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            snn.Leaky(beta=0.75, spike_grad=self.surrogate_grad()),
            nn.Conv2d(512, self.num_anchors * (5 + self.num_classes), 1)
        )
          
        
        self.temporal_attention = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(5,3,3), padding=(2,1,1))
        )
        
        
        #Change Needed?
        # Down to here
        
        # Define anchor boxes (width, height) - these should be tuned for your dataset
        #self.anchors = torch.tensor([
        #    [0.1, 0.2], [0.2, 0.3], [0.3, 0.5]  # Small, medium, large objects
        #], device=device)
        #self.anchors = torch.tensor([[0.05, 0.1], [0.15, 0.25], [0.3, 0.4]], device=device)
        # In SpikingYOLO.__init__:
        # Define anchor boxes more suitable for DSEC dataset (640x480)
        self.anchors = torch.tensor([
            [0.0323, 0.0459],  # Tiny
            [0.1108, 0.1335],  # Small
            [0.2763, 0.4073]   # Medium
        ], device=device)
          
    def surrogate_grad(self):
        """Zero-centered Integrate-and-Fire gradient approximation"""
        # Make sure this function returns a surrogate that allows gradient flow
        #return lambda x: torch.sigmoid(x * 5.0) * 5.0  # Steeper sigmoid for better approximation
        #return lambda x: torch.sigmoid(3*x) * (1 + 3*x*(1 - torch.sigmoid(3*x)))
        #return lambda x: torch.sigmoid(4*x) * (1 + 4*x*(1 - torch.sigmoid(4*x)))
        #pre_sigmoid_conf = outputs['confidence'].cpu().numpy()  # Before sigmoid
        #print("Edge Mean:", pre_sigmoid_conf[[0, -1], :, :].mean(), "Center Mean:", pre_sigmoid_conf[5:-5, 5:-5, :].mean())
        #return lambda x: torch.sigmoid(x) * (1 - torch.sigmoid(x))  # σ'(x) = σ(x)(1 - σ(x))
        return lambda x: torch.sigmoid(4*x) * (1 + 4*x*(1 - torch.sigmoid(4*x)))

    def forward(self, x, time_steps=10):
        batch_size = x.size(0)

        print(f"[DEBUG] SpikingYOLO input shape: {x.shape}")

        # Explicitly ensure input requires gradients
        if not x.requires_grad:
            x.requires_grad_(True)

        # Initialize membrane potentials for spiking layers
        mem_states = []
        for layer in self.backbone:
            if isinstance(layer, snn.Leaky):
                mem_states.append(layer.init_leaky())

        head_mem = None
        for layer in self.head:
            if isinstance(layer, snn.Leaky):
                head_mem = layer.init_leaky()

        # Track spike activity for regularization
        spike_activity = []

        # Process through time steps
        detector_outputs = []
        for t in range(time_steps):
            # Propagate through backbone - ensure gradient flow
            idx = 0
            y = x  # Input features
            for layer in self.backbone:
                if isinstance(layer, snn.Leaky):
                    # Make sure we're using the surrogate gradient function
                    spk, mem_states[idx] = layer(y, mem_states[idx])
                    y = spk
                    spike_activity.append(spk.detach().mean())  # Only detach for metrics, not computation
                    idx += 1
                else:
                    y = layer(y)

            # Propagate through detection head
            for layer in self.head:
                if isinstance(layer, snn.Leaky):
                    spk, head_mem = layer(y, head_mem)
                    y = spk
                    spike_activity.append(spk.detach().mean())
                else:
                    y = layer(y)

            # Save output for this time step
            detector_outputs.append(y)

        # ----- Begin Temporal Attention Block Replacement -----
        # Stack detector outputs: shape [time_steps, batch, channels, H, W]
        detector_stack = torch.stack(detector_outputs, dim=0)
        # Permute to get shape [batch, time_steps, channels, H, W]
        detector_stack = detector_stack.permute(1, 0, 2, 3, 4)
        # Average over the channel dimension to reduce to 1 channel: shape [batch, time_steps, 1, H, W]
        attn_input = detector_stack.mean(dim=2, keepdim=True)
        # Permute to match Conv3d expectation: [batch, 1, time_steps, H, W]
        attn_input = attn_input.permute(0, 2, 1, 3, 4)
        # Apply the temporal attention layer; output shape: [batch, 1, time_steps, H, W]
        temp_attn = self.temporal_attention(attn_input)
        # Average spatially (over H and W): shape becomes [batch, 1, time_steps]
        temp_attn = temp_attn.mean(dim=[3,4])
        # Apply softmax over the time dimension (dim=2): shape [batch, 1, time_steps]
        temporal_weights = torch.softmax(temp_attn, dim=2)
        # Permute to move time_steps to the second dimension: shape [batch, time_steps, 1]
        temporal_weights = temporal_weights.permute(0, 2, 1)
        # Unsqueeze to get shape [batch, time_steps, 1, 1, 1] for broadcasting
        temporal_weights = temporal_weights.unsqueeze(2).unsqueeze(3)
        # Weight the original detector_stack and sum over time dimension
        output = (detector_stack * temporal_weights).sum(dim=1)
        # ----- End Temporal Attention Block Replacement -----

        # Reshape output: [batch, num_anchors * (5 + num_classes), grid_h, grid_w]
        # -> [batch, grid_h, grid_w, num_anchors, (5 + num_classes)]
        grid_h, grid_w = output.shape[2:4]
        output = output.permute(0, 2, 3, 1).contiguous()
        output = output.view(batch_size, grid_h, grid_w, self.num_anchors, 5 + self.num_classes)

        # Extract components
        box_xy = torch.sigmoid(output[..., 0:2])  # Center x, y

        box_wh_raw = output[..., 2:4]
        box_wh = torch.sigmoid(box_wh_raw) * self.anchors.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        box_confidence = torch.sigmoid(output[..., 4:5])  # Objectness
        box_class_probs = torch.sigmoid(output[..., 5:])  # Class probabilities

        # Create grid offsets
        grid_y, grid_x = torch.meshgrid([torch.arange(grid_h, device=device), 
                                         torch.arange(grid_w, device=device)])
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).unsqueeze(3)
        grid_xy = grid_xy.float()
        # === DEBUG: Grid Offsets Distribution ===
        print("[DEBUG] Grid Offset Distribution:")
        print(" - First row y-values:", grid_xy[0, 0, :, 0, 1].unique().cpu().numpy())
        print(" - Middle row y-values:", grid_xy[0, grid_h//2, :, 0, 1].unique().cpu().numpy())
        print(" - Last row y-values:", grid_xy[0, -1, :, 0, 1].unique().cpu().numpy())
        # Adjust predictions to be relative to image
        box_xy = (box_xy + grid_xy) / torch.tensor([grid_w, grid_h], device=device)
        box_wh = box_wh / torch.tensor([1, 1], device=device)  # Already normalized

        # Clamp coordinates to valid range
        box_xy = torch.clamp(box_xy, 0.0, 1.0)
        box_wh = torch.clamp(box_wh, 0.0, 1.0)

        # Combine predictions into a single tensor
        boxes = torch.cat([box_xy, box_wh], dim=-1)

        # Calculate spike activity metric for regularization
        spike_map = torch.stack(spike_activity).mean()

        print(f"[DEBUG] Box confidence min: {box_confidence.min().item():.4f}, max: {box_confidence.max().item():.4f}, mean: {box_confidence.mean().item():.4f}")
        print(f"[DEBUG] Boxes grad_fn: {boxes.grad_fn}, Confidence grad_fn: {box_confidence.grad_fn}")

        # Return dictionary of outputs
        return {
            'boxes': boxes.view(batch_size, -1, 4),
            'confidence': box_confidence.view(batch_size, -1, 1),
            'class_probs': box_class_probs.view(batch_size, -1, self.num_classes),
            'spike_map': spike_map,
            'temporal_map': torch.ones(batch_size, time_steps, device=device) * box_confidence.mean(),
            't_mask': torch.ones(batch_size, time_steps, device=device) > 0.5
        }
          
def contiguous_sample_indices(timestamps, segments=3, window_size=10):
    """
    Divide timestamps into 'segments' parts and from each, select a contiguous window 
    of 'window_size' samples.
    
    Args:
        timestamps (np.array): Sorted unique timestamps.
        segments (int): Number of segments to split into.
        window_size (int): Number of consecutive samples to take from each segment.
        
    Returns:
        np.array: The selected timestamps.
    """
    total = len(timestamps)
    segment_length = total // segments
    selected = []
    
    for i in range(segments):
        start = i * segment_length
        # For the last segment, take until the end
        end = start + window_size if i < segments - 1 else min(total, start + window_size)
        # In case the segment is shorter than window_size, take the whole segment
        selected.extend(timestamps[start:end])
    
    return np.array(selected)
          
# --------------------------
# Dataset Implementation
# --------------------------

class EventDataset(Dataset):
    def __init__(self, event_file, label_file, max_samples=100):
        # Load labels
        self.labels = np.load(label_file, allow_pickle=True)

        # Load event data
        with h5py.File(event_file, "r") as f:
            # Store event data in memory (for compatibility with your existing code)
            self.event_data = {key: f["events/" + key][:] for key in ["t", "x", "y", "p"]}
            self.t_offset = f.get("t_offset", 0)[()]
            # Get time range for reporting
            self.t_min = self.event_data["t"][0]
            self.t_max = self.event_data["t"][-1]

        # Adjust label timestamps to match event timestamps
        self.labels["t"] -= self.t_offset

        # Get unique timestamps from labels
        #self.timestamps = np.unique(self.labels['t'])

        # If we have a lot of timestamps, sample uniformly across the recording
        #if len(self.timestamps) > max_samples:
        #    idx = np.linspace(0, len(self.timestamps)-1, max_samples, dtype=int)
        #    self.timestamps = self.timestamps[idx]
        # Get unique timestamps from labels
        all_timestamps = np.unique(self.labels['t'])
        # If max_samples is None or too high, use all timestamps; otherwise, sample uniformly
        if max_samples is None or max_samples >= len(all_timestamps):
            self.timestamps = all_timestamps
        else:
            idx = np.linspace(0, len(all_timestamps) - 1, max_samples, dtype=int)
            self.timestamps = all_timestamps[idx]
          
        self.time_window = 20000  # ±10ms window for events (20,000 microseconds)
        # Inside __init__ of EventDataset, after self.event_data is loaded

        # Filter timestamps to ensure they are not too close to start or end of events
        all_timestamps = np.unique(self.labels['t'])
        valid_mask = (
            (all_timestamps > self.t_min + self.time_window) &
            (all_timestamps < self.t_max - self.time_window)
        )
        filtered_timestamps = all_timestamps[valid_mask]

        # Now sample uniformly across these safe timestamps
        if max_samples is None or max_samples >= len(filtered_timestamps):
            self.timestamps = filtered_timestamps
        else:
            idx = np.linspace(0, len(filtered_timestamps) - 1, max_samples, dtype=int)
            self.timestamps = filtered_timestamps[idx]

    def __len__(self):
        return len(self.timestamps)
    
    def __getitem__(self, idx):
        # Get label timestamp
        label_timestamp = self.timestamps[idx]
        
        # Find labels for this timestamp
        matching_labels = self.labels[self.labels['t'] == label_timestamp]
        
        # Aggregate events in the time window
        event_indices = np.where(
            (self.event_data['t'] >= label_timestamp - self.time_window) &
            (self.event_data['t'] <= label_timestamp + self.time_window)
        )[0]
        
        # In your __getitem__ method, when extracting events:
        if len(event_indices) == 0:
            # Return empty arrays instead of scalars
            events = {
                't': np.array([], dtype=np.float64),
                'x': np.array([], dtype=np.int32),
                'y': np.array([], dtype=np.int32),
                'p': np.array([], dtype=np.int32)
            }
          
        # Extract events for this window
        # In EventDataset.__getitem__():
        events = {
            't': torch.from_numpy(self.event_data['t'][event_indices]).float(),
            'x': torch.from_numpy(self.event_data['x'][event_indices]).long(),
            'y': torch.from_numpy(self.event_data['y'][event_indices]).long(),
            'p': torch.from_numpy(self.event_data['p'][event_indices]).long()
        }
        
        # Apply a small vertical jitter (augmentation) to the y-coordinates
        # For example, shift by up to ±10 pixels (normalized: ±(10/height))
        jitter_pixels = int((torch.rand(1).item() - 0.5) * 10)  # ±5 pixels
        events['y'] = torch.clamp(events['y'] + jitter_pixels, 0, 479)
        
        # Normalize timestamps to [0, 1] within the window
        if len(event_indices) > 0:
            t_min, t_max = events['t'].min(), events['t'].max()
            events['t'] = (events['t'] - t_min) / (t_max - t_min + 1e-6)
          
        #if len(event_indices) > 100:
        #    drop_mask = np.random.choice([True, False], len(event_indices), p=[0.2, 0.8])
        #    event_indices = event_indices[drop_mask]
          
        max_events = 1000  # Adjust this value as needed
        if len(event_indices) > max_events:
            step = len(event_indices) // max_events
            event_indices = event_indices[::step]
        
        # Prepare labels in YOLO format
        yolo_labels = []
        for det in matching_labels:
            x, y, w, h = det["x"], det["y"], det["w"], det["h"]
            class_id = det["class_id"]
            
            # Normalize coordinates
            x_center = min(max((x + w / 2) / 640, 0), 1)
            y_center = min(max((y + h / 2) / 480, 0), 1)
            width = min(max(w / 640, 0), 1)
            height = min(max(h / 480, 0), 1)
            
            yolo_labels.append([class_id, x_center, y_center, width, height])
        
        # Convert labels to tensor
        if yolo_labels:
            labels_tensor = torch.tensor(yolo_labels, dtype=torch.float32)
        else:
            # Create a dummy tensor with one zero box if no labels
            labels_tensor = torch.zeros((1, 5), dtype=torch.float32)
        
        # Create temporal mask for conjoint learning
        temporal_mask = torch.zeros(10)  # 10 time bins
        temporal_mask[5] = 1.0  # Mark the center time bin as active
        
        sample = {
            'events': events,
            'labels': {
                'boxes': labels_tensor[:, 1:5],  # [x_center, y_center, width, height]
                'class_ids': labels_tensor[:, 0],  # class IDs
                'temporal_mask': temporal_mask,  # temporal activity mask
                't_mask': torch.ones(10) > 0.5  # binary temporal mask for IoU calculation
            },
            'timestamp': label_timestamp,
            'num_events': len(event_indices)
        }
        
        print(f"[DEBUG] Sample {idx}: Event t range = {events['t'].min():.2f}-{events['t'].max():.2f}, Label t = {label_timestamp}")
            
        return sample
              
class PreprocessedEventDataset(torch.utils.data.Dataset):
    def __init__(self, preprocessed_dir):
        self.sample_paths = sorted([
            os.path.join(preprocessed_dir, fname)
            for fname in os.listdir(preprocessed_dir)
            if fname.endswith(".pt")
        ])

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        sample = torch.load(self.sample_paths[idx])
        # Debug prints:
        print(f"Sample keys: {sample.keys()}")
        print(f"Type of sample['labels']: {type(sample['labels'])}")
        print(f"Labels content: {sample['labels']}")
        
        # Convert the NumPy array of tuples to the expected tensor format.
        # Here we assume each tuple is in the order: (t, x, y, w, h, class_id, conf, extra)
        if sample['labels'].size > 0:
            yolo_labels = []
            for det in sample['labels']:
                # Unpack the tuple
                t_val, x, y, w, h, class_id, conf, extra = det
                # Compute center coordinates and normalize
                x_center = (x + w / 2) / 640
                y_center = (y + h / 2) / 480
                width_norm = w / 640
                height_norm = h / 480
                yolo_labels.append([class_id, x_center, y_center, width_norm, height_norm])
            labels_tensor = torch.tensor(yolo_labels, dtype=torch.float32)
            # We'll also want to use the timestamp of the first detection as a sample timestamp.
            timestamp = sample['labels'][0][0]
        else:
            labels_tensor = torch.zeros((1, 5), dtype=torch.float32)
            timestamp = 0

        return {
            'voxels': sample['voxels'],  # Shape: [5, H, W]
            'labels': {
                'boxes': labels_tensor[:, 1:5],  # [num_detections, 4]
                'class_ids': labels_tensor[:, 0],  # [num_detections]
                'temporal_mask': torch.zeros(10).index_fill_(0, torch.tensor([5]), 1.0),
                't_mask': torch.ones(10, dtype=torch.bool)
            },
            'timestamp': timestamp,
            'num_events': sample['voxels'].sum().item()  # Approximate
        }
              
# Custom collate function - add this right after the EventDataset class
def custom_collate_fn(batch):
    """
    Custom collate function for PreprocessedEventDataset that handles variable-sized boxes
    """
    # Batch structure to return
    batched = {
        'voxels': [],  # Changed from 'events' to 'voxels'
        'labels': {
            'boxes': [],
            'class_ids': [],
            'temporal_mask': [],
            't_mask': []
        },
        'timestamp': [],
        'num_events': []
    }
    
    # Process each sample
    for sample in batch:
        # Add voxel data
        batched['voxels'].append(sample['voxels'])
        
        # Add labels
        batched['labels']['boxes'].append(sample['labels']['boxes'])
        batched['labels']['class_ids'].append(sample['labels']['class_ids'])
        
        # Add temporal data if present
        if 'temporal_mask' in sample['labels']:
            batched['labels']['temporal_mask'].append(sample['labels']['temporal_mask'])
        if 't_mask' in sample['labels']:
            batched['labels']['t_mask'].append(sample['labels']['t_mask'])
        
        # Add metadata
        if 'timestamp' in sample:
            batched['timestamp'].append(sample['timestamp'])
        if 'num_events' in sample:
            batched['num_events'].append(sample['num_events'])
    
    # Stack tensors that have fixed dimensions
    if batched['labels']['temporal_mask'] and len(batched['labels']['temporal_mask']) > 0:
        batched['labels']['temporal_mask'] = torch.stack(batched['labels']['temporal_mask'])
    if batched['labels']['t_mask'] and len(batched['labels']['t_mask']) > 0:
        batched['labels']['t_mask'] = torch.stack(batched['labels']['t_mask'])
    
    return batched

# --------------------------
# Training function
# --------------------------
def train_crest_sample(event_file, label_file, epochs=5, batch_size=4, max_samples=50, pretrained_path=None):
    """
    Train CREST model using preprocessed voxel grids.
    
    Args:
        event_file: Path to H5 event file (unused, kept for compatibility)
        label_file: Path to NPY label file (unused, kept for compatibility)
        epochs: Number of training epochs
        batch_size: Batch size
        max_samples: Maximum number of samples (unused with preprocessed data)
        pretrained_path: Path to pre-trained ANN weights
    Returns:
        mestor, model, train_losses, val_losses
    """
    mestor = MESTOR(image_size=(480, 640)).to(device)
    model = SpikingYOLO(num_classes=8).to(device)
    
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading pre-trained weights from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=device)
        mestor.load_state_dict(checkpoint['mestor_state_dict'])
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)  # Partial load for SNN
    
    criterion = ConjointLoss().to(device)
    optimizer = torch.optim.AdamW([
        {'params': mestor.parameters(), 'lr': 1e-4},
        {'params': model.parameters(), 'lr': 1e-4}
    ], weight_decay=1e-5)
    
    dataset = PreprocessedEventDataset(output_dir)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset = Subset(dataset, list(range(train_size)))
    val_dataset = Subset(dataset, list(range(train_size, len(dataset))))
    
    #train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate_fn)
    #val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate_fn)
    
    train_losses, val_losses = [], []
    best_loss = float('inf')
    
    for epoch in range(epochs):
        mestor.train()
        model.train()
        epoch_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            voxel_batch = torch.stack([sample['voxels'] for sample in batch]).to(device)  # [B, 5, H, W]
            features = mestor.process_voxels(voxel_batch)  # [B, 48, H/4, W/4]
            outputs = model(features)
            
            targets = [{
                'boxes': sample['labels']['boxes'].to(device),
                'temporal_mask': sample['labels']['temporal_mask'].to(device),
                't_mask': sample['labels']['t_mask'].to(device)
            } for sample in batch]
            
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(mestor.parameters()) + list(model.parameters()), max_norm=0.2)
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}")
        
        if epoch % 2 == 0 or epoch == epochs - 1:
            mestor.eval()
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    voxel_batch = torch.stack([sample['voxels'] for sample in batch]).to(device)
                    features = mestor.process_voxels(voxel_batch)
                    outputs = model(features)
                    targets = [{
                        'boxes': sample['labels']['boxes'].to(device),
                        'temporal_mask': sample['labels']['temporal_mask'].to(device),
                        't_mask': sample['labels']['t_mask'].to(device)
                    } for sample in batch]
                    val_loss += criterion(outputs, targets).item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save({
                    'mestor_state_dict': mestor.state_dict(),
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'loss': best_loss
                }, os.path.join(cache_dir, 'crest_best_model.pth'))
            print(f"Validation Loss: {avg_val_loss:.4f}")
    
    return mestor, model, train_losses, val_losses


# In[2]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
import time
from tqdm import tqdm

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Assuming these are defined in your existing code
# from your_code import MESTOR, EventDataset, custom_collate_fn, box_iou

class YOLOANN(nn.Module):
    """
    ANN version of the YOLO-like backbone from CREST, using ReLU activations instead of spiking neurons.
    Processes MESTOR output (voxel data) for object detection pre-training.
    """
    def __init__(self, num_classes=8, num_anchors=3):
        super(YOLOANN, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # Backbone: Convolutional layers with ReLU activations
        self.backbone = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=3, padding=1),  # Input: 48 channels from MESTOR (16 per scale × 3 scales)
            nn.ReLU(),
            nn.MaxPool2d(2),  # Downsample: 120×160
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Downsample: 60×80
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU()  # Output: [batch, 256, 30, 40]
        )

        # Detection head
        self.head = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, num_anchors * (5 + num_classes), kernel_size=1)  # 3×(5+8) = 39 channels
        )

        # Anchors (same as SpikingYOLO, tuned for DSEC Detection)
        self.anchors = torch.tensor([
            [0.0323, 0.0459],  # Tiny objects
            [0.1108, 0.1335],  # Small objects
            [0.2763, 0.4073]   # Medium objects
        ], device=device)

    def forward(self, x):
        """
        Forward pass through the ANN.
        Input: [batch, 48, 120, 160] from MESTOR
        Output: Dictionary with boxes, confidence, and class probabilities
        """
        # Backbone processing
        features = self.backbone(x)  # [batch, 256, 30, 40]

        # Head processing
        output = self.head(features)  # [batch, 39, 30, 40]

        # Reshape output to [batch, grid_h, grid_w, num_anchors, 5 + num_classes]
        batch_size, _, grid_h, grid_w = output.shape
        output = output.view(batch_size, grid_h, grid_w, self.num_anchors, 5 + self.num_classes)

        # Extract predictions
        box_xy = torch.sigmoid(output[..., 0:2])  # Center x, y in [0, 1]
        box_wh = torch.sigmoid(output[..., 2:4]) * self.anchors  # Width, height scaled by anchors
        box_confidence = torch.sigmoid(output[..., 4:5])  # Objectness score
        box_class_probs = torch.sigmoid(output[..., 5:])  # Class probabilities

        # Generate grid offsets
        grid_y, grid_x = torch.meshgrid([torch.arange(grid_h, device=device),
                                         torch.arange(grid_w, device=device)])
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).unsqueeze(3).float()
        # === DEBUG: Grid Offsets Distribution ===
        print("[DEBUG] Grid Offset Distribution:")
        print(" - First row y-values:", grid_xy[0, 0, :, 0, 1].unique().cpu().numpy())
        print(" - Middle row y-values:", grid_xy[0, grid_h//2, :, 0, 1].unique().cpu().numpy())
        print(" - Last row y-values:", grid_xy[0, -1, :, 0, 1].unique().cpu().numpy())
        # Adjust box_xy to image coordinates
        box_xy = (box_xy + grid_xy) / torch.tensor([grid_w, grid_h], device=device)
        boxes = torch.cat([box_xy, box_wh], dim=-1)  # [batch, grid_h, grid_w, num_anchors, 4]

        # Flatten outputs for loss computation
        return {
            'boxes': boxes.view(batch_size, -1, 4),  # [batch, num_preds, 4]
            'confidence': box_confidence.view(batch_size, -1, 1),  # [batch, num_preds, 1]
            'class_probs': box_class_probs.view(batch_size, -1, self.num_classes)  # [batch, num_preds, num_classes]
        }

class YOLOLoss(nn.Module):
    """
    Standard YOLO loss for ANN pre-training, including box regression, objectness, and classification losses.
    """
    def __init__(self, num_classes=8, num_anchors=3, grid_h=30, grid_w=40, anchors=None):
        super(YOLOLoss, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.anchors = anchors

        # Loss functions
        self.mse = nn.MSELoss()  # Box regression
        self.bce = nn.BCELoss()  # Objectness
        self.ce = nn.CrossEntropyLoss()  # Classification

    def forward(self, outputs, targets):
        """
        Compute the total loss for a batch.
        Inputs:
            outputs: Dict from YOLOANN with 'boxes', 'confidence', 'class_probs'
            targets: List of dicts with 'boxes' [num_boxes, 4] and 'class_ids' [num_boxes]
        """
        batch_size = outputs['boxes'].size(0)
        total_loss = 0.0

        for b in range(batch_size):
            pred_boxes = outputs['boxes'][b]  # [num_preds, 4]
            pred_conf = outputs['confidence'][b]  # [num_preds, 1]
            pred_class = outputs['class_probs'][b]  # [num_preds, num_classes]
            target_boxes = targets[b]['boxes']  # [num_boxes, 4]
            target_classes = targets[b]['class_ids']  # [num_boxes]

            # Handle empty target case
            if target_boxes.size(0) == 0:
                obj_loss = self.bce(pred_conf, torch.zeros_like(pred_conf))
                total_loss += obj_loss
                continue

            # Assign ground truth boxes to anchors
            gt_centers = target_boxes[:, :2]  # [num_boxes, 2]
            grid_x = (gt_centers[:, 0] * self.grid_w).floor().long()
            grid_y = (gt_centers[:, 1] * self.grid_h).floor().long()
            responsible_indices = []

            for k in range(target_boxes.size(0)):
                gt_box = target_boxes[k]
                gx, gy = grid_x[k], grid_y[k]

                # Define anchor boxes at this grid cell
                anchor_boxes = torch.zeros((self.num_anchors, 4), device=device)
                anchor_boxes[:, 0] = (gx + 0.5) / self.grid_w
                anchor_boxes[:, 1] = (gy + 0.5) / self.grid_h
                anchor_boxes[:, 2] = self.anchors[:, 0]
                anchor_boxes[:, 3] = self.anchors[:, 1]

                # Compute IoU and find best anchor
                ious = box_iou(anchor_boxes, gt_box.unsqueeze(0)).squeeze(1)
                best_anchor_idx = ious.argmax()
                pred_idx = (gy * self.grid_w * self.num_anchors) + (gx * self.num_anchors) + best_anchor_idx
                responsible_indices.append(pred_idx)

                # Losses for responsible anchor
                pred_box = pred_boxes[pred_idx]
                box_loss = self.mse(pred_box, gt_box)  # Box regression
                obj_loss = self.bce(pred_conf[pred_idx], torch.ones(1, device=device))  # Objectness
                #class_loss = self.ce(pred_class[pred_idx].unsqueeze(0), target_classes[k].unsqueeze(0))  # Classification
                class_loss = self.ce(pred_class[pred_idx].unsqueeze(0), target_classes[k].long().unsqueeze(0))
                total_loss += box_loss + obj_loss + class_loss

            # Objectness loss for non-responsible anchors
            all_pred_indices = torch.arange(pred_boxes.size(0), device=device)
            non_responsible_indices = [idx for idx in all_pred_indices if idx not in responsible_indices]
            if non_responsible_indices:
                non_responsible_conf = pred_conf[non_responsible_indices]
                obj_loss_noobj = self.bce(non_responsible_conf, torch.zeros_like(non_responsible_conf))
                total_loss += obj_loss_noobj

        total_loss /= batch_size
        return total_loss
def train_yolo_ann(output_dir, epochs=5, batch_size=4):
    mestor = MESTOR(image_size=(480, 640)).to(device)
    model = YOLOANN(num_classes=8).to(device)
    criterion = YOLOLoss(num_classes=8, num_anchors=3, grid_h=30, grid_w=40, anchors=model.anchors).to(device)
    optimizer = torch.optim.Adam(list(mestor.parameters()) + list(model.parameters()), lr=1e-3)
    
    dataset = PreprocessedEventDataset(output_dir)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset = Subset(dataset, list(range(train_size)))
    val_dataset = Subset(dataset, list(range(train_size, len(dataset))))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)
    
    for epoch in range(epochs):
        model.train()
        mestor.train()
        epoch_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            # Stack voxel tensors directly from the batch dictionary
            voxel_batch = torch.stack(batch['voxels']).to(device)  # [B, 5, H, W]
            features = mestor.process_voxels(voxel_batch)  # [B, 48, H/4, W/4]
            
            # Create targets by iterating over the batch size
            targets = [{
                'boxes': batch['labels']['boxes'][i].to(device),
                'class_ids': batch['labels']['class_ids'][i].to(device)
            } for i in range(len(batch['voxels']))]
            
            outputs = model(features)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return mestor, model


# In[43]:


import os
import torch
import h5py
import numpy as np
from tqdm import tqdm

# ==========================
# Directories Setup
# ==========================
cache_dir = "voxel_cache"
output_dir = "crest_data"  # Where preprocessed samples will be saved

# Create directories if they don't exist
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# ==========================
# Device and MESTOR Setup
# ==========================
# Replace the following MESTOR definition or import with your own model definition.
# Here’s an example placeholder MESTOR implementation:
class MESTOR(torch.nn.Module):
    def __init__(self, scales=[1, 2, 4], time_window=10e-3, image_size=(480, 640)):
        super(MESTOR, self).__init__()
        self.scales = scales
        self.time_window = time_window
        self.image_size = image_size
        # Create a list of simple conv layers for multi-scale processing.
        self.conv_layers = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(5, 16, kernel_size=3, stride=s, padding=1),
                torch.nn.BatchNorm2d(16),
                torch.nn.LeakyReLU(0.1)
            ) for s in scales
        ])

    def temporal_binning(self, events, num_bins=5):
        # Normalize time between 0 and 1
        t = events['t'].clone()
        t_min, t_max = t.min(), t.max()
        t = (t - t_min) / (t_max - t_min + 1e-6)
        # Create an empty tensor for bins
        bins = torch.zeros(num_bins, *self.image_size, device=t.device)
        bin_width = 1.0 / num_bins
        for i in range(num_bins):
            t_start = i * bin_width
            t_end = (i + 1) * bin_width
            mask = (t >= t_start) & (t < t_end)
            # Process both polarities
            for polarity in [0, 1]:
                pol_mask = mask & (events['p'] == polarity)
                if pol_mask.any():
                    x_pol = events['x'][pol_mask]
                    y_pol = events['y'][pol_mask]
                    bins[i, y_pol, x_pol] += 1
            bin_max = bins[i].max()
            if bin_max > 0:
                bins[i] = bins[i] / bin_max
        return bins

    def forward(self, events):
        # This forward just returns the voxel grid via temporal binning.
        # In your full training pipeline, you would further process the voxels.
        batch_size = 1  # In this preprocessing function, we process one sample at a time.
        voxel_grid = self.temporal_binning(events)  # [num_bins, H, W]
        return voxel_grid

# Set device and instantiate MESTOR
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
mestor = MESTOR(image_size=(480, 640)).to(device)
mestor.eval()  # Set MESTOR to evaluation mode

# ==========================
# Preprocessing Function
# ==========================
def preprocess_all_samples(event_file, label_file, output_dir, max_samples=None):
    """
    Preprocess raw event data into voxel grids and save as .pt files.

    Args:
        event_file (str): Path to raw H5 event file.
        label_file (str): Path to .npy label file.
        output_dir (str): Directory to save preprocessed .pt files.
        max_samples (int or None): Maximum number of samples to generate. If None, process all available.
    """
    # Load labels and event data
    labels = np.load(label_file, allow_pickle=True)
    with h5py.File(event_file, "r") as f:
        event_data = {key: f["events/" + key][:] for key in ["t", "x", "y", "p"]}
        t_offset = f.get("t_offset", 0)[()]
    labels["t"] -= t_offset  # Align label timestamps

    # Unique timestamps for samples
    all_timestamps = np.unique(labels["t"])
    if max_samples is not None and max_samples < len(all_timestamps):
        idx = np.linspace(0, len(all_timestamps) - 1, max_samples, dtype=int)
        timestamps = all_timestamps[idx]
    else:
        timestamps = all_timestamps

    time_window = 20000  # ±10ms window in microseconds

    print(f"Starting preprocessing of {len(timestamps)} samples...")

    for i, ts in enumerate(tqdm(timestamps, desc="Preprocessing samples")):
        sample_path = os.path.join(output_dir, f"sample_{i:05d}.pt")
        if os.path.exists(sample_path):
            continue  # Skip if file already exists

        # Extract labels for this timestamp
        sample_labels = labels[labels['t'] == ts]

        # Extract events within the time window
        event_indices = np.where(
            (event_data['t'] >= ts - time_window) & (event_data['t'] <= ts + time_window)
        )[0]
        if len(event_indices) == 0:
            continue  # Skip if no events found

        # Convert events to torch tensors
        sample_events = {
            't': torch.from_numpy(event_data['t'][event_indices]).float(),
            'x': torch.from_numpy(event_data['x'][event_indices]).long(),
            'y': torch.from_numpy(event_data['y'][event_indices]).long(),
            'p': torch.from_numpy(event_data['p'][event_indices]).long()
        }

        # Compute voxel grid with MESTOR
        with torch.no_grad():
            voxel_grid = mestor.temporal_binning(sample_events)

        # Save the preprocessed sample as a dictionary
        sample_dict = {
            'voxels': voxel_grid.cpu(),  # Save on CPU
            'labels': sample_labels  # Save the raw labels for this sample
        }

        torch.save(sample_dict, sample_path)

    print(f"\n✅ Preprocessing complete. Saved to: {output_dir}")

# ==========================
# Run Preprocessing
# ==========================
# To process all available samples, use max_samples=None.
preprocess_all_samples(
    event_file, 
    label_file, 
    output_dir=output_dir, 
    max_samples=None
)


# In[ ]:





# In[13]:


import h5py
import numpy as np

# === INPUT FILE PATHS ===
event_file = "../../Datasets/DSEC_Detection/dsec-det/train_events/train/zurich_city_18_a/events/left/events.h5"
label_file = "../../Datasets/DSEC_Detection/dsec-det/train_object_detections/train/zurich_city_18_a/object_detections/left/tracks.npy"


# === LOAD EVENT FILE AND LABEL TIMESTAMPS ===
with h5py.File(event_file, "r") as f:
    event_times = f["events/t"][:]
    t_min = event_times[0]
    t_max = event_times[-1]
    t_offset = f.get("t_offset", 0)[()]

labels = np.load(label_file, allow_pickle=True)
label_times = np.unique(labels['t'] - t_offset)

# === CHECK WHICH LABELS HAVE ENOUGH EVENT CONTEXT (±20,000 μs) ===
valid_count = 0
time_window = 20000  # ±20,000 µs = 40,000 µs total window

for t in label_times:
    if (t - time_window >= t_min) and (t + time_window <= t_max):
        valid_count += 1

print(f"Max usable samples with valid events: {valid_count}")


# In[3]:


import numpy as np

def run_diagnostics(outputs, confidence_threshold=0.6):
    """
    Diagnoses grid-based positional bias in model predictions.

    Args:
        outputs (dict): Model output with keys 'class_probs', 'confidence', 'boxes'
        confidence_threshold (float): Threshold for filtering predictions
    """
    print("\n=== Running Diagnostic Tests ===")
    
    # Extract outputs
    class_probs = outputs['class_probs'][0].detach().cpu().numpy()  # [3600, num_classes]
    confidence = outputs['confidence'][0].detach().cpu().numpy().squeeze(-1)  # [3600]
    boxes = outputs['boxes'][0].detach().cpu().numpy()  # [3600, 4]
    
    num_classes = class_probs.shape[1]
    grid_h, grid_w, anchors = 30, 40, 3

    # -------------------------------
    # 🔍 Test 1: Class Probability Heatmaps
    # -------------------------------
    print("[TEST 1] Visualizing Class Probability Maps (argmax over anchors)")
    class_maps = class_probs.reshape(grid_h, grid_w, anchors, num_classes)
    for cls in range(num_classes):
        cls_map = class_maps[:, :, :, cls].max(axis=2)  # Max over anchors
        plt.figure()
        plt.imshow(cls_map, cmap='viridis')
        plt.title(f"Class {cls} Probability Heatmap")
        plt.colorbar()
        plt.show()

    # -------------------------------
    # 🔍 Test 2: Print High-Confidence Grid Locations
    # -------------------------------
    print(f"\n[TEST 2] High-Confidence Predictions (Conf > {confidence_threshold}) Grid Locations:")
    high_conf_idx = np.where(confidence > confidence_threshold)[0]
    for idx in high_conf_idx[:15]:
        grid_cell = idx // anchors
        y = grid_cell // grid_w
        x = grid_cell % grid_w
        print(f" - Grid cell (x={x}, y={y}) → anchor={idx % anchors}")

    # -------------------------------
    # 🔍 Test 3: Print x-center and class of filtered boxes
    # -------------------------------
    print("\n[TEST 3] Class and x-center of Filtered Boxes:")
    filtered_boxes = boxes[high_conf_idx]
    filtered_class_probs = class_probs[high_conf_idx]
    predicted_classes = np.argmax(filtered_class_probs, axis=1)
    for i in range(min(15, len(filtered_boxes))):
        x_center = filtered_boxes[i][0] * 640  # denormalize
        print(f" - Box {i}: x_center = {x_center:.1f} px, class = {predicted_classes[i]}")


# In[ ]:





# In[3]:


import os
import torch

# Configuration
output_dir = "crest_data"
pretrained_path = "yoloann_pretrained.pth"
cache_dir = "cache"  # Assuming cache_dir is defined elsewhere; adjust as needed
model_path = os.path.join(cache_dir, 'crest_best_model.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Part 1: ANN Training
print("Part 1: ANN Training")
if not os.path.exists(pretrained_path):
    print("Training ANN model...")
    mestor_ann, pretrained_model = train_yolo_ann(output_dir, epochs=10, batch_size=4)
    torch.save({
        'mestor_state_dict': mestor_ann.state_dict(),
        'model_state_dict': pretrained_model.state_dict()
    }, pretrained_path)
    print(f"Pre-trained ANN weights saved to '{pretrained_path}'")
else:
    print(f"File '{pretrained_path}' already exists. Skipping ANN training.")


# In[4]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Assuming these are defined in your existing code
# from your_code import MESTOR, EventDataset, custom_collate_fn, box_iou, SpikingYOLO, ConjointLoss, YOLOANN, YOLOLoss, train_yolo_ann

def transfer_weights(ann_state_dict, snn_model):
    snn_state_dict = snn_model.state_dict()
    
    layer_mapping = {
        'backbone.0.weight': 'backbone.0.weight',
        'backbone.0.bias': 'backbone.0.bias',
        'backbone.3.weight': 'backbone.3.weight',
        'backbone.3.bias': 'backbone.3.bias',
        'backbone.6.weight': 'backbone.6.weight',
        'backbone.6.bias': 'backbone.6.bias',
        'head.0.weight': 'head.0.weight',
        'head.0.bias': 'head.0.bias',
        'head.2.weight': 'head.2.weight',
        'head.2.bias': 'head.2.bias',
    }
    
    for ann_key, snn_key in layer_mapping.items():
        if ann_key in ann_state_dict and snn_key in snn_state_dict:
            snn_state_dict[snn_key] = ann_state_dict[ann_key]
        else:
            print(f"Warning: {ann_key} not found in ANN state dict or {snn_key} not in SNN state dict")
    
    snn_model.load_state_dict(snn_state_dict)
    print("Weights transferred successfully 1")
    

def train_crest_sample(event_file, label_file, epochs=5, batch_size=4, max_samples=50, pretrained_path=None):
    mestor = MESTOR(image_size=(480, 640)).to(device)
    model = SpikingYOLO(num_classes=8).to(device)
    
    # Make sure the cache directory exists
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    if pretrained_path:
        transfer_weights(pretrained_path, model)
    
    criterion = ConjointLoss().to(device)
    optimizer = torch.optim.AdamW([
        {'params': mestor.parameters(), 'lr': 1e-4},
        {'params': model.parameters(), 'lr': 1e-4}
    ], weight_decay=1e-5)
    
    # Use preprocessed dataset
    dataset = PreprocessedEventDataset(output_dir)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    # Deterministic split based on temporal order
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, len(dataset)))
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    print(f"Train set size: {len(train_dataset)}, Val set size: {len(val_dataset)}")
    
    # Define custom collate function
    def custom_collate_fn(batch):
        voxels = torch.stack([sample['voxels'] for sample in batch])
        labels = [sample['labels'] for sample in batch]  # List of label dictionaries
        timestamps = [sample.get('timestamp', None) for sample in batch]
        return {
            'voxels': voxels,
            'labels': labels,
            'timestamps': timestamps
        }
    
    # DataLoader with custom collate_fn
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=False,  # Keep samples in timestamp order within each epoch
        num_workers=0,
        collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,  # Consistent validation order
        num_workers=0,
        collate_fn=custom_collate_fn
    )
    
    train_losses = []
    val_losses = []
    best_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        mestor.train()
        model.train()
        epoch_loss = 0.0
        
        for i, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
            try:
                voxel_batch = batch['voxels'].to(device)  # Already stacked by collate_fn
                features = mestor.process_voxels(voxel_batch)
                outputs = model(features)
                
                targets = {
                    'boxes': [label['boxes'].to(device) for label in batch['labels']],
                    'temporal_mask': torch.stack([label['temporal_mask'] for label in batch['labels']]).to(device),
                    't_mask': torch.stack([label['t_mask'] for label in batch['labels']]).to(device)
                }
                
                if any(box.size(0) > 0 for box in targets['boxes']):
                    loss = criterion(outputs, targets)
                else:
                    continue
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(mestor.parameters()) + list(model.parameters()), max_norm=0.2)
                optimizer.step()
                
                epoch_loss += loss.item()
            except Exception as e:
                print(f"Error in batch {i}: {e}")
                continue
        
        avg_train_loss = epoch_loss / (i + 1)
        train_losses.append(avg_train_loss)
        
        if epoch % 2 == 0 or epoch == epochs - 1:
            mestor.eval()
            model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for val_batch in tqdm(val_loader, desc="Validation"):
                    try:
                        voxel_batch = val_batch['voxels'].to(device)
                        val_features = mestor.process_voxels(voxel_batch)
                        val_outputs = model(val_features)
                        val_targets = {
                            'boxes': [label['boxes'].to(device) for label in val_batch['labels']],
                            'temporal_mask': torch.stack([label['temporal_mask'] for label in val_batch['labels']]).to(device),
                            't_mask': torch.stack([label['t_mask'] for label in val_batch['labels']]).to(device)
                        }
                        if any(box.size(0) > 0 for box in val_targets['boxes']):
                            batch_loss = criterion(val_outputs, val_targets)
                            val_loss += batch_loss.item()
                            val_batches += 1
                    except Exception as e:
                        print(f"Error in validation batch: {e}")
                        continue
                if val_batches > 0:
                    avg_val_loss = val_loss / val_batches
                    val_losses.append(avg_val_loss)
                    if avg_val_loss < best_loss:
                        best_loss = avg_val_loss
                        model_path = os.path.join(cache_dir, 'crest_best_model.pth')
                        torch.save({
                            'mestor_state_dict': mestor.state_dict(),
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'epoch': epoch,
                            'loss': best_loss
                        }, model_path)
                        print(f"Saved best model with val_loss: {best_loss:.4f}")
        
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Time: {elapsed:.1f}s")
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='Training Loss')
    if val_losses:
        val_epochs = list(range(1, len(val_losses) + 1))
        plt.plot(val_epochs, val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('CREST Training Progress')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return mestor, model, train_losses, val_losses


# In[5]:


# Part 2: Weight Transfer
print("\nPart 2: Weight Transfer")
mestor = MESTOR(image_size=(480, 640)).to(device)  # Initialize MESTOR
snn_model = SpikingYOLO(num_classes=8).to(device)   # Initialize SNN model

# Load the checkpoint
pretrained_path = "yoloann_pretrained.pth"
checkpoint = torch.load(pretrained_path, map_location=device)

# Extract the ANN model's state dictionary
ann_state_dict = checkpoint['model_state_dict']

print("Transferring weights from ANN to SNN...")
transfer_weights(ann_state_dict, snn_model)  # Pass the extracted state dict
print("Weights transferred successfully.")


# In[77]:


# Part 3: SNN Training
print("\nPart 3: SNN Training")
print("Training SNN model with transferred weights...")
mestor, snn_model, train_losses, val_losses = train_crest_sample(
    event_file, label_file, epochs=4, batch_size=4, max_samples=200, 
    pretrained_path=pretrained_path
)
print("SNN training completed.")

# Optional: Run Inference
print("\nRunning inference...")
sample, detections = run_sample_inference(output_dir, model_path=model_path, pretrained_path=pretrained_path)
print("Inference completed.")


# In[78]:


# Load trained model
mestor = MESTOR(image_size=(480, 640)).to(device)
snn_model = SpikingYOLO(num_classes=8).to(device)
model_path = os.path.join(cache_dir, 'crest_best_model.pth')
checkpoint = torch.load(model_path, map_location=device)
mestor.load_state_dict(checkpoint['mestor_state_dict'])
snn_model.load_state_dict(checkpoint['model_state_dict'])
mestor.eval()
snn_model.eval()

# Load dataset
dataset = PreprocessedEventDataset(output_dir)
sample = dataset[0]
voxels = sample['voxels'].unsqueeze(0).to(device)

# Inference
with torch.no_grad():
    features = mestor.process_voxels(voxels)
    outputs = snn_model(features)

# Prepare confidence and class maps
confidence = outputs['confidence'][0].squeeze(-1).cpu().numpy()
class_probs = outputs['class_probs'][0].cpu().numpy()
grid_h, grid_w, num_anchors = 30, 40, 3
confidence_map = np.max(confidence.reshape(grid_h, grid_w, num_anchors), axis=2)
dominant_class = np.argmax(class_probs.reshape(grid_h, grid_w, num_anchors, -1).mean(axis=2), axis=2)

# Plot
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(confidence_map, cmap='hot')
plt.title("Max Confidence per Grid Cell")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(dominant_class, cmap='tab10')
plt.title("Dominant Class per Grid Cell")
plt.colorbar()
plt.tight_layout()
plt.show()


# In[6]:


import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def custom_collate_fn(batch):
    # Collate function that stacks the voxel tensors and keeps the labels as a list
    voxels = torch.stack([sample['voxels'] for sample in batch])
    labels = [sample['labels'] for sample in batch]  # Each label is a dict with keys like 'boxes', 'temporal_mask', 't_mask'
    timestamps = [sample.get('timestamp', None) for sample in batch]
    return {
        'voxels': voxels,
        'labels': labels,
        'timestamps': timestamps
    }

def retrain_snn_with_preprocessed_voxels(pretrained_path, epochs=3, batch_size=4):
    # Load the preprocessed voxel dataset from the output directory
    dataset = PreprocessedEventDataset(output_dir)
    
    # Deterministic train/validation split (80/20)
    train_size = int(0.8 * len(dataset))
    train_dataset = Subset(dataset, list(range(train_size)))
    val_dataset = Subset(dataset, list(range(train_size, len(dataset))))
    
    print(f"Train set size: {len(train_dataset)}, Val set size: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # Keep samples in timestamp order
        num_workers=0,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=custom_collate_fn
    )
    
    # Initialize models
    mestor = MESTOR(image_size=(480, 640)).to(device)
    model = SpikingYOLO(num_classes=8).to(device)
    
    # Load the pretrained checkpoint and extract the ANN state dictionary
    if pretrained_path:
        checkpoint = torch.load(pretrained_path, map_location=device)
        ann_state_dict = checkpoint['model_state_dict']
        transfer_weights(ann_state_dict, model)
    
    # Use a higher learning rate (from Code 2) and set up the optimizer
    optimizer = torch.optim.AdamW([
        {'params': mestor.parameters(), 'lr': 1e-3},
        {'params': model.parameters(), 'lr': 1e-3}
    ], weight_decay=1e-5)
    
    criterion = ConjointLoss().to(device)
    
    train_losses = []
    val_losses = []
    best_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        mestor.train()
        model.train()
        epoch_loss = 0.0
        
        for i, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
            try:
                voxel_batch = batch['voxels'].to(device)
                features = mestor.process_voxels(voxel_batch)
                outputs = model(features)
                
                targets = {
                    'boxes': [label['boxes'].to(device) for label in batch['labels']],
                    'temporal_mask': torch.stack([label['temporal_mask'] for label in batch['labels']]).to(device),
                    't_mask': torch.stack([label['t_mask'] for label in batch['labels']]).to(device)
                }
                
                # Compute loss only if there is at least one valid bounding box
                if any(box.size(0) > 0 for box in targets['boxes']):
                    # Add regularization term from Code 2
                    loss = criterion(outputs, targets) + 0.001 * torch.mean(features**2)
                else:
                    continue

                optimizer.zero_grad()
                loss.backward()
                # Optionally: adjust gradient clipping as needed
                torch.nn.utils.clip_grad_norm_(list(mestor.parameters()) + list(model.parameters()), max_norm=0.2)
                optimizer.step()
                
                epoch_loss += loss.item()
            except Exception as e:
                print(f"Error in batch {i}: {e}")
                continue
        
        avg_train_loss = epoch_loss / (i + 1)
        train_losses.append(avg_train_loss)
        print(f"[Epoch {epoch+1}] Avg Train Loss: {avg_train_loss:.4f}")
        
        # Run validation every 2 epochs or on the final epoch
        if epoch % 2 == 0 or epoch == epochs - 1:
            mestor.eval()
            model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for j, val_batch in enumerate(tqdm(val_loader, desc="Validation")):
                    try:
                        voxel_batch = val_batch['voxels'].to(device)
                        val_features = mestor.process_voxels(voxel_batch)
                        val_outputs = model(val_features)
                        val_targets = {
                            'boxes': [label['boxes'].to(device) for label in val_batch['labels']],
                            'temporal_mask': torch.stack([label['temporal_mask'] for label in val_batch['labels']]).to(device),
                            't_mask': torch.stack([label['t_mask'] for label in val_batch['labels']]).to(device)
                        }
                        if any(box.size(0) > 0 for box in val_targets['boxes']):
                            batch_loss = criterion(val_outputs, val_targets)
                            val_loss += batch_loss.item()
                            val_batches += 1
                    except Exception as e:
                        print(f"Error in validation batch: {e}")
                        continue
                if val_batches > 0:
                    avg_val_loss = val_loss / val_batches
                    val_losses.append(avg_val_loss)
                    print(f"Validation Loss: {avg_val_loss:.4f}")
                    if avg_val_loss < best_loss:
                        best_loss = avg_val_loss
                        model_path = os.path.join(cache_dir, 'crest_best_model.pth')
                        torch.save({
                            'mestor_state_dict': mestor.state_dict(),
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'epoch': epoch,
                            'loss': best_loss
                        }, model_path)
                        print(f"Saved best model with val_loss: {best_loss:.4f}")
        
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Elapsed Time: {elapsed:.1f}s")
    
    # Plot training progress
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='Training Loss')
    if val_losses:
        plt.plot(range(1, len(val_losses) + 1), val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('SNN Training Progress')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return mestor, model, train_losses, val_losses

# === Running the workflow ===

# Define your pretrained checkpoint path (pointing to the voxel cache output)
pretrained_path = "yoloann_pretrained.pth"

# Note: No event_file or label_file parameters are needed
mestor, snn_model, train_losses, val_losses = retrain_snn_with_preprocessed_voxels(
    pretrained_path, epochs=3, batch_size=4
)

# Optional: Run inference (using the same dataset and processing pipeline)
dataset = PreprocessedEventDataset(output_dir)
sample = dataset[0]
voxels = sample['voxels'].unsqueeze(0).to(device)

mestor.eval()
snn_model.eval()

with torch.no_grad():
    features = mestor.process_voxels(voxels)
    outputs = snn_model(features)

# Prepare and display confidence and class maps
confidence = outputs['confidence'][0].squeeze(-1).cpu().numpy()
class_probs = outputs['class_probs'][0].cpu().numpy()
grid_h, grid_w, num_anchors = 30, 40, 3
confidence_map = np.max(confidence.reshape(grid_h, grid_w, num_anchors), axis=2)
dominant_class = np.argmax(class_probs.reshape(grid_h, grid_w, num_anchors, -1).mean(axis=2), axis=2)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(confidence_map, cmap='hot')
plt.title("Max Confidence per Grid Cell")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(dominant_class, cmap='tab10')
plt.title("Dominant Class per Grid Cell")
plt.colorbar()
plt.tight_layout()
plt.show()


# In[26]:


def diagnostic_test(mestor, snn_model, dataset, device, sample_idx=0, show_gt_boxes=False):
    print("=" * 80)
    print("SNN DIAGNOSTIC TEST")
    print("=" * 80)
    
    sample = dataset[sample_idx]
    print(f"\nTesting sample {sample_idx} (Timestamp: {sample.get('timestamp', 'N/A')})")
    
    # Get voxel grid and print basic statistics
    voxels = sample['voxels'].to(device).unsqueeze(0)
    print(f"Voxel shape: {voxels.shape}")
    print(f"Min: {voxels.min().item():.4f}, Max: {voxels.max().item():.4f}, Mean: {voxels.mean().item():.4f}")
    
    # Run the voxel grid through MESTOR and SNN
    with torch.no_grad():
        features = mestor.process_voxels(voxels)
        outputs = snn_model(features)
    
    # Get and print output stats
    confidence = outputs['confidence'][0].cpu().numpy()
    class_probs = outputs['class_probs'][0].cpu().numpy()
    print(f"\nConfidence stats: min={confidence.min():.4f}, max={confidence.max():.4f}, mean={confidence.mean():.4f}")
    
    # Plot confidence heatmap
    try:
        H, W, A = 30, 40, 3
        conf_map = confidence.reshape(H, W, A)
        max_conf_map = conf_map.max(axis=2)
        plt.figure(figsize=(8, 6))
        plt.imshow(max_conf_map, cmap='hot')
        plt.title("Max Confidence per Grid Cell")
        plt.colorbar()
        plt.show()
    except Exception as e:
        print(f"Error plotting confidence heatmap: {e}")
    
    # Confidence reshaping
    conf = confidence.squeeze(-1) if confidence.shape[-1] == 1 else confidence
    try:
        class_probs_reshaped = class_probs.reshape(H, W, A, -1)
    except Exception as e:
        print(f"Error reshaping class probabilities: {e}")
        return
    
    pred_classes = np.argmax(class_probs_reshaped, axis=-1).flatten()
    pred_boxes = outputs['boxes'][0].cpu().numpy()
    
    conf_threshold = 0.05
    mask = conf.flatten() > conf_threshold
    filtered_boxes = pred_boxes[mask]
    filtered_conf = conf.flatten()[mask]
    filtered_classes = pred_classes[mask]
    
    print(f"\nFiltered {len(filtered_boxes)} boxes with confidence > {conf_threshold}")
    
    # Create background image from voxel mean if no raw image
    if 'image' in sample:
        background = sample['image']
        if isinstance(background, torch.Tensor):
            background = background.cpu().numpy()
    else:
        background = voxels.squeeze(0).mean(dim=0).cpu().numpy()
        background = (255 * (background - background.min()) / (background.max() - background.min() + 1e-8)).astype(np.uint8)
        background = np.stack([background] * 3, axis=-1)

    # Plot predictions and GT (optional)
    plt.figure(figsize=(10, 8))
    plt.imshow(background)

    # Plot GT boxes (green)
    if show_gt_boxes:
        if isinstance(sample['labels'], dict):
            gt_boxes = sample['labels'].get('boxes', [])
        else:
            gt_boxes = sample['labels']
        
        for box in gt_boxes:
            try:
                box_vals = box.tolist() if isinstance(box, torch.Tensor) else box
                if len(box_vals) == 4:
                    cx, cy, w, h = map(float, box_vals)
                elif len(box_vals) >= 5:
                    cx, cy, w, h = map(float, box_vals[1:5])
                else:
                    raise ValueError("Not enough values to unpack")
                x1 = int((cx - w / 2) * 640)
                y1 = int((cy - h / 2) * 480)
                x2 = int((cx + w / 2) * 640)
                y2 = int((cy + h / 2) * 480)
                gt_rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                        edgecolor='lime', facecolor='none', linewidth=2)
                plt.gca().add_patch(gt_rect)
            except Exception as e:
                print(f"Skipping invalid ground truth box {box}: {e}")

    # Plot predicted boxes (red)
    for box, cls, conf_val in zip(filtered_boxes, filtered_classes, filtered_conf):
        x1 = int((box[0] - box[2] / 2) * 640)
        y1 = int((box[1] - box[3] / 2) * 480)
        x2 = int((box[0] + box[2] / 2) * 640)
        y2 = int((box[1] + box[3] / 2) * 480)
        pred_rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                  edgecolor='red', facecolor='none', linewidth=2)
        plt.gca().add_patch(pred_rect)
        plt.text(x1, y1, f"C{cls} ({conf_val:.2f})", color='white', fontsize=8,
                 bbox=dict(facecolor='red', alpha=0.5))
    
    plt.title("Predicted Boxes (Red) and Ground Truth Boxes (Green)" if show_gt_boxes else "Predicted Boxes")
    plt.axis('off')
    plt.show()
    
diagnostic_test(mestor, snn_model, dataset, device, sample_idx=500, show_gt_boxes=True)


# In[52]:


def diagnostic_test(mestor, snn_model, dataset, device, sample_idx=0, 
                    show_gt_boxes=False, ignore_edge=False):
    print("=" * 80)
    print("SNN DIAGNOSTIC TEST")
    print("=" * 80)
    
    sample = dataset[sample_idx]
    print(f"\nTesting sample {sample_idx} (Timestamp: {sample.get('timestamp', 'N/A')})")
    
    # Get voxel grid and print basic statistics
    voxels = sample['voxels'].to(device).unsqueeze(0)
    print(f"Voxel shape: {voxels.shape}")
    print(f"Min: {voxels.min().item():.4f}, Max: {voxels.max().item():.4f}, Mean: {voxels.mean().item():.4f}")
    
    # Run the voxel grid through MESTOR and SNN
    with torch.no_grad():
        features = mestor.process_voxels(voxels)
        outputs = snn_model(features)
    
    # Get and print output stats
    confidence = outputs['confidence'][0].cpu().numpy()
    class_probs = outputs['class_probs'][0].cpu().numpy()
    print(f"\nConfidence stats: min={confidence.min():.4f}, max={confidence.max():.4f}, mean={confidence.mean():.4f}")
    
    # Plot confidence heatmap (averaged across anchors)
    try:
        H, W, A = 30, 40, 3  # Grid dimensions
        conf_map = confidence.reshape(H, W, A)
        max_conf_map = conf_map.max(axis=2)
        plt.figure(figsize=(8, 6))
        plt.imshow(max_conf_map, cmap='hot')
        plt.title("Max Confidence per Grid Cell")
        plt.colorbar()
        plt.show()
    except Exception as e:
        print(f"Error plotting confidence heatmap: {e}")
    
    # Reshape confidence as needed
    conf = confidence.squeeze(-1) if confidence.shape[-1] == 1 else confidence
    try:
        class_probs_reshaped = class_probs.reshape(H, W, A, -1)
    except Exception as e:
        print(f"Error reshaping class probabilities: {e}")
        return
    pred_classes = np.argmax(class_probs_reshaped, axis=-1).flatten()
    pred_boxes = outputs['boxes'][0].cpu().numpy()
    
    # Filter predictions based on confidence threshold
    conf_threshold = 0.048  # Adjust as needed
    flat_mask = conf.flatten() > conf_threshold
    filtered_boxes = pred_boxes[flat_mask]
    filtered_conf = conf.flatten()[flat_mask]
    filtered_classes = pred_classes[flat_mask]
    
    print(f"\nFiltered {len(filtered_boxes)} boxes with confidence > {conf_threshold}")
    
    # Optionally, ignore boxes from outer grid cells
    if ignore_edge:
        # Get indices (in the flattened array) of the predictions that passed threshold
        flat_indices = np.nonzero(flat_mask)[0]
        # Unravel indices to (row, col, anchor) using grid shape (H, W, A)
        rows, cols, anchors = np.unravel_index(flat_indices, (H, W, A))
        # Create a boolean mask: only keep predictions not in the outermost rows/columns
        valid_mask = (rows != 0) & (rows != H - 1) & (cols != 0) & (cols != W - 1)
        # Apply this additional mask to filtered arrays
        filtered_boxes = filtered_boxes[valid_mask]
        filtered_conf = filtered_conf[valid_mask]
        filtered_classes = filtered_classes[valid_mask]
        print(f"After ignoring edge grid cells, {len(filtered_boxes)} boxes remain.")
    
    # Create a background image from voxel data if no raw image is provided
    if 'image' in sample:
        background = sample['image']
        if isinstance(background, torch.Tensor):
            background = background.cpu().numpy()
    else:
        background = voxels.squeeze(0).mean(dim=0).cpu().numpy()
        background = (255 * (background - background.min()) / (background.max() - background.min() + 1e-8)).astype(np.uint8)
        background = np.stack([background] * 3, axis=-1)
    
    # Plot predictions and ground truth
    plt.figure(figsize=(10, 8))
    plt.imshow(background)
    
    # Overlay ground truth boxes in green (if desired)
    if show_gt_boxes:
        if isinstance(sample['labels'], dict):
            gt_boxes = sample['labels'].get('boxes', [])
        else:
            gt_boxes = sample['labels']
        for box in gt_boxes:
            try:
                box_vals = box.tolist() if isinstance(box, torch.Tensor) else box
                if len(box_vals) == 4:
                    cx, cy, w, h = map(float, box_vals)
                elif len(box_vals) >= 5:
                    cx, cy, w, h = map(float, box_vals[1:5])
                else:
                    raise ValueError("Not enough values to unpack")
                x1 = int((cx - w / 2) * 640)
                y1 = int((cy - h / 2) * 480)
                x2 = int((cx + w / 2) * 640)
                y2 = int((cy + h / 2) * 480)
                gt_rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                        edgecolor='lime', facecolor='none', linewidth=2)
                plt.gca().add_patch(gt_rect)
            except Exception as e:
                print(f"Skipping invalid ground truth box {box}: {e}")
    
    # Overlay predicted boxes in red
    for box, cls, conf_val in zip(filtered_boxes, filtered_classes, filtered_conf):
        x1 = int((box[0] - box[2] / 2) * 640)
        y1 = int((box[1] - box[3] / 2) * 480)
        x2 = int((box[0] + box[2] / 2) * 640)
        y2 = int((box[1] + box[3] / 2) * 480)
        pred_rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                  edgecolor='red', facecolor='none', linewidth=2)
        plt.gca().add_patch(pred_rect)
        plt.text(x1, y1, f"C{cls} ({conf_val:.2f})", color='white', fontsize=8,
                 bbox=dict(facecolor='red', alpha=0.5))
    
    plt.title("Predicted Boxes (Red) and Ground Truth Boxes (Green)" if show_gt_boxes else "Predicted Boxes")
    plt.axis('off')
    plt.show()

# Example usage:
# For a specific sample (grid edge boxes will be ignored if ignore_edge=True)
diagnostic_test(mestor, snn_model, dataset, device, sample_idx=400, 
                show_gt_boxes=True, ignore_edge=True)

# To loop over several samples:
# for idx in range(5):
#     diagnostic_test(mestor, snn_model, dataset, device, sample_idx=idx, 
#                     show_gt_boxes=True, ignore_edge=True)


# In[ ]:


# KEY TESTER ABOVE ^^^^


# In[ ]:





# In[66]:


import torch
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# ASSUMED IMPORTS: Ensure these are defined in your notebook or imported appropriately.
# from your_code import MESTOR, PreprocessedEventDataset, YOLOANN
# Note: We use PreprocessedEventDataset instead of EventDataset to match the training setup.
# =============================================================================

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the checkpoint
checkpoint = torch.load("yoloann_pretrained.pth", map_location=device)

# Load your trained ANN model
model = YOLOANN(num_classes=8).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Initialize MESTOR and load its state dictionary
mestor = MESTOR(image_size=(480, 640)).to(device)
mestor.load_state_dict(checkpoint['mestor_state_dict'])
mestor.eval()

# Define your dataset path and load the dataset
# Adjust this path to point to your preprocessed data directory
dataset = PreprocessedEventDataset(output_dir)

# Diagnostic function (mostly unchanged, adjusted for preprocessed data)
def fixed_diagnostic(mestor, model, dataset, num_samples=1, device=device):
    print("=" * 80)
    print("ENHANCED ANN DIAGNOSTIC")
    print("=" * 80)
    
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for idx in indices:
        sample = dataset[idx]
        print(f"\nTesting sample {idx} (Timestamp: {sample.get('timestamp', 'N/A')})")
        print("-" * 40)

        # Access voxel grid and labels
        voxels = sample['voxels'].to(device).unsqueeze(0)  # Shape: [1, num_bins, H, W]
        labels = sample['labels']
        gt_boxes = labels['boxes'].cpu().numpy()

        # Voxel grid statistics
        print("Voxel Grid Statistics:")
        print(f" - Shape: {voxels.shape}")
        print(f" - Min: {voxels.min().item():.4f}, Max: {voxels.max().item():.4f}, Mean: {voxels.mean().item():.4f}")

        # Voxel grid visualization
        print("\nVoxel Grid Visualization")
        voxel_np = voxels[0].cpu().numpy()  # Shape: [num_bins, H, W]
        num_bins = min(5, voxel_np.shape[0])  # Visualize up to 5 bins
        fig, axes = plt.subplots(1, num_bins, figsize=(15, 3))
        if num_bins == 1:
            axes = [axes]  # Handle single subplot case
        for i in range(num_bins):
            im = axes[i].imshow(voxel_np[i], cmap='viridis')
            axes[i].set_title(f"Bin {i}")
            plt.colorbar(im, ax=axes[i])
        plt.tight_layout()
        plt.show()

        # Feature extraction
        print("\nMESTOR Feature Extraction Test")
        with torch.no_grad():
            features = mestor.process_voxels(voxels)  # Use process_voxels for precomputed voxels
            print(f"Feature tensor shape: {features.shape}")
            print(f"Feature stats: min={features.min().item():.4f}, max={features.max().item():.4f}, mean={features.mean().item():.4f}")
            avg_feature = features[0].mean(dim=0).cpu().numpy()
            plt.figure(figsize=(8, 6))
            plt.imshow(avg_feature, cmap='inferno')
            for box in gt_boxes:
                x1 = int((box[0] - box[2]/2) * 160)  # Assumes feature map size of 160x120; adjust if different
                y1 = int((box[1] - box[3]/2) * 120)
                x2 = int((box[0] + box[2]/2) * 160)
                y2 = int((box[1] + box[3]/2) * 120)
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='lime', facecolor='none')
                plt.gca().add_patch(rect)
            plt.title("Average Feature Activation with Ground Truth Overlay")
            plt.colorbar()
            plt.show()

        # Model predictions
        print("\nModel Prediction Test")
        with torch.no_grad():
            outputs = model(features)
            pred_boxes = outputs['boxes'][0].cpu().numpy()
            confidence = outputs['confidence'][0].cpu().numpy()
            class_probs = outputs['class_probs'][0].cpu().numpy()
            pred_classes = np.argmax(class_probs, axis=-1)
            print(f"Number of predicted boxes: {len(pred_boxes)}")
            print(f"Confidence stats: min={confidence.min():.4f}, max={confidence.max():.4f}, mean={confidence.mean():.4f}")
            
            plt.figure(figsize=(8, 4))
            plt.hist(confidence.flatten(), bins=20)
            plt.title("Confidence Score Distribution")
            plt.xlabel("Confidence")
            plt.ylabel("Count")
            plt.show()

            conf_threshold = 0.9
            mask = confidence.squeeze(-1) > conf_threshold
            filtered_boxes = pred_boxes[mask]
            filtered_conf = confidence.squeeze(-1)[mask]
            filtered_classes = pred_classes[mask]

            print(f"Filtered {len(filtered_boxes)} boxes with confidence > {conf_threshold}")

            plt.figure(figsize=(10, 8))
            plt.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
            for box, cls, conf in zip(filtered_boxes, filtered_classes, filtered_conf):
                x1 = int((box[0] - box[2]/2) * 640)
                y1 = int((box[1] - box[3]/2) * 480)
                x2 = int((box[0] + box[2]/2) * 640)
                y2 = int((box[1] + box[3]/2) * 480)
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='red', facecolor='none', linewidth=1)
                plt.gca().add_patch(rect)
                plt.text(x1, y1, f"C{cls} ({conf:.2f})", color='white', fontsize=8,
                         bbox=dict(facecolor='red', alpha=0.5))
            plt.title("Filtered Predicted Boxes with Class Labels")
            plt.axis('off')
            plt.show()

        # Confidence heatmap
        print("\nConfidence Heatmap Visualization")
        try:
            H, W, A = 30, 40, 3  # Adjust if model output grid size differs
            conf_map = confidence.squeeze(-1).reshape(H, W, A)
            max_conf_map = conf_map.max(axis=2)
            plt.figure(figsize=(8, 6))
            plt.imshow(max_conf_map, cmap='hot')
            plt.title("Max Confidence per Grid Cell")
            plt.colorbar()
            plt.show()
        except Exception as e:
            print(f"Error in confidence heatmap: {e}")

        # Class dominance heatmap
        print("\nClass Dominance Heatmap per Grid Cell")
        try:
            class_grid = class_probs.reshape(H, W, A, -1)
            avg_class_grid = class_grid.mean(axis=2)
            dominant_class = np.argmax(avg_class_grid, axis=2)
            plt.figure(figsize=(8, 6))
            plt.imshow(dominant_class, cmap='tab10')
            plt.title("Most Likely Class per Grid Cell")
            plt.colorbar()
            plt.show()
        except Exception as e:
            print(f"Error in class dominance visualization: {e}")

# Run the diagnostic visualization on a random sample from the dataset
fixed_diagnostic(mestor, model, dataset, num_samples=1, device=device)


# In[63]:


checkpoint = torch.load("yoloann_pretrained.pth", map_location=device)
print(checkpoint.keys())


# In[70]:


#SNN TRAINING

mestor, snn_model, train_losses, val_losses = train_crest_sample(
    event_file, label_file, epochs=3, batch_size=4, max_samples=200, 
    pretrained_path=pretrained_path
)


# In[6]:


import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def retrain_snn_with_pretrained(event_file, label_file, pretrained_path, epochs=3, batch_size=4, max_samples=200):
    #dataset = EventDataset(event_file, label_file, max_samples=max_samples)
    dataset = PreprocessedEventDataset(output_dir)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                              collate_fn=custom_collate_fn, num_workers=0)

    mestor = MESTOR(image_size=(480, 640)).to(device)
    model = SpikingYOLO(num_classes=8).to(device)
    criterion = ConjointLoss().to(device)

    # ✅ Transfer pretrained ANN weights into SNN
    transfer_weights(pretrained_path, model)

    optimizer = torch.optim.AdamW([
        {'params': mestor.parameters(), 'lr': 1e-3},
        {'params': model.parameters(), 'lr': 1e-3}
    ], weight_decay=1e-5)

    train_losses = []
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        mestor.train()
        model.train()
        total_loss = 0

        for batch in train_loader:
            events = {
                k: [v.to(device) for v in val]
                for k, val in batch['events'].items()
            }
            targets = {
                'boxes': [v.to(device) for v in batch['labels']['boxes']],
                'temporal_mask': batch['labels']['temporal_mask'].to(device),
                't_mask': batch['labels']['t_mask'].to(device)
            }

            bins_list = []
            for sample in zip(*[events[k] for k in ['t', 'x', 'y', 'p']]):
                sample_dict = {k: v for k, v in zip(['t', 'x', 'y', 'p'], sample)}
                bins = mestor.temporal_binning(sample_dict)
                bins_list.append(bins)
            events['bins'] = bins_list

            features = mestor(events)
            outputs = model(features)
            loss = criterion(outputs, targets) + 0.001 * torch.mean(features**2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"[Epoch {epoch+1}] Avg Loss: {avg_loss:.4f}")

    return mestor, model, train_losses, dataset


def visualize_confidence_heatmap(model, mestor, dataset, index=0):
    model.eval()
    mestor.eval()

    sample = dataset[index]
    events = {k: v.to(device).unsqueeze(0) for k, v in sample['events'].items()}

    with torch.no_grad():
        features = mestor(events)
        outputs = model(features)

    confidence = outputs['confidence'][0].squeeze(-1).cpu().numpy()
    class_probs = outputs['class_probs'][0].cpu().numpy()

    grid_h, grid_w, num_anchors = 30, 40, 3
    confidence_map = np.max(confidence.reshape(grid_h, grid_w, num_anchors), axis=2)
    dominant_class = np.argmax(class_probs.reshape(grid_h, grid_w, num_anchors, -1).mean(axis=2), axis=2)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(confidence_map, cmap='hot')
    plt.title("Max Confidence per Grid Cell")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(dominant_class, cmap='tab10')
    plt.title("Dominant Class per Grid Cell")
    plt.colorbar()
    plt.tight_layout()
    plt.show()
# === RUNNING THE WORKFLOW ===

pretrained_path = "yoloann_pretrained.pth"
mestor, snn_model, losses, dataset = retrain_snn_with_pretrained(
    event_file, label_file, pretrained_path, epochs=3
)

visualize_confidence_heatmap(snn_model, mestor, dataset, index=0)


# In[9]:


import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def retrain_snn_from_scratch(event_file, label_file, epochs=3, batch_size=4, max_samples=200):
    #dataset = EventDataset(event_file, label_file, max_samples=max_samples)
    dataset = PreprocessedEventDataset(output_dir)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                              collate_fn=custom_collate_fn, num_workers=0)

    mestor = MESTOR(image_size=(480, 640)).to(device)
    model = SpikingYOLO(num_classes=8).to(device)
    criterion = ConjointLoss().to(device)

    # 🚫 No pretrained weights loaded here

    optimizer = torch.optim.AdamW([
        {'params': mestor.parameters(), 'lr': 1e-3},
        {'params': model.parameters(), 'lr': 1e-3}
    ], weight_decay=1e-5)

    train_losses = []
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        mestor.train()
        model.train()
        total_loss = 0

        for batch in train_loader:
            events = {
                k: [v.to(device) for v in val]
                for k, val in batch['events'].items()
            }
            targets = {
                'boxes': [v.to(device) for v in batch['labels']['boxes']],
                'temporal_mask': batch['labels']['temporal_mask'].to(device),
                't_mask': batch['labels']['t_mask'].to(device)
            }

            bins_list = []
            for sample in zip(*[events[k] for k in ['t', 'x', 'y', 'p']]):
                sample_dict = {k: v for k, v in zip(['t', 'x', 'y', 'p'], sample)}
                bins = mestor.temporal_binning(sample_dict)
                bins_list.append(bins)
            events['bins'] = bins_list

            features = mestor(events)
            outputs = model(features)
            loss = criterion(outputs, targets) + 0.001 * torch.mean(features**2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"[Epoch {epoch+1}] Avg Loss: {avg_loss:.4f}")

    return mestor, model, train_losses, dataset


def visualize_confidence_heatmap(model, mestor, dataset, index=0):
    model.eval()
    mestor.eval()

    sample = dataset[index]
    events = {k: v.to(device).unsqueeze(0) for k, v in sample['events'].items()}

    with torch.no_grad():
        features = mestor(events)
        outputs = model(features)

    confidence = outputs['confidence'][0].squeeze(-1).cpu().numpy()
    class_probs = outputs['class_probs'][0].cpu().numpy()

    grid_h, grid_w, num_anchors = 30, 40, 3
    confidence_map = np.max(confidence.reshape(grid_h, grid_w, num_anchors), axis=2)
    dominant_class = np.argmax(class_probs.reshape(grid_h, grid_w, num_anchors, -1).mean(axis=2), axis=2)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(confidence_map, cmap='hot')
    plt.title("Max Confidence per Grid Cell")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(dominant_class, cmap='tab10')
    plt.title("Dominant Class per Grid Cell")
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    
mestor_scratch, snn_scratch, scratch_losses, dataset_scratch = retrain_snn_from_scratch(
    event_file, label_file, epochs=3
)

visualize_confidence_heatmap(snn_scratch, mestor_scratch, dataset_scratch, index=0)


# In[71]:


import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load saved model
checkpoint = torch.load(os.path.join(cache_dir, 'crest_best_model.pth'))
mestor.load_state_dict(checkpoint['mestor_state_dict'])
snn_model.load_state_dict(checkpoint['model_state_dict'])
mestor.eval()
snn_model.eval()

# Choose a sample from validation set
sample = val_dataset[0]  # You can change index
event_tensor = {k: [v.unsqueeze(0).to(device)] for k, v in sample['events'].items()}

# Run inference
with torch.no_grad():
    features = mestor(event_tensor)
    prediction = snn_model(features)

# Get predicted boxes and confidence scores
boxes = prediction['boxes'][0].cpu().numpy()  # shape [N, 4]
confidences = prediction['confidence'][0].cpu().numpy()  # shape [N]

# Plot
plt.figure(figsize=(10, 6))
plt.imshow(torch.zeros(120, 160), cmap='gray')  # Dummy background for visual

# Draw boxes
for box, conf in zip(boxes, confidences):
    if conf < 0.3:
        continue  # filter low-confidence boxes
    x, y, w, h = box
    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='lime', facecolor='none')
    plt.gca().add_patch(rect)
    plt.text(x, y - 5, f'{conf:.2f}', color='lime', fontsize=8)

plt.title("Predicted Bounding Boxes")
plt.axis('off')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[83]:


import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Subset

# --- Visualization for SNN Predictions and Diagnostics ---

def visualize_snn_predictions_and_maps(snn_model, mestor, dataset, num_samples=4, device='cpu', confidence_threshold=0.3):
    """
    Visualizes SNN predictions over the full event image (640x480) along with:
      - A confidence heatmap (maximum confidence per grid cell)
      - A class dominance heatmap (most likely class per grid cell)

    Args:
        snn_model: Trained SpikingYOLO model.
        mestor: Trained MESTOR model.
        dataset: A dataset or subset (e.g. validation set) containing event samples.
        num_samples: Number of samples to visualize.
        device: 'cuda' or 'cpu'.
        confidence_threshold: Filter predicted boxes below this threshold.
    """
    snn_model.eval()
    mestor.eval()
    
    # We'll assume a grid of 30 x 40 and 3 anchors (as in your network)
    H, W, A = 30, 40, 3

    # Loop over a few samples
    for i in range(num_samples):
        sample = dataset[i]
        # Wrap event data into a batch of size 1
        events = {k: [v.to(device)] for k, v in sample['events'].items()}
        gt_boxes = sample['labels']['boxes'].cpu()  # ground truth boxes
        
        # --- Create full-resolution grayscale event image ---
        # Here we use the full resolution of 640 x 480 for visualization.
        img = torch.zeros((480, 640))
        xs = events['x'][0].cpu().numpy()
        ys = events['y'][0].cpu().numpy()
        for x, y in zip(xs, ys):
            if 0 <= y < 480 and 0 <= x < 640:
                img[int(y), int(x)] += 1
        img = torch.clamp(img, 0, 5) / 5.0  # normalize

        # --- Get SNN outputs ---
        with torch.no_grad():
            features = mestor(events)
            outputs = snn_model(features)

        # The outputs are flattened: shapes [batch, num_preds, ...]
        # We know the grid is 30x40 and A=3 anchors.
        # So reshape confidence and class_probs for the diagnostics.
        conf_flat = outputs['confidence'][0].squeeze(-1)  # shape: [30*40*3]
        conf_map = conf_flat.view(H, W, A)  # shape: [30, 40, 3]
        # Take maximum confidence over the anchor dimension
        max_conf_map = conf_map.max(dim=2)[0].cpu().numpy()

        # Similarly, reshape class probabilities:
        class_probs = outputs['class_probs'][0]  # shape: [30*40*3, num_classes]
        num_classes = class_probs.shape[-1]
        class_grid = class_probs.view(H, W, A, num_classes).cpu().numpy()
        # Average over anchors and take argmax to get the dominant class per grid cell
        avg_class_grid = class_grid.mean(axis=2)  # shape: [30, 40, num_classes]
        dominant_class = np.argmax(avg_class_grid, axis=2)

        # --- For visualization of predicted boxes over the event image ---
        # First, filter predictions by confidence threshold.
        pred_conf = conf_flat.cpu()
        mask = pred_conf > confidence_threshold
        pred_boxes = outputs['boxes'][0].cpu()[mask]
        pred_class_probs = outputs['class_probs'][0].cpu()[mask]
        pred_classes = torch.argmax(pred_class_probs, dim=-1)
        pred_conf = pred_conf[mask]

        # Convert predicted boxes (normalized to [0,1] relative to 120×160 input) back to full resolution.
        # Note: The SNN input is 120x160. To map to 640x480,
        # multiply x by 640/160 = 4 and y by 480/120 = 4.
        converted_boxes = []
        for box in pred_boxes:
            xc, yc, w, h = box.numpy()
            # Convert from normalized (with respect to 120x160) to full resolution:
            x_center_full = xc * 160 * 4  # 160*4 = 640
            y_center_full = yc * 120 * 4   # 120*4 = 480
            w_full = w * 160 * 4
            h_full = h * 120 * 4
            x1 = x_center_full - w_full / 2
            y1 = y_center_full - h_full / 2
            x2 = x_center_full + w_full / 2
            y2 = y_center_full + h_full / 2
            converted_boxes.append([x1, y1, x2, y2])
        converted_boxes = np.array(converted_boxes)

        # --- Plotting ---
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        axs = axs.flatten()

        # Panel 1: Event image with predicted and ground truth boxes.
        ax = axs[0]
        ax.imshow(img.numpy(), cmap='gray', extent=[0, 640, 480, 0])
        ax.set_xlim(0, 640)
        ax.set_ylim(480, 0)
        ax.set_aspect('equal')
        ax.set_title("Event Image with Boxes")
        # Draw predicted boxes (red)
        for box, cls, conf in zip(converted_boxes, pred_classes, pred_conf):
            x1, y1, x2, y2 = box
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 edgecolor='red', facecolor='none', linewidth=2)
            ax.add_patch(rect)
            ax.text(x1, y1, f"C{cls.item()} ({conf:.2f})", color='white', fontsize=8,
                    bbox=dict(facecolor='red', alpha=0.5))
        # Draw ground truth boxes (green dashed)
        for box in gt_boxes:
            xc, yc, w, h = box.numpy()
            x1 = (xc - w / 2) * 640
            y1 = (yc - h / 2) * 480
            w_full = w * 640
            h_full = h * 480
            rect = plt.Rectangle((x1, y1), w_full, h_full,
                                 edgecolor='lime', facecolor='none', linewidth=2, linestyle='--')
            ax.add_patch(rect)

        # Panel 2: Confidence heatmap.
        ax = axs[1]
        im = ax.imshow(max_conf_map, cmap='hot', extent=[0, 160, 120, 0])
        ax.set_title("Confidence Heatmap (Grid 120x160)")
        plt.colorbar(im, ax=ax)

        # Panel 3: Class dominance heatmap.
        ax = axs[2]
        im = ax.imshow(dominant_class, cmap='tab10', extent=[0, 160, 120, 0])
        ax.set_title("Dominant Class per Grid Cell")
        plt.colorbar(im, ax=ax)

        # Panel 4: Confidence histogram.
        ax = axs[3]
        ax.hist(pred_conf.numpy().flatten(), bins=20, color='blue', alpha=0.7)
        ax.set_title("Confidence Histogram")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Frequency")

        plt.tight_layout()
        plt.show()


# --- Example usage ---
# (Make sure your SNN model 'snn_model' and MESTOR 'mestor' are already loaded/trained.)
# Also, create a validation subset (for example, use the last 40 samples).
from torch.utils.data import Subset
dataset = EventDataset(event_file, label_file, max_samples=200)
val_dataset = Subset(dataset, list(range(160, 200)))  # Adjust indices as needed

# Run the visualization (using CPU or CUDA as available)
visualize_snn_predictions_and_maps(snn_model, mestor, val_dataset, num_samples=4, device='cpu', confidence_threshold=0.3)


# In[86]:


import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader

# --- Use your current EventDataset class, but add a diagnostic function ---
class EventDataset(Dataset):
    def __init__(self, event_file, label_file, max_samples=100, time_window=20000):
        # Load labels
        self.labels = np.load(label_file, allow_pickle=True)
        # Load event data
        with h5py.File(event_file, "r") as f:
            # Store event data in memory (for compatibility with your existing code)
            self.event_data = {key: f["events/" + key][:] for key in ["t", "x", "y", "p"]}
            self.t_offset = f.get("t_offset", 0)[()]
            # Get time range for reporting
            self.t_min = self.event_data["t"][0]
            self.t_max = self.event_data["t"][-1]
        # Adjust label timestamps to match event timestamps
        self.labels["t"] -= self.t_offset
        # Get unique timestamps from labels and sample uniformly if necessary
        all_timestamps = np.unique(self.labels['t'])
        if max_samples is None or max_samples >= len(all_timestamps):
            self.timestamps = all_timestamps
        else:
            idx = np.linspace(0, len(all_timestamps) - 1, max_samples, dtype=int)
            self.timestamps = all_timestamps[idx]
        self.time_window = time_window  # in microseconds
        print(f"Dataset initialized with {len(self.timestamps)} samples spanning {(self.t_max-self.t_min)/1e6:.2f} seconds")

    def __len__(self):
        return len(self.timestamps)
    
    def __getitem__(self, idx):
        # Get the label timestamp (absolute)
        label_timestamp = self.timestamps[idx]
        # Find all labels at that timestamp
        matching_labels = self.labels[self.labels['t'] == label_timestamp]
        # Instead of normalizing, first extract raw event indices in an absolute time window around the label
        event_indices = np.where(
            (self.event_data['t'] >= label_timestamp - self.time_window) &
            (self.event_data['t'] <= label_timestamp + self.time_window)
        )[0]
        if len(event_indices) == 0:
            # Return empty events if none are found
            events = {
                't': np.array([], dtype=np.float64),
                'x': np.array([], dtype=np.int32),
                'y': np.array([], dtype=np.int32),
                'p': np.array([], dtype=np.int32)
            }
        else:
            events = {
                't': torch.from_numpy(self.event_data['t'][event_indices]).float(),
                'x': torch.from_numpy(self.event_data['x'][event_indices]).long(),
                'y': torch.from_numpy(self.event_data['y'][event_indices]).long(),
                'p': torch.from_numpy(self.event_data['p'][event_indices]).long()
            }
            # Normalize event times relative to the window for training purposes
            t_min, t_max = events['t'].min(), events['t'].max()
            events['t'] = (events['t'] - t_min) / (t_max - t_min + 1e-6)
        # Prepare labels in YOLO format (normalized spatially)
        yolo_labels = []
        for det in matching_labels:
            x, y, w, h = det["x"], det["y"], det["w"], det["h"]
            class_id = det["class_id"]
            x_center = min(max((x + w / 2) / 640, 0), 1)
            y_center = min(max((y + h / 2) / 480, 0), 1)
            width = min(max(w / 640, 0), 1)
            height = min(max(h / 480, 0), 1)
            yolo_labels.append([class_id, x_center, y_center, width, height])
        if yolo_labels:
            labels_tensor = torch.tensor(yolo_labels, dtype=torch.float32)
        else:
            labels_tensor = torch.zeros((1, 5), dtype=torch.float32)
        temporal_mask = torch.zeros(10)  # dummy temporal mask
        temporal_mask[5] = 1.0
        sample = {
            'events': events,
            'labels': {
                'boxes': labels_tensor[:, 1:5],
                'class_ids': labels_tensor[:, 0],
                'temporal_mask': temporal_mask,
                't_mask': torch.ones(10) > 0.5
            },
            'timestamp': label_timestamp,
            'num_events': len(event_indices),
            # For diagnostic purposes, also return raw event times (before normalization)
            'raw_t': torch.from_numpy(self.event_data['t'][event_indices]).float() if len(event_indices) > 0 else torch.tensor([])
        }
        return sample

# --- Standalone Diagnostic Function ---
def check_event_label_alignment(dataset, num_samples=10):
    """
    For a given dataset, prints out the raw event time window (before normalization)
    and the label timestamp for each sample.
    """
    print("="*80)
    print(f"Checking event-label alignment for {num_samples} samples...")
    print("="*80)
    for i in range(num_samples):
        sample = dataset[i]
        label_t = sample['timestamp']
        if sample['raw_t'].numel() > 0:
            raw_min = sample['raw_t'].min().item()
            raw_max = sample['raw_t'].max().item()
            print(f"[Sample {i}] Label t = {label_t:.2f}, Raw event times: {raw_min:.2f} to {raw_max:.2f}")
            # Check if the label time is within the window (or near the center)
            center = (raw_min + raw_max) / 2
            if abs(center - label_t) < (sample['num_events'] > 0 and (raw_max - raw_min) * 0.1 or 0):
                print("   → Seems aligned (label near the center).")
            else:
                print("   → MISALIGNED!")
        else:
            print(f"[Sample {i}] No events found in the window for label t = {label_t:.2f}")
    print("="*80)

# --- Example usage ---
event_file = "../../Datasets/DSEC_Detection/dsec-det/train_events/train/zurich_city_18_a/events/left/events.h5"
label_file = "../../Datasets/DSEC_Detection/dsec-det/train_object_detections/train/zurich_city_18_a/object_detections/left/tracks.npy"

# Create dataset with a specified max sample count and a chosen time window (in microseconds)
dataset = EventDataset(event_file, label_file, max_samples=200, time_window=20000)
# Run the diagnostic
check_event_label_alignment(dataset, num_samples=10)


# In[ ]:





# In[7]:


#Delete ANN Training:


# In[16]:


if os.path.exists(pretrained_path):
    os.remove(pretrained_path)
    print(f"Deleted '{pretrained_path}'")
else:
    print(f"No file to delete at '{pretrained_path}'")


# In[ ]:




