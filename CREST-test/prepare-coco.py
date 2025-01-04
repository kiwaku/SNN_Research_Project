import os
import torch
from pycocotools.coco import COCO
import numpy as np
import cv2  # For visualization

# Step 1: Load COCO Annotations
def load_coco_data(annotation_file, image_id):
    coco = COCO(annotation_file)
    anns = coco.loadAnns(coco.getAnnIds(imgIds=image_id))
    bboxes = torch.tensor([ann['bbox'] for ann in anns], dtype=torch.float32)
    return bboxes

# Step 2: Assign Synthetic Timestamps
def assign_timestamps(bboxes, start_time=0.0, interval=0.1):
    timestamps = torch.arange(0, len(bboxes)) * interval + start_time
    return torch.cat((bboxes, timestamps.unsqueeze(1)), dim=1)

# Step 3: Convert Bounding Boxes to Events
def convert_to_events(bboxes_with_timestamps):
    events = []
    for bbox in bboxes_with_timestamps:
        x, y = bbox[:2]  # Use top-left corner as the event location
        ts = bbox[-1]    # Timestamp
        events.append([x.item(), y.item(), ts.item()])
    return np.array(events)

# Step 4: Visualize Bounding Boxes
def visualize_bboxes(image_folder, annotation_file, image_id):
    """
    Visualize bounding boxes on the corresponding image.
    """
    coco = COCO(annotation_file)
    bboxes = load_coco_data(annotation_file, image_id)

    # Load the corresponding image
    img_info = coco.loadImgs(image_id)[0]
    img_path = os.path.join(image_folder, img_info['file_name'])
    image = cv2.imread(img_path)

    # Draw bounding boxes on the image
    for bbox in bboxes:
        x_min, y_min, width, height = bbox.int().tolist()
        x_max, y_max = x_min + width, y_min + height
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Green box

    # Display the image with bounding boxes
    cv2.imshow("Bounding Boxes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Step 5: Example Usage
if __name__ == "__main__":
    annotation_file = "/Users/kayraozturk/Downloads/annotations/instances_val2017.json"  # Update path
    image_folder = "/Users/kayraozturk/Downloads/val2017/"  # Update path
    image_id = 32735  # Example image ID

    # Load and process COCO data
    bboxes = load_coco_data(annotation_file, image_id)
    print(f"Bounding boxes:\n{bboxes}")

    # Assign synthetic timestamps
    bboxes_with_timestamps = assign_timestamps(bboxes)
    print(f"Bounding boxes with timestamps:\n{bboxes_with_timestamps}")

    # Convert to synthetic events
    events = convert_to_events(bboxes_with_timestamps)
    print(f"Synthetic events:\n{events}")

    # Visualize bounding boxes
    visualize_bboxes(image_folder, annotation_file, image_id)