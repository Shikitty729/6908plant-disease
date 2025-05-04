# Used to train YOLOv8 model (Nano) in terminal

from ultralytics import YOLO

# Initialize model, use 'yolov8n.yaml' to start from scratch, or 'yolov8n.pt' to continue training
model = YOLO('yolov8n.yaml')

# Training configuration
model.train(
    data='/home/by2387/leaf_dataset/data.yaml',  # Absolute path to the dataset configuration file
    epochs=50,                                   # Number of training epochs
    imgsz=640,                                   # Input image size
    batch=16,                                    # Batch size (reduce if memory is insufficient)
    device=0,                                    # Use GPU (set to 'cpu' to force CPU training)
    name='leaf_yolov8_terminal_run',             # Save directory name
    project='runs/train'                         # Output project path
)
