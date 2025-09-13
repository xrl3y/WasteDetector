from ultralytics import YOLO

# Load the pretrained model for classification
model = YOLO('yolov8n-cls.pt')  # You can use yolov8s-cls.pt, yolov8m-cls.pt depending on capacity

# Train the model with your images
model.train(
    data='C:\\Users\\YourUserName\\Desktop\\WasteDetector\\dataset',  # Path as raw string
    epochs=50,
    imgsz=224,
    batch=16,
    workers=4,
    verbose=True
)

# The model is automatically saved at the end of training.
# The .pt file will be saved in the runs/train/exp folder
