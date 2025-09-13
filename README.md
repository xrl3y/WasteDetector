
# Waste Detector ♻️

Waste Detector is a computer vision project that leverages a YOLOv8 deep learning model to classify waste in real time.  
The system detects categories such as **metal, organic, paper & cardboard, plastic, and glass**, providing an intuitive interface built with Tkinter to visualize the classification and guide proper recycling.  

---

## ⚙️ Requirements

This project was developed using **Python 3.10.0**, since this version is compatible with the AI models used (YOLOv8 from Ultralytics).  
It is strongly recommended to use the same version to avoid compatibility issues.

### 🛠️ Required Libraries

Make sure to install the following dependencies before running the project:

- **tkinter** → To build the graphical user interface.  
- **opencv-python (cv2)** → For real-time camera capture and image processing.  
- **numpy** → For numerical computations and array handling.  
- **imutils** → To handle image resizing and preprocessing.  
- **Pillow (PIL)** → To manage images in the Tkinter GUI.  
- **ultralytics** → Provides the YOLOv8 implementation for classification tasks.  

### 📦 Installation

You can install the required dependencies with:

```bash
pip install opencv-python numpy imutils pillow ultralytics
```


## 📥 Download / Clone the Repository

You can download the project manually as a `.zip` file or clone it directly from GitHub:

```bash
git clone https://github.com/xrl3y/WasteDetector.git
cd WasteDetector
```

📂 Project Structure
After downloading/cloning, the structure will look like this:

```bash
WasteDetector/
│── app.py                # Main application with GUI and real-time detection
│── Recibot/              # Core folder containing resources and models
│   ├── Train.py          # Script to re-train YOLOv8 model
│   ├── dataset.yaml      # Dataset configuration for training
│   ├── yolov8n-cls.pt    # Base pre-trained YOLOv8 model
│   ├── runs/             # Folder with training results (weights, logs)
│   └── setUp/            # GUI assets (backgrounds, class images, text images)
```

🔎 Explanation
- app.py → This is the main application. Running this file launches the graphical interface and real-time waste classification.

- Recibot/ → Contains all the internal logic of the project, including:

   - Code for retraining YOLOv8 models (Train.py, dataset.yaml).

   - Pre-trained models (yolov8n-cls.pt and trained weights in runs/).

   - Graphical assets used in the Tkinter interface (setUp/).

This design makes it easy to either run the pre-trained detector directly or extend the system by retraining the AI models with a custom dataset.



## 🖥️ Application (app.py)

The file **`app.py`** is the main entry point of the project.  
When executed, it launches a graphical interface called **"RECICLAJE INTELIGENTE"**, which opens the **Waste Detector** window.

### ▶️ How it works

1. The system activates the camera connected to your device.  
2. The live video feed appears inside the application window.  
3. You only need to **place the waste item in front of the camera**, and the detector will automatically classify it into one of the following categories:  
   - Metal  
   - Organic  
   - Paper & Cardboard  
   - Plastic  
   - Glass  

The interface will show:  
- The **predicted category name** and the **confidence percentage**.  
- Illustrative images and explanatory text for the detected class.

## 🖼️ Visual Example

<p align="center">
  <img src="https://github.com/user-attachments/assets/5ebd7196-0c78-44f9-b69c-da9d0315ca65" 
       alt="Waste Detector Interface" 
       width="80%" />
</p>


### 📊 Model Information

- The default training used was performed with **around 25,000 images**.  
- The model achieved an **accuracy of approximately 85%**.  
- The trained weights are located at:  WasteDetector\Recibot\runs\classify\train6\weights\best.pt

This pre-trained model is automatically loaded when running the application.  
If you want to retrain with a custom dataset, you can use the scripts provided in the `Recibot/` folder. 


---

## 🎨 Modifying the Graphical Interface

The graphical interface of the project is fully customizable.  
All the visual components such as backgrounds, class images, and text banners are stored inside the folder: WasteDetector\Recibot\setUp\


To **change or redesign the interface**, simply replace or edit the resources inside this folder.  

For example:  
- `Canva.png` → background of the main window.  
- `metal.png`, `plastico.png`, etc. → images shown for each class.  
- `metalTxt.png`, `vidrioTxt.png`, etc. → explanatory text for each category.  

This modular design allows you to adapt the GUI to your own style or branding.

---

## 📝 Dataset Configuration (dataset.yaml)

The file **`dataset.yaml`** defines the structure of the dataset used for training and retraining the YOLOv8 model.  
It specifies the paths to the training and validation datasets, as well as the classes to be detected (waste categories).  

Typical content includes:  
- Path to the dataset folder.  
- Number of classes.  
- Names of each class (e.g., `metal`, `organico`, `papel_y_carton`, `plastico`, `vidrio`).  

This file ensures that the YOLOv8 training process correctly maps your dataset to the classification labels.

---

## 🧠 Training Script (Train.py)

The file **`Train.py`** is used to train or retrain the YOLOv8 classification model.  

```python
from ultralytics import YOLO
```

# Load the pretrained model for classification
```python
model = YOLO('yolov8n-cls.pt')  # You can use yolov8s-cls.pt, yolov8m-cls.pt depending on capacity
```

# Train the model with your images
```python
model.train(
    data='C:\\Users\\YourUserName\\Desktop\\WasteDetector\\dataset',  # Path to dataset
    epochs=50,
    imgsz=224,
    batch=16,
    workers=4,
    verbose=True
)
```

# The model is automatically saved at the end of training.
# The .pt file will be saved in the runs/classify/train7 folder

🔎 Explanation

- YOLO('yolov8n-cls.pt') → Loads the pre-trained YOLOv8 classification model (lightweight version).

- model.train(...) → Starts the training process using your dataset with:

  - epochs=50 → number of iterations over the dataset.

  - imgsz=224 → image size (224x224 pixels).

  - batch=16 → number of images processed at once.

  - workers=4 → number of CPU threads for loading data.

At the end of training, the new model weights (best.pt) are saved in: WasteDetector\Recibot\runs\train\classify\train7

You can replace the default pre-trained weights with these to improve accuracy with your own dataset.


---

## ✅ Recommendations

- It is recommended to use **Python 3.10.0** to ensure compatibility with YOLOv8 models and libraries.  
- Make sure your **camera drivers** are correctly installed so the application can access the video stream.  
- Use a **well-lit environment** when testing detection to improve accuracy.  
- If you want to improve performance, consider retraining the model with **a larger or more specific dataset**.  
- When modifying the interface, keep consistent file names inside the `setUp/` folder to avoid runtime errors.  

---

## 🔮 Future Improvements

- Add support for **more waste categories** (e.g., batteries, electronics).  
- Optimize the model for **edge devices** such as Raspberry Pi or Jetson Nano.  
- Implement a **web-based interface** to allow remote use of the detector.  
- Add a **logging system** to save classification history and statistics.  
- Integrate **voice feedback** to announce detected categories in real time.  

---

## 🙌 Final Notes

Waste Detector is a project that combines **artificial intelligence, computer vision, and sustainability**.  
It aims to encourage proper recycling habits while showcasing how deep learning can be applied to solve everyday environmental challenges.  

Contributions, improvements, and suggestions are always welcome!  

---


## Author

This project was developed by **xrl3y**.

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif">



## 📜 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.


