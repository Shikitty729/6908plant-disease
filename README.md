# Plant Disease Detection
This project presents a deep learning pipeline for accurate plant disease detection and diagnosis. It uses a combination of image localization, classification, and language generation models to automate the end-to-end diagnosis of plant leaf disease on real-time.

Dataset used for classification models:
#https://drive.google.com/file/d/0B_voCy5O5sXMTFByemhpZllYREU/view?resourcekey=0-25uoBK9YYXXKnTliopPxDw

## Components:
### 1. Leaf Disease Detection using YOLOv8n:
- We have trained a YOLOv8n object detection model to first localize the leaf region from the input image as it also contains background.
- This is important because the background may contain noise, such as other plants or parts of the plant, sky etc, which can confuse the classifier model and result in incorrect predictions.
- By isolating the leaf from the background, we improve the classification accuracy significantly.

### 2. Leaf Disease Classification:
We implemented and evaluated 3 CNN architectures for classifying the cropped leaf image into one of 15 disease categories:
- MobileNetV2: Provides lightweight and efficient performance suitable for edge devices. At the end of epoch 10, we have a validation accuracy of 97.01% and validation loss of 0.0898, which shows that the model has learnt well and is able to classify unseen data accurately.
- DenseNet121: DenseNets offer several advantages over traditional CNNs, including improved feature propagation, reduced vanishing gradient problems, and efficient parameter usage. It is pretrained on large datasets like the ImageNet. We achieved a validation accuracy of 0.9460 and validation loss of 0.2089. Its dense connectivity promotes feature reuse and improves gradient flow, making it ideal for subtle disease variations.
- EfficientNetB0: Despite its theoretical performance, it underperformed on our dataset. It likely overfits or fails to generalize well because:
    - It uses aggressive downsampling early on, which may discard subtle visual cues needed for plant disease detection.
    - The architecture may not be optimal without fine-tuning on datasets with high intra-class similarity and background clutter.

### 3. LLM-based Diagnosis Report:
After predicting the disease class, we use a Large Language Model (LLM) to:
- Interpret the prediction
- Generate a natural language report that includes:
    - Disease name
    - Symptoms and diagnosis
    - Recommended treatment
    - Preventive measures
- We also send the LLM report to mobile phone device so that we can save the report for future use.

### Tech Stack:
- TensorFlow/Keras: for training and evaluation of CNN models
- Ultralytics YOLOv8: for object detection 
- LLM API: for generating diagnosis report
- OpenCV: for image processing

### Working:
1. YOLOv8n detects and crops the leaf portion from the images. It puts bounding box around the leaf region.
2. Cropped leaf: run MobileNetV2 or DenseNet121 classification model to get the disease prediction.
3. The prediction is then passed to the LLM to generate a user-friendly diagnostic report.

### Results:
Model	        Accuracy	Notes
DenseNet121	    94.6%	    Best overall performance
MobileNetV2	    ~92%	    Lightweight and fast
EfficientNetB0	<85%	    Skipped due to poor generalization

### Future Work:
- Integrate the system into a mobile app.
- Collect more diverse plant disease data for better generalization.
- Expand to even more crops and diseases.

### Further details:
The dataset used to train YOLOv8 model:
#https://universe.roboflow.com/yolo-0qx2l/leaf-e69nw 

Class Name	Description
Blight_a	Indicates mild leaf blight symptoms, with a few spots or yellowing on leaves—early-stage infection.
Blight_b	Represents moderate leaf blight, with increased spot density and spreading necrosis.
Blight_c	Severe leaf blight with large areas of dead tissue; overall plant health is significantly affected.
Danger_a	Mild symptoms of high-risk diseases (e.g., viral or bacterial infections), possibly early onset.
Danger_b	Severe high-risk disease presence, potentially including wilting, tissue death, or rapid plant decline.
Healthy_a	Healthy leaf with normal color and texture, but may show slight environmental stress.
Healthy_b	Very healthy leaf with vivid color, clear veins, and no visible stress or disease signs.

The results of YOLOv8 are in dict train and dict val. The dict weights keep the last model and best model with 40 epochs.


### Pipeline:
The pipeline.ipynb file containes the complete end-to-end implementation from loading DenseNet121 model to generating the LLM diagnostic report.