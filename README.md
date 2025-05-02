# 6908plant-disease

#https://drive.google.com/file/d/0B_voCy5O5sXMTFByemhpZllYREU/view?resourcekey=0-25uoBK9YYXXKnTliopPxDw The dataset is used to train the plant disease classification models. I trined EfficientNet and MobileNetV2 model

#I implemented real time inference in EfficientNet-Real-Time Detection.ipynb

#https://universe.roboflow.com/yolo-0qx2l/leaf-e69nw The dataset is used to train YOLOv8 model
Class Name	Description
Blight_a	Indicates mild leaf blight symptoms, with a few spots or yellowing on leaves—early-stage infection.
Blight_b	Represents moderate leaf blight, with increased spot density and spreading necrosis.
Blight_c	Severe leaf blight with large areas of dead tissue; overall plant health is significantly affected.
Danger_a	Mild symptoms of high-risk diseases (e.g., viral or bacterial infections), possibly early onset.
Danger_b	Severe high-risk disease presence, potentially including wilting, tissue death, or rapid plant decline.
Healthy_a	Healthy leaf with normal color and texture, but may show slight environmental stress.
Healthy_b	Very healthy leaf with vivid color, clear veins, and no visible stress or disease signs.

#The results of YOLOv8 are in dict train and dict val.
#The dict weights keep the last model and best model with 40 epoch

#For pipeline.ipynb
1 Load the YOLO model can take an image to save, it can show the label like Blight, Danger of Healthy which is a simple result.
2 Load the EfficientNet model to give an exact output like the disease name and the confience which will be save as JSON file.
3 Use the LLM to give the report from the JSON input.

I have used test.png for step2 and step3 and it work well.
