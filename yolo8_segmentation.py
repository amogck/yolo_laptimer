import ultralytics

from roboflow import Roboflow
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import sys
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from tqdm import tqdm 

from utils.boxes import show_mask, show_box, cropped_masked_image
from dataloader.image_loader import get_files_in_dir

sys.path.append("..")
device = "cuda"
ultralytics.checks()


result_dir = "/home/andreas/cv_workspace/motorcycles_persons"

file_path_list = get_files_in_dir("/home/andreas/cv_workspace/image_samples", [".jpg", ".png"])

class_ids = [0,3]


#Object Detection
model = YOLO('yolo11n-seg.pt')
model.fuse()

images = {}

for filename in tqdm(file_path_list):
    images[filename] = cv2.imread(filename)

for filename, image in tqdm(images.items()):    
    results = model.predict(image, classes=class_ids, conf=0.4) #inference

    segmented_patches = cropped_masked_image(results)
    labels = results[0].boxes.cls.tolist()

#    idx = 0
#    for patch, label in zip(segmented_patches,labels):
#        cv2.imwrite(f"{result_dir}/{filename.stem}_{idx}_{label}_patch.png", patch)
#        idx += 1
#
#    annotated_frame = results[0].plot()
#    cv2.imwrite(f"{result_dir}/{filename.stem}_annotated.png", annotated_frame)

