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

from utils.boxes import show_mask, show_box, plot_bboxes
from dataloader.image_loader import get_files_in_dir

sys.path.append("..")
device = "cuda"
ultralytics.checks()

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

result_dir = "/home/andreas/cv_workspace/results"

file_path_list = get_files_in_dir("/home/andreas/cv_workspace/image_samples", [".jpg", ".png"])

class_ids = [0,3]


#Object Detection
model = YOLO('yolov8n.pt')
model.fuse()

images = []

for filename in tqdm(file_path_list):
    images.append(cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB))

for image in tqdm(images):    
    results = model.predict(image, classes=class_ids, conf=0.5) #inference

    #box_image = plot_bboxes(results)
    #plt.figure(figsize=(10, 10))
    #plt.imshow(box_image)


#plt.show()

if False:
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device='cpu') # Run this in CPU not enough GPU memory
    predictor = SamPredictor(sam)
    predictor.set_image(image)



    input_box = np.array(bbox)

    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,
    )


    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(masks[0], plt.gca())
    show_box(input_box, plt.gca())
    plt.axis('off')
    plt.show()

    segmentation_mask = masks[0]

    binary_mask = np.where(segmentation_mask > 0.5, 1, 0)
    white_background = np.ones_like(image) * 255
    new_image = white_background * (1 - binary_mask[..., np.newaxis]) + image * binary_mask[..., np.newaxis]
    plt.imshow(new_image.astype(np.uint8))
    plt.axis('off')
    plt.show()
