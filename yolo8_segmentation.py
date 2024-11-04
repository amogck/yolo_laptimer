import ultralytics
from keras.applications.vgg16 import VGG16 
from keras.models import Model
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from pprint import pprint
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import sys
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm 


from keras.preprocessing.image import load_img 

from utils.boxes import show_mask, show_box, cropped_masked_image
from dataloader.image_loader import get_files_by_type_in_dir, get_files_by_pattern_in_dir, extract_features

sys.path.append("..")
device = "cuda"
ultralytics.checks()

dataset = Path("/home/andreas/cv_workspace/datasets/motorcycle_w_rider")

image_dir = Path.joinpath(dataset, "image_samples")
result_dir = Path.joinpath(dataset, "results")


file_path_list = get_files_by_type_in_dir(image_dir, [".jpg", ".png"])

class_ids = [0,3]


#Object Detection
model = YOLO('/home/andreas/cv_workspace/yolo11n-seg.pt')
model.fuse()

images = {}

for filename in tqdm(file_path_list):
    images[filename] = cv2.imread(filename)

# Run instance segmentation and save results annotated and segmented patches
for filename, image in tqdm(images.items()):    
    results = model.predict(image, classes=class_ids, conf=0.4) #inference

    segmented_patches = cropped_masked_image(results)
    labels = results[0].boxes.cls.tolist()

    idx = 0
    for patch, label in zip(segmented_patches,labels):
        cv2.imwrite(f"{result_dir}/{filename.stem}_{idx}_{model.names[label]}_patch.png", patch)
        idx += 1

    annotated_frame = results[0].plot()
    cv2.imwrite(f"{result_dir}/{filename.stem}_annotated.png", annotated_frame)

# Load segmentated patches and try clustering

motorcycle_patches = get_files_by_pattern_in_dir(result_dir, ["motorcycle"])

print(motorcycle_patches)

model = VGG16()
model = Model(inputs = model.inputs, outputs = model.layers[-2].output)

motorcycle_patches_vgg_features = {}

for patch_file in motorcycle_patches:
    feat = extract_features(patch_file,model)
    motorcycle_patches_vgg_features[patch_file] = feat

# get a list of the filenames
filenames = np.array(list(motorcycle_patches_vgg_features.keys()))

feat = np.array(list(motorcycle_patches_vgg_features.values())).reshape(-1,4096)

# reduce the amount of dimensions in the feature vector
pca = PCA(n_components=16, random_state=22)
pca.fit(feat)
x = pca.transform(feat)

# cluster feature vectors
kmeans = KMeans(n_clusters=3, random_state=22)
kmeans.fit(feat)

# holds the cluster id and the images { id: [images] }
groups = {}
for file, cluster in zip(filenames,kmeans.labels_):
    if cluster not in groups.keys():
        groups[cluster] = []
        groups[cluster].append(file)
    else:
        groups[cluster].append(file)

pprint(groups)

# function that lets you view a cluster (based on identifier)        
def view_cluster(groups, cluster):
    plt.figure(figsize = (25,25))
    # gets the list of filenames for a cluster
    files = groups[cluster]
    # only allow up to 30 images to be shown at a time
    if len(files) > 30:
        print(f"Clipping cluster size from {len(files)} to 30")
        files = files[:29]
    # plot each image in the cluster
    for index, file in enumerate(files):
        plt.subplot(10,10,index+1)
        img = load_img(file)
        img = np.array(img)
        plt.imshow(img)
        plt.axis('off')
        
for cluster in groups:
    view_cluster(groups, cluster)

plt.show()
'''
# this is just incase you want to see which value for k might be the best 
sse = []
list_k = list(range(3, 50))

for k in list_k:
    km = KMeans(n_clusters=k, random_state=22)
    km.fit(x)
    
    sse.append(km.inertia_)

# Plot sse against k
plt.figure(figsize=(6, 6))
plt.plot(list_k, sse)
plt.xlabel(r'Number of clusters *k*')
plt.ylabel('Sum of squared distance')
'''