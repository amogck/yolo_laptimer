from typing import List
from pathlib import Path


import numpy as np
from keras.preprocessing.image import load_img 
from keras.preprocessing.image import img_to_array 
from keras.applications.vgg16 import preprocess_input 

def get_files_by_type_in_dir(dir_path: str, file_type: List[str]) -> List[Path]:

    return[f for f in Path(dir_path).iterdir() if f.is_file() and f.suffix in file_type]


def get_files_by_pattern_in_dir(dir_path, patterns: List[str]) -> List[Path]:

    return[f for f in Path(dir_path).iterdir() if f.is_file() and any(pattern in f.stem for pattern in patterns)]

def extract_features(file, model):
    # load the image as a 224x224 array
    img = load_img(file, target_size=(224,224))
    # convert from 'PIL.Image.Image' to numpy array
    img = np.array(img) 
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
    reshaped_img = img.reshape(1,224,224,3) 
    # prepare image for model
    imgx = preprocess_input(reshaped_img)
    # get the feature vector
    features = model.predict(imgx, use_multiprocessing=True)
    return features