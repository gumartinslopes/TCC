
import deeptrack as dt
import numpy as np
import cv2
import os
from PIL import Image

def get_activation_image(weights):
    squeezed = weights.squeeze(0)
    _, binarized = cv2.threshold(squeezed, .5, 255, cv2.THRESH_BINARY)
    activation_image = Image.fromarray(binarized).convert('L')
    return activation_image

def crop_image(gt, crop_size):
    left = 0
    top = 0
    right =  gt.size[0] - crop_size
    bottom =  gt.size[1] - crop_size
    cropped = gt.crop((left, top, right, bottom))
    return cropped

def save_results(original_imgs, activations, save_dir):   
    original_save_path = save_dir + "/original"
    activation_save_path = save_dir + "/activations"

    if(not os.path.exists(original_save_path)):
        os.makedirs(original_save_path)
    
    if(not os.path.exists(activation_save_path)):
        os.makedirs(activation_save_path)
    for i in range(len(original_imgs)):
            original_imgs[i].save(f'{original_save_path}/{i+1}.jpg')
            activations[i].save(f'{activation_save_path}/{i+1}.jpg')
            
def get_prediction(image_path, model, alpha = 0.5, cutoff = 0.999):
    # reading the images
    input_img = dt.LoadImage(image_path)() / 256
    original_img = Image.open(image_path)

    # # cropping the image
    # input_img = input_img[:-400, :-400]
    # original_img = crop_image(original_img, 400)

    # obtaining the activation
    pred, weights = model.predict(input_img[np.newaxis])
    activation_img = get_activation_image(weights)
    
    activation_img = activation_img.resize(original_img.size).convert('RGB')
    return original_img, activation_img

def load_model(checkpoint_path):
    loaded = dt.models.LodeSTAR(input_shape=(None, None, 3))
    loaded.build(input_shape=())
    loaded.load_weights(checkpoint_path)
    return loaded

def calculate_iou(image1, image2):
    intersection = np.logical_and(image1, image2)
    union = np.logical_or(image1, image2)
    iou = np.sum(intersection) / np.sum(union)
    return iou