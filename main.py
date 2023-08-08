import sys
sys.path.append(".iDISF/python3/");
import os
from idisf import iDISF_scribbles
from PIL import Image
import numpy as np

from constants import *

from simplification import get_ultimate_erosion, get_markers
from  prediction import load_model, get_prediction

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def obtain_output_pil_image(label_arr):
    output_arr = np.where(label_arr == 1, 255, label_arr)
    output_arr = np.where(output_arr == 2, 0, output_arr)
    return Image.fromarray(output_arr).convert('L')

def save_segmentation(label_arr, original_img, save_folder, filename): 
    labels_folder = save_folder + "/segmentation"
    original_folder = save_folder + "/original"

    create_folder(labels_folder)
    create_folder(original_folder)

    output_img = obtain_output_pil_image(label_arr)
    
    labels_path = f'{labels_folder}/{filename}.png'
    original_path = f'{original_folder}/{filename}.png'
    
    output_img.save(labels_path) 
    original_img.save(original_path)

def segment_cells_img(img, marker_img):
    n0 = 0
    iterations = 1
    f = 1
    c1 = 0.1
    c2 = 0.1
    segm_method = 1
    all_borders = 1
    
    # applying ultimate erosion on the prediction image
    eroded_img = get_ultimate_erosion(marker_img)

    # generating scribbles for iDISF
    num_obj, markers_coords, scribbles_sizes = get_markers(eroded_img)
    
    # returning the segmentation and the borders
    return iDISF_scribbles(
        img, n0, iterations, np.array(markers_coords), np.array(scribbles_sizes), num_obj, f, c1, c2, segm_method, all_borders)

def main():
    model = load_model(MODEL_PATH)
    for complete_filename in os.listdir(INPUT_FOLDER):
        filename, extension = os.path.splitext(complete_filename)
        input_img_path = INPUT_FOLDER + "/" + filename + extension
        
        original, activations_img = get_prediction(input_img_path, model)
        labels_arr, borders_arr = segment_cells_img(np.asarray(original), np.asarray(activations_img))

        save_segmentation(labels_arr, original, OUTPUT_FOLDER, filename)

if __name__ == "__main__":
    main()