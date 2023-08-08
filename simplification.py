from __future__ import print_function
import cv2
import numpy as np
import os
from scipy import ndimage
from skimage import measure
from skimage.morphology import erosion, disk, dilation, erosion, reconstruction

# -----------------------------------Generating Markers-----------------------------------------
def get_markers(src_marker):
    src_marker = cv2.cvtColor(src_marker, cv2.COLOR_GRAY2RGB)
    markers = []
    num_obj = 0
    scribbles_sizes = []
    getBackground(src_marker)

    image, number_of_objects = ndimage.label(src_marker[:,:,2])
    blobs = ndimage.find_objects(image)
    for i, j in enumerate(blobs):
        marker = []
        for y in range(j[0].start,j[0].stop):
            for x in range(j[1].start,j[1].stop):
                if(image[y,x] != 0):
                    marker.append([x,y])
    
        scribbles_sizes.insert(0, len(marker))
        markers = marker + markers
        num_obj += 1

    image, number_of_objects = ndimage.label(src_marker[:,:,0])
    blobs = ndimage.find_objects(image)
    
    for i,j in enumerate(blobs):
      marker = []
      for y in range(j[0].start,j[0].stop):
        for x in range(j[1].start,j[1].stop):
          if(image[y,x] != 0):
            marker.append([x,y])
      
      scribbles_sizes.append(len(marker))
      markers = markers + marker    
    return num_obj, markers, scribbles_sizes

def getBackground(src_marker):
    for i in range(src_marker.shape[0]):
        for j in range(src_marker.shape[1]):
            if(i == 0 or i == (src_marker.shape[0]-1) or j == 0 or j == (src_marker.shape[1]-1)):
                if(src_marker[i,j,0] != 255):
                    src_marker[i,j,0] = 127
                else:
                    src_marker[i,j,0] = 0
            else:
                src_marker[i,j,0] = 0

def saveScribbles(filename, src_marker, output_dir):
    num_obj, markers, scribbles_sizes = get_markers(src_marker)
    if(len(markers) == 0): 
        return
    
    f = open(output_dir+"/"+os.path.splitext(filename)[0]+".txt", 'w')
    f.write("%d\n"%(len(scribbles_sizes)))
    f.write("%d\n"%(scribbles_sizes[0]))

    index_sizes=0
    acum=0

    for i in range(len(markers)-1):
        if(acum == scribbles_sizes[index_sizes]):
            index_sizes+=1
            acum=0
            f.write("%d\n"%(scribbles_sizes[index_sizes]))
        
        [x,y] = markers[i]
        f.write("%d;%d\n"%(x,y))
        acum+=1

    if(acum == scribbles_sizes[index_sizes]):
        index_sizes+=1
        acum=0
        f.write("%d\n"%(scribbles_sizes[index_sizes]))
    
    [x,y] = markers[-1]
    f.write("%d;%d\n"%(x,y))
    f.write("%d"%(num_obj))
    f.close()
    return num_obj, markers, scribbles_sizes

# --------------------------------------------------------------------------------------------
def ultimate_erosion(input_img, r):
    nb_iter = 0
    mask = generate_erosion_mask(input_img)
    img = mask
    img_niter = np.zeros_like(mask, dtype='uint16') 
    while(np.max(img) > 0):
        img_ero = erosion(img, disk(r))
        nb_iter = nb_iter+1
        reconst= reconstruction(img_ero, img,'dilation')
        residues = img - reconst
        img_niter = np.where(residues==1, nb_iter, img_niter)
        img = img_ero

    # residues relabel
    img_residue = img_niter
    img_residue[img_residue>0] = 1
    img_residue = dilation(img_residue, disk(3))
    img_residue = measure.label(img_residue, connectivity=2, background=0)
    img_residue = erosion(img_residue, disk(3))

    # reconstruction
    img_rc = np.zeros_like(img_residue, dtype='uint16') 
    i = np.max(img_niter)

    while i > 1:
        img_rc = np.where(img_niter == i, img, img_rc) 
        img_rc = dilation(img_rc, disk(r))
        i = i-1

    img_rc = np.where(img_niter == i, img_residue, img_rc) 
    return img_rc

def generate_erosion_mask(input_img):
    mask = np.zeros_like(input_img[..., 0], dtype='uint16') 
    mask = np.where((input_img[..., 0] > 0), 1, mask)
    mask = mask.astype(np.uint16)
    return mask

def get_ultimate_erosion(img):
    _, thresh_img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    return ultimate_erosion(thresh_img, 1).astype('uint8')