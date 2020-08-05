import time
import os
import numpy as np
import geodesic_distance
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filters
from PIL import Image
from scipy import ndimage
from skimage import measure
from mpl_toolkits.mplot3d import Axes3D
from level_set.ls_util.interactive_ls import *

def refine_dlls(image_name, seg_name, seed_name, display = True, intensity = False):
    # read images as gray cale, and normalize the input image
    img  = Image.open(image_name).convert('L')
    seg  = Image.open(seg_name).convert('L')
    seg  = np.asarray(seg, np.float32)/255.0
    seed = Image.open(seed_name).convert('L')
    seed = np.asarray(seed)
    seed_f = seed == 127
    seed_b = seed == 255

    params = {}
    params['mu'] = 0.003
    params['lambda'] = 0.3
    params['alpha']  = 0.1
    params['beta']   = 0.5
    new_seg, runtime = interactive_level_set(img, seg, seed_f, seed_b, params, display, intensity)
    
    return new_seg, runtime

def get_result_for_one_case():
    data_root = 'data/'
    img_name  = 'a03_04_11' #'a10_12_22' 
    img_full_name  = data_root + "{0:}img.png".format(img_name)
    seg_full_name  = data_root + "{0:}seg.png".format(img_name)
    scrb_full_name = data_root + "{0:}scrb.png".format(img_name)
    refine_dlls(img_full_name, seg_full_name, scrb_full_name, intensity = False)

if __name__ == "__main__":
    get_result_for_one_case()