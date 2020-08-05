import time
import os
import GeodisTK
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filters
from PIL import Image
from scipy import ndimage
from skimage import measure
from mpl_toolkits.mplot3d import Axes3D
from level_set.ls_util.drlse_reion import *

def show_leve_set(fig, phi):
    ax1 = fig.add_subplot(111, projection='3d')
    y, x = phi.shape
    x = np.arange(0, x, 1)
    y = np.arange(0, y, 1)
    X, Y = np.meshgrid(x, y)
    ax1.plot_surface(X, Y, phi, rstride=2, cstride=2, color='r', linewidth=0, alpha=0.6, antialiased=True)
    ax1.contour(X, Y, phi, 0, colors='g', linewidths=2)

def show_image_and_segmentation(fig, img, contours, seeds = None):
    ax2 = fig.add_subplot(111)
    ax2.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
    for n, contour in enumerate(contours):
        ax2.plot(contour[:, 1], contour[:, 0], linewidth=2, color='green')
    if(seeds is not None):
        h_idx, w_idx = np.where(seeds[0] > 0)
        ax2.plot(w_idx, h_idx, linewidth=2, color='red')
        h_idx, w_idx = np.where(seeds[1] > 0)
        ax2.plot(w_idx, h_idx, linewidth=2, color='blue')
    ax2.axis('off')

def get_distance_based_likelihood(img, seed, D):
    if(seed.sum() > 0):
        geoD = GeodisTK.geodesic2d_raster_scan(img, seed, 0.1, 2)
        geoD[geoD > D] = D
    else:
        geoD = np.ones_like(img)*D
    geoD = np.exp(-geoD)
    return geoD

def interactive_level_set(img, seg, seed_f, seed_b, param, display = True, intensity = False):
    """
    Refine an initial segmentation with interaction based level set
    Params:
        img: a 2D image array
        sed: a 2D image array representing the intial binary segmentation
        seed_f: a binary array representing the existence of foreground scribbles
        seed_b: a binary array representing the existence of background scribbles
        display: a bool value, whether display the segmentation result
        intensity: a bool value, whether define the region term based on intensity
    """
    img = np.asarray(img, np.float32)
    img = (img - img.mean())/img.std()
    seg = np.asarray(seg, np.float32)
    Df = get_distance_based_likelihood(img, seed_f, 4)
    Db = get_distance_based_likelihood(img, seed_b, 4)

    Pfexp = np.exp(Df); Pbexp = np.exp(Db)
    Pf = Pfexp / (Pfexp + Pbexp)
    # if(display):
    #   plt.subplot(1,3,1)
    #   plt.imshow(Df)
    #   plt.subplot(1,3,2)
    #   plt.imshow(Db)
    #   plt.subplot(1,3,3)
    #   plt.imshow(Pf)
    #   plt.show()
    
    [H, D] = img.shape
    zoom = [64.0/H, 64.0/D]
    img_d = ndimage.interpolation.zoom(img, zoom)
    seg_d = ndimage.interpolation.zoom(seg, zoom)
    Pf_d  = ndimage.interpolation.zoom(Pf,  zoom)
    if(intensity is True):
        print("use intensity")
        ls_img = img_d
    else:
        print("use segmentation")
        ls_img = seg_d

    # parameters
    timestep = 1           # time step
    iter_inner = 50
    iter_outer_max = 10
    mu   = param['mu']/timestep   # coefficient of the distance regularization term R(phi)
    lmda = param['lambda']         # coefficient of the weighted length term L(phi)
    alfa = param['alpha']           # coefficient of the weighted area term A(phi)
    beta = param['beta']             # coefficient for user interactin term
    epsilon = 1.5          # parameter that specifies the width of the DiracDelta function
    # initialize LSF as binary step function
    # the level set has positive value inside the contour and negative value outside
    # this is opposite to DRLSE
    c0 = 20
    initialLSF = -c0 * np.ones(seg_d.shape)
    initialLSF[seg_d > 0.5] = c0
    phi = initialLSF.copy()
    
    t0 = time.time()
    # start level set evolution
    seg_size0 = np.asarray(phi > 0).sum()
    for n in range(iter_outer_max):
        phi = drlse_region_interaction(phi, ls_img, Pf_d, lmda, mu, alfa, beta, epsilon, timestep, iter_inner, 'double-well')
        seg_size = np.asarray(phi > 0).sum()
        ratio = (seg_size - seg_size0)/float(seg_size0)
        if(abs(ratio) < 1e-3):
            print('iteration', n*iter_inner, ratio)
            break
        else:
            seg_size0 = seg_size
    runtime = time.time() - t0
    print('iteration', (n + 1)*iter_inner)
    print('running time', runtime)
   

    finalLSF = phi.copy()
    finalLSF = ndimage.interpolation.zoom(finalLSF,  [1.0/item for item in zoom])
    if(display):
        plt.ion()
        fig1 = plt.figure(1)
        fig2 = plt.figure(2)
        fig3 = plt.figure(3)    

        fig1.clf()
        init_contours =  measure.find_contours(seg, 0.5)
        show_image_and_segmentation(fig1, img, init_contours, [seed_f, seed_b])
        fig1.suptitle("(a) Initial Segmentation")
        # fig1.savefig("init_seg.png")

        fig2.clf()
        final_contours = measure.find_contours(finalLSF, 0)
        show_image_and_segmentation(fig2, img, final_contours)
        fig2.suptitle("(b) Refined Result")
        # fig2.savefig("refine_seg.png")

        fig3.clf()
        show_leve_set(fig3, finalLSF)
        fig3.suptitle("(c) Final Level Set Function")
        # fig3.savefig("levelset_func.png")
        plt.pause(10)
        plt.show()
    return finalLSF > 0, runtime

