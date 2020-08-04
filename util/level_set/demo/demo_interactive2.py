import time
import GeodisTK
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.ndimage.filters as filters
from skimage import measure
from  drlse_reion import *
import numpy as np

def show_leve_set(fig, phi):
    ax1 = fig.add_subplot(111, projection='3d')
    y, x = phi.shape
    x = np.arange(0, x, 1)
    y = np.arange(0, y, 1)
    X, Y = np.meshgrid(x, y)
    ax1.plot_surface(X, Y, phi, rstride=2, cstride=2, color='r', linewidth=0, alpha=0.6, antialiased=True)
    ax1.contour(X, Y, phi, 0, colors='g', linewidths=2)

def show_image_and_segmentation(fig, img, contours):
    ax2 = fig.add_subplot(111)
    ax2.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
    for n, contour in enumerate(contours):
        ax2.plot(contour[:, 1], contour[:, 0], linewidth=2)

def interactive_level_set(image_name, seg_name, seed_name):
    img0 = Image.open(image_name).convert('L')
    seg  = Image.open(seg_name).convert('L')
    seed = Image.open(seed_name).convert('L')
    img = np.asarray(img0, np.float32)
    img = (img - img.mean())/img.std()
    seg = np.asarray(seg, np.float32)

    seed = np.zeros(seg.shape, np.uint8)
    seed[100, 115] = 1
    seed[80, 148]  = 1
    seed[310, 112] = 1
    seed_d = GeodisTK.geodesic2d_raster_scan(img, seed, 0.0, 2)
    seed_d[seed_d > 80] = 80
    seed_d = seed_d/80.0
    g = seed_d
    plt.imshow(g)
    plt.show()
    # seed = np.asarray(seed)
    # seed_f = seed == 255
    # seed_b = seed == 128
    # print(seed_f.sum(), seed_b.sum())
    # Df = geodesic_distance.geodesic2d_raster_scan(img, seed_f, 1.0, 2)
    # Db = geodesic_distance.geodesic2d_raster_scan(img, seed_b, 1.0, 2)
    # Pf = np.exp(-Df); Pb = np.exp(-Db)
    # Pf = Pf * (Pf > 0.7)
    # Pb = Pb * (Pb > 0.4)
    # Pfexp = np.exp(Pf); Pbexp = np.exp(Pb)
    # Pf = Pfexp / (Pfexp + Pbexp)

    img = ndimage.interpolation.zoom(img, 0.5)
    seg = ndimage.interpolation.zoom(seg, 0.5)
    g = ndimage.interpolation.zoom(g, 0.5)
    # Pf  = ndimage.interpolation.zoom(Pf, 0.5)


    # parameters
    timestep = 1        # time step
    iter_inner = 10
    iter_outer = 10
    mu   = 0.1/timestep   # coefficient of the distance regularization term R(phi)
    lmda = 50               # coefficient of the weighted length term L(phi)
    alfa = 1               # coefficient of the weighted area term A(phi)
    beta = 1
    epsilon = 1.5          # parameter that specifies the width of the DiracDelta function

    # initialize LSF as binary step function
    # the level set has positive value inside the contour and negative value outside
    # this is opposite to DRLSE
    c0 = 20
    initialLSF = -c0 * np.ones(seg.shape)
    initialLSF[seg > 127] = c0
    phi = initialLSF.copy()
    plt.ion()

    fig1 = plt.figure(1)
    fig2 = plt.figure(2)
    fig3 = plt.figure(3)

    potential = 2
    if potential == 1:
        potentialFunction = 'single-well'  # use single well potential p1(s)=0.5*(s-1)^2, which is good for region-based model
    elif potential == 2:
        potentialFunction = 'double-well'  # use double-well potential in Eq. (16), which is good for both edge and region based models
    else:
        potentialFunction = 'double-well'  # default choice of potential function

    t0 = time.time()
    # start level set evolution
    for n in range(iter_outer):
        # phi = drlse_edge(phi, g, lmda, mu, alfa, epsilon, timestep, iter_inner, potentialFunction)
        # phi = drlse_region(phi, seg, lmda, mu, alfa, epsilon, timestep, iter_inner, potentialFunction)
        # phi = drlse_region_interaction2(phi, seg, Pf, lmda, mu, alfa, beta, epsilon, timestep, iter_inner, potentialFunction)
        phi = drlse_region_interaction2(phi, seg, g,lmda, mu, alfa,
        epsilon, timestep, iter_inner, potentialFunction)
    runtime = time.time() - t0
    print('running time', runtime)
    # # refine the zero level contour by further level set evolution with alfa=0
    # alfa = 0
    # iter_refine = 10
    # # phi = drlse_edge(phi, g, lmda, mu, alfa, epsilon, timestep, iter_refine, potentialFunction)
    # phi = drlse_region(phi, img,lmda, mu, alfa, epsilon, timestep, iter_refine, potentialFunction)

    finalLSF = phi.copy()
    print('show final fig 2')
    fig1.clf()
    init_contours =  measure.find_contours(seg, 128)
    show_image_and_segmentation(fig1, img, init_contours)

    fig2.clf()
    final_contours = measure.find_contours(finalLSF, 0)
    show_image_and_segmentation(fig2, img, final_contours)

    fig3.clf()
    show_leve_set(fig3, finalLSF)
    plt.pause(10)
    plt.show()

if __name__ == "__main__":
    # img_name = 'drlse_region/a02_02_41img.png'
    # seg_name = 'drlse_region/a02_02_41seg.png'
    # seed_name = 'drlse_region/a02_02_41seed.png'
    img_name = 'drlse_region/a13_08_35img_cut.png'
    seg_name = 'drlse_region/a13_08_35seg_cut.png'
    seed_name = 'drlse_region/a13_08_35seed_cut.png'
    interactive_level_set(img_name, seg_name, seed_name)
