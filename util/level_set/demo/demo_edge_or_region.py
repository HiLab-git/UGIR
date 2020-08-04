from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.ndimage.filters as filters
from skimage import measure
from ls_util.drlse_reion import *
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

def level_set_demo(method_id = 2):
    """
    param: method_id 0 -- edge-based (expand)
                     1 -- edge-based (shrink)
                     2 -- region-based (expand)
                     3 -- region-based (shrink)
    """
    img = Image.open('data/gourd.bmp')
    img = np.asarray(img, dtype='float32')

    # parameters
    timestep = 1        # time step
    iter_inner = 20
    iter_outer = 8
    mu   = 0.1/timestep   # coefficient of the distance regularization term R(phi)
    lmda = 2              # coefficient of the weighted length term L(phi)
    # coefficient of the weighted area term A(phi)
    # negative to expand, postive to shrink (for edge-based method)
    alfa = -1  
    if(method_id == 1):
        alfa = - alfa
    epsilon = 1.5         # parameter that specifies the width of the DiracDelta function

    sigma = 0.8           # scale parameter in Gaussian kernel
    img_smooth = filters.gaussian_filter(img, sigma)    # smooth image by Gaussian convolution
    [Iy, Ix] = np.gradient(img_smooth)
    f = np.square(Ix) + np.square(Iy)
    g = 1 / (1+f)        # edge indicator function.

    # initialize LSF as binary step function
    # inside is negative, outside is positive
    c0 = 2
    initialLSF = c0 * np.ones(img.shape)
    # generate the initial region R0 as two rectangles
    if(method_id == 0 or method_id == 2):
        initialLSF[25:35, 20:26] = -c0
        initialLSF[25:35, 40:50] = -c0
    else:
        initialLSF[10:50, 10:70] = -c0
    phi = initialLSF.copy()

    plt.ion()
    fig1 = plt.figure(1)
    fig2 = plt.figure(2)

    potential = 2
    if potential == 1:
        potentialFunction = 'single-well'  # use single well potential p1(s)=0.5*(s-1)^2, which is good for region-based model
    elif potential == 2:
        potentialFunction = 'double-well'  # use double-well potential in Eq. (16), which is good for both edge and region based models
    else:
        potentialFunction = 'double-well'  # default choice of potential function

    # start level set evolution
    for n in range(iter_outer):
        if(method_id < 2):
            phi = drlse_edge(phi, g, lmda, mu, alfa, epsilon, timestep, iter_inner, potentialFunction)
        else:
            phi = drlse_region(phi, img,lmda, mu, alfa, epsilon, timestep, iter_inner, potentialFunction)
        if np.mod(n, 2) == 0:
            print('show fig 2 for %i time' % n)
            fig2.clf()
            show_leve_set(fig2, phi)
            fig1.clf()
            contours = measure.find_contours(phi, 0)
            show_image_and_segmentation(fig1, img, contours)
            plt.pause(0.3)

    # refine the zero level contour by further level set evolution with alfa=0
    alfa = 0
    iter_refine = 10
    if(method_id < 2):
        phi = drlse_edge(phi, g, lmda, mu, alfa, epsilon, timestep, iter_refine, potentialFunction)
    else:
        phi = drlse_region(phi, img, lmda, mu, alfa, epsilon, timestep, iter_refine, potentialFunction)

    finalLSF = phi.copy()
    print('show final fig 2')
    fig2.clf()
    show_leve_set(fig2, phi)
    fig1.clf()
    contours = measure.find_contours(finalLSF, 0)
    show_image_and_segmentation(fig1, img, contours)
    plt.pause(10)
    plt.show()

if __name__ == "__main__":
    methods = ['edge_expand', 'edge shrink', 'region_expand', 'region_shrink']
    method_id = 0
    level_set_demo(method_id)

