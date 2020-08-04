
from PIL import Image
import scipy.ndimage.filters as filters
import numpy as np 
import matplotlib.pyplot as plt

img_name = 'drlse_region/a13_08_35seg_cut_edge.png'
img = Image.open(img_name).convert('L')
img = np.asarray(img, np.float32)
img = (img - img.mean())/img.std()

sigma = 1.5           # scale parameter in Gaussian kernel
img_smooth = filters.gaussian_filter(img, sigma)    # smooth image by Gaussian convolution
[Iy, Ix] = np.gradient(img_smooth)
f = np.square(Ix) + np.square(Iy)
g = 1 / (1+f)        # edge indicator function.
plt.imshow(g)
plt.show()