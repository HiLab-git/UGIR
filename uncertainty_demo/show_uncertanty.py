import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from PIL import Image
from PIL import ImageFilter

def add_countor(In, Seg, Color=(0, 255, 0)):
    Out = In.copy()
    [H, W] = In.size
    for i in range(H):
        for j in range(W):
            if(i==0 or i==H-1 or j==0 or j == W-1):
                if(Seg.getpixel((i,j))!=0):
                    Out.putpixel((i,j), Color)
            elif(Seg.getpixel((i,j))!=0 and  \
                 not(Seg.getpixel((i-1,j))!=0 and \
                     Seg.getpixel((i+1,j))!=0 and \
                     Seg.getpixel((i,j-1))!=0 and \
                     Seg.getpixel((i,j+1))!=0)):
                     Out.putpixel((i,j), Color)
    return Out

def gray_to_rgb(image):
    image_cat = np.asarray([image, image, image])
    image_cat = np.transpose(image_cat, [1, 2, 0])
    return image_cat


def map_scalar_to_color(x):
    x_list = [0.0, 0.25, 0.5, 0.75, 1.0]
    c_list = [[0, 0, 255],
              [0, 255, 255],
              [0, 255, 0],
              [255, 255, 0],
              [255, 0, 0]]
    for i in range(len(x_list)):
        if(x <= x_list[i + 1]):
            x0 = x_list[i]
            x1 = x_list[i + 1]
            c0 = c_list[i]
            c1 = c_list[i + 1]
            alpha = (x - x0)/(x1 - x0)
            c = [c0[j]*(1 - alpha) + c1[j] * alpha for j in range(3)]
            c = [int(item) for item in c]
            return tuple(c)

def get_attention_map(image, att):
    [H, W] = image.size
    img = Image.new('RGB', image.size, (255, 0, 0))
    
    for i in range(H):
        for j in range(W):
            p0 = image.getpixel((i,j))
            alpha = att.getpixel((i,j))
            p1 = map_scalar_to_color(alpha)
           # alpha = 0.1 + alpha*0.9
            p  = [int(p0[c] * (1 - alpha) + p1[c]*alpha) for c in range(3)]
            p = tuple(p)
            img.putpixel((i,j), p)
    return img


def show_seg_uncertainty():
    img_folder = "data"
    seg_folder = "result"
    uncertain_folder = "result"
    patient_id = "a26_12"
    slice_id   = 14
    img_name = img_folder + '/' + patient_id + ".nii.gz"
    seg_name = seg_folder + '/' + patient_id + ".nii.gz"
    uncertain_name = uncertain_folder + '/' + patient_id + "_var.nii.gz"
    img_obj = sitk.ReadImage(img_name)
    seg_obj = sitk.ReadImage(seg_name) 
    uct_obj = sitk.ReadImage(uncertain_name)
    img3d = sitk.GetArrayFromImage(img_obj)
    seg3d = sitk.GetArrayFromImage(seg_obj)
    uct3d = sitk.GetArrayFromImage(uct_obj)

    img3d = (img3d - img3d.min()) * 255.0 / (img3d.max() - img3d.min())
    img3d = np.asarray(img3d, np.uint8)
    uct3d = uct3d / uct3d.max()

    img = img3d[slice_id]
    seg = seg3d[slice_id]
    uct = uct3d[slice_id]
    img_show_raw = gray_to_rgb(img)
    img_show_raw = Image.fromarray(img_show_raw)
    seg = Image.fromarray(seg)
    img_show_seg = add_countor(img_show_raw, seg)
 
    uct = Image.fromarray(uct)
    img_show_uct = get_attention_map(img_show_raw, uct)
    fig = plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1); plt.axis('off'); plt.title('segmentation result')
    plt.imshow(img_show_seg)
    plt.subplot(1, 2, 2); plt.axis('off'); plt.title('uncertainty')
    plt.imshow(img_show_uct)
    plt.show()
    fig.savefig('./result/uncertainty.png')

if __name__ == "__main__":
    show_seg_uncertainty()
