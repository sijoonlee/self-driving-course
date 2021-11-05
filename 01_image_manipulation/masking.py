import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def create_mask(path, color_threshold):
    """
    create a binary mask of an image using a color threshold
    args:
    - path [str]: path to image file
    - color_threshold [array]: 1x3 array of RGB value
    returns:
    - img [array]: RGB image array
    - mask [array]: binary array
    """
    img = np.array(Image.open(path).convert('RGB'))
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    rt, gt, bt = color_threshold
    mask = (R > rt) & (G > gt) & (B > bt) 
    return img, mask


def mask_and_display(img, mask):
    """
    display 3 plots next to each other: image, mask and masked image
    args:
    - img [array]: HxWxC image array ( 1280 * 1090 * 3 )
    - mask [array]: HxW mask array
    """
    # numpy.stack(arrays, axis=0, out=None)
    #   Join a sequence of arrays along a new axis.
    #   The axis parameter specifies the index of the new axis in the dimensions of the result. 
    #   For example, if axis=0 it will be the first dimension and if axis=-1 it will be the last dimension.
    test = [[True, False]]
    print([test] * 3)
    # [[[True, False]], [[True, False]], [[True, False]]]
    print(np.stack([test]*3, axis=0))
    print(np.stack([test]*3, axis=0).shape) # 3, 1, 2
    # [[[True False]]
    #  [[True False]]
    #  [[True False]]]
    print(np.stack([test]*3, axis=1))
    print(np.stack([test]*3, axis=1).shape) # 1, 3, 2
    # [[[True False]
    #   [True False]
    #   [True False]]]
    print(np.stack([test]*3, axis=2))
    print(np.stack([test]*3, axis=2).shape) # 1 ,2, 3
    # [[[True True True]
    #   [False False False]]]
    masked_image = img * np.stack([mask]*3, axis=2)
    f, ax = plt.subplots(1, 3, figsize=(15, 10))
    ax[0].imshow(img)
    ax[1].imshow(mask)
    ax[2].imshow(masked_image)
    plt.show()


if __name__ == '__main__':
    path = '/home/sijoonlee/Documents/self-driving-course/01_image_manipulation/data/images/segment-1231623110026745648_480_000_500_000_with_camera_labels_38.png'
    color_threshold = [128, 128, 128]
    img, mask = create_mask(path, color_threshold)
    mask_and_display(img, mask)