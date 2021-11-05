import glob

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image, ImageStat

from utils import check_results


def calculate_mean_std(image_list):
    """
    calculate mean and std of image list
    args:
    - image_list [list[str]]: list of image paths
    returns:
    - mean [array]: 1x3 array of float, channel wise mean
    - std [array]: 1x3 array of float, channel wise std
    """
    means = []
    stds = []
    for path in image_list:
        img = Image.open(path).convert('RGB')
        stat = ImageStat.Stat(img)
        means.append(np.array(stat.mean))
        stds.append(np.array(stat.var)**0.5) # square root
    
    total_mean = np.mean(means, axis=0)
    total_std = np.mean(stds, axis=0)

    return total_mean, total_std


def channel_histogram(image_list):
    """
    calculate channel wise pixel value
    args:
    - image_list [list[str]]: list of image paths
    """
    red = []
    green = []
    blue = []
    for path in image_list:
        img = np.array(Image.open(path).convert('RGB'))
        R, G, B = img[..., 0], img[..., 1], img[..., 2]
        # Ellipsis is an object that can appear in slice notation. For example: myList[1:2, ..., 0]
        # As of Python 3.5 and PEP484, the literal ellipsis is used to denote certain types to a static type checker when using the typing module.
        # 1) Arbitrary-length homogeneous tuples can be expressed using one type and ellipsis, for example Tuple[int, ...]
        # 2) It is possible to declare the return type of a callable without specifying the call signature by substituting a literal ellipsis (three dots) for the list of arguments:
        #    def partial(func: Callable[..., str], *args) -> Callable[..., str]:
        #        Some Body
        red.extend(R.flatten().tolist()) # Extend the list by appending all the items from the iterable. Equivalent to a[len(a):] = iterable.
        green.extend(G.flatten().tolist())
        blue.extend(B.flatten().tolist())
    
    plt.figure()
    sns.kdeplot(red, color='r')
    sns.kdeplot(green, color='g')
    sns.kdeplot(blue, color='b')
    plt.show()


if __name__ == "__main__": 
    image_list = glob.glob('./data/images/*')
    # print(len(image_list))
    # print(*image_list)
    mean, std = calculate_mean_std(image_list)
    channel_histogram(image_list[:2])
    check_results(mean, std)