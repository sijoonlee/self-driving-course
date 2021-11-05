import argparse
from functools import partial

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

from utils import plot_batch

# https://albumentations.ai/docs/getting_started/image_augmentation/
transforms = A.Compose([A.Rotate(limit=30, p=0.5),
                        A.Blur(blur_limit=5, p=0.5)])


def aug_fn(image):
    """ augment an image """
    # https://numpy.org/doc/stable/reference/generated/numpy.squeeze.html
    # 
    aug_data = transforms(image=image.squeeze())
    # transform will return a dictionary with a single key image. Value at that key will contain an augmented image.
    aug_img = aug_data["image"]
    aug_img = tf.cast(aug_img/255.0, tf.float32) # tf.cast: Casts a tensor to a new type.
    return aug_img


def process_data(image, label):
    """ wrapper function to apply augmentation """
    # https://www.tensorflow.org/api_docs/python/tf/numpy_function
    # Given a python function func wrap this function as an operation in a TensorFlow function.
    # func must take numpy arrays as its arguments and return numpy arrays as its outputs.
    # Comparison to tf.py_function: tf.py_function and tf.numpy_function are very similar, 
    # except that tf.numpy_function takes numpy arrays, and not tf.Tensors. 
    # If you want the function to contain tf.Tensors, and have any TensorFlow operations executed in the function be differentiable, 
    # please use tf.py_function.
    aug_img = tf.numpy_function(func=aug_fn, inp=[image], Tout=tf.float32)
    return aug_img, label


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Augment dataset')
    parser.add_argument('-d', '--imdir', required=True, type=str,
                        help='data directory')
    args = parser.parse_args()    

    dataset = image_dataset_from_directory(args.imdir, 
                                           image_size=(32, 32),
                                           validation_split=0.1,
                                           subset='training',
                                           seed=123,
                                           batch_size=1)
    print(dataset.element_spec) # (TensorSpec(shape=(None, 32, 32, 3), dtype=tf.float32, name=None), TensorSpec(shape=(None,), dtype=tf.int32, name=None))
    dataset = dataset.map(process_data).batch(256)
    # https://stackoverflow.com/questions/53514495/what-does-batch-repeat-and-shuffle-do-with-tensorflow-dataset
    # The ds.batch() will take first batch_size entries and make a batch out of them. 
    # So, batch size of 3 for our example dataset will produce two batch records:
    # [1, 2, 3, 4, 5, 6] ===> [2,1,5] [3,6,4]
    for X,Y in dataset:
        batch_np = X.numpy()
        plot_batch(batch_np)
        break
