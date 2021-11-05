# Exercise 3 - Augmentations

## Objective

In this exercise, you will experiment with the [Albumentations](https://albumentations.ai/docs/) library
to perform different data augmentations. 

## Details

Write down a list of relevant augmentations and store them in the `transforms` variable. You should also
implement a quick script to visualize the batches and check your augmentations.

You can run `python augmentations.py` to display augmented images (in the Desktop window).

## Tips

You should use the `Compose` API to use multiple augmentations. You can find an example of an augmentation
pipeline using `Compose` [here](https://albumentations.ai/docs/examples/example/#define-an-augmentation-pipeline-using-compose-pass-the-image-to-it-and-receive-the-augmented-image).

This [Github repository](https://github.com/albumentations-team/albumentations_examples)
contains different examples of augmentations.

## Using Keras
## Option 1 - Keras Preprocessing layers
https://www.tensorflow.org/tutorials/images/data_augmentation
```
IMG_SIZE = 180

resize_and_rescale = tf.keras.Sequential([
  layers.Resizing(IMG_SIZE, IMG_SIZE),
  layers.Rescaling(1./255)
])

data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
])

model = tf.keras.Sequential([
  # Add the preprocessing layers you created earlier.
  resize_and_rescale,
  data_augmentation,
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  # Rest of your model.
])
```
## Option 2 - Apply preprocessing to the dataset
```
aug_ds = train_ds.map(
  lambda x, y: (resize_and_rescale(x, training=True), y))
```