## TFRecord file
- https://www.tensorflow.org/tutorials/load_data/tfrecord

### TFRecordDataset
- Using TFRecordDatasets can be useful for standardizing input data and optimizing performance.
```
filenames = [filename]
raw_dataset = tf.data.TFRecordDataset(filenames) # <TFRecordDatasetV2 shapes: (), types: tf.string>

for raw_record in raw_dataset.take(10):
  print(repr(raw_record))

```

## Matplotlib

### pyplot
- https://matplotlib.org/stable/tutorials/introductory/pyplot.html
- https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html
```
# using the variable ax for single a Axes
fig, ax = plt.subplots()

# using the variable axs for multiple Axes
fig, axs = plt.subplots(2, 2)

# using tuple unpacking for multiple Axes
fig, (ax1, ax2) = plt.subplots(1, 2)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
```