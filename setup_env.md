## update pip
```python3 -m pip install --upgrade pip```

## virtual env setup
- ```sudo apt install -y python3-venv```
- ```mkdir environment```
- ```pytho3 -m venv course_env```
- ```source environment/course_env/bin/activate``` at project root
- after done, ```deactivate```

## Pillow installation
- ```python3 -m pip install --upgrade pip```
- ```python3 -m pip install --upgrade Pillow```

## Matplotlib installation
- ```python -m pip install -U matplotlib```

## Numpy installation
- ```pip3 install numpy```

## Seaborn installation
- ```pip3 install seaborn```

## tk installation (for matplotlib GUI)
- ```sudo apt-get install python3-tk```

## Tensorflow
- ```pip3 install tensorflow``` # 2.6.0 was installed

## Albumentations
- ```pip3 install -U albumentations```

## Ray
- ```pip install -U ray``` # minimal install
- https://docs.ray.io/en/latest/installation.html

## Waymo-data-set
- ```pip3 install waymo-open-dataset-tf-2-6-0 --user```
- ```pip3 install waymo-open-dataset-tf-2-6-0``` # at virtual env
- https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/quick_start.md

## Install Protobuf
- https://askubuntu.com/questions/1072683/how-can-i-install-protoc-on-ubuntu-16-04
- https://github.com/protocolbuffers/protobuf/blob/master/src/README.md
- Prerequesites  
```sudo apt-get install autoconf automake libtool curl make g++ unzip```
- Installation
- From this page(https://github.com/protocolbuffers/protobuf/releases), download the protobuf-all-[VERSION].tar.gz.
Extract the contents and change in the directory
```
./configure
make
make check
sudo make install
sudo ldconfig # refresh shared library cache.
```
- Check if it works  
```
$ protoc --version
libprotoc 3.6.1
```

## Python protobuf
- ```pip3 install protobuf```

## Object detection API with Tensorflow 2
- https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md

```
git clone https://github.com/tensorflow/models.git

cd models/research

# Compile protos.
protoc object_detection/protos/*.proto --python_out=.

# Install TensorFlow Object Detection API.
cp object_detection/packages/tf2/setup.py .
python -m pip install --use-feature=2020-resolver .

# Test the installation.
python object_detection/builders/model_builder_tf2_test.py
```

## tf_slim
- https://github.com/google-research/tf-slim
- ```pip install --upgrade tf_slim```

## tensorflow.io
- ```pip install tensorflow.io```

## Jupyter notebook
- https://jupyter.org/install
- install
```
pip3 install jupyterlab
export PATH="$HOME/.local/bin:$PATH"
pip3 install notebook

```
- run jupyter lab
```
jupyter-lab
```
- run jupyter notebook
```
jupyter notebook
```