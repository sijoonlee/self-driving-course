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