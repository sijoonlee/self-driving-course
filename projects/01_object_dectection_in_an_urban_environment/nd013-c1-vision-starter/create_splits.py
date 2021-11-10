import argparse
import glob
import os
import random
import shutil

from utils import get_module_logger

# useful resources
# https://stackoverflow.com/questions/46444018/meaning-of-buffer-size-in-dataset-map-dataset-prefetch-and-dataset-shuffle
# https://stackoverflow.com/questions/51125266/how-do-i-split-tensorflow-datasets/51126863
# https://stackoverflow.com/questions/54519309/split-tfrecords-file-into-many-tfrecords-files
# https://www.tensorflow.org/api_docs/python/tf/data/Iterator
# https://www.tensorflow.org/api_docs/python/tf/experimental/Optional

def split_file_list(all_processed_files, magic_number = 7):
    # Train set: 60% if magic_number = 7
    def is_train(x):
        return x[0] % 10 < magic_number

    # Validation set: 30% if magic_number = 7
    def is_val(x):
        return x[0] % 10 > magic_number

    # Test set: always 10%
    def is_test(x):
        return x[0] % 10 == magic_number

    recover = lambda x : x[1] # get rid of i from (i, file)

    all_files_enumerated = [ (i, file) for (i, file) in enumerate(all_processed_files) ]

    train_files = list(map(recover, filter(is_train, all_files_enumerated)))
    val_files = list(map(recover, filter(is_val, all_files_enumerated)))
    test_files = list(map(recover, filter(is_test, all_files_enumerated)))

    return train_files, val_files, test_files

def copy_files(new_path, files):
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    
    for file in files:
        shutil.copy(file, new_path)
    return

def split(data_dir):
    """
    Create three splits from the processed records. The files should be moved to new folders in the 
    same directory. This folder should be named train, val and test.

    args:
        - data_dir [str]: data directory, /mnt/data
    """

    path_processed_files = os.path.join(data_dir, "waymo/processed")
    all_processed_files = glob.glob(os.path.join(path_processed_files, "*.tfrecord"))
    random.shuffle(all_processed_files)
    

    path_train_files = os.path.join(data_dir, "train")
    path_val_files = os.path.join(data_dir, "val")
    path_test_files = os.path.join(data_dir, "test")

    train_files, val_files, test_files = split_file_list(all_processed_files)

    copy_files(path_train_files, train_files)
    copy_files(path_val_files, val_files)
    copy_files(path_test_files, test_files)

    return
    

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--data_dir', required=True,
                        help='data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.data_dir)