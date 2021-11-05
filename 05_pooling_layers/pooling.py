import argparse

import numpy as np

from utils import check_output


def get_paddings(array, pool_size, pool_stride):
    """ 
    get padding sizes 
    args:
    - array [array]: input np array NxwxHxC
    - pool_size [int]: window size
    - pool_stride [int]: stride
    returns:
    - paddings [list[list]]: paddings in np.pad format
    """
    # IMPLEMENT THIS FUNCTION
    # the floor division // rounds the result down to the nearest whole number ex) 15 // 2 => 7
    _, w, h, _ = array.shape
    wpad = (w // pool_stride) * pool_stride + pool_size - w
    hpad = (h // pool_stride) * pool_stride + pool_size - h
    return [[0, 0], [0, wpad], [0, hpad], [0, 0]]


def get_output_size(shape, pool_size, pool_stride):
    """ 
    given input shape, pooling window and stride, output shape 
    args:
    - shape [list]: padded input shape
    - pool_size [int]: window size
    - pool_stride [int]: stride
    returns
    - output_shape [list]: output array shape

    cf) output size = ( original_input_size - pool_size + 2 * padding_size ) / stride + 1
    """
    # IMPLEMENT THIS FUNCTION
    w = shape[1]
    h = shape[2]
    new_w = int( (w - pool_size) / pool_stride + 1 )
    new_h = int( (h - pool_size) / pool_stride + 1 )
    return [shape[0], new_w, new_h, shape[3]]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download and process tf files')
    parser.add_argument('-f', '--pool_size', required=True, type=int, default=3,
                        help='pool filter size')
    parser.add_argument('-s', '--stride', required=True, type=int, default=3,
                        help='stride size')
    args = parser.parse_args()

    input_array = np.random.rand(1, 224, 224, 16)
    pool_size = args.pool_size
    pool_stride = args.stride

    # padd the input layer
    paddings = get_paddings(input_array, pool_size, pool_stride)
    padded = np.pad(input_array, paddings, mode='constant', constant_values=0)
    # >>> a = np.array([[ 1.,  1.,  1.,  1.,  1.],
    # ...               [ 1.,  1.,  1.,  1.,  1.],
    # ...               [ 1.,  1.,  1.,  1.,  1.]])
    # >>> np.pad(a, [(0, 1), (0, 1)], mode='constant')
    # [(0, 1), (0, 1)]
    #          ^^^^^^------ padding for second dimension
    #  ^^^^^^-------------- padding for first dimension
    #   ^------------------ no padding at the beginning of the first axis
    #      ^--------------- pad with one "value" at the end of the first axis.
    # array([[ 1.,  1.,  1.,  1.,  1.,  0.],
    #        [ 1.,  1.,  1.,  1.,  1.,  0.],
    #        [ 1.,  1.,  1.,  1.,  1.,  0.],
    #        [ 0.,  0.,  0.,  0.,  0.,  0.]])

    # get output size
    output_size = get_output_size(padded.shape, pool_size, pool_stride)
    output = np.zeros(output_size)

    # IMPLEMENT THE POOLING CALCULATION 
    check_output(output)