import copy

import cv2
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


def process(data, per_image=True):
    """
        Scales data between 0 and 1 on a per image basis

        data (np.array) is either the test or training data
        per_image (bool) determines if data is processed on a per image basis

    """
    output = copy.deepcopy(data)
    if per_image:
        output = output.astype('float32')
        for i, image in enumerate(data):
            x, y, z = image.shape
            output[i, ...] = MinMaxScaler(feature_range=(0, 1)
                                          ).fit_transform(image.reshape([x * y, z])).reshape(
                [x, y, z])
    else:
        mi, ma = np.min(data), np.max(data)
        output = (data - mi) / (ma - mi)
        output = output.astype('float32')
    return output


def resize(data, dim):
    """
        Overloaded method for resizing input image

        data (np.array)  3D matrix containing image data (#images,X,Y,RGB/G)
        dim  (tuple) Tuple with 4 entires (#images, X, Y, RGB)

    """
    # return transform.resize(data,(data.shape[0], dim[0], dim[1], dim[2]), anti_aliasing=False)
    return tf.image.resize(data, [dim[0], dim[1]], antialias=False).numpy()


def corrupt_masks(masks, kernel_size=(5, 5)):
    """
        Corrupts segmentation maps using cv2 morphology operators, only odd filter sizes work

        Parameters
        ----------
        masks (np.array) array of rfi masks for data
        kernel_size (tuple) the kernel size for the operator

        Returns
        -------
        np.array
    """
    kernel = np.ones(kernel_size, np.uint8)
    _masks = np.empty(masks.shape, dtype=np.bool)

    for i, m in enumerate(masks):
        img = cv2.cvtColor(m[..., 0].astype(np.uint8), cv2.COLOR_GRAY2BGR)
        out = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
        _masks[i, ..., 0] = out.astype(np.bool)[..., 0]

    return _masks


def rgb2gray(rgb):
    """
        Convert rgb images to gray

        Parameters
        ----------
        rgb (np.array) array of rgb imags 

        Returns
        -------
        np.array
    """
    if rgb.shape[-1] == 3:
        return np.expand_dims(np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140]), axis=-1)
    else:
        return rgb
