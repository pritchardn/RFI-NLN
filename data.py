import copy
import os
import numpy as np
import tensorflow as tf

from model_config import BUFFER_SIZE, BATCH_SIZE
from utils import args
from utils.data import (get_lofar_data,
                        get_hera_data,
                        process,
                        get_patches)
from utils.flagging import flag_data


def load_hera(args):
    """
        Load data from hera

    """
    (train_data, test_data,
     train_masks, test_masks) = get_hera_data(args)

    if args.limit is not None:
        train_indx = np.random.permutation(len(train_data))[:args.limit]
        train_data = train_data[train_indx]
        train_masks = train_masks[train_indx]

    test_masks_orig = copy.deepcopy(test_masks)
    if args.rfi_threshold is not None:
        test_masks = flag_data(test_data, args)
        train_masks = flag_data(train_data, args)
        test_masks = np.expand_dims(test_masks, axis=-1)
        train_masks = np.expand_dims(train_masks, axis=-1)

    _max = np.mean(test_data[np.invert(test_masks)]) + 4 * np.std(test_data[np.invert(test_masks)])
    _min = np.absolute(
        np.mean(test_data[np.invert(test_masks)]) - np.std(test_data[np.invert(test_masks)]))
    test_data = np.clip(test_data, _min, _max)
    test_data = np.log(test_data)
    test_data = process(test_data, per_image=False)  # .astype(np.float16)

    _max = np.mean(train_data[np.invert(train_masks)]) + 4 * np.std(
        train_data[np.invert(train_masks)])
    _min = np.absolute(
        np.mean(train_data[np.invert(train_masks)]) - np.std(train_data[np.invert(train_masks)]))
    train_data = np.clip(train_data, _min, _max)
    train_data = np.log(train_data)
    train_data = process(train_data, per_image=False)  # .astype(np.float16)

    if args.patches:
        p_size = (1, args.patch_x, args.patch_y, 1)
        s_size = (1, args.patch_stride_x, args.patch_stride_y, 1)
        rate = (1, 1, 1, 1)

        train_data = get_patches(train_data, None, p_size, s_size, rate, 'VALID')
        train_masks = get_patches(train_masks, None, p_size, s_size, rate, 'VALID').astype(bool)

        test_data = get_patches(test_data, None, p_size, s_size, rate, 'VALID')
        test_masks = get_patches(test_masks.astype('int'), None, p_size, s_size, rate,
                                 'VALID').astype(bool)

        test_masks_orig = get_patches(test_masks_orig.astype('int'), None, p_size, s_size, rate,
                                      'VALID').astype(bool)

        train_labels = np.empty(len(train_data), dtype='object')
        train_labels[np.any(train_masks, axis=(1, 2, 3))] = args.anomaly_class
        train_labels[np.invert(np.any(train_masks, axis=(1, 2, 3)))] = 'normal'

        test_labels = np.empty(len(test_data), dtype='object')
        test_labels[np.any(test_masks, axis=(1, 2, 3))] = args.anomaly_class
        test_labels[np.invert(np.any(test_masks, axis=(1, 2, 3)))] = 'normal'

        ae_train_data = train_data[np.invert(np.any(train_masks, axis=(1, 2, 3)))]
        ae_train_labels = train_labels[np.invert(np.any(train_masks, axis=(1, 2, 3)))]

    if str(args.model).find('AE') != -1:
        train_dataset = tf.data.Dataset.from_tensor_slices(ae_train_data).shuffle(BUFFER_SIZE,
                                                                                  seed=42).batch(
            BATCH_SIZE)
    else:
        train_dataset = tf.data.Dataset.from_tensor_slices(train_data).shuffle(BUFFER_SIZE,
                                                                               seed=42).batch(
            BATCH_SIZE)

    return (train_dataset,
            train_data,
            train_labels,
            train_masks,
            ae_train_data,
            ae_train_labels,
            test_data,
            test_labels,
            test_masks,
            test_masks_orig)


def load_lofar(args):
    """
        Load data from lofar 

    """
    train_data, train_masks, test_data, test_masks = get_lofar_data(args)

    if args.limit is not None:
        train_indx = np.random.permutation(len(train_data))[:args.limit]
        test_indx = np.random.permutation(len(test_data))[:args.limit]

        train_data = train_data[train_indx]
        train_masks = train_masks[train_indx]
        # test_data   = test_data  [test_indx]
        # test_masks  = test_masks [test_indx]

    if args.rfi_threshold is not None:
        train_masks = flag_data(train_data, args)
        train_masks = np.expand_dims(train_masks, axis=-1)

    _max = np.mean(test_data[np.invert(test_masks)]) + 95 * np.std(test_data[np.invert(test_masks)])
    _min = np.absolute(
        np.mean(test_data[np.invert(test_masks)]) - 3 * np.std(test_data[np.invert(test_masks)]))

    test_data = np.clip(test_data, _min, _max)
    test_data = np.log(test_data)
    test_data = process(test_data, per_image=False)

    train_data = np.clip(train_data, _min, _max)
    train_data = np.log(train_data)
    train_data = process(train_data, per_image=False)

    if args.patches:
        p_size = (1, args.patch_x, args.patch_y, 1)
        s_size = (1, args.patch_stride_x, args.patch_stride_y, 1)
        rate = (1, 1, 1, 1)

        train_data = get_patches(train_data, None, p_size, s_size, rate, 'VALID')
        test_data = get_patches(test_data, None, p_size, s_size, rate, 'VALID')
        train_masks = get_patches(train_masks, None, p_size, s_size, rate, 'VALID').astype(bool)
        test_masks = get_patches(test_masks.astype('int'), None, p_size, s_size, rate,
                                 'VALID').astype(bool)

        train_labels = np.empty(len(train_data), dtype='object')
        train_labels[np.any(train_masks, axis=(1, 2, 3))] = args.anomaly_class
        train_labels[np.invert(np.any(train_masks, axis=(1, 2, 3)))] = 'normal'

        test_labels = np.empty(len(test_data), dtype='object')
        test_labels[np.any(test_masks, axis=(1, 2, 3))] = args.anomaly_class
        test_labels[np.invert(np.any(test_masks, axis=(1, 2, 3)))] = 'normal'

        ae_train_data = train_data[np.invert(np.any(train_masks, axis=(1, 2, 3)))]
        ae_train_labels = train_labels[np.invert(np.any(train_masks, axis=(1, 2, 3)))]

    if str(args.model).find('AE') != -1:
        train_dataset = tf.data.Dataset.from_tensor_slices(ae_train_data).shuffle(BUFFER_SIZE,
                                                                                  seed=42).batch(
            BATCH_SIZE)
    else:
        train_dataset = tf.data.Dataset.from_tensor_slices(train_data).shuffle(BUFFER_SIZE,
                                                                               seed=42).batch(
            BATCH_SIZE)

    return (train_dataset,
            train_data,
            train_labels,
            train_masks,
            ae_train_data,
            ae_train_labels,
            test_data,
            test_labels,
            test_masks,
            test_masks)


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def tf_record(image: np.array, mask: np.array):
    image_shape = image.shape
    feature = {
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'channels': _int64_feature(image_shape[2]),
        'mask': _bytes_feature(mask.encode('utf-8')),
        'image': _bytes_feature(image.tobytes())
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def args_to_tf_filename(args):
    anomaly_string = "-all" if args.rfi is None else "-" + str(args.anomaly_type)
    dataset_type = 'AE' if str(args.model).find('AE') != -1 else 'STD'
    filename = f"{args.data}{anomaly_string}-{args.rfi_threshold}-{args.patch_x}-{dataset_type}"
    return filename


def process_into_tfrecords_dataset(train_data, train_masks, test_data, test_masks, filename,
                                   outputdir):
    outdir = os.path.join(outputdir, "processed_data")
    os.makedirs(outdir, exist_ok=True)

    with tf.io.TFRecordWriter(f'{outdir}{os.sep}{filename}-train.tfrecords') as writer:
        for x, y in zip(train_data, train_masks):
            record = tf_record(x, y)
            writer.write(record.SerializeToString())

    with tf.io.TFRecordWriter(f'{outdir}{os.sep}{filename}-test.tfrecords') as writer:
        for x, y in zip(test_data, test_masks):
            record = tf_record(x, y)
            writer.write(record.SerializeToString())


if __name__ == "__main__":
    with tf.device("/cpu:0"):
        data = None
        if args.args.data == "HERA":
            data = load_hera(args.args)
        elif args.args.data == "LOFAR":
            data = load_lofar(args.args)
        if not data:
            exit(0)
        (train_dataset, train_data, train_labels, train_masks, ae_train_data, ae_train_labels,
         test_data, test_labels, test_masks, test_masks_orig) = data
        filename = args_to_tf_filename(args.args)
        process_into_tfrecords_dataset(ae_train_data, ae_train_labels, test_data, test_labels, filename, '.')
