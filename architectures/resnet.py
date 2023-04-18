import os

from model_config import *
from .helper import end_routine

optimizer = tf.keras.optimizers.Adam()


def main(train_images, test_images, test_labels, test_masks,
         test_masks_orig, args):
    s = 256 // args.patch_x

    inputs = tf.keras.layers.Input(shape=args.input_shape)
    rgb = tf.keras.layers.Concatenate(axis=-1)([inputs, inputs, inputs])
    resize = tf.keras.layers.UpSampling2D(size=(s, s))(rgb)
    crop = tf.keras.layers.Cropping2D(16)(resize)

    resnet = tf.keras.applications.resnet.ResNet50(include_top=False, weights='imagenet',
                                                   input_shape=(224, 224, 3), pooling='max')(crop)

    model = tf.keras.Model(inputs=inputs, outputs=resnet)

    dir_path = 'outputs/{}/{}/{}'.format('DKNN',
                                         args.anomaly_class,
                                         args.model_name)
    os.makedirs(dir_path)

    end_routine(train_images, test_images, test_labels, test_masks, test_masks_orig, [model],
                'DKNN', args)


if __name__ == '__main__':
    main()
