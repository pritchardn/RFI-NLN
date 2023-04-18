import time

from model_config import *
from models import UNET
from utils import generate_and_save_images, generate_and_save_training, print_epoch, save_checkpoint
from .helper import end_routine

optimizer = tf.keras.optimizers.Adam()


@tf.function
def train_step(model, x, y):
    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape() as tape:
        x_hat = model(x, training=True)
        loss = bce(x_hat, y)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def train(unet, train_images, train_masks,
          args):
    unet_loss = []
    train_mask_dataset = tf.data.Dataset.from_tensor_slices(train_masks.astype('float32')).shuffle(
        BUFFER_SIZE, seed=42).batch(BATCH_SIZE)
    train_data_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE,
                                                                                  seed=42).batch(
        BATCH_SIZE)
    for epoch in range(args.epochs):
        start = time.time()

        for image_batch, mask_batch in zip(train_data_dataset, train_mask_dataset):
            auto_loss = train_step(unet, image_batch, mask_batch)

        generate_and_save_images(unet,
                                 epoch + 1,
                                 image_batch[:25, ...],
                                 'UNET',
                                 args)
        save_checkpoint(unet, epoch, args, 'UNET', 'unet')

        unet_loss.append(auto_loss)

        print_epoch('UNET', epoch, time.time() - start, {'UNET Loss': auto_loss.numpy()}, None)

    generate_and_save_training([unet_loss],
                               ['unet loss'],
                               'UNET', args)
    generate_and_save_images(unet, epoch, image_batch[:25, ...], 'UNET', args)

    return unet


def main(train_images, train_masks, test_images, test_labels,
         test_masks, test_masks_orig, args):
    unet = UNET(args)
    unet = train(unet, train_images, train_masks, args)
    end_routine(train_images, test_images, test_labels, test_masks, test_masks_orig, [unet], 'UNET',
                args)


if __name__ == '__main__':
    main()
