import tensorflow as tf
import numpy as np
from model_config import BATCH_SIZE


def infer(model, data, args, arch):
    """
        Performs inference in batches for given model on supplied data

        Parameters
        ----------
        model (tf.keras.Model or tf.keras.layers.Layer) 
        data (np.array) or [x,nln] in the case of NNAE 
        args (Namespace)
        arch (str)
        
        Returns
        -------
        np.array

    """
    data_tensor = tf.data.Dataset.from_tensor_slices(data).batch(BATCH_SIZE)

    if arch == 'AE' or arch == 'encoder' or arch == 'DKNN':
        if arch == 'encoder':
            output = np.empty([len(data), args.latent_dim], np.float32)
        elif arch == 'DKNN':
            output = np.empty([len(data), 2048], np.float32)
        else:
            output = np.empty(data.shape, dtype=np.float32)
        strt, fnnsh = 0, BATCH_SIZE
        for batch in data_tensor:
            output[strt:fnnsh, ...] = model(batch, training=False).numpy()
            strt = fnnsh
            fnnsh += BATCH_SIZE

    else:
        output = np.empty([len(data), args.latent_dim], dtype=np.float32)
        strt, fnnsh = 0, BATCH_SIZE
        for batch in data_tensor:
            output[strt:fnnsh, ...] = model(batch, training=False)[0].numpy()  # for disc
            strt = fnnsh
            fnnsh += BATCH_SIZE

    return output


def get_error(model_type,
              x,
              x_hat,
              ab=True,
              mean=True):
    """
        Gets the reconstruction error of a given model 

        Parameters
        ----------
        model_type (str) 
        x (np.array) 
        x_hat (np.array) 
        ab (bool) default True
        mean (bool) default True

        Returns
        -------
        np.array

    """

    if ((model_type == 'AE') or
            (model_type == 'AE-SSIM') or
            (model_type == 'DAE')):
        error = x - x_hat


    elif model_type == 'UNET' or model_type == 'RNET' or 'RFI_NET':
        error = x_hat

    np.abs(error, dtype=np.float32, out=error)

    if mean:
        error = np.mean(error, axis=tuple(range(1, error.ndim)))

    return error
