from architectures import *
from data import *
from utils import args


def main():
    """
        Reads data and cmd arguments and trains models
    """
    with tf.device("/cpu:0"):
        if args.args.data == 'HERA':
            data = load_hera(args.args)
        elif args.args.data == 'LOFAR':
            data = load_lofar(args.args)
        elif args.args.data == 'HIDE':
            data = load_hide(args.args)

    (train_dataset, train_data, train_labels, train_masks, ae_train_data, ae_train_labels,
     test_data, test_labels, test_masks, test_masks_orig) = data

    print(" __________________________________ \n Save name {}".format(
        args.args.model_name))
    print(" __________________________________ \n")

    if args.args.model == 'UNET':
        del train_dataset
        del train_labels
        del ae_train_data
        del ae_train_labels
        with tf.device('/cpu:0'):
            train_unet(train_data, train_masks, test_data,
                       test_labels, test_masks, test_masks_orig, args.args)

    if args.args.model == 'RNET':
        del train_dataset
        del train_labels
        del ae_train_data
        del ae_train_labels
        train_rnet(train_data, train_masks, test_data,
                   test_labels, test_masks, test_masks_orig, args.args)

    if args.args.model == 'RFI_NET':
        del train_dataset
        del train_labels
        del ae_train_data
        del ae_train_labels
        train_rfi_net(train_data, train_masks, test_data,
                      test_labels, test_masks, test_masks_orig, args.args)

    elif args.args.model == 'DKNN':
        del train_dataset
        del train_data
        del train_labels
        del train_masks
        del ae_train_labels
        train_resnet(ae_train_data, test_data, test_labels,
                     test_masks, test_masks_orig, args.args)

    elif args.args.model == 'AE':
        del train_dataset
        del train_data
        del train_labels
        del train_masks
        del ae_train_labels
        train_ae(ae_train_data, test_data, test_labels,
                 test_masks, test_masks_orig, args.args)

    elif args.args.model == 'AE-SSIM':
        del train_data
        del train_labels
        del train_masks
        del ae_train_labels
        train_ae_ssim(train_dataset, ae_train_data, test_data, test_labels,
                      test_masks, test_masks_orig, args.args)

    elif args.args.model == 'DAE':
        del train_data
        del train_labels
        del train_masks
        del ae_train_labels
        train_dae(train_dataset, ae_train_data, test_data, test_labels,
                  test_masks, test_masks_orig, args.args)


if __name__ == '__main__':
    main()
