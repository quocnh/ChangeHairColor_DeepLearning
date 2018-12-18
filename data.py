import argparse
import os
from glob import glob

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from scipy.misc import imresize
from scipy.ndimage import imread
from sklearn.model_selection import train_test_split
import cv2 as cv2
seed = 1


def standardize(images, mean=None, std=None):
    if mean is None:
        # These values are available from all images.
        mean = [[[29.24429131, 29.24429131, 29.24429131, 0]]]
    if std is None:
        # These values are available from all images.
        std = [[[69.8833313, 63.37436676, 61.38568878, 0]]]
    x = (images - np.array(mean)) / (np.array(std) + 1e-7)
    return x


def _create_datagen(images, masks, img_gen, mask_gen):
    img_iter = img_gen.flow(images, seed=seed)
    # only hair
    mask_iter = mask_gen.flow(np.expand_dims(masks[:, :, :, 0], axis=4),
                              # use same seed to apply same augmentation with image
                              seed=seed)

    def datagen():
        while True:
            img = img_iter.next()
            mask = mask_iter.next()
            # Modify to return multiple outputs
            yield img, mask

    return datagen

def create_data_combine(empty, affine):
    X_empty = np.load(empty)
    X_affine = np.load(affine)
    return np.concatenate((X_empty, X_affine))

def load_data(img_file, mask_file):
    images = np.load(img_file)
    masks = np.load(mask_file)
    X_train, X_val, Y_train, Y_val = train_test_split(images,
                                                      masks,
                                                      test_size=0.2,
                                                      random_state=seed)

    train_img_gen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        # rescale=1. / 255,
        rotation_range=20,
        shear_range=0.2,
        zoom_range=0.2,
        # vertical_flip=True,  # debug
        horizontal_flip=True,
    )
    train_img_gen.fit(images)
    train_mask_gen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        shear_range=0.2,
        zoom_range=0.2,
        # vertical_flip=True,  # debug
        horizontal_flip=True,
    )
    train_gen = _create_datagen(X_train, Y_train,
                                img_gen=train_img_gen,
                                mask_gen=train_mask_gen)

    validation_img_gen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        horizontal_flip=True,
    )
    validation_img_gen.fit(images)
    validation_mask_gen = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True,
    )
    validation_gen = _create_datagen(X_val, Y_val,
                                     img_gen=validation_img_gen,
                                     mask_gen=validation_mask_gen)

    return train_gen, validation_gen, images.shape[1:3]

def affine(mask, img_size):
    # Get 3 points on the mask
    pts1 = np.float32([[50,50],[100,50],[50,100]])
    rand_num = np.random.randint(0, 10, (3,2))
    # Calculate 3 points after affine transform
    pts2 = (pts1 + rand_num).astype(np.float32)
    # calculate affine transform matrix
    M = cv2.getAffineTransform(pts1,pts2)
    dst = cv2.warpAffine(mask, M,(img_size, img_size), borderMode=cv2.BORDER_REPLICATE)
    # return only first layer as mask
    return dst[:, :, 0:1]

# def spline(mask, img_size):
#     tps = cv2.createThinPlateSplineShapeTransformer()


def create_data(data_dir, out_dir, img_size, data_type):
    """
    It expects following directory layout in data_dir.

    images/
      0001.jpg
      0002.jpg
    masks/
      0001.ppm
      0002.ppm

    Mask image has 3 colors R, G and B. R is hair. G is face. B is bg.
    Finally, it will create images.npy and masks.npy in out_dir.

    :param data_dir:
    :param out_dir:
    :param img_size:
    :return:
    """
    img_files = sorted(glob(data_dir + '/images/*.jpg'))
    mask_files = sorted(glob(data_dir + '/masks/*.ppm'))
    X = []
    Y = []
    
    for img_path, mask_path in zip(img_files, mask_files):
        img = imread(img_path)
        img = imresize(img, (img_size, img_size))

        mask = imread(mask_path)
        mask = imresize(mask, (img_size, img_size), interp='nearest')

        if (data_type == 'empty'):
            # empty 4th channel
            new_dim = np.zeros((img_size, img_size, 1))
            # Add new channel to the 3rd dimension
            img = np.concatenate((img, new_dim), axis=2)
        elif (data_type == 'affine'):
            # affine transform of 4th channel
            new_dim = affine(mask, img_size)
            # Add new channel to the 3rd dimension
            img = np.concatenate((img, new_dim), axis=2)
        elif (data_type == 'spline'):
            # # affine transform of 4th channel
            # new_dim = spline(mask, img_size)
            # # Add new channel to the 3rd dimension
            # img = np.concatenate((img, new_dim), axis=2)
            pass

        # debug
        if False:
            import matplotlib.pyplot as plt
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.subplot(1, 2, 2)
            plt.imshow(mask)
            plt.show()

        X.append(img)
        Y.append(mask)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    np.save(out_dir + '/images-{}-{}.npy'.format(data_type, img_size), np.array(X))
    np.save(out_dir + '/masks-{}-{}.npy'.format(data_type, img_size), np.array(Y))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_type',
        type=str,
        default='empty',
        help='create data with different 4th channel data: empty, affine, spline.'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/raw',
        help='directory in which images and masks are placed.'
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        default='data',
        help='directory to put outputs.'
    )
    parser.add_argument(
        '--img_size',
        type=int,
        default=192,
    )
    args, _ = parser.parse_known_args()
    create_data(**vars(args))
