from __future__ import print_function

import argparse
import os

from keras import callbacks
from keras.optimizers import SGD, Adam

from data import load_data
from learning_rate import create_lr_schedule
from loss import dice_coef_loss, dice_coef, recall, precision
from nets.MobileUNet import MobileUNet
from nets.DenseNet import RefineDenseNet

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

checkpoint_path = 'artifacts/checkpoint_weights_dense_100.{epoch:02d}-{val_loss:.2f}.h5'
trained_model_path = 'artifacts/model_dense_100.h5'

nb_train_samples = 2341
nb_validation_samples = 586


def train(img_file, mask_file, top_model_weights_path, train_prev, epochs, batch_size):
    train_gen, validation_gen, img_shape = load_data(img_file, mask_file)

    img_height = img_shape[0]
    img_width = img_shape[1]
    lr_base = 0.01 * (float(batch_size) / 16)

    model = MobileUNet(input_shape=(img_height, img_width, 4), 
                       alpha=1.0, 
                       alpha_up=0.25)
    # model = MobileDeepLab(input_shape=(img_height, img_width, 3))
    if (train_prev == False):
    	model.load_weights(os.path.expanduser(top_model_weights_path), by_name=True)

    model = RefineDenseNet(model, blocks=3, growth_rate=3, train_prev=train_prev)
    if (train_prev == True):
        model.load_weights(os.path.expanduser(top_model_weights_path), by_name=True)

    model.summary()

    model.compile(
        #optimizer=SGD(lr=0.0001, momentum=0.9),
        optimizer=Adam(lr=0.0001),
        loss=dice_coef_loss,
        metrics=[
            dice_coef,
            # recall,
            # precision,
            'binary_crossentropy',
        ],
    )

    # callbacks
    scheduler = callbacks.LearningRateScheduler(
        create_lr_schedule(epochs, lr_base=lr_base, mode='progressive_drops'))
    tensorboard = callbacks.TensorBoard(log_dir='./logs')
    checkpoint = callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                           save_weights_only=True,
                                           save_best_only=True)
    model.fit_generator(
        generator=train_gen(),
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_gen(),
        validation_steps=nb_validation_samples // batch_size,
        callbacks=[scheduler, tensorboard, checkpoint],
    )

    model.save(trained_model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--img_file',
        type=str,
        default='data/images.npy',
        help='image file as numpy format'
    )
    parser.add_argument(
        '--mask_file',
        type=str,
        default='data/masks.npy',
        help='mask file as numpy format'
    )
    parser.add_argument(
        '--top_model_weights_path',
        type=str,
        # default='~/.keras/models/top_model_weights.h5',
        default=None,
        help='The best weights trained on encoder-decoder model'
    )
    parser.add_argument(
        '--train_prev',
        type=bool,
        default=False,
        help='Whether to train the encoder-decoder part or just train the DenseNet refine part'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=250,
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
    )
    args, _ = parser.parse_known_args()

    if not os.path.exists('artifacts'):
        os.makedirs('artifacts')

    train(**vars(args))
