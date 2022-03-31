import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import config_emotion as params

import tensorflow as tf
from tensorflow.keras.layers import *
import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from sklearn.metrics import *
from keras.engine import Model
from keras.layers import Input, Flatten, Dense, Activation, Conv2D, MaxPool2D, BatchNormalization, Dropout, MaxPooling2D


class Trainer:
    def __init__(self):
        self.pre_trained_model = keras.applications.ResNet152(
            include_top=False, weights='imagenet', input_tensor=None,
            input_shape=(params.RESIZE,
                         params.RESIZE, 3),
            pooling='avg', classes=1000
        )

    def modelFineTune(self):
        last_layer = self.pre_trained_model.layers[-1].output
        #x = Flatten(name='flatten')(last_layer)
        #x = Dropout(params.DROPOUT_RATE[0])(last_layer)
        x = Dense(4096, activation='relu', name='fc6')(last_layer)
        x = Dropout(params.DROPOUT_RATE[0])(x)
        x = Dense(1024, activation='relu', name='fc7')(x)
        x = Dropout(params.DROPOUT_RATE[1])(x)

        if params.FILTER_BATCH == 0:
            for i in range(len(self.pre_trained_model.layers)):
                try:
                    strType = str(type(self.pre_trained_model.layers[i])).split(
                        '.')[-1][:-2]
                    if strType == params.FRONZEN_LAYER:
                        continue
                    else:
                        self.pre_trained_model.layers[i].trainable = False
                except:
                    print(strType)
                    self.pre_trained_model.layers[i].trainable = False
        out = Dense(7, activation='softmax', name='classifier')(x)
        model = Model(self.pre_trained_model.input, out)
        return model

    def optimization(self):
        optim = keras.optimizers.Adam(
            lr=params.ADAM_LEARNING_RATE,
            beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        sgd = keras.optimizers.SGD(
            lr=params.SGD_LEARNING_RATE,
            momentum=0.9, decay=params.SGD_DECAY, nesterov=True)
        rlrop = keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy', mode='auto',
            factor=0.1, patience=10, min_lr=0.00001, verbose=1)
        return optim, sgd, rlrop

    def get_datagen(self, dataset, aug=False):
        if aug:
            datagen = ImageDataGenerator(
                rescale=1./255,
                featurewise_center=False,
                featurewise_std_normalization=False,
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.1,
                horizontal_flip=True)
        else:
            datagen = ImageDataGenerator(rescale=1./255)

        return datagen.flow_from_directory(
            dataset,
            target_size=(params.RESIZE,
                         params.RESIZE),
            color_mode='rgb',
            shuffle=True,
            class_mode='categorical',
            batch_size=params.BATCH_SIZE)

    def train(self):
        self.model = self.modelFineTune()
        optim, sgd, rlrop = self.optimization()
        self.model.compile(optimizer=optim, loss='categorical_crossentropy',
                           metrics=['accuracy'])

        train_generator = self.get_datagen(params.PATH_TRAIN, True)
        dev_generator = self.get_datagen(params.PATH_VAL)
        test_generator = self.get_datagen(params.PATH_TEST)

        history = self.model.fit_generator(
            generator=train_generator,
            validation_data=dev_generator,
            steps_per_epoch=params.NUM_TRAIN // params.BATCH_SIZE,
            validation_steps=params.NUM_VAL // params.BATCH_SIZE,
            shuffle=True,
            epochs=params.EPOCHS,
            callbacks=[rlrop],
            use_multiprocessing=True,
        )

        results_dev = model.evaluate_generator(dev_generator,
                                               params.NUM_VAL // params.BATCH_SIZE)
        results_test = model.evaluate_generator(test_generator,
                                                params.NUM_TEST // params.BATCH_SIZE)

        epoch_str = '-EPOCHS_' + str(params.EPOCHS)
        test_acc = 'test_acc_%.3f' % results_test[1]
        model.save('Model' + epoch_str + test_acc + '.h5')
        return results_dev, results_test


if __name__ == "__main__":
    trainer = Trainer()
    result = trainer.train()
    print(result)
