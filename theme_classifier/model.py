from ast import Or
from typing import Tuple
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, CSVLogger

from keras import Model, Input
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from tensorflow.keras import layers

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

from ordered_label_binarizer import OrderedLabelBinarizer

from categories import *
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
#from tensorflow.keras import layers
import os
import sys
from pathlib import Path
import logging
from datetime import datetime

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

logging.basicConfig(filename='model_debug.log', level=logging.DEBUG)

table_path = os.path.join(Path(__file__).parent, 'video_data', 'sampledImages.csv')
imagedir = os.path.join(Path(__file__).parent, 'videos', 'images')

# Create the Scikit-learn MultiLabelBinarizer
def get_multilabelbinarizer():
    labels = [("menu",),tuple(decode_styles.values()),tuple(decode_themes.values()),("day","night")]
    classes = [x for t in labels for x in t ]
    mlb = MultiLabelBinarizer(classes=classes)
    mlb.fit(labels)
    return mlb

def mlb_inverse_transform(pred):
    nstyles = len(decode_styles.values()) + 1
    nthemes = len(decode_themes.values())
    ntime = 2
    style = np.argmax(pred[:nstyles])
    theme = np.argmax(pred[nstyles:-2])
    time = np.argmax(pred[-2:])
    binarized = np.zeros(len(pred))
    binarized[style] = 1
    binarized[nstyles + theme] = 1
    binarized[nstyles+nthemes+time] = 1
    binarized = np.expand_dims(binarized, axis=0)
    return binarized

def mlb_inverse_transform_batch(preds: np.ndarray, one_hots: Tuple[OrderedLabelBinarizer, OrderedLabelBinarizer, OrderedLabelBinarizer]):
    style_indices = np.argmax(preds[0],axis=1)
    theme_indices = np.argmax(preds[1],axis=1)
    time_indices = np.argmax(preds[2],axis=1)

    style_classes = one_hots[0].inverse_transform_indices(style_indices)
    theme_classes = one_hots[1].inverse_transform_indices(theme_indices)
    time_classes = one_hots[2].inverse_transform_indices(time_indices)
    return list(zip(style_classes, theme_classes, time_classes))

def class_accuracy(y_true, y_pred):    
    style_slice_pred = tf.argmax(y_pred[:,0:6], axis=1)
    style_slice_true = tf.argmax(y_true[:,0:6], axis=1)
    style_mask = tf.cast(tf.math.equal(style_slice_pred, style_slice_true), tf.int8)
    
    menu_mask = tf.cast(tf.math.not_equal(style_slice_true, tf.constant(5, dtype=tf.int64)), tf.int8)

    theme_slice_pred = tf.argmax(y_pred[:,6:16], axis=1)
    theme_slice_true = tf.argmax(y_true[:,6:16], axis=1)
    theme_mask = tf.cast(tf.math.equal(theme_slice_pred, theme_slice_true), tf.int8)

    time_slice_pred = tf.argmax(y_pred[:,16:18], axis=1)
    time_slice_true = tf.argmax(y_true[:,16:18], axis=1)
    time_mask = tf.cast(tf.math.equal(time_slice_pred, time_slice_true), tf.int8)

    style_sum = tf.math.multiply(style_mask, tf.math.add(style_mask, tf.math.multiply(tf.constant(2, dtype=tf.int8), tf.math.subtract(tf.constant(1, dtype=tf.int8), menu_mask))))
    theme_sum = tf.math.multiply(theme_mask, menu_mask)
    time_sum = tf.math.multiply(time_mask, menu_mask)
    total_sum = tf.math.add(style_sum, tf.math.add(theme_sum, time_sum))
    average = tf.math.divide(total_sum, tf.constant(3, dtype=tf.int8))
    #print(style_slice_pred.numpy())
    #print(style_slice_true.numpy())
    #print(style_sum.numpy())
    #print(theme_sum.numpy())
    #print(time_sum.numpy())
    return average
    
def get_one_hots():
    style_oh = OrderedLabelBinarizer(list(encode_styles.keys()) + ['menu'])
    
    theme_oh = OrderedLabelBinarizer(list(encode_themes.keys()) + [''])

    time_oh = OrderedLabelBinarizer(["day","night",''])

    return (style_oh, theme_oh, time_oh)

# The helper function
def multilabel_flow_from_dataframe(data_generator, mlb, df):
    assert isinstance(mlb, MultiLabelBinarizer), \
            "MultiLabelBinarizer is required."
    for x, y in data_generator:
        indices = y.astype(np.int).tolist()
        rows = df.iloc[indices]
        tup = rows.apply(lambda r: strip_tuple((r['Style'],r['Theme'],r['Time'])), axis=1)
        y_multi = mlb.transform(
             tup
        )
        yield x, y_multi

# The helper function
def multioutput_flow_from_dataframe(data_generator, one_hots, df):
    for x, y in data_generator:
        indices = y.astype(np.int).tolist()
        rows = df.iloc[indices]
        styles = one_hots[0].transform(rows["Style"])
        themes = one_hots[1].transform(rows["Theme"])
        times = one_hots[2].transform(rows["Time"])
        yield x, [styles, themes, times]

# Create the Keras ImageDataGenerator
image_data_generator = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=.1
)

def get_generator(frame,mlb, df):
    gen = image_data_generator.flow_from_dataframe(
        dataframe=frame,
        directory=imagedir,
        x_col='Filename',
        y_col='index',
        class_mode='raw',
        target_size=(512,512),
        batch_size=16,
        shuffle=True
    )
    one_hots = get_one_hots()
    return multioutput_flow_from_dataframe(gen, one_hots, df)

def res_net_block(input_data, filters, conv_size):
  x = layers.Conv2D(filters, conv_size, activation='relu', padding='same')(input_data)
  x = layers.BatchNormalization()(x)
  x = layers.Conv2D(filters, conv_size, activation=None, padding='same')(x)
  x = layers.BatchNormalization()(x)
  x = layers.Add()([x, input_data])
  x = layers.Activation('relu')(x)
  return x

def prepare_model():
    input_layer = Input(shape=[512,512,3])
    x = layers.Conv2D(32, 7, strides=(2,2), activation='relu')(input_layer)
    x = layers.Conv2D(64, 3, strides=(2,2), activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    num_res_net_blocks = 5
    for i in range(num_res_net_blocks):
        x = res_net_block(x, 64, 3)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    #x = Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    def get_output_layers(ncat, name, x):
        y = Dense(units=64, activation='relu')(x)
        y = layers.Dropout(0.2)(y)
        y = Dense(units=128, activation='relu')(y)
        y = Dense(units=64, activation='relu')(y)
        y = Dense(ncat, activation='softmax', name=name)(y)
        return y
    style_out = get_output_layers(6, "style", x)
    theme_out = get_output_layers(11, "theme", x)
    time_out = get_output_layers(3, "time", x)

    model = Model(inputs=input_layer, outputs=[style_out, theme_out, time_out])
    model.compile(
        optimizer=keras.optimizers.Adam(epsilon=0.01),
        loss={'style':'categorical_crossentropy', 'theme':'categorical_crossentropy', 'time':'categorical_crossentropy'},
        metrics={'style':'accuracy', 'theme':'accuracy', 'time':'accuracy'},
    )
    return model

def fit_model(model, train_generator, valid_generator, custom_calls=[]):
    date = datetime.now().strftime("%m-%d")
    # Helper: Save the model.
    checkpointer = ModelCheckpoint(
        filepath=os.path.join('model', 'checkpoints', 'theme.' + date + '.{epoch:03d}-{val_loss:.2f}.hdf5'),
    verbose=1,
    save_best_only=True)

    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=10)

    # Helper: TensorBoard
    tensorboard = TensorBoard(log_dir=os.path.join('model', 'logs'))

    csvlog = CSVLogger(os.path.join('model','logs','csv-' + date + '.csv'), separator=',', append=False)

    model.fit(
        train_generator,
        validation_data = valid_generator,
        epochs = 200,
        steps_per_epoch = 200,
        validation_steps = 40,
        callbacks = [checkpointer, early_stopper, tensorboard, csvlog] + custom_calls
    )


def get_trained_model(should_train=True, weights=None, custom_calls=[]):
    mlb = get_multilabelbinarizer()

    # Prepare the model
    model = prepare_model()

    print(model.summary())
    #return

    if (weights is not None):
        print("loading weights from: " + weights)
        model.load_weights(weights)

    if (should_train):
        # Read the dataset
        df = pd.read_csv(table_path,keep_default_na=False)
        train_df = df.loc[df['Train'].values]
        valid_df = df[(df['Train'].values) == False]

        #get the generators
        train_generator = get_generator(train_df, mlb, df)
        valid_generator = get_generator(valid_df, mlb, df)

        fit_model(model, train_generator, valid_generator, custom_calls)
        model.save("model/theme_"+str(datetime.now()).replace(" ","_")+".h5")

    return model

if __name__ == "__main__":
    weights = None if len(sys.argv) < 2 else sys.argv[1]
    train = True if len(sys.argv) < 3 else bool(sys.argv[2])
    model = get_trained_model(train,weights)
        


