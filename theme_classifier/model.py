import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D

from sklearn.preprocessing import MultiLabelBinarizer
from categories import *
import numpy as np
from tensorflow import keras
#from tensorflow.keras import layers
import os
import sys
from pathlib import Path

table_path = os.path.join(Path(__file__).parent, 'video_data', 'imageTable.csv')
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

# The helper function
def multilabel_flow_from_dataframe(data_generator, mlb):
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

# Create the Keras ImageDataGenerator
image_data_generator = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=.1
)

def get_generator(frame,mlb):
    gen = image_data_generator.flow_from_dataframe(
        dataframe=frame,
        directory=imagedir,
        x_col='Filename',
        y_col='index',
        class_mode='raw',
        target_size=(512,512),
        batch_size=32,
        shuffle=True
    )
    return multilabel_flow_from_dataframe(gen, mlb)

def prepare_model():
    model = keras.Sequential([
        # Block One
        Conv2D(filters=32, kernel_size=3, activation='relu', padding='same',
                    input_shape=[512,512, 3]),
        MaxPooling2D(),

        # Block Two
        Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'),
        MaxPooling2D(),

        # Block Three
        Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
        Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
        MaxPooling2D(),

        
        Flatten(),
        Dense(6, activation='relu'),
        Dropout(0.2),
        Dense(18, activation='sigmoid'),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(epsilon=0.01),
        loss='binary_crossentropy',
        metrics=['binary_accuracy']
    )
    return model

def fit_model(model, train_generator, valid_generator):
    # Helper: Save the model.
    checkpointer = ModelCheckpoint(
        filepath=os.path.join('model', 'checkpoints', 'theme.{epoch:03d}-{val_loss:.2f}.hdf5'),
    verbose=1,
    save_best_only=True)

    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=10)

    # Helper: TensorBoard
    tensorboard = TensorBoard(log_dir=os.path.join('model', 'logs'))

    model.fit(
        train_generator,
        validation_data = valid_generator,
        epochs = 20,
        steps_per_epoch = 2000,
        validation_steps = 400,
        callbacks = [checkpointer, early_stopper, tensorboard]
    )


def get_trained_model(weights=None):
    mlb = get_multilabelbinarizer()

    # Prepare the model
    model = prepare_model()

    if (weights is None):
        # Read the dataset
        df = pd.read_csv(table_path)
        train_df = df.loc[df['Train'].values]
        valid_df = df[(df['Train'].values) == False]

        #get the generators
        train_generator = get_generator(train_df, mlb)
        valid_generator = get_generator(valid_df, mlb)

        fit_model(model, train_generator, valid_generator)
        model.save("model/theme_"+str(datetime.now()).replace(" ","_")+".h5")
    else:
        print("loading weights from: " + weights)
        model.load_weights(weights)

    return model

if __name__ == "__main__":
    weights = None if len(sys.argv) <= 1 else sys.argv[1]
    model = get_trained_model(weights)
        


