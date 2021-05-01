import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import MultiLabelBinarizer
from categories import *
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import os
from pathlib import Path

table_path = os.path.join(Path(__file__).parent, 'video_data', 'imageTable.csv')
#imagedir = os.path.join(Path(__file__).parent, 'videos', 'images')
imagedir = "/content/kAIzoVision/theme_classifier"

# Create the Scikit-learn MultiLabelBinarizer
labels = [("menu",),tuple(decode_styles.values()),tuple(decode_themes.values()),("day","night")]
classes = [x for t in labels for x in t ]
mlb = MultiLabelBinarizer(classes=classes)
mlb.fit(labels)


# Read the dataset
df = pd.read_csv(table_path)

# The helper function
def multilabel_flow_from_dataframe(data_generator, mlb):
    print("mffd called")
    for x, y in data_generator:
        assert isinstance(mlb, MultiLabelBinarizer), \
               "MultiLabelBinarizer is required."
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
        batch_size=32
    )
    return multilabel_flow_from_dataframe(gen, mlb)

def prepare_model():
    model = keras.Sequential([
        # Block One
        layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same',
                    input_shape=[512,512, 3]),
        layers.MaxPool2D(),

        # Block Two
        layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'),
        layers.MaxPool2D(),

        # Block Three
        layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
        layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
        layers.MaxPool2D(),

        # Head
        layers.Flatten(),
        layers.Dense(6, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(18, activation='sigmoid'),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(epsilon=0.01),
        loss='binary_crossentropy',
        metrics=['binary_accuracy']
    )
    return model


print(imagedir)
print(df.dtypes)
train_df = df.loc[df['Train'].values]
valid_df = df[(df['Train'].values) == False]
print(train_df)
print(valid_df)

#get the generators
train_generator = get_generator(train_df, mlb)
valid_generator = get_generator(valid_df, mlb)

print("got here2")

# Prepare the model
model = prepare_model()

print("gothere")
model.fit(
    train_generator,
    validation_data = valid_generator,
    epochs = 5
)