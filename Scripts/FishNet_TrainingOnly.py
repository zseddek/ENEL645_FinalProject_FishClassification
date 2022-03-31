import tensorflow as tf 
import os
from random import shuffle
import numpy as np
import os.path
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

os.chdir('/data')
working_directory = os.getcwd()
print("working directory:", working_directory)

def make_image_df(folder):
    test_image_dir = Path('fish_data/'+folder)
    test_filepaths = list(test_image_dir.glob(r'*/*.*'))
    test_labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], test_filepaths))

    test_filepaths = pd.Series(test_filepaths, name='Filepath').astype(str)
    test_labels = pd.Series(test_labels, name='Label')
    test_image_df = pd.concat([test_filepaths, test_labels], axis=1)
    return test_image_df

test_df = make_image_df('Test')
dev_df = make_image_df('Train_Val')
total_df = pd.concat([dev_df, test_df], axis=0, ignore_index=True)
total_df = total_df.sample(frac = 1, ignore_index=True, random_state = 32) # Adding extra shuffling/randomness for img visualization

dev_df, test_df = train_test_split(total_df, test_size=0.1, train_size=0.9, shuffle=True, random_state=42)
train_df, val_df = train_test_split(dev_df, test_size=0.2, train_size=0.8, shuffle=True, random_state=42)

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255 # Could apply additional augmentation here
)


train_images = image_generator.flow_from_dataframe(
    dataframe = train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=42
)

val_images = image_generator.flow_from_dataframe(
    dataframe = val_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=42
)

test_images = image_generator.flow_from_dataframe(
    dataframe = test_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=42
)

input = Input(shape =(224,224,3))
l1 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(input)
l2 = MaxPool2D(2,2)(l1)
l3 = Dropout(0.2)(l2)

# Regularizing using penalty instead of dropout (want to maintain feature extraction capabilities)
l4 = Conv2D(filters=64, kernel_size=(3,3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2=0.001))(l3) 
l5 = MaxPool2D(2,2)(l4)
l6 = Flatten()(l5)

l7 = Dense(256, activation='relu')(l6)
l8 = Dropout(0.2)(l7) # Only change after 94.9% test accuracy training run
l9 = Dense(256, activation='relu')(l8)
l10 = Dropout(0.2)(l9) # Only change after 94.9% test accuracy training run
output = Dense(9, activation='softmax')(l10)

model = Model (inputs=input, outputs =output)
model.compile(
    optimizer='adam', # Starting learning rate of 0.001 (default parameter)
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 10) # Fine tune
checkpoint_path = "training_1/cp.ckpt"
monitor = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss',
                                             verbose=1,save_best_only=True,
                                             save_weights_only=True,
                                             mode='min') # Only saves the best model (so far) in terms of min validation loss

def scheduler(epoch, lr):
    if epoch%10 == 0 and epoch!= 0:
        lr = lr/1.2
    return lr

lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler,verbose = 0)
lr_schedule_on_plateau = ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.1, patience=5, min_lr=0.000001, verbose=1)
callbacks = [early_stop, monitor, lr_schedule_on_plateau,lr_schedule]

print("\n************************ STARTING TRAINING ************************")

try:
    history = model.fit(
        train_images, 
        validation_data=val_images, 
        epochs=50, # Fine tune
        callbacks=callbacks
    )
    np.save('history.npy', history.history)
except KeyboardInterrupt:
    print("\nmodel training terminated\n")

print("\n************************ COMPLETED TRAINING ************************")

