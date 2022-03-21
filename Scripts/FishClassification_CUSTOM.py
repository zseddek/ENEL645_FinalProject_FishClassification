#!/usr/bin/env python
# coding: utf-8

# In[404]:


import tensorflow as tf 
import os
from random import shuffle
import numpy as np


# In[405]:


os.chdir('/root/fish_class')
working_directory = os.getcwd()
print("working directory:", working_directory)


# 1. Loading Data and Preprocessing

# In[406]:


# Potentially remove this and try again .. 
train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255, # min-max Normalization, shifting pixel value to [0,1], max 255 to max of 1 (domain shift)
    validation_split=0.20
)

test_generator = tf.keras.preprocessing.image.ImageDataGenerator(                                                    
    rescale=1./255 # Apply same normalization, not performing other preprocessing steps
)


# In[407]:


# BATCH SIZE WAS ORIGINALLY 32
train_images = train_generator.flow_from_directory(
    directory= './Data/Train_Val',
    target_size=(200, 200),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=24,
    shuffle=True,
    seed=42,
    subset='training'
)

val_images = train_generator.flow_from_directory(
    directory= './Data/Train_Val',
    target_size=(200, 200),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=24,
    shuffle=True,
    seed=42,
    subset='validation' # Will only take 20% of the total data as the validation data
)

test_images = test_generator.flow_from_directory(
    directory= './Data/Test',
    target_size=(200, 200),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=24,
    shuffle=False, # We need to be able to generate metrics at the end
    seed=42
)


# In[408]:


print("Training image shape:", train_images.image_shape)
print("Validation image shape:", val_images.image_shape)
print("Test image shape:", test_images.image_shape)


# In[409]:


train_images.class_indices


# In[410]:


val_images.class_indices


# In[411]:


test_images.class_indices


# In[412]:


import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


# 2. Defining VGG16 (CNN) Architecture

# In[413]:


model = tf.keras.models.Sequential([
    
    tf.keras.layers.Conv2D(32, (3,3), activation='relu',  input_shape=(200,200,3), kernel_regularizer=tf.keras.regularizers.l2(l2=0.001)),
    tf.keras.layers.MaxPool2D(pool_size = (2,2)),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2=0.001)),
    tf.keras.layers.MaxPool2D(pool_size = (2,2)),
    
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2=0.001)),
    tf.keras.layers.MaxPool2D(pool_size = (2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2=0.001)),
    tf.keras.layers.MaxPool2D(pool_size = (2,2)),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(9, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam()

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


# 3. Defining Schedulers and Callbacks

# In[414]:


early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 10) # Fine tune
checkpoint_path = "ENEL645_FinalProject_FishClassification/training_2_rof/cp.ckpt"
monitor = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss',
                                             verbose=1,save_best_only=True,
                                             save_weights_only=True,
                                             mode='min') # Only saves the best model (so far) in terms of min validation loss

def scheduler(epoch, lr):
    if epoch%5 == 0 and epoch!= 0:
        lr = lr/1.4
    return lr

lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler,verbose = 0)
lr_schedule_on_plateau = ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.1, patience=3, min_lr=0.00000001, verbose=1)
callbacks = [early_stop, monitor, lr_schedule_on_plateau,lr_schedule]


# 4. Training Model

# In[415]:


try:
    history = model.fit(
        train_images, 
        validation_data=val_images, 
        epochs=35, # Fine tune
        callbacks=callbacks
    )
except KeyboardInterrupt:
    print("\nmodel training terminated\n")


# In[132]:


np.save('ENEL645_FinalProject_FishClassification/history.npy', history.history)


# In[ ]:


model.save('ENEL645_FinalProject_FishClassification/Model_rof')


# In[ ]:


print("\n************************ COMPLETED TRAINING ************************")


# 5. Loading Best Model and Testing

# In[349]:


model.load_weights(checkpoint_path)


# In[195]:


history=np.load('ENEL645_FinalProject_FishClassification/history.npy', allow_pickle='TRUE').item()
print("Best training results:\n", history)


# In[197]:


results = model.evaluate(test_images, verbose=1)

print("Categorical Cross Entropy: {:.5f}".format(results[0]))
print("Test Accuracy: {:.2f}%".format(results[1] * 100))


# In[ ]:


print("\n************************ COMPLETED TESTING ************************")

