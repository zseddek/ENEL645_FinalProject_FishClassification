#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf 
import os
from random import shuffle
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


# In[2]:


os.chdir('/root/fish_class')
data_directory = os.getcwd()
print(data_directory)
get_ipython().system('ls')


# 1. Loading Data and Preprocessing

# In[3]:


# 20% Validation Set, 80% Training Set
# Input data is balanced across the number of fish classes
train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input, # Preprocessing function
    validation_split=0.2 
)

test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input # Preprocessing function
)


# In[4]:


# Shuffle = True randomly selects images from a random directory/class to meet the streaming batch size and send to the model for training
# Instead of flow_from_directory, the following article: https://www.kaggle.com/pavfedotov/fish-classifier-efficientnet-acc-100, uses flow_from_dataframe
# which simply contains the list of all image paths in directory and the corresponding class label, we can pivot to this method if it is difficult
# to visualize results, but the method below is actually more efficient...
train_images = train_generator.flow_from_directory(
    directory= './Data/Train_Val',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=128, # Will stream 64 images at a time to the model, helps to reduce RAM requirements, fine tune
    shuffle=True,
    seed=42,
    subset='training'
)

val_images = train_generator.flow_from_directory(
    directory= './Data/Train_Val',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=128,
    shuffle=True,
    seed=42,
    subset='validation' # Will only take 20% of the total data as the validation data
)

test_images = test_generator.flow_from_directory(
    directory= './Data/Test',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=128,
    shuffle=True,
    seed=42
)


# In[5]:


print("Training image shape:", train_images.image_shape)
print("Validation image shape:", val_images.image_shape)
print("Test image shape:", test_images.image_shape)


# In[6]:


train_images.class_indices


# In[7]:


val_images.class_indices


# In[8]:


test_images.class_indices


# In[9]:


import tensorflow.keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


# 2. Defining VGG16 (CNN) Architecture

# In[10]:


model = Sequential()
model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Flatten())
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=9, activation="softmax")) # Match number of fish classes/species (was 2 before - caused graph error)

optimizer = Adam(learning_rate=0.01) # Fine tune
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()


# 3. Defining Schedulers and Callbacks

# In[11]:


early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 5) # Fine tune
checkpoint_path = "training_1/cp.ckpt"
monitor = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss',
                                             verbose=1,save_best_only=True,
                                             save_weights_only=True,
                                             mode='min') # Only saves the best model (so far) in terms of min validation loss
# Learning rate schedule
def scheduler(epoch, lr): # Fine tune
    if epoch%10 == 0: # Occurs on 10, 20, 30, 40, 50
        lr = lr/2 
    return lr

lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler,verbose = 1)
callbacks = [early_stop, monitor, lr_schedule]


# 4. Training Model

# In[15]:


model.fit( # Batch size dictated by image generators (128 in base train)
    train_images, 
    validation_data=val_images, 
    epochs=50, # Fine tune
    callbacks=callbacks
)


# In[ ]:


model.save('Model')


# In[12]:


print("\n************************ COMPLETED ************************")


# ### Fine tuning will be performed using the best model weights saved as a checkpoint (model restored from this checkpoint)
