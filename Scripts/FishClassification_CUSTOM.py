#!/usr/bin/env python
# coding: utf-8

# In[4]:


import tensorflow as tf 
import os
from random import shuffle
import numpy as np
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


# In[5]:


os.chdir('/root/fish_class')
working_directory = os.getcwd()
print("working directory:", working_directory)


# 1. Loading Data and Preprocessing

# In[6]:


train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255, # min-max Normalization, shifting pixel value to [0,1], max 255 to max of 1 (domain shift)
    validation_split=0.10
)

test_generator = tf.keras.preprocessing.image.ImageDataGenerator(                                                    
    rescale=1./255 # Apply same normalization, not performing other preprocessing steps
)


# In[7]:


train_images = train_generator.flow_from_directory(
    directory= './Data/Train_Val',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=42,
    subset='training'
)

val_images = train_generator.flow_from_directory(
    directory= './Data/Train_Val',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=42,
    subset='validation' # Will only take 20% of the total data as the validation data
)

test_images = test_generator.flow_from_directory(
    directory= './Data/Test',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=False, # We need to be able to generate metrics at the end
    seed=42
)


# In[8]:


print("Training image shape:", train_images.image_shape)
print("Validation image shape:", val_images.image_shape)
print("Test image shape:", test_images.image_shape)


# In[9]:


train_images.class_indices


# In[10]:


val_images.class_indices


# In[11]:


test_images.class_indices


# In[12]:


import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


# 2. Defining VGG16 (CNN) Architecture

# In[13]:


# Novel model - add descriptive layer names?
input = Input(shape =(224,224,3))

l1 = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=0.001))(input)
l2 = MaxPool2D(2,2)(l1)

l3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=0.001))(l2)
l4 = MaxPool2D(2,2)(l3)

l5 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l1(l=0.001))(l4)
l6 = MaxPool2D(2,2)(l5)

l7 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l1(l=0.001))(l6)
l8 = MaxPool2D(2,2)(l7)

l9 = Flatten()(l8)
l10 = Dense(64, activation='relu')(l9)
l11 = Dropout(0.25)(l10)
l12 = Dense(32, activation='relu')(l11)
l13 = Dropout(0.25)(l12)
output = Dense(9, activation='softmax')(l13)
model = Model (inputs=input, outputs=output)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


# 3. Defining Schedulers and Callbacks

# In[14]:


early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 10) # Fine tune
checkpoint_path = "ENEL645_FinalProject_FishClassification/training_2_rof/cp.ckpt"
monitor = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss',
                                             verbose=1,save_best_only=True,
                                             save_weights_only=True,
                                             mode='min') # Only saves the best model (so far) in terms of min validation loss

lr_schedule = ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.1, patience=3, min_lr=0.0000001, verbose=1)
callbacks = [early_stop, monitor, lr_schedule]


# 4. Training Model

# In[15]:


try:
    history = model.fit(
        train_images, 
        validation_data=val_images, 
        epochs=50, # Fine tune
        callbacks=callbacks
    )
except KeyboardInterrupt:
    print("model training terminated\n")


# In[132]:


np.save('ENEL645_FinalProject_FishClassification/history.npy', history.history)


# In[ ]:


model.save('ENEL645_FinalProject_FishClassification/Model_rof')


# In[ ]:


print("\n************************ COMPLETED TRAINING ************************")


# 5. Loading Best Model and Testing

# In[89]:


model.load_weights(checkpoint_path)


# In[133]:


history=np.load('ENEL645_FinalProject_FishClassification/history.npy', allow_pickle='TRUE').item()
print("Best training results:\n", history)


# In[90]:


results = model.evaluate(test_images, verbose=1)

print("Categorical Cross Entropy: {:.5f}".format(results[0]))
print("Test Accuracy: {:.2f}%".format(results[1] * 100))


# In[ ]:


print("\n************************ COMPLETED TESTING ************************")

