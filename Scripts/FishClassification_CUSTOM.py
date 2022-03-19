#!/usr/bin/env python
# coding: utf-8

# In[31]:


import tensorflow as tf 
import os
from random import shuffle
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


# In[32]:


os.chdir('/root/fish_class')
working_directory = os.getcwd()
print("working directory:", working_directory)


# 1. Loading Data and Preprocessing

# In[48]:


# 20% Validation Set, 80% Training Set
# Input data is balanced across the number of fish classes
# Following generators and preprocessing used to help generalize the model
train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255, # min-max Normalization, shifting pixel value to [0,1], max 255 to max of 1 (domain shift)
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=40,
    shear_range=0.2, 
    zoom_range=0.2,
    validation_split=0.2
)

test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255 # Apply same normalization, not performing other preprocessing steps
)

# train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
#     preprocessing_function=tf.keras.applications.vgg16.preprocess_input, # Preprocessing function
#     validation_split=0.2 
# )

# test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
#     preprocessing_function=tf.keras.applications.vgg16.preprocess_input # Preprocessing function
# )


# In[52]:


# Shuffle = True randomly selects images from a random directory/class to meet the streaming batch size and send to the model for training
# Instead of flow_from_directory, the following article: https://www.kaggle.com/pavfedotov/fish-classifier-efficientnet-acc-100, uses flow_from_dataframe
# which simply contains the list of all image paths in directory and the corresponding class label, we can pivot to this method if it is difficult
# to visualize results, but the method below is actually more efficient...
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


# In[50]:


print("Training image shape:", train_images.image_shape)
print("Validation image shape:", val_images.image_shape)
print("Test image shape:", test_images.image_shape)


# In[51]:


train_images.class_indices


# In[47]:


val_images.class_indices


# In[38]:


test_images.class_indices


# In[39]:


import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


# 2. Defining VGG16 (CNN) Architecture

# In[40]:


# VGG16 from-scratch implementation
# input = Input(shape =(224,224,3))
# x = Conv2D (filters =64, kernel_size =3, padding ='same', activation='relu')(input)
# x = Conv2D (filters =64, kernel_size =3, padding ='same', activation='relu')(x)
# x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# x = Conv2D (filters =128, kernel_size =3, padding ='same', activation='relu')(x)
# x = Conv2D (filters =128, kernel_size =3, padding ='same', activation='relu')(x)
# x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# x = Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x) 
# x = Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x) 
# x = Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x) 
# x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
# x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
# x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
# x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
# x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
# x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
# x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# x = Flatten()(x) 
# x = Dense(units = 4096, activation ='relu')(x) 
# x = Dense(units = 4096, activation ='relu')(x) 
# output = Dense(units = 9, activation ='softmax')(x)
# model = Model (inputs=input, outputs =output)
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # Uses default LR or 0.001

# Novel model
input = Input(shape =(224,224,3))
l1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input)
l2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(l1)
l3 = BatchNormalization()(l2)
l4 = MaxPool2D(2,2)(l3)

l5 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(l4)
l6 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(l5)
l7 = BatchNormalization()(l6)
l8 = MaxPool2D(2,2)(l7)

l9 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(l8)
l10 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(l9)
l11 = BatchNormalization()(l10)
l12 = MaxPool2D(2,2)(l11)

l13 = Flatten()(l12)
l14 = Dense(350, activation='relu')(l13)
l15 = Dropout(0.1)(l14)
l16 = Dense(350, activation='relu')(l15)
l17 = Dropout(0.2)(l16)
output = Dense(9, activation='softmax')(l17)
model = Model (inputs=input, outputs=output)
model.compile(
    optimizer='adam', # Uses default LR of 0.001
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


# 3. Defining Schedulers and Callbacks

# In[41]:


# We should be monitoring validation loss calculation not validation accuracy in our callbacks
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 10) # Fine tune
checkpoint_path = "ENEL645_FinalProject_FishClassification/training_1/cp.ckpt"
monitor = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss',
                                             verbose=1,save_best_only=True,
                                             save_weights_only=True,
                                             mode='min') # Only saves the best model (so far) in terms of min validation loss
# # Learning rate schedule
# def scheduler(epoch, lr): # Fine tune
#     if epoch%10 == 0: # Occurs on 10, 20, 30, 40, 50
#         lr = lr/2 
#     return lr

# lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler,verbose = 1)
lr_schedule = ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.1, patience=3, min_lr=0.000001, verbose=1)
callbacks = [early_stop, monitor, lr_schedule]


# 4. Training Model

# In[42]:


model.fit(
    train_images, 
    validation_data=val_images, 
    epochs=50, # Fine tune
    callbacks=callbacks
)


# In[14]:


model.save('ENEL645_FinalProject_FishClassification/Model')


# In[ ]:


print("\n************************ COMPLETED TRAINING ************************")


# 5. Loading Best Model and Testing

# In[19]:


model.load_weights(checkpoint_path)


# In[20]:


results = model.evaluate(test_images, verbose=1)

print("Test Loss: {:.5f}".format(results[0]))
print("Test Accuracy: {:.2f}%".format(results[1] * 100))


# In[ ]:


print("\n************************ COMPLETED TESTING ************************")

