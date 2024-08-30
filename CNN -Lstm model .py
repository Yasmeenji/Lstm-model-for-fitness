#!/usr/bin/env python
# coding: utf-8

# ## Importing all necessary libraries

# In[1]:


import os               # Operating system functionality
import cv2              # OpenCV for image and video processing            
import math             # Mathematical functions
import random           # Random number generation
import numpy as np      # Numerical operations using NumPy
import datetime as dt   # Date and time handling
import tensorflow as tf # TensorFlow deep learning framework
from collections import deque 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D  # Add more layers as needed
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv3D


# In[2]:


seed_constant=27
np.random.seed(seed_constant)
random.seed(seed_constant)
tf.random.set_seed(seed_constant)


# In[3]:


# Define the base directory path
video_paths= 'C:\\Users\\yasme\\Music\\Documents\\squat dataset'

# Get the names of all classes/categories in the dataset
all_classes_names = os.listdir(video_paths)

# Generate a list of 20 random values between 0 and len(all_classes_names) - 1
random_range = random.sample(range(len(all_classes_names)), 2)

# Create a figure with a specific size
plt.figure(figsize=(20, 20))

# Iterate through all the generated random values
for counter, random_index in enumerate(random_range, 1):
    # Retrieve a class name using the random index
    selected_class_name = all_classes_names[random_index]
    
    # Retrieve the list of all the video files present in the randomly selected class directory
    selected_class_dir = os.path.join(video_paths, selected_class_name)
    video_files_names_list = os.listdir(selected_class_dir)
    
    # Randomly select a video file from the list retrieved from the randomly selected class directory
    selected_video_file_name = random.choice(video_files_names_list)
    
    # Initialize a video capture object to read from the video file
    video_reader = cv2.VideoCapture(os.path.join(selected_class_dir, selected_video_file_name))
    
    # Read the first frame of the video file
    success, bgr_frame = video_reader.read()
    
    # Release the video capture object
    video_reader.release()
    
    if not success:
        continue  # Skip this video if the frame couldn't be read
    
    # Convert the frame from BGR into RGB format
    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    
    # Write the class name on the video frame
    cv2.putText(rgb_frame, selected_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Display the frame
    plt.subplot(5, 4, counter)  # 5 rows, 4 columns layout for 20 subplots
    plt.imshow(rgb_frame)
    plt.axis('off')
    plt.title(selected_video_file_name)  # Optionally, show the video file name as title

# Adjust layout and display the figure
plt.tight_layout()
plt.show()


# In[4]:


## Preprocess video datset
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64 ##resize height and width of video
SEQUENCE_LENGTH = 20 ## NO OF Frames of video
DATASET_DIR ="C:\\Users\\yasme\\Music\\Documents\\squat dataset"
CLASS_LIST = ["deadlift","push-Up", "squat"]


# In[5]:


## Frame and preprocess it
def frames_extraction(video_paths, SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH):
    frame_list = []
    
    # Open the video file
    video_reader = cv2.VideoCapture(video_paths)
    
    # Get the total number of frames in the video
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Determine the interval to skip frames
    skip_frames_window = max(int(video_frames_count / SEQUENCE_LENGTH), 1)
    
    for frame_counter in range(SEQUENCE_LENGTH):
        # Set the frame position
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        
        # Read the frame
        success, frame = video_reader.read()
        
        if not success:
            break
        
        # Resize the frame
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))  # Corrected width and height order
        
        # Normalize the frame
        normalized_frame = resized_frame / 255.0
        
        # Append the normalized frame to the list
        frame_list.append(normalized_frame)
    
    video_reader.release()
    return frame_list


# In[6]:


def create_dataset():
    features = []
    labels = []
    video_files_paths = []
    
    for class_index, class_name in enumerate(CLASS_LIST):
        print(f'Extracting data of class: {class_name}')
        
        files_list = os.listdir(os.path.join(DATASET_DIR, class_name))
        
        for file_name in files_list:
            video_file_path = os.path.join(DATASET_DIR, class_name, file_name)
            frames = frames_extraction(video_file_path, SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH)
            
            if len(frames) == SEQUENCE_LENGTH:
                features.append(frames)
                labels.append(class_index)
                video_files_paths.append(video_file_path)
    
    features = np.asarray(features)
    labels = np.asarray(labels)
    
    return features, labels, video_files_paths


# In[7]:


features, labels, video_files_paths = create_dataset()


# In[8]:


one_hot_encoded_labels = to_categorical(labels)


# In[9]:


# Print shapes of feature and label arrays
print(f"Features shape: {features.shape}")
print(f"Labels shape: {labels.shape}")

# Ensure the number of samples is the same
assert features.shape[0] == labels.shape[0], "Number of samples in features and labels do not match"


# In[10]:


features_train, features_test, one_hot_encoded_labels_train, one_hot_encoded_labels_test= train_test_split(features, one_hot_encoded_labels,
                                                                          test_size=0.2, shuffle=True,
                                                                          random_state = seed_constant)


# In[11]:


# Print shapes of feature and label arrays
print(f"Features shape: {features_train.shape}")
print(f"Labels shape: {one_hot_encoded_labels_train.shape}")

# Ensure the number of samples is the same
assert features_train.shape[0] == one_hot_encoded_labels_train.shape[0], "Number of samples in features and labels do not match"


# In[19]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, MaxPooling3D, TimeDistributed, Dropout, Flatten, Dense

def create_convlstm_model(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, CLASS_LIST):
    model = Sequential()

    # Add a ConvLSTM2D layer
    model.add(ConvLSTM2D(filters=4, kernel_size=(3, 3), activation='tanh', data_format='channels_last',
                         recurrent_dropout=0.2, input_shape=(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3), return_sequences=True))

    model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    model.add(TimeDistributed(Dropout(0.2)))

    model.add(ConvLSTM2D(filters=8, kernel_size=(3, 3), activation='tanh', data_format='channels_last',
                         recurrent_dropout=0.2, return_sequences=True))

    model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))

    
    model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), activation='tanh', data_format='channels_last',
                         recurrent_dropout=0.2, return_sequences=True))

    model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    model.add(TimeDistributed(Dropout(0.2)))
    
    
    
    model.add(ConvLSTM2D(filters=16, kernel_size=(3, 3), activation='tanh', data_format='channels_last',
                         recurrent_dropout=0.2, return_sequences=True))

    model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    model.add(TimeDistributed(Dropout(0.2)))

    model.add(Flatten())

    model.add(Dense(len(CLASS_LIST), activation='softmax'))

    model.summary()

    return model


# In[20]:


# Create the model
convlstm_model = create_convlstm_model(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, CLASS_LIST)

print("Model created successfully!")


# In[21]:


from tensorflow.keras.callbacks import EarlyStopping

# Define the EarlyStopping callback
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)

# Assuming `convlstm_model` is the model you created
convlstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training the model
convlstm_model_training_history = convlstm_model.fit(x = features_train, y = one_hot_encoded_labels_train, epochs=20,
                                   batch_size= 2, shuffle=True, validation_split=0.2,callbacks=[early_stopping_callback]
                                   )


# In[22]:


model_evaluation_history = convlstm_model.evaluate(features_test, one_hot_encoded_labels_test)


# In[23]:


import matplotlib.pyplot as plt

def plot_metric(model_training_history, metric_name_1, metric_name_2, plot_name):
    metric_value_1 = model_training_history.history[metric_name_1]
    metric_value_2 = model_training_history.history[metric_name_2]

    epochs = range(len(metric_value_1))

    plt.plot(epochs, metric_value_1, 'blue', label=metric_name_1)
    plt.plot(epochs, metric_value_2, 'red', label=metric_name_2)

    plt.title(str(plot_name))
   
    plt.legend()


# In[24]:


plot_metric(convlstm_model_training_history, 'loss', 'val_loss', 'Total_loss vs Total validation loss')


# In[25]:


plot_metric(convlstm_model_training_history, 'accuracy', 'val_accuracy', 'Total_accuracy vs Total validation accuracy')


# In[ ]:





# In[ ]:




