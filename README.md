# VideoNetClassification

## Table of Contents
- [Introduction](#introduction)
- [Collaborators](#collaborators)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data Loading](#data-loading)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [License](#license)

## Introduction
VideoNetClassification is a project focused on classifying videos using a Recurrent Neural Network (RNN). This project leverages TensorFlow and TensorFlow Hub for model building and training. It uses the UCF101 dataset, a popular action recognition dataset.

## Collaborators
- Yahia Ehab
- Mariam Amr
- Mohamed Khaled

## Installation
To install the required dependencies, you can use the following commands:
```sh
pip install imageio
pip install opencv-python
pip install git+https://github.com/tensorflow/docs
```

## Usage
1. Clone the repository.
2. Install the required dependencies as mentioned above.
3. Run the `notebook.ipynb` notebook to train and evaluate the model.

## Project Structure
- `notebook.ipynb`: The main Jupyter Notebook containing all the code for data loading, preprocessing, model training, and evaluation.

## Data Loading
The UCF101 dataset is used in this project. The dataset is loaded and preprocessed using various helper functions. Videos are downloaded and their frames are extracted and resized.

## Model Training
An RNN model is created using Keras. The training process involves:
1. Extracting features from video frames using a pre-trained CNN.
2. Training the RNN model with the extracted features.

### Steps:
1. **Feature Extraction**:
   ```python
   base_model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, pooling='avg')
   video_features = []
   for video in video_frames:
       video_features.append(base_model.predict(video))
   ```

2. **Data Preparation**:
   ```python
   split = int(0.8 * len(video_features))
   train, test = video_features[:split], video_features[split:]
   train_labels, test_labels = labels[:split], labels[split:]
   ```

3. **Label Encoding**:
   ```python
   from tensorflow.keras.utils import to_categorical
   from sklearn.preprocessing import LabelEncoder

   label_encoder = LabelEncoder()
   train_labels_encoded = label_encoder.fit_transform(train_labels)
   train_labels_onehot = to_categorical(train_labels_encoded, num_of_classes)

   test_labels_encoded = label_encoder.fit_transform(test_labels)
   test_labels_onehot = to_categorical(test_labels_encoded, num_of_classes)
   ```

4. **Model Creation and Training**:
   ```python
   from keras.models import Sequential
   from keras.layers import SimpleRNN, Dense

   rnn_model = Sequential()
   rnn_model.add(SimpleRNN(50, input_shape=(sequence_length, 2048)))
   rnn_model.add(Dense(num_of_classes, activation="softmax"))

   rnn_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

   rnn_model.fit(x=train, y=train_labels_onehot, epochs=15, batch_size=64, validation_split=0.2)
   ```

## Evaluation
The trained model is evaluated using the test dataset:
```python
evaluation = rnn_model.evaluate(x=test, y=test_labels_onehot)
print("Evaluation results:", evaluation)
```

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

---
