🚦 Traffic Signal Detection Using CNN

This project is a Computer Vision application that uses a Convolutional Neural Network (CNN) to detect and classify traffic signal states—Red, Yellow, Green—from images. It helps beginners understand how deep learning can be applied to image classification tasks.

✅ Features

Classifies traffic signals into Red / Yellow / Green

Uses a custom CNN model for image classification

Includes data preprocessing and augmentation

Provides training and evaluation scripts

Predicts signal color from new images

Simple and clear project structure for learning

🛠️ Technologies Used

Python

TensorFlow / Keras

NumPy

OpenCV

Matplotlib

(Streamlit – if you used UI for predictions)

🎯 Use Case

This project demonstrates how machine learning can support:

Autonomous driving systems

Smart traffic monitoring

Traffic automation research

Computer vision learning and experimentation

It’s ideal for students and beginners practicing deep learning + image classification using CNNs.

▶️ How to Run
1. Install the required libraries
pip install -r requirements.txt

2. Prepare your dataset

Structure:

dataset/
 ├── red/
 ├── yellow/
 └── green/

3. Train the model
python train.py

4. Test the model
python test.py

5. Predict using a new image
python predict.py --image sample.jpg
