# Handwriting-Classification-Using-MNIST-dataset
Developed a Handwriting Recognition system using Convolutional Neural Networks (CNN) with the MNIST dataset. The project workflow involved the following steps:

1. **Data Loading and Preprocessing**: The MNIST dataset, consisting of 28x28 grayscale images of handwritten digits from 0 to 9, was loaded using TensorFlow's built-in dataset loader. The images were preprocessed by normalizing pixel values to a range between 0 and 1.

2. **Model Architecture**: A CNN architecture was designed and implemented using TensorFlow's Keras API. The model comprised convolutional layers for feature extraction, followed by max-pooling layers to reduce spatial dimensions. Dropout layers were added for regularization to prevent overfitting. The final layers included fully connected dense layers with softmax activation for multi-class classification.

3. **Training**: The model was trained on the training set using the Adam optimizer and categorical cross-entropy loss function. Training iterations were conducted over multiple epochs to optimize model parameters and improve accuracy.

4. **Evaluation**: After training, the model's performance was evaluated on the test set to assess its ability to generalize to unseen data. Metrics such as accuracy were used to measure the model's performance in classifying handwritten digits.

5. **Prediction**: Using the trained model, predictions were made on new handwritten digit images to classify them into one of the ten digit classes (0 to 9). The model's output provided probabilities for each class, enabling confident digit recognition.

Overall, the project showcased proficiency in CNN architectures, data preprocessing, model training, evaluation, and prediction, demonstrating the effectiveness of deep learning techniques in handwriting recognition tasks.
