# Machine Learning Neural Network Program that can  recognize the shape of a hand forming scissors, stone, or paper

This project is a deep learning neural network application that employs TensorFlow to classify hand gestures representing scissors, rock, or paper. Here's a breakdown of the project workflow:

1. **Data Preparation**: The dataset containing images of hand gestures for scissors, rock, and paper is downloaded from a provided URL using `wget`. The dataset is then extracted and organized into train and validation directories using the `splitfolders` library to facilitate training and evaluation.

2. **Image Augmentation**: Image augmentation techniques such as rotation, horizontal and vertical flips, brightness adjustment, shear, shift, and zoom are applied using the `ImageDataGenerator` class from TensorFlow's Keras API. This process helps increase the variability of the training data, which can improve the model's generalization ability.

3. **Model Architecture**: The neural network model is defined using the Sequential API from TensorFlow Keras. It consists of five convolutional layers with max-pooling followed by three dense (fully connected) layers. The convolutional layers extract features from the input images, and the dense layers classify those features.

4. **Model Compilation**: The model is compiled with the Stochastic Gradient Descent (SGD) optimizer and categorical cross-entropy loss function, which is suitable for multi-class classification problems.

5. **Model Training**: The model is trained using the `fit()` method with the specified number of epochs and batch size. During training, both training and validation accuracies are monitored to assess the model's performance.

6. **Model Evaluation**: The trained model achieves high accuracy on both the training and validation datasets, indicating its effectiveness in classifying hand gestures.

7. **Prediction**: Finally, the model is used to predict the class (scissors, rock, or paper) of new hand gesture images uploaded by the user. The uploaded images are processed, and the predicted class is displayed based on the highest probability among the output classes.

Overall, this project demonstrates the implementation of a convolutional neural network (CNN) for image classification, specifically for recognizing hand gestures of scissors, rock, and paper. It showcases the end-to-end workflow of building, training, evaluating, and deploying a deep learning model using TensorFlow.
