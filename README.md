# Speech_recognition
This is graduation project to recognize speech and made by python.

1. Overview
- This program learns 6 data of 'up, down, on, off, house, stop' existing in Google dataset by using machine learning algorithm.
- Algorithm used for data preprocessing: MFCC
- The algorithm used for learning: CNN
 
2. Development Environment
- Python 3.6.5
- Tensorflow 1.12.0 (with keras in the tensor flow)
- Librosa 0.6.2
- Numpy 1.15.4
- Scipy 1.2.0
- Matplotlib 3.0.2
3. The program process
- Loading audio data
- Data MFCC applied
- Learning CNN
- Model evaluation
- Save the model
 
4. Detailed model description
- MFCC is one of the speech data preprocessing methods and it is an algorithm that uses a harmonic structure. In the present model, data from MFCC is used as learning data by adjusting to 880 size.
- The model structure of CNN is as follows.
Input -> (8,11) -> (16,7) -> (32,5) -> (64,5) -> (128,3) -> (256,1) -> (128,1) -> output
Hidden layer count: 7
Learning rate: 0.001
Drop out rate: 0.5
Batch size: 256
Train data rate: 0.8
Test data rate: 0.2
Activation function: relu
Output activation function: softmax
Number of Lessons: 500
 
5. Model Evaluation
- Test data reference accuracy approx. 87%
