import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# # Define sigmoid function
# def sigmoid(x):
#     x = np.clip(x, -500, 500)
#     return 1 / (1 + np.exp(-x))

# # Define custom Logistic Regression class
# class LogisticRegression():
#     '''
#     Logistic Regression classifier.

#     Parameters:
#     -----------
#     lr : float, default=0.001
#         Learning rate.
#     n_iters : int, default=1000
#         Number of iterations.

#     Attributes:
#     -----------
#     weights : numpy.ndarray, shape (n_features,)
#         Weights after fitting the model.
#     bias : float
#         Bias term after fitting the model.

#     Methods:
#     --------
#     fit(X, y):
#         Fit the logistic regression model.
#     predict(X):
#         Predict binary class labels for the input data.
#     '''

#     def __init__(self, lr=0.001, n_iters=1000):
#         '''
#         Initialize the LogisticRegression model with given learning rate and number of iterations.

#         Parameters:
#         -----------
#         lr : float, optional, default=0.001
#             Learning rate for gradient descent optimization.
#         n_iters : int, optional, default=1000
#             Number of iterations for the gradient descent algorithm.
#         '''
#         self.lr = lr
#         self.n_iters = n_iters
#         self.weights = None
#         self.bias = None

#     def fit(self, X, y):
#         '''
#         Fit the logistic regression model to the training data.

#         Parameters:
#         -----------
#         X : numpy.ndarray, shape (n_samples, n_features)
#             Training data.
#         y : numpy.ndarray, shape (n_samples,)
#             Target values.

#         Returns:
#         --------
#         self : object
#             Fitted estimator.
#         '''
#         n_samples, n_features = X.shape
#         self.weights = np.zeros(n_features)
#         self.bias = 0

#         for _ in range(self.n_iters):
#             linear_pred = np.dot(X, self.weights) + self.bias
#             predictions = sigmoid(linear_pred)

#             dw = (1/n_samples) * np.dot(X.T, (predictions - y))
#             db = (1/n_samples) * np.sum(predictions - y)

#             self.weights -= self.lr * dw
#             self.bias -= self.lr * db
#         print(self.weights,'/n',self.bias)
#     def predict(self, X):
#                 '''
#         Predict binary class labels for the input data.

#         Parameters:
#         -----------
#         X : numpy.ndarray, shape (n_samples, n_features)
#             Input data.

#         Returns:
#         --------
#         class_pred : list of int
#             Predicted binary class labels (0 or 1).
#         '''
#                 print(self.weights,self.bias)
#                 linear_pred = np.dot(X, self.weights) + self.bias
#                 y_pred = sigmoid(linear_pred)
#                 class_pred = [0 if y<=0.5 else 1 for y in y_pred]
#                 return class_pred
#     def individual_predict(self, x):
#         '''
#         Predict binary class label for a single data point.

#         Parameters:
#         -----------
#         x : numpy.ndarray, shape (n_features,)
#             Single data point.

#         Returns:
#         --------
#         class_pred : int
#             Predicted binary class label (0 or 1).
#         '''
#         # print(self.weights,self.bias)
#         linear_pred = np.dot(x, self.weights) + self.bias
#         y_pred = sigmoid(linear_pred)
#         print('ERROR!')
#         class_pred = 0 if y_pred <= 0.5 else 1
#         # print(class_pred)
#         return class_pred

class GaussianNB:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        self.mean = np.zeros((n_classes, n_features))
        self.var = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)

        for idx, c in enumerate(self.classes):
            X_c = X[y==c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0)
            self.priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        n_samples, n_features = X.shape
        predictions = np.zeros(n_samples)

        for i in range(n_samples):
            posteriors = []
            for idx, c in enumerate(self.classes):
                prior = np.log(self.priors[idx])
                posterior = np.sum(np.log(self.pdf(idx, X[i])))
                posterior = prior + posterior
                posteriors.append(posterior)
            predictions[i] = self.classes[np.argmax(posteriors)]

        return predictions

    def pdf(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(-((x-mean)**2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
    def predict_row(self, row):
        posteriors = []
        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            posterior = np.sum(np.log(self.pdf(idx, row)))
            posterior = prior + posterior
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]





def main():
    st.title('Diabetes Predictor using Gaussian Naive Bayes (Classifier)')

    # Input fields
    gender = st.selectbox('Gender', ['Male', 'Female'])
    age = st.number_input('Age', min_value=20, max_value=79, value=30, step=1)
    urea = st.number_input('Urea', min_value=0.5, max_value=38.9, value=5.0, step=0.1)
    cr = st.number_input('Cr', min_value=6.0, max_value=800.0, value=20.0, step=0.1)
    hba1c = st.number_input('HbA1c', min_value=0.9, max_value=16.0, value=5.5, step=0.1)
    chol = st.number_input('Chol', min_value=0.0, max_value=10.3, value=9.0, step=1.0)
    tg = st.number_input('TG', min_value=0.3, max_value=13.8, value=1.0, step=1.0)
    hdl = st.number_input('HDL', min_value=0.2, max_value=9.9, value=5.0, step=1.0)
    ldl = st.number_input('LDL', min_value=0.3, max_value=9.9, value=1.0, step=1.0)
    vldl = st.number_input('VLDL', min_value=0.1, max_value=35.0, value=3.0, step=1.0)
    bmi = st.number_input('BMI', min_value=19.0, max_value=47.75, value=25.0, step=0.1)

    model = GaussianNB()

    # Predict button
    if st.button('Predict'):
        # Encode gender
        train_accuracy = accuracy_score(y_pred_train,Y_test)
        gender_encoded = 1 if gender == 'Female' else 0
        # Make prediction
        user_input = np.array([gender_encoded, age, urea, cr, hba1c, chol, tg, hdl, ldl, vldl, bmi])
        # print(user_input)
        prediction = model.predict_row(user_input)
        

        # Display prediction result
        # print(prediction)
        if prediction == 1:
            st.write('The predicted class is: Positive')
        elif prediction == 0:
            st.write('The predicted class is: Negative')

if __name__ == '__main__':
    main()
