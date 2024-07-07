import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

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

        gender_encoded = 1 if gender == 'Female' else 0
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
