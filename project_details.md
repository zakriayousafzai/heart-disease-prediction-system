My final year project: Heart Disease Prediction using Deep Learning with Multi-Class Classification
Introduction: Heart disease is one of the leading causes of death worldwide. Early and accurate detection can save millions of lives by enabling timely treatment. Traditional diagnosis relies on manual evaluation of clinical parameters, which can be time-consuming and prone to human error. This project proposes a Deep Learning–based heart disease prediction system that analyses patient health records and predicts the likelihood of developing heart disease. The system will utilize patient data such as age, gender, blood pressure, cholesterol level, blood sugar, ECG results, and other clinical features. By applying Artificial Neural Networks (ANN), Logistic Regression, and Random Forest Classifier, the system will classify patients into different risk categories (Low, Medium, High). The outcome will be a user-friendly web application where healthcare professionals can input patient data and instantly receive predictions with risk probabilities.
Functional Requirements:
• Input patient health data (age, sex, cholesterol, blood pressure, blood sugar, ECG, etc.).
• Preprocess data (handle missing values, normalization, categorical encoding).
• Use Dropout and Regularization techniques to reduce overfitting.
• Train a Feedforward Artificial Neural Network (ANN) for heart disease classification.
• Implement Logistic Regression as a baseline model for comparison.
• Apply Random Forest Classifier to evaluate performance against ANN.
• Provide multi-class output: Low Risk, Medium Risk, High Risk.
• Display probability scores for prediction.
• Generate visualizations of health risk trends (graphs, charts).
• Store patient history and past predictions in a database.
• Provide a Graphical User Interface (GUI) on a web-based platform, allowing users to easily input health parameters, view predictions, and download reports.
Tools:

1. Programming Language: Python
2. Libraries/Frameworks: TensorFlow, Keras, PyTorch, Scikit-learn
3. Database: MySQL / PostgreSQL
4. Visualization: Matplotlib, Seaborn, Plotly
5. Development Environment: Google Colab / Jupyter Notebook / VS Code
Dataset
6. Kaggle Heart Disease Dataset https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction