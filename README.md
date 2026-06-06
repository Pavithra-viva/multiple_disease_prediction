Multiple Disease Prediction System
📌 Overview

The Multiple Disease Prediction System is a machine learning-based web application that predicts the likelihood of various diseases such as Kidney Disease, Liver Disease, and Parkinson’s Disease.

The system helps in early detection by analyzing patient data and providing quick predictions along with probability scores and visual insights.

🎯 Objective
Assist in early detection of diseases
Improve decision-making using ML models
Reduce diagnostic time by providing quick predictions
Provide a simple and user-friendly interface
🏗️ System Architecture
🔹 Frontend
Built using Streamlit
Interactive UI for user input and results display
🔹 Backend
Python-based processing
Handles data preprocessing and model inference
🔹 Machine Learning Models
Logistic Regression
Random Forest (optional improvement)
⚙️ Features
✅ Multi-disease prediction (Kidney, Liver, Parkinson’s)
✅ User-friendly interface
✅ Real-time prediction results
✅ Probability-based output
✅ Visualization using graphs
✅ Sidebar navigation for disease selection
🔄 Workflow
User Input
Enter medical details such as age, blood pressure, glucose, etc.
Data Preprocessing
Handle missing values
Convert categorical data
Scale numerical features
Model Prediction
Input is passed to trained ML model
Model predicts disease probability
Output
Displays prediction result
Shows probability score
Visualizes confidence using graphs
🛠️ Technologies Used
Programming Language: Python
Libraries:
Pandas
NumPy
Scikit-learn
Matplotlib
Frontend: Streamlit
📊 Model Evaluation
Classification Metrics:
Accuracy
Precision
Recall
F1-Score
Confusion Matrix
📁 Project Structure
Multiple disease/
│
├── kidney_model.pkl
├── liver_model.pkl
├── parkinsons_model.pkl
├── scaler.pkl
├── scaler_liver.pkl
├── scaler_parkinsons.pkl
│
├── kidney_disease.csv
├── indian_liver_patient.csv
├── parkinsons.csv
│
└── app.py
▶️ How to Run the Project
1. Install Dependencies
pip install pandas numpy scikit-learn matplotlib streamlit
2. Run the Application
streamlit run app.py
3. Open in Browser
http://localhost:8501
