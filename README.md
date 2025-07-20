
📊 Smart Health Tracker - Sleep Well-Rested Prediction
------------------------------------------------------------------
This project uses Python, scikit-learn, PyTorch, and visualization tools to predict whether a user is well-rested based on various health features from a CSV dataset.

📂 Dataset
----------------------------------------------------------------
Input: smart_health_tracker_data.csv

Contains features like:

Age, Daily_Steps, Resting_Heart_Rate, Active_Heart_Rate

Hours_of_Sleep, Daily_Calorie_Intake, Sleep_Quality

Stress_Level, Daily_Activity_Type, Gender, Mood

Target:
--------------------------------------------------------

Well_Rested = 1 if Hours_of_Sleep >= 7, else 0

✅ What’s Implemented?
--------------------------------------------------------
This repo has 3 parts:
1️⃣ Logistic Regression with scikit-learn
2️⃣ Neural Network (NNet) with PyTorch
3️⃣ Perceptron model with PyTorch

⚙️ How It Works
--------------------------------------------------------
1️⃣ Logistic Regression
Libraries: pandas, scikit-learn, matplotlib, seaborn

Steps:

Load data and fill missing values (mean, median, mode)

Feature selection: ['Sleep_Quality','Stress_Level','Daily_Calorie_Intake','Active_Heart_Rate','Resting_Heart_Rate']

Data scaling with StandardScaler

Train-test split (80/20)

Train LogisticRegression with class_weight='balanced'

Evaluate using:

Accuracy, Precision, Recall, F1, ROC AUC

Confusion Matrix

ROC Curve

Probability Distribution

2️⃣ Neural Network (NNet) with PyTorch
Libraries: pandas, PyTorch, scikit-learn, matplotlib, seaborn

Steps:

Load and clean data

Feature selection: ['Resting_Heart_Rate','Sleep_Quality','Stress_Level']

Scale and split data

Convert data to PyTorch tensors

Define custom NNet:

3 -> 32 -> 16 -> 1 layers with ReLU and Sigmoid

Train with BCELoss and Adam

Evaluate with:

Accuracy, Precision, Recall, F1, ROC AUC

Confusion Matrix

ROC Curve

Probability Distribution

3️⃣ Perceptron with PyTorch
Libraries: pandas, PyTorch, scikit-learn, matplotlib, seaborn

Steps:

Same preprocessing as Logistic Regression

Feature selection: same 5 features

Convert to tensors

Define simple Perceptron:

5 -> 1 linear layer + Sigmoid

Train with BCELoss and Adam

Evaluate with:

Accuracy, Precision, Recall, F1, ROC AUC

Confusion Matrix

ROC Curve

Probability Distribution

📈 Plots & Evaluation
-----------------------------------------------------------
Each model outputs:

Confusion Matrix: Visual classification performance

ROC Curve: Model’s trade-off between TPR and FPR

Probability Histogram: How confident the model is

🚀 How to Run
---------------------------------------------------------------
Clone this repo:

bash
Copy
Edit
git clone (https://github.com/Apex-ace/Health_NN)
cd YOUR_REPO_NAME
Install dependencies:

bash
Copy
Edit
pip install pandas scikit-learn matplotlib seaborn torch
Add your smart_health_tracker_data.csv to the project folder.

Run each script:

🧑‍💻 Author
--------------------------------------------------------------------
JIT ACM Student Chapter

📜 License
---------------------------------------------------------------------
This project is open source and free to use for educational purposes
