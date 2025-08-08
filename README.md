# 📊 Smart Health Tracker - Sleep Well-Rested Prediction 🛌

Welcome to the **Smart Health Tracker** project! This tool is designed to predict whether a user is well-rested based on various health metrics collected from a smart health tracking device. The project combines classical machine learning and modern neural networks to analyze your health data and deliver meaningful insights about your sleep quality.

***

## 🌟 Project Overview

Good sleep is essential for health and productivity. This project evaluates if a person is "well-rested" by predicting sleep quality using daily health features such as heart rate, activity level, and stress. The prediction classifies users into:

- **Well_Rested (1):** Person has slept 7 or more hours.
- **Not Well_Rested (0):** Person has slept fewer than 7 hours.

To accomplish this, the project implements three predictive models:

1. **Logistic Regression:** A traditional statistical model effective for binary classification.
2. **Neural Network:** A deeper learning approach with one hidden layer, capturing complex patterns.
3. **Perceptron:** A simple neural network model for baseline performance.

Each model offers detailed metrics and visualizations to assess prediction accuracy.

***

## ⚙️ Installation Instructions (Step-by-Step)

1. **Clone the repository**

   Open your command line interface (CLI) and run:

   ```
   git clone https://github.com/Ashleesh/30-days-Neural-Network.git
   cd 30-days-Neural-Network
   ```

2. **Check Python Version**

   This project supports **Python 3.8 or later**. Confirm you have the right version by running:

   ```
  CMD : python --version
   ```

   If you need to install or upgrade Python, download it from the official Python website.

3. **Install Required Libraries**

   To run the models and visualization code, install the necessary Python packages:

   ```
  CMD :   pip install pandas scikit-learn matplotlib seaborn torch
   ```

   - `pandas` and `scikit-learn` for data handling and modeling.
   - `matplotlib` and `seaborn` for plotting graphs.
   - `torch` for building and training neural networks.

***

## 🧰 Project Dependencies & Environment

- The project is developed and tested on Python 3.8+.
- For GPU acceleration on neural network training, install the appropriate CUDA-enabled `torch` version from PyTorch's website depending on your system.
- All scripts are compatible with CPU-only execution but training may be slower.
- If you run into package issues, consider using a virtual environment to isolate dependencies.

***

## 📂 Dataset Details

The project uses a CSV file named `smart_health_tracker_data.csv`. This file should be placed in the project root directory beside the scripts.

### Dataset Columns:

- **Age:** User’s age in years.
- **Daily_Steps:** Number of steps taken daily.
- **Resting_Heart_Rate:** Beats per minute (bpm) when at rest.
- **Active_Heart_Rate:** Beats per minute during activity.
- **Hours_of_Sleep:** Total sleep hours (key feature).
- **Daily_Calorie_Intake:** Calories consumed per day.
- **Sleep_Quality:** Subjective or device-measured sleep rating.
- **Stress_Level:** User’s reported stress level.
- **Daily_Activity_Type:** Type of daily physical activity (e.g.walking, running).
- **Gender:** User’s gender.
- **Mood:** User’s mood during the day.

The target variable is derived as:

- **Well_Rested = 1:** If Hours_of_Sleep ≥ 7
- **Well_Rested = 0:** Otherwise

***

## 🚀 How the Project Works - In Depth

1. **Data Loading & Preprocessing**

   - The data is automatically loaded from the CSV file.
   - Missing numerical values are imputed using statistical methods like mean or median.
   - Missing categorical values use the most common category (mode).
   - Categorical variables are encoded to numerical formats as needed (e.g., one-hot encoding for Logistic Regression).
  
2. **Feature Selection**

   Each model uses a tailored set of input features to optimize prediction accuracy based on the algorithm’s strengths.

3. **Model Training**

   - **Logistic Regression:** Uses scikit-learn’s balanced class weights to handle imbalanced data.
   - **Neural Network:** PyTorch-based with:
     - One hidden layer
     - ReLU activation function for learning non-linear patterns
     - Sigmoid output layer for binary classification
     - Binary cross-entropy loss function to optimize the model.
   - **Perceptron:** Simple neural network layer with sigmoid activation and the same loss function.
   
4. **Model Evaluation**

   After training, the models calculate:

   - Accuracy: How often the model predicts correctly.
   - Precision, Recall, and F1-Score: For understanding model performance on each class.
   - ROC-AUC: Measures ability to distinguish between classes.
   - Confusion Matrix: Shows correct and incorrect classifications visually.
   - Visual plots are generated automatically for easy interpretation.

***

## 🎛️ What to Expect When Running the Models

When you execute each model script, expect:

- Output metrics printed to the console describing model performance.
- Graphs such as:
  - Confusion Matrix plot showing true/false positives and negatives.
  - ROC curve plot depicting sensitivity versus false alarm rate.
  - Probability distribution histograms illustrating model confidence.

These help you analyze strengths and weaknesses of each approach.

***

## 💡 How to Run the Project

In your terminal or command prompt, run one of the following commands to train and test each model:

- For Logistic Regression:

  ```
  python LogisticRegression.py
  ```

- For Neural Network:

  ```
  python Neural.py
  ```

- For Perceptron:

  ```
  python Perceptro.py
  ```

Each script loads the dataset, processes the data, trains the model, evaluates it, and then generates performance reports and visuals automatically.

***

## 🔧 Additional Notes & Tips

- Make sure the data file name is exactly `smart_health_tracker_data.csv` and located in the same folder as scripts.
- GPU support in PyTorch is optional but improves speed for bigger datasets.
- Feel free to experiment with hyperparameters (like learning rate and training epochs) inside the Python scripts to improve model accuracy.
- The project can be extended to include new features or models for further experimentation.
- If you encounter any errors related to package import or versions, consider reinstalling packages or using a virtual environment.

