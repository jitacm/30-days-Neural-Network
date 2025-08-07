import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

def main():
    # Load data
    df = pd.read_csv('smart_health_tracker_data.csv')
    
    # Fill missing values
    df.fillna(df.mode().iloc[0], inplace=True)
    
    # Categorical columns to encode
    categorical_cols = ['Gender', 'Daily_Activity_Type', 'Mood']  # Adjust as per your dataset columns
    
    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=categorical_cols)
    
    # Features and target
    X = df.drop('Sleep_Quality', axis=1)  # Replace with your target column name
    y = df['Sleep_Quality']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train model
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = lr_model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix - Logistic Regression')
    plt.show()
    
    # ROC curve plot
    y_pred_prob = lr_model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    plt.plot(fpr, tpr, label='Logistic Regression')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
