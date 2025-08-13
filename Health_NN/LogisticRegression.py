import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, accuracy_score, classification_report,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import shap

def main():
    # 1️⃣ Load dataset
    df = pd.read_csv("C:\\Users\\Asus\\Clone files\\30-days-Neural-Network\\Health_NN\\smart_health_tracker_data.csv")

    # 2️⃣ Fill missing values
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col].fillna(df[col].mean(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

    # 3️⃣ Create target variable Well_Rested
    df['Well_Rested'] = (df['Hours_of_Sleep'] >= 7).astype(int)

    # 4️⃣ Select relevant features
    features = [
        'Sleep_Quality', 'Stress_Level',
        'Daily_Calorie_Intake', 'Active_Heart_Rate', 'Resting_Heart_Rate'
    ]
    X = df[features]
    y = df['Well_Rested']

    # 5️⃣ Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 6️⃣ Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # 7️⃣ Train Logistic Regression with class_weight balanced
    lr_model = LogisticRegression(max_iter=1000, class_weight='balanced')
    lr_model.fit(X_train, y_train)

    # 8️⃣ Predictions
    y_pred = lr_model.predict(X_test)
    y_pred_prob = lr_model.predict_proba(X_test)[:, 1]

    # 9️⃣ Evaluation metrics
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_pred_prob))

    # 🔟 Confusion Matrix plot
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.title("Confusion Matrix - Logistic Regression")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # 1️⃣1️⃣ ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    plt.plot(fpr, tpr, label='Logistic Regression')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

    # 1️⃣2️⃣ Probability Distribution
    sns.histplot(y_pred_prob, bins=20, kde=True, color="green")
    plt.title("Predicted Probability Distribution")
    plt.xlabel("Probability of Well Rested")
    plt.ylabel("Frequency")
    plt.show()

    # 1️⃣3️⃣ SHAP Feature Importance
    explainer = shap.Explainer(lr_model, X_train)
    shap_values = explainer(X_test)

    shap.summary_plot(shap_values, X_test, feature_names=features)
    shap.summary_plot(shap_values, X_test, feature_names=features, plot_type="bar")


if __name__ == "__main__":
    main()
