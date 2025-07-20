import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('smart_health_tracker_data.csv')

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Daily_Steps'] = df['Daily_Steps'].fillna(df['Daily_Steps'].mean())
df['Resting_Heart_Rate'] = df['Resting_Heart_Rate'].fillna(df['Resting_Heart_Rate'].median())
df['Active_Heart_Rate'] = df['Active_Heart_Rate'].fillna(df['Active_Heart_Rate'].mean())
df['Hours_of_Sleep'] = df['Hours_of_Sleep'].fillna(df['Hours_of_Sleep'].mean())
df['Daily_Calorie_Intake'] = df['Daily_Calorie_Intake'].fillna(df['Daily_Calorie_Intake'].mean())
df['Sleep_Quality'] = df['Sleep_Quality'].fillna(df['Sleep_Quality'].mean())
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Stress_Level'] = df['Stress_Level'].fillna(df['Stress_Level'].mode()[0])
df['Daily_Activity_Type'] = df['Daily_Activity_Type'].fillna(df['Daily_Activity_Type'].mode()[0])
df['Mood'] = df['Mood'].fillna(df['Mood'].mode()[0])

df['Well_Rested'] = (df['Hours_of_Sleep'] >= 7).astype(int)

features = ['Sleep_Quality','Stress_Level','Daily_Calorie_Intake','Active_Heart_Rate','Resting_Heart_Rate']
X = df[features].values
y = df['Well_Rested'].values


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluation metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_prob))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Rested', 'Well Rested'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.2f}", color='darkorange')
plt.plot([0, 1], [0, 1], linestyle='--', color='navy')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()

# Probability distribution
plt.figure(figsize=(6, 4))
sns.histplot(y_prob, bins=30, kde=True, color='green')
plt.title("Predicted Probability Distribution")
plt.xlabel("Probability of Being Well Rested")
plt.ylabel("Frequency")
plt.show()
