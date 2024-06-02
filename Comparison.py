import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

# 加载之前保存的模型
model_files = {
    'Decision Tree': 'results/decision_tree_results/decision_tree_model.pkl',
    'Logistic Regression': 'results/logistic_regression_results/lr_model.pkl',
    'Gradient Boosting': 'results/gradient_boosting_results/gb_model.pkl',
    'Random Forest': 'results/random_forest_results/rf_model.pkl'
}

roc_curves = {}
auc_scores = {}

# 重新加载数据和标准化器，编码器
file_path = 'data/train.csv'
data = pd.read_csv(file_path)
data.fillna(data.mean(), inplace=True)
label_encoders = joblib.load('results/decision_tree_results/label_encoders.pkl')
scaler = joblib.load('results/decision_tree_results/scaler.pkl')
for col, le in label_encoders.items():
    data[col] = le.transform(data[col])
num_features = [col for col in data.columns if 'num' in col]
data[num_features] = scaler.transform(data[num_features])
X = data.drop(columns=['id', 'target'])
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 计算ROC曲线和AUC
for model_name, model_file in model_files.items():
    model = joblib.load(model_file)
    y_pred = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred)
    roc_curves[model_name] = (fpr, tpr)
    auc_scores[model_name] = auc_score

# 绘制所有模型的ROC曲线
plt.figure(figsize=(10, 6))
for model_name, (fpr, tpr) in roc_curves.items():
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_scores[model_name]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='best')
plt.grid()
plt.savefig('results/comparison/roc_curve_comparison.png')
plt.show()
