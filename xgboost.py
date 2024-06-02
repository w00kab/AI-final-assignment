import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib
import os
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
import time

# 创建保存结果的文件夹
# Create a folder where you can save the results
result_dir = 'results/gradient_boosting_results'
os.makedirs(result_dir, exist_ok=True)


# 加载数据
# Load data
print("加载数据")
start_time = time.time()
file_path = 'data/train.csv'
data = pd.read_csv(file_path)
print(f"数据加载完毕，耗时 {time.time() - start_time:.2f} 秒")

# 处理缺失值
# Handle missing values
print("处理缺失值...")
data.fillna(data.mean(), inplace=True)

# 编码离散特征
# Encode discrete features
print("编码离散特征")
label_encoders = {}
for col in tqdm(data.columns, desc="编码离散特征"):
    if 'cat' in col:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

# 标准化数值特征
# Normalize numeric features
print("标准化数值特征")
scaler = StandardScaler()
num_features = [col for col in data.columns if 'num' in col]
data[num_features] = scaler.fit_transform(data[num_features])

# 拆分数据集
# Split the dataset
print("拆分数据集")
X = data.drop(columns=['id', 'target'])
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 重新平衡训练数据
# Rebalance training data
print("重新平衡训练数据")
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# 训练梯度提升树模型
# Train a gradient boosting tree model
print("训练梯度提升树模型")
gb_model = GradientBoostingClassifier()
gb_model.fit(X_train_res, y_train_res)
y_pred_gb = gb_model.predict_proba(X_test)[:, 1]
auc_gb = roc_auc_score(y_test, y_pred_gb)

# 保存模型
# Save the model
model_file_gb = os.path.join(result_dir, 'gb_model.pkl')
joblib.dump(gb_model, model_file_gb)

# 输出AUC得分
# Output AUC score
print(f'梯度提升树模型的 AUC: {auc_gb}')

# 生成ROC曲线
# Generate and save ROC curves
fpr_gb, tpr_gb, _ = roc_curve(y_test, y_pred_gb)

plt.figure(figsize=(10, 6))
plt.plot(fpr_gb, tpr_gb, label=f'Gradient Boosting (AUC = {auc_gb:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.grid()
roc_curve_file_gb = os.path.join(result_dir, 'roc_curve_gb.png')
plt.savefig(roc_curve_file_gb)
plt.show()

# 计算混淆矩阵
# Calculate the confusion matrix
cm_gb = confusion_matrix(y_test, gb_model.predict(X_test))

# 显示混淆矩阵
# Show confusion matrix
disp_gb = ConfusionMatrixDisplay(confusion_matrix=cm_gb)
disp_gb.plot()
plt.title('Gradient Boosting Confusion Matrix')
confusion_matrix_file_gb = os.path.join(result_dir, 'confusion_matrix_gb.png')
plt.savefig(confusion_matrix_file_gb)
plt.show()

# 保存预测结果到CSV文件
# Save the prediction results to a CSV file
test_results_gb = pd.DataFrame({
    'id': data.iloc[X_test.index]['id'],
    'y_pred_gb': y_pred_gb,
    'true_label': y_test
})

test_results_gb.to_csv(os.path.join(result_dir, 'test_predictions_gb.csv'), index=False)
print(f"梯度提升树模型预测结果已保存到 {os.path.join(result_dir, 'test_predictions_gb.csv')}")
