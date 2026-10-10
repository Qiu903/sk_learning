# ============================================
# 第七课：线性回归完整流程（预测糖尿病患者的血糖值）
# 环境：skl_env
# ============================================

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ① 数据：442 个病人，10 个特征(年龄/性别/BMI/血压/6项血清指标)，目标是血糖值
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# ② 切分：80% 训练，20% 测试（测试集模型永远看不到）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ③ 创建模型：它的任务就是"找到最好的 w 和 b"
model = LinearRegression()

# ④ 训练：求出让 MSE 最小的 w、b
model.fit(X_train, y_train)

# ⑤ 看看机器学到了什么：10 个特征各有一个权重 w，外加截距 b
print("学到的权重 w（每个特征一个，正=越大越高，负=越大越低）：")
for name, w in zip(diabetes.feature_names, model.coef_):
    print(f"  {name:>12}: {w:8.2f}")
print(f"学到的截距 b: {model.intercept_:.2f}")

# ⑥ 预测：用没见过的测试集考它
y_pred = model.predict(X_test)

# ⑦ 评估：MSE 越小越好，R² 越接近 1 越好
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\n测试集 MSE（平均误差平方）: {mse:.2f}")
print(f"测试集 R²（解释力，0~1）: {r2:.4f}")

# 抽查：前 3 个测试样本的"预测 vs 真实"
print("\n抽查：预测值 vs 真实值")
for i in range(3):
    print(f"  预测 {y_pred[i]:7.2f}   真实 {y_test[i]:7.2f}")