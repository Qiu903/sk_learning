# ============================================
# 第三课：第一个机器学习模型 —— 决策树（鸢尾花分类）
# 环境：skl_env
# ============================================

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# 1. 加载数据：样本(特征) + 标签
iris = load_iris()
X = iris.data        # 特征：花瓣长度、花瓣宽度等 4 个属性
y = iris.target      # 标签：0=山鸢尾 1=变色鸢尾 2=维吉尼亚鸢尾

# 2. 划分：一部分给机器学(训练集)，一部分留着考它(测试集)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. 创建模型（一台"只会打分提问的机器"，暂时先把它当黑盒）
model = DecisionTreeClassifier(max_depth=3)

# 4. 训练：让机器从头到尾做"猜 → 对照答案 → 改正"的循环
model.fit(X_train, y_train)

# 5. 预测：用没见过的测试集考它
predictions = model.predict(X_test)

# 6. 评估：答对的比例
accuracy = (predictions == y_test).mean()
print(f"测试集准确率：{accuracy:.2%}")

print("预测结果：", predictions)
print("真实答案：", y_test)
print("它成功答对了", (predictions == y_test).sum(), "/", len(y_test), "朵")