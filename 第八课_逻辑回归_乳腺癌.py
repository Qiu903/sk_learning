# ============================================
# 第八课：逻辑回归 —— "分类版的线性回归"（预测乳腺肿块良/恶性）
# 环境：skl_env
# ============================================

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ① 数据：569 个乳腺肿块样本，30 个特征(半径/纹理/周长/面积等)，标签 0=恶性 1=良性
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# ② 切分：70% 训练，30% 测试
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ③ 创建模型：内部就是"线性回归算分数 → sigmoid 把分数压成 0~1 的概率"
model = LogisticRegression(max_iter=5000)

# ④ 训练
model.fit(X_train, y_train)

# ⑤ 机器学到的权重：和线性回归同一个思维，正=数值越大越推向"恶性"
print("权重 w（挑前 5 个看）：")
for name, w in zip(cancer.feature_names[:5], model.coef_[0][:5]):
    print(f"  {name:>20}: {w:8.2f}")
print(f"截距 b: {model.intercept_[0]:.2f}")

# ⑥ 逻辑回归的招牌能力：不只说"是什么"，还给出"有多确定"（概率）
proba = model.predict_proba(X_test)
print("\n前 5 个测试样本——机器的信心分：")
for i in range(5):
    p_bad, p_good = proba[i]
    real = "恶性" if y_test[i] == 0 else "良性"
    print(f"  真实={real}   恶性概率 {p_bad:5.1%}   良性概率 {p_good:5.1%}")

# ⑦ 默认规则：谁的概率超过 50% 就押谁
predictions = model.predict(X_test)
accuracy = (predictions == y_test).mean()
print(f"\n测试集准确率：{accuracy:.2%}")

# ⑧ 找出被判错的样本，看看机器"看走眼"的样子
wrong = 0
for i in range(len(y_test)):
    if predictions[i] != y_test[i]:
        p_bad, p_good = proba[i]
        real = "恶性" if y_test[i] == 0 else "良性"
        print(f"看走眼：真实{real}，机器按 恶性{p_bad:.1%}/良性{p_good:.1%} 押了{'恶性' if predictions[i] == 0 else '良性'}")
        wrong += 1
print(f"共 {wrong} 个判错")