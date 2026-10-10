# ============================================
# 第九课：为什么要切训练集/测试集 —— 过拟合与欠拟合
# 环境：skl_env
# ============================================

from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# ① 造一份"故意为难"的数据：两团交错的月牙形点，还掺了 0.3 的噪声
#    （纯净数据永远考不出过拟合，加噪声才能看出真本事）
X, y = make_moons(n_samples=300, noise=0.3, random_state=42)

# ② 切分：70% 训练，30% 测试（测过的题藏着，最后才拿出来考）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ③ 用不同的 max_depth（树"最多能追问几层"）各训练一个模型
#    depth 太小 → 欠拟合：太简单，训练/测试都考不好
#    depth 太大 → 过拟合：把训练集的怪癖都背下来，考试现原形
print(f"{'max_depth':>12} {'训练集准确率':>12} {'测试集准确率':>12}")
print("-" * 40)
for depth in range(1, 16):
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    train_acc = model.score(X_train, y_train)   # score() 对分类器=准确率
    test_acc = model.score(X_test, y_test)
    print(f"{depth:>12} {train_acc:>12.2%} {test_acc:>12.2%}")