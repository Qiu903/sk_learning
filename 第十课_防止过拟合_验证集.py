# ============================================
# 第十课：怎么防止过拟合（一）—— 用验证集挑参数，别偷看测试集
# 环境：skl_env
# ============================================

from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# ① 还是第九课那份带噪点的数据
X, y = make_moons(n_samples=300, noise=0.3, random_state=42)

# ② 先切出测试集，然后"锁起来"——挑参数阶段坚决不碰它
X_rest, X_test, y_rest, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ③ 从剩下的再切一刀：训练集 + 验证集（验证集专门用来挑参数）
X_train, X_val, y_train, y_val = train_test_split(
    X_rest, y_rest, test_size=0.25, random_state=42
)

# ④ 用"验证集"挑出最好的 max_depth（这不是偷看，是光明正大地选）
best_depth = 1
best_val = 0.0
print(f"{'max_depth':>12} {'验证集准确率':>12}")
print("-" * 28)
for depth in range(1, 16):
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    val_acc = model.score(X_val, y_val)
    print(f"{depth:>12} {val_acc:>12.2%}")
    if val_acc > best_val:
        best_val = val_acc
        best_depth = depth

# ⑤ 拿着挑好的 depth，最后才用"锁起来的测试集"验收一次
final = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
final.fit(X_train, y_train)
test_acc = final.score(X_test, y_test)
print(f"\n验证集帮我挑出的最佳 max_depth = {best_depth}")
print(f"最后拿锁着的测试集验收：{test_acc:.2%}")