# ============================================
# 第十一课：交叉验证 —— 多考几次取平均，挑参数不再靠运气
# 环境：skl_env
# ============================================

from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score

# ① 数据还是那份带噪点的月牙
X, y = make_moons(n_samples=300, noise=0.3, random_state=42)

# ② 测试集照旧锁起来，只在最后验收用
X_rest, X_test, y_rest, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ③ 交叉验证核心：cross_val_score
#    cv=5 表示把 X_rest 平均切成 5 份，
#    轮流拿 1 份当"验证集"、另外 4 份当"训练集"，重复 5 次
print(f"{'max_depth':>12} {'5次平均准确率':>12}")
print("-" * 28)
for depth in range(1, 11):
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    # 打分 5 次，得到 5 个分数
    scores = cross_val_score(model, X_rest, y_rest, cv=5)
    print(f"{depth:>12} {scores.mean():>12.2%}")

# ④ 选出"平均分最高"的 depth
best_depth = None
best_mean = 0.0
for depth in range(1, 11):
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    scores = cross_val_score(model, X_rest, y_rest, cv=5)
    if scores.mean() > best_mean:
        best_mean = scores.mean()
        best_depth = depth
print(f"\n交叉验证挑出的最佳 max_depth = {best_depth}（平均分 {best_mean:.2%}）")

# ⑤ 最后用锁起来的测试集验收一次
final = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
final.fit(X_rest, y_rest)   # 慎重点：也可以只 fit 训练部分，这里用全部剩余数据
test_acc = final.score(X_test, y_test)
print(f"最后拿锁着的测试集验收：{test_acc:.2%}")