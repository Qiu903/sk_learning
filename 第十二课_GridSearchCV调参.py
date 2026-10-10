# ============================================
# 第十二课：GridSearchCV —— 让机器自己穷举参数来找最优
# 环境：skl_env
# ============================================

from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

# ① 数据照旧
X, y = make_moons(n_samples=300, noise=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ② 给机器一个"候选清单"：两个旋钮各试哪几个值
param_grid = {
    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15],
    'min_samples_leaf': [1, 5, 10],
}

# ③ 一个 GridSearchCV 就把"穷举 + 交叉验证 + 找最优"全包了
#    cv=5：每种组合都在 5 折交叉验证下打分，取平均后再比较
model = DecisionTreeClassifier(random_state=42)
search = GridSearchCV(model, param_grid, cv=5)
search.fit(X_train, y_train)

# ④ 它找到了最优的旋钮组合
print(f"最优参数组合：{search.best_params_}")
print(f"最优参数在交叉验证下的平均分：{search.best_score_:.2%}")

# ⑤ 用锁起来的测试集验收这个"最优组合"
best = search.best_estimator_
test_acc = best.score(X_test, y_test)
print(f"最后拿测试集验收：{test_acc:.2%}")

# ⑥ 看看它到底试了多少种组合
total = 1
for v in param_grid.values():
    total *= len(v)
print(f"\n它心里穷举了 {total} 种组合，每种用 5 折交叉验证打分，加起来 {total * 5} 次训练")