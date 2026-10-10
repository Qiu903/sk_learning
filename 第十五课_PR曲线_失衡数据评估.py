# ============================================
# 第十五课：PR 曲线 —— 少数类很少时，ROC 会"说谎"，要看 PR
# 环境：skl_env
# ============================================

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

# ① 造一份 2% 正类（异常）的失衡数据，带 5% 标签噪声（模拟"特征不明显"的样本）
X, y = make_classification(
    n_samples=10000, n_features=10, n_informative=6,
    weights=[0.98, 0.02], flip_y=0.05, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)
proba = model.predict_proba(X_test)[:, 1]   # 被判为正类(异常)的概率

pos_ratio = y_test.mean()
print(f"测试集中正类(异常)占比：{pos_ratio:.2%}   ← 少数类，很少")

# ② 手动分类器，可调阈值
def classify(t):
    pred = (proba >= t).astype(int)
    tp = ((pred == 1) & (y_test == 1)).sum()
    fp = ((pred == 1) & (y_test == 0)).sum()
    tn = ((pred == 0) & (y_test == 0)).sum()
    fn = ((pred == 0) & (y_test == 1)).sum()
    return tp, fp, tn, fn

# ③ 逐个阈值看：精确率(precision) vs 召回率(recall)
print(f"\n{'阈值':>6} {'TP':>4} {'FP':>4} {'精确率':>8} {'召回率':>8} {'FPR':>8}")
for t in [0.9, 0.7, 0.5, 0.3, 0.1, 0.03]:
    tp, fp, tn, fn = classify(t)
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    fpr = fp / (fp + tn) if (fp + tn) else 0
    print(f"{t:>6.2f} {tp:>4} {fp:>4} {prec:>8.2%} {rec:>8.2%} {fpr:>8.2%}")

# ④ 用两个面积分数收尾
roc = roc_auc_score(y_test, proba)
aps = average_precision_score(y_test, proba)
print(f"\nROC AUC        = {roc:.3f}   （越大越好，0.5=瞎猜）")
print(f"PR AUC(AP)     = {aps:.3f}   （越大越好，随机基线={pos_ratio:.2%}）")
print(f"\n对比重点：ROC 很好看，但 PR 掉下来了——因为正类太少，误报的杀伤力被 ROC 掩盖了")
