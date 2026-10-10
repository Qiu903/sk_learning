# ============================================
# 第十四课：阈值、ROC 曲线与 AUC —— 一个旋钮如何决定"多报还是漏报"
# 环境：skl_env
# ============================================

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# ① 数据照旧：乳腺癌，0=恶性，1=良性
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

# ② 关键：predict_proba 会给每个样本一个"概率"，而不是直接给分类
proba = model.predict_proba(X_test)[:, 1]   # 被判为"良性(1)"的概率
print(f"概率范围：{proba.min():.2f} ~ {proba.max():.2f}")
print(f"说明：机器对每个样本先给一个 0~1 的信念值，再靠阈值决定押哪边")

# ③ 手动做一个"可调阈值"的分类器
def classify(threshold):
    pred = (proba >= threshold).astype(int)   # 概率 ≥ 阈值 → 判良性
    tp = ((pred == 1) & (y_test == 1)).sum()  # 真良性抓对
    fp = ((pred == 1) & (y_test == 0)).sum()  # 真恶性误判成良性（漏报！）
    tn = ((pred == 0) & (y_test == 0)).sum()  # 真恶性判对
    fn = ((pred == 0) & (y_test == 1)).sum()  # 真良性漏判
    return tp, fp, tn, fn

# ④ 看同一个模型、三个不同阈值下的表现
print("\n同一个模型，只换阈值：")
print(f"{'阈值':>6} {'TP抓对':>6} {'FP误报':>6} {'TN':>6} {'FN漏判':>6} {'召回率':>8}")
for t in [0.1, 0.5, 0.9]:
    tp, fp, tn, fn = classify(t)
    recall = tp / (tp + fn)
    print(f"{t:>6.1f} {tp:>6} {fp:>6} {tn:>6} {fn:>6} {recall:>8.2%}")

print("\n解读：")
print("  阈值 0.1 = 只要有一丁点像良性就报良性 → 召回率最高，但误报也多")
print("  阈值 0.9 = 非常确定才报良性 → 误报几乎为零，但漏判变多")
print("  阈值 0.5 = 默认折中")

# ⑤ ROC 的核心：把每个阈值下的两个比例画成曲线
#    TPR（真正例率/召回率）= tp/(tp+fn)：真良性里抓到了多少
#    FPR（假正例率）= fp/(fp+tn)：真恶性里有多少被误报成良性
print("\n把 0.00~0.99 每隔 0.01 当一次阈值，收集 TPR 和 FPR：")
print(f"{'阈值':>6} {'TPR(召回率)':>12} {'FPR(误报率)':>12}")
last_tpr, last_fpr = None, None
for i in range(101):
    t = i / 100
    tp, fp, tn, fn = classify(t)
    tpr = tp / (tp + fn) if (tp + fn) else 0
    fpr = fp / (fp + tn) if (fp + tn) else 0
    if i % 20 == 0 or i == 100:
        print(f"{t:>6.2f} {tpr:>12.2%} {fpr:>12.2%}")
    last_tpr, last_fpr = tpr, fpr

# ⑥ AUC：把这条曲线下的面积算成一个数
auc = roc_auc_score(y_test, proba)
print(f"\nAUC = {auc:.3f}")
print("AUC 的含义：随机抽一个真良性和一个真恶性，机器把良性排在前面的概率")
print("  0.5 = 瞎猜；越接近 1 = 区分能力越强")
