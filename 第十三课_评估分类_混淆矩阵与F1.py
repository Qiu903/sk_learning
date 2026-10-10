# ============================================
# 第十三课：评估分类模型 —— 混淆矩阵、精确率、召回率、F1
# 环境：skl_env
# ============================================

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# =========================================
# 第一部分：先在一份"正反均衡"的数据上看混淆矩阵
# =========================================
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target   # 目标: 0=恶性, 1=良性
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 准确率
acc = (y_pred == y_test).mean()
print("=" * 50)
print("第一部分：均衡数据上的混淆矩阵")
print("=" * 50)
print(f"准确率：{acc:.2%}")

# 混淆矩阵：我们把"恶性(0)"当作要抓的正类，行=真实，列=预测
# 返回 [[TN, FP], [FN, TP]]
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
print("\n                机器预测")
print("           ┌────────────┬────────────┐")
print("           │  恶性(0)   │  良性(1)   │")
print(" 真实恶性 ─┤ TN  {tn:>5} │ FP  {fp:>5} │".format(tn=tn, fp=fp))
print("           ├────────────┼────────────┤")
print(" 真实良性 ─┤ FN  {fn:>5} │ TP  {tp:>5} │".format(fn=fn, tp=tp))
print("           └────────────┴────────────┘")
print("\nTN = 真恶性判成恶性（判对）")
print("FP = 真良性被误判成恶性（好心办坏事）")
print("FN = 真恶性被漏判成良性（最危险的漏网之鱼）")
print("TP = 真良性判成良性（判对）")

# 精确率、召回率、F1（我们把"良性(1)"当正类来算，更贴近"检测到目标"）
print("\n以【良性=正类】重新计算三个指标：")
print(f"精确率 Precision = {precision_score(y_test, y_pred):.2%}  预测为良性里，有多少真良性")
print(f"召回率 Recall    = {recall_score(y_test, y_pred):.2%}   真实良性里，抓到了多少")
print(f"F1分数          = {f1_score(y_test, y_pred):.2%}   精确率与召回率的调和平均")

# =========================================
# 第二部分：造一份"正类很少"的数据，让准确率现原形
# =========================================
print("\n" + "=" * 50)
print("第二部分：人为造一份 95% 都是'正常'，只有 5% 是'异常'的数据")
print("=" * 50)
import numpy as np
rng = np.random.default_rng(42)
n = 10000
# 98% 是 0（正常），2% 是 1（异常——这才是你真正关心的）
y_imb = np.array([0] * 9800 + [1] * 200)
rng.shuffle(y_imb)

# 来个"躺平"模型：永远只猜 0（正常）。它一个异常都不报。
y_guess0 = np.zeros(n)

print("一个【永远只猜正常】的傻模型，算出来的准确率：")
acc_silly = (y_guess0 == y_imb).mean()
print(f"  它的准确率 = {acc_silly:.2%}")
print("  但它把 200 个异常【全部漏报了】!召回率 = 0")

print("\n对比：同样这个傻模型，精确率/召回率/F1 立刻露馅：")
print(f"  Precision = {precision_score(y_imb, y_guess0):.4f}")
# 为避免除零，手动说明:
print(f"  Recall    = 0（一个异常都没抓到）")
print(f"  F1        = 0")