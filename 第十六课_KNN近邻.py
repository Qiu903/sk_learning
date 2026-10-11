# ============================================
# 第十六课：KNN 近邻分类 —— "近朱者赤，近墨者黑"
# 环境：skl_env
#
# 本课结构：
#   ① 手算一次 KNN：理解"距离 + 投票"的完整过程
#   ② 用 sklearn 一键实现
#   ③ 换 k 值，对比欠拟合/过拟合（复习第九课）
#   ④ 进阶：特征尺度问题（KNN 的命门）+ 归一化修复
# ============================================

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X, y = iris.data, iris.target   # X=特征(花萼/花瓣长宽)，y=标签(3种鸢尾)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ============================================
# ① 手算一次 KNN（把算法从头到尾走一遍）
# ============================================
print("=" * 50)
print("① 手算一次 KNN：选测试集第 1 个样本")
print("=" * 50)

sample = X_test[0]                    # 一个"新来的样本"，只有特征
true_label = y_test[0]                # 它的真实标签（暂时藏起来）

# 第 1 步：计算它到所有训练样本的"欧氏距离"
# 欧氏距离公式：根号下 每个特征差的平方 之和
diffs = X_train - sample
dist = np.sqrt((diffs ** 2).sum(axis=1))

# 第 2 步：找出距离最近的 k 个邻居（这里 k=5）
k = 5
nearest_idx = np.argsort(dist)[:k]    # 按距离排序，取前 5 个的下标
neighbor_labels = y_train[nearest_idx]

print(f"样本特征：{sample}")
print(f"距离最近的 {k} 个邻居的训练标签：{neighbor_labels}")
print(f"这 {k} 个邻居的标签分布（投票）：")
for label in np.unique(neighbor_labels):
    count = (neighbor_labels == label).sum()
    print(f"  类别 {label}：{count} 票")
print(f"得票最多 → 预测类别：{np.bincount(neighbor_labels).argmax()}")
print(f"真实类别：{true_label}（{'猜对了' if np.bincount(neighbor_labels).argmax() == true_label else '猜错了'}）")
print("KNN 的本质：训练阶段啥也没学，只把数据记住；预测时才现算距离、现投票。")

# ============================================
# ② sklearn 一键实现
# ============================================
print("\n" + "=" * 50)
print("② sklearn 一键实现 KNN（k=5）")
print("=" * 50)

knn = KNeighborsClassifier(n_neighbors=5)   # n_neighbors 就是 k
knn.fit(X_train, y_train)
acc = knn.score(X_test, y_test)
print(f"测试集准确率：{acc:.2%}")

# ============================================
# ③ k 值大小 = 记忆力高低（对照第九课的过拟合/欠拟合）
# ============================================
print("\n" + "=" * 50)
print("③ 换不同的 k，看训练集/测试集准确率怎么变")
print("=" * 50)
print(f"{'k':>4} {'训练集准确率':>10} {'测试集准确率':>10}   ← 解读")
for k in [1, 3, 5, 10, 20, 50, 100]:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    tr = model.score(X_train, y_train)
    te = model.score(X_test, y_test)
    note = ""
    if k == 1:
        note = "k=1：只看最近1个 → 训练集100%，容易记噪点(过拟合)"
    elif k == 5:
        note = "k=5：甜点区"
    elif k == 100:
        note = "k太大：离得远的样本也来投票(欠拟合)"
    print(f"{k:>4} {tr:>10.2%} {te:>10.2%}   {note}")

# ============================================
# ④ 进阶：特征尺度问题 —— 距离会被"大数字"劫持
# ============================================
print("\n" + "=" * 50)
print("④ 特征尺度问题：把第2个特征放大100倍，KNN 会怎样？")
print("=" * 50)

X2 = X[:, :2].copy()                      # 只用前两个特征(花萼长宽，单位都是cm)
X2[:, 1] = X2[:, 1] * 100                 # 人为把"花萼宽"放大100倍
Xtr, Xte, ytr, yte = train_test_split(X2, y, test_size=0.3, random_state=42)

knn_bad = KNeighborsClassifier(n_neighbors=5)
knn_bad.fit(Xtr, ytr)
acc_bad = knn_bad.score(Xte, yte)

# 用 StandardScaler 做"归一化"：每个特征减去均值、除以标准差，拉到同一尺度
scaler = StandardScaler()
Xtr_s = scaler.fit_transform(Xtr)
Xte_s = scaler.transform(Xte)

knn_good = KNeighborsClassifier(n_neighbors=5)
knn_good.fit(Xtr_s, ytr)
acc_good = knn_good.score(Xte_s, yte)

print(f"不归一化（特征尺度差100倍）：测试集准确率 {acc_bad:.2%}")
print(f"归一化后（同一尺度）：        测试集准确率 {acc_good:.2%}")
print("原因：KNN 靠距离判断，哪个特征数值大，谁就主导距离，小尺度特征等于被无视。")
