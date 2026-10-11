# ============================================
# 第十九课：SVM 多分类 —— 只会画一条线的算法，怎么分三类、十类？
# 环境：skl_env
#
# 【先来个灵魂拷问】
#   前面两课我们一直在讲 SVM 怎么"画一条线，把两堆点分开"。
#   它的拿手好戏就是【二分类】：只有两个类别，你死我活。
#   可现实问题里，类别往往不止两个：
#       · 鸢尾花有 3 种（山鸢尾 / 变色鸢尾 / 维吉尼亚鸢尾）
#       · 手写数字有 10 种（0~9）
#   一个只会分两类的算法，怎么去分 3 类、10 类？
#
# 【本课要讲清楚的两大套路】
#   套路一：一对多（OvR，One-vs-Rest）—— 打"公投"
#   套路二：一对一（OvO，One-vs-One）—— 打"循环擂台赛"
#   最后用真实数据看看它们各自的效果和优缺点。
# ============================================

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations           # 进阶：专门生成"两两组合"，后面 OvO 会用到
from sklearn.datasets import load_iris, load_digits
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# matplotlib 默认字体不含中文，会显示成方框，所以指定微软雅黑
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 三个类别统一用这三种颜色，全课保持一致，看图不眼花
COLORS = ["#3b82f6", "#f97316", "#16a34a"]        # 蓝 / 橙 / 绿


# ============================================
# ① 先想明白：为什么 SVM "天生"只会分两类？
# ============================================
print("=" * 64)
print("① 为什么 SVM 天生只会分两类？")
print("=" * 64)
print("回忆第十七课：SVM 干的事就是找一条线（超平面），")
print("  让两边的点尽量被'最宽的马路'隔开。")
print()
print("它的数学目标从头到尾只有一个：")
print("  把【这边的点】和【那边的点】分开 —— 只有'这边'和'那边'两个说法。")
print("  也就是说，SVM 的脑子里根本没有'第三类'这个概念。")
print()
print("所以要分多类，必须'拼装'出一套办法，把多分类拆成一堆二分类。")
print("下面两个套路，就是最经典、也最常用的两种拆法。")


# ============================================
# ② 准备数据：鸢尾花 3 类
# ============================================
print("\n" + "=" * 64)
print("② 准备鸢尾花数据（3 类，方便先看清套路）")
print("=" * 64)

iris = load_iris()
# 鸢尾花原本有 4 个特征，我们只取后 2 个（花瓣长度、花瓣宽度），
# 因为 2 个特征才能画成平面图，看得见边界。
X = iris.data[:, 2:4]
y = iris.target

print(f"特征含义：第 1 列 = 花瓣长度，第 2 列 = 花瓣宽度")
print(f"X 的形状 = {X.shape}  →  150 朵花，每朵 2 个特征")
print(f"类别：{np.unique(y)}  →  0=山鸢尾，1=变色鸢尾，2=维吉尼亚鸢尾")

# 【重点】归一化！SVM 是"距离类"模型，忘了归一化，长量纲特征会压倒短量纲特征
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)     # 训练集：算出均值/标准差并转换
X_test = scaler.transform(X_test)           # 测试集：只用训练集的均值/标准差（不能重算）
print("已做归一化：训练集算均值标准差，测试集直接套用（防止信息泄露）")


# ============================================
# ③ 套路一：一对多 OvR（One-vs-Rest）—— 举办 3 场"公投"
# ============================================
print("\n" + "=" * 64)
print("③ 套路一：一对多 OvR —— 每一类办一场'公投'")
print("=" * 64)
print("【比喻：公投选举】")
print("  给每一类都办一场投票，题目是：'你是这一类，还是其他所有类？'")
print("  3 个类别 → 就办 3 场公投，也就是训练 3 个 SVM：")
print("    第 1 场：山鸢尾   vs  （变色鸢尾 + 维吉尼亚鸢尾）")
print("    第 2 场：变色鸢尾 vs  （山鸢尾 + 维吉尼亚鸢尾）")
print("    第 3 场：维吉尼亚 vs  （山鸢尾 + 变色鸢尾）")
print()
print("【预测一朵新花】")
print("  把它分别喂给 3 个 SVM，每个 SVM 都会给一个'自信分'（离边界多远）。")
print("  谁给的分最高，就判定它属于哪一类。")

classes = np.unique(y_train)     # [0, 1, 2]
ovr_models = []

for c in classes:
    # 造"二分类"标签：是第 c 类记 1，其余全记 0
    y_binary = (y_train == c).astype(int)
    # 进阶语法：astype(int) 把 True/False 转成 1/0
    m = SVC(kernel="linear", C=1)
    m.fit(X_train, y_binary)
    ovr_models.append((c, m))
    print(f"  训练完 第 {c} 类 vs 其余   （属于第 {c} 类的训练样本数 = {y_binary.sum()}，" 
          f"其余 = {(1 - y_binary).sum()}）")


def ovr_predict(X_input):
    """OvR 预测：把每朵花喂给全部 SVM，谁的分高算谁"""
    n = X_input.shape[0]
    scores = np.zeros((n, len(classes)))    # 每朵花 × 每个类别的得分表
    for j, (c, m) in enumerate(ovr_models):
        # 进阶：decision_function 返回"离决策边界的带符号距离"，绝对值越大越自信
        scores[:, j] = m.decision_function(X_input)
    # argmax(axis=1)：每行里分数最大的那个类别的下标
    best = np.argmax(scores, axis=1)
    return classes[best]


ovr_pred = ovr_predict(X_test)
ovr_acc = np.mean(ovr_pred == y_test)
print(f"\n【OvR 手写实现】测试集准确率 = {ovr_acc:.2%}")


# ============================================
# ④ 套路二：一对一 OvO（One-vs-One）—— 打"循环擂台赛"
# ============================================
print("\n" + "=" * 64)
print("④ 套路二：一对一 OvO —— 每两类之间打一场擂台赛")
print("=" * 64)
print("【比喻：循环擂台赛】")
print("  不让一类单挑全部，而是让每两类之间都单独打一场。")
print("  3 个类别 → 组合数 = 3×2÷2 = 3 场：")
print("    场次 1：山鸢尾   vs 变色鸢尾")
print("    场次 2：山鸢尾   vs 维吉尼亚")
print("    场次 3：变色鸢尾 vs 维吉尼亚")
print()
print("【预测一朵新花】")
print("  让这朵花参加全部 3 场比赛，每场投一票给它认为的赢家，")
print("  最后得票最多的类获胜（就像循环赛积分排名）。")

# 进阶语法：combinations(列表, 2) 会生成"从中任选 2 个"的所有组合
#   例：combinations([0,1,2], 2) → (0,1) (0,2) (1,2)
pairs = list(combinations(classes, 2))
ovo_models = []

for (a, b) in pairs:
    # 只挑出这两类的样本，其他类暂时不管
    mask = (y_train == a) | (y_train == b)
    m = SVC(kernel="linear", C=1)
    m.fit(X_train[mask], y_train[mask])
    ovo_models.append(((a, b), m))
    print(f"  训练完 第 {a} 类 vs 第 {b} 类   （用到样本数 = {mask.sum()}）")


def ovo_predict(X_input):
    """OvO 预测：每场比赛投一票，数票数，票多的赢"""
    n = X_input.shape[0]
    votes = np.zeros((n, len(classes)), dtype=int)   # 每朵花 × 每个类别的票数
    for (a, b), m in ovo_models:
        pred = m.predict(X_input)                    # 这场比赛的判决结果（a 或 b）
        ia = np.where(classes == a)[0][0]            # 进阶：np.where 找出类别 a 在表里的下标
        ib = np.where(classes == b)[0][0]
        votes[:, ia] += (pred == a)                  # 判给 a 的，a 加一票
        votes[:, ib] += (pred == b)                  # 判给 b 的，b 加一票
    return classes[np.argmax(votes, axis=1)]


ovo_pred = ovo_predict(X_test)
ovo_acc = np.mean(ovo_pred == y_test)
print(f"\n【OvO 手写实现】测试集准确率 = {ovo_acc:.2%}")
print(f"一共训练了 {len(ovo_models)} 个二分类 SVM")


# ============================================
# ⑤ 用 sklearn 现成的封装：和我们的手写结果对一对
# ============================================
print("\n" + "=" * 64)
print("⑤ 用 sklearn 现成封装，验证我们手写的对不对")
print("=" * 64)

# OneVsRestClassifier：把任意二分类器包装成"一对多"多分类器
sk_ovr = OneVsRestClassifier(SVC(kernel="linear", C=1))
sk_ovr.fit(X_train, y_train)
print(f"  sklearn OvR 准确率 = {sk_ovr.score(X_test, y_test):.2%}")

# OneVsOneClassifier：包装成"一对一"多分类器
sk_ovo = OneVsOneClassifier(SVC(kernel="linear", C=1))
sk_ovo.fit(X_train, y_train)
print(f"  sklearn OvO 准确率 = {sk_ovo.score(X_test, y_test):.2%}")

print("\n【重要】SVC 自己其实已经内置了多分类能力。")
print("  你直接 SVC().fit(三类的数据) 它也能跑，因为 libsvm 底层默认用的是 OvO，")
print("  只是把结果又换算成一对多的形式给你看，用起来像'天生支持多分类'。")
print("  参数 decision_function_shape='ovr'（默认）就是干这个换算的。")

direct = SVC(kernel="linear", C=1)
direct.fit(X_train, y_train)
print(f"  直接 SVC 准确率 = {direct.score(X_test, y_test):.2%}   ← 和上面 OvO 一个路子")


# ============================================
# ⑥ 可视化：把两套套路画的线都画出来看看
# ============================================
print("\n" + "=" * 64)
print("⑥ 看图：OvR 画 3 条线，OvO 也画 3 条线，但线的含义不同")
print("=" * 64)


def plot_line(ax, model, color, label):
    """把一个线性 SVM 的决策边界（一条直线）画到坐标轴上

    直线方程：w0·x + w1·y + b = 0
    变形一下：y = -(w0·x + b) / w1
    """
    w = model.coef_[0]        # 权重向量（决定线的方向）
    b = model.intercept_[0]   # 截距（决定线的位置）
    xs = np.array([-3.5, 3.5])
    if abs(w[1]) < 1e-9:
        # 特殊情况：w1 几乎为 0，线是竖直的，用 axvline 画
        ax.axvline(-b / w[0], color=color, linestyle="--", linewidth=1.6, label=label)
    else:
        ys = -(w[0] * xs + b) / w[1]
        ax.plot(xs, ys, color=color, linestyle="--", linewidth=1.6, label=label)


fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.5))

# ---- 左图：OvR 的 3 条线，每条线都是"某一类 vs 其余两类" ----
for j, (c, m) in enumerate(ovr_models):
    plot_line(axL, m, COLORS[j], f"第{c}类 vs 其余")

# ---- 右图：OvO 的 3 条线，每条线都是"两两之间"的分界 ----
for j, ((a, b), m) in enumerate(ovo_models):
    plot_line(axR, m, COLORS[j], f"第{a}类 vs 第{b}类")

# 两幅图都把三类散点画上去（归一化后的坐标）
for ax in (axL, axR):
    for c in classes:
        ax.scatter(X_train[y_train == c, 0], X_train[y_train == c, 1],
                   c=COLORS[c], s=30, edgecolors="white", linewidths=0.5,
                   label=f"第{c}类数据")
    ax.set_xlabel("花瓣长度（归一化后）")
    ax.set_ylabel("花瓣宽度（归一化后）")
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.legend(fontsize=8, loc="upper left")

axL.set_title("OvR 一对多：3 条线 = 3 场'某类 vs 其余'的公投")
axR.set_title("OvO 一对一：3 条线 = 3 场'两两单挑'的擂台赛")
plt.tight_layout()
plt.show()

print("看图感受：")
print("  OvR 的线：每一条都想把'某一类'从整体里圈出来，所以线常常偏在一角")
print("  OvO 的线：每一条只负责把相邻的两类分开，分工更细、更均匀")


# ============================================
# ⑦ 混淆矩阵：不只看准确率，还要看"错在哪里"
# ============================================
print("\n" + "=" * 64)
print("⑦ 混淆矩阵：看看 SVM 具体把哪类认成了哪类")
print("=" * 64)

best_pred = sk_ovo.predict(X_test)
cm = confusion_matrix(y_test, best_pred)

fig2, ax2 = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=["山鸢尾", "变色鸢尾", "维吉尼亚"])
disp.plot(ax=ax2, cmap="Blues", values_format="d")
ax2.set_title("OvO 多分类的混淆矩阵")
plt.tight_layout()
plt.show()

print("矩阵读法：行 = 真实类别，列 = 预测类别，对角线上的数是'判对'的个数")
print(f"本次测试共 {len(y_test)} 朵花，判对 {np.trace(cm)} 朵")
# 进阶：np.trace 求矩阵"主对角线元素之和"，正好就是判对的总数
print("非零的非对角线格子 = 判错的地方，能直接看出是哪两类容易混")


# ============================================
# ⑧ 升到 10 类：看看分类器数量怎么"爆炸"
# ============================================
print("\n" + "=" * 64)
print("⑧ 类别变多会怎样？用 10 类手写数字感受一下")
print("=" * 64)

digits = load_digits()
X_d = digits.data      # 每张图 8×8 = 64 个像素特征
y_d = digits.target    # 0~9 共 10 类

X_d = StandardScaler().fit_transform(X_d)
print(f"数据：{X_d.shape[0]} 张手写数字图，每张 {X_d.shape[1]} 个像素特征，共 10 类")

k = 10
n_ovo = k * (k - 1) // 2      # 进阶：// 是整数除法，用来算组合数 C(k,2)
n_ovr = k
print(f"\n如果要分 {k} 类：")
print(f"  OvR（一对多）：需要训练 {n_ovr} 个二分类器（每类一个）")
print(f"  OvO（一对一）：需要训练 {n_ovo} 个二分类器（两两组合）")
print(f"  → 10 类时 OvO 要训 {n_ovo} 个，如果是 100 类就要训 {100*99//2} 个，数量爆炸！")

# 实际跑一下 10 类，看看准确率
Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(
    X_d, y_d, test_size=0.3, random_state=42
)
m10 = SVC(kernel="rbf", C=1, gamma="scale")   # 进阶：gamma='scale' 让 sklearn 自动选一个合理的值
m10.fit(Xd_tr, yd_tr)
print(f"\n10 类手写数字（rbf 核）测试集准确率 = {m10.score(Xd_te, yd_te):.2%}")
print("注意：类别越多，OvO 要训练的分类器数量按'平方'增长，这也是它的最大软肋。")


# ============================================
# ⑨ 两种套路的取舍（考试/面试都爱问）
# ============================================
print("\n" + "=" * 64)
print("⑨ OvR 和 OvO 到底用哪个？")
print("=" * 64)
print("  OvR（一对多）：")
print("    + 分类器数量少（= 类别数），总训练次数少")
print("    - 每场都是'1 类 vs 一大堆'，正负样本不平衡，边界容易被大堆那边带偏")
print("  OvO（一对一）：")
print("    + 每场只用两类数据，样本少、训练快，每场都是平衡的二分类（SVM 最爱）")
print("    - 类别一多，分类器数量按平方涨（10 类 45 个，100 类 4950 个）")
print()
print("  实际使用建议：")
print("    · 类别很少（几类）：两者都行，OvO 通常略快")
print("    · 类别很多（几十上百类）：OvR 的分类器更少，反而更划算")
print("    · 用 sklearn 的 SVC 时，什么都不用管，它默认内部就是 OvO，直接拿来用即可")


# ============================================
# 本课小结
# ============================================
print("\n" + "=" * 64)
print("小结")
print("=" * 64)
print("  · SVM 天生只会分两类，多分类必须把问题'拆成很多个二分类'")
print("  · OvR 一对多：每类办一场'某类 vs 其余'的公投 → 得分最高者胜")
print("  · OvO 一对一：每两类打一场擂台赛 → 投票，票多者胜")
print("  · 分类器数量：OvR = 类别数 K；OvO = K×(K-1)/2（平方级增长）")
print("  · sklearn 的 SVC 默认内部就用 OvO，直接 fit 多类数据即可")
print("  · 无论哪种套路，SVM 使用前都别忘了先归一化")
