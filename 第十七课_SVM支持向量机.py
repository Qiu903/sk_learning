# ============================================
# 第十七课：SVM 支持向量机 —— 用图看懂"最宽的隔断"
# 环境：skl_env
#
# 一句话理解 SVM：
#   两类点混在平面上，SVM 要画一条直线把它们分开，
#   并且要求这条线"离两边都尽量远"（间隔最宽）→ 最稳。
#
# 三个核心名词（下面每节都会在图上指出来）：
#   ① 决策边界(超平面)：那条分界线本身
#   ② 间隔(margin)    ：分界线两边留出的"安全区"总宽度
#   ③ 支持向量        ：离分界线最近、真正撑住边界的那几个点
#
# 本课结构：
#   ① 造一批 2D 数据，先看看它长什么样
#   ② 训练 SVM，把决策边界 + 间隔 + 支持向量画出来（重点！）
#   ③ 打印 w、b、||w||、间隔宽度，和图上的数字对上
#   ④ 换 C 值画两张图，直观感受"宽容度"
#   ⑤ 真实数据(乳腺癌)：C 表格 + 为什么必须归一化
# ============================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# matplotlib 默认字体不含中文，直接写中文标题会变成一排方框，
# 所以这里指定用"微软雅黑"来画中文
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False   # 让负号正常显示（否则负号也是方框）

# ============================================
# ① 造一批 2D 数据，先看看它长什么样
# ============================================
print("=" * 60)
print("① 造 80 个样本，2 个特征（方便画在平面上），分成 2 类")
print("=" * 60)

# make_blobs：造假数据的小工具，专门生成"一簇一簇"的点
#   n_samples=80  一共 80 个点
#   centers=2     分成 2 簇（也就是 2 类）
#   cluster_std=1.2  每一簇散得有多开，数字越大越重叠
#   random_state=42  固定随机种子，保证你每次跑的结果都一样
X, y = make_blobs(n_samples=80, centers=2, cluster_std=1.2, random_state=42)

print(f"X 的形状 = {X.shape}  →  80 行(样本) × 2 列(特征)")
print(f"y 的取值 = {np.unique(y)}  →  只有 0 和 1 两个类别")

# 先把原始数据画出来看看：蓝点一堆、橙点一堆，中间空着一条缝
fig1, ax1 = plt.subplots(figsize=(7, 5))
ax1.scatter(X[y == 0, 0], X[y == 0, 1], c="#3b82f6", s=35, label="类别 0")
ax1.scatter(X[y == 1, 0], X[y == 1, 1], c="#f97316", s=35, label="类别 1")
ax1.set_xlabel("特征 1")
ax1.set_ylabel("特征 2")
ax1.set_title("原始数据：两堆点，中间有一条天然的缝")
ax1.legend()
plt.show()   # 弹出窗口显示这张图（关掉窗口后继续往下跑）

# ============================================
# ② 训练 SVM，把"决策边界 + 间隔 + 支持向量"画出来
# ============================================
print("\n" + "=" * 60)
print("② 训练线性 SVM，并在图上标出三个核心概念")
print("=" * 60)


def plot_boundary(ax, model, X_data, y_data, title):
    """在给定的坐标轴 ax 上画：两类散点 + 决策边界 + 间隔虚线 + 支持向量圈

    参数说明（进阶：函数可以接收参数，用参数比写死更灵活）：
      ax     : matplotlib 的"坐标轴对象"，往它上面画东西
      model  : 已经 fit 过的 SVM 模型
      X_data : 特征
      y_data : 标签
      title  : 这张图的标题
    """
    w = model.coef_[0]          # 权重向量 w = [w0, w1]
    b = model.intercept_[0]     # 截距 b

    # 先确定横轴的显示范围，左右各留一点边
    x_min, x_max = X_data[:, 0].min() - 0.5, X_data[:, 0].max() + 0.5
    xx = np.linspace(x_min, x_max, 200)   # 在横轴上均匀取 200 个点

    # 直线的方程是 w0*x + w1*y + b = 0，把它解成 y = ??? 的形式：
    #   w1*y = -(w0*x + b)   →   y = -(w0*x + b) / w1
    yy = -(w[0] * xx + b) / w[1]

    # 间隔的两条边界，就是方程等于 +1 和 -1 的两条线
    # （SVM 内部规定：支持向量恰好落在 ±1 这两条线上）
    yy_up = -(w[0] * xx + b - 1) / w[1]
    yy_down = -(w[0] * xx + b + 1) / w[1]

    # 画决策边界（实线）
    ax.plot(xx, yy, color="#16a34a", linewidth=2.5, label="决策边界(超平面)")
    # 画间隔的两条边界（虚线）
    ax.plot(xx, yy_up, color="#16a34a", linestyle="--", linewidth=1,
            label="间隔边界(安全区边缘)")
    ax.plot(xx, yy_down, color="#16a34a", linestyle="--", linewidth=1)

    # 画两类散点
    ax.scatter(X_data[y_data == 0, 0], X_data[y_data == 0, 1],
               c="#3b82f6", s=35, label="类别 0")
    ax.scatter(X_data[y_data == 1, 0], X_data[y_data == 1, 1],
               c="#f97316", s=35, label="类别 1")

    # 画支持向量：只画圈不填色，套在原来那些点外面
    # （进阶：support_vectors_ 是模型自带的属性，直接存着支持向量的坐标）
    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
               s=260, facecolors="none", edgecolors="red", linewidths=1.8,
               label="支持向量(撑住边界的少数点)")

    ax.set_xlabel("特征 1")
    ax.set_ylabel("特征 2")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")


# 造模型：kernel="linear" 表示"就用一条直线来分"（最基础的 SVM）
model = SVC(kernel="linear", C=1.0)
model.fit(X, y)   # 训练：让模型自己去找那条最宽的线

fig2, ax2 = plt.subplots(figsize=(7, 5))
plot_boundary(ax2, model, X, y, "SVM 学到的：绿实线=分界，绿虚线=安全区，红圈=支持向量")
plt.show()

print("看图要点：")
print("  1) 绿实线把蓝、橙两堆分开了 —— 这条线叫【决策边界/超平面】")
print("  2) 两条绿虚线之间的宽度 —— 叫【间隔 margin】，SVM 就是把它撑到最宽")
print("  3) 被红圈套住的那几个点 —— 叫【支持向量】，只有它们决定线的位置")
print("     其它离得远的点，你怎么挪都不影响这根线")

# ============================================
# ③ 打印数字，和图对得上
# ============================================
print("\n" + "=" * 60)
print("③ 把那根线用数字写出来（和图对照着看）")
print("=" * 60)

w = model.coef_[0]
b = model.intercept_[0]
norm_w = np.linalg.norm(w)   # 进阶：np.linalg.norm = 求向量的长度(模长)，高中数学的 |w|
margin = 2 / norm_w          # 间隔宽度公式：两条边界差 2，再除以 w 的长度

print(f"w（权重向量） = [{w[0]:.3f}, {w[1]:.3f}]")
print(f"b（截距）     = {b:.3f}")
print(f"决策边界方程  = {w[0]:.3f}*x1 + {w[1]:.3f}*x2 + ({b:.3f}) = 0")
print(f"||w||（w 的长度） = {norm_w:.3f}")
print(f"间隔宽度 = 2 / ||w|| = {margin:.3f}")
print(f"支持向量个数 = {len(model.support_)} 个（80 个样本里只靠这几个！）")
print(f"训练集准确率 = {model.score(X, y):.2%}")

# ============================================
# ④ 换 C 值，画两张图直观对比"宽容度"
# ============================================
print("\n" + "=" * 60)
print("④ 换 C 值：C 小 = 宽容(允许犯错)，C 大 = 严格(一点不让错)")
print("=" * 60)

# 故意造一批"有点重叠"的数据，让 C 的作用看得明显
# cluster_std=2.6 散得比较开，两类点会互相掺一点
X_c, y_c = make_blobs(n_samples=60, centers=2, cluster_std=2.6, random_state=3)
# 再手动加两个"叛徒"：把第 0 个点搬到对面堆里，故意制造错分点
X_c[0] = X_c[0] + np.array([6.0, 0.0])
X_c[1] = X_c[1] + np.array([0.0, 6.0])

# 1 行 2 列的子图：左边放 C 小的，右边放 C 大的
fig3, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))

model_small = SVC(kernel="linear", C=0.01)
model_small.fit(X_c, y_c)
plot_boundary(axL, model_small, X_c, y_c,
              f"C=0.01（宽容）：间隔宽，但会放过叛徒\n训练准确率 {model_small.score(X_c, y_c):.1%}")

model_big = SVC(kernel="linear", C=1000)
model_big.fit(X_c, y_c)
plot_boundary(axR, model_big, X_c, y_c,
              f"C=1000（严格）：为了全对而贴点画线，间隔变窄\n训练准确率 {model_big.score(X_c, y_c):.1%}")

plt.tight_layout()   # 自动调整子图间距，避免标题挤在一起
plt.show()

print("看图要点：")
print("  C 小(0.01)：线比较直、间隔宽，容忍把个别叛徒分错 → 太宽松容易【欠拟合】")
print("  C 大(1000)：线被迫扭一点去迁就叛徒，间隔变窄 → 太严格容易【过拟合】")
print("  结论：C 就是 SVM 版的旋钮，和决策树的 max_depth、KNN 的 k 是一回事")

# ============================================
# ⑤ 真实数据（乳腺癌）：C 表格 + 归一化
# ============================================
print("\n" + "=" * 60)
print("⑤ 换成真实数据（乳腺癌），先归一化，再看 C 的影响")
print("=" * 60)

cancer = load_breast_cancer()
Xr, yr = cancer.data, cancer.target
X_train, X_test, y_train, y_test = train_test_split(
    Xr, yr, test_size=0.3, random_state=42
)

# 归一化：把每个特征都缩放到"均值 0、标准差 1"的同一量级
# fit_transform = 先算训练集的均值和标准差，再按这个标准转换训练集
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
# transform = 测试集直接套用训练集的标准，绝对不能自己重算！
X_test_s = scaler.transform(X_test)

print(f"{'C':>8} {'训练集准确率':>10} {'测试集准确率':>10} {'支持向量数':>8}")
for C in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
    svm = SVC(kernel="linear", C=C)
    svm.fit(X_train_s, y_train)
    tr = svm.score(X_train_s, y_train)
    te = svm.score(X_test_s, y_test)
    print(f"{C:>8.3f} {tr:>10.2%} {te:>10.2%} {len(svm.support_):>8}")

print("解读：")
print("  C 很小(0.001)：宁可画很宽的间隔，即使错分几个也不管 → 容易欠拟合")
print("  C 很大(1000)：为了训练集全对，间隔被压窄、贴样本画线 → 容易过拟合")
print("  支持向量数随 C 变大而变少：C 小=间隔宽，很多样本卡在间隔里被迫当支持向量")

# 为什么 SVM 也必须归一化？（用图直接看）
print("\n" + "-" * 60)
print("为什么 SVM 也必须归一化？（看图比看数字清楚）")
print("-" * 60)

# 回到第①节那份 2D 数据，把第 2 个特征放大 1000 倍，人为制造"量级悬殊"
X_n = X.copy()
X_n[:, 1] = X_n[:, 1] * 1000

fig4, (axR1, axR2) = plt.subplots(1, 2, figsize=(13, 5))

# 左图：不归一化
m_raw = SVC(kernel="linear", C=1)
m_raw.fit(X_n, y)
plot_boundary(axR1, m_raw, X_n, y, "不归一化：线几乎水平，等于只看特征2")

# 右图：归一化后
scaler2 = StandardScaler()
X_n_s = scaler2.fit_transform(X_n)   # fit_transform 会返回新数组，原数据不变
m_s = SVC(kernel="linear", C=1)
m_s.fit(X_n_s, y)
plot_boundary(axR2, m_s, X_n_s, y, "归一化后：线斜着穿过，两个特征都参与")

plt.tight_layout()
plt.show()

print(f"不归一化时 w = [{m_raw.coef_[0][0]:.2e}, {m_raw.coef_[0][1]:.2e}]")
print("  → w[0] 小到接近 0，说明模型几乎完全忽略了【特征1】")
print(f"归一化后   w = [{m_s.coef_[0][0]:.3f}, {m_s.coef_[0][1]:.3f}]")
print("  → 两个权重在同一量级，两个特征都真正参与决策")
print("结论：SVM 靠'距离最近'找支持向量，跟 KNN 一样会被大数值特征绑架，必须先归一化。")
print("（注意：这份数据太简单，两边准确率可能都是 100%，所以别看准确率，看【线的方向和权重】）")

# ============================================
# 本课小结
# ============================================
print("\n" + "=" * 60)
print("小结：一句话记住 SVM")
print("=" * 60)
print("  SVM = 画一条线分开两类，并且让线离两边都尽量远（间隔最宽）；")
print("        真正决定这条线位置的，只有离它最近的几个点（支持向量）。")
print("  与 KNN 的区别：KNN 是'所有邻居投票'，SVM 只在乎'最近的那几个'。")
