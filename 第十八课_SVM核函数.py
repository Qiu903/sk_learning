# ============================================
# 第十八课：SVM 核函数 —— 直线切不开时怎么办
# 环境：skl_env
#
# 本课要讲清楚的一件事：
#   有些数据用直线（超平面）根本切不开（叫"线性不可分"）。
#   核函数的办法 = 偷偷把数据升到高维，在高维里用平面切开，
#                 投影回原来的维度看，边界就变成了弯曲的线。
#
# 本课结构（循序渐进）：
#   ① 造"环形"数据：内圈一类、外圈一类 —— 直线切不开
#   ② 用 linear 核试一下，看它怎么都切不对（画图）
#   ③ 手工升维：加一个新特征 z = x1² + x2²，用 3D 图看"一块板子切开"
#   ④ 用 rbf 核在 2 维里直接得到弯曲的决策边界（画图）
#   ⑤ rbf 的旋钮 gamma：0.1 / 1 / 100 三张图对比（欠拟合 → 刚好 → 过拟合）
# ============================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# matplotlib 默认字体不含中文，会显示成方框，所以指定微软雅黑
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================
# ① 造"环形"数据：内圈一类、外圈一类
# ============================================
print("=" * 60)
print("① 造环形数据：内圈是类别 0，外圈是类别 1")
print("=" * 60)

# make_circles：专门造"一大一小两个同心圆"的数据
#   n_samples=200  一共 200 个点
#   factor=0.4     内圈半径是小圈，0.4 表示内圈半径 = 0.4 × 外圈半径
#   noise=0.06     加一点点随机抖动，让数据更真实（不然太规整了）
X, y = make_circles(n_samples=200, factor=0.4, noise=0.06, random_state=42)

# 【踩坑提醒】make_circles 默认 y=0 是【外圈】、y=1 是【内圈】，跟直觉相反！
# 这里手动翻转一下，让 0=内圈、1=外圈，后面讲起来不容易绕晕
y = 1 - y

print(f"X 的形状 = {X.shape}  →  200 个样本，每个样本 2 个特征")
print(f"y 的取值 = {np.unique(y)}  →  0 = 内圈，1 = 外圈")
print("看点：内圈的点被外圈的点'包住'了，直线无论如何都切不开")


# ============================================
# ② 通用画图函数：把整个平面都拿去"问"模型
# ============================================
def plot_decision_region(ax, model, X_data, y_data, title):
    """在坐标轴 ax 上画出模型的"决策区域"（也就是它认为哪块地属于哪一类）

    思路（这是本课最需要理解的一段代码）：
      - 在平面上密密麻麻铺一张"棋盘"（网格），每个格子是一个坐标点
      - 把这些格子坐标全部丢给模型，问它"这个位置算哪一类"
      - 模型答完后，把答案按类别涂上不同颜色 → 就看到了边界形状

    进阶语法说明：
      np.meshgrid   把一个横轴数组 + 一个纵轴数组，交叉成一张二维网格
                    （可以理解为生成了棋盘上每个格子的行号和列号）
      .ravel()      把一个二维数组"拉直"成一维（铺成一长条）
      np.c_[]       把两个一维数组"上下并排"拼成两列的矩阵（每个样本一行）
      .reshape()    再变回二维，这样才画得回棋盘上
    """
    # 先圈定要画的区域：数据范围向外各留 0.3 的边
    x_min, x_max = X_data[:, 0].min() - 0.3, X_data[:, 0].max() + 0.3
    y_min, y_max = X_data[:, 1].min() - 0.3, X_data[:, 1].max() + 0.3

    # 横纵各取 300 个刻度，交叉成 300×300 的网格（9 万个格子）
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))

    # 把 9 万个格子坐标整理成"9 万行 × 2 列"的矩阵，喂给模型预测
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid)          # 每个格子的预测类别（0 或 1）
    Z = Z.reshape(xx.shape)          # 拉回棋盘形状，才能涂色

    # contourf = 按类别给区域填色，levels 表示"以 0.5 为分界切成两块"
    ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5],
                colors=["#dbeafe", "#ffedd5"])

    # 再把真实样本点画上去
    ax.scatter(X_data[y_data == 0, 0], X_data[y_data == 0, 1],
               c="#3b82f6", s=22, edgecolors="white", linewidths=0.5,
               label="类别 0（内圈）")
    ax.scatter(X_data[y_data == 1, 0], X_data[y_data == 1, 1],
               c="#f97316", s=22, edgecolors="white", linewidths=0.5,
               label="类别 1（外圈）")

    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="upper right")


# ============================================
# ③ 对比：linear 核（直线）vs rbf 核（曲线）
# ============================================
print("\n" + "=" * 60)
print("③ 同一个数据，换不同核函数试试")
print("=" * 60)

fig1, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5))

# 左边：linear 核 —— 只会画直线
m_linear = SVC(kernel="linear", C=1)
m_linear.fit(X, y)
acc_linear = m_linear.score(X, y)
plot_decision_region(axL, m_linear, X, y,
                     f"linear 核（只会画直线）\n准确率 {acc_linear:.1%}")

# 右边：rbf 核 —— 能画弯曲的边界
m_rbf = SVC(kernel="rbf", C=1, gamma=1)
m_rbf.fit(X, y)
acc_rbf = m_rbf.score(X, y)
plot_decision_region(axR, m_rbf, X, y,
                     f"rbf 核（能画曲线）\n准确率 {acc_rbf:.1%}")

plt.tight_layout()
plt.show()

print(f"linear 核准确率：{acc_linear:.1%}   ← 背景色分不干净，它只会画直直的线")
print(f"rbf    核准确率：{acc_rbf:.1%}   ← 背景色干净地分成一环一环")
print("结论：数据'线性不可分'时，换 rbf 核就能解决")

# ============================================
# ④ 手工升维：亲眼看见"二维切不开，三维一块板子切开"
# ============================================
print("\n" + "=" * 60)
print("④ 核函数背后的原理：升维。我们手工来做一次")
print("=" * 60)

# 造一个新特征 z：到原点距离的平方
#   内圈的点离原点近 → z 小；外圈的点离原点远 → z 大
z = X[:, 0] ** 2 + X[:, 1] ** 2

print(f"内圈（类别 0）的 z 范围：{z[y == 0].min():.3f} ~ {z[y == 0].max():.3f}")
print(f"外圈（类别 1）的 z 范围：{z[y == 1].min():.3f} ~ {z[y == 1].max():.3f}")
print("看：内圈的 z 明显比外圈小 —— 这就是能分开的原因！")

# 画 3D 图：横轴特征1、纵轴特征2、竖轴这个新特征 z
fig2 = plt.figure(figsize=(8, 6))
# 进阶：projection="3d" 表示"我要一个三维坐标系"
ax3 = fig2.add_subplot(projection="3d")

# 内圈的点（z 小，在下面）
ax3.scatter(X[y == 0, 0], X[y == 0, 1], z[y == 0],
            c="#3b82f6", s=25, label="类别 0（内圈，z 小）")
# 外圈的点（z 大，在上面）
ax3.scatter(X[y == 1, 0], X[y == 1, 1], z[y == 1],
            c="#f97316", s=25, label="类别 1（外圈，z 大）")

# 在两类中间的高度插一块"水平板子"，看它是不是刚好把两群点上下分开
thr = (z[y == 0].max() + z[y == 1].min()) / 2   # 取两边 z 的中间值当板子高度
gx = np.linspace(X[:, 0].min(), X[:, 0].max(), 10)
gy = np.linspace(X[:, 1].min(), X[:, 1].max(), 10)
GX, GY = np.meshgrid(gx, gy)                          # 板子的网格
GZ = np.full_like(GX, thr)                            # 板子高度处处等于 thr
ax3.plot_surface(GX, GY, GZ, alpha=0.35, color="#16a34a")

ax3.set_xlabel("特征 1")
ax3.set_ylabel("特征 2")
ax3.set_zlabel("新特征 z = 特征1² + 特征2²")
ax3.set_title("升维后：一块水平板子（平面）就把两类分开了")
ax3.legend(fontsize=8)
plt.show()

print("看懂这张 3D 图，就懂了核函数：")
print("  原来在 2 维里，两类点挤在一个平面上、互相包住 → 切不开；")
print("  加了 z 这个特征后，两类点被'拉'到不同高度 → 一块水平板子就切开了。")
print("  把这块板子投影回 2 维看，它的形状就是一个【圆】——")
print("  这就是右边 rbf 核那条弯曲边界是怎么来的。")

# ============================================
# ⑤ rbf 的旋钮 gamma：控制边界"弯得有多夸张"
# ============================================
print("\n" + "=" * 60)
print("⑤ rbf 核的旋钮 gamma：0.1 / 1 / 100 对比")
print("=" * 60)

# 上面那份数据太干净了（噪声很小），三种 gamma 都能做到 100%，看不出差别。
# 所以这里换一份"噪声更大"的数据：让点互相掺一点，区别才看得出来。
X_g, y_g = make_circles(n_samples=300, factor=0.4, noise=0.3, random_state=42)
y_g = 1 - y_g    # 同样翻转，保持 0=内圈、1=外圈

# 这次分训练集/测试集：训练集用来学，测试集用来看"是不是真的学会了"
Xg_tr, Xg_te, yg_tr, yg_te = train_test_split(
    X_g, y_g, test_size=0.3, random_state=42
)

# C 管"容忍度"，gamma 管"边界的弯曲程度/复杂程度"，都是旋钮
gammas = [0.1, 1, 100]
fig3, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, g in zip(axes, gammas):
    m = SVC(kernel="rbf", C=1, gamma=g)
    m.fit(Xg_tr, yg_tr)                     # 只在训练集上学习
    tr = m.score(Xg_tr, yg_tr)              # 训练集表现
    te = m.score(Xg_te, yg_te)              # 测试集表现
    # 画的时候把全部点都画上，方便看边界形状
    plot_decision_region(ax, m, X_g, y_g,
                         f"gamma = {g}\n训练 {tr:.0%} / 测试 {te:.0%}")
    print(f"gamma = {g:>5}  训练准确率 = {tr:.2%}   测试准确率 = {te:.2%}")

plt.tight_layout()
plt.show()

print("看图要点：")
print("  gamma 太小(0.1)：边界太圆滑、几乎又变成一条直线 → 训练/测试都低 → 欠拟合")
print("  gamma 适中(1)  ：边界正好贴合内外圈的形状 → 两边都最高 → 刚好")
print("  gamma 太大(100)：边界扭来扭去、围着每个点画小圈 → 训练高但测试掉下来 → 过拟合")
print("  记忆法：gamma 越大，模型越'敏感'，越容易过拟合；gamma 太小则欠拟合")

# ============================================
# 三种核的横向对比
# ============================================
print("\n" + "=" * 60)
print("三种核在同一份环形数据上的表现")
print("=" * 60)
for name, kernel in [("linear（直线）", "linear"),
                     ("poly（多项式曲线）", "poly"),
                     ("rbf（任意曲线）", "rbf")]:
    m = SVC(kernel=kernel, C=1, gamma=1)
    m.fit(X, y)
    print(f"  {name:<20} 训练准确率 = {m.score(X, y):.2%}")

print("\n实用建议：")
print("  1) 不确定用哪个时，先试 rbf，它几乎总是能画弯曲边界")
print("  2) 数据特征特别多（比如上千维）时，linear 常常反而更好、更快")
print("  3) 无论用哪个核，都别忘了先归一化（第十七课讲过原因）")

# ============================================
# 本课小结
# ============================================
print("\n" + "=" * 60)
print("小结")
print("=" * 60)
print("  · 直线切不开的数据叫【线性不可分】")
print("  · 核函数的办法 = 升维：高维里用平面切开，投影回来就是弯曲边界")
print("  · 【核技巧】= 不在计算上真的升维，但结果等价，省算力")
print("  · 常用核：linear(直线) / poly(曲线) / rbf(万能曲线，最常用)")
print("  · rbf 的旋钮 gamma：越大越敏感、越容易过拟合；太小则欠拟合")
