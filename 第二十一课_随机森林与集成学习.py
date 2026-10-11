# ============================================
# 第二十一课：随机森林与集成学习 —— 三个臭皮匠，顶个诸葛亮
# 环境：skl_env
#
# 【为什么讲这一课？】
#   第三课种下的第一棵决策树，有个毛病：
#   只要不限制深度，让它自由生长，它会把训练集背得滚瓜烂熟，
#   连数据里的噪声都当成真理记住，一到测试集就露馅 —— 这就是过拟合。
#
#   有没有办法既保留决策树的优点（直观、不用做量纲处理），
#   又让它别那么"死记硬背"？
#   答案：一个人不靠谱，那就组一个【团队】来投票。
#   这就是本课的主角 —— 集成学习，以及它最出名的成员：随机森林。
#
#   本课路线（由简到难）：
#       ① 看单棵树的病
#       ② 集成学习是什么、为什么有效
#       ③ 揭开 Bagging 抽样的秘密
#       ④ 随机森林的两重随机
#       ⑤~⑧ 四个实验逐个跑
#       ⑨ 袋外评分（OOB）
#       ⑩ 收官总结 + 顺带认识 Boosting
# ============================================

import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_classification, load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier              # 进阶：本课主角，集成学习家族的成员
from sklearn.model_selection import train_test_split, cross_val_score

# matplotlib 默认字体不含中文，会显示成方框，所以指定微软雅黑
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

COLOR_MAIN = "#3b82f6"     # 蓝
COLOR_WARN = "#ef4444"     # 红
COLOR_OK = "#16a34a"       # 绿
COLOR_GRAY = "#9ca3af"     # 灰


# ============================================
# ① 先看单棵决策树的『病』：记得太死，反而考不好
# ============================================
print("=" * 68)
print("① 先看单棵决策树的『病』：记得太死，反而考不好")
print("=" * 68)
print("回忆第三课：决策树靠一连串『这个值大于某个数吗？』把数据切开。")
print("它直观、好解释，还不用做量纲处理，但有致命弱点：")
print("  你若不给它加限制，它会一直切下去，直到每个训练样本都被单独分对 ——")
print("  连噪声都背了下来。训练集分数逼近 100%，测试集却掉链子，这就是【过拟合】。")
print()

# 【重点】造一份"两个交错月牙"的数据，再故意加噪声，专门考验模型会不会硬背
X, y = make_moons(n_samples=400, noise=0.35, random_state=0)     # 进阶：noise 越大，两类越犬牙交错、越难分
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)

tree = DecisionTreeClassifier(random_state=42)                   # 不传 max_depth，等于让它无限生长
tree.fit(X_tr, y_tr)
print("一棵无限深的决策树：")
print(f"  训练集准确率 {tree.score(X_tr, y_tr):.2%}   ← 几乎满分，因为它把训练集背下来了")
print(f"  测试集准确率 {tree.score(X_te, y_te):.2%}   ← 一遇新题就现原形")
print()
print("这个落差（背得住旧题、做不对新题）就是过拟合的典型症状。")
print("第九课我们用『剪枝』（限制 max_depth）治它，今天我们换一种更聪明的治法：组队。")


# ============================================
# ② 集成学习：一个人可能犯错，一群人投票就靠谱多了
# ============================================
print("\n" + "=" * 68)
print("② 集成学习（Ensemble）：把多个模型组队，靠『投票』做决定")
print("=" * 68)
print("【名词】集成学习：不指望单个模型完美，而是训练出【一群】模型，")
print("  让它们各自表态，最后按『少数服从多数』投票出结果。")
print()
print("为什么一群平庸的模型能胜过单个？先别急着信，我们用数学算给你看：")
print("  假设有 5 个互相独立的分类器，每个单独只有 60% 准确率（刚过及格线），")
print("  5 个人投票，只要【至少 3 个猜对】，结果就算对。")
print("  这个概率是多少？来算一算。")

# 【重点】用二项分布算"多数票正确"的概率：C(n,k) 是"从 n 个里挑 k 个"的组合数
def majority_acc(p, n_voters):
    """n_voters 个独立分类器、各自准确率 p，多数票（严格过半）后的整体准确率"""
    return sum(math.comb(n_voters, k) * p ** k * (1 - p) ** (n_voters - k)
               for k in range(n_voters // 2 + 1, n_voters + 1))   # 进阶：只累加"过半数正确"的那些情况

print(f"  5 个 60% 的模型投票  → 整体准确率 {majority_acc(0.6, 5):.2%}")
print(f"  51 个 60% 的模型投票 → 整体准确率 {majority_acc(0.6, 51):.2%}")
print()
print("看，单个 60% 的『臭皮匠』，组队之后战斗力飙升。")
print("前提只有一个字：【独立】。如果这 5 个人想法完全一样（一起对、一起错），")
print("那投票毫无意义 —— 这是集成的命门，后面随机森林的两重随机就是为它服务的。")

# 图1：投票人数越多，整体准确率越趋近 100%
voters = np.arange(1, 100, 2)                                    # 只取奇数个，避免出现平票
fig1, ax1 = plt.subplots(figsize=(9, 5))
for p, color in [(0.55, COLOR_WARN), (0.60, COLOR_MAIN), (0.70, COLOR_OK)]:
    accs = [majority_acc(p, n) for n in voters]                  # 进阶：列表推导式，一行算出所有投票人数下的准确率
    ax1.plot(voters, accs, marker="o", markersize=3, linewidth=1.8, color=color,
             label=f"单个模型准确率 {p:.0%}")
ax1.axhline(0.5, color=COLOR_GRAY, linestyle=":", linewidth=1.4)
ax1.text(58, 0.52, "瞎猜水平 50%", color="#6b7280", fontsize=10)
ax1.set_xlabel("团队里的模型个数（个）")
ax1.set_ylabel("投票后的整体准确率")
ax1.set_title("图1  三个臭皮匠定理：个体只要稍微好过瞎猜，组队后越投越准", fontsize=13)
ax1.legend(fontsize=11)
ax1.grid(alpha=0.3)
fig1.tight_layout()
plt.show()

print("\n注意两条规律：")
print("  · 个体准确率 > 50%（比瞎猜强一点点），队伍越大，整体越接近 100%；")
print("  · 个体准确率 < 50%（还不如瞎猜），组队反而越来越差。")
print("  所以集成的铁律是：成员必须是『比瞎猜强一点的弱模型』，且彼此独立。")


# ============================================
# ③ Bagging：随机森林的"抽签"智慧
# ============================================
print("\n" + "=" * 68)
print("③ Bagging 的第一招：有放回抽样（Bootstrap）")
print("=" * 68)
print("要让成员彼此独立，最直接的办法是：让每棵树只看一份【不同的】数据。")
print("可我们手上只有一份数据，怎么变出多份？答案是【有放回抽样】。")
print()
print("【名词】Bootstrap（自助抽样）：从 N 个样本里随机抽 N 个，")
print("  每抽一个就【放回去】，所以同一个样本可能被抽中好几次，")
print("  也可能一次都没被抽到。这样每棵树拿到的都是不同的『数据版本』。")

np.random.seed(42)
demo = np.random.randint(0, 10, 10)                              # 从 0~9 里抽 10 个，有放回
print(f"  小演示：从 10 个样本里抽 10 次 → 抽中编号 {list(demo)}")
print(f"  里面只有 {len(set(demo))} 个不同的样本，其余的都被重复抽中了。")
print()
print("  那平均有多少样本会被抽中？当 N 很大时有个漂亮的结论：")
for N in [10, 50, 100, 300, 1000, 10000]:
    ratio = 1 - (1 - 1 / N) ** N                                 # 进阶：一个样本一次都没被抽中的概率是 (1-1/N)^N
    print(f"    样本总数 {N:>6} → 平均被抽中的比例约 {ratio:.2%}")
print("  无论 N 多大，这个比例都稳定在 63.2% 附近。")
print("  也就是说：每棵树大约只见到 63% 的数据，剩下那约 37%『从没被它见过』的样本，")
print("  被称为【袋外样本 OOB】，第⑨节我们会拿它免费当验证集用。")


# ============================================
# ④ 随机森林 = Bagging 决策树 + 第二重随机
# ============================================
print("\n" + "=" * 68)
print("④ 随机森林的『两重随机』：这才是它比普通 Bagging 更厉害的地方")
print("=" * 68)
print("【名词】随机森林（Random Forest）：")
print("  就是用 Bagging 的方式，打包出一大群决策树，让它们投票。")
print("  它在普通 Bagging 之上，又加了一重随机，一共两重：")
print()
print("  第一重：样本随机 —— 每棵树用 Bootstrap 抽到的不同数据训练（第③节讲的）。")
print("  第二重：特征随机 —— 每到一次分裂，不让它看全部特征，")
print("          只从随机挑出的一部分特征里选『最好的那个』来切。")
print()
print("  为什么还要第二重？举个例子：假设有 1 个特征特别强，")
print("  若每次分裂都能看到它，那所有树都会抢着用它，长出来就会长得一模一样，")
print("  投票等于一个人投 N 次，白搭。")
print("  强制每次只看一部分特征，就逼着树去发掘不同特征，树与树的差异被拉开，")
print("  『独立性』保住了，投票才有意义 —— 这正是第②节说的那个命门。")
print()
print("  另外，随机森林完整继承了决策树的一个大优点：")
print("  【不用做量纲处理】。它比较的是大小顺序，收入除以一万它照样切，结论不变。")


# ============================================
# ⑤ 实验一：先看"长相"——单棵树 vs 随机森林的决策边界
# ============================================
print("\n" + "=" * 68)
print("⑤ 实验一：先看边界长相 —— 单棵树 vs 随机森林")
print("=" * 68)

# 画决策边界的公共函数：同一片网格上，让模型给每个点预测颜色
def plot_boundary(ax, model, X, y, title):
    # 进阶：meshgrid 生成一整片坐标网格，把每个格点喂给模型预测，就能画出一条分界线
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])             # 进阶：np.c_ 把两列坐标拼成"坐标对"再交给模型
    ax.contourf(xx, yy, Z.reshape(xx.shape), alpha=0.25, cmap="coolwarm")
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=15, edgecolor="k", linewidth=0.3)
    ax.set_title(title, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])

forest = RandomForestClassifier(n_estimators=100, random_state=42)   # n_estimators=100 表示种 100 棵树
forest.fit(X_tr, y_tr)
print(f"月牙数据（训练集 {len(X_tr)} 人 / 测试集 {len(X_te)} 人）：")
print(f"  单棵树   ：训练 {tree.score(X_tr, y_tr):.1%} / 测试 {tree.score(X_te, y_te):.1%}")
print(f"  随机森林 ：训练 {forest.score(X_tr, y_tr):.1%} / 测试 {forest.score(X_te, y_te):.1%}")
print()
print("月牙数据很简单，两者分数只差 1 个百分点，肉眼看不出高下。")
print("但把【决策边界】画出来，区别就藏不住了 —— 看下图。")

fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5.5))
plot_boundary(axes2[0], tree, X, y, f"单棵决策树\n训练 {tree.score(X_tr, y_tr):.1%} / 测试 {tree.score(X_te, y_te):.1%}")
plot_boundary(axes2[1], forest, X, y, f"随机森林（100 棵）\n训练 {forest.score(X_tr, y_tr):.1%} / 测试 {forest.score(X_te, y_te):.1%}")
fig2.suptitle("图2  左：单棵树的边界碎成锯齿，追着噪声跑；右：森林投票后边界平滑干净", fontsize=14)
fig2.tight_layout()
plt.show()

print("左图的边界坑坑洼洼，为了迁就个别噪声点，硬是绕了个小弯 —— 高方差的表现。")
print("右图 100 棵树各自犯错、互相拉平，边界就顺滑多了。")
print("数据越复杂、噪声越多，这个『平滑』带来的好处就越明显，实验二用数字说话。")


# ============================================
# ⑥ 实验二：把难度升级 —— 高维带噪声数据上，森林为什么更强？
# ============================================
print("\n" + "=" * 68)
print("⑥ 实验二：难度升级 —— 高维 + 标签错误的真实感数据")
print("=" * 68)
print("月牙太简单，换一份更接近现实的『模拟数据』：")
print("  15 个特征（其中 8 个真正有用）、1000 个样本，")
print("  并且故意把 8% 的标签【改错】（模拟现实中录错、标错的脏数据）。")
print("  这种被污染的数据，正是单棵树的死穴 —— 它会把错的也背下来。")

# 进阶：make_classification 是"造分类数据"的工厂；flip_y=0.08 表示随机翻转 8% 的标签（制造噪声）
Xh, yh = make_classification(n_samples=1000, n_features=15, n_informative=8,
                             n_redundant=0, n_clusters_per_class=2,
                             class_sep=1.0, flip_y=0.08, random_state=3)
Xh_tr, Xh_te, yh_tr, yh_te = train_test_split(Xh, yh, test_size=0.3, random_state=42)

tree_h = DecisionTreeClassifier(random_state=42)
tree_h.fit(Xh_tr, yh_tr)
forest_h = RandomForestClassifier(n_estimators=200, random_state=42)   # 这次种 200 棵，投得更稳
forest_h.fit(Xh_tr, yh_tr)

print(f"\n{'模型':<20}{'训练集准确率':>12}{'测试集准确率':>14}")
print("-" * 48)
print(f"{'单棵决策树(无限深)':<18}{tree_h.score(Xh_tr, yh_tr):>12.2%}{tree_h.score(Xh_te, yh_te):>14.2%}")
print(f"{'随机森林(200棵树)':<19}{forest_h.score(Xh_tr, yh_tr):>12.2%}{forest_h.score(Xh_te, yh_te):>14.2%}")
print()
print("这次差距就大了：两者训练集都满分（都背下了训练集），")
print(f"但测试集上森林 {forest_h.score(Xh_te, yh_te):.1%} 明显甩开单棵树 {tree_h.score(Xh_te, yh_te):.1%}。")
print("原因：训练集里那 8% 被改错的标签，单棵树当成真理死死记住；")
print("      森林里每棵树只看到一部分数据、一部分特征，各自记住的错各不相同，")
print("      投票时这些零散的错被互相抵消，正确答案反而浮了上来。")

print("\n【顺带把 n_estimators 的规律也跑了】树越多，真的越准吗？")
n_list = [1, 3, 5, 10, 20, 50, 100, 200, 300]
train_scores, test_scores = [], []
for k in n_list:
    rf = RandomForestClassifier(n_estimators=k, random_state=42)
    rf.fit(Xh_tr, yh_tr)
    train_scores.append(rf.score(Xh_tr, yh_tr))
    test_scores.append(rf.score(Xh_te, yh_te))

print(f"\n{'树的数量':>8}{'训练集':>12}{'测试集':>12}")
print("-" * 34)
for k, tr, te in zip(n_list, train_scores, test_scores):
    print(f"{k:>10}{tr:>12.2%}{te:>12.2%}")
print("\n规律很清楚：树从 1 棵加到几十棵，测试集分数一路爬升；")
print("          过了 100 棵基本就躺平了，再加只是白白多花时间。")
print("          所以经验值：n_estimators 取 100~300 通常就够。")

fig3, ax3 = plt.subplots(figsize=(9, 5))
ax3.plot(n_list, train_scores, marker="o", color=COLOR_WARN, linewidth=1.8, label="训练集准确率")
ax3.plot(n_list, test_scores, marker="s", color=COLOR_OK, linewidth=1.8, label="测试集准确率")
ax3.set_xscale("log")                                            # 进阶：log 让 1/3/10/100 间距均匀，曲线更好看
ax3.set_xticks(n_list)
ax3.set_xticklabels(n_list)
ax3.set_xlabel("树的数量 n_estimators（对数刻度）")
ax3.set_ylabel("准确率")
ax3.set_title("图3  树越多越准，但收益递减：一百棵之后基本躺平", fontsize=13)
ax3.legend(fontsize=11)
ax3.grid(alpha=0.3)
fig3.tight_layout()
plt.show()


# ============================================
# ⑦ 实验三：第二重随机的力度 —— max_features
# ============================================
print("\n" + "=" * 68)
print("⑦ 实验三：第二重随机的力度 —— max_features")
print("=" * 68)
print("【名词】max_features：每次分裂时，允许模型考虑的特征个数。")
print("  它就是第④节『第二重随机』的力度旋钮：")
print("      'sqrt' → 每次只看 根号下(特征总数) 个特征（分类任务的默认值）")
print("      'log2' → 每次只看 log2(特征总数) 个特征，随机性更强")
print("      None   → 每次看全部特征，等于关掉第二重随机，退化成普通 Bagging")
print()
print("还是用刚才那份 15 特征的数据，看旋钮拨到不同档位的效果：")

print(f"\n{'max_features':>16}{'5 折平均准确率':>16}")
print("-" * 32)
for mf in ["sqrt", "log2", None]:
    rf = RandomForestClassifier(n_estimators=100, max_features=mf, random_state=42)
    score = cross_val_score(rf, Xh, yh, cv=5).mean()
    label = "None（用全部）" if mf is None else mf
    print(f"{label:>18}{score:>16.2%}")
print()
print("看，把第二重随机关掉（None，每次都看全部 15 个特征），分数反而掉了一点。")
print("这说明『每次只挑一小撮特征』不是花架子，它确实靠拉开树之间的差异帮上了忙。")
print("（而且在特征成百上千时，它还能大幅提速。）")


# ============================================
# ⑧ 实验四：随机森林的附赠能力 —— 特征重要性
# ============================================
print("\n" + "=" * 68)
print("⑧ 实验四：特征重要性（feature_importances_）")
print("=" * 68)
print("随机森林有个免费附赠的超能力：能告诉你每个特征对判断有多重要。")
print("原理不复杂 —— 一个特征在树里被用来分裂时，越能一次把两类分干净，")
print("它得到的『贡献分』就越高；把所有树的贡献加起来一平均，就是特征重要性。")
print()
print("拿乳腺癌数据（30 个特征）实际跑一遍：")

bc = load_breast_cancer()
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(bc.data, bc.target, test_size=0.25, random_state=42)
rf_bc = RandomForestClassifier(n_estimators=200, random_state=42)
rf_bc.fit(Xc_tr, yc_tr)
print(f"  随机森林在乳腺癌测试集上的准确率：{rf_bc.score(Xc_te, yc_te):.2%}")

imp = rf_bc.feature_importances_
top10 = np.argsort(imp)[::-1][:10]                               # 进阶：argsort 返回排序下标，[::-1] 翻转成从大到小
print("\n  贡献最大的 10 个特征：")
for rank, i in enumerate(top10, 1):                              # 进阶：enumerate(..., 1) 让序号从 1 开始
    print(f"    {rank:>2}. {bc.feature_names[i]:<24} 重要性 {imp[i]:.3f}")

fig4, ax4 = plt.subplots(figsize=(9, 6))
names = [bc.feature_names[i] for i in top10]
vals = [imp[i] for i in top10]
ax4.barh(range(10), vals[::-1], color=COLOR_MAIN, alpha=0.85)    # 进阶：barh 画横向条形图
ax4.set_yticks(range(10))
ax4.set_yticklabels(names[::-1], fontsize=10)                    # 反过来排，让最长的条在最上方
ax4.set_xlabel("特征重要性")
ax4.set_title("图4  乳腺癌数据中随机森林选出的 Top-10 重要特征", fontsize=13)
ax4.grid(axis="x", alpha=0.3)
fig4.tight_layout()
plt.show()

print("\n实际用途：这份排名能帮你判断『哪些指标真正值得测』。")
print("若某特征重要性常年垫底，收集它的意义就不大，可以考虑删掉，给模型减负。")


# ============================================
# ⑨ 袋外评分 OOB：不额外切数据，就能白捡一个验证集
# ============================================
print("\n" + "=" * 68)
print("⑨ 袋外评分（OOB Score）：Bagging 送的免费验证集")
print("=" * 68)
print("还记得第③节的袋外样本吗？每棵树训练时大约 37% 的数据它没见过。")
print("把每棵树用它『没见过的那些样本』单独考一次，再汇总，")
print("就得到一份几乎等于交叉验证的评分 —— 而且不用额外切数据、不用多训一遍。")

rf_oob = RandomForestClassifier(n_estimators=200, oob_score=True, random_state=42)   # 进阶：打开 oob_score 开关
rf_oob.fit(Xc_tr, yc_tr)
print(f"\n  袋外评分 OOB Score ：{rf_oob.oob_score_:.2%}   ← 白捡的验证分")
print(f"  真正测试集准确率    ：{rf_oob.score(Xc_te, yc_te):.2%}   ← 两者非常接近")
print("\n两者接近，说明 OOB 是个可靠的『把脉』工具，")
print("数据不够多、舍不得再切验证集时，打开它就能顺手看一眼模型好坏。")


# ============================================
# ⑩ 收官：本课总结 + 顺带认识 Boosting
# ============================================
print("\n" + "=" * 68)
print("⑩ 收官总结")
print("=" * 68)
print("  1. 集成学习：单模型可能犯错，组队投票更稳（前提高于瞎猜、且彼此独立）")
print("  2. Bagging：有放回抽样（Bootstrap）制造多份不同的数据，每份约含 63% 样本")
print("  3. 随机森林 = Bagging 决策树 + 两重随机（样本随机 + 特征随机）")
print("  4. 关键参数：n_estimators（树数，一百棵后收益递减）")
print("               max_features（特征随机力度，分类默认 'sqrt'）")
print("  5. 附赠能力：feature_importances_ 看特征重要性；OOB 免费验证")
print("  6. 别忘优点：随机森林和决策树一样，不用做量纲处理")
print()
print("【顺带认识另一个流派 —— Boosting】")
print("  今天讲的随机森林属于 Bagging 流派：一堆树【平行地】【各自独立】地长，然后投票。")
print("  集成学习还有另一大流派 Boosting（代表：梯度提升树、XGBoost、LightGBM）：")
print("    它的树是【串行】的 —— 后一棵树专门去补前一棵树的错，一代代接力纠错。")
print("    通常分数更高，但更爱过拟合、更吃调参，是我们后面要单独开一课的主角。")
print()
print("  一句话对比：Bagging 靠『集思广益』降方差，Boosting 靠『接力纠错』降偏差。")