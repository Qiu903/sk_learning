# ============================================
# 第二十课：数据预处理与特征工程 —— 把"生数据"煮成模型能吃的"熟饭"
# 环境：skl_env
#
# 【为什么突然插一课讲预处理？】
#   回头看我们走过的路：鸢尾花、糖尿病、乳腺癌、月牙……
#   这些数据有个共同点 —— sklearn 早就替你洗好了：
#       全是数字、没有空白、量纲也都差不多。
#   所以你才能一路顺顺利利地 .fit() / .score()。
#
#   可现实里，从公司数据库导出来的表长这样：
#       年龄   月收入    城市    会员等级   是否续费
#       23    8000     北京    黄金       1
#       空    12000    上海    白银       0
#       45    520000   广州    普通       1
#
#   三个致命问题一眼可见：
#       ① 有空白（缺失值）—— 模型碰到空值直接报错
#       ② 单位不统一（月收入几万，年龄两位数）—— 距离类模型被大数字霸凌
#       ③ 有文字（城市、等级）—— 模型只认数字，不认汉字
#
#   预处理做不好，后面换再厉害的模型都是白搭。
#   本课按四步走，一步一张图：
#       缺失值 → 量纲 → 文字 → 串成流水线（顺带揭穿"数据泄露"）
# ============================================

import numpy as np
import pandas as pd                                        # 进阶：表格处理库，本课第一次正式用它
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer                   # 进阶：专门负责"填缺失值"
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif  # 进阶：特征选择，第⑤节演示数据泄露要用
from sklearn.compose import ColumnTransformer              # 进阶：让不同的列走不同的预处理流程
from sklearn.pipeline import Pipeline                      # 进阶：把多个步骤焊成一条流水线
from sklearn.model_selection import train_test_split, cross_val_score

# matplotlib 默认字体不含中文，会显示成方框，所以指定微软雅黑
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

COLOR_MAIN = "#3b82f6"     # 蓝
COLOR_WARN = "#ef4444"     # 红
COLOR_OK = "#16a34a"       # 绿


# ============================================
# ① 先造一份"脏数据"：真实世界的数据就长这样
# ============================================
print("=" * 68)
print("① 造一份脏数据（300 位会员，用来演示三个问题）")
print("=" * 68)

np.random.seed(42)          # 固定随机种子，保证每次跑出来的数据一模一样
n = 300

age = np.random.normal(36, 9, n).clip(18, 68).round(0)          # 年龄：正态分布，裁到 18~68
income = np.random.lognormal(mean=9.8, sigma=0.45, size=n).round(0)  # 收入：对数正态，天然右偏（多数人几千~几万）
# 三个超级大客户（也可能是录入错误），专门用来演示"均值会被极端值带跑偏"
income[:3] = [520000, 480000, 450000]

city = np.random.choice(["北京", "上海", "广州"], n)              # 城市：文字，无序
level = np.random.choice(["普通", "白银", "黄金"], n, p=[0.5, 0.35, 0.15])

# 【重点】造标签"是否续费"。故意让城市的影响很大、而且没有大小顺序：
#   上海 88% 续费、北京 15% 续费、广州 82% 续费        ← 注意：高-低-高，不单调
#   这样后面讲"文字编码"时，错误做法的后果才会暴露得非常明显。
city_prob = {"上海": 0.88, "北京": 0.15, "广州": 0.82}
level_bonus = {"普通": -0.05, "白银": 0.02, "黄金": 0.08}
p_renew = np.array([city_prob[c] for c in city]) + np.array([level_bonus[l] for l in level])
p_renew = p_renew.clip(0.02, 0.98)                              # 概率裁剪到合法区间
renew = (np.random.rand(n) < p_renew).astype(int)               # 按概率掷骰子，得到 0/1 标签

df = pd.DataFrame({
    "年龄": age,
    "月收入": income,
    "城市": city,
    "会员等级": level,
    "是否续费": renew,
})

# 【重点】故意挖几个坑：随机挑一些人，把年龄/收入/会员等级涂成空白
#    进阶：np.random.choice 从下标里抽样，df.loc[行下标, "列名"] = np.nan 精准置空
miss_idx = np.random.choice(n, 45, replace=False)
df.loc[miss_idx[:20], "年龄"] = np.nan
df.loc[miss_idx[20:38], "月收入"] = np.nan
df.loc[miss_idx[38:45], "会员等级"] = np.nan

print(df.head(6).to_string())            # 进阶：to_string() 让表格完整打印，不被省略号截断
print(f"\n一共 {len(df)} 行，{df.shape[1]} 列（4 个特征 + 1 个标签）")
print("\n看看每一列缺了多少（进阶：df.isnull().sum() 统计每列空值数量）：")
print(df.isnull().sum().to_string())
print("\n注意三件事：")
print("  · 年龄/月收入/会员等级都有空白 —— 模型碰到空白会直接报错")
print("  · 月收入动辄几万，年龄只有两位数 —— 单位不统一")
print("  · 城市、会员等级是汉字 —— 模型只认数字")
print("这三件事，就是本课要全部解决的。")


# ============================================
# ② 缺失值：不能一删了之，要"填"得聪明
# ============================================
print("\n" + "=" * 68)
print("② 缺失值处理：为什么不能删？怎么填才不吃亏？")
print("=" * 68)
print("最直觉的做法：谁有空值就把那一行删掉。")
print("  代价：45 行有坑，至少要删 45 行，白白丢掉 15% 的样本。")
print("  样本就是信息，删一行就少一份经验，所以更常用的做法是【填】。")
print()
print("【名词】填补（Imputation）：把空值替换成一个合理的数。")
print("  最常用的三种策略：")
print("    strategy='mean'          填这一列的【平均值】")
print("    strategy='median'        填这一列的【中位数】")
print("    strategy='most_frequent' 填这一列的【众数】，也就是出现最多的值（给文字列用）")
print()

# 【重点】均值和中位数只差一个字，遇到极端值时的表现却天差地别
income_notnull = df["月收入"].dropna()      # 进阶：dropna() 丢掉空值，只留有效数字
mean_val = income_notnull.mean()
median_val = income_notnull.median()
print(f"月收入这一列：均值 = {mean_val:,.0f} 元，中位数 = {median_val:,.0f} 元")
print(f"  差了 {mean_val - median_val:,.0f} 元，为什么？")
print(f"  因为刚才那三个几十万的极端值，把【均值】硬生生拽高了，")
print(f"  而【中位数】是按排队站在正中间的那个人，极端值站多远都影响不了它。")
print()
print("  所以：收入、房价这类'少数大户能拉爆全场'的数据，中位数更稳健（robust）。")
print("  换句话说 —— 用均值填缺失，等于把大多数人的收入悄悄抬高了。")

# 图1：收入分布 + 均值线 + 中位数线
fig1, ax1 = plt.subplots(figsize=(9, 5))
ax1.hist(income_notnull, bins=40, color=COLOR_MAIN, alpha=0.75, edgecolor="white")
ax1.axvline(mean_val, color=COLOR_WARN, linestyle="--", linewidth=2.2, label=f"均值 {mean_val:,.0f}")
ax1.axvline(median_val, color=COLOR_OK, linestyle="-", linewidth=2.2, label=f"中位数 {median_val:,.0f}")
ax1.set_title("图1  月收入分布：3 个极端值把均值拉到了右边，中位数稳如泰山", fontsize=13)
ax1.set_xlabel("月收入（元）")
ax1.set_ylabel("人数")
ax1.legend(fontsize=11)
fig1.tight_layout()
plt.show()

print("\n现在正式动手填：数字列用中位数，文字列用众数。")
imputer_num = SimpleImputer(strategy="median")
df["月收入"] = imputer_num.fit_transform(df[["月收入"]])   # 进阶：fit_transform 一步完成"算中位数 + 填进去"
df["年龄"] = SimpleImputer(strategy="median").fit_transform(df[["年龄"]])
df["会员等级"] = SimpleImputer(strategy="most_frequent").fit_transform(df[["会员等级"]]).ravel()
# 进阶：上面 .ravel() 把"二维的列"压回"一维"，因为 fit_transform 永远返回二维表格
print(f"填补完成，现在还剩 {df.isnull().sum().sum()} 个空值")
print("（进阶提醒：fit 算出来的中位数要【存下来】，将来新数据必须用同一个中位数填，")
print("  这也是后面 Pipeline 存在的原因之一。）")


# ============================================
# ③ 量纲统一：标准化 vs 归一化，它俩不是一回事
# ============================================
print("\n" + "=" * 68)
print("③ 量纲处理：标准化 vs 归一化（先把这两个词彻底分清）")
print("=" * 68)
print("【为什么必须处理】")
print("  月收入范围 5000~520000，年龄范围 18~68。")
print("  计算「两点的距离」时，收入因子动辄十几万，年龄因子最多差 50。")
print("  结果：距离几乎只由收入决定，年龄直接被无视。")
print("  所以 KNN、SVM 这类【靠距离】的模型，不做这一步就是废的。")
print()
print("【两个词的区别（很多人混着叫，其实不一样）】")
print("  标准化 StandardScaler：(x - 均值) / 标准差")
print("      → 结果均值变 0、标准差变 1，范围不固定，可能有负数。这是 z-score 打分。")
print("  归一化 MinMaxScaler：(x - 最小值) / (最大值 - 最小值)")
print("      → 结果被硬压到 [0, 1] 区间，但一旦有极端值，其他人全被挤成一团。")
print()
print("  口语里大家常把两者都叫'归一化'。前几课我说的'归一化'，")
print("  代码里用的其实是 StandardScaler（标准化）—— 名字错了，功能没差，现在把它纠正过来。")

X_raw = df[["年龄", "月收入"]].values          # 进阶：.values 把 DataFrame 转成纯 numpy 数组
X_std = StandardScaler().fit_transform(X_raw)
X_mm = MinMaxScaler().fit_transform(X_raw)

print(f"\n原始数据        ：年龄 {X_raw[:,0].min():.0f}~{X_raw[:,0].max():.0f}，"
      f"收入 {X_raw[:,1].min():,.0f}~{X_raw[:,1].max():,.0f}  ← 差 4 个数量级")
print(f"标准化后        ：年龄 {X_std[:,0].min():.2f}~{X_std[:,0].max():.2f}，"
      f"收入 {X_std[:,1].min():.2f}~{X_std[:,1].max():.2f}  ← 两者在同一个量级了")
print(f"归一化后        ：年龄 {X_mm[:,0].min():.2f}~{X_mm[:,0].max():.2f}，"
      f"收入 {X_mm[:,1].min():.2f}~{X_mm[:,1].max():.2f}  ← 都被压进 0~1")
print(f"  看出问题了吗：归一化后，除了 3 个大客户是 1.0，")
print(f"  其余 {(X_mm[:,1] < 0.1).sum()} 个人全被挤在 0.1 以下的角落里，几乎分不出差别。")
print(f"  这就是有极端值时，MinMax 不如 Standard 耐用的原因。")

# 图2：原始 / 标准化 / 归一化 三连对比
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4.6))
titles = ["原始数据\n(收入轴被拉得极长)", "标准化 StandardScaler\n(变成均值0、标准差1)", "归一化 MinMaxScaler\n(挤进 0~1 的小角落)"]
datas = [X_raw, X_std, X_mm]
for ax, data, title in zip(axes2, datas, titles):
    ax.scatter(data[:, 0], data[:, 1], s=18, color=COLOR_MAIN, alpha=0.6)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("年龄（处理后）")
    ax.set_ylabel("月收入（处理后）")
fig2.suptitle("图2  同一份数据，三种量纲状态", fontsize=14)
fig2.tight_layout()
plt.show()

print("\n【补充一句，能帮你少做无用功】")
print("  决策树、随机森林【不需要】量纲处理。")
print("  因为它只看'这个值大于某个阈值吗'，比较的是大小顺序，不计算距离，")
print("  你把收入除以 10000，它的切分点跟着变，结论完全一样。")


# ============================================
# ④ 文字变数字：编号编码的坑，和独热编码的解法
# ============================================
print("\n" + "=" * 68)
print("④ 类别特征编码：城市是文字，模型看不懂，怎么办？")
print("=" * 68)
print("【做法一（有坑）】编号编码 LabelEncoder：给每个类别编个号")
print("  顾名思义，就是把文字直接换成 0、1、2 这样的编号。")
print("  看起来问题解决了，其实埋了颗雷：模型会以为 2 > 1 > 0，")
print("  于是它认定'广州比北京大'，凭空造出一个【根本不存在的大小顺序】。")
print("  现实中城市之间哪有什么大小？这就是假信息。")
print()
print("【做法二（正解）】独热编码 OneHotEncoder：一个类别拆成一列 0/1")
print("      上海 → [1, 0, 0]，北京 → [0, 1, 0]，广州 → [0, 0, 1]")
print("  三列之间没有大小关系，谁也不比谁大，模型就不会误读。")
print()

# 【重点】光说没用，跑个实验看差距。特征只留"城市"一列，把编码方式的影响单独拎出来看
X_city = df[["城市"]].copy()      # 进阶：.copy() 复制一份，避免改动影响原表
y_renew = df["是否续费"].values

# 先看清 LabelEncoder 到底编成了几号 —— 它按字符编码自动排序，不由你决定
le = LabelEncoder()
codes = le.fit_transform(X_city["城市"])
print("LabelEncoder 实际编出来的号，以及每个城市的真实续费率：")
for name, code in zip(le.classes_, le.transform(le.classes_)):     # 进阶：classes_ 是它排好序的类别清单
    print(f"    {name} → 编号 {code} 号，真实续费率 {y_renew[codes == code].mean():.0%}")
chain = " → ".join(f"{c}号:{y_renew[codes == c].mean():.0%}" for c in sorted(set(codes)))
print(f"  按编号顺序连起来看：{chain}")
print("  这是个『高 → 低 → 高』的 V 字形！而编号编码只能表达单调关系（一路升或一路降），")
print("  根本画不出 V 字 —— 模型被这个假顺序捆住了手脚。")
print()

# 做法一：编号编码 + 逻辑回归
X_label = codes.reshape(-1, 1)   # 进阶：一维要 reshape 成"列"，模型只吃二维表格
acc_label = cross_val_score(LogisticRegression(max_iter=1000), X_label, y_renew, cv=5).mean()

# 做法二：独热编码 + 逻辑回归
X_onehot = OneHotEncoder(sparse_output=False).fit_transform(X_city)     # 进阶：sparse_output=False 表示返回普通数组而不是稀疏矩阵，方便查看
acc_onehot = cross_val_score(LogisticRegression(max_iter=1000), X_onehot, y_renew, cv=5).mean()

# 参照物：不管三七二十一全猜"续费"，准确率是多少？
acc_dummy = max(y_renew.mean(), 1 - y_renew.mean())

print(f"编号编码 LabelEncoder  的 5 折平均准确率：{acc_label:.2%}")
print(f"独热编码 OneHotEncoder 的 5 折平均准确率：{acc_onehot:.2%}")
print(f"摆烂基线（全猜'续费'）      的准确率：{acc_dummy:.2%}")
print()
print("结论一目了然：")
print("  · 编号编码下，模型只能沿着 0→1→2 单调地判断，")
print(f"    分数最高也就 {acc_label:.1%}，跟'全猜续费'的 {acc_dummy:.1%} 基本一个水平，")
print("    等于这一列特征白给了。")
print("  · 独热编码拆出的三个 0/1 列，让每个城市能独立拥有自己的结论，")
print(f"    高高低低随便摆，分数立刻涨到 {acc_onehot:.1%}。")
print()
print("【注意别走极端】会员等级 = 普通 < 白银 < 黄金，这是【真的】有顺序，")
print("  这种情况用编号编码反而更合适（还不占额外列）。")
print("  判断口诀：类别之间有大小/高低含义吗？有 → 编号；没有 → 独热。")

# 图3：两种编码方式对比
fig3, ax3 = plt.subplots(figsize=(7.5, 5))
bars = ax3.bar(["编号编码\nLabelEncoder", "独热编码\nOneHotEncoder", "摆烂基线\n全猜续费"],
               [acc_label, acc_onehot, acc_dummy],
               color=[COLOR_WARN, COLOR_OK, "#9ca3af"], alpha=0.85, width=0.55)
for bar, v in zip(bars, [acc_label, acc_onehot, acc_dummy]):
    ax3.text(bar.get_x() + bar.get_width() / 2, v + 0.012, f"{v:.2%}",
             ha="center", fontsize=12, fontweight="bold")
ax3.set_ylim(0, 1.05)
ax3.set_ylabel("5 折交叉验证平均准确率")
ax3.set_title("图3  同样是'城市'，编码方式不同，效果天差地别", fontsize=13)
ax3.grid(axis="y", alpha=0.3)
fig3.tight_layout()
plt.show()


# ============================================
# ⑤ 数据泄露：一个让分数"虚高"的隐形陷阱
# ============================================
print("\n" + "=" * 68)
print("⑤ 数据泄露：预处理顺序错了，你的分数就是假的")
print("=" * 68)
print("【名词】数据泄露（Data Leakage）：")
print("  在训练模型时，不小心让模型'偷看'到了本不该看到的信息。")
print("  最典型的偷看对象就是【测试集的标签】。")
print()
print("  前面第②③④节，我都是拿【全量数据】去 fit 中位数、均值、编码表的。")
print("  在'先切分、后处理'的正确流程下，这在正式建模时是有问题的：")
print("  交叉验证里的'验证折'也被算进了均值和编码表 —— 相当于考前偷看了答案。")
print()
print("  平时它被抓住的幅度很小（均值泄露一点点），但用【特征选择】演示会特别明显。")
print()

# 【重点】制造一场"纯噪声考试"：30 个特征全是随机数，跟标签毫无关系
np.random.seed(0)
X_noise = np.random.randn(120, 30)                  # 120 个样本，30 个特征，全是随机噪声
y_noise = np.random.randint(0, 2, 120)              # 标签也是随机的 —— 理论上不可能学会
print("说明：30 个特征全是随机数，标签也是随机掷的，两者毫无关系。")
print("      所以真实水平应该是 50% 左右（瞎猜），任何显著高于 50% 都是假象。")

# 做法一（错误）：先在全量数据上挑出"和标签最相关的 3 个特征"，再做交叉验证
selector = SelectKBest(score_func=f_classif, k=3)   # 进阶：f_classif 用方差分析给特征和标签的相关性打分
X_selected = selector.fit_transform(X_noise, y_noise)
acc_leak = cross_val_score(LogisticRegression(max_iter=1000), X_selected, y_noise, cv=5).mean()

# 做法二（正确）：把"挑特征"也塞进流水线，让它每折都在训练集内部重挑
pipe = Pipeline([
    ("select", SelectKBest(score_func=f_classif, k=3)),
    ("model", LogisticRegression(max_iter=1000)),
])
acc_clean = cross_val_score(pipe, X_noise, y_noise, cv=5).mean()

print(f"\n做法一：全量挑特征 → 再交叉验证 ：{acc_leak:.2%}   ← 假象！")
print(f"做法二：把挑特征装进 Pipeline   ：{acc_clean:.2%}   ← 真相（约等于瞎猜）")
print()
print("差这么多，是因为做法一在挑特征时，已经用到了全部标签（包括验证折的标签）。")
print("它挑出来的'最相关特征'，纯属运气好、恰好蒙对了验证折的答案。")
print("一旦真的拿新数据来（做法二），这个'好特征'立刻现出原形。")
print()
print("同理，标准化、填缺失、独热编码，只要用了全量数据 fit，全都构成泄露。")

# 图4：泄露 vs 正确做法
fig4, ax4 = plt.subplots(figsize=(7.5, 5))
bars4 = ax4.bar(["错误做法\n全量挑特征后验证", "正确做法\n全塞进 Pipeline", "理论真实水平\n瞎猜"],
                [acc_leak, acc_clean, 0.5], color=[COLOR_WARN, COLOR_OK, "#9ca3af"], alpha=0.85, width=0.55)
for bar, v in zip(bars4, [acc_leak, acc_clean, 0.5]):
    ax4.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.2%}",
             ha="center", fontsize=12, fontweight="bold")
ax4.axhline(0.5, color="#6b7280", linestyle=":", linewidth=1.6)
ax4.set_ylim(0, 1.05)
ax4.set_ylabel("5 折交叉验证平均准确率")
ax4.set_title("图4  纯噪声数据上的「成绩单」：左边的高分是偷来的", fontsize=13)
ax4.grid(axis="y", alpha=0.3)
fig4.tight_layout()
plt.show()


# ============================================
# ⑥ 收官：用 Pipeline 把四步焊接成一条流水线
# ============================================
print("\n" + "=" * 68)
print("⑥ 收官：Pipeline + ColumnTransformer 手把手处理真正的脏数据")
print("=" * 68)
print("【名词】Pipeline（流水线）：把'填缺失 → 量纲处理 → 编码 → 训练模型'")
print("  按顺序串成一个整体对象。调用一次 fit，它自己按顺序走完全部步骤。")
print("  好处：交叉验证时，所有预处理都只在【训练折】里 fit，测试折只被 transform，")
print("        泄露问题从源头上被堵死了。")
print()
print("【名词】ColumnTransformer（列级别分流器）：不同列走不同流程。")
print("  数值列（年龄、月收入）→ 中位数填 + 标准化")
print("  类别列（城市、会员等级）→ 众数填 + 独热编码")
print()

num_cols = ["年龄", "月收入"]
cat_cols = ["城市", "会员等级"]

# 数值列的处理流程：先填，再标准化
num_pipe = Pipeline([
    ("fill", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])
# 类别列的处理流程：先填空，再独热
cat_pipe = Pipeline([
    ("fill", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),   # 进阶：handle_unknown="ignore" 让将来冒出的新城市不会让程序崩，而是编成全 0
])

# 分流器：把上面两条流程分别挂到对应的列上
preprocess = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols),
])

# 完整流水线 = 预处理 + 模型
full_pipe = Pipeline([
    ("prep", preprocess),
    ("model", KNeighborsClassifier(n_neighbors=5)),
])

X_df = df[num_cols + cat_cols]
y = df["是否续费"].values

# 【重点】现在整条流水线可以像单个模型一样被交叉验证 —— 每一折内部独立完成预处理
scores = cross_val_score(full_pipe, X_df, y, cv=5)
print(f"整条流水线的 5 折分数：{np.round(scores, 4)}")
print(f"平均：{scores.mean():.2%}\n")

# 正式训练 + 测试集验收
X_tr, X_te, y_tr, y_te = train_test_split(X_df, y, test_size=0.25, random_state=42)
full_pipe.fit(X_tr, y_tr)
print(f"锁起来的测试集验收准确率：{full_pipe.score(X_te, y_te):.2%}")

# 【彩蛋】看看"经过预处理后"，模型眼里到底长什么样
feat_names = full_pipe.named_steps["prep"].get_feature_names_out()   # 进阶：拿到每列处理后的名字
print(f"\n原始 4 个特征，经过处理后变成 {len(feat_names)} 个特征：")
for name in feat_names:
    print(f"  · {name}")
print("  看到了吗：'城市'和'会员等级'被拆成了一堆 0/1 列，")
print("  而'年龄''月收入'被压到了同一个量级。这些才是模型真正看到的东西。")

print("\n" + "=" * 68)
print("本课总结（这四步以后会出现在你每一次建模里）")
print("=" * 68)
print("  1. 缺失值 → SimpleImputer；收入类偏态数据用 median 比 mean 稳")
print("  2. 量纲   → 标准化 StandardScaler / 归一化 MinMaxScaler")
print("              距离类模型（KNN、SVM）必须做；树模型不用做")
print("  3. 文字   → 无序类别用 OneHotEncoder，有序类别才用 LabelEncoder")
print("  4. 以上三步统统塞进 Pipeline，交给交叉验证自动、正确地执行")
print("              否则你看到的分数，可能是偷来的（数据泄露）")
