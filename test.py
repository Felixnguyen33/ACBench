import matplotlib.pyplot as plt

# NF3值列表
nf3_values = [
    -1.0,
    -0.5350227355957031,
    -0.2469314038753510,
    0.0,
    0.1833375245332718,
    0.3819939494132996,
    0.6229856610298157,
    1.0,
]

# 索引列表
indices = range(1, len(nf3_values) + 1)

# 绘制折线图
plt.figure(figsize=(8, 6))
plt.plot(indices, nf3_values, marker="o", linestyle="-", color="b")

# 添加标题和轴标签
plt.title("NF3")
plt.xlabel("INDEX")
plt.ylabel("VALUE")

# 添加网格线
plt.grid(True)

# 设置x轴刻度
plt.xticks(indices)

# 显示图形
plt.savefig("nf3_values.png")
