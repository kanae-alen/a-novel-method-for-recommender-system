import matplotlib.pyplot as plt
import numpy as np

# 假设这是A方法和B方法在四个指标上的数值数据
indicators = ['Accuracy', 'Average-diversity', 'Coverage', 'Average-novelty']
method_A_values = [0.877, 0.909, 0.039, 0.958]
method_B_values = [0.873, 0.966, 0.052, 0.966]

# 设置柱状图的宽度
bar_width = 0.30

# 生成 x 坐标轴的位置
r1 = np.arange(len(indicators))
r2 = [x + bar_width for x in r1]

# 创建柱状图
plt.bar(r1, method_A_values, color='#A1A9D0', width=bar_width, edgecolor='grey', label='Original Method')
plt.bar(r2, method_B_values, color='#F0988C', width=bar_width, edgecolor='grey', label='New Method')

# 添加标签，标题和图例
plt.xticks([r + bar_width/2 for r in range(len(indicators))], indicators)
plt.legend()

# 打开网格
plt.grid(True, alpha=0.3)

plt.ylim(0, 1.3)

plt.savefig('mf_0.25_0.75.pdf', format='pdf')

# 获取当前的 axes 对象
axes = plt.gca()

# 设置背景颜色
axes.set_facecolor('white')

# 显示图表
plt.show()

