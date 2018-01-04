import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

fig = plt.figure(facecolor='w')
# 注意位置坐标，数字表示的是坐标的比例
ax1 = fig.add_subplot(111)
s = ['after', 'using', 'it', 'for', 'just', 'a', 'month', ',', 'i', 'could', 'not', 'play', 'any', 'dvd']
aa = plt.imshow(np.random.randint(1, 10, (1, 14)), plt.get_cmap('gray'))

haha = np.random.randint(1, 10, (1, 14))
ax1.set_xticklabels(s)
plt.rc('xtick', labelsize = len(s))
plt.yticks([])
# aa = plt.imshow(np.random.randint(1,10,(5,5)),cmap=cm.autumn)  这2中写法作用一样
#添加一个colorbar   #颜色渐变条
#plt.colorbar(aa)
plt.show()