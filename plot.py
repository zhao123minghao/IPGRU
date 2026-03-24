import matplotlib.pyplot as plt
import numpy as np

'''
PBMC_G949_21K
GTEx_4tissues
MAGIC_mouse
PBMC_G949_10K
PBMC_G5561
'''

labels = ['PBMC_G949_21K', 'GTEx_4tissues', 'MAGIC_mouse', 'PBMC_G949_10K', 'PBMC_G5561']

# "mse_nz"
# DGRU = [0.007483,0.05437,0.003348,0.006922,0.001338]
# DeepImpute = [0.008298,0.1133,0.002992,0.008473,0.001487]
# LATE=[0.007559,0.09740,0.007696,0.006782,0.001385]
# DCA=[0.008842,0.07218,0.002355,0.009805,0.001366]
# AutoImpute=[0.008523,0.09236,0.002994,0.007534,0.001369]
# #magic第二行太大设置为0.12
# MAGIC=[0.01191,0.12,0.01434,0.01163,0.003866]
#
# x=np.arange(len(labels))
# width=0.1
#
# fig,ax=plt.subplots()
#
# rects1 = ax.bar(x + width/6, DGRU, width, label='DGRU')
# rects2 = ax.bar(x + (width/6)*6, DeepImpute, width, label='DeepImpute')
# rects3 = ax.bar(x + (width/6)*12, LATE, width, label='LATE')
# rects4 = ax.bar(x + (width/6)*18, DCA, width, label='DCA')
# rects5 = ax.bar(x + (width/6)*24, AutoImpute, width, label='AutoImpute')
# rects6 = ax.bar(x + (width/6)*30, MAGIC, width, label='MAGIC')
#
# plt.tick_params(labelsize=8)
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('mse_nz')
# ax.set_title('mse_nz by impute methods')
# ax.set_xticks(x+width*2)
# ax.set_xticklabels(labels)
# ax.legend()
#
# # ax.bar_label(rects1, padding=3)
# # ax.bar_label(rects2, padding=3)
# # ax.bar_label(rects3, padding=3)
# # ax.bar_label(rects4, padding=3)
# # ax.bar_label(rects5, padding=3)
# # ax.bar_label(rects6, padding=3)
# plt.savefig('./mse_nz.png',format='png')
# fig.tight_layout()
#
# plt.show()



"mse"
DGRU = [0.007482,0.05437,0.003348,0.006923,0.001342]
DeepImpute = [0.008298	,0.11328	,0.002992	,0.008473	,0.001487]
LATE=[0.088101	,0.1637	,0.007697,	0.09140,	0.09388]
DCA=[0.08114,	0.1006,	0.002355,	0.08552,	0.09256]
AutoImpute=[0.08246,	0.2450,	0.003004,	0.08666,	0.09388]
MAGIC=[0.05222,	0.12,	0.01433,	0.05543,	0.01304]

x=np.arange(len(labels))
width=0.1

fig,ax=plt.subplots()

rects1 = ax.bar(x + width/6, DGRU, width, label='DGRU')
rects2 = ax.bar(x + (width/6)*6, DeepImpute, width, label='DeepImpute')
rects3 = ax.bar(x + (width/6)*12, LATE, width, label='LATE')
rects4 = ax.bar(x + (width/6)*18, DCA, width, label='DCA')
rects5 = ax.bar(x + (width/6)*24, AutoImpute, width, label='AutoImpute')
rects6 = ax.bar(x + (width/6)*30, MAGIC, width, label='MAGIC')

plt.tick_params(labelsize=8)
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('mse')
ax.set_title('mse by impute methods')

ax.set_xticks(x+width*2)
ax.set_xticklabels(labels)
ax.legend()
plt.savefig('./mse.png',format='png')
fig.tight_layout()
plt.show()