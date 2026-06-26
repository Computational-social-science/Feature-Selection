import Orange  # version: Orange3==3.32.0
import pandas as pd
# import matplotlib
# matplotlib.use('TkAgg')  # 不显示图则加上这两行
import matplotlib.pyplot as plt
T= '1'
ranktable = pd.read_excel( '../tables/'+T+'/ranktable.xlsx',index_col=0).T
print(ranktable)
datasets_num = 23
for i in ranktable:
    CD = Orange.evaluation.scoring.compute_CD(ranktable[i], datasets_num, alpha='0.05', test='nemenyi')
    Orange.evaluation.scoring.graph_ranks(ranktable[i], ranktable.index, cd=CD,width=6, textspace=1.5,cdmethod=7, reverse=True)
    # plt.savefig("picture/" + i + T +".png", dpi=300, bbox_inches='tight')
    plt.show()