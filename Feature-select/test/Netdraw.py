import json
import networkx as nx
import matplotlib.pyplot as plt
# 文件路径
# file_path = "data_dict_alarm.txt"
# #
# # # 读取 JSON 文件
# with open(file_path, "r") as file:
#     result_dict = json.load(file)
result_dict = {'HISTORY': {'LVFAILURE'}, 'CVP': {'LVEDVOLUME', 'BP'}, 'PCWP': {'LVEDVOLUME'}, 'HYPOVOLEMIA': {'LVEDVOLUME', 'BP', 'CO'}, 'LVEDVOLUME': {'HYPOVOLEMIA', 'CO', 'PCWP', 'CVP'}, 'LVFAILURE': {'HISTORY', 'CO', 'BP'}, 'STROKEVOLUME': {'CO', 'BP'}, 'ERRLOWOUTPUT': {'HRBP'}, 'HRBP': {'ERRLOWOUTPUT', 'HR'}, 'HREKG': {'CO', 'HRSAT', 'ERRCAUTER', 'HR'}, 'ERRCAUTER': {'HREKG'}, 'HRSAT': {'CO', 'HREKG', 'HR'}, 'INSUFFANESTH': {'EXPCO2'}, 'ANAPHYLAXIS': {'TPR'}, 'TPR': {'HR', 'ANAPHYLAXIS', 'CATECHOL', 'CO', 'BP'}, 'EXPCO2': {'VENTLUNG'}, 'KINKEDTUBE': {'PRESS'}, 'MINVOL': {'VENTTUBE', 'SAO2', 'CO', 'VENTALV', 'VENTLUNG', 'INTUBATION', 'SHUNT'}, 'FIO2': {'PVSAT'}, 'PVSAT': {'SAO2', 'BP'}, 'SAO2': {'PVSAT', 'MINVOL'}, 'PAP': {'PULMEMBOLUS'}, 'PULMEMBOLUS': {'SHUNT', 'PAP'}, 'SHUNT': {'INTUBATION', 'MINVOL', 'PULMEMBOLUS'}, 'INTUBATION': {'MINVOL', 'SHUNT', 'VENTALV'}, 'PRESS': {'CO', 'KINKEDTUBE', 'VENTTUBE', 'VENTALV'}, 'DISCONNECT': {'VENTTUBE'}, 'MINVOLSET': {'VENTMACH'}, 'VENTMACH': {'VENTTUBE', 'MINVOLSET'}, 'VENTTUBE': {'PRESS', 'VENTMACH', 'MINVOL', 'DISCONNECT', 'VENTALV'}, 'VENTLUNG': {'MINVOL', 'EXPCO2', 'VENTALV'}, 'VENTALV': {'PRESS', 'VENTTUBE', 'MINVOL', 'CO', 'ARTCO2', 'VENTLUNG', 'INTUBATION'}, 'ARTCO2': {'VENTALV'}, 'CATECHOL': {'TPR', 'BP', 'ARTCO2', 'HR'}, 'HR': {'HREKG', 'HRBP', 'HRSAT', 'CATECHOL', 'TPR', 'CO', 'VENTLUNG'}, 'CO': {'HREKG', 'HYPOVOLEMIA', 'HR', 'HRSAT', 'LVFAILURE', 'LVEDVOLUME', 'BP', 'STROKEVOLUME'}, 'BP': {'HYPOVOLEMIA', 'CATECHOL', 'TPR', 'LVFAILURE', 'CO', 'STROKEVOLUME'}}
# result_dict={'asia': [], 'tub': ['smoke',  'lung', 'either', 'xray', 'dysp'], 'smoke': [ 'tub', 'lung', 'bronc', 'either', 'xray', 'dysp'], 'lung': ['tub', 'bronc',  'smoke', 'either', 'xray', 'dysp'], 'bronc': [ 'lung', 'either', 'smoke', 'either', 'xray', 'dysp'], 'either': ['dysp', 'bronc', 'tub', 'smoke', 'lung', 'bronc', 'xray', 'dysp'], 'xray': ['dysp', 'bronc', 'tub', 'smoke', 'lung', 'bronc', 'either', 'dysp'], 'dysp': ['either', 'xray', 'bronc', 'bronc', 'tub', 'smoke', 'lung', 'bronc', 'either', 'xray']}
# result_dict={'HISTORY': {'LVFAILURE'}, 'CVP': {'LVEDVOLUME', 'BP'}, 'PCWP': {'LVEDVOLUME'}, 'HYPOVOLEMIA': {'LVEDVOLUME', 'CO', 'BP'}, 'LVEDVOLUME': {'CVP', 'PCWP', 'CO', 'HYPOVOLEMIA'}, 'LVFAILURE': {'HISTORY', 'CO', 'BP'}, 'STROKEVOLUME': {'CO', 'BP'}, 'ERRLOWOUTPUT': {'HRBP'}, 'HRBP': {'ERRLOWOUTPUT', 'HR'}, 'HREKG': {'HR', 'CO', 'HRSAT', 'ERRCAUTER'}, 'ERRCAUTER': {'HREKG'}, 'HRSAT': {'HR', 'CO', 'HREKG'}, 'INSUFFANESTH': {'EXPCO2'}, 'ANAPHYLAXIS': {'TPR'}, 'TPR': {'BP', 'CATECHOL', 'ANAPHYLAXIS', 'HR', 'CO'}, 'EXPCO2': {'VENTLUNG'}, 'KINKEDTUBE': {'PRESS'}, 'MINVOL': {'VENTTUBE', 'SAO2', 'VENTLUNG', 'SHUNT', 'INTUBATION', 'VENTALV', 'CO'}, 'FIO2': {'PVSAT'}, 'PVSAT': {'SAO2', 'BP'}, 'SAO2': {'MINVOL', 'PVSAT'}, 'PAP': {'PULMEMBOLUS'}, 'PULMEMBOLUS': {'PAP', 'SHUNT'}, 'SHUNT': {'PULMEMBOLUS', 'MINVOL', 'INTUBATION'}, 'INTUBATION': {'SHUNT', 'MINVOL', 'VENTALV'}, 'PRESS': {'KINKEDTUBE', 'VENTTUBE', 'CO', 'VENTALV'}, 'DISCONNECT': {'VENTTUBE'}, 'MINVOLSET': {'VENTMACH'}, 'VENTMACH': {'VENTTUBE', 'MINVOLSET'}, 'VENTTUBE': {'PRESS', 'DISCONNECT', 'VENTMACH', 'MINVOL', 'VENTALV'}, 'VENTLUNG': {'EXPCO2', 'MINVOL', 'VENTALV'}, 'VENTALV': {'VENTTUBE', 'PRESS', 'VENTLUNG', 'CO', 'INTUBATION', 'MINVOL', 'ARTCO2'}, 'ARTCO2': {'VENTALV'}, 'CATECHOL': {'TPR', 'HR', 'ARTCO2', 'BP'}, 'HR': {'TPR', 'VENTLUNG', 'CO', 'CATECHOL', 'HREKG', 'HRSAT', 'HRBP'}, 'CO': {'BP', 'STROKEVOLUME', 'HREKG', 'HR', 'HRSAT', 'LVFAILURE', 'LVEDVOLUME', 'HYPOVOLEMIA'}, 'BP': {'TPR', 'CO', 'STROKEVOLUME', 'CATECHOL', 'LVFAILURE', 'HYPOVOLEMIA'}}

# 打印加载的字典
print("从文件加载的字典：")
print(result_dict)

# 使用字典中的数据
for key, value in result_dict.items():
    print(f"节点 {key} 的连接节点: {value}")
# 创建一个无向图
G = nx.Graph()

# 添加节点和边
for node, neighbors in result_dict.items():
    # 添加节点
    G.add_node(node)
    # 添加边
    for neighbor in neighbors:
        G.add_edge(node, neighbor)

# 绘制图形
plt.figure(figsize=(6, 6))  # 设置图形大小
pos = nx.spring_layout(G)  # 使用 spring 布局算法
nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=30, font_size=8, font_weight="bold")

# 显示图形
# plt.title("无向图表示节点及其联系", fontsize=20)
plt.show()



