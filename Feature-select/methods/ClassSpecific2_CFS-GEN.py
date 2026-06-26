import copy
import random

import weka
from sklearn.preprocessing import LabelEncoder
from weka.core.classes import Random
from weka.core.converters import Loader
from weka.attribute_selection import ASSearch, ASEvaluation, AttributeSelection
from weka.core.classes import from_commandline, get_classname
from weka.attribute_selection import ASSearch
from weka.attribute_selection import ASEvaluation

# search = from_commandline('weka.attributeSelection.GreedyStepwise -B -T -1.7976931348623157E308 -N -1 -num-slots 1', classname=get_classname(ASSearch))
# evaluation = from_commandline('weka.attributeSelection.CfsSubsetEval -P 1 -E 1', classname=get_classname(ASEvaluation))

# # 加载数据集
# loader = Loader(classname="weka.core.converters.ArffLoader")
# data = loader.load_file("path/to/your/data.arff")
# data.class_is_last()
#
# # 创建特征选择器
# selector = AttributeSelection()
#
# # 创建评估准则
# evaluator = CfsSubsetEval()
# search = BestFirst()
#
# # 设置评估准则和搜索策略
# selector.evaluator = evaluator
# selector.search = search
#
# # 应用特征选择器到数据集上
# selected_attributes = selector.select_attributes(data)
#
# # 输出所选择的特征索引
# print("Selected attribute indices:", selected_attributes.selected_attributes)
# from weka.attribute_selection import GeneticSearch
#
# # 创建搜索策略
# search = GeneticSearch()
#
# # 设置评估准则和搜索策略
# selector.evaluator = evaluator
# selector.search = search
#
# # 应用特征选择器到数据集上
# selected_attributes = selector.select_attributes(data)
#
# # 输出所选择的特征索引
# print("Selected attribute indices:", selected_attributes.selected_attributes)
# from weka.attribute_selection import ConsistencySubsetEval
#
# # 创建评估准则
# evaluator = ConsistencySubsetEval()
#
# # 设置评估准则和搜索策略
# selector.evaluator = evaluator
# selector.search = search
#
# # 应用特征选择器到数据集上
# selected_attributes = selector.select_attributes(data)
#
# # 输出所选择的特征索引
# print("Selected attribute indices:", selected_attributes.selected_attributes)
# # 设置评估准则和搜索策略
# selector.evaluator = evaluator
# selector.search = search
#
# # 应用特征选择器到数据集上
# selected_attributes = selector.select_attributes(data)
#
# # 输出所选择的特征索引
# print("Selected attribute indices:", selected_attributes.selected_attributes)
#

import pandas as pd
from weka.core.converters import Loader,Saver
from weka.core.dataset import Instances
import weka.core.jvm as jvm
from weka.filters import AttributeSelection,ASEvaluation,ASSearch
import scipy.io as scio
import numpy as np
# 读取mat文件
datanames = ['Movement_libras','Musk1','Sorlie','Yale','WarpAR10P','GLIOMA','Arcene','Waveform','Wdbc','Synthetic_control','Authorship','Factors','Pixel','Isolet','ORL','WarpPIE10P','Su','Prostate_GE','TOX_171','Orlraws10P','CLL_SUB_111','Yeoh']

# datanames = ['Dermatology',]#'Waveform','Factors','Synthetic_control','Authorship','Pixel','Wdbc','Movement_libras','Musk1','Sorlie','Yale','WarpAR10P','GLIOMA','Arcene','Isolet','ORL','WarpPIE10P','Su','Prostate_GE','TOX_171','Orlraws10P','CLL_SUB_111',
base_url = r"./dataset/"
method = 'CFS-GEN'
# 启动jvm
jvm.start(packages=True)
for dataname in datanames:
    # print(dataname)
    # data_url = dataname + '.mat'
    # url = base_url + data_url #数据路径
    # data0 = scio.loadmat(url)# 读取数据文件
    # X0 = pd.DataFrame(data0['X'])#读取训练数据
    # y0 = pd.DataFrame(data0['Y'])#读取标签
    # label_encoder = LabelEncoder()
    # y_encoded = label_encoder.fit_transform(X0[33])
    # data = {33:y_encoded}
    # newf = pd.DataFrame(data)
    # X0[33]=newf[33]
    # def changetosinge(x):
    #     return float(x)
    # # 应用函数到 DataFrame 的每个元素
    # X0 = X0.applymap(changetosinge)
    # y0 = y0.applymap(changetosinge)
    # X = pd.DataFrame()
    # for col in X0.columns:
    #     X[col] = pd.cut(X0[col], bins=5, labels=False)
    # # 使用rename函数重新命名列名，将x列名控制在0-n
    # new_columns = list(range(X.shape[1]+1))
    # X = X.rename(columns=dict(zip(X.columns, new_columns)))
    # newdata = X
    # newdata['class'] = y0
    # # print(newdata)
    # #==========================================================读取文件
    # # print(X0,y0)
    # newfile = base_url+ dataname + '.arff'
    # # 将数据保存为ARFF文件
    # with open(newfile, 'w') as f:
    #     # 写入ARFF文件头部信息
    #     f.write('@relation MyDataset\n\n')
    #
    #     # 写入属性信息
    #     for column in newdata.columns:
    #         f.write('@attribute ' + str(column) + ' numeric\n')
    #     f.write('\n@data\n')
    #
    #     # 写入数据
    #     for index, row in newdata.iterrows():
    #         f.write(','.join(map(str, row.values)) + '\n')
    # #==============================================================将mat转化为arff
    for i in range(0,10):
        filename = base_url +'arff/' + dataname + str(i) + '.arff'
        print(filename)
        #加载文件
        loader = Loader(classname="weka.core.converters.ArffLoader")

        data0 = loader.load_file(filename)
        data0.class_is_last()
        last_column_values = []
        for k in range(data0.num_instances):
            last_column_values.append(data0.get_instance(k).get_value(data0.class_index))
        # 去重并打印所有取值
        unique_values = set(last_column_values)
        # print(unique_values)
        # print(data0)
        allresult = []
        for u in unique_values:
            # print(u)
            othercount = 0
            data = copy.deepcopy(data0)
            for j in range(data.num_instances):
                if data.get_instance(j).get_value(data.class_index) != u:
                    data.get_instance(j).set_value(data.class_index, 0.0)
                    othercount += 1
                else:
                    data.get_instance(j).set_value(data.class_index, 1.0)
            count = data.num_instances - othercount
            # Oversample by repeating instances in each class
            if count < othercount:
                # Calculate the number of instances to repeat
                repeat_count = othercount - count
                # Find instances in class u
                instances_to_repeat = [inst for inst in data if inst.get_value(data0.class_index) == 1.0]
                # Repeat instances randomly until the class is balanced
                for _ in range(repeat_count):
                    instance_to_add = random.choice(instances_to_repeat)
                    data.add_instance(instance_to_add)
        if data.class_attribute.is_numeric:
            # 转换数值类标签为分类类型标签
            from weka.filters import Filter
            num_to_nom = Filter(classname="weka.filters.unsupervised.attribute.NumericToNominal", options=["-R", str(data.class_index + 1)])
            num_to_nom.inputformat(data)
            data = num_to_nom.filter(data)

        #使用filter进行特征选取
        import weka.core.packages as packages
        packages.install_package("attributeSelectionSearchMethods")
        # # 获取已安装的包列表
        # installed_packages = packages.installed_packages()
        # print(installed_packages)
        #=================================搜索器和评估器的选择
        search=ASSearch(classname='weka.attributeSelection.GeneticSearch')
        # search=ASSearch(classname='weka.attributeSelection.BestFirst',options=["-N", "50"])
        evaluation = ASEvaluation(classname="weka.attributeSelection.CfsSubsetEval")
        # evaluation = ASEvaluation(classname="weka.attributeSelection.ConsistencySubsetEval")
        attsel=AttributeSelection()
        attsel.evaluator=evaluation
        attsel.search=search
        #===========================================================训练数据提取特征
        # options = ["-N", "10"]  # Set the number of results to 10
        attsel.inputformat(data)
        select=attsel.filter(data)
        # print(select.index())
        # selected_attributes = [i for i in select if i != data.class_index]
        # print(attsel.capabilities())
        # print(train_data_1)
        # data_1=attsel.filter(data)
        # selected_attribute_names = [data.attribute(i).name for i in select]
        # print("Selected attribute names:", selected_attributes)
        # print(select.attribute_names())
        file_name = method+'_' + dataname + str(i) +'.txt'
        result = pd.DataFrame(select.attribute_names())
        np.savetxt(file_name, result.values, fmt='%s')  # 使用savetxt()函数保存数组到文件

        # print(len(data_1))
        # print(train_data_1)
        # print(data)

jvm.stop()

#
# import pandas as pd
# from scipy import stats
# import arff
#
# # 1. 读取 ARFF 文件
# def read_arff(file_path):
#     with open(file_path, 'r') as f:
#         data = arff.load(f)
#     return data
#
# # 2. 转换数值类别属性为标称类型
# def convert_numeric_to_nominal(data):
#     # Assuming `data` is a dictionary containing ARFF data
#     attributes = data['attributes']
#     instances = data['data']
#
#     # Iterate through attributes to identify numeric class attribute
#     for attr in attributes:
#         if attr[1] == 'numeric':  # Check if attribute is numeric
#             # Perform discretization or other transformation here
#             # For example, you can discretize using equal-frequency binning
#             instances = discretize_numeric_attribute(instances, attr[0])
#
#     # Update data with transformed instances
#     data['data'] = instances
#     return data
#
# def discretize_numeric_attribute(instances, attr_name):
#     # Assuming `instances` is a list of instances
#     # Assuming `attr_name` is the name of the numeric attribute to be discretized
#     # Here you can implement your discretization method
#     # For example, using equal-frequency binning
#     values = [instance[attr_name] for instance in instances]
#     bins = pd.qcut(values, q=5, labels=False, duplicates='drop')
#     for i, instance in enumerate(instances):
#         instance[attr_name] = bins[i]  # Replace numeric value with bin number
#     return instances
#
# # 3. 保存处理后的 ARFF 文件
# def save_arff(data, file_path):
#     with open(file_path, 'w') as f:
#         arff.dump(data, f)
#
# # Main function
# def main():
#     # Read ARFF file
#     arff_data = read_arff('your_arff_file.arff')
#
#     # Convert numeric class attribute to nominal
#     arff_data = convert_numeric_to_nominal(arff_data)
#
#     # Save processed ARFF file
#     save_arff(arff_data, 'processed_arff_file.arff')
#
# if __name__ == "__main__":
#     main()
