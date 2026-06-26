import warnings
from sklearn.model_selection import KFold
import pandas as pd
import scipy.io as scio
from itertools import combinations
def get_equivalence_classes(data, attributes):#计算等价类
    return data.groupby(list(attributes)).groups

data = pd.DataFrame({
    0: [0,0,0,1,0,1,1,1,1,],
    1: [1,1,1,0,0,0,0,0,1,],
    2: [3,3,3,1,1,2,1,1,2,],
    3: [0,0,2,0,0,2,0,0,2,],
    4: [0,0,2,1,1,0,2,2,1,],
    'label': [1,1,1,2,2,2,3,3,3]
})
X = data.drop(columns=['label'])
y = data['label']

def positive_region(data, attributes):
    pos_reg = set()
    for _, group in data.groupby(attributes):
        if len(group['label'].unique()) == 1:
            pos_reg.update(group.index)
    return pos_reg
def find_reducts(data, all_attributes, full_positive_region):
    reducts = []
    for i in range(1, len(all_attributes) + 1):
        print(i)
        for subset in combinations(all_attributes, i):
            print(subset,111112222233333)
            if full_positive_region.issubset(positive_region(data, list(subset))):
                reducts.append(subset)
    return reducts

#给X是特征集，y是label集，名为label不可随意更改
def CORE_POS(X,y,d):
    #X还是特征，y是标签
    C = X.columns.tolist()
    core = []
    ally = CPRASpecial(X,C,y,d)
    for c in C:
        newC = [element for element in C if element != c]
        if CPRASpecial(X,newC,y,d) < ally:
            core.append(c)
    return core

def CPRA(X,attributes,y):
    y.rename('label', inplace=True)
    data = pd.concat([X, y], axis=1)
    pos_reg = positive_region(data,attributes)
    total_samples = len(data)

    positive_region_cardinality = len(pos_reg)
    positive_region_ratio = positive_region_cardinality / total_samples
    return positive_region_ratio
#attributes是特证名，data是X和y的合并dataframe

def CPRASpecial(X,attributes,y,d):
    data = pd.concat([X, y], axis=1)
    pos_reg = positive_region(data,attributes)
    total_samples = len(data)

    subset_data = data[data['label'] == d]
    pos_reg_for_label = pos_reg.intersection(subset_data.index)
    positive_region_cardinality = len(pos_reg_for_label)
    positive_region_ratio = positive_region_cardinality / total_samples
    return positive_region_ratio
#attributes是特证名，data是X和y的合并dataframe

def get_samples_by_decision(data, decision_class):
    return set(data.index[data['label'] == decision_class])

# 忽略特定类型的警告
warnings.filterwarnings("ignore", message="A column-vector y was passed when a 1d array was expected.*")
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
from skfeature.utility.entropy_estimators import *
from sklearn.preprocessing import LabelEncoder

def SigMI(X,R,c,d):
    newR = list(R)
    t2 = CPRASpecial(X,R,y,d)
    newR.append(c)
    t1 = CPRASpecial(X,newR,y,d)
    return t1-t2
kfold = KFold(n_splits=10, shuffle=True, random_state=19)  # 十折交叉验证，固定种子
datanames = ['Factors','Pixel' ]#'Authorship', 'Factors', 'Pixel', 'ORL', 'WarpPIE10P', 'Su',
            # 'Prostate_GE', 'TOX_171', 'CLL_SUB_111',
for dataname in datanames:
    print("====================================================================================================")
    print(dataname)
    base_url = r"./dataset/"
    data_url = dataname + '.mat'
    url = base_url + data_url  # 数据路径
    data = scio.loadmat(url)  # 读取数据文件
    X0 = pd.DataFrame(data['X'])  # 读取训练数据
    y0 = pd.DataFrame(data['Y'])  # 读取标签
    def changetosinge(x):
        return float(x)
    # 应用函数到 DataFrame 的每个元素
    X0 = X0.applymap(changetosinge)
    y0 = y0.applymap(changetosinge)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y0)
    Class = set(y_encoded)
    y = pd.DataFrame(y_encoded)
    y = y.rename(columns={y.columns[-1]: 'label'})
    # 数据离散化
    X = pd.DataFrame()
    #===========================dertology=====
    # label_encoder = LabelEncoder()
    # y_encoded = label_encoder.fit_transform(X0[33])
    # data = {33:y_encoded}
    # newf = pd.DataFrame(data)
    # X0[33]=newf[33]
    #===========================dertology=====
    for col in X0.columns:
        X[col] = pd.cut(X0[col], bins=5, labels=False)
    # 使用rename函数重新命名列名，将x列名控制在0-n
    new_columns = list(range(X.shape[1] + 1))
    X = X.rename(columns=dict(zip(X.columns, new_columns)))
    for fold, (train_index, test_index) in enumerate(kfold.split(X, y)):  # fold赋值之后就是0-9
        if fold in [8,9]:
            print("当前处理到第", fold, "折\n")
            bacc, kacc, sacc, racc = [], [], [], []
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]  # 提取训练集条目
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]  # 提取测试集条目
            result = []  # 特征提取结果数组
            C = list(set(X.columns))
            label = y_train['label']
            D = list(set(label))

            result = []
            for i in D:
                R = []
                R = CORE_POS(X,label,i)
                MI = CPRASpecial(X,X.columns.tolist(),y,i)
                if  not R:
                    mi = np.zeros(len(C))
                    for q in C:
                        f = X[q]
                        mi[q] = midd(f, label)
                    idx = np.argmax(mi)
                    R.append(idx)
                while CPRASpecial(X,list(R),y,i) < MI:
                    C_R = [element for element in C if element not in R]
                    sig = np.zeros(len(C_R))
                    for j in range(len(C_R)):
                        sig[j] = SigMI(X,R,C_R[j],i)
                    idx = np.argmax(sig)
                    R.append(C_R[idx])
                newmi = CPRASpecial(X,R,y,i)
                oR = list(R)
                nR = list(R)
                for r in oR:
                    nowR = [element for element in nR if element != r]
                    if not nowR:
                        break;
                    if CPRASpecial(X,nowR,y,i) == newmi :
                        R.remove(r)
                print(i,":    ",R)
                result.append(R)
                file_name = r"./feature_select_result/POS-CSR/" + dataname + str(fold) + "_result.txt"
                np.savetxt(file_name, result, fmt='%s')  # 使用savetxt()函数保存数组到文件

