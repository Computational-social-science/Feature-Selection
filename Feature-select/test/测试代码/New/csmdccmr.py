import pandas as pd
from skfeature.utility.entropy_estimators import *

# from main_X import *

def csmdccmr(X, y,ys, **kwargs):
    n_samples, n_features = X.shape
    # index of selected features, initialized to be empty
    F = []
    # Objective function value for selected features
    J_CMI = []
    # Mutual information between feature and response
    MIfy = []
    # indicate whether the user specifies the number of features
    is_n_selected_features_specified = False
    # initialize the parameters
    if 'beta' in kwargs.keys():
        beta = kwargs['beta']
    if 'gamma' in kwargs.keys():
        gamma = kwargs['gamma']
    if 'n_selected_features' in kwargs.keys():
        n_selected_features = kwargs['n_selected_features']
        is_n_selected_features_specified = True

    # select the feature whose j_cmi is the largest
    # t1 stores I(f;y) for each feature f
    t1 = np.zeros(n_features)
    # t2 stores sum_j(Iy(fj;f)) for each feature f
    t2 = np.zeros(n_features)
    # t3 stores sum_j(I(fj;y|f)+I(f;y|fj)) for each feature f
    t3 = np.zeros(n_features)
    #对第i个特征，求其midd，类与特征之间的信息量
    for i in range(n_features):
        f = X[:, i]
        t1[i] = middspecial(f, y, ys)
        # t1[i] = mutualInfo(f, y)

    # make sure that j_cmi is positive at the very beginning
    j_cmi = 1

    while True:
        if len(F) == 0:
            # select the feature whose mutual information is the largest
            idx = np.argmax(t1)
            F.append(idx)
            J_CMI.append(t1[idx])
            MIfy.append(t1[idx])
            f_select = X[:, idx]

        if is_n_selected_features_specified:
            if len(F) == n_selected_features:
                break
        else:
            if j_cmi < 0:
                break

        # we assign an extreme small value to j_cmi to ensure it is smaller than all possible values of j_cmi
        j_cmi = -1E30
        # if 'function_name' in kwargs.keys():
            # if kwargs['function_name'] == 'MRMR':
            #     beta = 1.0 / len(F)
            # elif kwargs['function_name'] == 'JMI':
        beta = 1.0 / len(F)
        gamma = 1.0 / len(F)
        for i in range(n_features):
            if i not in F:
                f = X[:, i]
                # con = cmiddspecial(f,f_select, y, ys) + cmiddspecial(f_select,f, y, ys)
                # con1,red =ConAndRed(f_select,y,f,ys)
                # red = redundancy(f,f_select, y, ys)
                t2[i] += redundancy(f,f_select, y, ys)
                t3[i] += cmiddspecial(f,f_select, y, ys) + cmiddspecial(f_select,f, y, ys)
                # calculate j_cmi for feature i (not in F)
                t = t1[i] - beta*t2[i] + gamma*t3[i]
                # record the largest j_cmi and the corresponding feature index
                if t > j_cmi:
                    j_cmi = t
                    idx = i
        F.append(idx)
        J_CMI.append(j_cmi)
        MIfy.append(t1[idx])
        f_select = X[:, idx]
    # return np.array(F), np.array(J_CMI), np.array(MIfy)
    return np.array(F)
def middspecial(x, y, c):
    """
    Discrete mutual information estimator given a list of samples which can be any hashable object
    """
    return -entropyed(list(zip(x, y)),c)+entropydpro(list(zip(x, y)),c)+entropyed(y,c)
def cmiddspecial(x, y, z, c):#I(y;x|z)=-[H(x,y,z)+H(z)-H(x,z)-H(y,z)]  此处y为待选和x为已选特征，z为目标特征
    # print('!!',entropyd(list(zip(y, z)),c),entropyd(list(zip(x, z)),c),entropyd(list(zip(x, z, y)),c),entropyd(y,c))
    #H(x,y,z)+H(z)-H(x,z)-H(y,z)
    # print(1,x,2,y,3,z)
    if isinstance(y, pd.DataFrame):
        y = [tuple(row) for row in y.values]
        # print(123,y.dtype,y,x)
    # print(x,y,z,12321,list(zip(x, y, z)))
    # print(entropyed(list(zip(x, y, z)),c),entropydpro(list(zip(y, z)),c),entropydpro(list(zip(x, y, z)),c),entropyed(list(zip(y, z)),c))
    return -entropyed(list(zip(x, y, z)),c)-entropydpro(list(zip(y, z)),c)+entropydpro(list(zip(x, y, z)),c)+entropyed(list(zip(y, z)),c)
def redundancy(x, y, z, c):#Iz(y,x)
    return -entropydpro(list(zip(x, y, z)),c)+entropydpro(list(zip(y, z)),c)+entropydpro(list(zip(x, z)),c)
# Discrete estimators
def entropyed(sx,sy,base=2):#直接进行统计计算
    return entropyfromprobs(histspecial(sx,sy), base=base)#求f(x)logf(x)的函数
def entropydpro(sx,sy,base=2):#在本不应该计算y的情况下仍要规定y进行统计计算
    sumnum = 0
    d = dict()
    ds = dict()
    for s in sx:#先确定目标特征每一种取值的个数，（被指定的，输入的时候放在最后一个输入）
        s = s[:-1]
        d[s] = d.get(s, 0) + 1#计数
    for s in sx:#所有排列组合中找到指定目标确定的组合，计数不同组合的数量
        # 让最后一个是需要指定的，就可以无限循环了
        if s[-1] == sy:
            ds[s] = ds.get(s, 0) + 1
    for p in ds:#对所有，求p(sx)*log(sx[-1])
        sumnum += ds.get(p, 0)/len(sx)*log(d.get(p[:-1], 0)/len(sx))
    return -sumnum/log(base)
def histspecial(sx,sy):
    # Histogram from list of samples
    d = dict()
    for s in sx:
        # print(s)
        if isinstance(s, tuple):#最多只允许两个输入，若一个则当前输入，若两个则后面那个输入为限定输入
            nowy = s[-1]
        else:
            nowy = s
        # print(".....",nowy)
        if (nowy == sy):#找到符合条件的（在目标特征对应情况下计算概率）
            d[s] = d.get(s, 0) + 1
    return map(lambda z: float(z)/len(sx), d.values())#p(x)对应特征取值出现概率相加

