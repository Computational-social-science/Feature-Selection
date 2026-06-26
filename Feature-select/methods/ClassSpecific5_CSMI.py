from skfeature.utility.entropy_estimators import *

# from main_X import *

def csmi(X, y,ys, **kwargs):
    ys=0
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
    t4 = np.zeros(n_features)
    t5 = np.zeros(n_features)
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
        # beta = 1.0 / len(F)
        # gamma = 1.0 / len(F)
        for i in range(n_features):
            if i not in F:
                f = X[:, i]
                # con = cmiddspecial(f,f_select, y, ys) + cmiddspecial(f_select,f, y, ys)
                # con1,red =ConAndRed(f_select,y,f,ys)
                # red = redundancy(f,f_select, y, ys)
                t2[i] += cmiddspecial(f,f_select, y, ys)+cmiddspecial(f_select,f, y, ys)
                t3[i] += cmidd(y,f, f_select)+midd(f, f_select)
                # calculate j_cmi for feature i (not in F)
                t = t2[i] - t3[i]
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
def cmiddspecial(x, y, z, c):#I(y,x|z)
    # print('!!',entropyd(list(zip(y, z)),c),entropyd(list(zip(x, z)),c),entropyd(list(zip(x, z, y)),c),entropyd(y,c))
    return entropydpro(list(zip(x, y, z)),c)+entropyed(list(zip(x, z)),c)-entropyed(list(zip(x, z, y)),c)-entropydpro(list(zip(x, z)),c)
def redundancy(x, y, z, c):#Iz(y,x)
    return -entropydpro(list(zip(x, y, z)),c)+entropydpro(list(zip(y, z)),c)+entropydpro(list(zip(x, z)),c)
# Discrete estimators
def entropyed(sx,sy,base=2):
    return entropyfromprobs(histspecial(sx,sy), base=base)
def entropydpro(sx,sy,base=2):
    sumnum = 0
    d = dict()
    ds = dict()
    for s in sx:
        s = s[:-1]
        d[s] = d.get(s, 0) + 1
    for s in sx:
        # 让最后一个是需要指定的，就可以无限循环了
        if s[-1] == sy:
            ds[s] = ds.get(s, 0) + 1
    for p in ds:
        sumnum += ds.get(p, 0)/len(sx)*log(d.get(p[:-1], 0)/len(sx))
    return -sumnum/log(base)
def histspecial(sx,sy):
    # Histogram from list of samples
    d = dict()
    for s in sx:
        # print(s)
        if isinstance(s, tuple):
            nowy = s[1]
        else:
            nowy = s
        # print(".....",nowy)
        if (nowy == sy):
            d[s] = d.get(s, 0) + 1
    return map(lambda z: float(z)/len(sx), d.values())

