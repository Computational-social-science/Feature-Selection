from csmdccmr import *


# midd(x,y)前后顺序随意，求I(x,y)
# cmidd(x,y,z)前后顺序指定，求I(y;x|z)
# I(x;y;c)  =  I(x,y;c)-I(x;c)-I(y;c)
#           =  I(x;c|y)+I(y;c)-I(x;c)-I(y;c)
#           =  I(x;c|y)-I(x;c)

def jmim(X, y, **kwargs):#J(k) = arg max(min( I(fk,fs;C))),where I(fk,fs;C) = I(fs;C) + I(fk;C|fs)
    n_samples, n_features = X.shape
    F = [] # Selected features, initialized to be empty
    J_CMI = [] # Objective function value for selected features
    MIfy = [] # Mutual information between feature and response
    # initialize the parameters
    is_n_selected_features_specified = False
    if 'n_selected_features' in kwargs.keys():
        n_selected_features = kwargs['n_selected_features']
        is_n_selected_features_specified = True

    # select the feature whose j_cmi is the largest
    t1 = np.zeros(n_features) # t1 stores I(fk;y) for each feature f
    t2 = np.zeros(n_features) # t2 stores I(fk;C|fs) for each feature f
    min = np.zeros(n_features) # min stores min_J_cmi for each feature f
    # Calculate all the features' mutual information with label
    for i in range(n_features):
        f = X[:, i]
        t1[i] = midd(f, y)

    j_cmi = 1 # make sure that j_cmi is positive at the very beginning
    while True:
        # select the feature whose mutual information is the largest
        if len(F) == 0:#先选择一个特征，第一个特征选择的条件是max I（C;fi）
            idx = np.argmax(t1)
            F.append(idx)
            J_CMI.append(t1[idx])
            MIfy.append(t1[idx])
            f_select = X[:, idx]

        # Continue selecting features until a specified number is reached
        if is_n_selected_features_specified:
            if len(F) == n_selected_features:
                break


        for i in range(n_features):
            j_cmi = 1E30
            if i not in F:
                f = X[:, i]
                t2[i] = cmidd(y, f_select, f)
                # calculate j_cmi for feature i (not in F)
                t = t2[i] + t1[i]
                # record the largest j_cmi and the corresponding feature index
                if t < j_cmi:
                    j_cmi = t
            else:
                j_cmi = -1E30
            min[i] = j_cmi
        idx = np.argmax(min)
        F.append(idx)
        J_CMI.append(min[idx])
        min[idx] = -1E30
        f_select = X[:, idx]
    return np.array(F)
def mri(X, y, **kwargs):
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
    #对第i个特征，求其midd，类与特征之间的信息量
    for i in range(n_features):
        f = X[:, i]
        t1[i] = midd(f, y)
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
        beta = 0
        gamma = 0
        for i in range(n_features):
            if i not in F:
                f = X[:, i]
                # 累加的部分在这里
                t2[i] += cmidd(y,f,f_select)+ cmidd(y,f_select,f)
                # calculate j_cmi for feature i (not in F)
                t = t1[i] + t2[i]
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
def cfr(X, y, **kwargs):
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
    # select the feature whose j_cmi is the largest
    # t1 stores I(f;y) for each feature f
    t0 = np.zeros(n_features)

    t1 = np.zeros(n_features)
    # t2 stores sum_j(I(fj;f)) for each feature f
    t2 = np.zeros(n_features)
    # t3 stores sum_j(I(fj;f|y)) for each feature f
    t3 = np.zeros(n_features)
    if 'beta' in kwargs.keys():
        beta = kwargs['beta']
    if 'gamma' in kwargs.keys():
        gamma = kwargs['gamma']
    if 'n_selected_features' in kwargs.keys():
        n_selected_features = kwargs['n_selected_features']
        is_n_selected_features_specified = True

    for i in range(n_features):
        f = X[:, i]
        t0[i] = midd(f, y)
    # make sure that j_cmi is positive at the very beginning
    j_cmi = 1
    while True:
        if len(F) == 0:
            # select the feature whose mutual information is the largest
            idx = np.argmax(t0)
            F.append(idx)
            J_CMI.append(t0[idx])
            MIfy.append(t0[idx])
            f_select = X[:, idx]

        if is_n_selected_features_specified:
            if len(F) == n_selected_features:
                break
        else:
            if j_cmi < 0:
                break
        # we assign an extreme small value to j_cmi to ensure it is smaller than all possible values of j_cmi
        j_cmi = -1E30
        for i in range(n_features):
            if i not in F:
                f = X[:, i]
                t1[i] += cmidd(y, f, f_select)
                t2[i] += midd(f_select, y)
                # t3[i] += cmidd(y, f_select, f)
                # calculate j_cmi for feature i (not in F)
                t = 2*t1[i] -t2[i]
                # record the largest j_cmi and the corresponding feature index
                if t > j_cmi:
                    j_cmi = t
                    idx = i
        F.append(idx)
        J_CMI.append(j_cmi)
        MIfy.append(t1[idx])
        f_select = X[:, idx]

    return np.array(F)
def dcsf(X, y, **kwargs):
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
    t0 = np.zeros(n_features)
    # t1 stores I(f;y) for each feature f
    t1 = np.zeros(n_features)
    # t2 stores sum_j(I(fj;f)) for each feature f
    t2 = np.zeros(n_features)
    # t3 stores sum_j(I(fj;f|y)) for each feature f
    t3 = np.zeros(n_features)
    for i in range(n_features):
        f = X[:, i]
        t0[i] = midd(f, y)
    # make sure that j_cmi is positive at the very beginning
    j_cmi = 1
    while True:
        if len(F) == 0:
            # select the feature whose mutual information is the largest
            idx = np.argmax(t0)
            F.append(idx)
            J_CMI.append(t0[idx])
            MIfy.append(t0[idx])
            f_select = X[:, idx]

        if is_n_selected_features_specified:
            if len(F) == n_selected_features:
                break
        else:
            if j_cmi < 0:
                break

        # we assign an extreme small value to j_cmi to ensure it is smaller than all possible values of j_cmi
        j_cmi = -1E30
        for i in range(n_features):
            if i not in F:
                f = X[:, i]
                t1[i] += cmidd(y, f_select, f)
                t2[i] += cmidd(y,f, f_select)
                t3[i] += midd(f_select, f)
                # calculate j_cmi for feature i (not in F)
                t = t1[i] + t2[i] - t3[i]
                # record the largest j_cmi and the corresponding feature index
                if t > j_cmi:
                    j_cmi = t
                    idx = i
        F.append(idx)
        J_CMI.append(j_cmi)
        MIfy.append(t1[idx])
        f_select = X[:, idx]

    return np.array(F)
def mrmd(X, y, **kwargs):
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
    # t2 stores sum_j(I(fj;f)) for each feature f
    t2 = np.zeros(n_features)
    for i in range(n_features):
        f = X[:, i]
        t1[i] = midd(f, y)

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
        for i in range(n_features):
            if i not in F:
                f = X[:, i]
                t2[i] += midd(f_select, f) - cmidd(y, f, f_select)
                # calculate j_cmi for feature i (not in F)
                t = t1[i] - (1.0 / len(F))*t2[i]
                # record the largest j_cmi and the corresponding feature index
                if t > j_cmi:
                    j_cmi = t
                    idx = i
        F.append(idx)
        J_CMI.append(j_cmi)
        MIfy.append(t1[idx])
        f_select = X[:, idx]

    return np.array(F)
def iwfs(X, y, **kwargs):
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
    su = np.zeros(n_features)
    # t2 stores sum_j(I(fj;f)) for each feature f
    r = np.zeros(n_features)
    # t3 stores sum_j(I(fj;f|y)) for each feature f
    iw= np.zeros(n_features)

    fw = np.ones(n_features)
    for i in range(n_features):
        f = X[:, i]
        su[i] = 2*midd(f, y)/(entropyd(f)+entropyd(y))

    # make sure that j_cmi is positive at the very beginning
    j_cmi = 1

    while True:
        # if len(F) == 0:
        #     # select the feature whose mutual information is the largest

        #
        if is_n_selected_features_specified:
            if len(F) == n_selected_features:
                break
        else:
            if j_cmi < 0:
                break
        for i in range(n_features):
            if i not in F:
                f = X[:, i]
                r[i] = fw[i] * (1 + su[i])
        idx = np.argmax(r)
        F.append(idx)
        r[idx]= -1e30
        f_select = X[:, idx]
        for i in range(n_features):
            if i not in F:
                f = X[:, i]
                iw[i] = 1+(cmidd(y,f,f_select)-midd(f, y)) / (entropyd(f)+entropyd(f_select))
                # calculate j_cmi for feature i (not in F)
                fw[i] = fw[i] * iw[i]
                # record the largest j_cmi and the corresponding feature index

    return np.array(F)
def ucrfs(X, y, **kwargs):
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
    # t2 stores sum_j(I(fj;f)) for each feature f
    t2 = np.zeros(n_features)
    # t3 stores sum_j(I(fj;f|y)) for each feature f
    t3 = np.zeros(n_features)
    for i in range(n_features):
        f = X[:, i]
        t1[i] = midd(f, y)

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
        for i in range(n_features):
            if i not in F:
                f = X[:, i]
                t2[i] += midd(f_select, f)
                # print(2*entropyd(y)-cmidd(y,f_select,f)-2*midd(f_select,y))
                # print(2*entropyd(y),cmidd(f_select,y,f),2*midd(f_select,y))
                t3[i] += cmidd(y, f_select ,f)/(2*entropyd(y)-cmidd(y,f,f_select)-midd(f_select,y)-t1[i])
                # calculate j_cmi for feature i (not in F)
                t = t1[i] - (1.0 / len(F))*t2[i] +t3[i]
                # record the largest j_cmi and the corresponding feature index
                if t > j_cmi:
                    j_cmi = t
                    idx = i
        F.append(idx)
        J_CMI.append(j_cmi)
        MIfy.append(t1[idx])
        f_select = X[:, idx]

    return np.array(F)




