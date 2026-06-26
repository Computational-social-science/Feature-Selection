import os
import numpy as np
import scipy.io as scio
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import json
from multiprocessing import Pool, cpu_count
from eamb_cs import EAMB_sp  # 你自定义的模块

def changetosinge(x):
    return float(x)
def prepare_data(dataname, base_url):
    url = os.path.join(base_url, dataname + '.mat')
    data = scio.loadmat(url)
    X0 = pd.DataFrame(data['X'])
    y0 = pd.DataFrame(data['Y'])

    if dataname == 'Dermatology':
        Special = X0.iloc[:, -1]
        a = np.array([item[0] for item in Special])
        label_encoder = LabelEncoder()
        a33 = label_encoder.fit_transform(a)
        X0 = X0.iloc[:, :-1]
        X0[33] = a33

    X0 = X0.applymap(changetosinge)
    y0 = y0.applymap(changetosinge)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y0)
    y = pd.DataFrame(y_encoded)
    X = pd.DataFrame()

    for col in X0.columns:
        X[col] = pd.cut(X0[col], bins=5, labels=False)

    new_columns = [str(i) for i in range(X.shape[1] + 1)]
    X = X.rename(columns=dict(zip(X.columns, new_columns[:-1])))
    y = y.rename(columns=dict(zip(y.columns, [new_columns[-1]])))
    data_processed = pd.concat([X, y], axis=1)

    return data_processed, list(set(y_encoded))

def process_fold_class(args):
    fold, train_index, test_index, nowy, data_processed, dataname = args
    data_train = data_processed.iloc[train_index]
    target_index = str(data_train.shape[1] - 1)
    result = EAMB_sp(target_index, data_train, 0.15, nowy)

    result_dict = {int(nowy): [int(x) for x in result]}
    save_path = f"../feature_select_result/EAMB11/{dataname}_sp_result{fold}.txt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(result_dict, f, indent=4)
    print(f"[Fold {fold}] nowy={nowy} -> 完成，保存至 {save_path}")
    return True

def run_parallel(dataname, base_url, n_jobs=2):
    data_processed, class_set = prepare_data(dataname, base_url)
    kfold = KFold(n_splits=10, shuffle=True, random_state=19)

    tasks = []
    for fold, (train_idx, test_idx) in enumerate(kfold.split(data_processed)):
        if fold != 1:
            continue
        for nowy in class_set:
            tasks.append((fold, train_idx, test_idx, nowy, data_processed, dataname))

    print(f"共生成任务：{len(tasks)} 个，启动 {n_jobs} 个进程并行处理")
    with Pool(processes=n_jobs) as pool:
        pool.map(process_fold_class, tasks)

if __name__ == "__main__":
    datasets = ["Orlraws10P","ORL"]
    base_url = "../dataset/"
    for dataname in datasets:
        print("="*30 + f" 正在处理数据集 {dataname} " + "="*30)
        run_parallel(dataname, base_url, n_jobs=2)
