# src/sandbox/datasets.py

"""
测试数据集
"""
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from typing import List, Tuple, Optional
import os
import pickle
import scipy.io as scio
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import glob


@dataclass
class TestDataset:
    """测试数据集"""
    name: str
    X: np.ndarray
    y: np.ndarray
    metadata: dict
    
    @property
    def n_samples(self) -> int:
        return self.X.shape[0]
    
    @property
    def n_features(self) -> int:
        return self.X.shape[1]
    
    @property
    def n_classes(self) -> int:
        return len(np.unique(self.y))


class DatasetGenerator:
    """数据集生成器"""
    
    @staticmethod
    def generate_classification(
        name: str,
        n_samples: int,
        n_features: int,
        n_informative: int,
        n_redundant: int = 0,
        n_clusters_per_class: int = 2,
        flip_y: float = 0.01,
        random_state: int = 42
    ) -> TestDataset:
        """生成分类数据集"""
        
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_clusters_per_class=n_clusters_per_class,
            flip_y=flip_y,
            random_state=random_state
        )
        
        # 标准化
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        metadata = {
            'n_informative': n_informative,
            'n_redundant': n_redundant,
            'n_clusters_per_class': n_clusters_per_class,
            'flip_y': flip_y,
            'type': 'classification'
        }
        
        return TestDataset(name=name, X=X, y=y, metadata=metadata)
    
    @staticmethod
    def create_standard_benchmark() -> List[TestDataset]:
        """创建标准基准数据集"""
        datasets = []
        
        # # 1. Madelon-like: 高维，少量相关特征
        # datasets.append(DatasetGenerator.generate_classification(
        #     name="madelon_like",
        #     n_samples=2000,
        #     n_features=500,
        #     n_informative=20,
        #     n_redundant=20,
        #     n_clusters_per_class=5,
        #     random_state=42
        # ))
        
        # # 2. Gisette-like: 高维，平衡
        # datasets.append(DatasetGenerator.generate_classification(
        #     name="gisette_like",
        #     n_samples=6000,
        #     n_features=500,
        #     n_informative=50,
        #     n_redundant=50,
        #     n_clusters_per_class=2,
        #     random_state=42
        # ))
        
        # # 3. Arcene-like: 高维小样本
        # datasets.append(DatasetGenerator.generate_classification(
        #     name="arcene_like",
        #     n_samples=200,
        #     n_features=10000,
        #     n_informative=100,
        #     n_redundant=100,
        #     n_clusters_per_class=1,
        #     random_state=42
        # ))
        
        # # 4. High correlation: 特征间高相关
        # datasets.append(DatasetGenerator.generate_classification(
        #     name="high_corr",
        #     n_samples=1000,
        #     n_features=100,
        #     n_informative=20,
        #     n_redundant=60,
        #     n_clusters_per_class=2,
        #     random_state=42
        # ))
        
        # # 5. Low signal: 弱信号
        # datasets.append(DatasetGenerator.generate_classification(
        #     name="low_signal",
        #     n_samples=2000,
        #     n_features=200,
        #     n_informative=10,
        #     n_redundant=10,
        #     flip_y=0.1,
        #     random_state=42
        # ))
        
        return datasets




class DatasetManager:
    """数据集管理器：支持内置基准测试与本地 .mat/.csv 数据加载"""
    
    def __init__(self, base_dir: str = None, cache_dir: str = "data/cache", raw_dir: str = "data/raw"):
        # 自动定位项目根目录
        if base_dir is None:
            # 假设当前文件在 src/sandbox/datasets/ 目录下，向上退三级到达项目根目录
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        self.base_dir = base_dir
        self.cache_dir = os.path.join(base_dir, cache_dir)
        self.raw_dir = os.path.join(base_dir, raw_dir)
        self.datasets: dict = {}
        
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.raw_dir, exist_ok=True)

    def get_dataset(self, name_or_path: str) -> 'TestDataset':
        """
        统一的数据获取入口。
        支持：1. 完整路径 2. data/raw 下的文件名 3. 内置数据集名称
        """
        # 1. 检查是否是直接可用的路径
        if os.path.exists(name_or_path):
            return self._load_by_extension(name_or_path)
            
        # 2. 检查是否是 data/raw 下的文件名
        raw_path = os.path.join(self.raw_dir, name_or_path)
        if os.path.exists(raw_path):
            return self._load_by_extension(raw_path)
            
        # 3. 否则，走缓存或标准生成逻辑
        return self.load_or_generate(name_or_path)

    def _load_by_extension(self, path: str) -> 'TestDataset':
        """内部辅助：根据后缀名路由加载函数"""
        if path.endswith('.mat'):
            return self.load_mat_dataset(path)
        elif path.endswith('.csv'):
            return self.load_csv_dataset(path)
        else:
            raise ValueError(f"不支持的文件格式: {path}")

    def load_or_generate(self, name: str) -> 'TestDataset':
        """加载缓存或生成标准基准数据集"""
        if name in self.datasets:
            return self.datasets[name]
        
        cache_path = os.path.join(self.cache_dir, f"{name}.pkl")
        
        # 尝试从缓存加载
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                dataset = pickle.load(f)
        else:
            # 生成标准数据集并缓存
            standard_datasets = DatasetGenerator.create_standard_benchmark()
            for ds in standard_datasets:
                self.datasets[ds.name] = ds
                with open(os.path.join(self.cache_dir, f"{ds.name}.pkl"), 'wb') as f:
                    pickle.dump(ds, f)
            
            if name in self.datasets:
                dataset = self.datasets[name]
            else:
                raise ValueError(f"未知数据集名称: {name}")
        
        self.datasets[name] = dataset
        return dataset

    def get_all_datasets(self) -> List['TestDataset']:
        """
        核心功能：一键获取所有数据集。
        包含：内置标准数据 + data/raw 目录下的所有 .mat 文件
        """
        all_ds = []
        
        # 1. 加载内置标准数据集
        try:
            all_ds.extend(DatasetGenerator.create_standard_benchmark())
        except Exception as e:
            print(f"⚠️ 无法生成标准数据集: {e}")

        # 2. 自动扫描 data/raw 目录
        mat_files = glob.glob(os.path.join(self.raw_dir, "*.csv"))
        for f in mat_files:
            try:
                all_ds.append(self.load_csv_dataset(f))
            except Exception as e:
                print(f"❌ 自动加载失败 {f}: {e}")
        if all_ds == []:
            mat_files = glob.glob(os.path.join(self.raw_dir, "*.mat"))
            for f in mat_files:
                try:
                    all_ds.append(self.load_mat_dataset(f))
                except Exception as e:
                    print(f"❌ 自动加载失败 {f}: {e}")
                
        return all_ds

    def load_mat_dataset(self, mat_path: str, target_key: str = 'Y', data_key: str = 'X') -> 'TestDataset':
        """加载 MAT 数据集并进行预处理（编码与离散化）"""
        data = scio.loadmat(mat_path)
        X0 = pd.DataFrame(data[data_key])
        y0 = pd.DataFrame(data[target_key])

        # 标签编码
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y0.values.ravel())

        # 数据离散化 (等宽分箱)
        X = pd.DataFrame()
        for col in X0.columns:
            X[col] = pd.cut(X0[col], bins=5, labels=False)
        X.columns = list(range(X.shape[1]))

        return TestDataset(
            name=os.path.basename(mat_path),
            X=X.values,
            y=y,
            metadata={
                'source': 'custom_mat',
                'target_key': target_key,
                'classes': list(label_encoder.classes_)
            }
        )

    def load_csv_dataset(self, csv_path: str, target_column: str = 'target') -> 'TestDataset':
        """加载自定义 CSV 数据集"""
        df = pd.read_csv(csv_path)
        if target_column not in df.columns:
            # 如果没指定标签列，默认取最后一列
            target_column = df.columns[-1]
            
        y = df[target_column].values
        X = df.drop(columns=[target_column]).values
        
        return TestDataset(
            name=os.path.basename(csv_path),
            X=X,
            y=y,
            metadata={'source': 'custom_csv', 'target_column': target_column}
        )
    