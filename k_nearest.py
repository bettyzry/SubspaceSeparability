import pandas as pd
import numpy as np


class kNN:
    def __init__(self, X, k):
        self.X = X      # DataFrame
        self.k = k      # int
        return

    def __del__(self):
        return

    def get_distances(self, point):  # 计算训练集每个点与计算点的欧几米得距离
        points = np.zeros_like(self.X)  # 获得与训练集X一样结构的0集
        points[:] = point
        minusSquare = (self.X - points) ** 2
        EuclideanDistances = np.sqrt(minusSquare.sum(axis=1))  # 训练集每个点与特殊点的欧几米得距离
        return EuclideanDistances

    def get_k_nearest(self, loc):
        """
        :param loc: 异常点p的位置
        :return: 与该点距离最近的k个数据的索引
        """
        p = self.X.iloc[loc, :]
        distances = self.get_distances(p)
        argsort = distances.argsort(axis=0)  # 根据数值大小，进行索引排序
        Rk = argsort[:self.k].values
        return Rk


if __name__ == '__main__':
    df = pd.read_csv('data/cardio.csv')
    X = df.drop(columns=['label'], axis=0)
    y = df['label'].values
    myknn = kNN(X, 4)
    outlier_index = np.where(y == 1)[0]
    for ii in outlier_index:
        points = myknn.get_k_nearest(ii)
        print(points)
