import k_nearest
from feature_select import FS_SVM, LARS_lasso
from ODModel import IForest
import numpy as np
import pandas as pd
from evaluation import Mutiple_OD_jaccard, Mutiple_OD_precision

class EOSS:
    def __init__(self, X, y, k, a, r=None, subspaces_size=5, ODModel=IForest.IForestOD(), feature_select_Model=LARS_lasso.LARS_lasso()):
        """
        :param X: data on all space, DataFrame
        :param k: k-nearest
        :param a:
        :param r:
        """
        self.k = k
        self.r = r or k     # 填写r时为r，否则为k
        self.a = a
        self.subspaces_size = subspaces_size
        self.X = X
        self.y = y
        self.ODModel = ODModel
        self.feature_select_Model = feature_select_Model
        return

    def evaluation(self, reason_true, reason):
        reason_pre = reason['reason'].values
        jaccard = Mutiple_OD_jaccard.avg_jaccard(reason_true, reason_pre)
        precision = Mutiple_OD_precision.avg_precision(reason_true, reason_pre)
        return jaccard, precision

    def get_expalinable_subspace(self):
        """
        :return: self.reason( 'explainable_subspace': index of outlier, 'reason', 'value')
        """
        outliers = np.where(self.y == 1)[0]
        reason = pd.DataFrame(outliers, columns=['outlier'])
        reason['explainable_subspace'] = ''
        reason['value'] = 0.1
        for ii, p in enumerate(outliers):
            explainable_subspace, score = self.get_single_explainable_subspace(p)
            reason['explainable_subspace'][ii] = explainable_subspace
            reason['value'][ii] = score
            print(reason['value'][ii])
        print(reason['value'].values)
        return reason

    def get_single_explainable_subspace(self, p):
        # 通过特征选择确定可能的根因子空间
        subspaces = self.feature_select_Model.feature_select(self.X, self.y, self.subspaces_size)
        accuracies = np.zeros(len(subspaces))
        for ii, subspace in enumerate(subspaces):
            sub_X = self.X[subspace]                            # 该子空间对应的数据
            kn = k_nearest.kNN(sub_X, self.k)

            # --------------------- 数据点采样，构建二分类的数据集 --------------------- #
            Ip = self.get_inlier_index(kn, p)                   # 获取正常对应的索引
            Ip_data_df = kn.X.iloc[Ip, :]                       # 正常对应的数据
            Ip_data_df['label'] = 0
            Op_data_df = self.get_outlier(kn, p)                # 获取正常对应的数据
            Op_data_df['label'] = 1
            Tp_data_df = Ip_data_df.append(Op_data_df)          # 混合两类数据
            p_data_df = kn.X.iloc[[p], :]                       # p对应的数据
            p_data_df['label'] = 1
            Tp_data_df = Tp_data_df.append(p_data_df)           # 混合两类数据
            label = Tp_data_df['label'].values
            Tp_data_df = Tp_data_df.drop('label', axis=1)

            # --------------------- 异常检测 ---------------------- #
            scores = self.ODModel.detect(Tp_data_df, label)     # 进行异常检测
            accuracies[ii] = scores[-1]                         # 点p的得分

            kn.__del__()                                        # 释放kn
        argsort = accuracies.argsort(axis=0)                    # 根据数值大小，进行索引排序
        result = argsort[-1]
        explainable_subspace = subspaces[result]
        print(accuracies, accuracies[result])
        return explainable_subspace, accuracies[result]

    def get_inlier_index(self, kn, p):
        """
        :param kn: class k_nearest
        :param p: the loc of an outlier
        :return Ip: the sampled inlier set of p
        """
        datasize = len(kn.X)
        Rk = kn.get_k_nearest(p)
        Q = [-1]*self.r
        i = 0
        while i < self.r:
            d = int(np.random.uniform(0, datasize))
            if d in Rk or d in Q:
                continue
            else:
                Q[i] = d
                i += 1
        Ip = np.concatenate([Rk, Q])
        return Ip

    def get_outlier(self, kn, p):
        """
        :param kn: class k_nearest
        :param p: the loc of an outlier
        :return Op: the sampled outlier set of p
        """
        columnsize = len(kn.X.columns)
        distances = kn.get_distances(p)
        d = max(distances)
        k_distance = distances[distances.argsort(axis=0)[self.k]]
        l = self.a * (1 / np.sqrt(d)) * k_distance

        mean = kn.X.iloc[p, :].values
        conv = np.ones([columnsize, columnsize]) * l            # 协方差矩阵
        Op = np.random.multivariate_normal(mean=mean, cov=conv, size=self.k+self.r)
        Op = pd.DataFrame(Op, columns=kn.X.columns)
        return Op

def do_eoss(relative_df):
    """
    :param relative_df: ('trace_id', 'device_id', 'cluster_id', 'span_name', cols, 'label')
    :return: predict_df('trace_id', 'device_id', 'cluster_id', 'span_name', 'reason', 'value')
    """
    X = relative_df.drop(['trace_id', 'device_id', 'cluster_id', 'span_name', 'label'], axis=1)
    y = relative_df['label'].values
    k = 35
    a = 0.35

    eoss = EOSS(X, y, k, a, subspaces_size=1)
    reason = eoss.get_expalinable_subspace()  # self.reason( 'outlier': index of outlier, 'reason')
    reason.to_csv('result/reason.csv')
    print(reason['value'].values)
    predict_df = relative_df[['trace_id', 'device_id', 'cluster_id', 'span_name', 'label']]
    predict_df = predict_df[predict_df.label == 1]  # fail=1, api=0

    predict_df['reason'] = [i[0] for i in reason['explainable_subspace'].values]
    predict_df['value'] = reason['value'].values
    predict_df = predict_df.drop(['label'], axis=1)
    return predict_df

if __name__ == '__main__':
    df = pd.read_csv('data/cardio.csv')
    X = df.drop(columns=['label'], axis=0)
    y = df['label'].values
    realtive_df = df.copy()
    realtive_df['trace_id'] = [i for i in range(len(realtive_df))]
    realtive_df['device_id'] = [i for i in range(len(realtive_df))]
    realtive_df['cluster_id'] = [i for i in range(len(realtive_df))]
    realtive_df['span_name'] = 'a'
    realtive_df.to_csv('result/relative_df.csv')
    predict_df = do_eoss(realtive_df)
    predict_df.to_csv('result/result.csv')