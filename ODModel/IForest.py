from pyod.models.iforest import IForest
import numpy as np


class IForestOD:
    def __init__(self):
        return

    def detect(self, X, y=None):
        """
        :param X: Dataframe
        :param y: np.array
        :return: outlier scores
        """
        rng = np.random.RandomState(42)
        # 构造训练样本
        n_estimators = 200  # 森林中树的棵数
        outliers_fraction = 0.5  # 异常样本比例
        clf = IForest(max_samples='auto', random_state=rng, contamination=outliers_fraction, n_estimators=n_estimators)
        clf.fit(X)
        scores = clf.decision_function(X)
        return scores
