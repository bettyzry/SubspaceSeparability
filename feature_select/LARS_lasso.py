import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso


class LARS_lasso:
    def __init__(self):
        return

    def feature_select(self, X, y, subspaces_size=5, alpha=0.01):
        """
        :param X: Dataframe
        :param y: np.array
        :return: outlier scores
        """
        lasso = Lasso(alpha=alpha)
        lasso.fit(X.values, y)
        names = X.columns
        feature = pd.DataFrame()
        feature['name'] = names
        feature['score'] = abs(lasso.coef_)
        feature = feature.sort_values(by='score', axis=0, ascending=False)

        names = feature['name'].values
        if subspaces_size > 1:
            subspaces = [names[:i].tolist() for i in range(1, subspaces_size+1)]
        else:
            subspaces = [names[i:i+1].tolist() for i in range(5)]
        return subspaces


if __name__ == '__main__':
    df = pd.read_csv('../data/cardio.csv')
    X = df.drop(columns=['label'], axis=0)
    y = df['label'].values
    lars = LARS_lasso()
    subspaces = lars.feature_select(X, y)
    print(subspaces)