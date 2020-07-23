import EOSS
import numpy as np
import pandas as pd
import time
c_time = str(time.strftime('%Y%m%d %H.%M.%S', time.localtime(time.time())))

if __name__ == '__main__':
    df = pd.read_csv('data/cardio.csv')
    X = df.drop(columns=['label'], axis=0)
    y = df['label'].values
    k = 35
    a = 0.35
    eoss = EOSS.EOSS(X, y, k, a, subspaces_size=1)
    reason = eoss.get_expalinable_subspace()
    reason.to_csv('result/result' + str(c_time) + '.csv')

    reason_true = [['A6']] * len(reason)
    jaccard, precision = eoss.evaluation(reason_true, reason)
    print(jaccard, precision)

