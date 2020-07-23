from svmutil import *

class FS_SVM:
    def __init__(self):
        return

    def feature_select(self, X, y):
        """
        :param X: Dataframe
        :param y: np.array
        :return: outlier scores
        """
        parameter = '-s 0 -c 1.0'  # -s 0: C-SVM， -c 1.0 设置参数
        model = svm_train(y, X, parameter)


        scores = [1] * len(X)
        return scores

if __name__ == '__main__':
    y, x = svm_read_problem('data/train.txt')
    yt, xt = svm_read_problem('data/test.txt')
    parameter = '-s 0 -c 1.0'  # -s 0: C-SVM， -c 1.0 设置参数
    model = svm_train(y, x, parameter)

    p_label, p_acc, p_val = svm_predict(yt[0:], xt[0:], model)

