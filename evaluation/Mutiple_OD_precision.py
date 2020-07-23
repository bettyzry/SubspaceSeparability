def precision(T, P):
    '''
    :return: (T and P)/P
    '''
    unions = len(set(T).union(set(P)))
    intersections = len(P)
    return intersections / unions

def avg_precision(y_true, y_pre):
    s = 0
    for i in range(len(y_pre)):
        s += precision(y_true[i], y_pre[i])
    return s / len(y_pre)

if __name__ == '__main__':
    y_true = [['a', 'b'], ['a', 'd'], ['a', 'c']]
    y_pre = [['a', 'c'], ['a', 'b'], ['a', 'c']]
    print(avg_precision(y_true, y_pre))
