def jaccard_sim(T, P):
    '''
    :return: (T and  P)/(T or P)
    '''
    unions = len(set(T).union(set(P)))
    intersections = len(set(T).intersection(set(P)))
    return intersections / unions


def avg_jaccard(y_true, y_pre):
    s = 0
    for i in range(len(y_pre)):
        s += jaccard_sim(y_true[i], y_pre[i])
    return s / len(y_pre)

if __name__ == '__main__':
    y_true = [['a', 'b'], ['a', 'd'], ['a', 'c']]
    y_pre = [['a'], ['a', 'b'], ['a', 'c']]
    print(avg_jaccard(y_true, y_pre))