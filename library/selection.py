import numpy as np
from scipy import stats
from .regression import LinearRegression

def ForwardSelection(df, col_X, col_y, alpha = 0.05, detail = True):
    columns = [ ]
    model = LinearRegression(df)
    model.fit(columns, col_y)
    RES_SS = model.param['Res S.S.']
    while len(col_X) != len(columns):
        res_min = None
        index = 0
        for i in range(len(col_X)):
            if col_X[i] not in columns:
                cols = columns[ : ]
                cols.append(col_X[i])
                model.df = df[cols + [col_y]]
                model.fit(cols, col_y)
                if res_min is None or res_min > model.param['Res S.S.']:
                    res_min = model.param['Res S.S.']
                    index = i
        test_stat = (RES_SS - res_min) / (res_min / (len(df) - len(columns) - 2))
        p_value = 1 - stats.f.cdf(test_stat, 1, len(df) - len(columns) - 2)
        if detail is True or p_value > alpha or len(col_X) == len(columns) + 1:
            print('Variable in model:', ', '.join(columns))
            print('{0: <50}     {1: <30}     {2: <15}'.format('Variable Entered', 'Res S.S. before Entered',
                                                              'Res S.S. after Entered'))
            print('{0: <50}     {1: <30}     {2: <15}'.format(col_X[index], "{0:.4f}".format(RES_SS), "{0:.4f}".format(res_min)))
            print('Test Statistic:', test_stat)
            print('p-value:', p_value)
            print('Decision:', 'Do not enter' if p_value > alpha else 'Enter')
            print('------------------------------------------------------------------------------------------------------------------------')
        if p_value > alpha:
            break
        else:
            columns.append(col_X[index])
            RES_SS = res_min

def BackwardSelection(df, col_X, col_y, alpha = 0.05, detail = True):
    if 'Intercept' in df.columns:
        df = df.drop('Intercept', axis = 1)
    if 'Intercept' in col_X:
        col_X.remove('Intercept')
    model = LinearRegression(df)
    model.fit(col_X, col_y)
    RES_SS = model.param['Res S.S.']
    while len(col_X) != 0:
        res_min = None
        index = 0
        for i in range(len(col_X)):
            cols = col_X[ : ]
            cols.pop(i)
            model.df = df[cols + [col_y]]
            model.fit(cols, col_y)
            if res_min is None or res_min > model.param['Res S.S.']:
                res_min = model.param['Res S.S.']
                index = i
        test_stat = (res_min - RES_SS) / (RES_SS / (len(df) - len(col_X) - 1))
        p_value = 1 - stats.f.cdf(test_stat, 1, len(df) - len(col_X) - 1)
        if detail is True or p_value <= alpha or len(col_X) == 1:
            print('Variable in model:', ', '.join(col_X))
            print('{0: <50}     {1: <30}     {2: <15}'.format('Variable Removed', 'Res S.S. before Removal',
                                                              'Res S.S. after Removal'))
            print('{0: <50}     {1: <30}     {2: <15}'.format(col_X[index], "{0:.4f}".format(RES_SS), "{0:.4f}".format(res_min)))
            print('Test Statistic:', test_stat)
            print('p-value:', p_value)
            print('Decision:', 'Remove' if p_value > alpha else 'Do not remove')
            print('------------------------------------------------------------------------------------------------------------------------')
        if p_value <= alpha:
            break
        else:
            col_X.pop(index)
            RES_SS = res_min

def StepwiseSelection(df, col_X, col_y, in_alpha = 0.05, out_alpha = 0.05, detail = True):
    if 'Intercept' in df.columns:
        df = df.drop('Intercept', axis = 1)
    if 'Intercept' in col_X:
        col_X.remove('Intercept')

    columns = [ ]
    model = LinearRegression(df)
    model.fit(columns, col_y)
    RES_SS = model.param['Res S.S.']
    while len(columns) != len(col_X):
        res = None
        index = 0
        for i in range(len(col_X)):
            if col_X[i] not in columns:
                cols = columns[ : ]
                cols.append(col_X[i])
                model.df = df[cols + [col_y]]
                model.fit(cols, col_y)
                if res is None or res > model.param['Res S.S.']:
                    res = model.param['Res S.S.']
                    index = i
        test_stat = (RES_SS - res) / (res / (len(df) - len(columns) - 2))
        p_value = 1 - stats.f.cdf(test_stat, 1, len(df) - len(columns) - 2)
        if detail is True or p_value > in_alpha or len(col_X) == len(columns) + 1:
            print('Variable in model:', ', '.join(columns))
            print('{0: <50}     {1: <30}     {2: <15}'.format('Variable Entered', 'Res S.S. before Entered',
                                                              'Res S.S. after Entered'))
            print('{0: <50}     {1: <30}     {2: <15}'.format(col_X[index], "{0:.4f}".format(RES_SS),
                                                              "{0:.4f}".format(res)))
            print('Test Statistic:', test_stat)
            print('p-value:', p_value)
            print('Decision:', 'Do not enter' if p_value > in_alpha else 'Enter')
            print('------------------------------------------------------------------------------------------------------------------------')

        if p_value <= in_alpha:
            columns.append(col_X[index])
            RES_SS = res
        else:
            break

        if len(columns) != 1:
            for i in range(len(columns)):
                res = None
                index = 0
                cols = columns[ : ]
                cols.pop(i)
                model.df = df[cols + [col_y]]
                model.fit(cols, col_y)
                if res is None or res < model.param['Res S.S.']:
                    res = model.param['Res S.S.']
                    index = i
            test_stat = (res - RES_SS) / (RES_SS / (len(df) - len(columns) - 1))
            p_value = 1 - stats.f.cdf(test_stat, 1, len(df) - len(columns) - 1)
            if detail is True or p_value <= out_alpha:
                print('Variable in model:', ', '.join(columns))
                print('{0: <50}     {1: <30}     {2: <15}'.format('Variable Removed', 'Res S.S. before Removal',
                                                                  'Res S.S. after Removal'))
                print('{0: <50}     {1: <30}     {2: <15}'.format(columns[index], "{0:.4f}".format(RES_SS),
                                                                  "{0:.4f}".format(res)))
                print('Test Statistic:', test_stat)
                print('p-value:', p_value)
                print('Decision:', 'Remove' if p_value > out_alpha else 'Do not remove')
                print('------------------------------------------------------------------------------------------------------------------------')

            if p_value > out_alpha:
                columns.pop(index)
