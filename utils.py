import pandas as pd
import numpy as np
from scipy import stats
import math

# design for dataframe
class LinearRegression:

    def __init__(self, df, fit_intercept = True):
        self.df = df
        self.fit_intercept = fit_intercept

    def fit(self, col_X, col_y, categorical = None, interaction = None, ascending = True):
        self.X = self.df[col_X]
        self.y = self.df[col_y]

        if categorical is not None:
            self.X = self.X.drop(categorical, axis = 1)
            for cat in categorical:
                values = self.df[cat].unique()
                if ascending is False:
                    values = values[::-1]
                for i in range(len(values) - 1):
                    temp = [ ]
                    for j in range(len(self.df)):
                        if self.df[cat].loc[j] == values[i]:
                            temp.append(1)
                        else:
                            temp.append(0)
                    col = cat + str(values[i])
                    self.X[col] = temp

        if interaction is not None:
            for inter in interaction:
                if inter[0] in categorical and inter[1] in categorical:
                    col_1 = self.df[inter[0]].unique()
                    col_2 = self.df[inter[1]].unique()
                    if ascending is False:
                        col_1 = col_1[::-1]
                        col_2 = col_2[::-1]
                        for i in range(len(col_1) - 1):
                            for j in range(len(col_2) - 1):
                                temp = [ ]
                                for k in range(len(self.df)):
                                    if self.df[inter[0]].loc[k] == col_1[i] and self.df[inter[1]].loc[k] == col_2[j]:
                                        temp.append(1)
                                    else:
                                        temp.append(0)
                                col = inter[0] + str(col_1[i]) + '_' + inter[1] + str(col_2[j])
                                self.X[col] = temp
                else:
                    if inter[0] in categorical and inter[1] not in categorical:
                        cat = inter[0]
                        con = inter[1]
                    else:
                        cat = inter[1]
                        con = inter[0]
                    cols = self.df[cat].unique()
                    if ascending is False:
                        cols = cols[::-1]
                    for i in range(len(cols) - 1):
                        temp = [ ]
                        for j in range(len(self.df)):
                            if self.df[cat].loc[i] == cols[i]:
                                temp.append(self.df[con].loc[i])
                            else:
                                temp.append(0)
                        col = cat + str(cols[i]) + con
                        self.df[col] = temp


        # if interaction is not None:
        #     for inter in interaction:
        #         TEMP = [ ]
        #         for i in range(len(self.df)):
        #             temp = [ ]
        #             if inter[0] in categorical and inter[1] in categorical:
        #                 col_1 = self.df[inter[0]].unique()
        #                 col_2 = self.df[inter[1]].unique()
        #                 if ascending is False:
        #                     col_1 = col_1[::-1]
        #                     col_2 = col_2[::-1]
        #                 for j in range(len(col_1) - 1):
        #                     for k in range(len(col_2) - 1):
        #                         if self.df[inter[0]].loc[i] == col_1[j] and self.df[inter[1]].loc[i] == col_2[k]:
        #                             temp.append(1)
        #                         else:
        #                             temp.append(0)
        #             else:
        #                 if inter[0] in categorical and inter[1] not in categorical:
        #                     cat = inter[0]
        #                     con = inter[1]
        #                 else:
        #                     cat = inter[1]
        #                     con = inter[0]
        #                 col = self.df[cat].unique()
        #                 if ascending is False:
        #                     col = col[::-1]
        #                 for j in range(len(col) - 1):
        #                     if self.df[cat].loc[i] == col[j]:
        #                         temp.append(self.df[con].loc[i])
        #                     else:
        #                         temp.append(0)
        #
        #             TEMP.append(temp)
        #         TEMP = np.array(TEMP)
        #         if inter[0] in categorical and inter[1] in categorical:
        #             col_1 = self.df[inter[0]].unique()
        #             col_2 = self.df[inter[1]].unique()
        #             if ascending is False:
        #                 col_1 = col_1[::-1]
        #                 col_2 = col_2[::-1]
        #             for j in range(len(self.df[inter[0]].unique()) - 1):
        #                 for k in range(len(self.df[inter[1]].unique()) - 1):
        #                     col = inter[0] + '_' + str(col_1[j]) + '_' + inter[1] + '_' + str(col_2[k])
        #                     self.X[col] = TEMP.transpose()[j * (len(col_2) - 1) + k]
        #         else:
        #             if inter[0] in categorical and inter[1] not in categorical:
        #                 cat = inter[0]
        #                 con = inter[1]
        #             else:
        #                 cat = inter[1]
        #                 con = inter[0]
        #             col = self.df[cat].unique()
        #             if ascending is False:
        #                 col = col[::-1]
        #             for j in range(len(col) - 1):
        #                 col = cat + '_' + str(col[j]) + '_' + con
        #                 self.X[col] = TEMP.transpose()[j]

        if self.fit_intercept is True:
            length = len(self.df) - len(self.X.columns.values.tolist()) - 1
            X = np.c_[np.ones(len(self.df)), np.array(self.X)]

        else:
            length = len(self.df) - len(self.X.columns.values.tolist())
            X = self.X
            X = np.array(X)

        y = self.df[col_y]
        y = np.array(y)
        self.coef = np.dot(np.linalg.inv(np.dot(X.transpose(), X)), np.dot(X.transpose(), y.transpose()))
        self.param = { }
        self.param['coef'] = self.coef

        RES_SS = np.dot(y, np.transpose(y)) - np.dot(self.coef, np.dot(np.transpose(X), np.transpose(y)))
        self.param['Res S.S.'] = RES_SS
        if self.fit_intercept is True:
            TOTAL_SS = np.dot(y, (np.dot((np.identity(len(self.df)) - 1/len(df) * np.ones((len(self.df), len(self.df)))), y.transpose())))
        else:
            TOTAL_SS = np.dot(y.transpose(), y)
        self.param['Reg S.S.'] = TOTAL_SS - RES_SS
        self.param['Total S.S.'] = TOTAL_SS

        self.sigma = math.sqrt(RES_SS / length)
        self.r2 = 1 - RES_SS/TOTAL_SS
        self.std = np.linalg.inv(np.dot(np.transpose(X), X)) * (self.sigma)**2
        return self

    def predict(self, X):
        if self.fit_intercept is True:
            X = np.c_[np.ones(len(X)), np.array(X)]
        else:
            X = np.array(X)
        return np.dot(self.coef, X.transpose())

    def change_sigma(self, sigma):
        self.sigma = sigma
        if self.fit_intercept is True:
            X = np.c_[np.ones(len(self.df)), np.array(self.X)]
        else:
            X = np.array(self.X)
        self.std = np.linalg.inv(np.dot(np.transpose(X), X)) * (self.sigma) ** 2
        return self

    def hypothesis_testing(self, col_X, col_y):
        C = [ ]
        for i in range(len(col_X)):
            temp = [ ]
            if self.fit_intercept is True:
                if col_X[i].get('Intercept') is not None:
                    temp.append(col_X[i].get('Intercept'))
                else:
                    temp.append(0)
            for col in self.X.columns.values.tolist():
                if col_X[i].get(col) is not None:
                    temp.append(col_X[i].get(col))
                else:
                    temp.append(0)
            C.append(temp)
        C = np.array(C)
        d = np.array(col_y)
        test_stat = np.dot(np.transpose(np.dot(C, self.coef.transpose()) - d), np.dot(np.linalg.inv(np.dot(C, np.dot(self.std / self.sigma ** 2, C.transpose()))), (np.dot(C, self.coef.transpose()) - d)))/(np.linalg.matrix_rank(C) * self.sigma ** 2)
        p_value = 1 - stats.f.cdf(test_stat, np.linalg.matrix_rank(C), len(self.df) - len(self.X.columns.values.tolist()) - self.fit_intercept)
        print('Test statistic:', test_stat)
        print('Pr:', p_value)

    def predict_interval(self, X, alpha = 0.05, mode = 'individual'):
        if self.fit_intercept is True:
            X = np.c_[np.ones(len(X)), np.array(X)]
        else:
            X = np.array(X)

        return_list = [ ]
        for i in range(len(X)):
            std = np.dot(X[i], np.dot(self.std, X[i].transpose())) / self.sigma ** 2
            est = np.dot(X[i], self.coef.transpose())
            if mode == 'individual':
                return_list.append([est - stats.t.ppf(1 - alpha / 2, len(self.df) - len(self.X.columns) - self.fit_intercept) * self.sigma * math.sqrt(1 + std),
                                    est + stats.t.ppf(1 - alpha / 2, len(self.df) - len(self.X.columns) - self.fit_intercept) * self.sigma * math.sqrt(1 + std)])
            elif mode == 'mean':
                return_list.append([est - stats.t.ppf(1 - alpha / 2, len(self.df) - len(self.X.columns) - self.fit_intercept) * self.sigma * math.sqrt(std),
                                    est + stats.t.ppf(1 - alpha / 2, len(self.df) - len(self.X.columns) - self.fit_intercept) * self.sigma * math.sqrt(std)])
        return return_list

    def lack_of_fit(self, result = True, epsilon = 1e-10):
        lack_fit = {}
        for i in range(len(self.X)):
            temp = []
            for col in self.X.columns.values.tolist():
                temp.append(self.X[col].loc[i])
            temp = tuple(temp)
            if lack_fit.get(temp) is None:
                lack_fit[temp] = [self.y.loc[i]]

            else:
                lack_fit.get(temp).append(self.y.loc[i])

        pure_error = 0
        for key in lack_fit.keys():
            mean = sum(lack_fit.get(key)) / len(lack_fit.get(key))
            for i in range(len(lack_fit.get(key))):
                pure_error += (lack_fit.get(key)[i] - mean) ** 2

        if result is False:
            return (self.param['Res S.S.'] - pure_error) / (len(lack_fit) - len(self.X.columns) - self.fit_intercept)

        else:
            if pure_error - self.param['Res S.S.'] > epsilon or pure_error - self.param['Res S.S.'] < - epsilon:
                test_stat = ((self.param['Res S.S.'] - pure_error) / (len(lack_fit) - len(self.X.columns) - self.fit_intercept)) / (pure_error / (len(self.df) - len(lack_fit)))
                print('{0: <10}        {1: <10}'.format('Source', 'Sum of Square'))
                print('{0: <10}        {1: <10}'.format('Lack of Fit', "{0:.4f}".format(self.param['Res S.S.'] - pure_error)))
                print('{0: <10}        {1: <10}'.format('Pure Error', "{0:.4f}".format(pure_error)))
                print('F-value:', test_stat)
                print('Pr(Lack of Fit):', 1 - stats.f.cdf(test_stat, len(lack_fit) - len(self.X.columns) - self.fit_intercept, len(self.df) - len(lack_fit)))
                print('')
            else:
                print('The lack of fit cannot be measured as there are no repeated records')

    def summary(self):
        columns = self.X.columns.values.tolist()
        print('{0: <15}     {1: <15}     {2: <15}'.format('Factor', 'Coefficient', 'Pr(|t|>0)'))
        if self.fit_intercept is True:
            length = len(self.df) - len(self.X.columns.values.tolist()) - 1
            print('{0: <15}     {1: <15}     {2: <15}'.format('Intercept', "{0:.4f}".format(self.coef[0]), "{0:.4f}".format(2 * (1 - stats.t.cdf(abs(self.coef[0]) / math.sqrt(self.std[0][0]), length)))))
        else:
            length = len(self.df) - len(self.X.columns.values.tolist())
        for i in range(len(columns)):
            print('{0: <15}     {1: <15}     {2: <15}'.format(columns[i], "{0:.4f}".format(self.coef[i + self.fit_intercept]), "{0:.4f}".format(2 * (1 - stats.t.cdf(abs(self.coef[i + self.fit_intercept]) / math.sqrt(self.std[i + self.fit_intercept][i + self.fit_intercept]), length)))))
        print('------------------------------------------------------------')
        print('{0: <10}        {1: <10}'.format('Source', 'Sum of Square'))
        print('{0: <10}        {1: <10}'.format('Total S.S.', "{0:.4f}".format(self.param['Total S.S.'])))
        print('{0: <10}        {1: <10}'.format('Reg S.S.', "{0:.4f}".format(self.param['Reg S.S.'])))
        print('{0: <10}        {1: <10}'.format('Res S.S.', "{0:.4f}".format(self.param['Res S.S.'])))
        test_stat = (self.param['Reg S.S.'] / len(self.X.columns)) / (self.param['Res S.S.'] / (len(self.df) - len(self.X.columns) - self.fit_intercept))
        print('F-value:', test_stat)
        print('Pr(F):', 1 - stats.f.cdf(test_stat, len(self.X.columns) + self.fit_intercept,
                                                  len(self.df) - len(self.X.columns) - self.fit_intercept))
        print('------------------------------------------------------------')
        print('R-squared:', self.r2)
        print('Adjusted R-squared:', (1 - (self.param['Res S.S.'] / (len(self.df) - len(self.X.columns) - self.fit_intercept)) / (self.param['Total S.S.'] / (len(self.df) - self.fit_intercept))))
        print(' ')



# df = pd.DataFrame({'X1' : [1, 2, 2, 3, 3, 4, 5, 5, 5, 6, 7, 8, 8, 9, 9, 10, 11, 11, 11, 12],
#                    'X2' : [-1, 2, 2, 2, 2, 7, 7, 8, 8, 8, 3, 4, 4, 5, 6, 9, 10, 10, 11, 12],
#                    'X1c' : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#                    'X2c' : [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
#                    'y' : [-3, 7, 8, 5, 9, 20, 19, 19, 18, 20, 15, 15, 16, 18, 22, 32, 31, 34, 33, 37]})
# df = pd.DataFrame({'X' : [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
#                    'y' : [79.5, 82.0, 80.6, 84.9, 81.0, 81.5, 82.3, 81.4, 79.5, 83.0, 78.1, 80.2, 81.5, 83.0, 82.1]})
# model = LinearRegression(df, fit_intercept = True)
# model.fit(['X1c', 'X2c'], 'y', categorical = ['X1c', 'X2c'], interaction = [['X1c', 'X2c']], ascending = False)
# model.change_sigma(1)
# model.summary()
# model.hypothesis_testing([{'X1c1' : 1, 'X2c1' : 1, 'X1c1_X2c1' : 1}], [0])
# model.lack_of_fit()
# print(model.predict([[1, 0, 1], [0, 1, 0]]))








