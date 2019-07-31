import pandas as pd
import numpy as np
from scipy import stats
import math

# design for dataframe
class LinearRegression:

    def __init__(self, df, fit_intercept = True):
        self.df = df
        self.fit_intercept = fit_intercept

    def fit(self, col_X, col_y):
        self.X = self.df[col_X]
        self.y = self.df[col_y]

        if self.fit_intercept is True:
            length = len(self.df) - len(self.X.columns.values.tolist()) - 1
            X = np.c_[np.ones(len(self.df)), np.array(self.df[col_X])]
        else:
            length = len(self.df) - len(self.X.columns.values.tolist())
            X = self.df[col_X]
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
        if self.fit_intercept is True:
            p_value = 1 - stats.f.cdf(test_stat, np.ndim(C), len(self.df) - len(self.X.columns.values.tolist()) - 1)
        else:
            p_value = 1 - stats.f.cdf(test_stat, np.ndim(C), len(self.df) - len(self.X.columns.values.tolist()))
        print('Test: statistic:', test_stat)
        print('Pr:', p_value)

    def lack_of_fit(self, result = True):
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

        test_stat = ((self.param['Res S.S.'] - pure_error) / (len(lack_fit) - len(self.X.columns) - self.fit_intercept)) / (pure_error / (len(self.df) - len(lack_fit)))
        if result is False:
            return (self.param['Res S.S.'] - pure_error) / (len(lack_fit) - len(self.X.columns) - self.fit_intercept)

        else:
            print('{}\t\t\t\t\t{}'.format('Source', 'Sum of Square'))
            print('{}:\t\t\t{}'.format('Lack of Fit', "{0:.4f}".format(self.param['Res S.S.'] - pure_error)))
            print('{}:\t\t\t\t{}'.format('Pure Error', "{0:.4f}".format(pure_error)))
            print('F-value:', test_stat)
            print('Pr(Lack of Fit):', 1 - stats.f.cdf(test_stat, len(lack_fit) - len(self.X.columns) - self.fit_intercept, len(self.df) - len(lack_fit)))

    def summary(self):
        columns = self.X.columns.values.tolist()
        print('{}\t\t\t{}\t\t\t{}'.format('Factor', 'Coefficient', 'Pr(|t|>0)'))
        if self.fit_intercept is True:
            length = len(self.df) - len(self.X.columns.values.tolist()) - 1
            print('{}:\t\t{}\t\t\t\t{}'.format('Intercept', "{0:.4f}".format(self.coef[0]), "{0:.4f}".format(1 - stats.t.cdf(abs(self.coef[0]), length))))
        else:
            length = len(self.df) - len(self.X.columns.values.tolist())
        for i in range(len(columns)):
            print('{}:\t\t\t\t{}\t\t\t\t{}'.format(columns[i], "{0:.4f}".format(self.coef[i + self.fit_intercept]), "{0:.4f}".format(1 - stats.t.cdf(abs(self.coef[i + self.fit_intercept]), length))))
        print('------------------------------------------------------------')
        print('{}\t\t\t\t\t{}'.format('Source', 'Sum of Square'))
        print('{}:\t\t\t\t{}'.format('Total S.S.', "{0:.4f}".format(self.param['Total S.S.'])))
        print('{}:\t\t\t\t{}'.format('Reg S.S.', "{0:.4f}".format(self.param['Reg S.S.'])))
        print('{}:\t\t\t\t{}'.format('Res S.S.', "{0:.4f}".format(self.param['Res S.S.'])))
        test_stat = (self.param['Reg S.S.'] / len(self.X.columns)) / (self.param['Res S.S.'] / (len(self.df) - len(self.X.columns) - self.fit_intercept))
        print('F-value:', test_stat)
        print('Pr(F):', 1 - stats.f.cdf(test_stat, len(self.X.columns) + self.fit_intercept,
                                                  len(self.df) - len(self.X.columns) - self.fit_intercept))
        print('------------------------------------------------------------')
        print('R-squared:', self.r2)
        print('Adjusted R-squared:', (1 - (self.param['Res S.S.'] / (len(self.df) - len(self.X.columns) - self.fit_intercept)) / (self.param['Total S.S.'] / (len(self.df) - self.fit_intercept))))

df = pd.DataFrame({'X1' : [1, 2, 2, 3, 3, 4, 5, 5, 5, 6, 7, 8, 8, 9, 9, 10, 11, 11, 11, 12],
                   'X2' : [-1, 2, 2, 2, 2, 7, 7, 8, 8, 8, 3, 4, 4, 5, 6, 9, 10, 10, 11, 12],
                   'y' : [-3, 7, 8, 5, 9, 20, 19, 19, 18, 20, 15, 15, 16, 18, 22, 32, 31, 34, 33, 37]})

model = LinearRegression(df, fit_intercept = True)
model.fit(['X1', 'X2'], 'y')
# model.change_sigma(1)
# model.hypothesis_testing([{'X1' : 1, 'X2' : -2}], [0])
model.summary()
print(' ')
# model.lack_of_fit()
# print(model.predict([[5, 10], [6, 9]]))








