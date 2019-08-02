import numpy as np
from scipy import stats
import math
import pandas as pd
import altair as alt

# design for dataframe
class LinearRegression:

    def __init__(self, df, fit_intercept = True):
        self.df = df
        self.fit_intercept = fit_intercept

    def fit(self, col_X, col_y, categorical = None, interaction = None, ascending = True):
        self.columns  = { }
        self.columns['x'] = [ ]
        self.columns['y'] = col_y

        if self.fit_intercept is True:
            self.df['Intercept'] = [1] * len(self.df)
            self.columns.get('x').append('Intercept')

        if categorical is not None:
            self.df = self.df.drop(categorical, axis = 1)
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
                    self.df[col] = temp
                    self.columns.get('x').append(col)
        else:
            for col in col_X:
                self.columns.get('x').append(col)

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
                                self.df[col] = temp
                                self.columns.get('x').append(col)
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
                        self.columns.get('x').append(col)

        X = self.df[self.columns.get('x')]
        X = np.array(X)

        y = self.df[col_y]
        y = np.array(y)
        self.coef = np.dot(np.linalg.inv(np.dot(X.transpose(), X)), np.dot(X.transpose(), y.transpose()))
        self.param = { }
        self.param['coef'] = self.coef

        RES_SS = np.dot(y, np.transpose(y)) - np.dot(self.coef, np.dot(np.transpose(X), np.transpose(y)))
        self.param['Res S.S.'] = RES_SS
        if self.fit_intercept is True:
            TOTAL_SS = np.dot(y, (np.dot((np.identity(len(self.df)) - 1/len(self.df) * np.ones((len(self.df), len(self.df)))), y.transpose())))
        else:
            TOTAL_SS = np.dot(y.transpose(), y)
        self.param['Reg S.S.'] = TOTAL_SS - RES_SS
        self.param['Total S.S.'] = TOTAL_SS

        self.sigma = math.sqrt(RES_SS / (len(self.df) - len(self.columns.get('x'))))
        self.r2 = 1 - RES_SS/TOTAL_SS
        self.std = np.linalg.inv(np.dot(np.transpose(X), X)) * (self.sigma)**2
        return self

    def predict(self, X):
        TEMP = [ ]
        for i in range(len(X)):
            if type(X[i]) == dict:
                temp = []
                for col in self.columns.get('x'):
                    if X[i].get(col) is not None:
                        temp.append(X[i].get(col))
                    else:
                        temp.append(0)
                TEMP.append(temp)
            else:
                TEMP.append(X[i])
        X = np.array(TEMP)
        return np.dot(self.coef, X.transpose())

    def change_sigma(self, sigma):
        self.sigma = sigma
        X = np.array(self.df[self.columns.get('x')])
        self.std = np.linalg.inv(np.dot(np.transpose(X), X)) * (self.sigma) ** 2
        return self

    def hypothesis_testing(self, col_X, col_y):
        C = [ ]
        for i in range(len(col_X)):
            if type(col_X[i]) == dict:
                temp = [ ]
                for col in self.columns.get('x'):
                    if col_X[i].get(col) is not None:
                        temp.append(col_X[i].get(col))
                    else:
                        temp.append(0)
                C.append(temp)
            else:
                C.append(col_X[i])
        C = np.array(C)
        d = np.array(col_y)
        test_stat = np.dot(np.transpose(np.dot(C, self.coef.transpose()) - d), np.dot(np.linalg.inv(np.dot(C, np.dot(self.std / self.sigma ** 2, C.transpose()))), (np.dot(C, self.coef.transpose()) - d)))/(np.linalg.matrix_rank(C) * self.sigma ** 2)
        p_value = 1 - stats.f.cdf(test_stat, np.linalg.matrix_rank(C), len(self.df) - len(self.columns.get('x')))
        print('Test statistic:', test_stat)
        print('Pr:', p_value)

    def predict_interval(self, X, alpha = 0.05, mode = 'individual'):
        TEMP = []
        for i in range(len(X)):
            if type(X[i]) == dict:
                temp = []
                for col in self.columns.get('x'):
                    if X[i].get(col) is not None:
                        temp.append(X[i].get(col))
                    else:
                        temp.append(0)
                TEMP.append(temp)
            else:
                TEMP.append(X[i])
        X = np.array(TEMP)

        return_list = [ ]
        for i in range(len(X)):
            std = np.dot(X[i], np.dot(self.std, X[i].transpose())) / self.sigma ** 2
            est = np.dot(X[i], self.coef.transpose())
            if mode == 'individual':
                return_list.append([est - stats.t.ppf(1 - alpha / 2, len(self.df) - len(self.columns.get('x')) - self.fit_intercept) * self.sigma * math.sqrt(1 + std),
                                    est + stats.t.ppf(1 - alpha / 2, len(self.df) - len(self.columns.get('x')) - self.fit_intercept) * self.sigma * math.sqrt(1 + std)])
            elif mode == 'mean':
                return_list.append([est - stats.t.ppf(1 - alpha / 2, len(self.df) - len(self.columns.get('x')) - self.fit_intercept) * self.sigma * math.sqrt(std),
                                    est + stats.t.ppf(1 - alpha / 2, len(self.df) - len(self.columns.get('x')) - self.fit_intercept) * self.sigma * math.sqrt(std)])
        return return_list

    def lack_of_fit(self, result = True, epsilon = 1e-10):
        lack_fit = {}
        for i in range(len(self.df)):
            temp = []
            for col in self.columns.get('x'):
                temp.append(self.df[col].loc[i])
            temp = tuple(temp)
            if lack_fit.get(temp) is None:
                lack_fit[temp] = [self.df[self.columns.get('y')].loc[i]]

            else:
                lack_fit.get(temp).append(self.df[self.columns.get('y')].loc[i])

        pure_error = 0
        for key in lack_fit.keys():
            mean = sum(lack_fit.get(key)) / len(lack_fit.get(key))
            for i in range(len(lack_fit.get(key))):
                pure_error += (lack_fit.get(key)[i] - mean) ** 2

        if result is False:
            return (self.param['Res S.S.'] - pure_error) / (len(lack_fit) - len(self.columns.get('x')) - self.fit_intercept)

        else:
            if pure_error - self.param['Res S.S.'] > epsilon or pure_error - self.param['Res S.S.'] < - epsilon:
                test_stat = ((self.param['Res S.S.'] - pure_error) / (len(lack_fit) - len(self.columns.get('x')) - self.fit_intercept)) / (pure_error / (len(self.df) - len(lack_fit)))
                print('{0: <12}        {1: <10}'.format('Source', 'Sum of Square'))
                print('{0: <12}        {1: <10}'.format('Lack of Fit', "{0:.4f}".format(self.param['Res S.S.'] - pure_error)))
                print('{0: <12}        {1: <10}'.format('Pure Error', "{0:.4f}".format(pure_error)))
                print('F-value:', test_stat)
                print('Pr(Lack of Fit):', 1 - stats.f.cdf(test_stat, len(lack_fit) - len(self.columns.get('x')) - self.fit_intercept, len(self.df) - len(lack_fit)))
                print('')
            else:
                print('The lack of fit cannot be measured as there are no repeated records')

    def summary(self):
        print('{0: <15}     {1: <15}     {2: <15}'.format('Factor', 'Coefficient', 'Pr(|t|>0)'))
        for i in range(len(self.columns.get('x'))):
            print('{0: <15}     {1: <15}     {2: <15}'.format(self.columns.get('x')[i], "{0:.4f}".format(self.coef[i]), "{0:.4f}".format(2 * (1 - stats.t.cdf(abs(self.coef[i]) / math.sqrt(self.std[i][i]), len(self.df) - len(self.columns.get('x')))))))
        print('------------------------------------------------------------')
        print('{0: <10}        {1: <10}'.format('Source', 'Sum of Square'))
        print('{0: <10}        {1: <10}'.format('Total S.S.', "{0:.4f}".format(self.param['Total S.S.'])))
        print('{0: <10}        {1: <10}'.format('Reg S.S.', "{0:.4f}".format(self.param['Reg S.S.'])))
        print('{0: <10}        {1: <10}'.format('Res S.S.', "{0:.4f}".format(self.param['Res S.S.'])))
        test_stat = (self.param['Reg S.S.'] / (len(self.columns.get('x')) - self.fit_intercept)) / (self.param['Res S.S.'] / (len(self.df) - len(self.columns.get('x'))))
        print('F-value:', test_stat)
        print('Pr(F):', 1 - stats.f.cdf(test_stat, len(self.columns.get('x')) - self.fit_intercept, len(self.df) - len(self.columns.get('x'))))
        print('------------------------------------------------------------')
        print('R-squared:', self.r2)
        print('Adjusted R-squared:', (1 - (self.param['Res S.S.'] / (len(self.df) - len(self.columns.get('x')))) / (self.param['Total S.S.'] / (len(self.df) - self.fit_intercept))))
        print('AIC:', len(self.df) * math.log(self.param['Res S.S.'] / len(self.df)) + 2 * len(self.columns.get('x')))
        print('BIC:', len(self.df) * math.log(self.param['Res S.S.'] / len(self.df)) + math.log(len(self.df)) * len(self.columns.get('x')))
        print(' ')

    def residual(self, type = 'residual'):
        residual = []
        X = np.array(self.df[self.columns.get('x')])

        if type == 'residual':
            for i in range(len(self.df)):
                residual.append(self.df[self.columns.get('y')].loc[i] - np.dot(self.coef, X.transpose())[i])
            return np.array(residual)
        elif type == 'standardised':
            for i in range(len(self.df)):
                residual.append(self.df[self.columns.get('y')].loc[i] - np.dot(self.coef, X.transpose())[i])
            residual = np.array(residual)
            return residual / math.sqrt(np.dot(residual, residual.transpose()) / (len(self.df) - 1))

    def residual_plot(self):
        residual = self.residual('standardised')
        X = np.array(self.df[self.columns.get('x')])
        y = np.dot(self.coef, X.transpose())
        df = pd.DataFrame({'residual' : residual,
                           'predict' : y})
        df['index'] = df.index
        line = np.arange(min(df['predict']) - 0.1, max(df['predict']) + 0.1,
                         (max(df['predict']) - min(df['predict']) + 0.2) / 100)
        line = np.array(line)
        line = pd.DataFrame({'x': line, 'y': np.array([0] * len(line))})
        line = alt.Chart(line).mark_line().encode(x = 'x', y = 'y')
        chart = alt.Chart(df, title='Residual plot').mark_point().encode(x = 'predict', y = 'residual', tooltip=['index'])
        chart.encoding.x.title = 'Predicted'
        chart.encoding.y.title = 'Standardised residual'
        return line + chart

    def normal_plot(self):
        residual = self.residual()

        if len(residual) > 10:
            a = 0.5
        else:
            a = 0.375

        df = pd.DataFrame({'residual' : residual})
        df['index'] = df.index
        df = df.sort_values(by = 'residual', ascending = True)
        z = [ ]
        for i in range(len(df)):
            value = (i + 1 - a) / (len(df) + 1 - 2 * a)
            z.append(stats.norm.ppf(value))
        df['quantile'] = z

        residual = np.array(df['residual'])
        z = np.array(df['quantile'])
        coef = np.dot(residual, z.transpose()) / np.dot(z, z.transpose())
        line = np.arange(min(df['quantile']) - 0.1, max(df['quantile']) + 0.1, (max(df['quantile']) - min(df['quantile']) + 0.2) / 100)
        line = np.array(line)
        line = pd.DataFrame({'x' : line, 'y' : coef * line})
        line = alt.Chart(line).mark_line().encode(x = 'x', y = 'y')
        chart = alt.Chart(df, title = 'Normal plot').mark_point().encode(x = 'quantile', y = 'residual', tooltip = ['index'])
        chart.encoding.x.title = 'Quantile'
        chart.encoding.y.title = 'Residual'
        return line + chart

    def normal_test(self):
        residue = [ ]
        X = np.array(self.df[self.columns.get('x')])
        for i in range(len(self.df)):
            residue.append(self.df[self.columns.get('y')].loc[i] - np.dot(self.coef, X.transpose())[i])
        residue = np.array(residue)
        print('{0: <30}     {1: <20}     {2: <20}'.format('Test', 'Test Statistic', 'Pr'))
        stat, p = stats.shapiro(residue)
        print('{0: <30}     {1: <20}     {2: <20}'.format('Shapiro-Wilk', "{0:.4f}".format(stat), "{0:.4f}".format(p)))
        stat, p = stats.kstest(residue, 'norm')
        print('{0: <30}     {1: <20}     {2: <20}'.format('Kolmogorov-Smirnov', "{0:.4f}".format(stat), "{0:.4f}".format(p)))
        stat, p = stats.normaltest(residue)
        print('{0: <30}     {1: <20}     {2: <20}'.format('D’Agostino’s K^2', "{0:.4f}".format(stat), "{0:.4f}".format(p)))
