import numpy as np
from scipy import stats
import math
import pandas as pd
import altair as alt
from .transformation import design_matrix

# design for design matrix
class LinearRegression:

    def __init__(self, df, intercept = True):
        self.df = df
        self.intercept = intercept

    def fit(self, col_X, col_y, category = None, interaction = None, ascending = True):
        self.columns  = { }
        self.columns['y'] = col_y
        if self.intercept is True or category is not None:
            self.df = self.df.copy()
            self.df = design_matrix(self.df[col_X + [col_y]], intercept = self.intercept,
                                    category = category, interaction = interaction, ascending = ascending)
            self.columns['x'] = self.df.drop(col_y, axis = 1).columns.tolist()
        else:
            self.columns['x'] = col_X

        X = self.df[self.columns.get('x')]
        X = np.array(X)

        y = self.df[col_y]
        y = np.array(y)

        self.std = np.linalg.inv(np.dot(np.transpose(X), X))
        xty = np.dot(X.transpose(), y.transpose())
        self.coef = np.dot(self.std, xty)
        self.param = { }
        self.param['coef'] = self.coef

        RES_SS = np.dot(y, np.transpose(y)) - np.dot(self.coef, xty)
        self.param['Res S.S.'] = RES_SS
        if self.intercept is True:
            TOTAL_SS = np.dot(y, (np.dot((np.identity(len(self.df)) - 1/len(self.df) * np.ones((len(self.df), len(self.df)))), y.transpose())))
        else:
            TOTAL_SS = np.dot(y.transpose(), y)
        self.param['Reg S.S.'] = TOTAL_SS - RES_SS
        self.param['Total S.S.'] = TOTAL_SS

        self.sigma = math.sqrt(RES_SS / (len(self.df) - len(self.columns.get('x'))))
        self.std = self.std * (self.sigma)**2
        return self

    def change_sigma(self, sigma):
        self.sigma = sigma
        X = np.array(self.df[self.columns.get('x')])
        self.std = np.linalg.inv(np.dot(np.transpose(X), X)) * (self.sigma) ** 2
        return self

    def vif(self, col_X):
        if col_X == 'Intercept':
            return None
        else:
            cols = self.columns.get('x')[ : ]
            if self.intercept is False:
                self.df['Intercept'] = np.ones(len(self.df))
                cols.append('Intercept')
            cols.remove(col_X)
            X = self.df[cols]
            X = np.array(X)
            y = np.array(self.df[col_X])
            xty = np.dot(X.transpose(), y.transpose())
            coef = np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), xty)
            RES_SS = np.dot(y, np.transpose(y)) - np.dot(coef, xty)
            TOTAL_SS = np.dot(y, (np.dot((np.identity(len(self.df)) - 1 / len(self.df) * np.ones((len(self.df), len(self.df)))), y.transpose())))
            if self.intercept is False:
                self.df = self.df.drop('Intercept', axis = 1)
            return TOTAL_SS / RES_SS

    # http://web.vu.lt/mif/a.buteikis/wp-content/uploads/PE_Book/4-5-Multiple-collinearity.html

    def predict(self, X):
        if type(X) == pd.core.frame.DataFrame:
            X = X.values.tolist()
            X = np.array(X)
        else:
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
        if type(X) == pd.core.frame.DataFrame:
            X = X.values.tolist()
            X = np.array(X)
        else:
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
                return_list.append([est - stats.t.ppf(1 - alpha / 2, len(self.df) - len(self.columns.get('x')) - self.intercept) * self.sigma * math.sqrt(1 + std),
                                    est + stats.t.ppf(1 - alpha / 2, len(self.df) - len(self.columns.get('x')) - self.intercept) * self.sigma * math.sqrt(1 + std)])
            elif mode == 'mean':
                return_list.append([est - stats.t.ppf(1 - alpha / 2, len(self.df) - len(self.columns.get('x')) - self.intercept) * self.sigma * math.sqrt(std),
                                    est + stats.t.ppf(1 - alpha / 2, len(self.df) - len(self.columns.get('x')) - self.intercept) * self.sigma * math.sqrt(std)])
        return return_list

    def lack_of_fit(self, result = True):
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
            return (self.param['Res S.S.'] - pure_error) / (len(lack_fit) - len(self.columns.get('x')) - self.intercept)

        else:
            if len(self.df) != len(lack_fit):
                test_stat = ((self.param['Res S.S.'] - pure_error) / (len(lack_fit) - len(self.columns.get('x')) - self.intercept)) / (pure_error / (len(self.df) - len(lack_fit)))
                print('{0: <12}        {1: <10}'.format('Source', 'Sum of Square'))
                print('{0: <12}        {1: <10}'.format('Lack of Fit', "{0:.4f}".format(self.param['Res S.S.'] - pure_error)))
                print('{0: <12}        {1: <10}'.format('Pure Error', "{0:.4f}".format(pure_error)))
                print('F-value:', test_stat)
                print('Pr(Lack of Fit):', 1 - stats.f.cdf(test_stat, len(lack_fit) - len(self.columns.get('x')) - self.intercept, len(self.df) - len(lack_fit)))
                print('')
            else:
                print('The lack of fit cannot be measured as there are no repeated records')

    def summary(self, VIF = False):
        if VIF is True:
            vif = ["{0:.4f}".format(self.vif(col)) if col != 'Intercept' else '' for col in self.columns.get('x')]
        else:
            vif = [''] * len(self.columns.get('x'))
        print('{0: <50}     {1: <15}     {2: <15}     {3: <15}'.format('Factor', 'Coefficient', 'Pr(|t|>0)', 'VIF' if VIF is True else ''))
        for i in range(len(self.columns.get('x'))):
            print('{0: <50}     {1: <15}     {2: <15}     {3: <15}'.format(self.columns.get('x')[i], "{0:.4f}".format(self.coef[i]), "{0:.4f}".format(2 * (1 - stats.t.cdf(abs(self.coef[i]) / math.sqrt(self.std[i][i]), len(self.df) - len(self.columns.get('x'))))), vif[i]))
        print('------------------------------------------------------------------------------------------------------------------------')
        print('{0: <10}        {1: <10}'.format('Source', 'Sum of Square'))
        print('{0: <10}        {1: <10}'.format('Total S.S.', "{0:.4f}".format(self.param['Total S.S.'])))
        print('{0: <10}        {1: <10}'.format('Reg S.S.', "{0:.4f}".format(self.param['Reg S.S.'])))
        print('{0: <10}        {1: <10}'.format('Res S.S.', "{0:.4f}".format(self.param['Res S.S.'])))
        test_stat = (self.param['Reg S.S.'] / (len(self.columns.get('x')) - self.intercept)) / (self.param['Res S.S.'] / (len(self.df) - len(self.columns.get('x'))))
        print('F-value:', test_stat)
        print('Pr(F):', 1 - stats.f.cdf(test_stat, len(self.columns.get('x')) - self.intercept, len(self.df) - len(self.columns.get('x'))))
        print('------------------------------------------------------------------------------------------------------------------------')
        print('R-squared:', 1 - self.param['Res S.S.'] / self.param['Total S.S.'])
        print('Adjusted R-squared:', (1 - (self.param['Res S.S.'] / (len(self.df) - len(self.columns.get('x')))) / (self.param['Total S.S.'] / (len(self.df) - self.intercept))))
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
        if len(self.df) > 10:
            a = 0.5
        else:
            a = 0.375

        residual = self.residual()
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



