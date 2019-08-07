import numpy as np
import pandas as pd

def design_matrix(df, intercept = True, categorical = None, interaction = None, ascending = True, copy = False):
    if copy is True:
        df = df.copy()
    if intercept is True:
        df['Intercept'] = np.ones(len(df))
        columns = df.columns.tolist()
        columns = [columns[-1]] + columns[:-1]
        df = df[columns]

    if categorical is not None:
        for col in categorical:
            if ascending is True:
                df1 = pd.get_dummies(df[col], prefix=col)
                df1 = df1.drop(df1.columns[len(df1.columns) - 1], axis=1)
            else:
                df1 = pd.get_dummies(df[col], prefix=col, drop_first = True)
            df = pd.merge(df, df1, left_index = True, right_index = True)

    if interaction is not None:
        for inter in interaction:
            if inter[0] in categorical and inter[1] in categorical:
                col_1 = df[inter[0]].unique()
                col_2 = df[inter[1]].unique()
                if ascending is False:
                    col_1 = col_1[::-1]
                    col_2 = col_2[::-1]
                for i in range(len(col_1) - 1):
                    for j in range(len(col_2) - 1):
                        temp = [ ]
                        for k in range(len(df)):
                            if df[inter[0]].loc[k] == col_1[i] and df[inter[1]].loc[k] == col_2[j]:
                                temp.append(1)
                            else:
                                temp.append(0)
                        col_name = inter[0] + '_' + str(col_1[i]) + '_' + inter[1] + '_' + str(col_2[j])
                        df[col_name] = temp
            else:
                if inter[0] in categorical and inter[1] not in categorical:
                    col_1 = inter[0]
                    col_2 = inter[1]
                else:
                    col_1 = inter[1]
                    col_2 = inter[0]
                cols = df[col_1].unique()
                if ascending is False:
                    cols = cols[::-1]
                for i in range(len(cols) - 1):
                    temp = []
                    for j in range(len(df)):
                        if df[col_1].loc[j] == cols[i]:
                            temp.append(df[col_2].loc[j])
                        else:
                            temp.append(0)
                    col_name = col_1 + '_' + str(cols[i]) + '_' + col_2
                    df[col_name] = temp

    if categorical is not None:
        df = df.drop(categorical, axis=1)

    return df