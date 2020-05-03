import numpy as np
import pandas as pd

def design_matrix(df, columns = None, intercept = True, category = None, interaction = None, ascending = True):
    if columns is not None:
        df = df[columns]
    df = df.copy()
    if intercept is True:
        df['Intercept'] = np.ones(len(df))
        columns = df.columns.tolist()
        columns = [columns[-1]] + columns[:-1]
        df = df[columns]

    if category is not None:
        for col in category:
            if ascending is True:
                df1 = pd.get_dummies(df[col].sort_values(), prefix=col)
                df1 = df1.drop(df1.columns[len(df1.columns) - 1], axis=1)
            else:
                df1 = pd.get_dummies(df[col].sort_values(ascending = False), prefix=col, drop_first = True)
            df = pd.merge(df, df1, left_index = True, right_index = True)

    if interaction is not None:
        for inter in interaction:
            if inter[0] in category and inter[1] in category:
                col_1 = df[inter[0]].unique().tolist()
                col_1.sort()
                col_2 = df[inter[1]].unique().tolist()
                col_2.sort()
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
                if inter[0] in category and inter[1] not in category:
                    col_1 = inter[0]
                    col_2 = inter[1]
                else:
                    col_1 = inter[1]
                    col_2 = inter[0]
                cols = df[col_1].unique().tolist()
                cols.sort()
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

    if category is not None:
        df = df.drop(category, axis=1)

    return df

def