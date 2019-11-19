#!/usr/bin/env python
import csv
import math
import numpy as np
import pandas as pd
#import xgboost as xgb
from scipy import optimize
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from  sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame


def data_read(path,test_path):
    mixed = (50,53,54,55,56,255,256,257,258,260,268,376)

    data_all = pd.read_csv(path)
    data_test_all = pd.read_csv(test_path)

    data_all.replace([np.inf, -np.inf], np.nan)
    data_test_all.replace([np.inf, -np.inf], np.nan)


    train_len = data_all.shape[0]
    data_all = pd.concat([data_all,data_test_all])


    attri = data_all.columns.values.tolist()
    result = np.zeros(shape=(data_all.shape[0],1))


    label = []
    attri_record = []


    current = -1
    for item in attri:
        current = current+1
        if current in mixed:
            continue
        if item=="job_performance":
            label = list(data_all[item])
            continue


        if not data_all[item].isnull().any():#not data_all[item].isnull().any()
            if (data_all[item].dtype == "object"):
                tmp = OneHotEncoder(sparse=False).fit_transform(np.array(list(data_all[item])).reshape(-1, 1))
                result = np.hstack((result, tmp))
                attri_record.append(item)
            else:
                #data_all[item] = data_all[item].astype(np.int)
                result = np.hstack((result, np.array(data_all[item]).reshape(-1,1) ))
                attri_record.append(item)

        else:
            num = data_all[item].isna().sum()
            #uni = len(data_all[item].unique())

            if num>(data_all.shape[0]) * 0.35:
                continue
            else:
                if (data_all[item].dtype == "object"):
                    data_all[item] = data_all[item].fillna("missing")
                    tmp = OneHotEncoder(sparse=False).fit_transform(np.array(list(data_all[item])).reshape(-1, 1))
                    result = np.hstack((result, tmp))
                    attri_record.append(item)
                else:
                    #data_all[item] = data_all[item].astype(np.int)
                    average = data_all[item].mean()
                    data_all[item] = data_all[item].fillna(average)
                    #data_all[item] = data_all[item].astype(np.int)

                    result = np.hstack((result, np.array(data_all[item]).reshape(-1,1) ))
                    attri_record.append(item)



    print(result.shape)
    print(len(attri_record))
    print(attri_record)

    return result,label,train_len




def mse(gen,bench):
    num = 0

    x = []
    y = []
    for i in range(len(gen)):
        num = num+(gen[i]-bench[i])**2
        x.append(i)
        y.append(abs(gen[i]-bench[i]))

    plt.figure()
    plt.plot(x,y)
    plt.savefig("easyplot.png")

    return num/len(gen)




if __name__ == '__main__':

    feature,label,train_len = data_read("hw4-trainingset-yf2502.csv","hw4-testset-yf2502.csv")



    #print(feature)
    #print(len(feature))
    print(train_len)

    print("___COME___OUT___")

    feature_train,feature_valid,label_train,label_valid = train_test_split(feature[0:train_len],label[0:train_len],test_size=0.2,random_state=0)


    estimator = RandomForestRegressor(n_estimators=1000, oob_score = True, random_state = 42)

    estimator.fit(feature_train, label_train)
    predicted_forest = estimator.predict(feature_valid)
    print(predicted_forest)
    print("forest")
    print(mse(predicted_forest, label_valid))
    #98577.90046009945  n_estimator=160
    #97875              n_estimator=1000
    generate = list(estimator.predict(feature[train_len:]))

    data = pd.read_csv("hw4-testset-yf2502.csv")
    # 任意的多组列表

    data["job_performance"] = generate

    # 将DataFrame存储为csv,index表示是否显示行名，default=True
    data.to_csv("tmp1.csv", index=False, sep=',')



    '''
    model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=1000, silent=False, objective='reg:gamma')
    model.fit(feature_train, np.array(label_train))
    whatget = list(model.predict(feature_valid))
    print(whatget)
    print("xgboost")
    print(mse(whatget,label_valid))

    realget = list(model.predict(feature[train_len:]))

    data = pd.read_csv("hw4-testset-yf2502.csv")
    # 任意的多组列表

    data["job_performance"] = realget

    # 将DataFrame存储为csv,index表示是否显示行名，default=True
    data.to_csv("tmp2.csv", index=False, sep=',')
    '''




























