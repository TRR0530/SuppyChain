# -*- coding: utf-8 -*-
# @Author : TRR
# @Time : 2022/11/6 11:20

"""数据可视化"""
from datetime import datetime
# load and plot the time series dataset
# from pandas import read_csv
from matplotlib import pyplot as plt


import seaborn as sns
import pandas as pd
import numpy as np


# 统一接口
def frequency_chart(x):
    import numpy as np
    import pandas as pd

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="darkgrid")

    xlabel, = x.columns
    sns.histplot(data=x, x=xlabel, bins=int(np.sqrt(len(x))))
    plt.show()


if __name__ == '__main__':
    import pandas as pd
    from statsmodels.tsa.stattools import adfuller

    def test_stationarity(data, timeseries):

        # Determing rolling statistics
        rolmean = timeseries.rolling(window=12).mean()
        rolstd = timeseries.rolling(window=12).std()

        # Plot rolling statistics:
        orig = plt.plot(timeseries, color='blue', label='Original')
        mean = plt.plot(rolmean, color='red', label='Rolling Mean')
        std = plt.plot(rolstd, color='black', label='Rolling Std')
        plt.legend(loc='best')
        plt.title('Rolling Mean & Standard Deviation')
        plt.xlabel(data)
        plt.show(block=False)

        # Perform Dickey-Fuller test:
        print('Results of Dickey-Fuller Test:')
        dftest = adfuller(timeseries, autolag='AIC')
        dfoutput = pd.Series(dftest[0:4],
                             index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
        for key, value in dftest[4].items():
            dfoutput['Critical Value (%s)' % key] = value
        print(dfoutput)


    """获取销售表中每日总销量"""
    df = pd.read_csv("dataset/goodsale.csv", low_memory=False)
    # print(df.groupby(['data_date'])['goods_num'].sum().reset_index())
    df['data_date'] = pd.to_datetime(df['data_date'], format="%Y%m%d")
    goods = df.groupby(['data_date'], as_index=False)['goods_num'].agg({"sum": np.sum})
    ts = goods['sum']
    test_stationarity(goods['data_date'], ts)

    # df.groupby(['data_date'])['goods_num'].sum().reset_index().to_csv("1.csv", index=False)
    # goods = pd.read_csv("1.csv", dtype={"data_date": object, "goods_num": int})
    # goods['data_date'] = pd.to_datetime(goods['data_date'], format="%Y%m%d")
    # plt.figure(figsize=(12, 5))
    # plt.plot(goods["data_date"], goods["goods_num"])
    # # plt.setp(ylim=(1000, 5000))
    # plt.show()

    """获取销售表中每周总销量"""
    # data = pd.read_csv("dataset/goodsale_modified1.csv", index_col=0, low_memory=False)
    # # data['data_date'] = pd.to_datetime(data['data_date'], format="%Y%m%d")
    # print(data.dtypes)
    #
    # result_columns = data.groupby(['own_week'],
    #                               as_index=False)['goods_num'].agg({"总销量": np.sum})
    # print(result_columns)
    # index = np.arange(result_columns["总销量"].size)
    # print(index)
    # # plt.plot(index, result_columns["总销量"])
    # plt.bar(index, result_columns["总销量"])
    # plt.show()

    """判断表格基本信息"""
    # 行数、字段、缺失值
    # goods_sale
    # goods_sale = pd.read_csv("dataset/goodsale.csv", low_memory=False)
    # print("goods_sale")
    # print("总行数", goods_sale.size)
    # print(goods_sale.info())
    # print("缺失值")
    # print(goods_sale.isnull().sum())
    # # goods_daily
    # goods_daily = pd.read_csv("dataset/goodsdaily.csv", low_memory=False)
    # print("goods_daily")
    # print("总行数", goods_daily.size)
    # print(goods_daily.info())
    # print("缺失值")
    # print(goods_daily.isnull().sum())
    # # goods_info
    # goods_info = pd.read_csv("dataset/goodsinfo.csv", low_memory=False)
    # print("goods_info")
    # print("总行数", goods_info.size)
    # print(goods_info.info())
    # print("缺失值")
    # print(goods_info.isnull().sum())
    # # goods_promote_price
    # goods_promote_price = pd.read_csv("dataset/goods_promote_price.csv", low_memory=False)
    # print("goods_promote_price")
    # print("总行数", goods_promote_price.size)
    # print(goods_promote_price.info())
    # print("缺失值")
    # print(goods_promote_price.isnull().sum())
    # # goods_sku_relation
    # goods_sku_relation = pd.read_csv("dataset/goods_sku_relation.csv", low_memory=False)
    # print("goods_sku_relation")
    # print("总行数", goods_sku_relation.size)
    # print(goods_sku_relation.info())
    # print("缺失值")
    # print(goods_sku_relation.isnull().sum())
    # # marketing
    # marketing = pd.read_csv("dataset/marketing.csv", low_memory=False)
    # print("marketing")
    # print("总行数", marketing.size)
    # print(marketing.info())
    # print("缺失值")
    # print(marketing.isnull().sum())

    """最终预测结果每周的总销售量"""
    # data = pd.read_csv("dataset/result.csv", index_col=0,
    #                    dtype={"sku_id": str, "week1": int, "week2": int, "week3": int, "week4": int, "week5": int})
    # data1 = data["week1"].sum()
    # print(data1)
    # data2 = data["week2"].sum()
    # data3 = data["week3"].sum()
    # data4 = data["week4"].sum()
    # data5 = data["week5"].sum()
    # x = np.array(['week1', 'week2', 'week3', 'week4', 'week5'])
    # y = np.array([data1, data2, data3, data4, data5])
    # plt.bar(x, y, width=0.8)
    # plt.show()

    # 加强版
    # data = pd.read_csv("dataset/submit_example_2.csv", low_memory=False)
    # data1 = data["week1"].sum()
    # print(data1)
    # data2 = data["week2"].sum()
    # data3 = data["week3"].sum()
    # data4 = data["week4"].sum()
    # data5 = data["week5"].sum()
    # x = np.array(['week1', 'week2', 'week3', 'week4', 'week5'])
    # y = np.array([data1, data2, data3, data4, data5])
    # plt.bar(x, y, width=0.8)
    # plt.show()




