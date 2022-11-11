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

# sns.set_theme(style="darkgrid")
# df = pd.read_csv("dataset/goodsale.csv",index_col = 0)
#
# sns.displot(
#     df, x="flipper_length_mm", col="species", row="sex",
#     binwidth=3, height=3, facet_kws=dict(margin_titles=True),
# )
# 统一接口
def frequency_chart(x: list):
    import numpy as np
    import pandas as pd
    import seaborn as sns

    sns.set_theme(style="darkgrid")

    xlabel, = x.columns
    sns.histplot(data=x, x=xlabel, bins=int(np.sqrt(len(x))))


if __name__ == '__main__':
    import pandas as pd

    df = pd.read_csv("dataset/goodsale.csv", index_col=0, dtype={"data_date": datetime, "goods_id": str, "goods_price": str, "orginal_shop_price": str, "goods_num": int})
    print(df.groupby(['data_date'])['goods_num'].sum().reset_index())
    df.groupby(['data_date'])['goods_num'].sum().reset_index().to_csv("1.csv", index=False)
    goods = pd.read_csv("1.csv", dtype={"data_date": object, "goods_num": int})
    plt.figure(figsize=(12, 5))
    plt.plot(goods["data_date"], goods["goods_num"])
    # plt.setp(ylim=(1000, 5000))
    plt.show()

    # frequency_chart(goods[["data_date"]])
