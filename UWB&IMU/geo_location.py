#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   geo_location.py    
@Contact :   1094404954@qq.com
@License :   (C)Copyright Cheng Hoiyuen

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2023/5/2 17:07   郑海远      1.0         None
'''

# import lib
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd



with open(r"../data/2023_4_26/corrected222.csv") as file:
    df = pd.read_csv(file)
    print(df.head())
    x= []
    y =[]
    for i in range(df.shape[0]):
        g = 0.6
        a = df['l'][i]
        b = df['r'][i]
        print(a)
        try:
            alpha = math.acos((g ** 2 + b ** 2 - a ** 2) / abs(2 * g * b))
        except Exception as e:
            print(e)
        d1 = b*math.sin(alpha)
        d2 = b*math.cos(alpha)
        x.append(g/2-d2)
        y.append(d1)
    pd.DataFrame({'x':x,'y':y}).to_csv(r"../data/2023_4_26/corrected_uwb2.csv")