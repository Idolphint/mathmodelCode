# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 16:07:44 2020

@author: litia
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
PLTDIR = "pltdir1/"

def sigmoid(x):
    s = 5 / (1+np.exp(-x))
    #=10/(1+EXP(-I1589/5)) - 4  in excel
    return s

def get_date_list(min_date, max_date):
    max_y, max_m = max_date.split("-")
    min_y, min_m = min_date.split("-")
    m = int(min_m)
    d_list = []
    for y in range(int(min_y), int(max_y)):
        while m <= 1:
            d_list.append(str(y)+"-"+str(m).zfill(2))
            m+=1
        m=0
    while m<=int(max_m):
        d_list.append(str(max_y)+"-"+str(m).zfill(2))
        m+=1
    return d_list

def date_pro(ori_data):
    
    max_date = "1000-00"
    min_date = "9999-99"
    i=0
    for item in ori_data:
        
        m,d, y = item.split("/")
        m = int((int(m) - 1)/6)
        new_d = y+"-"+str(m).zfill(2)
        ori_data[i] = new_d
        if new_d > max_date:
            max_date = new_d
        if new_d < min_date:
            min_date = new_d
        i+=1
    #print(min_date, max_date)
    date_list = get_date_list(min_date, max_date)
    #print(max_date, min_date, date_list)
    return ori_data, date_list
        
def get_part(info, m):
    part = []
    for item in info:
        if item[2] == m:
            part.append(item)
    
    return np.array(part)


def add_data(sales, sen_score, star, final_info):
    l = sales.shape[0]
    print(l)
    for i in range(2,l):
        final_info.append({'sales_1now': sales[i], 'sales_before2':sales[i-2],
                           'sales_before1': sales[i-1], 'sen_score':
            sen_score[i-2], 'star':star[i-2]})
    return final_info


def get_data():
    hair = pd.read_csv("Problem_C_Data\microutf8.csv", sep=',', header=0)
    #hair = hair.to_dict(orient='records')
    
    #获得字典？？根据产品不同而分类，要求的是产品销量与星级以及评价得分之间的关系
    product = list(set(hair['product_id']))
    #print(product)
    final_info = []
    print(hair.shape)
    for item in product:
        data4item = hair[hair['product_id']==item]
        if (data4item.shape[0]<50):
            continue
        #print(data4item.shape)
        star_rating = data4item['star_rating'].tolist()
        vote_multi = data4item['vote_multi'].tolist()
        review_date = data4item['review_date'].tolist()
        positive_prob = data4item['positive_prob'].tolist()
        
        review_date, mon_list = date_pro(review_date)
        star = np.zeros_like(mon_list)
        sen_score = np.zeros_like(mon_list)
        sales = np.zeros_like(mon_list)
        info = np.array([star_rating, vote_multi, review_date, positive_prob])
        info = info.transpose(1,0)
        print(info.shape)
    
#        info = {'star_rating': data4item['star_rating'],
#                                       'vote_multi': data4item['vote_multi'],
#                                       'review_date':data4item['review_date'], 
#                                       'positive_prob':data4item['positive_prob']}
        
        #info = np.array(info)
        #print(info)
        #break
        #获得这个日期下，产品销售量，星级平均值，情感得分，对这些内容计算权重，根据方程式拟合
        cnt=0
        
        for m in mon_list:
            #part = (lambda x: x[2]==m, info[:, ])
            part = get_part(info, m)
            
            sale_m = part.shape[0]
            if sale_m <= 0:
                star[cnt] = 0
                sen_score[cnt] = 0
                sales[cnt] = 0
                cnt+=1
                continue
            part = part.transpose(1,0)
            part_np = np.array([part[0], part[1], part[3]]).astype(np.float64)
            
            star_m = np.mean(part_np[0])
            score_m = np.mean(part_np[1]*part_np[2])
            star[cnt] = star_m
            sen_score[cnt] = score_m
            sales[cnt] = sale_m
            cnt+=1
            
        sales = sales.astype(np.float64)
        sen_score = sen_score.astype(np.float64)
        star = star.astype(np.float64)
        
        final_info = add_data(sales, sen_score, star, final_info)
#        plt.figure()
#        plt.subplot(311)
#        plt.plot(mon_list, sales)
#        plt.title("sales")
#        
#        plt.subplot(312)
#        plt.plot(mon_list, star)
#        plt.title("star")
#        plt.subplot(313)
#        plt.plot(mon_list, sen_score)
#        plt.title("sen_score")
#        plt.savefig(os.path.join(PLTDIR, item))
#        plt.cla()
#        plt.close("all")
#        print(item)
    df = pd.DataFrame(final_info)
    rDf = df.corr()
    print(df.iloc[1])
    X_train, X_test, Y_train, Y_test = train_test_split(df.iloc[:, 1:], df['sales_1now'], train_size=.8)
    print(X_train.shape, X_test.shape)
    print(Y_train.shape, Y_test.shape)
    
    ################begin train
    model = LinearRegression()
    model.fit(X_train, Y_train)
    a = model.intercept_
    b = model.coef_
    print("最佳拟合线:截距",a,",回归系数：",b)
    
    score = model.score(X_test, Y_test)
    print(score)
    Y_pred = model.predict(X_test)
    print(Y_pred)
    plt.plot(range(len(Y_pred)), Y_pred,'r', label='predict')
    plt.plot( range(len(Y_pred)), Y_test, 'b', label='ori')
    plt.legend()
    plt.ylabel("sales")
    plt.savefig("predict.jpg")
    plt.show()
    
        

if __name__ == "__main__":
    get_data()
    