#!/usr/bin/env python
# coding: utf-8

# 数据资源下载：链接: https://pan.baidu.com/s/1b7l6NtPbr5bbxzyV-INY0A 提取码: qe8a 

# ### 说明
# 汇集了北京地区餐饮商家的详细数据，综合分析店铺热度、顾客印象、评分、评论等大量信息，经过数据清洗和抽取特征之后，研究热门店铺类型、位置、顾客口碑及关注点，使用对比分析、构成分析以及文本挖掘建模等方法，为新店开设、服务品质升级提供指导意见。
# #### 步骤
# 1）数据准备    
# * 准备好的大众点评商家数据为：“大众点评商家数据.csv”, 截取15家最热门的商家评论数据“大众点评评论数据.xlsx”，可直接使用。  

# 2）创建数据表&数据清洗    
# * 明确各字段含义，初步探索数据类型及缺失值情况，使用pandas对缺失值进行合适的处理；    
# * 将评分数据拆分成“口味”、“环境”和“服务”三类；  

# 3）数据分析    
# * 对大众点评店家数据的分类、评分、价格、评论数等特征进行可视化分析，包括但不限于大小比较，构成情况分析；
# * 以商家级别为分析目标，通过构建多元回归模型，分析影响商家级别的各因素；
# * 通过主成分分析对回归分析结果进行降维；
# * 利用主题模型，对所有的评论数据进行文本分析，探索顾客偏好及口碑；

# ## 1.数据预处理

# #### 1.1 加载数据

# 导入模块
import pandas as pd
import numpy as np
#关闭警告和loging
import logging
import warnings
logging.disable(logging.WARNING)
warnings.filterwarnings('ignore')
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.decomposition import PCA
from snownlp import SnowNLP
import jieba #结巴
import lda
from collections import Counter
#更改目录
df1=pd.read_csv(r'D:\大众点评商家数据1.csv',encoding='utf-8',engine='python')
dianping=pd.read_excel(r'D:\大众点评评论数据.xlsx')
df2=pd.read_csv(r'D:\大众点评商家数据2.csv',encoding='utf-8',engine='python')

df1.info()

df2.info()

# 查看数据
dianping.info()

dianping.columns

dianping.iloc[0]['评论.1']

# 如果存在 Unnamed: 0 数据列则删除该列数据
df1=df1.drop(columns=['Unnamed: 0'])

# 加载商家数据2
dianping['shopname'].unique()

# 如果存在 Unnamed: 22 数据列则删除该列数据
df2=df2.drop(columns=[ 'Unnamed: 22'])

len(df1['商家名(name)'].unique())

# 合并数据表

datamer = pd.merge(df1,df2,on ='商家Id(sid)')

datamer.info()

#data = pd.merge(dianping,df,left_on = 'shopname',right_on ='商家名(name)',how = "right")

datamer.columns

# 对数据进行重命名

datamer.columns = ['爬取链接', '商家ID', '商家名', '省份', '城市',
       '区域', '地址', '纬度', '经度',
       '分类', '电话', '营业时间',
       '商家级别', '商家介绍', '图片', '评价数',
       '评分', '评价标签', '商品', '促销优惠',
       '平均价格']

# 删除经纬的缺失数据

datamer.info()

datamer=datamer.dropna(subset=['经度'])

# 删除商家介绍数据列

datamer=datamer.drop(columns=['商家介绍'])

datamer=datamer.drop(columns=['省份', '城市','区域','评价标签','图片','商品'])
#datamer=datamer.drop(columns=['新分类'])


# ##### 价格缺失数据暂时不做处理

datamer.loc[0]['评分']

labels=['口味','环境','服务']

for label in labels:
    def get_value(param):
        for i in eval(param):
            if i['label']==label:
                return i['value']
    datamer[label] = datamer['评分'].apply(get_value)

# 导出数据

datamer.info()


# ####  处理评论数据

# 处理缺失数据

datamer=datamer.dropna(subset=['口味'])

a = datamer['分类'].tolist()

# 保存处理后的评论
def get_fenlei(param):
    return eval(param)[0]  
datamer['新分类']=datamer['分类'].apply(get_fenlei)

datamer['新分类']
with open(r'D:\BA训练营\20200302大众点评项目\word.txt','w') as f:
    for i in datamer['新分类']:
        f.write(i+',')


# ## 2. 可视化

# #### 2.1 Tableau 可视化

# ### 对店铺数据和评论数据做可视化分析

def get_level(p):
    level = p 
    if level>45:
        return "45-50"
    elif level>40:
        return "40-45"
    elif level>35:
        return "35-40"
    elif level>30:
        return "30-35"
    elif level<=30:
        return "0-30"
    else :
        return "计算出错"

datamer['商家区间'] = datamer['商家级别'].apply(get_level)
datamer.to_excel(r'D:\BA训练营\20200302大众点评项目\全面商家数据.xlsx')


# #### 2.2. 使用wordArt制作词云
# 
# * 店铺类别词云

# ## 3. 挖掘建模

# ## 3.1 构建回归分析模型

# 参考：
#去除价格的缺失数据，以商家级别(50为五星)作为因变量，
# 以'纬度', '经度','评价数','avg_price', '口味', '环境', '服务'作为自变量
# 构建多元回归模型和PCA回归模型（n = 2）


datacle = datamer[datamer['平均价格'].notnull()]

datacle.info()

datacle['口味'] = pd.to_numeric(datacle['口味'],errors='coerce')

datacle['环境'] = pd.to_numeric(datacle['环境'],errors='coerce')

datacle['服务'] = pd.to_numeric(datacle['服务'],errors='coerce')

datacle.columns

datacle.iloc[0]

datacle.to_excel(r'D:\BA训练营\20200302大众点评项目\干净商家数据.xlsx')

Yvar=datacle['商家级别']
Xvar=datacle[['纬度', '经度','评价数','平均价格', '口味', '环境', '服务']]

#热力图绘制

plt.rcParams['font.sans-serif'] = ['SimHei']
t=datacle[['商家级别','纬度', '经度','评价数','平均价格', '口味', '环境', '服务']]
plt.figure(figsize = (10,8))
sns.heatmap(np.abs(t.corr()),annot = True)
plt.show()

std = StandardScaler()
Xstd = std.fit_transform(Xvar)
Y = Yvar.values
X = Xstd
X = sm.add_constant(X)
lm = sm.OLS(Y,X).fit()
print('=======================多元线性回归结果=======================')
print(lm.summary())

#y = 39.5+0.19*x3+0.14*x4+3.02*x5+1.67*x7
#商家级别 = 39.5+0.19*评价数+0.14*平均价格+3.02*口味+1.67*服务

# 回归建模

pca_model = PCA(n_components = 2)#减半
Y = Yvar.values
X = Xstd
pca_model.fit(X)
x_pca = pca_model.transform(X)
x_pca = sm.add_constant(x_pca)
lm = sm.OLS(Y,x_pca).fit()
print('=============================主成分结果=====================')
print(lm.summary())

np.round(pca_model.components_,2)

#y = 39.50+2.74*p1-0.3*p2'纬度', '经度','评价数','平均价格', '口味', '环境', '服务'
#p1=0.01*纬度+0.13*经度+0.25*评价数+0.25*平均价格+0.53*口味+0.54*环境+0.54*服务
#p2=0.74*纬度+0.67*经度-0.05*评价数+0.03*平均价格-0.06*口味-0.05*环境-0.05*服务
#p1=0.01*x1+0.13*x2+0.25*x3+0.25*x4+0.53*x5+0.54*x6+0.54*x7
#p2=0.74*x1+0.67*x2-0.05*x3+0.03*x4-0.06*x5-0.05*x6-0.05*x7


# ### 3.2 构建情感分析和LDA主题模型

# * 对TOP15店铺进行情感得分和LDA主题分析

shops = dianping['shopname'].unique().tolist()
shops

dianping.info()

for i in shops:
    uid=dianping[dianping['shopname']==i]
    print(i,'评论数量：',len(uid))

def emotion(s):
    positive=0
    negative=0
    smooth=0
    for i in s:
        if i>0.6:
            positive+=1
        elif i<0.4:
            negative+=1
        else:
            smooth+=1
    counts=positive+negative+smooth
    print('积极情绪：',str(round(positive/counts*100,0))+'%')
    print('消极情绪：',str(round(negative/counts*100,0))+'%')
    print('平和情绪：',str(round(smooth/counts*100,0))+'%')

# 对TOP15每一家店铺进行分析
for i in shops:
    comments=dianping[dianping['shopname']==i]['评论.1']
    comments.index=np.arange(len(comments))
    index=np.random.randint(1,len(comments),500)
    data=comments.iloc[index]
    a=[np.round(SnowNLP(sen).sentiments,2) for sen in data]
    print(i,'TOP15店铺情感分析结果：')
    emotion(a)

# #### 2.2 LDA主题模型分析


string=open(r'D:\BA训练营\20200302大众点评项目\stopwords.txt','r',encoding='UTF-8').read()
filterwords=string.split('\n')

again = [
 '海底捞火锅(牡丹园店)',
 '满恒记清真涮羊肉']

def word_cut(coms):
    b=[]
    for i in jieba.cut(coms):
        if i not in filterwords:
            b.append(i)
    return b
def get_vector(sentence,vocab):
    temp=[]
    for word in vocab:
        if word in sentence:
            temp.append(1)
        else:
            temp.append(0)
    return temp
def get_lda(params):
    corpora_words=[]
    for i in params:
        ss=word_cut(i)
        corpora_words.append(ss)
    words=[]
    for i in corpora_words:
        words+=i
    word_count=Counter(words)
    vocab=[]
    for word in word_count.keys():
        if word_count[word]>1:
            vocab.append(word)
    X=[]
    for se in corpora_words:
        X.append(get_vector(se,vocab))
    X=np.array(X)
    lda_model=lda.LDA(n_topics=10,n_iter=100,random_state=1)
    lda_model.fit(X)
    topic_word=lda_model.topic_word_
    for i in range(10):
        index=np.argsort(topic_word[i])[::-1]
        print('主题',i,':',end='')
        for j in np.array(vocab)[index][0:10]:
            print(j,end=' ')
        print()

for i in again:
    comments=dianping[dianping['shopname']==i]['评论.1']
    index=np.arange(len(comments))
    data=comments.iloc[index]
    neg=[]
    pos=[]
    mid=[]
    for sen in data:
        s=SnowNLP(sen).sentiments
        if s<0.1:
            neg.append(sen)
        elif s>0.6:
            pos.append(sen)
        else:
            mid.append(sen)
    print(i,'TOP15店铺LDA主题情感分析结果：')
    print(i,'积极的十个主题：')
    get_lda(pos)
    print(i,'消极的十个主题：')
    get_lda(neg)

for i in shops:
    comments=dianping[dianping['shopname']==i]['评论.1']
    index=np.arange(len(comments))
    data=comments.iloc[index]
    neg=[]
    pos=[]
    mid=[]
    for sen in data:
        s=SnowNLP(sen).sentiments
        if s<0.1:
            neg.append(sen)
        elif s>0.6:
            pos.append(sen)
        else:
            mid.append(sen)
    print(i,'TOP15店铺LDA主题情感分析结果：')
    print('积极的十个主题：')
    get_lda(pos)
    print('消极的十个主题：')
    get_lda(neg)
    print('中立的十个主题：')
    get_lda(mid)
