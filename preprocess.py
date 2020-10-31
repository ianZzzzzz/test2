import pandas as pd
import numpy as np
import pickle as pkl
import math
from sklearn.preprocessing import StandardScaler

train_feat= pd.read_csv('train_features.csv', index_col=0) # 每个注册号下各种行为的数量汇总
test_feat= pd.read_csv('test_features.csv', index_col=0) # 
user_profile = pd.read_csv('user_info.csv', index_col='user_id') # user_id,gender,education,birth
courseinfo = pd.read_csv('course_info.csv', index_col='id') # id,course_id,start,end,course_type,category
# pkl.dump(act_feats, open('act_feats.pkl','wb')) # 
all_feat = pd.concat([train_feat, test_feat]) # 



# extract user age
birth_year = user_profile['birth'].to_dict()
def age_convert(y):
    
    if y == None or math.isnan(y):
        return 0
    a = 2018 - int(y)
    if a> 70 or a< 10:
        a = 0
    return a
all_feat['age'] = [age_convert(birth_year.get(int(u),None)) for u in all_feat['username']]

# extract user gender
user_gender = user_profile['gender'].to_dict()
def gender_convert(g):
    if g == 'm':
        return 1
    elif g == 'f':
        return 2
    else:
        return 0


all_feat['gender'] = [gender_convert(user_gender.get(int(u),None)) for u in all_feat['username']]

user_edu = user_profile['education'].to_dict()
def edu_convert(x):
    edus = ["Bachelor's","High", "Master's", "Primary", "Middle","Associate","Doctorate"]
    #if x == None or or math.isnan(x):
    #    return 0
    if not isinstance(x, str):
        return 0
    ii = edus.index(x)
    return ii+1

all_feat['education'] = [edu_convert(user_edu.get(int(u), None)) for u in all_feat['username']]

user_enroll_num = all_feat.groupby('username').count()[['course_id']]
course_enroll_num = all_feat.groupby('course_id').count()[['username']]

user_enroll_num.columns = ['user_enroll_num']
course_enroll_num.columns = ['course_enroll_num']

all_feat = pd.merge(all_feat, user_enroll_num, left_on = 'username', right_index = True)
all_feat = pd.merge(all_feat, course_enroll_num, left_on='course_id', right_index=True)


#extract user cluster
user_cluster_id = pkl.load(open('cluster/user_dict','r')) #
cluster_label = np.load('cluster/label_5_10time.npy') #
all_feat['cluster_label'] = [cluster_label[user_cluster_id[u]] for u in all_feat['username']]


#extract course category
en_categorys = ['math','physics','electrical', 'computer','foreign language', 'business', 'economics','biology','medicine','literature','philosophy','history','social science', 'art','engineering','education','environment','chemistry']

def category_convert(cc):
    if isinstance(cc, str):
        for i, c in zip(range(len(en_categorys)), en_categorys):
            if cc == c:
                return i+1
    else:
        return 0
category_dict = courseinfo['category'].to_dict()

all_feat['course_category'] = [category_convert(category_dict.get(str(x), None)) for x in all_feat['course_id']]

act_feats = [c for c in train_feat.columns if 'count' in c or 'time' in c or 'num' in c] 
# 只提取计数数据的列名 不包含id和统计值列名

pkl.dump(act_feats, open('act_feats.pkl','wb')) # 只写了列名进去 没有数据 

num_feats = act_feats + ['age','course_enroll_num','user_enroll_num']
# 计数数据的列名 + 年龄 + 课程数 + 用户数

scaler= StandardScaler()
newX = scaler.fit_transform(all_feat[num_feats]) # 对train+test归一化 方差为一 均值为零
print(newX.shape)
for i, n_f in enumerate(num_feats):
    all_feat[n_f] = newX[:,i]   

all_feat.loc[train_feat.index].to_csv('train_feat.csv')
all_feat.loc[test_feat.index].to_csv('test_feat.csv')

