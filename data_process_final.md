# 缺失特征补全
# 比对两级特征处理
#### feat_extract.py
    对流水账数据基于注册号的汇总
#### preprocess.py
    拼接汇总数据和用户及课程数据
    数据归一化
    from sklearn.preprocessing import StandardScaler
#### main.py
    计算行为计数值的统计值
#### model.py
    在 _init_graph中建立网络的数据流图
    似乎只对a做bn操作
# 训练中使用的dfTrain_a 行为特征===用户行为简单汇总数据
