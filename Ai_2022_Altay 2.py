#!/usr/bin/python
# -*- coding: cp1251 -*-
# -*- coding: utf-8 -*-



from pickle import FALSE
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
import joblib
import csv

path = 'G:\\Code_K\\AI_projects_all\\Altay_krai\\'

data = []

feature_cols1 = ['ID','���_������','���_�����������','���','���������','���������_����','����_��������',
                '��_���������','���_���������_��','���_���������_��','�������','������_��',
                '������_��','�����_��','���������','�������_������','�������_����',
                '������_��������','����������','����','����������','�������������','���������������'

]



feature_cols = ['ID','���_������','���_�����������','���','���������','���������_����','����_��������',
                '��_���������','���_���������_��','���_���������_��','�������','������_��',
                '������_��','�����_��','���������','�������_������','�������_����',
                '������_��������','����������','����','����������','�������������','���������������','������'

]

with open(path+"test_dataset_test.csv", "r", encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    lineSwap=''
    nullpoint =0

    bugs =0

    for i,line in enumerate(reader):
        if nullpoint==0:
            nullpoint=nullpoint+1
            continue

        dataBin = []
        for beta in line:
            if beta=='': 
                #dataBin.append('00000000')
                dataBin.append('99999')
                #print('00000000++++')

            else:
                #dataBin.append(''.join(format(i, '08b') for i in bytearray(beta, encoding ='utf-8')))
                #print(''.join(format(i, '08b') for i in bytearray(beta, encoding='utf-8'))+' ++++ ')
                dataBin.append(beta)
            #print(count)
       
        data.append(dataBin)

print('BURGLADER ',bugs)

df = pd.DataFrame(data, columns=feature_cols1)
 
df.to_csv(path+"0_test_dataset_test.csv", index=False)

print("===Null_kill===")

data = []
with open(path+"train_dataset_train.csv", "r", encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    lineSwap=''
    nullpoint =0

    bugs =0

    for i,line in enumerate(reader):
        if nullpoint==0:
            nullpoint=nullpoint+1
            continue

        dataBin = []
        for beta in line:
            if beta=='': 
                #dataBin.append('00000000')
                dataBin.append('99999')
                #print('00000000++++')

            else:
                #dataBin.append(''.join(format(i, '08b') for i in bytearray(beta, encoding ='utf-8')))
                #print(''.join(format(i, '08b') for i in bytearray(beta, encoding='utf-8'))+' ++++ ')
                dataBin.append(beta)
            #print(count)
       
        data.append(dataBin)

print('BURGLADER ',bugs)

df = pd.DataFrame(data, columns=feature_cols)
 
df.to_csv(path+"0_train_dataset_train.csv", index=False)

print("===Null_kill===")

data = []
#
path_dir = 'G:\\Code_K\\AI_projects_all\\Altay_krai\\'

columnsT= ['ID','���_������','���_�����������','���','���������','���������_����','����_��������',
                '��_���������','���_���������_��','���_���������_��','�������','������_��',
                '������_��','�����_��','���������','�������_������','�������_����',
                '������_��������','����������','����','����������','�������������','���������������','������','M_1','D_1']

columnsT1= ['ID','���_������','���_�����������','���','���������','���������_����','����_��������',
                '��_���������','���_���������_��','���_���������_��','�������','������_��',
                '������_��','�����_��','���������','�������_������','�������_����',
                '������_��������','����������','����','����������','�������������','���������������','M_1','D_1']
#with open(path_dir+"/train/train.csv", "r") as f:

col3=[]
col4=[]
col5=[]
col6=[]
col7=[]
col8=[]
col11=[]
col12=[]
col13=[]
col17=[]

with open(path_dir+"0_train_dataset_train.csv", "r", encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=",")
    count =0
    for i, line in enumerate(reader):
        if count==0:
            count=count+1
            continue

        if line[3] not in col3:
             col3.append(line[3])

        if line[4] not in col4:
            col4.append(line[4])
            
        if line[5] not in col5:
            col5.append(line[5])
        
        god = line[6]
        str = ""
        for char1 in god:
            if char1=='0' or char1=='1' or char1=='2' or char1=='3' or char1=='4' or char1=='5' or char1=='6' or char1=='7' or char1=='8' or char1=='9':
                str = str + char1
        if str not in col6:
            col6.append(str)
          
        if line[7] not in col7:
            col7.append(line[7]) 

        if line[8] not in col8:
            col8.append(line[8])

        if line[11] not in col11:
            col11.append(line[11])
            
        if line[12] not in col12:
            col12.append(line[12])
            
        if line[13] not in col13:
            col13.append(line[13])
            
        if line[17] not in col17:
            col17.append(line[17])

with open(path_dir+"0_test_dataset_test.csv", "r", encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=",")
    count =0
    for i, line in enumerate(reader):
        if count==0:
            count=count+1
            continue

        if line[3] not in col3:
             col3.append(line[3])

        if line[4] not in col4:
            col4.append(line[4])
            
        if line[5] not in col5:
            col5.append(line[5])

        god = line[6]
        str = ""
        for char1 in god:
            if char1=='0' or char1=='1' or char1=='2' or char1=='3' or char1=='4' or char1=='5' or char1=='6' or char1=='7' or char1=='8' or char1=='9':
                str = str + char1
        if str not in col6:
            col6.append(str)
            
        if line[7] not in col7:
            col7.append(line[7]) 

        if line[8] not in col8:
            col8.append(line[8])

        if line[11] not in col11:
            col11.append(line[11])
            
        if line[12] not in col12:
            col12.append(line[12])
            
        if line[13] not in col13:
            col13.append(line[13])
            
        if line[17] not in col17:
            col17.append(line[17])
         
#print(col3)


with open(path_dir+"0_train_dataset_train.csv", "r", encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=",")
    count =0
    for i, line in enumerate(reader):
        if count==0:
            count=count+1
            continue
        c3 = col3.index(line[3])+1
        c4 = col4.index(line[4])+1
        c5 = col5.index(line[5])+1

        god = line[6]
        str = ""
        for char1 in god:
            if char1=='0' or char1=='1' or char1=='2' or char1=='3' or char1=='4' or char1=='5' or char1=='6' or char1=='7' or char1=='8' or char1=='9':
                str = str + char1

        c6 = str[0]+str[1]+str[2]+str[3]
        M_1 = str[4]+ str[5]
        D_1 = str[6]+ str[7]
        c7 = col7.index(line[7])+1
        c8 = col8.index(line[8])+1
        c11 = col11.index(line[11])+1
        c12 = col12.index(line[12])+1
        c13 = col13.index(line[13])+1
        c17 = col17.index(line[17])+1
        #print(line)
        data.append([line[0],line[1],line[2],c3,
                    c4,c5,c6,
                     c7,c8
                     ,line[9],line[10],c11,c12,
                    c13,line[14],line[15],
                     line[16],c17,line[18],line[19],line[20],line[21],line[22],line[23],M_1,D_1])

df = pd.DataFrame(data, columns=columnsT)
#df.to_csv(path_dir+"/train/Strain.csv")
df.to_csv(path_dir+"Num_train_dataset_train.csv",index=False)

data = []

with open(path_dir+"0_test_dataset_test.csv", "r", encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=",")
    count =0
    for i, line in enumerate(reader):
        if count==0:
            count=count+1
            continue
        c3 = col3.index(line[3])+1
        c4 = col4.index(line[4])+1
        c5 = col5.index(line[5])+1
        
        god = line[6]
        str = ""
        for char1 in god:
            if char1=='0' or char1=='1' or char1=='2' or char1=='3' or char1=='4' or char1=='5' or char1=='6' or char1=='7' or char1=='8' or char1=='9':
                str = str + char1

        c6 = str[0]+str[1]+str[2]+str[3]
        M_1 = str[4]+ str[5]
        D_1 = str[6]+ str[7]
        c7 = col7.index(line[7])+1
        c8 = col8.index(line[8])+1
        c11 = col11.index(line[11])+1
        c12 = col12.index(line[12])+1
        c13 = col13.index(line[13])+1
        c17 = col17.index(line[17])+1
        #print(line)
        data.append([line[0],line[1],line[2],c3,
                    c4,c5,c6,
                     c7,c8
                     ,line[9],line[10],c11,c12,
                    c13,line[14],line[15],
                     line[16],c17,line[18],line[19],line[20],line[21],line[22],M_1,D_1])

df = pd.DataFrame(data, columns=columnsT1)
#df.to_csv(path_dir+"/train/Strain.csv")
df.to_csv(path_dir+"Num_test_dataset_test.csv",index=False)

print("===Num_swap===")

df_train = pd.read_csv(path+"Num_train_dataset_train.csv",encoding = "utf-8")
print(df_train.head())

df_test = pd.read_csv(path+"Num_test_dataset_test.csv",encoding = "utf-8")

print(df_test.head())

'''

df_train.fillna('100001')
df_test.fillna('100001')

frames = [df_train, df_test]
df_hub = pd.concat(frames)

#print(df_hub.head())
#df_hub.to_csv(path+"=horror.csv",index=False)

col3 = list(pd.unique(df_hub['���']))
df_train['���'] = df_train['���'].apply(lambda x: col3.index(x))
df_test['���'] = df_test['���'].apply(lambda x: col3.index(x))

col4 = list(pd.unique(df_hub['���������']))
df_train['���������'] = df_train['���������'].apply(lambda x: col4.index(x))
df_test['���������'] = df_test['���������'].apply(lambda x: col4.index(x))

col5 = list(pd.unique(df_hub['���������_����']))
df_train['���������_����'] = df_train['���������_����'].apply(lambda x: col5.index(x))
df_test['���������_����'] = df_test['���������_����'].apply(lambda x: col5.index(x))

col7 = list(pd.unique(df_hub['��_���������']))
df_train['��_���������'] = df_train['��_���������'].apply(lambda x: col7.index(x))
df_test['��_���������'] = df_test['��_���������'].apply(lambda x: col7.index(x))

col8 = list(pd.unique(df_hub['���_���������_��']))
df_train['���_���������_��'] = df_train['���_���������_��'].apply(lambda x: col8.index(x))
df_test['���_���������_��'] = df_test['���_���������_��'].apply(lambda x: col8.index(x))

col11 = list(pd.unique(df_hub['������_��']))
df_train['������_��'] = df_train['������_��'].apply(lambda x: col11.index(x))
df_test['������_��'] = df_test['������_��'].apply(lambda x: col11.index(x))

col12 = list(pd.unique(df_hub['������_��']))
df_train['������_��'] = df_train['������_��'].apply(lambda x: col12.index(x))
df_test['������_��'] = df_test['������_��'].apply(lambda x: col12.index(x))

col13 = list(pd.unique(df_hub['�����_��']))
df_train['�����_��'] = df_train['�����_��'].apply(lambda x: col13.index(x))
df_test['�����_��'] = df_test['�����_��'].apply(lambda x: col13.index(x))

col17 = list(pd.unique(df_hub['������_��������']))
df_train['������_��������'] = df_train['������_��������'].apply(lambda x: col17.index(x))
df_test['������_��������'] = df_test['������_��������'].apply(lambda x: col17.index(x))
'''
#
'''
df_train = df_train.drop(['���������'],axis=1)
df_train = df_train.drop(['�������_������'],axis=1)
df_train = df_train.drop(['�������_����'],axis=1)
df_train = df_train.drop(['����������'],axis=1)
df_train = df_train.drop(['����'],axis=1)
df_train = df_train.drop(['����������'],axis=1)
df_train = df_train.drop(['���'],axis=1)
df_train = df_train.drop(['������_��'],axis=1)
df_train = df_train.drop(['���������_����'],axis=1)
df_train = df_train.drop(['������_��'],axis=1)
df_train = df_train.drop(['�������'],axis=1)
#print(df_train.head())
df_test = df_test.drop(['���������'],axis=1)
df_test = df_test.drop(['�������_������'],axis=1)
df_test = df_test.drop(['�������_����'],axis=1)
df_test = df_test.drop(['����������'],axis=1)
df_test = df_test.drop(['����'],axis=1)
df_test = df_test.drop(['����������'],axis=1)
df_test = df_test.drop(['���'],axis=1)
df_test = df_test.drop(['������_��'],axis=1)
df_test = df_test.drop(['���������_����'],axis=1)
df_test = df_test.drop(['������_��'],axis=1)
df_test = df_test.drop(['�������'],axis=1)
'''

print(df_train.info())
print('\n')
print(df_test.info())


X = df_train.drop('������',axis=1).values
y = df_train['������'].values


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=4)

from sklearn.ensemble import RandomForestClassifier

#0.76
#11

'''
model=RandomForestClassifier(n_estimators=600,
               max_features = 50,
              max_depth = 52,
               min_samples_split=2,
               min_samples_leaf=3,
               random_state=4
               )
'''

import lightgbm as lgb

model = lgb.LGBMClassifier(
   max_depth=-1,
    learning_rate=0.01,
    n_estimators=10000, # How many weak classifiers to use
    objective='multiclass',
    num_class=3,
    boosting_type='dart',
    min_child_weight=10,
    subsample=1,
    colsample_bytree=1,
    reg_alpha=1,
    reg_lambda=1,
    seed=0
)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
print('\n')
print(model.score(X_test,y_test))
print('\n')


predatel1 = model.predict(df_test)

predictions = pd.DataFrame(predatel1)
predictions = predictions.rename(columns={0:"������"})

#print(predictions.info())
print('\n')
#print(df_test.info())
print('\n')

df_test = pd.read_csv(path+"test_dataset_test.csv",encoding = "utf-8")
submission = pd.concat([df_test["ID"],predictions],axis=1)
print(submission.info())
submission.to_csv(path+"1_submission.csv",index=False)

print('=======END========')
