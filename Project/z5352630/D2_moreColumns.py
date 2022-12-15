# %%
# Read files
import os
import pandas as pd

#calculation duration
from datetime import datetime

#clean_operating_system
import re

#one-hot
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import numpy as np
from scipy.sparse import hstack

#catboost
import catboost as cbt
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
import gc
from sklearn.model_selection import cross_validate

#DT test dataset
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

#TestModels
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.tree
import sklearn.linear_model
import sklearn.neural_network
import sklearn.metrics
import sklearn.discriminant_analysis
import sklearn.naive_bayes
import sklearn.neighbors
import sklearn.svm
import matplotlib.pyplot as plt


# %%
os.chdir("/data/team15/data")
ebb_1 = pd.read_csv("ebb_set1.csv")
ebb_2 = pd.read_csv("ebb_set2.csv")
eval_set = pd.read_csv("eval_set.csv")
### activations_ebb_set1.csv#####
ebb_1_act = pd.read_csv("activations_ebb_set1.csv")
ebb_2_act = pd.read_csv("activations_ebb_set2.csv")
eval_act = pd.read_csv("activations_eval_set.csv")

def count_act_times(df):
    counts = df['customer_id'].value_counts(sort=False).rename_axis('customer_id').reset_index(name='activation_counts')
    df = counts.drop_duplicates(subset=['customer_id'])
    return df

ebb_1_act = count_act_times(ebb_1_act)
ebb_2_act = count_act_times(ebb_2_act)
eval_act = count_act_times(eval_act)

ebb_1_o_act = ebb_1.merge(ebb_1_act, how='left', on='customer_id')
ebb_2_o_act = ebb_2.merge(ebb_2_act, how='left', on='customer_id')
eval_o_act = eval_set.merge(eval_act, how='left', on='customer_id')


### deactivations_ebb_set#####
ebb_1_deact = pd.read_csv("deactivations_ebb_set1.csv")
ebb_2_deact = pd.read_csv("deactivations_ebb_set2.csv")
eval_deact = pd.read_csv("deactivations_eval_set.csv")

def count_times(df):
    counts = df['customer_id'].value_counts(sort=False).rename_axis('customer_id').reset_index(name='deactivation_counts')
    df = counts.drop_duplicates(subset=['customer_id'])
    return df

ebb_1_deact = count_times(ebb_1_deact)
ebb_2_deact = count_times(ebb_2_deact)
eval_deact = count_times(eval_deact)

ebb_1_act_de = ebb_1_o_act.merge(ebb_1_deact, how='left', on='customer_id')
ebb_2_act_de = ebb_2_o_act.merge(ebb_2_deact, how='left', on='customer_id')
eval_act_de_ = eval_o_act.merge(eval_deact, how='left', on='customer_id')

########interactions_ebb_set1.csv############
ebb_1_inter = pd.read_csv("interactions_ebb_set1.csv")
ebb_2_inter = pd.read_csv("interactions_ebb_set2.csv")
eval_inter = pd.read_csv("interactions_eval_set.csv")

def count_times_inter(df):
    counts = df['customer_id'].value_counts(sort=False).rename_axis('customer_id').reset_index(name='interection_counts')
    df = counts.drop_duplicates(subset=['customer_id'])
    return df
ebb_1_inter = count_times_inter(ebb_1_inter)
ebb_2_inter = count_times_inter(ebb_2_inter)
eval_inter = count_times_inter(eval_inter)


ebb_1_adi = ebb_1_act_de.merge(ebb_1_inter, how='left', on='customer_id')
ebb_2_adi = ebb_2_act_de.merge(ebb_2_inter, how='left', on='customer_id')
eval_adi = eval_act_de_.merge(eval_inter, how='left', on='customer_id')



########ivr_calls_ebb_set1############
ebb_1_ival = pd.read_csv("ivr_calls_ebb_set1.csv")
ebb_2_ival = pd.read_csv("ivr_calls_ebb_set2.csv")
eval_ival = pd.read_csv("ivr_calls_eval_set.csv")

def count_times_ival(df):
    #how many calls
    counts = df['customer_id'].value_counts(sort=False).rename_axis('customer_id').reset_index(name='ival_counts')
    #iscompleted ->mean value
    df['iscompleted'] = df['iscompleted'].replace(0, -1)
    df['iscompleted'] = df['iscompleted'].fillna(0)
    df['iscompleted'] = df['iscompleted'].astype(int)
    df_kb = df['iscompleted'].groupby(df['customer_id'],sort = False).mean()
    df_kb = pd.DataFrame(df_kb).reset_index()  
    df_kb['iscompleted'][df_kb.iscompleted>0] = 1
    df_kb['iscompleted'][df_kb.iscompleted<0] = -1
    df = pd.merge(counts,df_kb)
    
    return df

ebb_1_ival = count_times_ival(ebb_1_ival)
ebb_2_ival = count_times_ival(ebb_2_ival)
eval_ival = count_times_ival(eval_ival)


ebb_1_adii = ebb_1_adi.merge(ebb_1_ival, how='left', on='customer_id')
ebb_2_adii = ebb_2_adi.merge(ebb_2_ival, how='left', on='customer_id')
eval_adii = eval_adi.merge(eval_ival, how='left', on='customer_id')

##############network_ebb_set1.csv#################
ebb_1_net = pd.read_csv("network_ebb_set1.csv")
ebb_2_net = pd.read_csv("network_ebb_set2.csv")
eval_net = pd.read_csv("network_eval_set.csv")

def convert_network(df):
    df_ka = df['total_kb'].groupby(df['customer_id'],sort = False).sum()
    df_ka = pd.DataFrame(df_ka).reset_index() 
    df_kb = df['voice_minutes'].groupby(df['customer_id'],sort = False).sum()
    df_kb = pd.DataFrame(df_kb).reset_index() 
    
    df_ka = df_ka.merge(df_kb, how='left', on='customer_id')
    
    df_kc = df['total_sms'].groupby(df['customer_id'],sort = False).sum()
    df_kc = pd.DataFrame(df_kc).reset_index() 
    
    df = df_ka.merge(df_kc, how='left', on='customer_id')
    return df
ebb_1_net = convert_network(ebb_1_net)
ebb_2_net = convert_network(ebb_2_net)
eval_net = convert_network(eval_net)

ebb_1_adiin = ebb_1_adii.merge(ebb_1_net, how='left', on='customer_id')
ebb_2_adiin = ebb_2_adii.merge(ebb_2_net, how='left', on='customer_id')
eval_adiin = eval_adii.merge(eval_net, how='left', on='customer_id')

#############notifying_ebb_set1#####################
ebb_1_not = pd.read_csv("notifying_ebb_set1.csv")
ebb_2_not = pd.read_csv("notifying_ebb_set2.csv")
eval_not = pd.read_csv("notifying_eval_set.csv")

def convert_notify_date(df):
    #how many notifiactions
    counts = df['customer_id'].value_counts(sort=False).rename_axis('customer_id').reset_index(name='notify_counts')
    #mean of date
    today_str = '2022-4-15'
    today = datetime.strptime(today_str, "%Y-%m-%d")
    df['notify_date']=pd.to_datetime(df['notify_date'],format='%Y %m %d')
    df['notify_date'] = today-df['notify_date']
    df['notify_date'] = df['notify_date'].astype('timedelta64[D]')
    df['notify_date'] = df['notify_date'].astype(int)
    df_kb = df['notify_date'].groupby(df['customer_id'],sort = False).mean() 
    df_kb = pd.DataFrame(df_kb).reset_index() 
    df_kb['notify_date'] = df_kb['notify_date'].astype(int)
    #df = pd.merge(counts,df_kb)
    df = counts.merge(df_kb, how='left', on='customer_id')

    return df
              
ebb_1_not = convert_notify_date(ebb_1_not)
ebb_2_not = convert_notify_date(ebb_2_not)
eval_not = convert_notify_date(eval_not)


ebb_1_adiinn = ebb_1_adiin.merge(ebb_1_not, how='left', on='customer_id')
ebb_2_adiinn = ebb_2_adiin.merge(ebb_2_not, how='left', on='customer_id')
eval_adiinn = eval_adiin.merge(eval_not, how='left', on='customer_id')

##############phone_data_ebb_set1##################
ebb_1_pho = pd.read_csv("phone_data_ebb_set1.csv")
ebb_2_pho = pd.read_csv("phone_data_ebb_set2.csv")
eval_pho = pd.read_csv("phone_data_eval_set.csv")


def convert_pho(df):
    #battery_ava
    df['battery_available'][df.battery_available<0] = 0
    df_ka = df['battery_available'].groupby(df['customer_id'],sort = False).mean()
    df_ka = pd.DataFrame(df_ka).reset_index() 
    df_ka['battery_available'] = df_ka['battery_available'].astype(int)
    
    #data_roaming
    df['data_roaming'] = df['data_roaming'].replace([True,False], [1,-1])
    df['data_roaming'] = df['data_roaming'].fillna(0)
    df_kb = df['data_roaming'].groupby(df['customer_id'],sort = False).mean()
    df_kb = pd.DataFrame(df_kb).reset_index()  
    df_kb['data_roaming'][df_kb.data_roaming>0] = 1
    df_kb['data_roaming'][df_kb.data_roaming<0] = -1
    df_kb['data_roaming'] = df_kb['data_roaming'].astype(int)
    
    #df_1 = pd.merge(df_ka,df_kb)
    df_1 = df_ka.merge(df_kb, how='left', on='customer_id')
    #memory_ava
    df_kc = df['memory_available'].groupby(df['customer_id'],sort = False).mean()
    df_kc = pd.DataFrame(df_kc).reset_index() 
    df_kc['memory_available'] = df_kc['memory_available'].fillna(0)
    df_kc['memory_available'] = df_kc['memory_available'].astype(int)
    
    #df_2 = pd.merge(df_1,df_kc)
    df_2 = df_1.merge(df_kc, how='left', on='customer_id')
    #temperature
    df_kd = df['temperature'].groupby(df['customer_id'],sort = False).mean()
    df_kd = pd.DataFrame(df_kd).reset_index() 
    df_kd['temperature'] = df_kd['temperature'].fillna(0)
    df_kd['temperature'] = df_kd['temperature'].astype(int)
    
    df = df_2.merge(df_kd, how='left', on='customer_id')

    return df
              
ebb_1_pho = convert_pho(ebb_1_pho)
ebb_2_pho = convert_pho(ebb_2_pho)
eval_pho = convert_pho(eval_pho)

ebb_1_adiinnp = ebb_1_adiinn.merge(ebb_1_pho, how='left', on='customer_id')
ebb_2_adiinnp = ebb_2_adiinn.merge(ebb_2_pho, how='left', on='customer_id')
eval_adiinnp = eval_adiinn.merge(eval_pho, how='left', on='customer_id')

############reactivations_ebb_set1##################
ebb_1_rea = pd.read_csv("reactivations_ebb_set1.csv")
ebb_2_rea = pd.read_csv("reactivations_ebb_set2.csv")
eval_rea = pd.read_csv("reactivations_eval_set.csv")


def count_times_rea(df):
    counts = df['customer_id'].value_counts(sort=False).rename_axis('customer_id').reset_index(name='reactivitaion_counts')
    df = counts.drop_duplicates(subset=['customer_id'])
    return df
ebb_1_rea = count_times_rea(ebb_1_rea)
ebb_2_rea = count_times_rea(ebb_2_rea)
eval_rea = count_times_rea(eval_rea)


ebb_1_adiinnpr = ebb_1_adiinnp.merge(ebb_1_rea, how='left', on='customer_id')
ebb_2_adiinnpr = ebb_2_adiinnp.merge(ebb_2_rea, how='left', on='customer_id')
eval_adiinnpr = eval_adiinnp.merge(eval_rea, how='left', on='customer_id')
##############redemptions_ebb_set1###################
ebb_1_red = pd.read_csv("redemptions_ebb_set1.csv")
ebb_2_red = pd.read_csv("redemptions_ebb_set2.csv")
eval_red = pd.read_csv("redemptions_eval_set.csv")


def convert_red_date(df):
    #how many redemption
    counts = df['customer_id'].value_counts(sort=False).rename_axis('customer_id').reset_index(name='redenption_counts')
    
    today_str = '2022-4-15'
    today = datetime.strptime(today_str, "%Y-%m-%d")
    df['date']= pd.to_datetime(df['date'],format='%Y %m %d')
    df['date'] = today-df['date']
    df['date'] = df['date'].astype('timedelta64[D]')
    df['date'] = df['date'].astype(int)
    df_kb= df['date'].groupby(df['customer_id'],sort = False).mean()
    df_kb = pd.DataFrame(df_kb).reset_index()
    df_kb['date'] = df_kb['date'].astype(int)
    #df = df_kb.join(df, how='right')
    df = counts.merge(df_kb, how='left', on='customer_id')
    return df
              
ebb_1_red = convert_red_date(ebb_1_red)
ebb_2_red = convert_red_date(ebb_2_red)
eval_red = convert_red_date(eval_red)


ebb_1_adiinnprr = ebb_1_adiinnpr.merge(ebb_1_red, how='left', on='customer_id')
ebb_2_adiinnprr = ebb_2_adiinnpr.merge(ebb_2_red, how='left', on='customer_id')
eval_adiinnprr = eval_adiinnpr.merge(eval_red, how='left', on='customer_id')
###############throttling_ebb_set1####################
ebb_1_thr = pd.read_csv("throttling_ebb_set1.csv")
ebb_2_thr = pd.read_csv("throttling_ebb_set2.csv")
eval_thr = pd.read_csv("throttling_eval_set.csv")


def count_times_thr(df):
    #how many redemption
    counts = df['customer_id'].value_counts(sort=False).rename_axis('customer_id').reset_index(name='throttled_counts')
    
    today_str = '2022-4-15'
    today = datetime.strptime(today_str, "%Y-%m-%d")
    df['throttled_date']= pd.to_datetime(df['throttled_date'],format='%Y %m %d')
    df['throttled_date'] = today-df['throttled_date']
    df['throttled_date'] = df['throttled_date'].astype('timedelta64[D]')
    df['throttled_date'] = df['throttled_date'].astype(int)
    df_kb= df['throttled_date'].groupby(df['customer_id'],sort = False).mean()
    df_kb = pd.DataFrame(df_kb).reset_index()
    df_kb['throttled_date'] = df_kb['throttled_date'].astype(int)

    df = counts.merge(df_kb, how='left', on='customer_id')
    return df

ebb_1_thr = count_times_thr(ebb_1_thr)
ebb_2_thr = count_times_thr(ebb_2_thr)
eval_thr = count_times_thr(eval_thr)


ebb_1_adiinnprrt = ebb_1_adiinnprr.merge(ebb_1_thr, how='left', on='customer_id')
ebb_2_adiinnprrt = ebb_2_adiinnprr.merge(ebb_2_thr, how='left', on='customer_id')
eval_adiinnprrt = eval_adiinnprr.merge(eval_thr, how='left', on='customer_id')
###################fianl ebb_1############################
ebb_1 = ebb_1_adiinnprrt 
ebb_2 = ebb_2_adiinnprrt 
eval_set = eval_adiinnprrt 
#########################################same as the origianl dataset#############################################
def convert_date(df):
    today_str = '2022-4-15'
    today = datetime.strptime(today_str, "%Y-%m-%d")
    df['last_redemption']=pd.to_datetime(df['last_redemption_date'],format='%Y %m %d')
    df['first_activation']=pd.to_datetime(df['first_activation_date'],format='%Y %m %d')
    df['redemption_duaration'] = df['last_redemption'] - df['first_activation']
    df['last_redemption'] = today-df['last_redemption']
    df['first_activation'] = today-df['first_activation']
    df['last_redemption'] = df['last_redemption'].astype('timedelta64[D]')
    df['first_activation'] = df['first_activation'].astype('timedelta64[D]')
    df['redemption_duaration'] = df['redemption_duaration'].astype('timedelta64[D]')
    return df

convert_date(ebb_1)
convert_date(ebb_2)
convert_date(eval_set)

def clean_operating_system(x):
    if pd.isna(x):
        return 'Not Known'
    t=x.lower()
    if re.search(r'\bandroid\b', t):
        return 'ANDROID'
    elif re.search(r'\bios\b', t):
        return 'IOS'
    elif t.find('not known')!=-1:
        return 'Not Known'
    return 'Other'

ebb_1['operating_system']=ebb_1['operating_system'].apply(clean_operating_system)
ebb_2['operating_system']=ebb_2['operating_system'].apply(clean_operating_system)
eval_set['operating_system']=eval_set['operating_system'].apply(clean_operating_system)
df=pd.concat([ebb_1,ebb_2])
df.info()




# %%
df['ebb_eligible']=df['ebb_eligible'].fillna(0)
df['ebb_eligible']=df['ebb_eligible'].astype(int)
dfy = df[['ebb_eligible']]
dfy.value_counts()

# %%
######################## append the column to the dataset if the sum of its items is bigger than 45000 ############################

#after convert_date, clean_operating_system
def convert_dataset_no_one_hot_more(df, isEval=False):
    df['state'] = df['state'].replace([None],'Not Known') 
    df['state'] = df['state'].astype(str)
    df['operating_system'] = df['operating_system'].astype(str)
    
    df = df.fillna('0') 
    
    columns=['total_redemptions','tenure','number_upgrades','year','total_revenues_bucket',\
             'state','last_redemption','first_activation','redemption_duaration',\
             'operating_system','opt_out_mobiles_ads','deactivation_counts',\
             'interection_counts','ival_counts','iscompleted','total_kb','voice_minutes',\
             'total_sms','reactivitaion_counts','redenption_counts','date',\
             'notify_counts','notify_date','battery_available','data_roaming','memory_available',\
             'temperature','throttled_counts','throttled_date']
    if isEval:
        columns.insert(0, 'customer_id')
        
    df['opt_out_mobiles_ads'] = df['opt_out_mobiles_ads'].astype(int)
    df['last_redemption'] = df['last_redemption'].astype(int)
    df['first_activation'] = df['first_activation'].astype(int)
    df['redemption_duaration'] = df['redemption_duaration'].astype(int)
    df['deactivation_counts']=df['deactivation_counts'].astype(int)
    df['interection_counts']=df['interection_counts'].astype(int)
    df['ival_counts']=df['ival_counts'].astype(int)
    df['iscompleted']=df['iscompleted'].astype(int)
    df['total_kb']=df['total_kb'].astype(int)
    df['voice_minutes']=df['voice_minutes'].astype(int)
    df['total_sms']=df['total_sms'].astype(int)
    df['reactivitaion_counts']=df['reactivitaion_counts'].astype(int)
    df['redenption_counts']=df['redenption_counts'].astype(int)
    df['date']=df['date'].astype(int)
    df['notify_counts']=df['notify_counts'].astype(int)
    df['notify_date']=df['notify_date'].astype(int)
    df['battery_available']=df['battery_available'].astype(int)
    df['data_roaming']=df['data_roaming'].astype(int)
    df['memory_available']=df['memory_available'].astype(int)
    df['temperature']=df['temperature'].astype(int)
    df['throttled_counts']=df['throttled_counts'].astype(int)
    df['throttled_date']=df['throttled_date'].astype(int)
    return df[columns]
    

        


    
    


dfx=convert_dataset_no_one_hot_more(df)
dfe=convert_dataset_no_one_hot_more(eval_set, True)

dfx.info()
dfx.head()

# %%
X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(dfx, dfy, test_size= 0.1, random_state = 1) 
cat_features = [5,9]

clf_o= cbt.CatBoostClassifier(iterations=2000,learning_rate=0.1,depth = 6,eval_metric='F1',loss_function='CrossEntropy',verbose=500)
clf_o.fit(X_train_o, y_train_o,cat_features=cat_features,plot=True)


train_report=classification_report(y_train_o, clf_o.predict(X_train_o))
test_report=classification_report(y_test_o, clf_o.predict(X_test_o))
print(train_report)
print(test_report)

# %%
X_train, X_test, y_train, y_test = train_test_split(dfx, dfy, test_size=0.1, random_state = 1)   
cat_features = [5,9]
NFOLDS = 5
folds = KFold(n_splits=NFOLDS)
columns = X_train.columns
splits = folds.split(X_train, y_train)
y_preds = np.zeros(X_test.shape[0])
y_oof = np.zeros(X_train.shape[0])
score = 0
  
for fold_n, (train_index, valid_index) in enumerate(splits):
    X_tr, X_val = X_train[columns].iloc[train_index], X_train[columns].iloc[valid_index]
    y_tr, y_val = y_train.iloc[train_index], y_train.iloc[valid_index]    
    clf= cbt.CatBoostClassifier(iterations=4000,learning_rate=0.1,eval_metric='F1',loss_function='CrossEntropy',verbose=500) 

    clf.fit(X_tr, y_tr,eval_set=(X_val, y_val),cat_features=cat_features)
    y_pred_valid = clf.predict_proba(X_val)[:,1]
    y_oof[valid_index] = y_pred_valid
    
    y_preds += clf.predict_proba(X_test)[:,1]/ NFOLDS    
    del X_tr, X_val, y_tr, y_val
    gc.collect()    

print('F1 score test', f1_score(y_test, np.round(y_preds)))

# %%



