# %%
import matplotlib.pyplot as plt
from sklearn import metrics 
import os
from datetime import datetime,time
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.metrics import f1_score
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

import re
%matplotlib inline

# %%
# Print files in current working directory 
os.chdir("/data/team15/data")
ebb_1 = pd.read_csv("ebb_set1.csv")
ebb_2 = pd.read_csv("ebb_set2.csv")
eval_set = pd.read_csv("eval_set.csv")
os.listdir(os.curdir)

# %%
ebb_1.info()

# %%
ebb_1_act = pd.read_csv("activations_ebb_set1.csv")
ebb_2_act = pd.read_csv("activations_ebb_set2.csv")
eval_act = pd.read_csv("activations_eval_set.csv")
#ebb_1_act.info()
#ebb_1_act.head()

def count_act_times(df):
    counts = df['customer_id'].value_counts(sort=False).rename_axis('customer_id').reset_index(name='activation_counts')
    df = counts.drop_duplicates(subset=['customer_id'])
    return df

ebb_1_act = count_act_times(ebb_1_act)
ebb_2_act = count_act_times(ebb_2_act)
eval_act = count_act_times(eval_act)

#ebb_1_act.head()

ebb_1_o_act = ebb_1.merge(ebb_1_act, how='left', on='customer_id')
ebb_2_o_act = ebb_2.merge(ebb_2_act, how='left', on='customer_id')
eval_o_act = eval_set.merge(eval_act, how='left', on='customer_id')
ebb_1_o_act.head()

# %%
ebb_1_deact = pd.read_csv("deactivations_ebb_set1.csv")
ebb_2_deact = pd.read_csv("deactivations_ebb_set2.csv")
eval_deact = pd.read_csv("deactivations_eval_set.csv")
#ebb_1_deact.head()
#ebb_1_deact.info()
def count_times(df):
    counts = df['customer_id'].value_counts(sort=False).rename_axis('customer_id').reset_index(name='deactivation_counts')
    df = counts.drop_duplicates(subset=['customer_id'])
    return df

ebb_1_deact = count_times(ebb_1_deact)
ebb_2_deact = count_times(ebb_2_deact)
eval_deact = count_times(eval_deact)

#ebb_1_deact.head()
ebb_1_act_de = ebb_1_o_act.merge(ebb_1_deact, how='left', on='customer_id')
ebb_2_act_de = ebb_2_o_act.merge(ebb_2_deact, how='left', on='customer_id')
eval_act_de_ = eval_o_act.merge(eval_deact, how='left', on='customer_id')
ebb_1_act_de.head()

# %%
ebb_1_inter = pd.read_csv("interactions_ebb_set1.csv")
ebb_2_inter = pd.read_csv("interactions_ebb_set2.csv")
eval_inter = pd.read_csv("interactions_eval_set.csv")
ebb_1_inter.info()
#ebb_1_inter.head()
def count_times_inter(df):
    counts = df['customer_id'].value_counts(sort=False).rename_axis('customer_id').reset_index(name='interection_counts')
    df = counts.drop_duplicates(subset=['customer_id'])
    return df
ebb_1_inter = count_times_inter(ebb_1_inter)
ebb_2_inter = count_times_inter(ebb_2_inter)
eval_inter = count_times_inter(eval_inter)

#ebb_1_inter.head()
ebb_1_adi = ebb_1_act_de.merge(ebb_1_inter, how='left', on='customer_id')
ebb_2_adi = ebb_2_act_de.merge(ebb_2_inter, how='left', on='customer_id')
eval_adi = eval_act_de_.merge(eval_inter, how='left', on='customer_id')
ebb_1_adi.info()
ebb_1_adi.head()

# %%
ebb_1_ival = pd.read_csv("ivr_calls_ebb_set1.csv")
ebb_2_ival = pd.read_csv("ivr_calls_ebb_set2.csv")
eval_ival = pd.read_csv("ivr_calls_eval_set.csv")
ebb_1_ival.info()
#ebb_1_ival.head()

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

#ebb_1_ival.head()
ebb_1_adii = ebb_1_adi.merge(ebb_1_ival, how='left', on='customer_id')
ebb_2_adii = ebb_2_adi.merge(ebb_2_ival, how='left', on='customer_id')
eval_adii = eval_adi.merge(eval_ival, how='left', on='customer_id')
ebb_1_adii.info()
ebb_1_adii.head()

# %%
ebb_1_net = pd.read_csv("network_ebb_set1.csv")
ebb_2_net = pd.read_csv("network_ebb_set2.csv")
eval_net = pd.read_csv("network_eval_set.csv")
ebb_1_net.info()
#ebb_1_net.head()

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
#ebb_1_net.info()
#ebb_1_net.head()
ebb_1_adiin = ebb_1_adii.merge(ebb_1_net, how='left', on='customer_id')
ebb_2_adiin = ebb_2_adii.merge(ebb_2_net, how='left', on='customer_id')
eval_adiin = eval_adii.merge(eval_net, how='left', on='customer_id')
ebb_1_adiin.info()
ebb_1_adiin.head()

# %%
ebb_1_not = pd.read_csv("notifying_ebb_set1.csv")
ebb_2_not = pd.read_csv("notifying_ebb_set2.csv")
eval_not = pd.read_csv("notifying_eval_set.csv")
ebb_1_not.info()
#ebb_1_not.head()

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
#ebb_1_not.head()

ebb_1_adiinn = ebb_1_adiin.merge(ebb_1_not, how='left', on='customer_id')
ebb_2_adiinn = ebb_2_adiin.merge(ebb_2_not, how='left', on='customer_id')
eval_adiinn = eval_adiin.merge(eval_not, how='left', on='customer_id')
ebb_1_adiinn.info()
ebb_1_adiinn.head()

# %%
ebb_1_pho = pd.read_csv("phone_data_ebb_set1.csv")
ebb_2_pho = pd.read_csv("phone_data_ebb_set2.csv")
eval_pho = pd.read_csv("phone_data_eval_set.csv")
ebb_1_pho.info()
ebb_1_pho['battery_available'].value_counts()

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
ebb_1_adiinnp.info()
ebb_1_adiinnp.head()

# %%
ebb_1_rea = pd.read_csv("reactivations_ebb_set1.csv")
ebb_2_rea = pd.read_csv("reactivations_ebb_set2.csv")
eval_rea = pd.read_csv("reactivations_eval_set.csv")
ebb_1_rea.info()
#ebb_1_rea.head()

def count_times_rea(df):
    counts = df['customer_id'].value_counts(sort=False).rename_axis('customer_id').reset_index(name='reactivitaion_counts')
    df = counts.drop_duplicates(subset=['customer_id'])
    return df
ebb_1_rea = count_times_rea(ebb_1_rea)
ebb_2_rea = count_times_rea(ebb_2_rea)
eval_rea = count_times_rea(eval_rea)
#ebb_1_rea.head()

ebb_1_adiinnpr = ebb_1_adiinnp.merge(ebb_1_rea, how='left', on='customer_id')
ebb_2_adiinnpr = ebb_2_adiinnp.merge(ebb_2_rea, how='left', on='customer_id')
eval_adiinnpr = eval_adiinnp.merge(eval_rea, how='left', on='customer_id')
ebb_1_adiinnpr.info()
ebb_1_adiinnpr.head()

# %%
ebb_1_red = pd.read_csv("redemptions_ebb_set1.csv")
ebb_2_red = pd.read_csv("redemptions_ebb_set2.csv")
eval_red = pd.read_csv("redemptions_eval_set.csv")
ebb_1_red.info()
#ebb_1_red.head()

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
#ebb_1_red.head()

ebb_1_adiinnprr = ebb_1_adiinnpr.merge(ebb_1_red, how='left', on='customer_id')
ebb_2_adiinnprr = ebb_2_adiinnpr.merge(ebb_2_red, how='left', on='customer_id')
eval_adiinnprr = eval_adiinnpr.merge(eval_red, how='left', on='customer_id')
ebb_1_adiinnprr.info()
ebb_1_adiinnprr.head()

# %%
ebb_1_sup = pd.read_csv("support_ebb_set1.csv")
ebb_2_sup = pd.read_csv("support_ebb_set2.csv")
eval_sup = pd.read_csv("support_eval_set.csv")
#ebb_1_act.info()
#ebb_1_act.head()

def count_act_times(df):
    counts = df['customer_id'].value_counts(sort=False).rename_axis('customer_id').reset_index(name='support_counts')
    df = counts.drop_duplicates(subset=['customer_id'])
    return df

ebb_1_sup = count_act_times(ebb_1_sup)
ebb_2_sup = count_act_times(ebb_2_sup)
eval_sup = count_act_times(eval_sup)

#ebb_1_act.head()

ebb_1_o_sup = ebb_1_adiinnprr.merge(ebb_1_sup, how='left', on='customer_id')
ebb_2_o_sup = ebb_2_adiinnprr.merge(ebb_2_sup, how='left', on='customer_id')
eval_o_sup = eval_adiinnprr.merge(eval_sup, how='left', on='customer_id')
ebb_1_o_sup.head()

# %%
ebb_1_thr = pd.read_csv("throttling_ebb_set1.csv")
ebb_2_thr = pd.read_csv("throttling_ebb_set2.csv")
eval_thr = pd.read_csv("throttling_eval_set.csv")
ebb_1_thr.info()
#ebb_1_thr.head()

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
#ebb_1_thr.head()

ebb_1_adiinnprrt = ebb_1_o_sup.merge(ebb_1_thr, how='left', on='customer_id')
ebb_2_adiinnprrt = ebb_2_o_sup.merge(ebb_2_thr, how='left', on='customer_id')
eval_adiinnprrt = eval_o_sup.merge(eval_thr, how='left', on='customer_id')
ebb_1_adiinnprrt.info()
ebb_1_adiinnprrt.head()

# %%
ebb_1_loy = pd.read_csv("loyalty_program_ebb_set1.csv")
ebb_2_loy = pd.read_csv("loyalty_program_ebb_set2.csv")
eval_loy = pd.read_csv("loyalty_program_eval_set.csv")
#ebb_1_act.info()
#ebb_1_act.head()

def count_act_times(df):
    counts = df['customer_id'].value_counts(sort=False).rename_axis('customer_id').reset_index(name='loy_counts')
    df = counts.drop_duplicates(subset=['customer_id'])
    return df

ebb_1_loy1 = count_act_times(ebb_1_loy)
ebb_2_loy1 = count_act_times(ebb_2_loy)
eval_loy1 = count_act_times(eval_loy)

ebb_1_loy = ebb_1_loy.merge(ebb_1_loy1, how='left', on='customer_id')
ebb_2_loy = ebb_2_loy.merge(ebb_2_loy1, how='left', on='customer_id')
eval_loy = eval_loy.merge(eval_loy1, how='left', on='customer_id')

for col in ebb_1_loy.columns:
    if col == 'lrp_enrolled' or col == 'date':
        ebb_1_loy = ebb_1_loy.drop(col, axis=1)
        ebb_2_loy = ebb_2_loy.drop(col, axis=1)
        eval_loy = eval_loy.drop(col, axis=1)
ebb_1_loy.head()

ebb_1_loyy = ebb_1_adiinnprrt.merge(ebb_1_loy, how='left', on='customer_id')
ebb_2_loyy = ebb_2_adiinnprrt.merge(ebb_2_loy, how='left', on='customer_id')
eval_loyy = eval_adiinnprrt.merge(eval_loy, how='left', on='customer_id')
ebb_1_loyy['total_quantity']=ebb_1_loyy['total_quantity'].apply(lambda x: 0 if pd.isnull(x) else x)
ebb_1_loyy['loy_counts']=ebb_1_loyy['loy_counts'].apply(lambda x: 0 if pd.isnull(x) else x)
ebb_1_loyy.info()
ebb_1_loyy.head()

# %%
ebb_1_sus = pd.read_csv("suspensions_ebb_set1.csv")
ebb_2_sus = pd.read_csv("suspensions_ebb_set2.csv")
eval_sus = pd.read_csv("suspensions_eval_set.csv")
#ebb_1_act.info()
#ebb_1_act.head()

for col in ebb_1_sus:
    if col == 'start_date' or col == 'end_date':
        ebb_1_sus[col] = pd.to_datetime(ebb_1_sus[col])
        ebb_2_sus[col] = pd.to_datetime(ebb_1_sus[col])
        eval_sus[col] = pd.to_datetime(ebb_1_sus[col])
        
ebb_1_sus['sus_time'] = ebb_1_sus['end_date'] - ebb_1_sus['start_date']
ebb_1_sus['sus_time'] = ebb_1_sus['sus_time'].astype('timedelta64[D]').astype(int)
ebb_2_sus['sus_time'] = ebb_2_sus['end_date'] - ebb_2_sus['start_date']
ebb_2_sus['sus_time'] = ebb_2_sus['sus_time'].astype('timedelta64[D]').astype(int)
eval_sus['sus_time'] = eval_sus['end_date'] - eval_sus['start_date']
eval_sus['sus_time'] = eval_sus['sus_time'].astype('timedelta64[D]').astype(int)

ebb_1_sus1 = ebb_1_sus.groupby(by=['customer_id'])['sus_time'].sum().rename_axis('customer_id').reset_index(name='sus_sum')
ebb_2_sus1 = ebb_2_sus.groupby(by=['customer_id'])['sus_time'].sum().rename_axis('customer_id').reset_index(name='sus_sum')
eval_sus1 = eval_sus.groupby(by=['customer_id'])['sus_time'].sum().rename_axis('customer_id').reset_index(name='sus_sum')

ebb_1_sus = ebb_1_loyy.merge(ebb_1_sus1, how='left', on='customer_id')
ebb_2_sus = ebb_2_loyy.merge(ebb_2_sus1, how='left', on='customer_id')
eval_sus = eval_loyy.merge(eval_sus1, how='left', on='customer_id')

ebb_1_sus['sus_sum']=ebb_1_sus['sus_sum'].apply(lambda x: 0 if pd.isnull(x) else x)

ebb_1_sus.info()
ebb_1_sus.head()


# %%
ebb_1 = ebb_1_sus
ebb_2 = ebb_2_sus
eval_set = eval_sus

# %%
#########################################same, just for saving time##############################################
# Print files in current working directory 
#os.chdir("/data/team15/data")
#os.listdir(os.curdir)
#ebb_1 = pd.read_csv("ebb_set1.csv")
#ebb_2 = pd.read_csv("ebb_set2.csv")
#eval_set = pd.read_csv("eval_set.csv")


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


# %%
df.info()
df.head()

# %%
temp_columns = []
for col in df.columns:
    counter = df[col].isnull().sum()
    ratio = counter / len(df[col])
    if ratio > 0.7:
         temp_columns.append(col)
df = df.drop(temp_columns, axis=1)
dfe = eval_set.drop(temp_columns, axis=1)

df['ebb_eligible']=df['ebb_eligible'].apply(lambda x: 0 if pd.isnull(x) else 1)
dfy=df[['ebb_eligible']]
dfx = df.drop('ebb_eligible', axis=1)
dfx = dfx.drop('customer_id', axis=1)
dfx.info()
dfx.head()

# %%
dfe.info()
dfe.head()

# %%
label = LabelEncoder()
def transform_to_int(df):
    for col in df:
        if col == 'operating_system' or col == 'state' or col == 'manufacturer':
            df[col] = label.fit_transform(df[col])
    for col in df:
        if col == 'last_redemption_date' or col == 'first_activation_date':
            df[col] = pd.to_datetime(df[col])
            df[col] = df[col].astype(int)
transform_to_int(dfx)
transform_to_int(dfe)
dfx.info()
dfe.info()

# %%
#def fill_nan(df):
#    for col in df:
#        df[col].fillna(df[col].median(), inplace=True)
#fill_nan(dfx)

for col in dfx:
    dfx[col]=dfx[col].apply(lambda x: 0 if pd.isnull(x) else x)
dfx.info()

# %%
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score,recall_score,f1_score,classification_report
import catboost as cbt
from sklearn.ensemble import HistGradientBoostingClassifier

X_train, X_test, Y_train, Y_test = train_test_split(dfx, dfy, test_size= 0.3)
X_train=X_train.values
Y_train=Y_train.values
X_test=X_test.values
Y_test=Y_test.values
y_train=Y_train.reshape(-1)
y_test=Y_test.reshape(-1)

def testModels(X_train, y_train, X_test, y_test):
    models=[]
    models.append(DecisionTreeClassifier())
    models.append(SGDClassifier())
    models.append(KNeighborsClassifier())
    models.append(LinearDiscriminantAnalysis())
    # models.append(sklearn.linear_model.LogisticRegression())
    models.append(GaussianNB())
    # models.append(sklearn.svm.SVC())
    models_name=['DecisionTree','SGD', 'KNN', 'LinearDiscriminant', 'GaussianNB']
    models_precision=[]
    models_recall=[]
    models_f1=[]
    models_report=[]
    for _ in models:
        _.fit(X_train, y_train)
        y_pred=_.predict(X_test)
        models_precision.append(precision_score(y_test,y_pred, average='macro'))
        models_recall.append(recall_score(y_test,y_pred, average='macro'))
        models_f1.append(f1_score(y_test,y_pred, average='macro'))
        models_report.append(classification_report(y_test, y_pred))
    
    models_name.append('Catboost')
    added_model1=cbt.CatBoostClassifier(iterations=2000,learning_rate=0.1,eval_metric='F1',loss_function='CrossEntropy',verbose=False) 
    added_model1.fit(X_train, y_train)
    y_pred=added_model1.predict(X_test)
    #print(y_pred)
    models_precision.append(precision_score(y_test,y_pred, average='macro'))
    models_recall.append(recall_score(y_test,y_pred, average='macro'))
    models_f1.append(f1_score(y_test,y_pred, average='macro'))
    models_report.append(classification_report(y_test, y_pred))
    
    models_name.append('XGboost')
    added_model2=XGBClassifier()
    added_model2.fit(X_train, y_train)
    y_pred=added_model2.predict(X_test)
    #print(y_pred)
    models_precision.append(precision_score(y_test,y_pred, average='macro'))
    models_recall.append(recall_score(y_test,y_pred, average='macro'))
    models_f1.append(f1_score(y_test,y_pred, average='macro'))
    models_report.append(classification_report(y_test, y_pred))
    
    models_name.append('HisGradientBoosting')
    added_model3=HistGradientBoostingClassifier(random_state = 1)
    added_model3.fit(X_train, y_train)
    y_pred=added_model3.predict(X_test)
    #print(y_pred)
    models_precision.append(precision_score(y_test,y_pred, average='macro'))
    models_recall.append(recall_score(y_test,y_pred, average='macro'))
    models_f1.append(f1_score(y_test,y_pred, average='macro'))
    models_report.append(classification_report(y_test, y_pred))

    plt.clf()
    plt.figure(figsize=(15,10))
    plt.title('Different Models')
    bar_x=np.arange(len(models_name))
    bar_width=0.3
    plt.bar(x=bar_x-bar_width, height=models_precision, label='precision', width=bar_width)
    plt.bar(x=bar_x, height=models_recall, label='recall', width=bar_width)
    plt.bar(x=bar_x+bar_width, height=models_f1, label='f1-score', width=bar_width)
    for a,b in zip(bar_x-bar_width, models_precision):
        plt.text(a,b+0.01,b.round(2),ha='center')
    for a,b in zip(bar_x, models_recall):
        plt.text(a,b+0.01,b.round(2),ha='center')
    for a,b in zip(bar_x+bar_width, models_f1):
        plt.text(a,b+0.01,b.round(2),ha='center')
    plt.xticks(bar_x, labels=models_name)
    plt.legend()
    plt.ylim(0,1)
    plt.show()

    print('Summary:')
    for i in range(len(models_name)):
        print(models_name[i])
        print(models_report[i])
    print('End Summary')
testModels(X_train, y_train, X_test, y_test)

# %%



