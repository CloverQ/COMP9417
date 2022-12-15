# %% [markdown]
# # Import labs

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
import catboost as cbt
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier


# %% [markdown]
# # Read the data
# ### origianl dataset

# %%
os.chdir("/data/team15/data")
ebb_1 = pd.read_csv("ebb_set1.csv")
ebb_2 = pd.read_csv("ebb_set2.csv")
eval_set = pd.read_csv("eval_set.csv")

# %% [markdown]
# ## Feature Processing (no one-hot) and pass the processed dataset to the catboost classifier

# %%
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

######################## append the column to the dataset if the sum of its items is bigger than 45000 ############################

#after convert_date, clean_operating_system
def convert_dataset_no_one_hot(df, isEval=False):
    df['state'] = df['state'].replace([None],'Not Known')   
    
    df = df.fillna('0') 
    
    columns=['total_redemptions','tenure','number_upgrades','year','total_revenues_bucket',\
             'state','last_redemption','first_activation','redemption_duaration',\
             'operating_system','opt_out_mobiles_ads']
    if isEval:
        columns.insert(0, 'customer_id')
           
    df['state'] = df['state'].astype(str)
    df['operating_system'] = df['operating_system'].astype(str)
    df['opt_out_mobiles_ads'] = df['opt_out_mobiles_ads'].astype(int)
    df['last_redemption'] = df['last_redemption'].astype(int)
    df['first_activation'] = df['first_activation'].astype(int)
    df['redemption_duaration'] = df['redemption_duaration'].astype(int)
    return df[columns]

df['ebb_eligible']=df['ebb_eligible'].apply(lambda x: 0 if pd.isnull(x) else 1)
dfy=df[['ebb_eligible']]
dfx=convert_dataset_no_one_hot(df)
dfe=convert_dataset_no_one_hot(eval_set, True)
dfx.info()
dfx.head()

# %% [markdown]
# #### The names of result pictures are 'origianl_simple_catboost_f1' and 'origianl_simple_catboost_ce'. (The change curves of F1 and cross entrofy during the experiment).

# %%
########################### test_size is the same as the submission one ###############################################
X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(dfx, dfy, test_size= 0.1, random_state = 1) 
cat_features = [5,9]

clf_o= cbt.CatBoostClassifier(iterations=2000,learning_rate=0.1,eval_metric='F1',loss_function='CrossEntropy',verbose=500)
clf_o.fit(X_train_o, y_train_o,cat_features=cat_features,plot=True)


train_report=classification_report(y_train_o, clf_o.predict(X_train_o))
test_report=classification_report(y_test_o, clf_o.predict(X_test_o))
print(train_report)
print(test_report)



# %% [markdown]
# ### Adding more sheets

# %%
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

# %%
ebb_1.info()

# %% [markdown]
# ## Feature Processing (no one-hot) and pass the processed dataset to the catboost classifier

# %%
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


######################## append the column to the dataset if the sum of its items is bigger than 45000 ############################

#after convert_date, clean_operating_system
def convert_dataset_no_one_hot(df, isEval=False):
    df['state'] = df['state'].replace([None],'Not Known')   
    
    df = df.fillna('0') 
    
    columns=['total_redemptions','tenure','number_upgrades','year','total_revenues_bucket',\
             'state','last_redemption','first_activation','redemption_duaration',\
             'operating_system','opt_out_mobiles_ads','deactivation_counts',\
             'interection_counts','ival_counts','iscompleted','total_kb','voice_minutes',\
             'total_sms','reactivitaion_counts','redenption_counts','date']
    if isEval:
        columns.insert(0, 'customer_id')
           
    df['state'] = df['state'].astype(str)
    df['operating_system'] = df['operating_system'].astype(str)
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
    return df[columns]

df['ebb_eligible']=df['ebb_eligible'].apply(lambda x: 0 if pd.isnull(x) else 1)
dfy=df[['ebb_eligible']]
dfx=convert_dataset_no_one_hot(df)
dfe=convert_dataset_no_one_hot(eval_set, True)
dfx.info()
dfx.head()

# %% [markdown]
# ## Pass the dataset to the simpliest Catboost, to check whether adding more sheets will increas the F1 score
# #### The names of result pictures are 'origianl_simple_catboost_f1' and 'origianl_simple_catboost_ce'. (The change curves of F1 and cross entrofy during the experiment).

# %%
X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(dfx, dfy, test_size= 0.1, random_state = 1) 
cat_features = [5,9]

clf_o= cbt.CatBoostClassifier(iterations=2000,learning_rate=0.1,eval_metric='F1',loss_function='CrossEntropy',verbose=500)
clf_o.fit(X_train_o, y_train_o,cat_features=cat_features,plot=True)


train_report=classification_report(y_train_o, clf_o.predict(X_train_o))
test_report=classification_report(y_test_o, clf_o.predict(X_test_o))
print(train_report)
print(test_report)

# %% [markdown]
# ##  Checking whether adding a cross validition will increase the F1

# %% [markdown]
# ### KFold = 5 (the best)

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
    clf= cbt.CatBoostClassifier(iterations=2000,learning_rate=0.1,eval_metric='F1',loss_function='CrossEntropy',verbose=500) 

    clf.fit(X_tr, y_tr,eval_set=(X_val, y_val),cat_features=cat_features)
    y_pred_valid = clf.predict_proba(X_val)[:,1]
    y_oof[valid_index] = y_pred_valid
    
    y_preds += clf.predict_proba(X_test)[:,1]/ NFOLDS    
    del X_tr, X_val, y_tr, y_val
    gc.collect()    

print('F1 score test', f1_score(y_test, np.round(y_preds)))

# %% [markdown]
# ### StratifiedShuffleSplit, n_split = 5

# %%
X_train, X_test, y_train, y_test = train_test_split(dfx, dfy, test_size=0.1, random_state = 1)    
y_oof = np.zeros(X_train.shape[0]) 
y_preds = np.zeros(X_test.shape[0])
skf = StratifiedShuffleSplit(n_splits=5) 
cat_features = [5,9]


for index, (train_index, valid_index) in enumerate(skf.split(X_train, y_train)): 
    X_tr, X_val, y_tr, y_val = \
    X_train.iloc[train_index], \
    X_train.iloc[valid_index],\
    y_train.iloc[train_index],\
    y_train.iloc[valid_index]
    clf= cbt.CatBoostClassifier(iterations=2000,learning_rate=0.1,eval_metric='F1',loss_function='CrossEntropy',verbose=500) 
    
    #train dataset after 5-fold
    clf.fit(X_tr, y_tr, eval_set=(X_val, y_val),cat_features = cat_features) 

    prediction_cat = clf.predict_proba(X_val)[:,1]
    y_preds += clf.predict_proba(X_test)[:,1]/ 5
    del X_tr, X_val, y_tr, y_val
    gc.collect() 

print('score', f1_score(y_test, np.round(y_preds)))

# %% [markdown]
# ## Check whether default parameters will increase the F1 score ->NO!

# %%
X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(dfx, dfy, test_size= 0.1, random_state = 1) 
cat_features = [5,9]

clf_o= cbt.CatBoostClassifier(verbose=500)
clf_o.fit(X_train_o, y_train_o,cat_features=cat_features,plot=True)


train_report=classification_report(y_train_o, clf_o.predict(X_train_o))
test_report=classification_report(y_test_o, clf_o.predict(X_test_o))
print(train_report)
print(test_report)

# %% [markdown]
# ## Grid Search ->success

# %%
from sklearn.model_selection import GridSearchCV
X_train, X_test, y_train, y_test = train_test_split(dfx, dfy, test_size= 0.1, random_state = 1) 
model = cbt.CatBoostClassifier(verbose = 500)
cat_features = [5,9]

grid = {'learning_rate': [0.01, 0.05, 0.1],
        'depth': [4, 6, 10]}

grid = GridSearchCV(estimator=model, param_grid = grid, n_jobs=1)
grid.fit(X_train, y_train, cat_features = cat_features)

# %%
print("\n The best estimator across ALL searched params:\n", grid.best_estimator_)
print("\n The best score across ALL searched params:\n", grid.best_score_)
print("\n The best parameters across ALL searched params:\n", grid.best_params_)

# %% [markdown]
# ## Submit the prediction of eval set

# %%
dfe_test = dfe.iloc[:,1:]
dfe_test.info()

########################### still have train and test ###############################################
X_train, X_test, y_train, y_test = train_test_split(dfx, dfy, test_size= 0.1, random_state = 1) 
X_test = dfe_test
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
    clf= cbt.CatBoostClassifier(iterations=3000,learning_rate=0.1,eval_metric='F1',loss_function='CrossEntropy',verbose=500) 

    clf.fit(X_tr, y_tr,eval_set=(X_val, y_val),cat_features=cat_features)
    y_pred_valid = clf.predict_proba(X_val)[:,1]
    y_oof[valid_index] = y_pred_valid
    
    y_preds += clf.predict_proba(X_test)[:,1]/ NFOLDS    
    del X_tr, X_val, y_tr, y_val
    gc.collect()    

y_preds = np.round(y_preds)
print(y_preds)


# %%
def getDateString():
    return pd.Timestamp.now().strftime('%Y-%m-%d')

def submit_prediction(prediction_list, eval_df, csv_path = None):
    if csv_path==None:
        csv_path = getDateString() + '.csv'
    df_csv = pd.DataFrame()
    df_csv['customer_id'] = dfe['customer_id']
    df_csv['ebb_eligible'] = prediction_list
    df_csv = df_csv.drop_duplicates(['customer_id'])
    df_csv.to_csv(csv_path, index=False)

# %%
submit_prediction(y_preds, dfe)

# %% [markdown]
# ## Result of TestModels

# %% [markdown]
# ### Generate TestModles on my final dataset

# %%
# need to delete the cat_features()->no state and no operating system
dfx_no_cat = dfx[['total_redemptions',
 'tenure',
 'number_upgrades',
 'year',
 'total_revenues_bucket',
 'last_redemption',
 'first_activation',
 'redemption_duaration',
 'opt_out_mobiles_ads',
 'deactivation_counts',
 'interection_counts',
 'ival_counts',
 'iscompleted',
 'total_kb',
 'voice_minutes',
 'total_sms',
 'reactivitaion_counts',
 'redenption_counts',
 'date']]
dfx_no_cat.info()

# %%
from xgboost import XGBClassifier
X_train, X_test, Y_train, Y_test = train_test_split(dfx_no_cat, dfy, test_size= 0.3)
X_train=X_train.values
Y_train=Y_train.values
X_test=X_test.values
Y_test=Y_test.values
y_train=Y_train.reshape(-1)
y_test=Y_test.reshape(-1)

def testModels(X_train, y_train, X_test, y_test):
    models=[]
    models.append(sklearn.tree.DecisionTreeClassifier())
    models.append(sklearn.linear_model.SGDClassifier())
    models.append(sklearn.neighbors.KNeighborsClassifier())
    models.append(sklearn.discriminant_analysis.LinearDiscriminantAnalysis())
    # models.append(sklearn.linear_model.LogisticRegression())
    models.append(sklearn.naive_bayes.GaussianNB())
    # models.append(sklearn.svm.SVC())
    models_name=['DecisionTree','SGD', 'KNN', 'LinearDiscriminant', 'GaussianNB']
    models_precision=[]
    models_recall=[]
    models_f1=[]
    models_report=[]
    for _ in models:
        _.fit(X_train, y_train)
        y_pred=_.predict(X_test)
        models_precision.append(sklearn.metrics.precision_score(y_test,y_pred, average='macro'))
        models_recall.append(sklearn.metrics.recall_score(y_test,y_pred, average='macro'))
        models_f1.append(sklearn.metrics.f1_score(y_test,y_pred, average='macro'))
        models_report.append(sklearn.metrics.classification_report(y_test, y_pred))
    
    models_name.append('Catboost')
    added_model1=cbt.CatBoostClassifier(iterations=2000,learning_rate=0.1,eval_metric='F1',loss_function='CrossEntropy',verbose=False) 
    added_model1.fit(X_train, y_train)
    y_pred=added_model1.predict(X_test)
    #print(y_pred)
    models_precision.append(sklearn.metrics.precision_score(y_test,y_pred, average='macro'))
    models_recall.append(sklearn.metrics.recall_score(y_test,y_pred, average='macro'))
    models_f1.append(sklearn.metrics.f1_score(y_test,y_pred, average='macro'))
    models_report.append(sklearn.metrics.classification_report(y_test, y_pred))
    
    models_name.append('XGboost')
    added_model2=XGBClassifier()
    added_model2.fit(X_train, y_train)
    y_pred=added_model2.predict(X_test)
    #print(y_pred)
    models_precision.append(sklearn.metrics.precision_score(y_test,y_pred, average='macro'))
    models_recall.append(sklearn.metrics.recall_score(y_test,y_pred, average='macro'))
    models_f1.append(sklearn.metrics.f1_score(y_test,y_pred, average='macro'))
    models_report.append(sklearn.metrics.classification_report(y_test, y_pred))
    
    models_name.append('HisGradientBoosting')
    added_model3=HistGradientBoostingClassifier(random_state = 1)
    added_model3.fit(X_train, y_train)
    y_pred=added_model3.predict(X_test)
    #print(y_pred)
    models_precision.append(sklearn.metrics.precision_score(y_test,y_pred, average='macro'))
    models_recall.append(sklearn.metrics.recall_score(y_test,y_pred, average='macro'))
    models_f1.append(sklearn.metrics.f1_score(y_test,y_pred, average='macro'))
    models_report.append(sklearn.metrics.classification_report(y_test, y_pred))

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


# %%
testModels(X_train, y_train, X_test, y_test)

# %% [markdown]
# ## One-hot operating system on origianl dataset, to see whether it will increase the F1 score

# %%
ebb_1_op = pd.read_csv("ebb_set1.csv")
ebb_2_op = pd.read_csv("ebb_set2.csv")
eval_set_op = pd.read_csv("eval_set.csv")

def convert_date(df):
    today_str = '2022-4-10'
    today = datetime.strptime(today_str, "%Y-%m-%d")
    df['last_redemption']=pd.to_datetime(df['last_redemption_date'],format='%Y %m %d')
    df['first_activation']=pd.to_datetime(df['first_activation_date'],format='%Y %m %d')
    df['last_redemption'] = today-df['last_redemption']
    df['first_activation'] = today-df['first_activation']
    df['redemption_activition'] = df['first_activation']-df['last_redemption'] 
    df['last_redemption'] = df['last_redemption'].astype('timedelta64[D]')
    df['last_redemption'] = df['last_redemption'].astype(int)
    df['first_activation'] = df['first_activation'].astype('timedelta64[D]')
    df['first_activation'] = df['first_activation'].astype(int)
    df['redemption_activition'] = df['redemption_activition'].astype('timedelta64[D]')
    df['redemption_activition'] = df['redemption_activition'].astype(int)
    return df
convert_date(ebb_1_op)
convert_date(ebb_2_op)
convert_date(eval_set_op)

# %% [markdown]
# ### apply One-hot only operating_system, not on state(too many dimensions)
# Keep 4 labels
# 
#  ### processing operating_system before one-hot code   
#     1. convert non-standard labels into canonical labels (such as convert'Android,Android' to 'Android')
#     2. in addition to 'Android','iOS','Not Known', all encoded as one feature 'Other' (combine left systems to one feature)
#     3. fill the missing data in the operating_syatem column with 'Not Known'

# %%
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

ebb_1_op['operating_system']=ebb_1_op['operating_system'].apply(clean_operating_system)
print(ebb_1_op['operating_system'].value_counts())
ebb_2_op['operating_system']=ebb_2_op['operating_system'].apply(clean_operating_system)
print(ebb_2_op['operating_system'].value_counts())
eval_set_op['operating_system']=eval_set_op['operating_system'].apply(clean_operating_system)
print(eval_set_op['operating_system'].value_counts())

# %%
ebb_1_op.info()

# %% [markdown]
# ### operating_system one-hot 
# 
# Why only one feature one-hot->In order to ensure the independence between the variables of logistic regression, only one-hot encoding of one feature is reserved. Therefore, if multiple features need to be one-hot encoded, the results of several one-hots of a certain sample may be 1 at the same time. At this time, the two features are completely related, which will lead to a singular error. That is, a non-singular matrix cannot have a unique solution.(https://blog.csdn.net/weixin_39750084/article/details/81432619?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1.pc_relevant_antiscanv2&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1.pc_relevant_antiscanv2&utm_relevant_index=2) Also someone comments that this is wrong...
# 
# 
# Tree model->Seem that do not need one-hot

# %%
#Step 1: By using LabelEncoder, the discrete categories are numbered, and it will be labeled according to the type of value.
#Since the labelEncoder is set in advance,the same transformation setting could be used for different datasets.
onehot = OneHotEncoder()
label = LabelEncoder()
temp = label.fit_transform(['ANDROID','IOS','Other','Not Known'])
print('Labels after LabelEncoder',temp)


#Step2: Fit OneHotEncoder
onehot.fit(temp.reshape(-1,1))

#Step 3: Apply OneHotEncoder transformation to different dataset
def onehot_operating_system(df):
    temp_column = np.array(df['operating_system']).reshape(-1,1)
    temp_column = label.transform(temp_column).reshape(-1, 1)
    #print('The result of LabelEncoder',temp_column[0:10])
    temp_column = onehot.transform(temp_column)
    #print('The result of OneHotEncoder',temp_column[0:10])
    temp_column = pd.DataFrame(temp_column.todense())
    #print('Dense result',temp_column)
    df = pd.concat([df, temp_column], axis=1)
    df = df.rename(columns={0: 'ANDROID',1: 'IOS',2: 'Not Known',3: 'Other'})
    
    return df

ebb_1_op = onehot_operating_system(ebb_1_op)
ebb_2_op = onehot_operating_system(ebb_2_op)
eval_set_op = onehot_operating_system(eval_set_op)
#OK, correct
print(ebb_1_op['Other'].value_counts())
print(ebb_2_op['Other'].value_counts())
print(eval_set_op['Other'].value_counts())

# %% [markdown]
# ### fianl dataset + operating_system one-hot + state still nouns ->CatBoost's dataset
# (Guess it will works better on operating_system without one-hot, who knows, let's try XD)

# %%
df_op=pd.concat([ebb_1_op,ebb_2_op])
#fill missing state with Not Known

#check for decision tree
def convert_dataset(df, isEval=False):
    #df.loc[df['state'].notnull(), 'state'] = 1
    #df['state'] = df['state'].replace([None],-1)
    df['state'] = df['state'].replace([None],'NULL')  
    
    df['opt_out_mobiles_ads']= df['opt_out_mobiles_ads'].fillna('0')
    df['opt_out_mobiles_ads'] = df['opt_out_mobiles_ads'].astype(int)
    
    #print(df['opt_out_mobiles_ads'].value_counts())
    columns=['last_redemption','first_activation','total_redemptions','tenure',\
             'number_upgrades','year','total_revenues_bucket','state',\
             'ANDROID', 'IOS', 'Not Known', 'Other','opt_out_mobiles_ads','redemption_activition']
    if isEval:
        columns.insert(0, 'customer_id')
    df['ANDROID'] = df['ANDROID'].astype(int)
    df['IOS'] = df['IOS'].astype(int)
    df['Not Known'] = df['Not Known'].astype(int)
    df['Other'] = df['Other'].astype(int)    
    return df[columns]

df_op['ebb_eligible']=df_op['ebb_eligible'].apply(lambda x: 0 if pd.isnull(x) else 1)
dfx_op=convert_dataset(df_op)
dfy_op=df_op[['ebb_eligible']]
dfe_op=convert_dataset(eval_set_op, True)

# %%
dfx_op.info()

# %% [markdown]
# ### Pass the processed dataset to CatBoostClassifier
# based on https://www.kaggle.com/code/xwxw2929/catboost-kfold/noteboo

# %%
X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(dfx_op, dfy_op, test_size= 0.1, random_state = 1) 
cat_features = [7]

clf_o= cbt.CatBoostClassifier(iterations=2000,learning_rate=0.1,eval_metric='F1',loss_function='CrossEntropy',verbose=500)
clf_o.fit(X_train_o, y_train_o,cat_features=cat_features,plot=True)


train_report=classification_report(y_train_o, clf_o.predict(X_train_o))
test_report=classification_report(y_test_o, clf_o.predict(X_test_o))
print(train_report)
print(test_report)

# %% [markdown]
# ## Draft of my code, useless
# ### Do not need to run these code to formulate the final dataset, we've already get this from the previous block
# ### deactivations_ebb_set

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

# %% [markdown]
# ### interection_set

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

# %% [markdown]
# ### ival_call set

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

# %% [markdown]
# ### Network_ebb

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

# %% [markdown]
# ### notify_ebb

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

# %% [markdown]
# ### phone_data_ebb_set1.csv

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

# %% [markdown]
# ### reactivations_ebb_set1.csv

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

# %% [markdown]
# ### redemptions_ebb_set1.csv

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

# %% [markdown]
# ### throttling_ebb_set1.csv

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

ebb_1_adiinnprrt = ebb_1_adiinnprr.merge(ebb_1_thr, how='left', on='customer_id')
ebb_2_adiinnprrt = ebb_2_adiinnprr.merge(ebb_2_thr, how='left', on='customer_id')
eval_adiinnprrt = eval_adiinnprr.merge(eval_thr, how='left', on='customer_id')
ebb_1_adiinnprrt.info()
ebb_1_adiinnprrt.head()

# %%
ebb_1 = ebb_1_adiinnprrt 
ebb_2 = ebb_2_adiinnprrt 
eval_set = eval_adiinnprrt 

# %% [markdown]
# # Feature engineering and preprocessing

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
######################## append the column to the dataset if the sum of its items is bigger than 45000 ############################

#after convert_date, clean_operating_system
def convert_dataset_no_one_hot(df, isEval=False):
    df['state'] = df['state'].replace([None],'Not Known')   
    
    df = df.fillna('0') 
    
    columns=['total_redemptions','tenure','number_upgrades','year','total_revenues_bucket',\
             'state','last_redemption','first_activation','redemption_duaration',\
             'operating_system','opt_out_mobiles_ads','deactivation_counts',\
             'interection_counts','ival_counts','iscompleted','total_kb','voice_minutes',\
             'total_sms','reactivitaion_counts','redenption_counts','date']
    if isEval:
        columns.insert(0, 'customer_id')
           
    df['state'] = df['state'].astype(str)
    df['operating_system'] = df['operating_system'].astype(str)
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
    return df[columns]

df['ebb_eligible']=df['ebb_eligible'].apply(lambda x: 0 if pd.isnull(x) else 1)
dfy=df[['ebb_eligible']]
dfx=convert_dataset_no_one_hot(df)
dfe=convert_dataset_no_one_hot(eval_set, True)
dfx.info()
dfx.head()

# %% [markdown]
# ### Pass this fianl dataset to 3.

# %% [markdown]
# ### Fit the different datasets to CatboostClassifier with opearation system+state all string type and 5 Fold

# %% [markdown]
# ### 1. original dataset ->0.818

# %%
########################### 5-Hold+catboost ###############################################
X_train, X_test, y_train, y_test = train_test_split(dfx, dfy, test_size=0.2, random_state = 1)   
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
    clf= cbt.CatBoostClassifier(iterations=1100,learning_rate=0.1,eval_metric='F1',loss_function='CrossEntropy',verbose=100) 

    clf.fit(X_tr, y_tr,eval_set=(X_val, y_val),cat_features=cat_features,plot=True)
    y_pred_valid = clf.predict_proba(X_val)[:,1]
    print('F1 score validation',f1_score(y_val, np.round(y_pred_valid)))
    
    y_preds += clf.predict_proba(X_test)[:,1]/ NFOLDS    
    del X_tr, X_val, y_tr, y_val
    gc.collect()    

print('F1 score testing', f1_score(y_test, np.round(y_preds)))

# %% [markdown]
# ### 2.  +activations_ebb_set1.csv ->0.819

# %%
dfx.head()

# %%
########################### 5-Hold+catboost ###############################################
X_train, X_test, y_train, y_test = train_test_split(dfx, dfy, test_size=0.2, random_state = 1)   
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
    clf= cbt.CatBoostClassifier(iterations=1100,learning_rate=0.1,eval_metric='F1',loss_function='CrossEntropy',verbose=100) 

    clf.fit(X_tr, y_tr,eval_set=(X_val, y_val),cat_features=cat_features,plot=True)
    y_pred_valid = clf.predict_proba(X_val)[:,1]
    y_oof[valid_index] = y_pred_valid
    
    y_preds += clf.predict_proba(X_test)[:,1]/ NFOLDS    
    del X_tr, X_val, y_tr, y_val
    gc.collect()    

print('F1 score test', f1_score(y_test, np.round(y_preds)))

# %% [markdown]
# ### 3. original+activations_ebb_set1.csv+ deactivations_ebb_set+interaction+ival_call+network+notifying+phone+reactivation+redemption+throttling

# %% [markdown]
# ### final prediction of eval_set

# %%
#dfe.info()
dfe_test = dfe.iloc[:,1:]
dfe_test.info()

# %% [markdown]
# ### confusing, why test 0.0?

# %%
########################### cunfusing??????？ ###############################################
X_train = dfx
y_train = dfy
X_test = dfe_test
#X_train, X_test, y_train, y_test = train_test_split(dfx, dfy, test_size=0.2, random_state = 1)   
cat_features = [5,9]
NFOLDS = 5
folds = KFold(n_splits=NFOLDS)
columns = X_train.columns
#split train and validation
splits = folds.split(X_train, y_train)
y_preds = np.zeros(X_test.shape[0])
y_oof = np.zeros(X_train.shape[0])
score = 0
  
for fold_n, (train_index, valid_index) in enumerate(splits):
    X_tr, X_val = X_train[columns].iloc[train_index], X_train[columns].iloc[valid_index]
    y_tr, y_val = y_train.iloc[train_index], y_train.iloc[valid_index]    
    clf= cbt.CatBoostClassifier(iterations= 3000,learning_rate=0.1,eval_metric='F1',loss_function='CrossEntropy',verbose=500) 

    clf.fit(X_tr, y_tr,eval_set=(X_val, y_val),cat_features=cat_features,plot=True)
    y_pred_valid = clf.predict_proba(X_val)[:,1]
    y_oof[valid_index] = y_pred_valid
    
    y_preds += clf.predict_proba(X_test)[:,1]/ NFOLDS    
    del X_tr, X_val, y_tr, y_val
    gc.collect()    

#print('F1 score test', f1_score(y_test, np.round(y_preds)))
y_preds = np.round(y_preds)
print(y_preds)

# %% [markdown]
# ### iteration = 2000 ->F1  = 0.86

# %%
########################### still have train and test ###############################################
X_train, X_test, y_train, y_test = train_test_split(dfx, dfy, test_size= 0.1, random_state = 1) 
X_test = dfe_test
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
    clf= cbt.CatBoostClassifier(iterations=2000,learning_rate=0.1,eval_metric='F1',loss_function='CrossEntropy',verbose=500) 

    clf.fit(X_tr, y_tr,eval_set=(X_val, y_val),cat_features=cat_features,plot=True)
    y_pred_valid = clf.predict_proba(X_val)[:,1]
    y_oof[valid_index] = y_pred_valid
    
    y_preds += clf.predict_proba(X_test)[:,1]/ NFOLDS    
    del X_tr, X_val, y_tr, y_val
    gc.collect()    

y_preds = np.round(y_preds)
print(y_preds)

# %%
submit_prediction(y_preds, dfe)

# %% [markdown]
# ### grid search to find the best parameters
# #### https://www.projectpro.io/recipes/find-optimal-parameters-for-catboost-using-gridsearchcv-for-classification

# %%
from sklearn.model_selection import GridSearchCV

X_train, X_test, y_train, y_test = train_test_split(dfx, dfy, test_size= 0.1, random_state = 1) 
cat_features = [5,9]

model = cbt.CatBoostClassifier(verbose = 500)
grid = {'learning_rate': [0.03, 0.1],
        'depth': [4, 6, 10],
        'iterations'    : [2000,4000,6000],
        'loss_function': ['Logloss', 'CrossEntropy'],
       'cat_features' :cat_features}

Grid_CBC = GridSearchCV(estimator= model, param_grid = grid, cv = 2, n_jobs=-1)
Grid_CBC.fit(X_train, y_train)

# %% [markdown]
# ## Other ways
# ### Beginning with one-hot system, first we need to convert date
# 1. only string -> cat_feature 0.78 No
# 
# 2. how long is it from today -> int 0.803
# 
# 3. add how long is redemption from activation -> 0.809

# %%
def convert_date(df):
    today_str = '2022-4-10'
    today = datetime.strptime(today_str, "%Y-%m-%d")
    df['last_redemption']=pd.to_datetime(df['last_redemption_date'],format='%Y %m %d')
    df['first_activation']=pd.to_datetime(df['first_activation_date'],format='%Y %m %d')
    df['last_redemption'] = today-df['last_redemption']
    df['first_activation'] = today-df['first_activation']
    df['redemption_activition'] = df['first_activation']-df['last_redemption'] 
    df['last_redemption'] = df['last_redemption'].astype('timedelta64[D]')
    df['last_redemption'] = df['last_redemption'].astype(int)
    df['first_activation'] = df['first_activation'].astype('timedelta64[D]')
    df['first_activation'] = df['first_activation'].astype(int)
    df['redemption_activition'] = df['redemption_activition'].astype('timedelta64[D]')
    df['redemption_activition'] = df['redemption_activition'].astype(int)
    return df
convert_date(ebb_1)
convert_date(ebb_2)
convert_date(eval_set)

# %%
X_train, X_test, y_train, y_test = train_test_split(dfx, dfy, test_size=0.2, random_state = 1)      
NFOLDS = 5
folds = KFold(n_splits=NFOLDS)
columns = X_train.columns
splits = folds.split(X_train, y_train)
y_preds = np.zeros(X_test.shape[0])
y_oof = np.zeros(X_train.shape[0])
score = 0
cat_features = [5,7,8]  
for fold_n, (train_index, valid_index) in enumerate(splits):
    X_tr, X_val = X_train[columns].iloc[train_index], X_train[columns].iloc[valid_index]
    y_tr, y_val = y_train.iloc[train_index], y_train.iloc[valid_index]    
    clf= cbt.CatBoostClassifier(iterations=1000,learning_rate=0.1,eval_metric='F1',loss_function='CrossEntropy',verbose=100) 

    clf.fit(X_tr, y_tr,eval_set=(X_val, y_val),cat_features=cat_features,plot=True)
    y_pred_valid = clf.predict_proba(X_val)[:,1]
    y_oof[valid_index] = y_pred_valid
    
    y_preds += clf.predict_proba(X_test)[:,1]/ NFOLDS    
    del X_tr, X_val, y_tr, y_val
    gc.collect()    

print('F1 score test', f1_score(y_test, np.round(y_preds)))


# %% [markdown]
# Check the dataset:
#     onehot system;
#     if state!= null, state = 1
#     else state = -1
#     F1 score of desition tree ->0.8 not so good

# %%
X_train, X_test, Y_train, Y_test = train_test_split(dfx, dfy, test_size=0.2)
X_train=X_train.values
Y_train=Y_train.values
X_test=X_test.values
Y_test=Y_test.values
y_train=Y_train.reshape(-1)
y_test=Y_test.reshape(-1)
decision_tree=DecisionTreeClassifier(max_depth=10, criterion='gini')
decision_tree.fit(X_train,y_train)
train_report=classification_report(y_train, decision_tree.predict(X_train))
test_report=classification_report(y_test, decision_tree.predict(X_test))
print(train_report)
print(test_report)

# %%
model = cbt.CatBoostClassifier()

grid = {'learning_rate': [0.3, 0.1],
        'depth': [1, 3, 5],
        'l2_leaf_reg': [ 5, 9, 11]}

grid_search_result = model.grid_search(grid, 
                                       X=X_train, 
                                       y=y_train, 
                                       plot=True)

# %%
print('Best parameters of catboost',grid_search_result)

# %% [markdown]
# ### 5 fold CatBoostClassifier - StratifiedKShufflesplit->0.814
# Best parameters not fit better than lr=0.1 only

# %%
X_train, X_test, y_train, y_test = train_test_split(dfx, dfy, test_size=0.2, random_state = 1)    
y_oof = np.zeros(X_train.shape[0]) 
y_preds = np.zeros(X_test.shape[0])
skf = StratifiedShuffleSplit(n_splits=5, random_state=0, test_size=0.6) 
cat_features = [5,7,12]

for index, (train_index, valid_index) in enumerate(skf.split(X_train, y_train)): 
    X_tr, X_val, y_tr, y_val = \
    X_train.iloc[train_index], \
    X_train.iloc[valid_index],\
    y_train.iloc[train_index],\
    y_train.iloc[valid_index]
    cbt_model = cbt.CatBoostClassifier(iterations=1000,\
                                       learning_rate=0.1,\
                                       eval_metric='F1',\
                                       loss_function='CrossEntropy',\
                                       verbose=500) 
    #train dataset after 5-fold
    cbt_model.fit(X_tr, y_tr, eval_set=(X_val, y_val),cat_features = cat_features,plot=True) 

    prediction_cat = cbt_model.predict_proba(X_val)[:,1]
    y_oof[valid_index] = prediction_cat
    y_preds += cbt_model.predict_proba(X_test)[:,1]/ 5
    del X_tr, X_val, y_tr, y_val
    gc.collect() 

print('score', f1_score(y_test, np.round(y_preds)))

# %% [markdown]
# ## CatBoostEncoder ->opearation system+state->Random forest
# ### 11 Categorical Encoders and Benchmark https://www.kaggle.com/code/subinium/11-categorical-encoders-and-benchmark/notebook
# It says that CatBoostEncoder works bettter than one-hot, and do not need to turn nouns to numbers
# 
# Only encoding by CatBoostEncoder, after that, it takes linear regression.
# 
# ### Chinese version https://mattzheng.blog.csdn.net/article/details/107851162?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1.pc_relevant_default&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1.pc_relevant_default&utm_relevant_index=2
# It says that CatBoostEncoder performs best combining with cross validition
# 

# %% [markdown]
# Multiple values dataset ->Catboost

# %%



