# %% [markdown]
# ### import section

# %%
import sys

print(sys.version)
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch
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

from xgboost import XGBClassifier, plot_importance

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score


# %% [markdown]
# ### Constants

# %%
# FOLDER='data/'

FOLDER='../data/'


FILE1, FILE2, FILE_EVAL = 'ebb_set1.csv', 'ebb_set2.csv', 'eval_set.csv'
SUPPORT_FILE1, SUPPORT_FILE2, SUPPORT_FILE_EVAL = 'support_ebb_set1.csv', 'support_ebb_set2.csv', 'support_eval_set.csv'
INTERACTIONS_FILE1, INTERACTIONS_FILE2, INTERACTIONS_FILE_EVAL = 'interactions_ebb_set1.csv', 'interactions_ebb_set2.csv', 'interactions_eval_set.csv'
DEACTIVATIONS_FILE1, DEACTIVATIONS_FILE2, DEACTIVATION_FILE_EVAL = 'deactivations_ebb_set1.csv', 'deactivations_ebb_set2.csv', 'deactivations_eval_set.csv'

NETWORK_FILE1, NETWORK_FILE2, NETWORK_FILE_EVAL = 'network_ebb_set1.csv', 'network_ebb_set2.csv', 'network_eval_set.csv'
SUSPENSION_FILE1, SUSPENSION_FILE2, SUSPENSION_FILE_EVAL = 'suspensions_ebb_set1.csv', 'suspensions_ebb_set2.csv', 'suspensions_eval_set.csv'
PHONE_FILE1, PHONE_FILE2, PHONE_FILE_EVAL = 'phone_data_ebb_set1.csv', 'phone_data_ebb_set2.csv', 'phone_data_eval_set.csv'

LEASE_FILE1, LEASE_FILE2, LEASE_FILE_EVAL = 'lease_history_ebb_set1.csv', 'lease_history_ebb_set2.csv', 'lease_history_eval_set.csv'
LOYALTY_FILE1, LOYALTY_FILE2, LOYALTY_FILE_EVAL = 'loyalty_program_ebb_set1.csv', 'loyalty_program_ebb_set2.csv', 'loyalty_program_eval_set.csv'
NOTIFYING_FILE1, NOTIFYING_FILE2, NOTIFYING_FILE_EVAL = 'notifying_ebb_set1.csv', 'notifying_ebb_set2.csv', 'notifying_eval_set.csv'
REACTIVATIONS_FILE1, REACTIVATIONS_FILE2, REACTIVATIONS_FILE_EVAL = 'reactivations_ebb_set1.csv', 'reactivations_ebb_set2.csv', 'reactivations_eval_set.csv'
REDEMPTIONS_FILE1, REDEMPTIONS_FILE2, REDEMPTIONS_FILE_EVAL = 'redemptions_ebb_set1.csv', 'redemptions_ebb_set2.csv', 'redemptions_eval_set.csv'
# SUSPENSIONS_FILE1, SUSPENSIONS_FILE2, SUSPENSIONS_FILE_EVAL = 'suspensions_ebb_set1.csv', 'suspensions_ebb_set2.csv', 'suspensions_eval_set.csv'
THROTTLING_FILE1, THROTTLING_FILE2, THROTTLING_FILE_EVAL = 'throttling_ebb_set1.csv', 'throttling_ebb_set2.csv', 'throttling_eval_set.csv'

AUTO_FILE1, AUTO_FILE2, AUTO_FILE_EVAL = 'auto_refill_ebb_set1.csv', 'auto_refill_ebb_set2.csv', 'auto_refill_eval_set.csv'

IVR_FILE1, IVR_FILE2, IVR_FILE_EVAL = 'ivr_calls_ebb_set1.csv', 'ivr_calls_ebb_set2.csv', 'ivr_calls_eval_set.csv'


ACTIVATIONS_FILE1, ACTIVATIONS_FILE2, ACTIVATIONS_FILE_EVAL = 'activations_ebb_set1.csv', 'activations_ebb_set2.csv', 'activations_eval_set.csv'



# %% [markdown]
# Extra conversion functions

# %%

def get_os(k):
    return 'ios' in str(k).lower()

# https://www.counterpointresearch.com/global-smartphone-revenue-hits-record-450-billion-2021-apple-captures-highest-ever-share-q4-2021/

# motorola and lg are estimates couldn't find exact figures
manufacturers = {
    'samsung' : 263, 'apple': 825 , 'motorola':200, 'lg':200
}

def get_manufacturer(k):
    for m, asp in manufacturers.items():
        # print(asp)
        if m in str(k).lower(): return asp
    return 200


def get_state_income(state):
    if state in state_income.keys():
        return state_income[state]
    else:
        return 60000 # roughly us per capita income across whole country

# Source https://www.bea.gov/data/income-saving/real-personal-income-states-and-metropolitan-areas
# https://www.forbes.com/sites/andrewdepietro/2021/12/28/us-per-capita-income-by-state-in-2021/?sh=2cc69f5737be
# Real average income
state_income = {
    'TN':49955, 'AL':46963, 'MS':43284, 'GA':49392, 'FL':49853, 'KY':47551, 'LA':49483, 
    'NC':49396, 'SC':47252, 'VA':55333,'IN':50624,'IL':56482,'WY':60463,'NY':60936,
    'WV':46130,'AR':47765,'ND':60286,'TX':49945,'MO':50404,'CO':55911,
    'ID':48216,'MI':51071,'NJ':59594,'MD':56578,'NE':55891,
    'MN':56696,'SD':58414,'VT':53726,'AZ':45193,'DE':51689,
    'WA':56385,'OK':49254,'CT':68533,'WI':53798,'OH':52758,
    'MT':52054,'CA':57347,'PA':57030,'MA':65853,'ME':50516,
    'KS':54773,'HI':47234,'NV':49914,'OR':49485,'NH':58342,
    'IA':52969,'NM':45637,'UT':49388,'AK':55470,
    # 'DC':,
    # 'VI':,
    'RI':53859,
    # 'PR':
}

def mode(x):
    m = pd.Series.mode(x)
    return m.values[0] if not m.empty else np.nan



# %%
def load_data(csvs):
    return pd.concat([pd.read_csv(FOLDER + c) for c in csvs])

# %%
# Standard
df1=pd.read_csv(FOLDER + FILE1)
df2=pd.read_csv(FOLDER + FILE2)
df_eval=pd.read_csv(FOLDER +FILE_EVAL)
df=pd.concat([df1,df2])

# %%

def convert_dataset(df, isEval=False):
    df = df.reset_index()
    df['manufacturer'] = df['manufacturer'].apply(lambda x : get_manufacturer(x))
    df['state'] =df['state'].apply(lambda x : get_state_income(x))
    # df['state'] = df['state'].apply(lambda x : hash(x))
    df['operating_system'] = df['operating_system'].apply(lambda x : get_os(x))
    # cols = ['total_redemptions', 'tenure', 'number_upgrades', 'year', 'manufacturer', 'operating_system', 'state', 'total_revenues_bucket']
   
    # for c in ('opt_out_email', 'opt_out_loyalty_sms', 'opt_out_mobiles_ads', 'opt_out_loyalty_email', 'opt_out_phone'):
    #     df[c] = df[c].apply(lambda x: x ==1)

    df['marketing_comms_1'] = df['marketing_comms_1'].fillna(0)
    df['marketing_comms_2'] = df['marketing_comms_2'].fillna(0)

    df['opt_out_mobiles_ads'] = df['opt_out_mobiles_ads'].apply(lambda x: x == -1)

    cols = ['ebb_eligible', 'last_redemption_date', 'first_activation_date', 'language_preference', 'opt_out_email', 'opt_out_loyalty_sms', 'opt_out_loyalty_email', 'opt_out_phone']
    if not isEval:
        cols.append('customer_id')
    else:
        cols.remove('ebb_eligible')

    df = df.drop(cols, axis=1)

    print(df)
    
    return df


id = lambda x: x

df['ebb_eligible']=df['ebb_eligible'].apply(lambda x: 0 if pd.isnull(x) else 1)
df = df.set_index('customer_id')
df_eval = df_eval.set_index('customer_id')

def add_network_mean(df, df_n):
    df_n = df_n.drop(['date'], axis = 1)

    for t in ['total_kb', 'voice_minutes', 'total_sms', 'hotspot_kb']:
        df_kb = df_n[t].groupby(df_n['customer_id']).mean()
        df_kb = pd.DataFrame(df_kb)
        # print(df_kb)
        df = df_kb.join(df, how='right')
        df[t] = df[t].fillna(0)

    # print(*[print(k) for k in df_n['total_kb'].groupby(df_n['customer_id'])])
    return df


df = add_network_mean(df, load_data([NETWORK_FILE1, NETWORK_FILE2]))
df_eval = add_network_mean(df_eval, load_data([NETWORK_FILE_EVAL]))
# Small positive


def add_phone_data(df, df_p):
    for c, f, fill in (('data_roaming', lambda x : x == 'true', False),
                       ('language', lambda x: hash(x), 0),
                       ('memory_total', lambda x: x, 0),
                       ('bluetooth_on', lambda x: x =='true', False),
                       ('sd_storage_present', lambda x: x =='YES', 0),
                       ('data_network_type', lambda x : hash(x), 0)):
        df_r = df_p[c].groupby(df_p['customer_id']).agg(mode)
        df_r = pd.DataFrame(df_r)
        df_r[c] = df_r[c].apply(lambda x: f(x))
        
        df = df_r.join(df, how = 'right')
        df[c] = df[c].fillna(fill)

    df_a = df_p['memory_available'].groupby(df_p['customer_id']).mean()
    df_a = pd.DataFrame(df_a)
    
    df = df_a.join(df, how = 'right')
    df['memory_available'] = df['memory_available'].fillna(0)
    
    return df

df = add_phone_data(df, load_data([PHONE_FILE1, PHONE_FILE2]))
df_eval = add_phone_data(df_eval, load_data([PHONE_FILE_EVAL]))
# neglible/positive impact

def add_redemptions(df, df_r):
    df_a = df_r['revenues'].groupby(df_r['customer_id']).mean() # Mean slightly better than sum
    df_a = pd.DataFrame(df_a)
    df = df_a.join(df, how ='right')
    df['revenues'] = df['revenues'].fillna(0)
    return df
    
df = add_redemptions(df, load_data([REDEMPTIONS_FILE1, REDEMPTIONS_FILE2]))
df_eval = add_redemptions(df_eval, load_data([REDEMPTIONS_FILE_EVAL]))
# Positive difference

def add_activations(df, df_a):
    df_a = df_a['activation_channel'].groupby(df_a['customer_id']).agg(mode)
    df_a = pd.DataFrame(df_a)
    df_a['activation_channel'] = df_a['activation_channel'].apply(lambda x: hash(x))
    df = df_a.join(df, how ='right')
    df['activation_channel'] = df['activation_channel'].fillna(0)
    return df
    
df = add_activations(df, load_data([ACTIVATIONS_FILE1, ACTIVATIONS_FILE2]))
df_eval = add_activations(df_eval, load_data([ACTIVATIONS_FILE_EVAL]))
# Very Minor positive

def add_support(df, df_s):
    df_a = df_s['case_type'].groupby(df_s['customer_id']).agg(mode)
    df_a = pd.DataFrame(df_a)
    df_a['case_type'] = df_a['case_type'].apply(lambda x: hash(x))
    df = df_a.join(df, how='right')
    df['case_type'] = df['case_type'].fillna(0)
    return df

df = add_support(df, load_data([SUPPORT_FILE1, SUPPORT_FILE2]))
df_eval = add_support(df_eval, load_data([SUPPORT_FILE_EVAL]))
# Very large difference

def add_lease_history(df, df_s):
    df_a = df_s['lease_status'].groupby(df_s['customer_id']).agg(mode)
    df_a = pd.DataFrame(df_a)
    df_a['lease_status'] = df_a['lease_status'].apply(lambda x: hash(x))
    df = df_a.join(df, how='right')
    df['lease_status'] = df['lease_status'].fillna(0)
    return df

df = add_lease_history(df, load_data([LEASE_FILE1, LEASE_FILE2]))
df_eval = add_lease_history(df_eval, load_data([LEASE_FILE_EVAL]))
# Minor positive impact

def add_react(df, df_a):
    df_a = df_a['reactivation_channel'].groupby(df_a['customer_id']).agg(mode)
    df_a = pd.DataFrame(df_a)
    df_a['reactivation_channel'] = df_a['reactivation_channel'].apply(lambda x: hash(x))
    df = df_a.join(df, how ='right')
    df['reactivation_channel'] = df['reactivation_channel'].fillna(0)
    return df

df = add_react(df, load_data([REACTIVATIONS_FILE1, REACTIVATIONS_FILE2]))
df_eval = add_react(df_eval, load_data([REACTIVATIONS_FILE_EVAL]))
# Noticable positive impact

def add_auto_refill_date(df, df_a):
    df_a = df_a['auto_refill_enroll_date'].groupby(df_a['customer_id']).agg(mode)
    df_a = pd.DataFrame(df_a)
    df_a['auto_refill_enroll_date'] = df_a['auto_refill_enroll_date'].apply(lambda x: 1)
    df = df_a.join(df, how ='right')
    df['auto_refill_enroll_date'] = df['auto_refill_enroll_date'].fillna(0)
    return df

df = add_auto_refill_date(df, load_data([AUTO_FILE1, AUTO_FILE2]))
df_eval = add_auto_refill_date(df_eval, load_data([AUTO_FILE_EVAL]))
# Marginal positive

def add_loyalty(df, df_a):
    df_a = df_a['total_quantity'].groupby(df_a['customer_id']).sum()
    df_a = pd.DataFrame(df_a)
    # df_a['total_quantity'] = df_a['total_quantity'].apply(lambda x: hash(x))
    df = df_a.join(df, how ='right')
    df['total_quantity'] = df['total_quantity'].fillna(0)
    return df

df = add_loyalty(df, load_data([LOYALTY_FILE1, LOYALTY_FILE2]))
df_eval = add_loyalty(df_eval, load_data([LOYALTY_FILE_EVAL]))
# Minor positive impact

df.info()

dfx=convert_dataset(df)
print(len(dfx))
dfy=df[['ebb_eligible']]

dfe=convert_dataset(df_eval, isEval=True)


# %%
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(dfx, dfy, test_size=0.3)
X_train=X_train.values
Y_train=Y_train.values
X_test=X_test.values
Y_test=Y_test.values
y_train=Y_train.reshape(-1)
y_test=Y_test.reshape(-1)
# X_train, X_test, Y_train, Y_test, y_train, y_test

# %%
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
    
    
    models_name.append('HistGradientBoosting')
    added_model3=HistGradientBoostingClassifier(random_state=1)
    added_model3.fit(X_train, y_train)
    y_pred=added_model3.predict(X_test)
    y_pred=np.round(y_pred)
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

# %% [markdown]
# ### Decision Tree

# %%
def testDecisionTree(X_train, y_train, X_test, y_test):
    max_depths=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,25,30,35]
    models_train_precision=[]
    models_train_recall=[]
    models_train_f1=[]
    models_precision=[]
    models_recall=[]
    models_f1=[]
    models_train_report=[]
    models_report=[]
    for _ in max_depths:
        decision_tree=sklearn.tree.DecisionTreeClassifier(max_depth=_)
        decision_tree.fit(X_train,y_train)
        y_pred=decision_tree.predict(X_train)
        models_train_precision.append(sklearn.metrics.precision_score(y_train,y_pred, average='macro'))
        models_train_recall.append(sklearn.metrics.recall_score(y_train,y_pred, average='macro'))
        models_train_f1.append(sklearn.metrics.f1_score(y_train,y_pred, average='macro'))
        models_train_report.append(sklearn.metrics.classification_report(y_train, y_pred))
        y_pred=decision_tree.predict(X_test)
        models_precision.append(sklearn.metrics.precision_score(y_test,y_pred, average='macro'))
        models_recall.append(sklearn.metrics.recall_score(y_test,y_pred, average='macro'))
        models_f1.append(sklearn.metrics.f1_score(y_test,y_pred, average='macro'))
        models_report.append(sklearn.metrics.classification_report(y_test,y_pred))

    precision_max=max(models_precision)
    precision_max_depth=max_depths[models_precision.index(precision_max)]
    recall_max=max(models_recall)
    recall_max_depth=max_depths[models_recall.index(recall_max)]
    f1_max=max(models_f1)
    f1_max_depth=max_depths[models_f1.index(f1_max)]
    
    plt.clf()
    plt.figure(figsize=(15,15))
    plt.suptitle('Diffrent Max-Depth in DecisionTree')
    ax = plt.subplot(311)
    ax.set_title(f'precision, the optimal max_depth = {precision_max_depth}, the precision = {precision_max}')
    plt.plot(max_depths, models_train_precision, label='train')
    plt.plot(max_depths, models_precision, label='test')
    plt.legend()
    ax = plt.subplot(312)
    ax.set_title(f'recall, the optimal max_depth = {recall_max_depth}, the precision = {recall_max}')
    plt.plot(max_depths, models_train_recall, label='train')
    plt.plot(max_depths, models_recall, label='test')
    plt.legend()
    ax = plt.subplot(313)
    ax.set_title(f'f1-score, the optimal max_depth = {f1_max_depth}, the precision = {f1_max}')
    plt.plot(max_depths, models_train_f1, label='train')
    plt.plot(max_depths, models_f1, label='test')
    plt.legend()
    plt.show()


    print('Summary:')
    for i in range(len(max_depths)):
        print('max_depth:', max_depths[i])
        print('train report:')
        print(models_train_report[i])
        print('test report:')
        print(models_report[i])
    print('End Summary')

# %%
testModels(X_train,y_train, X_test, y_test)

# %%
testDecisionTree(X_train,y_train, X_test, y_test)

# %%

model = XGBClassifier()
model.fit(X_train, y_train)

plot_importance(model)
plt.show()

y_pre = model.predict(X_test)
label = np.round(y_pre)
print("The f1 score is ", f1_score(y_test, label))

# %%

model = HistGradientBoostingClassifier(random_state=1)
model.fit(X_train, y_train)

y_pre = model.predict(X_test)
label = np.round(y_pre)
print("The f1 score is ", f1_score(y_test, label))


# %%
def getDateString():
    return pd.Timestamp.now().strftime('%Y-%m-%d')

print(getDateString())
'''
model: The model of sklearn
eval_df: The Dataframe of eval dataset
csv_path: The path of output csv, if ignored, the file of current date .csv will be outputed.
'''
def submit(model, eval_df, csv_path = None):
    if csv_path==None:
        csv_path = getDateString() + '.csv'
    df_eval = eval_df.drop(columns=['customer_id'])
    df_csv = pd.DataFrame()
    df_csv['customer_id'] = eval_df['customer_id']
    print(df_eval)
    df_csv['ebb_eligible'] = model.predict(df_eval.values)
    df_csv = df_csv.drop_duplicates(['customer_id'])
    df_csv.to_csv(csv_path, index=False)

'''
prediction_list: The prediction list like [0,1,0,0,1]
eval_df: The Dataframe of eval dataset
csv_path: The path of output csv, if ignored, the file of current date .csv will be outputed.
'''    
def submit_prediction(prediction_list, eval_df, csv_path = None):
    if csv_path==None:
        csv_path = getDateString() + '.csv'
    df_csv = pd.DataFrame()
    df_csv['customer_id'] = eval_df['customer_id']
    df_csv['ebb_eligible'] = prediction_list
    df_csv = df_csv.drop_duplicates(['customer_id'])
    df_csv.to_csv(csv_path, index=False)

'''
prediction_list: The prediction list like [0,1,0,0,1]
customer_id: The customer id list
csv_path: The path of output csv, if ignored, the file of current date .csv will be outputed.
'''
def submit_prediction_id(prediction_list, customer_id, csv_path = None):
    if csv_path==None:
        csv_path = getDateString() + '.csv'
    df_csv['customer_id'] = customer_id
    df_csv['ebb_eligible'] = prediction_list
    df_csv = df_csv.drop_duplicates(['customer_id'])
    df_csv.to_csv(csv_path, index=False)

# %%
submit(model, dfe)


