# %% [markdown]
# ### import section

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
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
from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
# import torch

# %% [markdown]
# ### Constants

# %%
FOLDER='../data/'
FILE1=FOLDER+'ebb_set1.csv'
FILE2=FOLDER+'ebb_set2.csv'
FILE_EVAL=FOLDER+'eval_set.csv'

# %% [markdown]
# ### Data Prepration

# %%
def read_data(file_prefix, columns=[], rename_map={}):
    file_prefix = FOLDER + file_prefix
    tdf1=pd.read_csv(file_prefix+'_ebb_set1.csv')
    tdf2=pd.read_csv(file_prefix+'_ebb_set2.csv')
    tdf=pd.concat([tdf1,tdf2])
    tdf_eval=pd.read_csv(file_prefix+'_eval_set.csv')
    if len(columns)>0:
        if 'customer_id' not in columns:
            columns.append('customer_id')
        tdf=tdf[columns]
        tdf_eval=tdf_eval[columns]
    if len(rename_map)>0:
        tdf.rename(columns=rename_map)
        tdf_eval.rename(columns=rename_map)
    tdf=tdf.drop_duplicates()
    tdf_eval=tdf_eval.drop_duplicates()
    # df=df.merge(tdf, on='customer_id', how='left')
    # df_eval=df_eval.merge(tdf_eval, on='customer_id', how='left')
    return tdf, tdf_eval


def add_data(df, df_eval, tdf, tdf_eval):
    df=df.merge(tdf, on='customer_id', how='left', )
    df_eval=df_eval.merge(tdf_eval, on='customer_id', how='left')
    return df, df_eval


# %%
def convert_string_to_integer_index(x: str, l: list):
    if x not in l:
        l.append(x)
    return l.index(x)

def one_hot_interger_encode(x: str, l: list):
    xs=x.split(',')
    sum=0
    for _ in xs:
        if _ not in l:
            l.append(_)
        sum+=2**l.index(_)
    return sum

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

def clean_manufacturer(x):
    if pd.isna(x):
        return 'Not Known'
    t=x.lower()
    if re.search(r'\bapple\b', t):
        return 'APPLE'
    elif re.search(r'\bsamsumg\b', t):
        return 'SAMMSUNG'
    elif re.search(r'\blg\b', t):
        return 'LG'
    elif re.search(r'\bmotorola\b', t):
        return 'MOTOROLA'
    elif t.find('not known')!=-1:
        return 'Not Known'
    return 'Other'


# %%
def preparation_convert_string_to_integer_index(df, columns):
    for _ in columns:
        l=[]
        df[_]=df[_].apply(convert_string_to_integer_index, args=(l,))
    return df

def preparation_one_hot_integer_encode(df, columns):
    for _ in columns:
        l=[]
        df[_]=df[_].apply(one_hot_interger_encode, args=(l,))
    return df

def preparation_convert_date_to_days(df, columns):
    for _ in columns:
        df[_]=pd.to_datetime(df[_])
        df[_]=((df[_]-pd.Timestamp('1970-1-1'))/pd.Timedelta(days=1)).fillna(0).astype(int)
    return df

def preparation_fillna_zero(df, columns):
    for _ in columns:
        df[_]=df[_].fillna(0)
    return df

# %%
df1=pd.read_csv(FILE1)
df2=pd.read_csv(FILE2)
df_eval=pd.read_csv(FILE_EVAL)
df=pd.concat([df1,df2])
df.info()

# tdf, tdf_eval = read_data('activations', ['activation_date'])
# tdf=tdf.drop_duplicates('customer_id')
# tdf_eval=tdf_eval.drop_duplicates('customer_id')
# df, df_eval = add_data(df, df_eval, tdf, tdf_eval)

# tdf, tdf_eval = read_data('activations', ['activation_channel'])
# tdf=tdf.groupby(by='customer_id', as_index=False)['activation_channel'].apply(lambda x: ','.join(x))
# tdf=preparation_one_hot_integer_encode(tdf, ['activation_channel'])
# tdf_eval=tdf_eval.groupby(by='customer_id', as_index=False)['activation_channel'].apply(lambda x: ','.join(x))
# tdf_eval=preparation_one_hot_integer_encode(tdf_eval, ['activation_channel'])
# df, df_eval = add_data(df, df_eval, tdf, tdf_eval)

# tdf, tdf_eval = read_data('deactivations', ['deactivation_reason'])
# tdf=tdf.groupby(by='customer_id', as_index=False)['deactivation_reason'].apply(lambda x: ','.join(x))
# tdf=preparation_one_hot_integer_encode(tdf, ['deactivation_reason'])
# tdf_eval=tdf_eval.groupby(by='customer_id', as_index=False)['deactivation_reason'].apply(lambda x: ','.join(x))
# tdf_eval=preparation_one_hot_integer_encode(tdf_eval, ['deactivation_reason'])
# df, df_eval = add_data(df, df_eval, tdf, tdf_eval)

# tdf, tdf_eval = read_data('loyalty_program', ['total_quantity'])
# df, df_eval = add_data(df, df_eval, tdf, tdf_eval)
# df=df.drop_duplicates('customer_id')
# df_eval=df_eval.drop_duplicates('customer_id')
# df.info()
# df_eval.info()

# %%
def convert_dataset(df, isEval=False):
    df['manufacturer']=df['manufacturer'].apply(clean_manufacturer)
    df['operating_system']=df['operating_system'].apply(clean_operating_system)
    date_columns=['last_redemption_date', 'first_activation_date']
    preparation_convert_date_to_days(df, date_columns)
    string_to_int_columns=['manufacturer', 'operating_system', 'state']
    preparation_convert_string_to_integer_index(df, string_to_int_columns)
    # fillna_zero_columns=['total_quantity']
    # preparation_fillna_zero(df, fillna_zero_columns)
    df['date1_diff']=df['last_redemption_date']-df['first_activation_date']
    columns=['total_redemptions','tenure','number_upgrades','year','total_revenues_bucket','operating_system','manufacturer','state', 'last_redemption_date','first_activation_date', 'date1_diff']
    if isEval:
        columns.insert(0, 'customer_id')
    return df[columns]

df['ebb_eligible']=df['ebb_eligible'].apply(lambda x: 0 if pd.isnull(x) else 1)
dfx=convert_dataset(df)
dfy=df[['ebb_eligible']]
dfe=convert_dataset(df_eval, True)

df.info()
df

# %%
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(dfx, dfy, test_size=0.3)
X_train.info()
X_test.info()
X_train=X_train.values
Y_train=Y_train.values
X_test=X_test.values
Y_test=Y_test.values
y_train=Y_train.reshape(-1)
y_test=Y_test.reshape(-1)
# X_train, X_test, Y_train, Y_test, y_train, y_test

# %% [markdown]
# ### Test and Plot Different Models (For Report)

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
testModels(X_train, y_train, X_test, y_test)
testDecisionTree(X_train, y_train, X_test, y_test)

# %% [markdown]
# ### Submission

# %%
# def getDateString():
#     return pd.Timestamp.now().strftime('%Y-%m-%d')
# '''
# model: The model of sklearn
# eval_df: The Dataframe of eval dataset
# csv_path: The path of output csv, if ignored, the file of current date .csv will be outputed.
# '''
# def submit(model, eval_df, csv_path = None):
#     if csv_path==None:
#         csv_path = getDateString() + '.csv'
#     df_eval = eval_df.drop(columns=['customer_id'])
#     df_csv = pd.DataFrame()
#     df_csv['customer_id'] = eval_df['customer_id']
#     df_csv['ebb_eligible'] = model.predict(df_eval.values)
#     df_csv = df_csv.drop_duplicates(['customer_id'])
#     df_csv.to_csv(csv_path, index=False)

# '''
# prediction_list: The prediction list like [0,1,0,0,1]
# eval_df: The Dataframe of eval dataset
# csv_path: The path of output csv, if ignored, the file of current date .csv will be outputed.
# '''    
# def submit_prediction(prediction_list, eval_df, csv_path = None):
#     if csv_path==None:
#         csv_path = getDateString() + '.csv'
#     df_csv = pd.DataFrame()
#     df_csv['customer_id'] = eval_df['customer_id']
#     df_csv['ebb_eligible'] = prediction_list
#     df_csv = df_csv.drop_duplicates(['customer_id'])
#     df_csv.to_csv(csv_path, index=False)

# '''
# prediction_list: The prediction list like [0,1,0,0,1]
# customer_id: The customer id list
# csv_path: The path of output csv, if ignored, the file of current date .csv will be outputed.
# '''
# def submit_prediction_id(prediction_list, customer_id, csv_path = None):
#     if csv_path==None:
#         csv_path = getDateString() + '.csv'
#     df_csv = pd.DataFrame()
#     df_csv['customer_id'] = customer_id
#     df_csv['ebb_eligible'] = prediction_list
#     df_csv = df_csv.drop_duplicates(['customer_id'])
#     df_csv.to_csv(csv_path, index=False)

# %%
# submit(decision_tree, dfe)


