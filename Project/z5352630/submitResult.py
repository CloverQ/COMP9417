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