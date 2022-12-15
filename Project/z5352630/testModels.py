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