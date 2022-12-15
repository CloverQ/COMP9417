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
