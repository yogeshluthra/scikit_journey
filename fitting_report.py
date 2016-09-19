import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc


def FittingReport(clf, X_test, y_test):
    print("Best parameters set found on development set:")
    print(clf.best_params_)
    print("\n")
    print("Grid scores on development set:")
    print("--------------------")
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() * 2, params))
    print("\n")
    #fig1 = plt.figure()
    #fig1.suptitle('Bias/Variance tradeoff (Optimized via GridSearch)', fontsize=12)
    plt.plot([mean_score for (params, mean_score, scores) in clf.grid_scores_],
             label="best score= %0.2f" % clf.best_score_)
    plt.xlabel('Hyperparam combinations (increment in groups) -->', fontsize=12)
    plt.ylabel('score (stratified cv)', fontsize=12)
    plt.legend(loc="lower right")
    #plt.show()
    print("The model fit on training set; scores computed on the test set.\n")
    print("Detailed classification report:")
    print("-------------------------------")
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print("-------------------------------")

