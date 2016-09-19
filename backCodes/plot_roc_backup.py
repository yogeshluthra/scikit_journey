

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

def plotRoc(clf, X, y, class_labels, title='Multi Class ROC', addNoisyFeature=False, testSize=0.2):
    # Binarize the output
    ybin = label_binarize(y, classes=class_labels)
    n_classes = ybin.shape[1]

    # Add noisy features to make the problem harder
    if addNoisyFeature is True:
        random_state = np.random.RandomState(0)
        n_samples, n_features = X.shape
        X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, ybin, test_size=testSize,
                                                        random_state=0)

    # Learn to predict each class against the other
    classifier = OneVsRestClassifier(clf, n_jobs=-1)
    y_score = classifier.fit(X_train, y_train).predict_proba(X_test)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    if len(class_labels) > 2:
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


        ##############################################################################
        # Plot of a ROC curve for a specific class
        # plt.figure()
        # plt.plot(fpr[2], tpr[2], label='ROC curve (area = %0.2f)' % roc_auc[2])
        # plt.plot([0, 1], [0, 1], 'k--')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Receiver operating characteristic example')
        # plt.legend(loc="lower right")
        # plt.show()


        ##############################################################################
        # Plot ROC curves for the multiclass problem

        # Compute macro-average ROC curve and ROC area

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        ##plt.figure()
        # plt.plot(fpr["micro"], tpr["micro"],
        #          label='micro-average ROC curve (area = {0:0.2f})'
        #                ''.format(roc_auc["micro"]),
        #          linewidth=2)

        # plt.plot(fpr["macro"], tpr["macro"],
        #          label='macro-average ROC curve (area = {0:0.2f})'
        #                ''.format(roc_auc["macro"]),
        #          linewidth=2)

        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], label='class {0} AUC = {1:0.2f}'
                                           ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title(title)
        plt.legend(loc="lower right")
        #plt.show()
    else:

        probas_ = classifier.fit(X_train, y_train).predict_proba(X_test)
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='AUC= %0.2f' % roc_auc)

        plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

        # mean_tpr = 0.0
        # mean_fpr = np.linspace(0, 1, 100)
        # mean_tpr += interp(mean_fpr, fpr, tpr)
        # mean_tpr[0] = 0.0
        # mean_tpr[-1] = 1.0
        # mean_auc = auc(mean_fpr, mean_tpr)
        # plt.plot(mean_fpr, mean_tpr, 'k--',
        #          label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title(title)
        plt.legend(loc="lower right")
        #plt.show()
    return plt