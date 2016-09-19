from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from fitting_report import FittingReport


def gridSearch(classifier, parDict, X, y, scoreMethod=None):
    X_train, X_test, y_train, y_test=cross_validation.train_test_split(X,y, test_size=0.2, random_state=0)
    clf=GridSearchCV(estimator=classifier, param_grid=parDict, cv=5, scoring=scoreMethod)
    clf.fit(X_train, y_train)
    FittingReport(clf, X_test, y_test)

