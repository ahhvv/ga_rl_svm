from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn import datasets
from sklearn.model_selection import train_test_split as ts
import data
# print(data.val_data)
# print(data.val_res)
import data
# kernel = 'rbf'
#gamma = 0.1
# param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]}
# clf_rbf = svm.LinearSVC()
# grid_search = GridSearchCV(clf_rbf, param_grid,  verbose=1)
# grid_search.fit(data.train_data, data.train_res)
# best_parameters = grid_search.best_estimator_.get_params()
# for para, val in list(best_parameters.items()):
#     print(para, val)
# model = svm.LinearSVC(C=best_parameters['C'])
# model.fit(data.val_data, data.val_res)

for _ in range(10):

    clf = svm.SVC(kernel='rbf', C=100, gamma=0.1)
    # clf = svm.LinearSVC(C=20)
    clf.fit(data.train_data, data.train_res)
    a = clf.score(data.val_data, data.val_res)
    print(a)