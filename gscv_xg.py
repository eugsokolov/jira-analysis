'''
use xg + grid search to analyze JIRA data
'''

from template import *

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

xgmodel = XGBClassifier(nthread=-1, objective='reg:logistic',
            max_depth=7,
            n_estimators=80,
            learning_rate=0.1,
            reg_lambda=1)
cv_params = {
    'max_depth': [5, 7, 10],
    'min_child_weight': [5, 6, 7],
    'n_estimators': [70, 80, 90],
    'learning_rate': [0.1],
    'reg_lambda': [1, 10],
}

model = GridSearchCV(xgmodel, cv_params, scoring='accuracy', cv=5, n_jobs=-1, refit=True, )
model.fit(xtrain, ytrain)

print(model)
print("best score: {}, best params: {}".format(model.best_score_, model.best_params_))
predicted = model.predict(xtest)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(ytest, predicted))
table = classification_report(ytest, predicted, target_names=vals)
print(table)
print('Accuracy: {}'.format(accuracy_score(ytest, predicted, normalize=True)))

for i in range(len(X.columns)):
    print(X.columns[i], model.feature_importances_[i])

