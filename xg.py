'''
use boosting techniques to analyze JIRA data
'''
from template import *

from xgboost import XGBClassifier, plot_importance

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

model = xgmodel
a = model.fit(xtrain, ytrain)

print(model)
predicted = model.predict(xtest)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(ytest, predicted))
table = classification_report(ytest, predicted, target_names=vals)
print(table)
print('Accuracy: {}'.format(accuracy_score(ytest, predicted, normalize=True)))

