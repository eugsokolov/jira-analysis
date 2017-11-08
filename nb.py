'''
use naive bayes techniques to analyze JIRA data
'''
from template import *

from sklearn import naive_bayes as nb
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

alphas = [0.5, 1, 10, 50, 100, 200, 500]
for a in alphas:
#    model = nb.MultinomialNB(fit_prior=True, alpha=a)
    model = LogisticRegression(C=a)
#model = GridSearchCV(nbmodel, cv_params, scoring='accuracy', cv=5, n_jobs=-1, refit=True, )

    model.fit(xtrain, ytrain)


    print(model)
#print("best score: {}, best params: {}".format(model.best_score_, model.best_params_))
    predicted = model.predict(xtrain)

    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    print(confusion_matrix(ytrain, predicted))
    table = classification_report(ytrain, predicted, target_names=vals)
    print(table)
    print('Accuracy: {}'.format(accuracy_score(ytrain, predicted, normalize=True)))



