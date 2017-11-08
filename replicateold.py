'''
starter script to play with data and attempt to replicate
results in published papers

'''
import sys
import datetime
import numpy
from pandas import DataFrame, Series, read_sql_query
from sqlalchemy import create_engine

limit = 10000 # save some time

db = 'postgresql://eugene:eugene@localhost:5432/jira_dataset'
engine = create_engine(db)
conn = engine.connect()
query = 'select * from jira_issue_report as r inner join (select * from jira_issue_comment where sentiment is NOT NULL) as c on c.issue_report_id = r.id where r.resolved is not NULL and r.created is not NULL and r.assignee_id is not NULL and r.priority is not NULL and r.resolution is not NULL limit {}'.format(limit)
joined = conn.execute(query)
df = read_sql_query(query, conn)

from sklearn.model_selection import train_test_split

df.assignee_id = df.assignee_id.astype(str)
# don't play with null values
df.sentiment = df.sentiment + 1
df.modality = df.modality + 1

# bin time as short/long
realTime = df['resolved'] - df['created']
Y = Series(len(realTime))
for i in range(len(realTime)):
    days = realTime[i].days
    if days <= 100:
        Y[i] = 'short'
    elif  days > 100:
        Y[i] = 'longterm'

tmp = Series(len(df['politeness']))
for i in range(len(df['politeness'])):
    if df['politeness'][i] == 'POLITE':
        tmp[i] = 1
    else:
        tmp[i] = 0

tmp2 = Series(len(df['resolution']))
for i in range(len(df['resolution'])):
    if df['resolution'][i] == 'Cannot Reproduce':
        tmp2[i] = 0
    elif df['resolution'][i] == 'Duplicate':
        tmp2[i] = 1
    elif df['resolution'][i] == 'Fixed':
        tmp2[i] = 2
    elif df['resolution'][i] == 'Incomplete':
        tmp2[i] = 3
    elif df['resolution'][i] == 'Out of Date':
        tmp2[i] = 4
    elif df['resolution'][i] == 'Rejected':
        tmp2[i] = 5
    elif df['resolution'][i] == 'Won\'t Fix':
        tmp2[i] = 6

tmp3 = Series(len(df['priority']))
for i in range(len(df['priority'])):
    if df['priority'][i] == 'Major':
        tmp3[i] = 0
    elif df['priority'][i] == 'Blocker':
        tmp3[i] = 1
    elif df['priority'][i] == 'Minor':
        tmp3[i] = 2
    elif df['priority'][i] == 'Trivial':
        tmp3[i] = 3
    elif df['priority'][i] == 'Critical':
        tmp3[i] = 4

X = DataFrame({'resolution': tmp2,
                'priority': tmp3,
                'assignee': df['assignee_id'].astype(int),
                'reportee': df['reporter_id'],
                'watchers': df['watchers'],
                'votes': df['votes'],
                'sentiment': df['sentiment'],
               'mood': df['mood'],
               'modality': df['modality'],
               'politeness_confidence_level': df['politeness_confidence_level'],
               'politness': tmp
               })


xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2)
vals = list(numpy.unique(Y.values))

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

#model = SVC(C=1, kernel='rbf')
model = GridSearchCV(LogisticRegression(), {'C': [0.1, 0.5, 1, 5, 10]}, scoring='accuracy', cv=5, n_jobs=-1,refit=True)
a = model.fit(xtrain, ytrain)

print(model)
predicted = model.predict(xtest)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(ytest, predicted))
table = classification_report(ytest, predicted, target_names=vals)
print(table)
print('Accuracy: {}'.format(accuracy_score(ytest, predicted, normalize=True)))

for i in range(len(X.columns)):
    print(X.columns[i], model.feature_importances_[i])

