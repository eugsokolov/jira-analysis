'''
template script to massage JIRA data
provides an 80/20 train/test split
'''
import sys
import datetime
import numpy

from pandas import DataFrame, Series


def getRawData():
    '''
    get SQL data, use README to download postgresql dataset
    '''
    from pandas import read_sql_query
    from sqlalchemy import create_engine

    db = 'postgresql://eugene:eugene@localhost:5432/jira_dataset'
    engine = create_engine(db)
    conn = engine.connect()

    # massage the sql data
    limit = 10000 # save some time
    #limit = int(raw_input('Enter a query limit: '))
    query = "select * from jira_issue_report as r inner join (select * from jira_issue_comment where sentiment is NOT NULL) as c on c.issue_report_id = r.id where r.resolved is not NULL and r.created is not NULL and r.assignee_id is not NULL and r.priority is not NULL and r.resolution is not NULL and r.resolved-r.created > '5 minutes'::interval"
    query = "{} limit {}".format(query, limit)

    joined = conn.execute(query)
    return read_sql_query(query, conn)


def getProcessedData(df):
    '''
    massage data and split to train and test (80/20)
    '''
    # we want to work with non zero ints
    df.assignee_id = df.assignee_id.astype(int)
    df.sentiment = df.sentiment + 1
    df.modality = df.modality + 1

    #Y = df['resolution']
    #Y = df['priority']
    realTime = df['resolved'] - df['created']
    Y = Series(len(realTime))
    # bucket into practical times
    for i in range(len(realTime)):
        days = realTime[i].days
        if days <= 1:
            Y[i] = 'day'
        elif days <= 7 and days > 1:
            Y[i] = 'week'
        elif days <= 31 and days > 7:
            Y[i] = 'month'
        elif days <= 365 and days > 31:
            Y[i] = 'year'
        elif  days > 365:
            Y[i] = 'longterm'

    # categorize values as integers
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

    # work with different X variables 
    X = DataFrame({
'resolution': tmp2,
'priority': tmp3,
'assignee': df['assignee_id'],
'reportee': df['reporter_id'],
'watchers': df['watchers'],
'votes': df['votes'],
'sentiment': df['sentiment'],
'mood': df['mood'],
'modality': df['modality'],
'politeness_confidence_level': df['politeness_confidence_level'],
'politness': tmp,
'anger': df['anger_count'],
'joy': df['joy_count'],
'love': df['love_count'],
'sadness': df['sadness_count'],
                   })

    from sklearn.model_selection import train_test_split

    vals = list(numpy.unique(Y.astype(str).values))
    a, b, c, d = train_test_split(X, Y, test_size=0.2)
    return a, b, c, d, vals


df = getRawData()
xtrain, xtest, ytrain, ytest, vals = getProcessedData(df)


