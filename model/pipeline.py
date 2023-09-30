import datetime

import dill
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

import numpy as np


def categorical_emissions(data):
    copy_data = data.copy()
    for name, column in copy_data.iteritems():
        seria = column.value_counts() / column.value_counts().sum() * 100
        keep_cat = seria[seria > 5].index
        copy_data[name] = np.where(column.isin(keep_cat), column, 'other')
    return copy_data


def main():
    print('Loan Prediction Pipeline')
    data_sessions = pd.read_csv('data/ga_sessions.csv', low_memory=False)
    data_sessions_utm_device_geo = data_sessions.iloc[
                                   :,
                                   lambda data_sessions:
                                   data_sessions.columns.str.contains('utm|device|geo|session_id', case=False)]

    data_hits = pd.read_csv('data/ga_hits.csv')
    df = data_hits[['session_id', 'event_action']].copy()

    targets_1 = ['sub_car_claim_click', 'sub_car_claim_submit_click',
                 'sub_open_dialog_click', 'sub_custom_question_submit_click',
                 'sub_call_number_click', 'sub_callback_submit_click', 'sub_submit_success',
                 'sub_car_request_submit_click']

    df['event_action'] = df['event_action'].apply(lambda x: 1 if x in targets_1 else 0)

    df_group = df.groupby(['session_id']).agg({'event_action': ['max']}).reset_index()
    df_group.columns = ['_'.join(col).rstrip('_') for col in df_group.columns.values]

    df_merged = pd.merge(data_sessions_utm_device_geo, df_group, how='left', on='session_id')
    df_merged = df_merged.rename(columns={'event_action_max': 'target'})
    df_merged.drop(['session_id'], axis=1, inplace=True)
    df_merged.target = df_merged.target.fillna(0)
    df_merged.target = df_merged.target.astype('int')

    x = df_merged.drop('target', axis=1)
    x_data = categorical_emissions(x)
    y = df_merged.target

    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('categorical', categorical_transformer, make_column_selector(dtype_include=object))
        ])

    models = [
        {'model': DecisionTreeClassifier(),
         'param_grid': {
             'model__criterion': ['gini', 'entropy'],
             'model__max_features': [20, 30],
             'model__class_weight': ['balanced'],
             'model__random_state': [42]}
         },
        {'model': RandomForestClassifier(),
         'param_grid': {
             'model__n_estimators': [10, 11],
             'model__max_depth': [3, 4],
             'model__class_weight': ['balanced'],
             'model__random_state': [42]}
         }
    ]

    Xtrain, Xtest, ytrain, ytest = train_test_split(x_data, y, train_size=0.7, random_state=42)

    best_score = 0
    best_pipe = None
    best_model = None
    for model in models:
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model['model'])
        ])

        grid_search = GridSearchCV(estimator=pipe, param_grid=model['param_grid'], cv=3, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(Xtrain, ytrain)

        print('model:', model['model'])
        print('best score:', grid_search.best_score_)
        print('best params:', grid_search.best_params_)
        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_pipe = grid_search
            if __name__ == '__main__':
                best_model = str(type(grid_search.best_estimator_.named_steps['model']))

    pred = best_pipe.predict_proba(Xtest)[:, 1]
    best_roc_auc = roc_auc_score(ytest, pred)
    print(f'best model: {best_model}, roc_auc: {best_roc_auc:.4f}')
    best_pipe.fit(x_data, y)

    object_to_dump = {
        'model': best_pipe,
        'meta': {
            'name': 'target action prediction pipeline',
            'author': 'Maria',
            'version': 1,
            'date': datetime.datetime.now(),
            'type': best_model,
            'roc_auc_score': best_roc_auc
        }
    }

    with open('target_action_pipe.pkl', 'wb') as file:
        dill.dump(object_to_dump, file)


if __name__ == '__main__':
    main()
