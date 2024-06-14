#!/usr/bin/env python3
"""
Tune model hyperparameters
"""

import argparse
import pandas as pd
import sys
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from hyperopt import fmin, tpe, hp, STATUS_OK


def minimize(objective, space, max_evals):

    # Perform the optimization
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=max_evals)

    return best


def main(args):

    # Show args
    if args.verbose:
        print(args, file=sys.stderr)

    if args.verbose:
        print(f"Reading {args.input_filename}", file=sys.stderr)

    df = pd.read_csv(args.input_filename)

    # Sort the data by dataset ID
    df = df.sort_values(by=['dataset_id'])

    if args.verbose:
        print(df, file=sys.stderr)

    # Get rid of dataset_id column
    df = df.loc[:, df.columns != 'dataset_id']

    # Separate into labels, features
    y = df['label']

    if args.verbose:
        print(y, file=sys.stderr)

    x = df.loc[:, df.columns != 'label']

    if args.verbose:
        print(x, file=sys.stderr)

    # The rows are already in a random order, except we have sorted by
    # dataset_id. Therefore, if we don't shuffle, and we do a k-fold
    # split, the splits will (mostly) not cross dataset IDs.
    test_size = 0.2
    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size=test_size,
                                                        shuffle=False)
    if args.verbose:
        print('x_train', file=sys.stderr)
        print(x_train, file=sys.stderr)
        print('x_test', file=sys.stderr)
        print(x_test, file=sys.stderr)
        print('y_train', file=sys.stderr)
        print(y_train, file=sys.stderr)
        print('y_test', file=sys.stderr)
        print(y_test, file=sys.stderr)

    if args.verbose:
        print(f"Searching parameter space...", file=sys.stderr)

    # Define the objective function to minimize
    def objective(p):
        xgb_model = xgb.XGBClassifier(**p)
        xgb_model.set_params(device="cuda")
        xgb_model.fit(x_train, y_train)
        y_pred = xgb_model.predict(x_test)
        score = accuracy_score(y_test, y_pred)
        return {'loss': -score, 'status': STATUS_OK}

    # Group 0
    space = {'objective': 'multi:softmax',
             'num_class': 3,
             'seed': 123}
    max_evals = 100

    # Group 1:
    max_depth_choices = range(3, 10, 1)
    space['max_depth'] = hp.choice('max_depth',
                                   max_depth_choices)
    min_child_weight_choices = range(1, 6, 1)
    space['min_child_weight'] = hp.choice('min_child_weight',
                                          min_child_weight_choices)

    if args.verbose:
        print(f"Tuning group 1", file=sys.stderr)

    best = minimize(objective, space, max_evals)
    print('best:', best)

    space['max_depth'] = max_depth_choices[
        best['max_depth']]
    space['min_child_weight'] = min_child_weight_choices[
        best['min_child_weight']]

    space['max_depth'] = 4
    space['min_child_weight'] = 4

    if args.verbose:
        print(f"Tuning group 2", file=sys.stderr)

    # Group 2:
    space['gamma'] = hp.uniform('gamma', 0.0, 1.0)
    best = minimize(objective, space, max_evals)
    print('best:', best)
    space['gamma'] = best['gamma']

    if args.verbose:
        print(f"Tuning group 3", file=sys.stderr)

    # Group 3:
    space['subsample'] = hp.uniform('subsample', 0.5, 1)
    space['colsample_bytree'] = hp.uniform('colsample_bytree', 0.5, 1)
    best = minimize(objective, space, max_evals)
    print('best:', best)
    space['subsample'] = best['subsample']
    space['colsample_bytree'] = best['colsample_bytree']

    if args.verbose:
        print(f"Tuning group 4", file=sys.stderr)

    # Group 4:
    n_estimators_choices = range(50, 150, 10)
    space['learning_rate'] = hp.uniform('learning_rate', 0, 1)
    space['n_estimators'] = hp.choice('n_estimators',
                                      n_estimators_choices)
    best = minimize(objective, space, max_evals)
    print('best:', best)
    space['learning_rate'] = best['learning_rate']
    space['n_estimators'] = n_estimators_choices[best['n_estimators']]

    # Done
    print('Best hyper-parameters:')
    print(space)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Hype parameter tuner')
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Show verbose output')
    parser.add_argument(
        'input_filename',
        type=str,
        help='Feature/label CSV filename')

    args = parser.parse_args()

    main(args)
