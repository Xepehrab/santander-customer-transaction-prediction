from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from prepare import prepare_santander_data


def load_prepared_data(
    train_path='train_prepared.feather',
    test_path='test_prepared.feather',
):
    """Load prepared feather files, running prepare.py first if needed."""
    if not Path(train_path).exists() or not Path(test_path).exists():
        print('Prepared data not found. Running prepare.py...')
        prepare_santander_data()

    train = pd.read_feather(train_path)
    test = pd.read_feather(test_path)
    return train, test


def get_feature_columns(df):
    """All columns except ID and target."""
    return [col for col in df.columns if col not in ['ID_code', 'target']]


def train_and_predict(train, test, n_splits=5, random_state=42):
    """Train LightGBM with cross-validation and predict on the test set."""
    features = get_feature_columns(train)

    X = train[features]
    y = train['target']
    X_test = test[features]

    oof_preds = np.zeros(len(train))
    test_preds = np.zeros(len(test))

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.05,
        'num_leaves': 64,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'verbose': -1,
        'seed': random_state,
    }

    print(f'Training LightGBM with {n_splits}-fold cross-validation...')

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = lgb.LGBMClassifier(n_estimators=2000, **params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )

        valid_preds = model.predict_proba(X_valid)[:, 1]
        oof_preds[valid_idx] = valid_preds
        test_preds += model.predict_proba(X_test)[:, 1] / n_splits

        fold_auc = roc_auc_score(y_valid, valid_preds)
        print(f'  Fold {fold} AUC: {fold_auc:.5f}')

    print(f'Overall OOF AUC: {roc_auc_score(y, oof_preds):.5f}')
    return test_preds


def save_submission(test, predictions, output_path='submission.csv'):
    """Save predictions in Kaggle submission format."""
    submission = pd.DataFrame({
        'ID_code': test['ID_code'],
        'target': predictions,
    })
    submission.to_csv(output_path, index=False)
    print(f'Saved submission to {output_path}')


def main():
    train, test = load_prepared_data()
    test_preds = train_and_predict(train, test)
    save_submission(test, test_preds)


if __name__ == '__main__':
    main()
