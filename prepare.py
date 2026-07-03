import numpy as np
import pandas as pd


def prepare_santander_data(train_path='train.csv', test_path='test.csv'):
    """
    Prepare the Santander Customer Transaction dataset for modeling.

    Steps:
    1. Load the train and test datasets.
    2. Identify 'real' rows in the test set using the unique-value trick
       discovered during the Kaggle competition.
    3. Create frequency-encoding features based on the combined train data
       and real test data.
    4. Assign a frequency of 1 to synthetic test rows.
    5. Save the prepared datasets in Feather format for fast loading.

    Frequency Encoding:
        For each feature value, create a new feature containing the number
        of times that value appears in the dataset.
    """

    print("Loading data...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # Keep only the actual feature columns
    features = [
        col for col in train.columns
        if col not in ['ID_code', 'target']
    ]

    print("Identifying real test rows...")

    # Convert feature columns to a NumPy array for faster processing
    df_test = test[features].values

    # Stores how many times each value appears in its column
    unique_count = np.zeros_like(df_test)

    # Count occurrences of every value in every feature column
    for i in range(df_test.shape[1]):
        _, index, count = np.unique(
            df_test[:, i],
            return_inverse=True,
            return_counts=True
        )

        unique_count[:, i] = count[index]

    # A row is considered "real" if at least one of its feature values
    # appears only once in the test set
    real_test_idx = np.argwhere(
        np.sum(unique_count == 1, axis=1) > 0
    ).reshape(-1)

    real_test = test.iloc[real_test_idx]

    print("Creating frequency features...")

    # Rows not identified as real are treated as synthetic
    fake_test_idx = np.setdiff1d(
        np.arange(len(test)),
        real_test_idx
    )

    # Build all frequency maps from train + real test data
    combined = pd.concat([train[features], real_test[features]], ignore_index=True)
    freq_maps = {col: combined[col].value_counts() for col in features}

    # Add all freq columns at once (avoids DataFrame fragmentation)
    train_freq = pd.DataFrame(
        {f'{col}_freq': train[col].map(freq_maps[col]) for col in features}
    )
    test_freq = pd.DataFrame(
        {f'{col}_freq': test[col].map(freq_maps[col]).fillna(1) for col in features}
    )
    test_freq.iloc[fake_test_idx] = 1

    train = pd.concat([train, train_freq], axis=1)
    test = pd.concat([test, test_freq], axis=1)

    print("Saving prepared data...")

    # Feather format loads much faster than CSV
    train.to_feather('train_prepared.feather')
    test.to_feather('test_prepared.feather')

    print("Done!")


if __name__ == "__main__":
    prepare_santander_data()