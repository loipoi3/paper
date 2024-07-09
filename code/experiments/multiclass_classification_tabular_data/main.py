from typing import Any
import pandas as pd
import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler, LabelEncoder
from experiments.multiclass_classification_tabular_data.pipelines.mlp_with_ga import run_mlp_with_ga
from experiments.multiclass_classification_tabular_data.pipelines.classic_mlp import run_classic_mlp
from experiments.multiclass_classification_tabular_data.pipelines.one_plus_lambda_ea_with_gp import \
    run_one_plus_lambda_ea_with_gp

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def multiclass_tabular_pipeline(args: Any) -> None:
    """
    Execute the multiclass tabular classification pipeline based on the provided arguments.

    Args:
        args: Command-line arguments specifying the task and model hyperparameters.
    """
    df_train = pd.read_csv('datasets/human_activity_recognition_with_smartphones/train.csv')
    df_test = pd.read_csv('datasets/human_activity_recognition_with_smartphones/test.csv')

    label_encoder = LabelEncoder()
    df_train['Activity'] = label_encoder.fit_transform(df_train['Activity'])
    df_test['Activity'] = label_encoder.transform(df_test['Activity'])

    x_train = df_train.iloc[:, :-1].values
    y_train = df_train.iloc[:, -1].values
    x_test = df_test.iloc[:, :-1].values
    y_test = df_test.iloc[:, -1].values

    shuffle_index_train = np.random.permutation(len(x_train))
    x_train = x_train[shuffle_index_train]
    y_train = y_train[shuffle_index_train]

    shuffle_index_test = np.random.permutation(len(x_test))
    x_test = x_test[shuffle_index_test]
    y_test = y_test[shuffle_index_test]

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    if "mlp_with_gd" in args.models:
        run_classic_mlp(x_train_scaled, x_test_scaled, y_train, y_test, args)
        print()
    if "mlp_with_spm" in args.models:
        run_mlp_with_ga(x_train_scaled, x_test_scaled, y_train, y_test, args)
        print()
    if "ea" in args.models:
        run_one_plus_lambda_ea_with_gp(x_train_scaled, x_test_scaled, y_train, y_test, args)
