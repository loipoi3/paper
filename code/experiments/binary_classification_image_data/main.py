import pandas as pd
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from experiments.binary_classification_image_data.pipelines.mlp_with_ga import run_mlp_with_ga
from experiments.binary_classification_image_data.pipelines.classic_mlp import run_classic_mlp
from experiments.binary_classification_image_data.pipelines.one_plus_lambda_ea_with_gp import \
    run_one_plus_lambda_ea_with_gp

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def binary_image_pipeline(args) -> None:
    """
    Pipeline for binary classification of image data.

    Args:
        args: Arguments for configuring the pipeline and models.
    """
    df_train = pd.read_csv('datasets/chest_xray/train_embeddings.csv')
    df_test = pd.read_csv('datasets/chest_xray/test_embeddings.csv')

    x_train = df_train.iloc[:, :-1].values
    y_train = df_train.iloc[:, -1].values
    x_test = df_test.iloc[:, :-1].values
    y_test = df_test.iloc[:, -1].values

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    pca = PCA(n_components=0.99)
    x_train_pca = pca.fit_transform(x_train_scaled)
    x_test_pca = pca.transform(x_test_scaled)

    if "mlp_with_gd" in args.models:
        run_classic_mlp(x_train_pca, x_test_pca, y_train, y_test, args)
        print()

    if "mlp_with_spm" in args.models:
        run_mlp_with_ga(x_train_pca, x_test_pca, y_train, y_test, args)
        print()

    if "ea" in args.models:
        run_one_plus_lambda_ea_with_gp(x_train_pca, x_test_pca, y_train, y_test, args)
