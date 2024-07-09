import pandas as pd
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
import numpy as np
from experiments.binary_classification_tabular_data.pipelines.mlp_with_ga import run_mlp_with_ga
from experiments.binary_classification_tabular_data.pipelines.classic_mlp import run_classic_mlp
from experiments.binary_classification_tabular_data.pipelines.one_plus_lambda_ea_with_gp import \
    run_one_plus_lambda_ea_with_gp
from sklearn.neighbors import LocalOutlierFactor

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def binary_tabular_pipeline(args) -> None:
    """
    Pipeline for binary classification of tabular data.

    Args:
        args: Arguments for configuring the pipeline and models.
    """
    df = pd.read_csv('datasets/pima_indians_diabetes_database/diabetes.csv')
    df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction',
        'Age']] = df[
        ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction',
         'Age']].replace(0, np.NaN)

    columns = df.columns
    columns = columns.drop("Outcome")

    def median_target(var):
        temp = df[df[var].notnull()]
        temp = temp[[var, 'Outcome']].groupby(['Outcome'])[[var]].median().reset_index()
        return temp

    for i in columns:
        median_target(i)
        df.loc[(df['Outcome'] == 0) & (df[i].isnull()), i] = median_target(i)[i][0]
        df.loc[(df['Outcome'] == 1) & (df[i].isnull()), i] = median_target(i)[i][1]

    q1 = df.Insulin.quantile(0.25)
    q3 = df.Insulin.quantile(0.75)
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    df.loc[df['Insulin'] > upper, "Insulin"] = upper

    lof = LocalOutlierFactor(n_neighbors=10)
    lof.fit_predict(df)
    df_scores = lof.negative_outlier_factor_
    threshold = np.sort(df_scores)[7]
    outlier = df_scores > threshold
    df = df[outlier]

    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    if "mlp_with_gd" in args.models:
        run_classic_mlp(x_scaled, y, args)
        print()

    if "mlp_with_spm" in args.models:
        run_mlp_with_ga(x_scaled, y, args)
        print()

    if "ea" in args.models:
        run_one_plus_lambda_ea_with_gp(x_scaled, y, args)
