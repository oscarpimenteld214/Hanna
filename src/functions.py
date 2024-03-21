from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.metrics import roc_auc_score
from scipy import stats
import pickle


def get_data(data_file: str, dtypes_file: str) -> pd.DataFrame:
    """Load the data from a csv file and set the correct data types.

    Args:
        data_file (str): File path to the data
        dtypes_file (str): File path to the dictionary with the data types

    Returns:
        pd.DataFrame: Dataframe with the data
    """
    with open(dtypes_file, "rb") as fp:
        dtypes_dict = pickle.load(fp)
    date_dtypes = ["datetime64[ns]", "datetime64", "datetime"]
    parse_dates = []
    for key, value in dtypes_dict.items():
        if value in date_dtypes:
            dtypes_dict[key] = "object"
            parse_dates.append(key)
    data = pd.read_csv(
        data_file, dtype=dtypes_dict, parse_dates=parse_dates, encoding="latin"
    )
    return data


def prepare_for_modeling(
    train: pd.DataFrame, test: pd.DataFrame, variables: list[str], target: str
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Prepare the train, test sets to use in the scikit-learn objects

    Args:
        train (pd.DataFrame): Train dataset
        test (pd.DataFrame): Test dataset
        variables (list[str]): Variables to use in the model
        target (str): Variable to predict

    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]: Variables and targets of the train and test sets
    """
    X_train = train[variables]
    X_test = test[variables]
    for var in variables:
        if X_train[var].dtype.name == "object":
            X_train.loc[:, var] = X_train[var].astype("category")
            X_test.loc[:, var] = X_test[var].astype("category")
    y_train = train[target]
    y_test = test[target]

    if target in X_train.columns:
        X_train = X_train.drop(columns=target)
        X_test = X_test.drop(columns=target)

    # Convert columns to dummy variables
    X_train = pd.get_dummies(X_train, drop_first=True, dtype=float)
    X_test = pd.get_dummies(X_test, drop_first=True, dtype=float)
    return X_train, y_train, X_test, y_test


def remove_outliers(
    data: pd.DataFrame, variables: List[str], threshold: float
) -> pd.DataFrame:
    """Remove outliers from the series

    Args:
        data (pd.DataFrame): Dataframe
        variables (List[str]): Variables to remove outliers
        threshold (float): Threshold to remove outliers

    Returns:
        pd.DataFrame: Dataframe without outliers
    """
    z_scores = np.abs(stats.zscore(data[variables], nan_policy="omit"))
    condition1 = z_scores < threshold
    condition2 = np.isnan(z_scores)
    data = data[(condition1 | condition2).all(axis=1)]
    return data.reset_index(drop=True)


def feature_plot(data: pd.DataFrame, features: List, file_path: str) -> None:
    """Plot the distribution of the features in the dataset.

    Args:
        data (pd.DataFrame): dataframe
        features (List): Names of columns to plot
        file_name (str): File name to save the plot
    """
    fig, axes = plt.subplots(len(features), 1, figsize=(10, 4 * len(features)))
    for i, feature in enumerate(features):
        if data[feature].dtype.name == "category":
            data[feature].value_counts().plot(
                kind="bar", ax=axes[i], color="tab:blue", ec="black", linewidth=2
            )
            axes[i].set_xlabel(feature)
        elif data[feature].dtype.name == "period[M]":
            data[feature].value_counts().sort_index().plot(
                kind="bar", ax=axes[i], color="tab:blue", ec="black", linewidth=2
            )
            axes[i].set_xlabel(feature)
        else:
            data[feature].plot(
                kind="hist",
                ax=axes[i],
                bins=20,
                color="tab:blue",
                ec="black",
                linewidth=2,
            )
            axes[i].set_xlabel(feature)
    fig.tight_layout()
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()


def woe_iv(
    df: pd.DataFrame, target: str, bins: int = 10, event_ident: int = 0
) -> Tuple[pd.DataFrame, pd.Series]:
    #   df: pd.DataFrame, target: str, bins: int=10, event_ient:int =0
    """
    Compute the Weight of Evidence table and the Information value for all the variables
    in dataframe df.

    Parameters:
    --------------

    df: dataframe
        Usually the train set

    target: str
        Name of the column with the dependent variable

    bins: int
        Number of bins or categories in which the continues variables will be classing

    event_ident: int, float, or str
        Category label which represents the occurrence of an event. For instance, by default
        the event of paying the loan is denoted by 0

    Output

    """
    cols = df.columns
    woe_df = pd.DataFrame()
    for variable in cols[~cols.isin([target])]:
        var_dtype = df[variable].dtype.name
        df_filter = df.reset_index()[[variable, target, "index"]]

        if var_dtype not in ["object", "category", "interval", "period[M]"]:
            df_filter[variable] = pd.qcut(
                df_filter[variable], q=bins, duplicates="drop"
            )

        woe = df_filter.groupby(
            by=[variable, target], as_index=False, observed=False
        ).count()
        woe = woe.pivot(index=variable, columns=target, values="index")
        woe.columns.name = None
        woe = woe.reset_index()
        woe.insert(0, "Variable", variable)
        target_cats = df[target].unique()
        col_names = {
            variable: "Categories",
            event_ident: "N_Event",
            np.setdiff1d(target_cats, event_ident)[0]: "N_NonEvent",
        }
        woe = woe.rename(columns=col_names)
        woe.insert(2, "N_tot", woe["N_Event"] + woe["N_NonEvent"])
        woe["Perc_Event"] = woe["N_Event"] / woe["N_Event"].sum()
        woe["Perc_NonEvent"] = woe["N_NonEvent"] / woe["N_NonEvent"].sum()
        woe["Bad_Rate"] = woe["N_NonEvent"] / woe["N_tot"]
        woe["WoE"] = np.log(woe["Perc_Event"] / woe["Perc_NonEvent"])
        woe["IV"] = woe["WoE"] * (woe["Perc_Event"] - woe["Perc_NonEvent"])

        woe_df = pd.concat([woe_df, woe], ignore_index=True)
    iv_df = woe_df.groupby("Variable").IV.sum().sort_values(ascending=False)
    return woe_df, iv_df


def woe_plot(woe_table: pd.DataFrame, var_toplot: str, path: str) -> None:
    """Get the weight of evidence plot for each variable

    Args:
        woe_table (pd.DataFrame): Weight of evidence (WoE) table with variables in the column
        'Variable', the categories or bins in the column 'Categories', and WoE values in
        the column 'WoE'.
        var_toplot (str): It is the variable for which the plot is obtained
        path (str): Path to save the plot
    """
    woe_variable = woe_table[woe_table["Variable"] == var_toplot]
    x = [str(i) for i in woe_variable["Categories"].values]
    y = woe_variable["WoE"].values.tolist()
    fig = plt.figure(figsize=(12, 6))
    plt.xticks(rotation=45, ha="right")
    plt.plot(x, y, color="red", marker="o", linewidth=3)
    plt.bar(x, y, width=0.5)
    plt.xlabel(var_toplot)
    plt.ylabel("WoE")
    save_path = path + var_toplot + "_woe.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def coerse_classing(
    serie: pd.Series, left_bounds: list[float], right_bounds: list[float]
) -> pd.Series:
    # Create groups
    groups = []
    classif_list = []
    for left, right in zip(left_bounds, right_bounds):
        if left == right:
            groups.append(left)
            continue
        groups.append(pd.Interval(left=left, right=right))
    # Iterate over pandas series
    for value in serie.values:
        if np.isnan(value):
            classif_list.append(np.nan)
            continue
        # Iterate over groups to select the group for each value
        for group in groups:
            if type(group) == float:
                if value == group:
                    classif_list.append(str(group))
                continue
            if value in group:
                classif_list.append(str(group))
                continue
    final_series = pd.Series(classif_list, index=serie.index)
    categories = [str(cat) for cat in groups]
    return final_series.astype(CategoricalDtype(categories=categories, ordered=True))


def coerse_classing_results(
    data: pd.DataFrame,
    variable: str,
    left_bounds: list[float],
    right_bounds: list[float],
    target: str,
    path: str,
) -> None:
    """Performs coarse classing, computes the WoE table, and plots the WoE for the variable.

    Args:
        data (pd.DataFrame): Dataframe
        variable (str): Name of the variable to be coerced
        left_bounds (list): List of left bounds for the groups
        right_bounds (list): List of right bounds for the groups
        target (str): Name of the target variable
    """
    coerse_var = coerse_classing(data[variable], left_bounds, right_bounds)

    # Create the DataFrame with the new variable and the target to compute the WoE table
    df_VarTest_dict = {
        variable: coerse_var,
        target: data[target],
    }
    df_VarTest = pd.DataFrame(data=df_VarTest_dict)
    woe_test, _ = woe_iv(df_VarTest, target)
    path_tosave = path + "woe_" + variable + ".csv"
    woe_test.to_csv(path_tosave)

    # Plot the WoE for the specific variable
    woe_plot(woe_test, variable, path)


def auroc(X: pd.DataFrame, y: pd.Series, model: object, dataset: str) -> str:
    """Compute the AUROC for a model

    Args:
        X (pd.DataFrame): Variables
        y (pd.Series): Target
        model (object): Model object

    Returns:
        str: AUROC value in a string format
    """
    y_prob = model.predict_proba(X)[:, 1]
    AUROC = np.round(roc_auc_score(y, y_prob) * 100, 2)
    return f"- AUROC in the {dataset} set: {AUROC} %"


def score_scale(
    estimates: np.ndarray, ref_odds: int, ref_score: int, pdo: int, beta_0: float
) -> List[float]:
    """
    Creates the scores by scaling the log(odds), which is the output of the
    logistic regression. It uses a linear tranformation to convert each
    estimate (beta) to a score.

    Args:
        estimates (List[floats]): Parameters from logistic regression.
        ref_odds (int): The odds the client want to have at the reference score.
        ref_score (int): The score the client wants to have at the reference odds.
        pdo (int): The points that increments the score when the odds doubles.
        beta_0 (float): Intercept of the logistic regression.

    Returns:
        scores (List[floats]): The scores for each parameter.
    """
    factor = pdo / np.log(2)
    offset = ref_score - factor * np.log(ref_odds)
    n = len(estimates)
    scores = [-round(float(beta * factor)) for beta in estimates]
    score_intercept = offset - factor * beta_0
    scores.append(round(float(score_intercept)))
    return scores


def get_scores_table(
    data_filtered: pd.DataFrame, model: object, ref_odds: int, ref_score: int, pdo: int
) -> pd.DataFrame:
    scores_table = pd.DataFrame()

    def var_cats(x: str, var_names: List[str], result: str) -> str:
        """Takes the variable name from logistic regression in the format 'varname_category', adn extract the variable name or the category depending on the result parameter.

        Args:
            x (str): variable name from logistic regression
            var_names (List[str]): variables used in to train the logistic regression model
            result (str): 'variable' or 'category' to be extracted from the variable name

        Returns:
            str: the variable or the category
        """
        for var in var_names:
            if var in x:
                if result == "variable":
                    return var
                else:
                    return x.replace(var + "_", "")
            else:
                continue

    var_list = data_filtered.columns.to_list()
    X_train_withfirst = pd.get_dummies(data_filtered, drop_first=False)
    scores_table["Varname"] = X_train_withfirst.columns.values
    # Extract the variable name
    scores_table["Variable"] = scores_table["Varname"].apply(
        var_cats,
        args=(
            var_list,
            "variable",
        ),
    )
    # Extract the category name
    scores_table["Category"] = scores_table["Varname"].apply(
        var_cats,
        args=(
            var_list,
            "category",
        ),
    )
    # Estimates (coefficients)
    params = model.coef_[0]
    features = model.feature_names_in_
    par_dict = {feature: par for feature, par in zip(features, params)}
    scores_table["Estimates"] = scores_table["Varname"].replace(par_dict)
    scores_table["Estimates"] = scores_table["Estimates"].replace(
        to_replace=r"_", value=0, regex=True
    )

    estimates_tot = scores_table["Estimates"].values
    score_scales = score_scale(
        estimates_tot, ref_odds, ref_score, pdo, model.intercept_
    )

    # drop the 'varname' column
    scores_table = scores_table.drop(columns=["Varname"])

    # Add a new row for the intercept
    scores_table.loc[len(scores_table)] = [
        "Intercept",
        "Intercept",
        float(model.intercept_),
    ]

    # Add a new column with the new_scale and name it score
    scores_table["Score"] = score_scales
    return scores_table

def get_scores_distribution(X: pd.DataFrame, y: pd.Series, score_table: pd.DataFrame, bins: int, path: str) -> pd.DataFrame:
    """Get the scores distribution plot and table

    Args:
        X (pd.DataFrame): Dataframe with the filtered data
        y (pd.Series): Target variable
        score_table (pd.DataFrame): Dataframe with the scores
        bins (int): Number of bins to use in the bar plot
        path (str): Path to save the plot

    Returns:
        pd.DataFrame: Dataframe with the distribution of the scores
    """
    X_train_withfirst = pd.get_dummies(X, drop_first=False)
    X_train_withfirst["Intercept"] = 1

    X_train_scores = (X_train_withfirst * score_table["Score"].values).sum(axis=1)

    # Segment X_train_scores into the given number of bins
    scorebands = pd.cut(X_train_scores, bins)
    bins_dict = {cat: i for cat, i in zip(scorebands.cat.categories, range(1, bins + 1))}
    scorebands = scorebands.replace(bins_dict).cat.set_categories(
        [i for i in range(1, bins + 1)], ordered=True
    )

    # Create a bar plot
    plt.figure(figsize=(10, 6))
    scorebands.value_counts(normalize=True).sort_index().plot.bar()
    plt.xlabel("Score Bins")
    plt.ylabel("Count")
    plt.title("Distribution of Scores")

    # Create line plot with the bad rate
    score_bad_rate = pd.DataFrame(scorebands, columns=["Score_bins"])
    score_bad_rate["GB_predict"] = y
    tot_perband = scorebands.value_counts().sort_index()
    tot_bads = (
        score_bad_rate[score_bad_rate["GB_predict"] == 1]["Score_bins"]
        .value_counts()
        .sort_index()
    )
    bad_rate = tot_bads / tot_perband
    print(bad_rate)
    bad_rate.plot.line(color="red", marker="o", linewidth=3, label="Bad Rate")
    plt.legend()

    plt.savefig(path, bbox_inches="tight")
    plt.close()
