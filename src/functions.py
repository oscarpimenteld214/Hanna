from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.api.types import CategoricalDtype

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
            print(variable)
            df_filter[variable] = pd.qcut(df_filter[variable], q=bins, duplicates="drop")

        woe = df_filter.groupby(by=[variable, target], as_index=False).count()
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
