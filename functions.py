from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def feature_plot(data: pd.DataFrame, features: List, file_name: str) -> None:
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
    save_path = "images/portfolio_analysis/" + file_name
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

# Computing the WOE and IV tables to select variables
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
            df_filter[variable] = pd.qcut(df_filter[variable], bins, duplicates="drop")

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

def woe_plot(woe_table: pd.DataFrame, var_toplot: str) -> None:
    """Get the weight of evidence plot for each variable

    Args:
        woe_table (pd.DataFrame): Weight of evidence (WoE) table with variables in the column
        'Variable', the categories or bins in the column 'Categories', and WoE values in
        the column 'WoE'.
        var_toplot (str): It is the variable for which the plot is obtained
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
    save_path = "images/univariate_analysis/" + str(var_toplot) + "_woe.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()