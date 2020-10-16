#!/Users/michellean/workspace/courses/bda696/src/BDA696/.venv/bin/python3

import os
import statistics
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def bool_check(column):
    uniques = set()
    for ea in column:
        uniques.add(ea)
    if len(uniques) == 2:
        return True
    else:
        return False


def make_clickable(val):
    return '<a href="{}">{}</a>'.format(val, val)


def plot_msd(table, feature, response):
    pop_mean = table["population mean"][1]
    x = [xx for xx in table["median"].to_list() if xx != 0]
    c = [xx for xx in table["count"].to_list() if xx != 0]
    y = [yy for yy in table["bin mean"].to_list() if yy != 0]
    plt = [
        go.Scatter(x=x, y=y, name="bin mean"),
        go.Bar(x=x, y=c, yaxis="y2", name="population", opacity=0.5),
        go.Scatter(
            x=[min(x), max(x)],
            y=[pop_mean, pop_mean],
            name="population mean",
            mode="lines",
        ),
    ]
    layout = go.Layout(
        title="Binned Response Mean vs Population Mean",
        xaxis_title=f"predictor: {feature}",
        yaxis_title=f"response: {response}",
        yaxis2=dict(title="Population", overlaying="y", anchor="y3", side="right"),
    )
    fig = go.Figure(data=plt, layout=layout)
    fig_loc = f"html/{feature}_msd_plot.html"
    fig.write_html(
        file=fig_loc,
        include_plotlyjs="cdn",
    )
    fig.show()
    return fig_loc


def html_write(table, feature, weighting):
    table = add_average_row(table)
    print(f"\n{feature}\n{table}")
    html = table.to_html()
    file_loc = f"html/{feature}_{weighting}_table.html"
    text_file = open(file_loc, "w")
    text_file.write(html)
    text_file.close()
    return file_loc


def add_average_row(table):
    columns = table.columns.to_list()
    msd_list = table[columns[-1]].to_list()
    msd_wo_zeros = [a for a in msd_list if a != 0]
    msd_average = statistics.mean(msd_wo_zeros)
    values = {x: [""] for x in columns}
    values[columns[-1]] = msd_average
    new_row = pd.DataFrame.from_dict(values, orient="columns")
    table = table.append(new_row, ignore_index=True)
    return table


def main():
    # create path for html output
    if not os.path.exists("html"):
        os.makedirs("html")

    # import data to pandas
    data = load_wine()
    input_df = pd.DataFrame(data.data)

    # find which columns are predictors and which is response
    cols = input_df.columns.to_list()
    check = False
    while not check:
        response = input(f"Which column is the response? \n {cols}? \n")
        if response in cols:
            check = True
        elif int(response) in cols:
            response = int(response)
            check = True
        else:
            print("Incorrect user input.")
    else:
        response = 1
    predictors = [x for x in cols if x != response]

    # determine which columns are categorical and which are continuous
    bool_dict = {response: bool_check(input_df[response])}
    plot_dict = {}
    for predictor in predictors:
        bool_dict[predictor] = bool_check(input_df[predictor])

    # generate plots if response is categorical
    if bool_dict[response]:
        for predictor in predictors:
            if bool_dict[predictor]:
                # heat plot
                df = input_df[[response, predictor]].copy()
                hist_2d = px.density_heatmap(df, x=predictor, y=response)
                hist_2d.update_xaxes(title=predictor)
                hist_2d.update_yaxes(title=response)
                hist_2d.show()
                plot_loc = f"html/{predictor}_plot.html"
                hist_2d.write_html(
                    file=plot_loc,
                    include_plotlyjs="cdn",
                )
                plot_dict[predictor] = plot_loc
            else:
                # violin plot
                df = input_df[[response, predictor]].copy()
                violin = px.violin(
                    df, y=predictor, color=response, violinmode="overlay"
                )
                violin.update_layout(
                    title_text=f"violin plot of {predictor} grouped by {response}",
                )
                violin.update_yaxes(title_text=predictor)
                violin.show()
                plot_loc = f"html/{predictor}_plot.html"
                violin.write_html(
                    file=plot_loc,
                    include_plotlyjs="cdn",
                )
                plot_dict[predictor] = plot_loc

    # generate plots if response is continuous
    else:
        for predictor in predictors:
            if bool_dict[predictor]:
                # histogram plot
                df = input_df[[response, predictor]].copy()
                fig = px.histogram(
                    df,
                    x=response,
                    y=response,
                    color=predictor,
                    marginal="box",
                    hover_data=df.columns,
                )
                fig.show()
                plot_loc = f"html/{predictor}_plot.html"
                fig.write_html(
                    file=plot_loc,
                    include_plotlyjs="cdn",
                )
                plot_dict[predictor] = plot_loc
            else:
                # scatter plot with trend line
                df = input_df[[response, predictor]].copy()
                scatter = px.scatter(df, x=predictor, y=response, trendline="ols")
                scatter.update_layout(title_text=f"{predictor} v. {response}")
                scatter.update_xaxes(ticks="inside", title_text=predictor)
                scatter.update_yaxes(ticks="inside", title_text=response)
                scatter.show()
                plot_loc = f"html/{predictor}_plot.html"
                scatter.write_html(
                    file=plot_loc,
                    include_plotlyjs="cdn",
                )
                plot_dict[predictor] = plot_loc

    # generate stats data inputs
    X_cols = input_df.drop(response, axis=1).columns.to_list()
    X = input_df.drop(response, axis=1).to_numpy()
    y = input_df[response].to_numpy()
    t_val, p_val, stat_plots = {}, {}, {}

    # linear regression stats if response is continuous
    if not bool_dict[response]:
        for idx, column in enumerate(X.T):
            column = X[:, idx]
            feature_name = X_cols[idx]
            predictor = statsmodels.api.add_constant(column)
            linear_regression_model = statsmodels.api.OLS(y, predictor, missing="drop")
            linear_regression_fitted = linear_regression_model.fit()
            print(linear_regression_fitted.summary())
            t_value = round(linear_regression_fitted.tvalues[1], 6)
            p_value = "{:.6e}".format(linear_regression_fitted.pvalues[1])
            p_val[feature_name], t_val[feature_name] = t_value, p_value
            fig = px.scatter(x=column, y=y, trendline="ols")
            fig.update_layout(
                title=f"Variable: {feature_name}: (t-value={t_value}) (p-value={p_value})",
                xaxis_title=f"Variable: {feature_name}",
                yaxis_title=f"Response: {response}",
            )
            fig.show()
            plot_loc = f"html/{feature_name}_stats_plot.html"
            fig.write_html(
                file=plot_loc,
                include_plotlyjs="cdn",
            )
            stat_plots[feature_name] = plot_loc

    # logistic regression stats if response is boolean
    else:
        for idx, column in enumerate(X.T):
            column = X[:, idx]
            feature_name = X_cols[idx]
            predictor = statsmodels.api.add_constant(column)
            logistic_regression_model = statsmodels.api.Logit(
                y, predictor, missing="drop"
            )
            logistic_regression_fitted = logistic_regression_model.fit()
            print(logistic_regression_fitted.summary())
            t_value = round(logistic_regression_fitted.tvalues[1], 6)
            p_value = "{:.6e}".format(logistic_regression_fitted.pvalues[1])
            p_val[feature_name], t_val[feature_name] = t_value, p_value
            fig = px.scatter(x=column, y=y, trendline="ols")
            fig.update_layout(
                title=f"Variable: {feature_name}: (t-value={t_value}) (p-value={p_value})",
                xaxis_title=f"Variable: {feature_name}",
                yaxis_title=f"Response: {response}",
            )
            fig.show()
            plot_loc = f"html/{feature_name}_stats_plot.html"
            fig.write_html(
                file=plot_loc,
                include_plotlyjs="cdn",
            )
            stat_plots[feature_name] = plot_loc

    # mean square difference setup
    msd_plots, msd_tables = {}, {}
    for feature in X_cols:
        data = input_df[feature].to_list()
        data.sort()
        data_range = max(data) - min(data)
        step = data_range / 10
        table = pd.DataFrame(
            columns=[
                "lower bin",
                "upper bin",
                "median",
                "count",
                "bin mean",
                "population mean",
                "mean square diff",
            ]
        )
        weighted_table = pd.DataFrame(
            columns=[
                "lower bin",
                "upper bin",
                "median",
                "count",
                "bin mean",
                "population mean",
                "mean square diff",
                "pop proportion",
                "weighted MSD",
            ]
        )

        # mean square unweighted table
        for n in range(10):
            low, high = min(data) + (step * n), min(data) + (step * (n + 1))
            if n == 9:
                b = [y for y in data if low <= y <= high]
            else:
                b = [y for y in data if low <= y < high]
            if not b:
                new_row = {
                    "lower bin": low,
                    "upper bin": high,
                    "median": 0,
                    "count": 0,
                    "bin mean": 0,
                    "population mean": np.nanmean(data),
                    "mean square diff": 0,
                }
            else:
                med, count, mean = (
                    statistics.median(b),
                    int(len(b)),
                    statistics.mean(b),
                )
                pop_mean = np.nanmean(data)
                mean_sq_diff = abs((mean - pop_mean) ** 2)
                new_row = {
                    "lower bin": low,
                    "upper bin": high,
                    "median": med,
                    "count": count,
                    "bin mean": mean,
                    "population mean": pop_mean,
                    "mean square diff": mean_sq_diff,
                }
            table = table.append(new_row, ignore_index=True)
        msd_tables[feature] = html_write(table, feature, "unweighted")

        # mean square weighted table
        for n in range(10):
            low, high = min(data) + (step * n), min(data) + (step * (n + 1))
            if n == 9:
                b = [y for y in data if low <= y <= high]
            else:
                b = [y for y in data if low <= y < high]
            if not b:
                new_row = {
                    "lower bin": low,
                    "upper bin": high,
                    "median": 0,
                    "count": 0,
                    "bin mean": 0,
                    "population mean": np.nanmean(data),
                    "mean square diff": 0,
                    "pop proportion": 0,
                    "weighted MSD": 0,
                }
            else:
                med, count, mean = (
                    statistics.median(b),
                    int(len(b)),
                    statistics.mean(b),
                )
                pop_prop = count / len(data)
                pop_mean = np.nanmean(data)
                mean_sq_diff = abs((mean - pop_mean) ** 2)
                weighted_msd = mean_sq_diff * pop_prop
                new_row = {
                    "lower bin": low,
                    "upper bin": high,
                    "median": med,
                    "count": count,
                    "bin mean": mean,
                    "population mean": pop_mean,
                    "mean square diff": mean_sq_diff,
                    "pop proportion": pop_prop,
                    "weighted MSD": weighted_msd,
                }
            weighted_table = weighted_table.append(new_row, ignore_index=True)
        table = weighted_table
        msd_tables[feature] = html_write(table, feature, "weighted")
        # plot from table
        msd_plots[feature] = plot_msd(table, feature, response)

    # feature importance calculations
    y = input_df[response].values
    X = input_df.drop(response, axis=1)
    if bool_dict[response]:
        rf = RandomForestClassifier()
        rf.fit(X, y)
        feature_importance = rf.feature_importances_
    else:
        rf = RandomForestRegressor()
        rf.fit(X, y)
        feature_importance = rf.feature_importances_
    feature_importance_dict = {
        predictors[i]: feature_importance[i] for i in range(len(predictors))
    }

    # generate final output
    output_list = [
        {i: bool_dict[i] for i in bool_dict if i != response},
        plot_dict,
        p_val,
        t_val,
        stat_plots,
        msd_tables,
        msd_plots,
        feature_importance_dict,
    ]
    output_names = [
        "boolean",
        "plots",
        "p values",
        "t values",
        "statistics plots",
        "msd table",
        "msd plots",
        "feature importances",
    ]
    html = ""
    for i in range(len(output_list)):
        df = pd.DataFrame.from_dict(
            output_list[i],
            orient="index",
        )
        try:
            if df[0].str.contains("html").any():
                df = df.style.format(make_clickable).render()
            else:
                df = df.style.render()
        except AttributeError:
            df = df.style.set_precision(4).render()
        html = html + "\n<br><br>" + output_names[i] + "\n" + df
    with open("output.html", "w") as f:
        f.write(html)
        f.close()


if __name__ == "__main__":
    sys.exit(main())
