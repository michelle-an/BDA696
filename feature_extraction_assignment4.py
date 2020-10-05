#!/Users/michellean/workspace/courses/bda696/src/BDA696/.venv/bin/python3

import statistics
import sys

import pandas as pd
import plotly.express as px
import statsmodels.api


def bool_check(column):
    uniques = set()
    for ea in column:
        uniques.add(ea)
    if len(uniques) == 2:
        return True
    else:
        return False


def html_write(table, feature):
    html = table.to_html()
    text_file = open(f"{feature}_unweighted.html", "w")
    text_file.write(html)
    text_file.close()
    return


def main():
    debug = True  # TODO: remove later
    # TODO: replace this dataset with a dataset input?
    # import data to pandas
    input_df = pd.read_csv(
        "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
    )
    input_df = input_df.drop("embarked", axis=1)  # TODO: remove later
    input_df = input_df.drop("class", axis=1)  # TODO: remove later
    input_df = input_df.drop("who", axis=1)  # TODO: remove later
    input_df = input_df.drop("deck", axis=1)  # TODO: remove later
    input_df = input_df.drop("alone", axis=1)  # TODO: remove later
    input_df = input_df.drop("alive", axis=1)  # TODO: remove later
    input_df = input_df.drop("adult_male", axis=1)  # TODO: remove later
    input_df = input_df.drop("sex", axis=1)  # TODO: remove later
    input_df = input_df.drop("embark_town", axis=1)  # TODO: remove later
    # find which columns are predictors and which is response
    cols = input_df.columns.to_list()
    print("column names after dropped columns", cols)
    check = False
    if not debug:  # TODO: remove later
        while not check:  # check == False
            response = input(f"Which column is the response? \n {cols}? \n")
            print("response type", type(response))
            if response in cols:
                check = True
            else:
                print("Incorrect user input.")
    else:
        response = "survived"
    predictors = [x for x in cols if x != response]

    # determine which columns in response and predictor are categorical and which are continuous
    bool_dict = {response: bool_check(input_df[response])}
    print("bool_dict", bool_dict)
    for predictor in predictors:
        bool_dict[predictor] = bool_check(input_df[predictor])
    print("updated bool_dict", bool_dict)

    # If response is categorical generate plots
    if not debug:  # debug = False     # TODO: remove later
        if bool_dict[response]:  # bool_dict[response] == True
            for predictor in predictors:
                if bool_dict[predictor]:  # if predictor is categorical
                    # heat plot             # categorical vs categorical
                    df = input_df[[response, predictor]].copy()
                    hist_2d = px.density_heatmap(df, x=predictor, y=response)
                    hist_2d.update_xaxes(title=predictor)
                    hist_2d.update_yaxes(title=response)
                    hist_2d.show()
                else:
                    # violin plot           # categorical response vs continuous predictor
                    df = input_df[[response, predictor]].copy()
                    # TODO: figure out predictor response in graph. Which is which?
                    violin = px.violin(
                        df, y=predictor, color=response, violinmode="overlay"
                    )
                    violin.update_layout(
                        title_text=f"violin plot of {predictor} grouped by {response}",
                    )
                    violin.update_yaxes(title_text=predictor)
                    violin.show()

        # generate plots if response is continuous
        else:
            for predictor in predictors:
                if bool_dict[predictor]:  # continuous response vs categorical predictor
                    # histogram plot
                    df = input_df[[response, predictor]].copy()
                    hist = px.histogram(
                        df,
                        x=response,
                        y=response,
                        color=predictor,
                        marginal="box",
                        hover_data=df.columns,
                    )
                    hist.show()
                else:  # continuous response vs continuous predictor
                    # scatter plot with trend line
                    df = input_df[[response, predictor]].copy()
                    scatter = px.scatter(df, x=predictor, y=response, trendline="ols")
                    scatter.update_layout(title_text=f"{predictor} v. {response}")
                    scatter.update_xaxes(ticks="inside", title_text=predictor)
                    scatter.update_yaxes(ticks="inside", title_text=response)
                    scatter.show()

    # generate stats data inputs
    X_cols = input_df.drop(response, axis=1).columns.to_list()
    print("X_cols - ", X_cols)
    X = input_df.drop(response, axis=1).to_numpy()
    y = input_df[response].to_numpy()
    scores = {}
    # linear regression stats if response is continuous
    if not bool_dict[response]:
        for idx, column in enumerate(X.T):
            column = X[:, idx]
            feature_name = X_cols[idx]
            print(f"Variable: {feature_name}")
            predictor = statsmodels.api.add_constant(column)
            linear_regression_model = statsmodels.api.OLS(y, predictor, missing="drop")
            linear_regression_fitted = linear_regression_model.fit()
            print(linear_regression_fitted.summary())
            t_value = round(linear_regression_fitted.tvalues[1], 6)
            p_value = "{:.6e}".format(linear_regression_fitted.pvalues[1])
            if not debug:  # TODO: remove later
                fig = px.scatter(x=column, y=y, trendline="ols")
                fig.update_layout(
                    title=f"Variable: {feature_name}: (t-value={t_value}) (p-value={p_value})",
                    xaxis_title=f"Variable: {feature_name}",
                    yaxis_title=f"Response: {response}",
                )
                fig.show()
            scores[feature_name] = [p_value, t_value]
    # logistic regression stats if response is boolean
    else:
        for idx, column in enumerate(X.T):
            column = X[:, idx]
            feature_name = X_cols[idx]
            print(f"Variable: {feature_name}")
            predictor = statsmodels.api.add_constant(column)
            logistic_regression_model = statsmodels.api.Logit(
                y, predictor, missing="drop"
            )
            logistic_regression_fitted = logistic_regression_model.fit()
            print(logistic_regression_fitted.summary())
            t_value = round(logistic_regression_fitted.tvalues[1], 6)
            p_value = "{:.6e}".format(logistic_regression_fitted.pvalues[1])
            if not debug:  # TODO: remove later
                fig = px.scatter(x=column, y=y, trendline="ols")
                fig.update_layout(
                    title=f"Variable: {feature_name}: (t-value={t_value}) (p-value={p_value})",
                    xaxis_title=f"Variable: {feature_name}",
                    yaxis_title=f"Response: {response}",
                )
                fig.show()
            scores[feature_name] = [p_value, t_value]
    print(scores)

    print(" \n\n\n\n")
    # mean square difference setup
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
            if n == 9:
                b = [
                    y
                    for y in data
                    if (min(data) + (step * n)) <= y <= (min(data) + step * (n + 1))
                ]
            else:
                b = [
                    y
                    for y in data
                    if (min(data) + (step * n)) <= y < (min(data) + step * (n + 1))
                ]
            if not b:
                new_row = {
                    "lower bin": 0,
                    "upper bin": 0,
                    "median": 0,
                    "count": 0,
                    "bin mean": 0,
                    "population mean": 0,
                    "mean square diff": 0,
                }
            else:
                mi, ma, med, count, mean = (
                    min(b),
                    max(b),
                    statistics.median(b),
                    int(len(b)),
                    statistics.mean(b),
                )
                pop_mean = statistics.mean(data)
                mean_sq_diff = abs(mean ** 2 - pop_mean ** 2)
                new_row = {
                    "lower bin": mi,
                    "upper bin": ma,
                    "median": med,
                    "count": count,
                    "bin mean": mean,
                    "population mean": pop_mean,
                    "mean square diff": mean_sq_diff,
                }
            table = table.append(new_row, ignore_index=True)
        print(f"\n{feature}\n{table}")
        html_write(table, feature)
        # mean square weighted table
        for n in range(10):
            if n == 9:
                b = [
                    y
                    for y in data
                    if (min(data) + (step * n)) <= y <= (min(data) + step * (n + 1))
                ]
            else:
                b = [
                    y
                    for y in data
                    if (min(data) + (step * n)) <= y < (min(data) + step * (n + 1))
                ]
            if not b:
                new_row = {
                    "lower bin": 0,
                    "upper bin": 0,
                    "median": 0,
                    "count": 0,
                    "bin mean": 0,
                    "population mean": 0,
                    "mean square diff": 0,
                    "pop proportion": 0,
                    "weighted MSD": 0,
                }
            else:
                mi, ma, med, count, mean = (
                    min(b),
                    max(b),
                    statistics.median(b),
                    int(len(b)),
                    statistics.mean(b),
                )
                pop_prop = count / len(data)
                pop_mean = statistics.mean(data)
                mean_sq_diff = abs(mean ** 2 - pop_mean ** 2)
                weighted_msd = mean_sq_diff * pop_prop
                new_row = {
                    "lower bin": mi,
                    "upper bin": ma,
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
        print(f"\n{feature} weighted\n{table}")
        html_write(table, feature)
        # histogram/line plots


if __name__ == "__main__":
    sys.exit(main())
    # sys.exit()
