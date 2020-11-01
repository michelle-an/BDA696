#!/Users/michellean/workspace/courses/bda696/src/BDA696/.venv/bin/python3

import os
import sys
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy
import seaborn as sns
from scipy import stats


def categorical_check(column):
    uniques = set()
    all_values = []
    for ea in column:
        uniques.add(ea)
        all_values.append(ea)
        if type(ea) is str:
            return True
        if type(ea) is float:
            return False
    if len(uniques) / len(all_values) <= 0.1:
        return True
    else:
        return False


def find_response(columns):
    check = False
    response = ""
    while not check:
        response = input(f"Which column is the response? \n {columns}? \n")
        try:
            response = int(response)
            if response in columns:
                check = True
        except ValueError:
            if response in columns:
                check = True
        else:
            print("Incorrect user input.")
    predictors = [x for x in columns if x != response]
    return response, predictors, columns


def correlation_table(dictionary, output_name=None, simplify=False):
    keys_used = []
    df = pd.DataFrame(columns=["predictor1", "predictor2", "score"])
    for key in dictionary:
        if simplify:
            items = [x for x in list(dictionary[key].items()) if x[0] not in keys_used]
        else:
            items = list(dictionary[key].items())
        keys_used.append(key)
        temp_df = pd.DataFrame(items, columns=["predictor2", "score"])
        temp_df.insert(0, "predictor1", key)
        df = df.append(temp_df, ignore_index=True)
    if simplify:
        df = df.sort_values("score", ascending=False)
    return df


def make_clickable(val):
    folder = os.path.basename(os.getcwd())
    if ".html" in str(val):
        name = val.split("/")[-1][:-5]
        name = name.replace("_", " ")
        if name.endswith("plot"):
            name = name[:-5]
            val = f"{folder}/" + val
        if name.endswith("heatmap"):
            val = f"../{folder}/output/" + val
        return '<a href="{}">{}</a>'.format(val, name)
    elif type(val) == np.float64:
        val = round(val, 8)
    return val


def add_plot_links(df, plot_link_dict, output_name):
    pred1, pred2 = df["predictor1"].to_list(), df["predictor2"].to_list()
    pred_link1, pred_link2 = [], []
    for ea in pred1:
        pred_link1.append(f"../{plot_link_dict[ea]}")
    for ea in pred2:
        pred_link2.append(f"../{plot_link_dict[ea]}")
    df["predictor1"] = pred_link1
    df["predictor2"] = pred_link2
    df = df.style.format(make_clickable).render()
    file_loc = f"output/{output_name}_correlation_table.html"
    with open(file_loc, "w") as txt_file:
        txt_file.write(df)
    return df, file_loc


def correlation_matrix_plot(score_dict, title):
    table = correlation_table(score_dict)
    x_axis = list(score_dict.keys())
    y_axis = list(score_dict[x_axis[0]].keys())
    data = []
    for xx in y_axis:
        temp = []
        for yy in x_axis:
            temp2 = list(
                table.loc[(table["predictor2"] == xx) & (table["predictor1"] == yy)][
                    "score"
                ]
            )
            for val in temp2:
                temp.append(val)
        data.append(temp)
    figure = px.imshow(
        data,
        labels=dict(
            x="Continuous Variables",
            y="Categorical Variables",
            color="correlation score",
        ),
        x=x_axis,
        y=y_axis,
        color_continuous_scale=px.colors.sequential.Burg,
        title=title,
    )
    fig_loc = f"output/{title}_heatmap.html"
    fig_3d_loc = f"output/{title}_3D_heatmap.html"
    figure.write_html(file=fig_loc, default_width="40%", include_plotlyjs="cdn")
    figure.show()

    fig_3d = go.Figure(data=[go.Surface(z=data)])
    fig_3d.write_html(file=fig_3d_loc, default_width="45%", include_plotlyjs="cdn")
    fig_3d.show()
    return fig_loc, fig_3d_loc


def brute_force_cont_cont(feature_1_name, feature_2_name, response, input_df):
    n_bins = 10
    response_pop_mean, response_pop = float(np.mean(input_df[response].to_list())), len(
        input_df[response].to_list()
    )
    feature_1, feature_2 = (
        input_df[feature_1_name].to_list(),
        input_df[feature_2_name].to_list(),
    )
    min_1, max_1, min_2, max_2 = (
        min(feature_1),
        max(feature_1),
        min(feature_2),
        max(feature_2),
    )
    size_1, size_2 = max_1 - min_1, max_2 - min_2
    bin_size_1, bin_size_2 = size_1 / n_bins, size_2 / n_bins
    bin_centers_1, bin_centers_2 = [], []
    msd_array = np.empty((n_bins, n_bins))
    w_msd_array = np.empty((n_bins, n_bins))
    msd_array[:], w_msd_array[:] = np.nan, np.nan
    for i in range(n_bins):
        low_1, high_1 = min_1 + bin_size_1 * i, min_1 + bin_size_1 * (i + 1)
        bin_centers_1.append((high_1 - low_1) + low_1)
        if i == 0:
            bin_1_indices = [
                index for index, b1 in enumerate(feature_1) if low_1 <= b1 <= high_1
            ]
        else:
            bin_1_indices = [
                index for index, b1 in enumerate(feature_1) if low_1 < b1 <= high_1
            ]
        for j in range(n_bins):
            low_2, high_2 = min_2 + bin_size_2 * j, min_2 + bin_size_2 * (j + 1)
            if j == 0:
                bin_centers_2.append((high_2 - low_2) + low_2)
                bin_2_indices = [
                    index for index, b2 in enumerate(feature_2) if low_2 <= b2 <= high_2
                ]
            else:
                bin_2_indices = [
                    index for index, b2 in enumerate(feature_2) if low_2 < b2 <= high_2
                ]
            indices = list(set(bin_1_indices) & set(bin_2_indices))
            responses_in_bin = [input_df[response].to_list()[x] for x in indices]
            if len(responses_in_bin) != 0:
                bin_average_resp = float(np.mean(responses_in_bin))
                bin_pop = len(responses_in_bin)
                bin_msd = (bin_average_resp - response_pop_mean) ** 2
                pop_proportion = bin_pop * response_pop
                bin_w_msd = bin_msd / pop_proportion
                w_msd_array[i][j] = bin_w_msd
                msd_array[i][j] = bin_msd
    # call plotting function here, also return plot link
    fig = go.Figure(data=go.Heatmap(z=w_msd_array, hoverongaps=False))
    fig.update_xaxes(title_text=feature_1_name)
    fig.update_yaxes(title_text=feature_2_name)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(
        title=f"{feature_1_name} v. {feature_2_name} Weighted MSD heatmap"
    )
    fig_loc = f"output/{feature_1_name}_v_{feature_2_name}_w_msd_heatmap.html"
    fig.write_html(file=fig_loc, include_plotlyjs="cdn")
    fig.show()
    return w_msd_array, msd_array, fig_loc


def brute_force_cat_cont(cont_feature_name, cat_feature_name, response, input_df):
    n_bins_cont = 10
    n_bins_cat = len(set(input_df[cat_feature_name].to_list()))
    response_pop_mean, response_pop = float(np.mean(input_df[response].to_list())), len(
        input_df[response].to_list()
    )
    cont_feature, cat_feature = (
        input_df[cont_feature_name].to_list(),
        input_df[cat_feature_name].to_list(),
    )
    cat_bins = list(set(cat_feature))
    min_1, max_1 = min(cont_feature), max(cont_feature)
    size_1 = max_1 - min_1
    bin_size_1 = size_1 / n_bins_cont
    bin_centers_1 = []
    msd_array = np.empty((n_bins_cont, n_bins_cat))
    w_msd_array = np.empty((n_bins_cont, n_bins_cat))
    msd_array[:], w_msd_array[:] = np.nan, np.nan
    for i in range(n_bins_cont):
        low_1, high_1 = min_1 + bin_size_1 * i, min_1 + bin_size_1 * (i + 1)
        bin_centers_1.append((high_1 - low_1) + low_1)
        if i == 0:
            bin_1_indices = [
                index for index, b1 in enumerate(cont_feature) if low_1 <= b1 <= high_1
            ]
        else:
            bin_1_indices = [
                index for index, b1 in enumerate(cont_feature) if low_1 < b1 <= high_1
            ]
        for j in cat_bins:
            k_index = cat_bins.index(j)
            bin_2_indices = [index for index, b2 in enumerate(cat_feature) if b2 == j]
            indices = list(set(bin_1_indices) & set(bin_2_indices))
            responses_in_bin = [input_df[response].to_list()[x] for x in indices]
            if len(responses_in_bin) != 0:
                bin_average_resp = float(np.mean(responses_in_bin))
                bin_pop = len(responses_in_bin)
                bin_msd = (bin_average_resp - response_pop_mean) ** 2
                pop_proportion = bin_pop / response_pop
                bin_w_msd = bin_msd * pop_proportion
                w_msd_array[i][k_index] = bin_w_msd
                msd_array[i][k_index] = bin_msd
    # call plotting function here, also return plot link
    fig = go.Figure(data=go.Heatmap(z=w_msd_array, hoverongaps=False))
    fig.update_xaxes(title_text=cont_feature_name)
    fig.update_yaxes(title_text=cat_feature_name)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(
        title=f"{cont_feature_name} v. {cat_feature_name} Weighted MSD heatmap"
    )
    fig_loc = f"output/{cont_feature_name}_v_{cat_feature_name}_w_msd_heatmap.html"
    fig.write_html(file=fig_loc, include_plotlyjs="cdn")
    fig.show()
    return w_msd_array, msd_array, fig_loc


def brute_force_cat_cat(feature_name_1, feature_name_2, response, input_df):
    n_bins_cat_1, n_bins_cat_2 = len(set(input_df[feature_name_1].to_list())), len(
        set(input_df[feature_name_2].to_list())
    )
    response_pop_mean, response_pop = float(np.mean(input_df[response].to_list())), len(
        input_df[response].to_list()
    )
    cat_feature_1, cat_feature_2 = (
        input_df[feature_name_1].to_list(),
        input_df[feature_name_2].to_list(),
    )
    cat_bins_1, cat_bins_2 = list(set(cat_feature_1)), list(set(cat_feature_2))
    msd_array = np.empty((n_bins_cat_1, n_bins_cat_2))
    w_msd_array = np.empty((n_bins_cat_1, n_bins_cat_2))
    msd_array[:], w_msd_array[:] = np.nan, np.nan
    for i in cat_bins_1:
        k_index = cat_bins_1.index(i)
        bin_1_indices = [index for index, b1 in enumerate(cat_feature_1) if b1 == i]
        for j in cat_bins_2:
            l_index = cat_bins_2.index(j)
            bin_2_indices = [index for index, b2 in enumerate(cat_feature_2) if b2 == j]
            indices = list(set(bin_1_indices) & set(bin_2_indices))
            responses_in_bin = [input_df[response].to_list()[x] for x in indices]
            if len(responses_in_bin) != 0:
                bin_average_resp = float(np.mean(responses_in_bin))
                bin_pop = len(responses_in_bin)
                bin_msd = (bin_average_resp - response_pop_mean) ** 2
                pop_proportion = bin_pop / response_pop
                bin_w_msd = bin_msd * pop_proportion
                w_msd_array[k_index][l_index] = bin_w_msd
                msd_array[k_index][l_index] = bin_msd
    # call plotting function here, also return plot link
    fig = go.Figure(data=go.Heatmap(z=w_msd_array, hoverongaps=False))
    fig.update_xaxes(title_text=feature_name_1)
    fig.update_yaxes(title_text=feature_name_2)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(
        title=f"{feature_name_1} v. {feature_name_2} Weighted MSD heatmap"
    )
    fig_loc = f"output/{feature_name_1}_v_{feature_name_2}_w_msd_heatmap.html"
    fig.write_html(file=fig_loc, include_plotlyjs="cdn")
    fig.show()
    return w_msd_array, msd_array, fig_loc


def brute_force_table(dictionary, filename):
    df = pd.DataFrame(
        columns=[
            "predictor1",
            "predictor2",
            "unweighted msd",
            "weighted msd",
            "plot link",
        ]
    )
    for key in dictionary:
        predictor1, predictor2 = key
        weighted_msd, msd, plot_link = (
            dictionary[key][0],
            dictionary[key][1],
            dictionary[key][2],
        )
        plot_link = plot_link.split("/")[-1]
        df = df.append(
            {
                "predictor1": predictor1,
                "predictor2": predictor2,
                "weighted msd": weighted_msd,
                "unweighted msd": msd,
                "plot link": plot_link,
            },
            ignore_index=True,
        )
    df = df.sort_values("weighted msd", ascending=False)
    df = df.style.format(make_clickable).render()
    file_loc = f"output/{filename}_brute_force_table.html"
    with open(file_loc, "w") as txt_file:
        txt_file.write(df)
    return df, file_loc


def fill_na(data):
    if isinstance(data, pd.Series):
        return data.fillna(0)
    else:
        return np.array([value if value is not None else 0 for value in data])


def cat_correlation(x, y, bias_correction=True, tschuprow=False):
    """
    Calculates correlation statistic for categorical-categorical association.
    The two measures supported are:
    1. Cramer'V ( default )
    2. Tschuprow'T

    SOURCES:
    1.) CODE: https://github.com/MavericksDS/pycorr
    2.) Used logic from:
        https://stackoverflow.com/questions/20892799/using-pandas-calculate-cram%C3%A9rs-coefficient-matrix
        to ignore yates correction factor on 2x2
    3.) Haven't validated Tschuprow

    Bias correction and formula's taken from :
    https://www.researchgate.net/publication/270277061_A_bias-correction_for_Cramer's_V_and_Tschuprow's_T

    Wikipedia for Cramer's V: https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V
    Wikipedia for Tschuprow' T: https://en.wikipedia.org/wiki/Tschuprow%27s_T
    Parameters:
    -----------
    x : list / ndarray / Pandas Series
        A sequence of categorical measurements
    y : list / NumPy ndarray / Pandas Series
        A sequence of categorical measurements
    bias_correction : Boolean, default = True
    tschuprow : Boolean, default = False
               For choosing Tschuprow as measure
    Returns:
    --------
    float in the range of [0,1]
    """
    corr_coeff = np.nan
    try:
        x, y = fill_na(x), fill_na(y)
        crosstab_matrix = pd.crosstab(x, y)
        n_observations = crosstab_matrix.sum().sum()

        yates_correct = True
        if bias_correction:
            if crosstab_matrix.shape == (2, 2):
                yates_correct = False

        chi2, _, _, _ = stats.chi2_contingency(
            crosstab_matrix, correction=yates_correct
        )
        phi2 = chi2 / n_observations

        # r and c are number of categories of x and y
        r, c = crosstab_matrix.shape
        if bias_correction:
            phi2_corrected = max(0, phi2 - ((r - 1) * (c - 1)) / (n_observations - 1))
            r_corrected = r - ((r - 1) ** 2) / (n_observations - 1)
            c_corrected = c - ((c - 1) ** 2) / (n_observations - 1)
            if tschuprow:
                corr_coeff = np.sqrt(
                    phi2_corrected / np.sqrt((r_corrected - 1) * (c_corrected - 1))
                )
                return corr_coeff
            corr_coeff = np.sqrt(
                phi2_corrected / min((r_corrected - 1), (c_corrected - 1))
            )
            return corr_coeff
        if tschuprow:
            corr_coeff = np.sqrt(phi2 / np.sqrt((r - 1) * (c - 1)))
            return corr_coeff
        corr_coeff = np.sqrt(phi2 / min((r - 1), (c - 1)))
        return corr_coeff
    except Exception as ex:
        print(ex)
        if tschuprow:
            warnings.warn("Error calculating Tschuprow's T", RuntimeWarning)
        else:
            warnings.warn("Error calculating Cramer's V", RuntimeWarning)
        return corr_coeff


def cat_cont_correlation_ratio(categories, values):
    """
    Correlation Ratio: https://en.wikipedia.org/wiki/Correlation_ratio
    SOURCE:
    1.) https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
    :param categories: Numpy array of categories
    :param values: Numpy array of values
    :return: correlation
    """
    f_cat, _ = pd.factorize(categories)
    cat_num = np.max(f_cat) + 1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0, cat_num):
        cat_measures = values[np.argwhere(f_cat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
    numerator = np.sum(
        np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2))
    )
    denominator = np.sum(np.power(np.subtract(values, y_total_avg), 2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator / denominator)
    return eta


def main():
    # create path for html output
    if not os.path.exists("html"):
        os.makedirs("html")
    if not os.path.exists("output"):
        os.makedirs("output")

    # import data to pandas
    input_df = sns.load_dataset("iris")

    if not isinstance(input_df, pd.DataFrame):
        input_df = pd.DataFrame(data=input_df.data, columns=input_df.feature_names)
    input_df = input_df.dropna(axis=0, how="any")

    # find which columns are predictors and which is response
    print(input_df.head(5))
    print("\n")
    response, predictors, cols = find_response(input_df.columns.to_list())

    # determine which columns are categorical and which are continuous
    cat_dict = {}
    for column in cols:
        cat_dict[column] = categorical_check(input_df[column])

    # calculate correlation metrics
    # split dataset between categorical and continuous
    cont_features = [x for x in predictors if not cat_dict[x]]
    cat_features = [x for x in predictors if cat_dict[x]]

    # continuous-continuous pairs
    pearson_correlation_scores = {}
    for feature1 in cont_features:
        pearson_correlation_scores[feature1] = {}
        for feature2 in cont_features:
            pearson_correlation_scores[feature1][feature2] = round(
                scipy.stats.pearsonr(input_df[feature1], input_df[feature2])[0], 5
            )

    # continuous-categorical pairs
    cat_cont_ratio_scores = {}
    for feature1 in cont_features:
        cat_cont_ratio_scores[feature1] = {}
        for feature2 in cat_features:
            cat_cont_ratio_scores[feature1][feature2] = round(
                cat_cont_correlation_ratio(input_df[feature2], input_df[feature1]), 5
            )

    # categorical-categorical pairs
    cat_cat_correlation_scores = {}
    for feature1 in cat_features:
        cat_cat_correlation_scores[feature1] = {}
        for feature2 in cat_features:
            cat_cat_correlation_scores[feature1][feature2] = cat_correlation(
                input_df[feature1], input_df[feature2]
            )

    # generate plots
    plot_links_dict = {}
    # generate plots if response is categorical
    if cat_dict[response]:
        for predictor in predictors:
            if cat_dict[predictor]:
                # heat plot
                df = input_df[[response, predictor]].copy()
                hist_2d = px.density_heatmap(df, x=predictor, y=response)
                hist_2d.update_xaxes(title=predictor)
                hist_2d.update_yaxes(title=response)
                plot_loc = f"html/{predictor}_plot.html"
                hist_2d.write_html(
                    file=plot_loc,
                    include_plotlyjs="cdn",
                )
                plot_links_dict[predictor] = plot_loc
                hist_2d.show()

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
                plot_loc = f"html/{predictor}_plot.html"
                violin.write_html(
                    file=plot_loc,
                    include_plotlyjs="cdn",
                )
                plot_links_dict[predictor] = plot_loc
                violin.show()

    # generate plots if response is continuous
    else:
        for predictor in predictors:
            if cat_dict[predictor]:
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
                plot_loc = f"html/{predictor}_plot.html"
                fig.write_html(
                    file=plot_loc,
                    include_plotlyjs="cdn",
                )
                plot_links_dict[predictor] = plot_loc
                fig.show()

            else:
                # scatter plot with trend line
                df = input_df[[response, predictor]].copy()
                scatter = px.scatter(df, x=predictor, y=response, trendline="ols")
                scatter.update_layout(title_text=f"{predictor} v. {response}")
                scatter.update_xaxes(ticks="inside", title_text=predictor)
                scatter.update_yaxes(ticks="inside", title_text=response)
                plot_loc = f"html/{predictor}_plot.html"
                scatter.write_html(
                    file=plot_loc,
                    include_plotlyjs="cdn",
                )
                plot_links_dict[predictor] = plot_loc
                scatter.show()

    # correlation metric tables
    cont_cont_table = correlation_table(pearson_correlation_scores, simplify=True)
    cat_cont_table = correlation_table(cat_cont_ratio_scores, simplify=True)
    cat_cat_table = correlation_table(cat_cat_correlation_scores, simplify=True)

    # add plot links to correlation metric tables
    cont_cont_table, cont_cont_loc = add_plot_links(
        cont_cont_table, plot_links_dict, "cont_cont"
    )
    cat_cont_table, cat_cont_loc = add_plot_links(
        cat_cont_table, plot_links_dict, "cat_cont"
    )
    cat_cat_table, cat_cat_loc = add_plot_links(
        cat_cat_table, plot_links_dict, "cat_cat"
    )

    # generate correlation matrix plots
    correlation_matrix_plot(pearson_correlation_scores, "pearson correlation scores")
    correlation_matrix_plot(
        cat_cont_ratio_scores, "continuous v. categorical correlation scores"
    )
    correlation_matrix_plot(
        cat_cat_correlation_scores, "categorical v. categorical correlation scores"
    )

    # response check for brute force, update to 1s and 0s if necessary
    if cat_dict[response]:
        if set(input_df[response].to_list()) != {0, 1}:
            column_values = input_df[response]
            categories = list(set(column_values))
            column_values_updated = [
                0 if x == categories[0] else 1 for x in column_values
            ]
            input_df[response] = column_values_updated

    # cont-cont binning, MSD, plotting
    predictors_used = []
    cont_cont_brute_force_dict = {}
    for predictor_1 in cont_features:
        for predictor_2 in [x for x in cont_features if x not in predictors_used]:
            w_msd_array, msd_array, plot_link = brute_force_cont_cont(
                predictor_1, predictor_2, response, input_df
            )
            predictors_used.append(predictor_1)
            cont_cont_brute_force_dict[predictor_1, predictor_2] = (
                np.nanmean(w_msd_array),
                np.nanmean(msd_array),
                plot_link,
            )

    # cat-cont binning, MSD, plotting
    cat_cont_brute_force_dict = {}
    for predictor_1 in cont_features:
        for predictor_2 in cat_features:
            w_msd_array, msd_array, plot_link = brute_force_cat_cont(
                predictor_1, predictor_2, response, input_df
            )
            cat_cont_brute_force_dict[predictor_1, predictor_2] = (
                np.nanmean(w_msd_array),
                np.nanmean(msd_array),
                plot_link,
            )

    # cat-cat binning, MSD, plotting
    predictors_used = []
    cat_cat_brute_force_dict = {}
    for predictor_1 in cat_features:
        for predictor_2 in [x for x in cat_features if x not in predictors_used]:
            w_msd_array, msd_array, plot_link = brute_force_cat_cat(
                predictor_1, predictor_2, response, input_df
            )
            predictors_used.append(predictor_1)
            cat_cat_brute_force_dict[predictor_1, predictor_2] = (
                np.nanmean(w_msd_array),
                np.nanmean(msd_array),
                plot_link,
            )

    # generate brute force tables
    cont_cont_brute_force_df, cont_cont_brute_force_loc = brute_force_table(
        cont_cont_brute_force_dict, "cont_cont"
    )
    cat_cont_brute_force_df, cont_cont_brute_force_loc = brute_force_table(
        cat_cont_brute_force_dict, "cont_cat"
    )
    cat_cat_brute_force_df, cont_cont_brute_force_loc = brute_force_table(
        cat_cat_brute_force_dict, "cat_cat"
    )

    heatmaps = []
    with open("output/pearson correlation scores_heatmap.html", "r") as file:
        heatmaps.append(file.read())
    with open("output/pearson correlation scores_3D_heatmap.html", "r") as file:
        heatmaps.append(file.read())
    with open(
        "output/continuous v. categorical correlation scores_heatmap.html", "r"
    ) as file:
        heatmaps.append(file.read())
    with open(
        "output/continuous v. categorical correlation scores_3D_heatmap.html", "r"
    ) as file:
        heatmaps.append(file.read())
    with open(
        "output/categorical v. categorical correlation scores_heatmap.html", "r"
    ) as file:
        heatmaps.append(file.read())
    with open(
        "output/categorical v. categorical correlation scores_3D_heatmap.html", "r"
    ) as file:
        heatmaps.append(file.read())

    output_list = [
        "continuous v continuous correlation table",
        cont_cont_table,
        "<br> continuous v categorical correlation table",
        cat_cont_table,
        "<br> categorical v categorical correlation table",
        cat_cat_table,
        "<br> continuous v continuous correlation matrix",
        heatmaps[0],
        "<br> continuous v continuous correlation matrix 3-D",
        heatmaps[1],
        "<br> continuous v categorical correlation matrix",
        heatmaps[2],
        "<br> continuous v categorical correlation matrix 3-D",
        heatmaps[3],
        "<br> categorical v categorical correlation matrix",
        heatmaps[4],
        "<br> categorical v categorical correlation matrix 3-D",
        heatmaps[5],
        "<br> continuous v continuous brute force table",
        cont_cont_brute_force_df,
        "<br> continuous v categorical brute force table",
        cat_cont_brute_force_df,
        "<br> categorical v categorical brute force table",
        cat_cat_brute_force_df,
    ]

    output_str = ""
    for item in output_list:
        output_str = output_str + str(item)
        output_str = output_str + "\n<br>\n"
    with open("midterm.html", "w") as html:
        html.write(output_str)


if __name__ == "__main__":
    sys.exit(main())
