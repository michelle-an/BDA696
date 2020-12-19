import os
import pickle
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api
from plotly.subplots import make_subplots
from scipy import stats
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle


def write_html_plot(filename, dir, plot):
    plot.write_html(file=f"{dir}/{filename}", include_plotlyjs=True)
    return


def make_clickable1(val):
    if val == "-":
        return val
    else:
        url = val.split("#")[1]
        name = val.split("#")[0]
        return '<a href="{}">{}</a>'.format(url, name)


def brute_force_cont_msd(items_list1, items_list2, response_list, num_bins=10):
    mi1, ma1, mi2, ma2 = (
        min(items_list1),
        max(items_list1),
        min(items_list2),
        max(items_list2),
    )
    sw1, sw2 = (ma1 - mi1) / 10, (ma2 - mi2) / 10
    response_mean = float(np.mean(response_list))
    centers1, centers2, _, _, _, _ = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    msd_array = np.zeros((10, 10))
    for i in range(num_bins):
        for j in range(num_bins):
            if i != (num_bins - 1):
                start1 = i * sw1 + mi1
                end1 = (i + 1) * sw1 + mi1
            else:
                start1 = i * sw1 + mi1
                end1 = ma1 + 1 + mi1
            if j != (num_bins - 1):
                start2 = j * sw2 + mi2
                end2 = (j + 1) * sw2 + mi2
            else:
                start2 = j * sw2 + mi2
                end2 = ma2 + 1 + mi2
            idx1 = [i1 for i1, x in enumerate(items_list1) if start1 <= x < end1]
            idx2 = [i2 for i2, z in enumerate(items_list2) if start2 <= z < end2]
            idx = list(set(idx1 + idx2))

            resp_in_bin = [response_list[z] for z in idx]
            pop_prop = len(resp_in_bin) / len(response_list)
            resp_bin_mean = float(np.mean(resp_in_bin))

            centers1.append(sw1 * i + sw1 / 2)
            centers2.append(sw2 * i + sw2 / 2)

            msd_array[i][j] = ((resp_bin_mean - response_mean) ** 2) * pop_prop
    return msd_array


def fill_na(data):
    if isinstance(data, pd.Series):
        return data.fillna(0)
    else:
        return np.array([value if value is not None else 0 for value in data])


def cat_correlation(x, y, bias_correction=True, tschuprow=False):
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


def plot_violin(n1, n2, cont_legend, df):
    better_violin = go.Figure()
    filename = ""
    for violin_feature in cont_legend[n1:n2]:
        filename = filename + str(violin_feature) + " "
        better_violin.add_trace(
            go.Violin(
                y=df[violin_feature][df["classification"] == "portal"],
                legendgroup="portal",
                name=violin_feature,
                side="negative",
            )
        )
        better_violin.add_trace(
            go.Violin(
                y=df[violin_feature][df["classification"] == "not portal"],
                legendgroup="not portal",
                name=violin_feature,
                side="positive",
            )
        )
    better_violin.update_traces(width=1, points=False)
    better_violin.update_layout(violinmode="group")
    better_violin.update_layout(xaxis_showgrid=False, xaxis_zeroline=False)
    better_violin.update_layout(
        xaxis_title="Violin Plot of values split by portal (left) and not portal (right)"
    )
    # better_violin.show()
    filename = filename + "violin plot.html"
    write_html_plot(filename, "feature plots", better_violin)
    return


def main():
    # load csv
    df1 = pd.read_csv("features_data1.csv", header=0)
    df2 = pd.read_csv("features_data2.csv", header=0)
    df = df1.append(df2, ignore_index=True, sort=False)

    print(df.head())

    # for debugging purposes
    # df = df.sample(frac=1).reset_index(drop=True)
    # df = df.head(1000)

    cont_legend = [
        "Isoelectric point",
        "Isoelectric point q1",
        "Isoelectric point q2",
        "Isoelectric point q3",
        "Isoelectric point q4",
        "Isoelectric point q5",
        "instability index",
        "aromaticity index",
        "acidic fraction",
        "basic fraction",
        "acidic fraction q1",
        "acidic fraction q2",
        "acidic fraction q3",
        "acidic fraction q4",
        "acidic fraction q5",
        "basic fraction q1",
        "basic fraction q2",
        "basic fraction q3",
        "basic fraction q4",
        "basic fraction q5",
        "iso * acid and base",
        "molar extinction coefficient 1",
        "molar extinction coefficient 2",
        "gravy score",
        "gravy score q1",
        "gravy score q2",
        "gravy score q3",
        "gravy score q4",
        "gravy score q5",
        "molecular weight",
        "flexibility mean",
        "flexibility mean q1",
        "flexibility mean q2",
        "flexibility mean q3",
        "flexibility mean q4",
        "flexibility mean q5",
        "peak flexibility smoothed",
        "peak flexibility position",
        "flex ratio above 1",
        "flex ratio above 2",
        "flex ratio above 3",
        "flex ratio below 1",
        "flex ratio below 2",
        "flex ratio below 3",
        "mean flexibility at end",
        "Leu-Leu dimer frequency",
        "molar ext coeff div by Trp frequency ",
        "Helix secondary fraction",
        "Turn secondary fraction",
        "Sheet secondary fraction",
    ]
    cat_legend = [
        "more acidic",
        "more flex",
        "acidic isoelectric point",
        "long repeat",
        "short repeat",
    ]

    # Better plotting of cont features
    plot_violin(0, 10, cont_legend, df)
    plot_violin(10, 20, cont_legend, df)
    plot_violin(20, 30, cont_legend, df)
    plot_violin(30, 40, cont_legend, df)
    plot_violin(40, 50, cont_legend, df)

    # Better plotting of cat features
    true_row = df[df["classification"] == "portal"]
    false_row = df[df["classification"] == "not portal"]
    port, nport = [], []
    for cat in cat_legend:
        portal_true = true_row[true_row[cat] == 1][cat]
        nportal_true = false_row[false_row[cat] == 1][cat]
        port.append(len(portal_true))
        nport.append(len(nportal_true))

    for i in range(len(port)):
        # tot = port[i] + nport[i]
        port[i] = port[i] / true_row.shape[0]
        nport[i] = nport[i] / false_row.shape[0]

    better_heatmap = go.Figure(
        data=go.Heatmap(z=[port, nport], x=cat_legend, y=["portal", "not portal"])
    )
    # better_heatmap.show()
    filename = ", ".join([x for x in cat_legend]) + " cat heatplot.html"
    write_html_plot(filename, "feature plots", better_heatmap)

    # checking correlations between cont variables and response
    cat_cont_ratios = {}
    for feature in cont_legend:
        cat_cont_ratios[feature] = round(
            cat_cont_correlation_ratio(df["classification"], df[feature]), 5
        )

    # checking correlations between cat features and response
    cat_cat_ratios = {}
    for feature in cat_legend:
        cat_cat_ratios[feature] = cat_correlation(df["classification"], df[feature])

    # plotting correlations
    x_key = list(cat_cat_ratios.keys())
    x_key = x_key + (list(cat_cont_ratios.keys()))
    y_val = list(cat_cat_ratios.values())
    y_val = y_val + (list(cat_cont_ratios.values()))
    z_legend = ["categorical" for x in range(len(cat_cat_ratios.keys()))]
    z_legend = z_legend + ["continuous" for x in range(len(cat_cont_ratios.keys()))]
    y_val, x_key, z_legend = (
        list(t) for t in zip(*sorted(zip(y_val, x_key, z_legend)))
    )

    data = px.bar(y=y_val, x=x_key, color=z_legend, title="correlation ratios")
    correlation_plot = go.Figure(data=data)
    # correlation_plot.show()
    write_html_plot("correlation plot.html", "ranking plots", correlation_plot)

    # solving feature importances and plotting
    y = df["classification"].replace("portal", 1).replace("not portal", 0)
    X = (
        df.drop("classification", axis=1)
        .drop("Protein name [Species]", axis=1)
        .drop("Fasta protein sequence", axis=1)
        .drop("Length", axis=1)
    )

    rf = RandomForestRegressor()
    rf.fit(X, y)
    feature_importance = rf.feature_importances_
    feature_importances = {
        list(X.columns)[i]: feature_importance[i]
        for i in range(len(feature_importance))
    }
    x_key = list(feature_importances.keys())
    y_val = list(feature_importances.values())
    z_legend = ["continuous" for x in range(len(cont_legend))] + [
        "categorical" for x in range(len(cat_legend))
    ]
    y_val, x_key, z_legend = (
        list(t) for t in zip(*sorted(zip(y_val, x_key, z_legend)))
    )
    data = px.bar(
        y=y_val,
        x=x_key,
        color=z_legend,
        title="Feature Importances using Random Forrest",
    )
    feature_importance_plot = go.Figure(data=data)
    #  feature_importance_plot.show()
    write_html_plot(
        "feature importances.html", "ranking plots", feature_importance_plot
    )

    # p value and t score plots
    p_val_dict, t_val_dict = {}, {}
    y = df["classification"].replace("portal", 1).replace("not portal", 0).to_numpy()
    for col in cont_legend:
        column = df[col].to_numpy()
        predictor = statsmodels.api.add_constant(column)
        logistic_regression_model = statsmodels.api.Logit(y, predictor, missing="drop")
        logistic_regression_fitted = logistic_regression_model.fit()
        t_value = round(logistic_regression_fitted.tvalues[1], 4)
        p_value = float("{:.6e}".format(logistic_regression_fitted.pvalues[1]))
        p_val_dict[col] = p_value
        t_val_dict[col] = t_value

    p, t, tlegend, plegend = (
        list(p_val_dict.values()),
        list(t_val_dict.values()),
        list(t_val_dict.keys()),
        list(p_val_dict.keys()),
    )
    t = [abs(x) for x in t]
    t, tlegend = (list(z) for z in zip(*sorted(zip(t, tlegend))))
    p, plegend = (list(z) for z in zip(*sorted(zip(p, plegend))))

    ptvalplot = make_subplots(rows=2, cols=1, vertical_spacing=0.35)
    ptvalplot.add_trace(go.Bar(name="t-values", y=t, x=tlegend), row=1, col=1)
    # ptvalplot.add_trace(go.Bar(name="p-values", y=p, x=plegend), row=2, col=1)
    ptvalplot.update_xaxes(tickangle=45)
    # ptvalplot.show()
    write_html_plot("p-t values plot.html", "ranking plots", ptvalplot)

    # MSD difference with Mean of Response plotting
    response_int = list(y)
    resp_mean = float(np.mean(response_int))
    wmsd_dict = {}
    for item in cont_legend + cat_legend:
        item_list = list(df[item])
        mi, ma = min(item_list), max(item_list)
        sw = (ma - mi) / 10
        item_sorted, response_sorted = (
            list(t) for t in zip(*sorted(zip(item_list, response_int)))
        )
        msd, bin_mean_list, centers, pop_prop = [], [], [], []
        for i in range(10):
            if i != 9:
                start = i * sw + mi
                end = (i + 1) * sw + mi
            else:
                start = i * sw + mi
                end = ma + 1 + mi
            idx = [idx for idx, x in enumerate(item_sorted) if start <= x < end]

            resp_bin = [response_sorted[z] for z in idx]

            pop_prop = len(resp_bin) / len(response_sorted)
            resp_bin_mean = float(np.mean(resp_bin))
            centers.append(sw * i + sw / 2)

            msd.append(((resp_bin_mean - resp_mean) ** 2) * pop_prop)
            bin_mean_list.append(resp_bin_mean)
        plt = [
            go.Scatter(
                x=centers, y=bin_mean_list, name="response bin means", mode="lines"
            ),
            go.Bar(x=centers, y=msd, yaxis="y2", name="Weighted MSD", opacity=0.5),
            go.Scatter(
                x=[min(centers), max(centers)],
                y=[resp_mean, resp_mean],
                name="population response mean",
                mode="lines",
            ),
        ]
        layout = go.Layout(
            title="Binned Response Mean vs Population Mean",
            xaxis_title=f"predictor: {item}",
            yaxis_title=f"response: {item}",
            yaxis2=dict(title="Population", overlaying="y", anchor="y3", side="right"),
        )
        fig = go.Figure(data=plt, layout=layout)
        # fig.show()
        write_html_plot(f"{item} MSD plot.html", "msd plots", fig)
        wmsd_dict[item] = float(np.nansum(msd))

    # Plot wMSD scores
    wmsd_keys = list(wmsd_dict.keys())
    wmsd_vals = list(wmsd_dict.values())
    wmsd_vals, wmsd_keys = (list(t) for t in zip(*sorted(zip(wmsd_vals, wmsd_keys))))
    data = px.bar(y=wmsd_vals, x=wmsd_keys, title="Weighted MSD")
    wmsd_plot = go.Figure(data=data)
    # wmsd_plot.show()
    write_html_plot("wMSD plot.html", "ranking plots", wmsd_plot)

    # reduce dataset to only best scoring features
    updated_cont_legend = [
        "Isoelectric point",
        "Isoelectric point q1",
        "Isoelectric point q5",
        "instability index",
        "aromaticity index",
        "acidic fraction",
        "basic fraction",
        "acidic fraction q1",
        "acidic fraction q4",
        "acidic fraction q5",
        "basic fraction q1",
        "basic fraction q2",
        "basic fraction q3",
        "basic fraction q4",
        "basic fraction q5",
        "iso * acid and base",
        "molar extinction coefficient 1",
        "molar extinction coefficient 2",
        "gravy score",
        "gravy score q5",
        "molecular weight",
        "peak flexibility smoothed",
        "peak flexibility position",
        "mean flexibility at end",
        "Leu-Leu dimer frequency",
        "Sheet secondary fraction",
    ]
    updated_cat_legend = ["more acidic", "acidic isoelectric point"]

    cat_remove = [x for x in cat_legend if x not in updated_cat_legend]
    cont_remove = [y for y in cont_legend if y not in updated_cont_legend]
    for each in cat_remove:
        df = df.drop(each, axis=1)
    for each in cont_remove:
        df = df.drop(each, axis=1)
    df.reset_index(inplace=True, drop=True)

    # brute force variable combinations MSD scores
    for feat1 in updated_cont_legend:
        for feat2 in updated_cont_legend:
            feat1_list = list(df[feat1])
            feat2_list = list(df[feat2])
            response_list = list(
                df["classification"].replace("portal", 1).replace("not portal", 0)
            )
            z = brute_force_cont_msd(feat1_list, feat2_list, response_list, 10)
            surface_msd_plot = go.Figure(data=[go.Surface(z=z)])

            surface_msd_plot.update_layout(
                title=f"{feat1} vs {feat2}", autosize=False, width=500, height=500
            )
            write_html_plot(
                f"{feat1} v {feat2} brute force plot.html",
                "brute force plots",
                surface_msd_plot,
            )

    # setup df as final output table 1
    output_table = pd.DataFrame(
        columns=[
            "Name",
            "t-val",
            "RF Feature Importance",
            "Correlation Ratio",
            "wMSD score",
        ]
    )
    cwd = os.getcwd()
    for feature1 in updated_cont_legend:
        fname = feature1
        t_val = t_val_dict[feature1]
        rf = feature_importances[feature1]
        corr = cat_cont_ratios[feature1]
        wmsd = wmsd_dict[feature1]

        for (root, dirs, files) in os.walk(cwd):
            for name in files:
                fullname = os.path.join(root, name)
                if feature1 in fullname:
                    if "MSD plot" in fullname:
                        if fullname[:-14].endswith(feature1):
                            wmsd = str(round(float(wmsd), 4)) + "#" + fullname
                    if "feature plots" in fullname:
                        fname = str(fname) + "#" + fullname
                if "p-t" in fullname:
                    t_val = str(round(float(t_val), 4)) + "#" + fullname
                if "importances" in fullname:
                    rf = str(round(float(rf), 4)) + "#" + fullname
                if "correlation" in fullname:
                    corr = str(round(float(corr), 4)) + "#" + fullname

        output_table.loc[len(output_table)] = [fname, t_val, rf, corr, wmsd]

    for feature2 in updated_cat_legend:
        fname = feature2
        rf = feature_importances[feature2]
        corr = cat_cat_ratios[feature2]
        wmsd = wmsd_dict[feature2]

        for (root, dirs, files) in os.walk(cwd):
            for name in files:
                fullname = os.path.join(root, name)
                if feature2 in fullname:
                    if "MSD plot" in fullname:
                        if fullname[:-14].endswith(feature2):
                            wmsd = str(round(float(wmsd), 4)) + "#" + fullname
                    if "feature plots" in fullname:
                        fname = str(fname) + "#" + fullname
                if "importances" in fullname:
                    rf = str(round(float(rf), 4)) + "#" + fullname
                if "correlation" in fullname:
                    corr = str(round(float(corr), 4)) + "#" + fullname

        output_table.loc[len(output_table)] = [fname, "-", rf, corr, wmsd]

    output_table = output_table.style.format(make_clickable1).render()
    with open("final assignment output.html", "w") as file:
        file.write(output_table)

    # train-test split
    df_shuffled = shuffle(df)
    df_shuffled.reset_index(inplace=True, drop=True)

    y = df_shuffled["classification"].replace("portal", 1).replace("not portal", 0)
    X = (
        df_shuffled.drop("classification", axis=1)
        .drop("Protein name [Species]", axis=1)
        .drop("Fasta protein sequence", axis=1)
        .drop("Length", axis=1)
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=200
    )

    # build SVM model
    svc_pipeline = Pipeline(
        [
            ("standard scaler", StandardScaler()),
            ("SVC", svm.SVC(probability=True, random_state=12345)),
        ]
    )
    svc_pipeline.fit(X_train, y_train)
    pickle.dump(svc_pipeline, open("svc_model.pkl", "wb"))
    svcprediction = list(svc_pipeline.predict(X_test))

    # check accuracy
    svc_accuracy = sum(1 for a, b in zip(svcprediction, y_test) if a == b) / len(
        svcprediction
    )

    svc_positive_pred_idx = [i for i, x in enumerate(svcprediction) if x == 1]
    svc_negative_pred_idx = [i for i, x in enumerate(svcprediction) if x == 0]

    svc_pos_real = [list(y_test)[x] for x in svc_positive_pred_idx]
    svc_neg_real = [list(y_test)[x] for x in svc_negative_pred_idx]

    svc_TP = svc_pos_real.count(1) / len(svc_positive_pred_idx)
    svc_FN = svc_neg_real.count(1) / len(svc_negative_pred_idx)
    svc_FP = svc_pos_real.count(0) / len(svc_positive_pred_idx)
    svc_TN = svc_neg_real.count(0) / len(svc_negative_pred_idx)

    svc_positive_predictive_value = svc_TP / (svc_TP + svc_FP)
    svc_negative_predictive_value = svc_TN / (svc_TN + svc_FN)

    print(svc_accuracy, svc_positive_predictive_value, svc_negative_predictive_value)

    with open("model_test_score.txt", "w") as model_score_txt:
        model_score_txt.write(
            f"""
SVM model testing output using randomized train-test split of 80/20.

ACCURACY: \t\t\t{round(svc_accuracy, 4)}
POSITIVE PREDICTIVE VALUE: \t{round(svc_positive_predictive_value, 4)}
NEGATIVE PREDICTIVE VALUE: \t{round(svc_negative_predictive_value, 4)}

These values are not related to the user input. This is purely for assessing
model performance prior to running the user provided sequences."""
        )


if __name__ == "__main__":
    main()
