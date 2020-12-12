#!/Users/michellean/workspace/courses/bda696/src/BDA696/.venv/bin/python3

# import csv
# import fnmatch
# import os
import pickle
# import random
import warnings

import numpy as np
import pandas as pd
# import plotly.express as px
import plotly.graph_objects as go
# import statsmodels.api
# from Bio.SeqUtils.ProtParam import ProteinAnalysis
# from plotly.subplots import make_subplots
from scipy import stats
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.utils import shuffle

# from statistics import mean


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
    for violin_feature in cont_legend[n1:n2]:
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
    better_violin.show()
    return


def main():
    # limit = 10000000
    # num_rows = 0
    # for filename in os.listdir(os.getcwd()):
    #     if filename.endswith(".fasta"):
    #         with open(filename, "r") as file:
    #             num_rows += min(file.read().count(">"), limit)
    #
    # # num_rows = 11000  # DELETE LATER!!!
    #
    # array_float = np.zeros(shape=(num_rows, 51))
    # array_str = np.empty(shape=(num_rows, 3), dtype="O")
    # array_bool = np.zeros(shape=(num_rows, 5))
    # # array = np.concatenate((array_str, array_float), axis=1)
    #
    # row_num = 0
    #
    # for filename in os.listdir(os.getcwd()):
    #     if filename.endswith(".fasta"):
    #         with open(filename, "r") as file:
    #             file_string = file.read()
    #             file_list = [x.strip("\n") for x in file_string.split(">")]
    #             counter = 0
    #             count = file_string.count(">")
    #             print(f"{filename} \t({count})")
    #             for each in file_list[1:]:
    #                 classification_str = (
    #                     filename.split(".")[0].strip("_100").strip("_100x")
    #                 )
    #                 classification = (
    #                     "portal" if classification_str == "portal" else "not portal"
    #                 )
    #                 counter += 1
    #                 lines = each.split("\n")
    #                 name_of_protein = lines[0]
    #                 sequence = "".join(lines[1:])
    #                 # convert to uppercase so Protein Analysis can handle it okay
    #                 seqq = sequence.upper()
    #                 # take sequence, convert to uppercase, and update any weird findings to correct amino acid
    #                 seqqq = (
    #                     seqq.replace("X", "A")
    #                     .replace("J", "L")
    #                     .replace("*", "A")
    #                     .replace("Z", "E")
    #                     .replace("B", "D")
    #                 )
    #                 X = ProteinAnalysis(seqqq)
    #                 length = len(seqqq)
    #
    #                 # setting up quintiles
    #                 quint_size = int(round(length / 5, 0))
    #                 seqqq_1 = seqqq[:quint_size]
    #                 seqqq_2 = seqqq[quint_size : quint_size * 2]
    #                 seqqq_3 = seqqq[quint_size * 2 : quint_size * 3]
    #                 seqqq_4 = seqqq[quint_size * 3 : quint_size * 4]
    #                 seqqq_5 = seqqq[quint_size * 4 :]
    #                 quintiles = [seqqq_1, seqqq_2, seqqq_3, seqqq_4, seqqq_5]
    #                 quintiles_analyzed = [ProteinAnalysis(x) for x in quintiles]
    #
    #                 # isoelectric points
    #                 isoelectric = X.isoelectric_point()
    #                 quintile_isoelectric = [
    #                     x.isoelectric_point() for x in quintiles_analyzed
    #                 ]
    #
    #                 # instability
    #                 instability = X.instability_index()
    #
    #                 # aromaticity
    #                 aromaticity = X.aromaticity()
    #
    #                 # acidic/basic
    #                 acidic_fraction = (seqqq.count("E") + seqqq.count("D")) / length
    #                 basic_fraction = (
    #                     seqqq.count("R") + seqqq.count("H") + seqqq.count("K")
    #                 ) / length
    #                 quintile_acidic = [
    #                     (x.count("E") + x.count("D")) / length for x in quintiles
    #                 ]
    #                 quintile_basic = [
    #                     (x.count("R") + x.count("H") + x.count("K")) / length
    #                     for x in quintiles
    #                 ]
    #
    #                 # isoelectric point * (acidic + basic)
    #                 iso_times_a_and_b = isoelectric * (acidic_fraction + basic_fraction)
    #
    #                 # extinction coefficients
    #                 molar_extinction_1 = X.molar_extinction_coefficient()[0]
    #                 molar_extinction_2 = X.molar_extinction_coefficient()[1]
    #
    #                 # gravy scores
    #                 gravy_score = X.gravy()
    #                 quintile_gravy = [x.gravy() for x in quintiles_analyzed]
    #
    #                 # molecular weight
    #                 mol_weight = X.molecular_weight()
    #
    #                 # flexibility
    #                 flex_list = X.flexibility()
    #                 flex_mean = np.mean(X.flexibility())
    #                 quintile_flex_mean = [
    #                     np.mean(x.flexibility()) for x in quintiles_analyzed
    #                 ]
    #                 # smoothing
    #                 flex_list_smoothed = [
    #                     np.mean(flex_list[i : i + 5]) for i in range(len(flex_list) - 5)
    #                 ]
    #                 highest_flex_smoothed = max(flex_list_smoothed)
    #                 highest_flex_position = flex_list_smoothed.index(
    #                     highest_flex_smoothed
    #                 ) / len(flex_list_smoothed)
    #                 # thresholds
    #                 high_thresholds = [1.01, 1.03, 1.04]
    #                 low_thresholds = [0.99, 0.97, 0.96]
    #                 flex_above_1 = sum(x > high_thresholds[0] for x in flex_list) / len(
    #                     flex_list
    #                 )
    #                 flex_above_2 = sum(x > high_thresholds[1] for x in flex_list) / len(
    #                     flex_list
    #                 )
    #                 flex_above_3 = sum(x > high_thresholds[2] for x in flex_list) / len(
    #                     flex_list
    #                 )
    #                 flex_below_1 = sum(x > low_thresholds[0] for x in flex_list) / len(
    #                     flex_list
    #                 )
    #                 flex_below_2 = sum(x > low_thresholds[1] for x in flex_list) / len(
    #                     flex_list
    #                 )
    #                 flex_below_3 = sum(x > low_thresholds[2] for x in flex_list) / len(
    #                     flex_list
    #                 )
    #                 # ends
    #                 range20pct = int(round(len(flex_list) / 10, 0))
    #                 flex_ends = flex_list[:range20pct] + flex_list[-range20pct:]
    #                 mean_flex_ends = np.mean(flex_ends)
    #
    #                 # Leu-Leu dimer freq
    #                 leu_leu_dimer_freq = seqqq.count("LL") / length
    #
    #                 # trp / molar extinction coefficient 1
    #                 try:
    #                     molar_ext_coeff_div_trp = molar_extinction_1 / (
    #                         seqqq.count("W") / length
    #                     )
    #                 except ZeroDivisionError:
    #                     molar_ext_coeff_div_trp = 0
    #
    #                 # secondary fractions
    #                 (
    #                     helix_fraction,
    #                     turn_fraction,
    #                     sheet_fraction,
    #                 ) = X.secondary_structure_fraction()
    #
    #                 # T/F more acidic
    #                 more_acidic = 1 if acidic_fraction >= basic_fraction else 0
    #
    #                 # T/F flexibility > 1.0
    #                 more_flex = 1 if flex_mean > 1 else 0
    #
    #                 # T/F acidic isoelectric point
    #                 acidic_isoelectric_point = 1 if isoelectric < 7 else 0
    #
    #                 # T/F is there a long repeat?
    #                 long_repeat = 0
    #                 forward, backward = seqqq, seqqq[::-1]
    #                 while len(forward) > 8:
    #                     check = forward[:8]
    #                     if check in forward[1:]:
    #                         long_repeat = 1
    #                         forward = "done"
    #                     else:
    #                         forward, backward = forward[1:], backward[:-1]
    #
    #                 # T/F is there a short repeat?
    #                 short_repeat = 0
    #                 forward, backward = seqqq, seqqq[::-1]
    #                 while len(forward) > 5:
    #                     check = forward[:5]
    #                     if check in forward[1:]:
    #                         short_repeat = 1
    #                         forward = "done"
    #                     else:
    #                         forward, backward = forward[1:], backward[:-1]
    #
    #                 # add new values to 1-D arrays for strings and floats
    #                 new_array_row_floats = [
    #                     length,
    #                     isoelectric,
    #                     quintile_isoelectric[0],
    #                     quintile_isoelectric[1],
    #                     quintile_isoelectric[2],
    #                     quintile_isoelectric[3],
    #                     quintile_isoelectric[4],
    #                     instability,
    #                     aromaticity,
    #                     acidic_fraction,
    #                     basic_fraction,
    #                     quintile_acidic[0],
    #                     quintile_acidic[1],
    #                     quintile_acidic[2],
    #                     quintile_acidic[3],
    #                     quintile_acidic[4],
    #                     quintile_basic[0],
    #                     quintile_basic[1],
    #                     quintile_basic[2],
    #                     quintile_basic[3],
    #                     quintile_basic[4],
    #                     iso_times_a_and_b,
    #                     molar_extinction_1,
    #                     molar_extinction_2,
    #                     gravy_score,
    #                     quintile_gravy[0],
    #                     quintile_gravy[1],
    #                     quintile_gravy[2],
    #                     quintile_gravy[3],
    #                     quintile_gravy[4],
    #                     mol_weight,
    #                     flex_mean,
    #                     quintile_flex_mean[0],
    #                     quintile_flex_mean[1],
    #                     quintile_flex_mean[2],
    #                     quintile_flex_mean[3],
    #                     quintile_flex_mean[4],
    #                     highest_flex_smoothed,
    #                     highest_flex_position,
    #                     flex_above_1,
    #                     flex_above_2,
    #                     flex_above_3,
    #                     flex_below_1,
    #                     flex_below_2,
    #                     flex_below_3,
    #                     mean_flex_ends,
    #                     leu_leu_dimer_freq,
    #                     molar_ext_coeff_div_trp,
    #                     helix_fraction,
    #                     turn_fraction,
    #                     sheet_fraction,
    #                 ]
    #                 new_array_row_str = [classification, name_of_protein, sequence]
    #
    #                 new_array_row_bool = [
    #                     more_acidic,
    #                     more_flex,
    #                     acidic_isoelectric_point,
    #                     long_repeat,
    #                     short_repeat,
    #                 ]
    #
    #                 array_float[row_num] = new_array_row_floats
    #                 array_str[row_num] = new_array_row_str
    #                 array_bool[row_num] = new_array_row_bool
    #                 row_num += 1
    #
    #                 if counter % 100 == 0:
    #                     print(
    #                         f"\t {counter} / {min(count, limit)} "
    #                         f"({round(counter / min(count, limit) * 100, 2)}%)",
    #                         end="\r",
    #                     )
    #
    #                 if counter >= limit:
    #                     break
    #
    # # cast array as dict
    # print("casting array as dict...", end="\r")
    # array_to_dict = {
    #     "classification": array_str[:, 0],
    #     "Protein name [Species]": array_str[:, 1],
    #     "Fasta protein sequence": array_str[:, 2],
    #     "Length": array_float[:, 0],
    #     "Isoelectric point": array_float[:, 1],
    #     "Isoelectric point q1": array_float[:, 2],
    #     "Isoelectric point q2": array_float[:, 3],
    #     "Isoelectric point q3": array_float[:, 4],
    #     "Isoelectric point q4": array_float[:, 5],
    #     "Isoelectric point q5": array_float[:, 6],
    #     "instability index": array_float[:, 7],
    #     "aromaticity index": array_float[:, 8],
    #     "acidic fraction": array_float[:, 9],
    #     "basic fraction": array_float[:, 10],
    #     "acidic fraction q1": array_float[:, 11],
    #     "acidic fraction q2": array_float[:, 12],
    #     "acidic fraction q3": array_float[:, 13],
    #     "acidic fraction q4": array_float[:, 14],
    #     "acidic fraction q5": array_float[:, 15],
    #     "basic fraction q1": array_float[:, 16],
    #     "basic fraction q2": array_float[:, 17],
    #     "basic fraction q3": array_float[:, 18],
    #     "basic fraction q4": array_float[:, 19],
    #     "basic fraction q5": array_float[:, 20],
    #     "iso * acid and base": array_float[:, 21],
    #     "molar extinction coefficient 1": array_float[:, 22],
    #     "molar extinction coefficient 2": array_float[:, 23],
    #     "gravy score": array_float[:, 24],
    #     "gravy score q1": array_float[:, 25],
    #     "gravy score q2": array_float[:, 26],
    #     "gravy score q3": array_float[:, 27],
    #     "gravy score q4": array_float[:, 28],
    #     "gravy score q5": array_float[:, 29],
    #     "molecular weight": array_float[:, 30],
    #     "flexibility mean": array_float[:, 31],
    #     "flexibility mean q1": array_float[:, 32],
    #     "flexibility mean q2": array_float[:, 33],
    #     "flexibility mean q3": array_float[:, 34],
    #     "flexibility mean q4": array_float[:, 35],
    #     "flexibility mean q5": array_float[:, 36],
    #     "peak flexibility smoothed": array_float[:, 37],
    #     "peak flexibility position": array_float[:, 38],
    #     "flex ratio above 1": array_float[:, 39],
    #     "flex ratio above 2": array_float[:, 40],
    #     "flex ratio above 3": array_float[:, 41],
    #     "flex ratio below 1": array_float[:, 42],
    #     "flex ratio below 2": array_float[:, 43],
    #     "flex ratio below 3": array_float[:, 44],
    #     "mean flexibility at end": array_float[:, 45],
    #     "Leu-Leu dimer frequency": array_float[:, 46],
    #     "molar ext coeff div by Trp frequency ": array_float[:, 47],
    #     "Helix secondary fraction": array_float[:, 48],
    #     "Turn secondary fraction": array_float[:, 49],
    #     "Sheet secondary fraction": array_float[:, 50],
    #     "more acidic": array_bool[:, 0],
    #     "more flex": array_bool[:, 1],
    #     "acidic isoelectric point": array_bool[:, 2],
    #     "long repeat": array_bool[:, 3],
    #     "short repeat": array_bool[:, 4],
    # }
    #
    # # write file
    # print("casting dict as dataframe...", end="\r")
    # df = pd.DataFrame.from_dict(array_to_dict)
    #
    # # drop rows where length is < 120 AA long or where too many (more than half) of values in row are 0
    # df = df[df["Length"] > 120]
    # n = (df.shape[1] - 1) // 2
    # df = df[df.eq(0).sum(1) < n]
    # df.reset_index(inplace=True, drop=True)
    #
    # print("writing file...", end="\r")
    # df.to_csv(r"flexibility_secondary_etc.csv", index=False, header=True)

    # load csv
    df = pd.read_csv("flexibility_secondary_etc.csv", header=0)
    print(df.head())

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
    # plot_violin(0, 10, cont_legend, df)
    # plot_violin(10, 20, cont_legend, df)
    # plot_violin(20, 30, cont_legend, df)
    # plot_violin(30, 40, cont_legend, df)
    # plot_violin(40, 50, cont_legend, df)

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
    better_heatmap.show()
    exit()
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

    # data = px.bar(y=y_val, x=x_key, color=z_legend, title="correlation ratios")
    # correlation_plot = go.Figure(data=data)
    # correlation_plot.show()

    # solving feature importances and plotting
    # y = df["classification"].replace("portal", 1).replace("not portal", 0)
    # X = (
    #     df.drop("classification", axis=1)
    #     .drop("Protein name [Species]", axis=1)
    #     .drop("Fasta protein sequence", axis=1)
    #     .drop("Length", axis=1)
    # )
    #
    # rf = RandomForestRegressor()
    # rf.fit(X, y)
    # feature_importance = rf.feature_importances_
    # feature_importances = {
    #     list(X.columns)[i]: feature_importance[i]
    #     for i in range(len(feature_importance))
    # }
    # x_key = list(feature_importances.keys())
    # y_val = list(feature_importances.values())
    # z_legend = ["continuous" for x in range(len(cont_legend))] + [
    #     "categorical" for x in range(len(cat_legend))
    # ]
    # y_val, x_key, z_legend = (
    #     list(t) for t in zip(*sorted(zip(y_val, x_key, z_legend)))
    # )
    # data = px.bar(
    #     y=y_val,
    #     x=x_key,
    #     color=z_legend,
    #     title="Feature Importances using Random Forrest",
    # )
    # feature_importance_plot = go.Figure(data=data)
    # # feature_importance_plot.show()

    # p value and t score plots
    # p_val_dict, t_val_dict = {}, {}
    # y = df["classification"].replace("portal", 1).replace("not portal", 0).to_numpy()
    # for col in cont_legend:
    #     column = df[col].to_numpy()
    #     predictor = statsmodels.api.add_constant(column)
    #     logistic_regression_model = statsmodels.api.Logit(y, predictor, missing="drop")
    #     logistic_regression_fitted = logistic_regression_model.fit()
    #     t_value = round(logistic_regression_fitted.tvalues[1], 4)
    #     p_value = float("{:.6e}".format(logistic_regression_fitted.pvalues[1]))
    #     print(p_value)
    #     p_val_dict[col] = p_value
    #     t_val_dict[col] = t_value
    #
    # p, t, tlegend, plegend = (
    #     list(p_val_dict.values()),
    #     list(t_val_dict.values()),
    #     list(t_val_dict.keys()),
    #     list(p_val_dict.keys()),
    # )
    # t = [abs(x) for x in t]
    # t, tlegend = (list(z) for z in zip(*sorted(zip(t, tlegend))))
    # p, plegend = (list(z) for z in zip(*sorted(zip(p, plegend))))
    #
    # ptvalplot = make_subplots(rows=2, cols=1, vertical_spacing=0.35)
    # ptvalplot.add_trace(go.Bar(name="t-values", y=t, x=tlegend), row=1, col=1)
    # # ptvalplot.add_trace(go.Bar(name="p-values", y=p, x=plegend), row=2, col=1)
    # ptvalplot.update_xaxes(tickangle=45)
    # # ptvalplot.show()

    # MSD difference with Mean of Response plotting
    # response_int = list(y)
    # resp_mean, resp_count = float(np.mean(response_int)), len(response_int)
    # wmsd_dict = {}
    # for item in cont_legend + cat_legend:
    #     item_list = list(df[item])
    #     mi, ma = min(item_list), max(item_list)
    #     sw = (ma - mi) / 10
    #     item_sorted, response_sorted = (
    #         list(t) for t in zip(*sorted(zip(item_list, response_int)))
    #     )
    #     msd, bin_mean_list, centers, pop_prop = [], [], [], []
    #     for i in range(10):
    #         if i != 9:
    #             start = i * sw + mi
    #             end = (i + 1) * sw + mi
    #         else:
    #             start = i * sw + mi
    #             end = ma + 1 + mi
    #         idx = [idx for idx, x in enumerate(item_sorted) if start <= x < end]
    #
    #         resp_bin = [response_sorted[z] for z in idx]
    #
    #         pop_prop = len(resp_bin) / len(response_sorted)
    #         resp_bin_mean = float(np.mean(resp_bin))
    #         centers.append(sw * i + sw / 2)
    #
    #         msd.append(((resp_bin_mean - resp_mean) ** 2) * pop_prop)
    #         bin_mean_list.append(resp_bin_mean)
    #     plt = [
    #         go.Scatter(
    #             x=centers, y=bin_mean_list, name="response bin means", mode="lines"
    #         ),
    #         go.Bar(x=centers, y=msd, yaxis="y2", name="Weighted MSD", opacity=0.5),
    #         go.Scatter(
    #             x=[min(centers), max(centers)],
    #             y=[resp_mean, resp_mean],
    #             name="population response mean",
    #             mode="lines",
    #         ),
    #     ]
    #     layout = go.Layout(
    #         title="Binned Response Mean vs Population Mean",
    #         xaxis_title=f"predictor: {item}",
    #         yaxis_title=f"response: {item}",
    #         yaxis2=dict(title="Population", overlaying="y", anchor="y3", side="right"),
    #     )
    #     fig = go.Figure(data=plt, layout=layout)
    #     # fig.show()
    #     wmsd_dict[item] = float(np.nansum(msd))

    # Plot wMSD scores
    # wmsd_keys = list(wmsd_dict.keys())
    # wmsd_vals = list(wmsd_dict.values())
    # wmsd_vals, wmsd_keys = (list(t) for t in zip(*sorted(zip(wmsd_vals, wmsd_keys))))
    # data = px.bar(y=wmsd_vals, x=wmsd_keys, title="Weighted MSD")
    # wmsd_plot = go.Figure(data=data)
    # # wmsd_plot.show()

    # reduce dataset to only best scoring features
    updated_cont_legend = [
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
        "gravy score q5",
        "molecular weight",
        "peak flexibility smoothed",
        "peak flexibility position",
        "mean flexibility at end",
        "Leu-Leu dimer frequency",
        "Turn secondary fraction",
        "Sheet secondary fraction",
    ]
    updated_cat_legend = ["more acidic", "more flex", "acidic isoelectric point"]

    cat_remove = [x for x in cat_legend if x not in updated_cat_legend]
    cont_remove = [y for y in cont_legend if y not in updated_cont_legend]
    for each in cat_remove:
        df = df.drop(each, axis=1)
    for each in cont_remove:
        df = df.drop(each, axis=1)
    df.reset_index(inplace=True, drop=True)

    # brute force variable combinations MSD scores
    count = 0
    for feat1 in updated_cont_legend:
        for feat2 in updated_cont_legend:
            count += 1
            print(count)
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
            if count % 5 == 0:
                surface_msd_plot.show()
                count += 1
                break
            if count > 40:
                exit()

    # train-test split
    df_shuffled = shuffle(df)
    df_shuffled.reset_index(inplace=True, drop=True)

    y = df_shuffled["classification"].replace("portal", 1).replace("not portal", 0)
    X = (
        df_shuffled.drop("classification", axis=1)
        .drop("Protein name [Species]", axis=1)
        .drop("Fasta protein sequence", axis=1)
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=200
    )

    # train_test_split_score = roc_auc_score(z, z_pred)
    # cross_validation_score = cross_val_score(random_forrest, concat_df, z, cv=5, scoring='roc_auc')

    # build RF model
    rf_pipeline = Pipeline(
        [
            ("Normalizer", Normalizer()),
            ("RandomForest", RandomForestClassifier(random_state=54321)),
        ]
    )
    rf_pipeline.fit(X_train, y_train)
    pickle.dump(rf_pipeline, open("rf_model.pkl", "wb"))
    rfprediction = list(rf_pipeline.predict(X_test))
    # rfprobability = rf_pipeline.predict_proba(X_test)

    # build KNN model
    np.random.seed(11111)
    knn_pipeline = Pipeline(
        [
            ("Normalize and Transform", Normalizer()),
            ("KNN", KNeighborsClassifier(n_neighbors=10, weights="distance")),
        ]
    )
    knn_pipeline.fit(X_train, y_train)
    pickle.dump(knn_pipeline, open("knn_model.pkl", "wb"))
    knnprediction = list(knn_pipeline.predict(X_test))
    # knnprobability = list(knn_pipeline.predict_proba(X_test))

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
    rf_accuracy = sum(1 for a, b in zip(rfprediction, y_test) if a == b) / len(
        rfprediction
    )

    rf_positive_pred_idx = [i for i, x in enumerate(rfprediction) if x == 1]
    rf_negative_pred_idx = [i for i, x in enumerate(rfprediction) if x == 0]

    rf_pos_real = [list(y_test)[x] for x in rf_positive_pred_idx]
    rf_neg_real = [list(y_test)[x] for x in rf_negative_pred_idx]

    rf_TP = rf_pos_real.count(1) / len(rf_positive_pred_idx)
    rf_FN = rf_neg_real.count(1) / len(rf_negative_pred_idx)
    rf_FP = rf_pos_real.count(0) / len(rf_positive_pred_idx)
    rf_TN = rf_neg_real.count(0) / len(rf_negative_pred_idx)

    rf_positive_predictive_value = rf_TP / (rf_TP + rf_FP)
    rf_negative_predictive_value = rf_TN / (rf_TN + rf_FN)

    knn_accuracy = sum(1 for a, b in zip(knnprediction, y_test) if a == b) / len(
        knnprediction
    )

    knn_positive_pred_idx = [i for i, x in enumerate(knnprediction) if x == 1]
    knn_negative_pred_idx = [i for i, x in enumerate(knnprediction) if x == 0]

    knn_pos_real = [list(y_test)[x] for x in knn_positive_pred_idx]
    knn_neg_real = [list(y_test)[x] for x in knn_negative_pred_idx]

    knn_TP = knn_pos_real.count(1) / len(knn_positive_pred_idx)
    knn_FN = knn_neg_real.count(1) / len(knn_negative_pred_idx)
    knn_FP = knn_pos_real.count(0) / len(knn_positive_pred_idx)
    knn_TN = knn_neg_real.count(0) / len(knn_negative_pred_idx)

    knn_positive_predictive_value = knn_TP / (knn_TP + knn_FP)
    knn_negative_predictive_value = knn_TN / (knn_TN + knn_FN)

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

    print(rf_accuracy, rf_positive_predictive_value, rf_negative_predictive_value)
    print(knn_accuracy, knn_positive_predictive_value, knn_negative_predictive_value)
    print(svc_accuracy, svc_positive_predictive_value, svc_negative_predictive_value)

    model_scores = pd.DataFrame(
        columns=["Random Forrest", "K Nearest Neighbors", "Support Vector Machine"],
        index=[
            "Positive Predictive Value",
            "Negative Predictive Value",
            "Overall Acuracy",
        ],
        data=[
            [
                rf_positive_predictive_value,
                knn_positive_predictive_value,
                svc_positive_predictive_value,
            ],
            [
                rf_negative_predictive_value,
                knn_negative_predictive_value,
                svc_negative_predictive_value,
            ],
            [rf_accuracy, knn_accuracy, svc_accuracy],
        ],
    )

    print(model_scores)

    x = ["Random Forrest", "K Nearest Neighbors", "Support Vector Machine"]
    fig = go.Figure(
        data=[
            go.Bar(
                name="Positive Predictive Value",
                x=x,
                y=[
                    rf_positive_predictive_value,
                    knn_positive_predictive_value,
                    svc_positive_predictive_value,
                ],
            ),
            go.Bar(
                name="Negative Predictive Value",
                x=x,
                y=[
                    rf_negative_predictive_value,
                    knn_negative_predictive_value,
                    svc_negative_predictive_value,
                ],
            ),
            go.Bar(
                name="Overall Accuracy",
                x=x,
                y=[rf_accuracy, knn_accuracy, svc_accuracy],
            ),
        ]
    )
    fig.update_layout(barmode="group", width=600, height=600)
    fig.show()


# try to get to a third model if possible
# compare models performance. get data and show it. plot?


if __name__ == "__main__":
    main()
