import os
import pickle

import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis


def main():
    for filename in os.listdir(os.getcwd() + "/user_inputs"):
        if filename.endswith(".fasta"):
            with open("user_inputs/" + filename, "r") as file:
                file_string = file.read().split(">")[-1]
                lines = file_string.split("\n")
                name_of_protein = lines[0]
                sequence = "".join(lines[1:])
                # convert to uppercase so Protein Analysis can handle it okay
                seqq = sequence.upper()
                # take sequence, convert to uppercase, and update any weird findings to correct amino acid
                seqqq = (
                    seqq.replace("X", "A")
                    .replace("J", "L")
                    .replace("*", "A")
                    .replace("Z", "E")
                    .replace("B", "D")
                )
                X = ProteinAnalysis(seqqq)
                length = len(seqqq)

                # setting up quintiles
                quint_size = int(round(length / 5, 0))
                seqqq_1 = seqqq[:quint_size]
                seqqq_2 = seqqq[quint_size : quint_size * 2]  # noqa: E203
                seqqq_3 = seqqq[quint_size * 2 : quint_size * 3]  # noqa: E203
                seqqq_4 = seqqq[quint_size * 3 : quint_size * 4]  # noqa: E203
                seqqq_5 = seqqq[quint_size * 4 :]  # noqa: E203
                quintiles = [seqqq_1, seqqq_2, seqqq_3, seqqq_4, seqqq_5]
                quintiles_analyzed = [ProteinAnalysis(x) for x in quintiles]

                # isoelectric points
                isoelectric = X.isoelectric_point()
                quintile_isoelectric = [
                    x.isoelectric_point() for x in quintiles_analyzed
                ]

                # instability
                instability = X.instability_index()

                # aromaticity
                aromaticity = X.aromaticity()

                # acidic/basic
                acidic_fraction = (seqqq.count("E") + seqqq.count("D")) / length
                basic_fraction = (
                    seqqq.count("R") + seqqq.count("H") + seqqq.count("K")
                ) / length
                quintile_acidic = [
                    (x.count("E") + x.count("D")) / length for x in quintiles
                ]
                quintile_basic = [
                    (x.count("R") + x.count("H") + x.count("K")) / length
                    for x in quintiles
                ]

                # isoelectric point * (acidic + basic)
                iso_times_a_and_b = isoelectric * (acidic_fraction + basic_fraction)

                # extinction coefficients
                molar_extinction_1 = X.molar_extinction_coefficient()[0]
                molar_extinction_2 = X.molar_extinction_coefficient()[1]

                # gravy scores
                gravy_score = X.gravy()
                quintile_gravy = [x.gravy() for x in quintiles_analyzed]

                # molecular weight
                mol_weight = X.molecular_weight()

                # flexibility
                flex_list = X.flexibility()

                # smoothing
                flex_list_smoothed = [
                    np.mean(flex_list[i : i + 5])  # noqa: E203
                    for i in range(len(flex_list) - 5)
                ]
                highest_flex_smoothed = max(flex_list_smoothed)
                highest_flex_position = flex_list_smoothed.index(
                    highest_flex_smoothed
                ) / len(flex_list_smoothed)

                # ends
                range20pct = int(round(len(flex_list) / 10, 0))
                flex_ends = flex_list[:range20pct] + flex_list[-range20pct:]
                mean_flex_ends = np.mean(flex_ends)

                # Leu-Leu dimer freq
                leu_leu_dimer_freq = seqqq.count("LL") / length

                # secondary fractions
                (
                    helix_fraction,
                    turn_fraction,
                    sheet_fraction,
                ) = X.secondary_structure_fraction()

                # T/F more acidic
                more_acidic = 1 if acidic_fraction >= basic_fraction else 0

                # T/F acidic isoelectric point
                acidic_isoelectric_point = 1 if isoelectric < 7 else 0

            # cast values as dict
            user_dict = {
                "Protein name [Species]": name_of_protein,
                "Fasta protein sequence": seqqq,
                "Length": length,
                "Isoelectric point": isoelectric,
                "Isoelectric point q1": quintile_isoelectric[0],
                "Isoelectric point q5": quintile_isoelectric[4],
                "instability index": instability,
                "aromaticity index": aromaticity,
                "acidic fraction": acidic_fraction,
                "basic fraction": basic_fraction,
                "acidic fraction q1": quintile_acidic[0],
                "acidic fraction q4": quintile_acidic[3],
                "acidic fraction q5": quintile_acidic[4],
                "basic fraction q1": quintile_basic[0],
                "basic fraction q2": quintile_basic[1],
                "basic fraction q3": quintile_basic[2],
                "basic fraction q4": quintile_basic[3],
                "basic fraction q5": quintile_basic[4],
                "iso * acid and base": iso_times_a_and_b,
                "molar extinction coefficient 1": molar_extinction_1,
                "molar extinction coefficient 2": molar_extinction_2,
                "gravy score": gravy_score,
                "gravy score q5": quintile_gravy[4],
                "molecular weight": mol_weight,
                "peak flexibility smoothed": highest_flex_smoothed,
                "peak flexibility position": highest_flex_position,
                "mean flexibility at end": mean_flex_ends,
                "Leu-Leu dimer frequency": leu_leu_dimer_freq,
                "Sheet secondary fraction": sheet_fraction,
                "more acidic": more_acidic,
                "acidic isoelectric point": acidic_isoelectric_point,
            }
            df = pd.DataFrame(user_dict, index=[0])
            X = df.drop(
                ["Protein name [Species]", "Fasta protein sequence", "Length"], axis=1
            )
            svc_pipeline = pickle.load(open("svc_model.pkl", "rb"))
            svcprediction = svc_pipeline.predict(X)

            svcprediction = "portal" if svcprediction == [1] else "not portal"

            df = df.transpose()
            df_csv = df.to_csv(sep="\t", index=True, header=False)

            with open(f"user_inputs/{filename}_predictions.txt", "w") as file:
                file.write(
                    f"""
PREDICTION:\t{svcprediction}

{filename}
{name_of_protein}



{df_csv}
"""
                )


if __name__ == "__main__":
    main()
