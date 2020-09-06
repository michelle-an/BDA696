#!/Users/michellean/workspace/courses/bda696/src/BDA696/venv/bin/python3

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import Normalizer

# Load Fisher's Iris dataset into a Pandas DataFrame
df = pd.read_csv('iris.data', header=None)
df.columns = ['sepal length in cm', 'sepal width in cm', 'petal length in cm', 'petal width in cm', 'class']
print(df.head())

data = df.to_numpy()

# Simple summary statistics(mean, min, max, quartiles) using numpy
sepal_length_mean, sepal_width_mean, petal_length_mean, petal_width_mean = data[:, 0:4].mean(axis=0)
sepal_length_mean, sepal_width_mean, petal_length_mean, petal_width_mean = round(sepal_length_mean, 3), \
                        round(sepal_width_mean, 3), round(petal_length_mean, 3), round(petal_width_mean, 3)

## Repeat for min 
sepal_length_min, sepal_width_min, petal_length_min, petal_width_min = data[:, 0:4].min(axis=0)
sepal_length_min, sepal_width_min, petal_length_min, petal_width_min = round(sepal_length_min, 3), \
                        round(sepal_width_min, 3), round(petal_length_min, 3), round(petal_width_min, 3)


## Repeat for max
sepal_length_max, sepal_width_max, petal_length_max, petal_width_max = data[:, 0:4].max(axis=0)
sepal_length_max, sepal_width_max, petal_length_max, petal_width_max = round(sepal_length_max, 3), \
                        round(sepal_width_max, 3), round(petal_length_max, 3), round(petal_width_max, 3)

print()
print("mean sepal length:\t {}\tmin: {}\tmax: {}\n"
     "mean sepal width:\t {}\tmin: {}\tmax: {}\n"
     "mean petal length:\t {}\tmin: {}\tmax: {}\n"
     "mean petal width:\t {}\tmin: {}\tmax: {}".format(sepal_length_mean, sepal_length_min,
                                                       sepal_length_max, sepal_width_mean,
                                                       sepal_width_min, sepal_width_max,
                                                       petal_length_mean, petal_length_min,
                                                       petal_length_max, petal_width_mean,
                                                       petal_width_min, petal_width_max))
## Generate quartiles
sepal_len_q0, sepal_len_q1, sepal_len_q2, sepal_len_q3, sepal_len_q4 = np.quantile(data[:, 0], 0), \
               np.quantile(data[:, 0], 0.25), np.quantile(data[:, 0], 0.50), np.quantile(data[:, 0], 0.75), \
                                                                                 np.quantile(data[:, 0], 1)

sepal_width_q0, sepal_width_q1, sepal_width_q2, sepal_width_q3, sepal_width_q4 = np.quantile(data[:, 1], 0), \
                np.quantile(data[:, 1], 0.25), np.quantile(data[:, 1], 0.5), np.quantile(data[:, 1], 0.75), \
                                                                                 np.quantile(data[:, 1], 1)

petal_len_q0, petal_len_q1, petal_len_q2, petal_len_q3, petal_len_q4 = np.quantile(data[:, 2], 0), \
                np.quantile(data[:, 2], 0.25), np.quantile(data[:, 2], 0.50), np.quantile(data[:, 2], 0.75), \
                                                                       np.quantile(data[:, 2], 1)


petal_width_q0, petal_width_q1, petal_width_q2, petal_width_q3, petal_width_q4 = np.quantile(data[:, 3], 0), \
                np.quantile(data[:, 3], 0.25), np.quantile(data[:, 3], 0.5), np.quantile(data[:, 3], 0.75), \
                                                                                 np.quantile(data[:, 3], 1)

print()
print("sepal quartiles: {}\t{}\t{}\t{}\t{}".format(sepal_len_q0, sepal_len_q1, sepal_len_q2, sepal_len_q3,
                                                   sepal_len_q4))
print("sepal quartiles: {}\t{}\t{}\t{}\t{}".format(sepal_width_q0, sepal_width_q1, sepal_width_q2, sepal_width_q3,
                                                   sepal_width_q4))
print("petal quartiles: {}\t{}\t{}\t{}\t{}".format(petal_len_q0, petal_len_q1, petal_len_q2, petal_len_q3,
                                                   petal_len_q4))
print("petal quartiles: {}\t{}\t{}\t{}\t{}".format(petal_width_q0, petal_width_q1, petal_width_q2, petal_width_q3,
                                                   petal_width_q4))


# --------PLOT DATA--------
# scatter plot of petal lengths vs. petal widths
petal_length_v_width_plot = px.scatter(data, x=2, y=3)
petal_length_v_width_plot.update_layout(title_text="petal lengths vs. petal widths")
petal_length_v_width_plot.update_xaxes(ticks="inside", title_text='Petal Lengths (cm)')
petal_length_v_width_plot.update_yaxes(ticks="inside", title_text='Petal Widths (cm)')

# violin plot of sepal lengths
# sepal_lengths_violin_plot = px.violin(data, y=0:4)

sepal_lengths_violin_plot = px.violin(data, y=0, color=4, violinmode='overlay')
sepal_lengths_violin_plot.update_layout(title_text="violin plot of sepal lengths",)
sepal_lengths_violin_plot.update_yaxes(title_text='sepal length (cm)')

# pie chart of different classes observed
setosa = data[:, 4].tolist().count('Iris-setosa')
versicolor = data[:, 4].tolist().count('Iris-versicolor')
virginica = data[:, 4].tolist().count('Iris-virginica')
labels = ['Setosa', 'Versicolor', 'Virginica']
values = [setosa, versicolor, virginica]
pie_chart_classes = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+text+value+percent',
                                           insidetextorientation='auto', hole=.3)])

# box plot of petal lengths of each class
box_plot_of_petal_lengths = px.box(data, x=4, y=2, points="all")
box_plot_of_petal_lengths.update_xaxes(title='Unique Species')
box_plot_of_petal_lengths.update_yaxes(title='Petal Lengths (cm)')

# 2-D histogram (heatmap) of petal width and sepal width
hist_2d = px.density_heatmap(data, x=3, y=1)
hist_2d.update_xaxes(title='Petal Width (cm)')
hist_2d.update_yaxes(title='Sepal Width (cm)')

# show plots
petal_length_v_width_plot.show()
sepal_lengths_violin_plot.show()
pie_chart_classes.show()
box_plot_of_petal_lengths.show()
hist_2d.show()





# -----Analyze and build models-----
# Normalizer transformer
transformer = Normalizer().fit(data)
transformer.transform(data)

# Random forest classifier (try other classifers)