import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/02_outliers_removed_chauvenets.pkl")
predictor_columns = list(df.columns[:6])

plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------
df.info()
df.set.value_counts()
df[df["set"] == 35]["gyro_y"].plot()

# interpolating the missing values (outliers) by a line
for col in predictor_columns:
    df[col] = df[col].interpolate()

df.info()
df.category.value_counts()

# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------
df[df["set"] == 1]["gyro_y"].plot()

duration = df[df["set"] == 1].index[-1] - df[df["set"] == 1].index[0]
duration.seconds

for s in df["set"].unique():
    start = df[df["set"] == s].index[0]
    stop = df[df["set"] == s].index[-1]
    duration = stop - start

    df.loc[df["set"] == s, "duration"] = duration.seconds

df[df["set"] == 64]

# dataframes allow groupby

duration_df = df.groupby(["category"])["duration"].mean()
duration_df

# heavy sets has 5 repetitions
# medium sets has 10 repetitions
# so...
duration_df.iloc[0] / 5
duration_df.iloc[1] / 10

# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------
df_lowpass = df.copy()
low_pass = LowPassFilter()
fs = 1000 / 200
cutoff = 1.2  # increasing the cutoff frequency increases rough lines


# sampling frequency is 200ms
# cutoff frequency is kind of the threshold to reduce noise

df_low_pass = low_pass.low_pass_filter(df_lowpass, "acc_x", fs, cutoff, order=5)

subset = df_low_pass[df_low_pass["set"] == 7]
fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 15))
subset.columns
ax[0].plot(subset["acc_x"])
ax[1].plot(subset["acc_x_lowpass"])

# apply it to all columns
for col in predictor_columns:
    df_low_pass = low_pass.low_pass_filter(df_lowpass, col, fs, cutoff, order=5)
    df_low_pass[col] = df_low_pass[col + "_lowpass"]
    del df_low_pass[col + "_lowpass"]

# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------
df_pca = df_low_pass.copy()
PCA = PrincipalComponentAnalysis()
pc_values = PCA.determine_pc_explained_variance(df_pca, predictor_columns)
# determine_pc_explained_variance returns the variancd explained by each column

# use elbow method (same as k means clustering) to determine the optimal number of pc
plt.plot(range(1, len(pc_values) + 1), pc_values)
plt.xlabel("Number of Principal components")
plt.ylabel("Explained Variance (cumulative variance)")
plt.show()

# we select the # PC as 3
df_pca = PCA.apply_pca(df_pca, predictor_columns, 3)

subset = df_pca[df_pca["set"] == 10]
subset[["pca_1", "pca_2", "pca_3"]].plot()


# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------

df_squared = df_pca.copy()

acc_r = df_squared["acc_x"] ** 2 + df_squared["acc_y"] ** 2 + df_squared["acc_z"] ** 2
gyro_r = (
    df_squared["gyro_x"] ** 2 + df_squared["gyro_y"] ** 2 + df_squared["gyro_z"] ** 2
)

df_squared["acc_r"] = np.sqrt(acc_r)
df_squared["gyro_r"] = np.sqrt(gyro_r)

subset = df_squared[df_squared["set"] == 30]

subset[["acc_r", "gyro_r"]].plot(subplots=True)
plt.show()

# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------
df_temportal = df_squared.copy()
len(df_temportal.columns)
num_Abstraction = NumericalAbstraction()
predictor_columns = predictor_columns + ["acc_r", "gyro_r"]

# rolling window is the size tht we want to look back
ws = int(1000 / 200)  # we fix the window size as 1 sec
# the step here is 200ms so to get 1 sec we need 5 steps

# for col in predictor_columns:
#     df_temportal = num_Abstraction.abstract_numerical(df_temportal,[col],ws,"mean")
#     df_temportal = num_Abstraction.abstract_numerical(df_temportal,[col],ws,"std")

df_temportal_list = []
for s in df_temportal["set"].unique():
    subset = df_temportal[df_temportal["set"] == s]
    for col in predictor_columns:
        subset = num_Abstraction.abstract_numerical(subset, [col], ws, "mean")
        subset = num_Abstraction.abstract_numerical(subset, [col], ws, "std")
    df_temportal_list.append(subset)
df_temportal = pd.concat(df_temportal_list)

df_temportal.info()

subset[["acc_y", "acc_y_temp_mean_ws_5", "acc_y_temp_std_ws_5"]].plot()
subset[["gyro_y", "gyro_y_temp_mean_ws_5", "gyro_y_temp_std_ws_5"]].plot()

# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------
df_frequency = df_temportal.copy().reset_index()
freq_abs = FourierTransformation()
# using fourier transformation to find the underlying frequencies and amplitudes
# helps for activity recognition

fs = int(1000 / 200)  # sampling rate
ws = int(2000 / 200)  # window size avg len of each repetition

df_frequency = freq_abs.abstract_frequency(df_frequency, ["acc_y"], ws, fs)
subset = df_frequency[df_frequency["set"] == 10]
subset["acc_y"].plot()
subset.columns
subset[
    [
        "acc_y_max_freq",
        "acc_y_freq_weighted",
        "acc_y_pse",
        "acc_y_temp_mean_ws_5",
        "acc_y_temp_std_ws_5",
    ]
].plot()

df_frequency_list = []

for s in df_frequency["set"].unique():
    print(f"applying fourier transfomation on the set {s}")

    subset = df_frequency[df_frequency["set"] == s].copy().reset_index(drop=True)
    subset = freq_abs.abstract_frequency(subset, predictor_columns, ws, fs)
    df_frequency_list.append(subset)

df_frequency = pd.concat(df_frequency_list).set_index("epoch (ms)", drop=True)

for col in df_frequency.columns:
    print(col)
# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------

# since all columns are inter-related we eliminate certain overlaps
df_frequency = df_frequency.dropna()
# recommended to get rid of 50% data ---> make data less prone to overfitting
df_frequency
df_frequency = df_frequency.iloc[::2]  # step size 2


# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------
df_cluster = df_frequency.copy()
cluster_columns = ["acc_x", "acc_y", "acc_z"]
k_values = range(2, 10)
inertia = []
# k_values = [2,3,4,5,6,7,8,9,10]
for k in k_values:
    subset = df_cluster[cluster_columns]
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
    cluster_labels = kmeans.fit_predict(subset)
    inertia.append(
        kmeans.inertia_
    )  # sum of squared distances of sample to the closest cluster
    # saving it to plot the wss curve and find the elbow point
plt.title("WSS plot to find elbow point")
plt.xlabel("k - values")
plt.ylabel("sum of squares of distance of the sample to the closest centroid ")
plt.plot(k_values, inertia)

# fixing k to 5
subset = df_cluster[cluster_columns]
kmeans = KMeans(n_clusters=5, n_init=20, random_state=0)
df_cluster["cluster"] = kmeans.fit_predict(subset)
kmeans.inertia_
unique_labels = np.unique(cluster_labels)

fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(projection="3d")
for c in unique_labels:
    subset = df_cluster[df_cluster["cluster"] == c]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=c)
plt.legend()
plt.show()

# unique labels
fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(projection="3d")
for c in df_cluster["label"].unique():
    subset = df_cluster[df_cluster["label"] == c]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=c)
plt.legend()
plt.show()


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

df_cluster.to_pickle("../../data/interim/03_data_features.pkl")
df_cluster.to_csv("../../data/interim/03_data_features.csv")
