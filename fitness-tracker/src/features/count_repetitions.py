import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter
from scipy.signal import argrelextrema
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score

pd.options.mode.chained_assignment = None


# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/01_data_processed.pkl")
df = df[df["label"] != "rest"]

squat_df = df[df["set"] == df["set"].unique()[0]]
# acc_r and gyro_r
acc_r = df["acc_x"] ** 2 + df["acc_y"] ** 2 + df["acc_z"] ** 2
gyro_r = df["gyro_x"] ** 2 + df["gyro_y"] ** 2 + df["gyro_z"] ** 2
acc_r = np.sqrt(acc_r)
gyro_r = np.sqrt(gyro_r)
df["acc_r"] = acc_r
df["gyro_r"] = gyro_r


# --------------------------------------------------------------
# Split data
# --------------------------------------------------------------
df.label.unique()
df_bench = df[df["label"] == "bench"]
df_ohp = df[df["label"] == "ohp"]
df_squat = df[df["label"] == "squat"]
df_dead = df[df["label"] == "dead"]
df_row = df[df["label"] == "row"]


# --------------------------------------------------------------
# Visualize data to identify patterns
# --------------------------------------------------------------


plot_df = squat_df

plot_df["acc_x"].plot()
plot_df["acc_y"].plot()
plot_df["acc_z"].plot()
plot_df["acc_r"].plot()

plot_df["gyro_x"].plot()

plot_df["gyro_y"].plot()

plot_df["gyro_z"].plot()
plot_df["gyro_r"].plot()


# --------------------------------------------------------------
# Configure LowPassFilter
# --------------------------------------------------------------

fs = 1000 / 200
Low_pass = LowPassFilter()

# --------------------------------------------------------------
# Apply and tweak LowPassFilter
# --------------------------------------------------------------
unique_set = df_row["set"].unique()[1]

bench_set = df_bench[df_bench["set"] == df_bench["set"].unique()[0]]
squat_set = df_squat[df_squat.set == unique_set]
df.label.unique()
ohp_set = df_ohp[df_ohp.set == unique_set]
dead_set = df_dead[df_dead.set == unique_set]
row_set = df_row[df_row.set == unique_set]

squat_set["acc_r"].plot()
col = "acc_r"
# setting cutoff frequency by trial and error   (cuts out the noise)
Low_pass.low_pass_filter(row_set, col, fs, 0.4, 5)[col + "_lowpass"].plot()

# -----------------------------------------     ---------------------
# Create function to count repetitions
# --------------------------------------------------------------


def count_reps(dataset, cutoff=0.4, order=10, column="acc_r"):
    col = column
    data = Low_pass.low_pass_filter(
        dataset, col=column, cutoff_frequency=cutoff, sampling_frequency=fs, order=order
    )

    indexes = argrelextrema(data[col + "_lowpass"].values, np.greater)
    peaks = data.iloc[indexes]

    # plotting the columns and its peaks
    fig, ax = plt.subplots()
    plt.plot(dataset[column + "_lowpass"])
    plt.plot(peaks[column + "_lowpass"], "v", color="red")
    ax.set_ylabel(column + "_lowpass")
    exercise = dataset["label"].iloc[0].title()
    category = dataset["category"].iloc[0].title()
    plt.title(f"{category} {exercise} : {len(peaks)} totalreps")
    plt.show()

    return len(peaks)


# cutoff = 0.4 best for bench
count_reps(squat_set)

# note: medium has 10 reps and heavy has 5 reps
count_reps(squat_set, cutoff=0.565)

# the best cutoff frequency for every label
# bench = 0.4
# squat = 0.35
# dead = 0.4
# row = 0.65 column = "gyro_x"
# ohp = 0.35


# --------------------------------------------------------------
# Create benchmark dataframe
# --------------------------------------------------------------

df["reps"] = df["category"].apply(lambda x: 5 if x == "heavy" else 10)

rep_df = df.groupby(["label", "category", "set"])["reps"].max().reset_index()

rep_df["reps_pred"] = 0

for s in df["set"].unique():
    subset = df[df["set"] == s]
    column = "acc_r"
    cutoff = 0.4
    if subset["label"].iloc[0] == "row":
        column = "gyro_x"
        cutoff = 0.65
    if subset["label"].iloc[0] == "squat":
        cutoff = 0.35
    if subset["label"].iloc[0] == "ohp":
        cutoff = 0.35

    reps = count_reps(subset, cutoff=cutoff, column=column)
    rep_df.loc[rep_df["set"] == s, "reps_pred"] = reps

# --------------------------------------------------------------
# Evaluate the results
# --------------------------------------------------------------

error = mean_absolute_error(rep_df["reps"], rep_df["reps_pred"]).round(2)

rep_df.groupby(["label", "category"])["reps", "reps_pred"].mean().plot.bar(
    figsize=(20, 10)
)
diff_df = rep_df.groupby(["label", "category"])["reps", "reps_pred"].mean()
diff = diff_df["reps"] - diff_df["reps_pred"]
diff

# printing if the absolute difference is more than 1
diff_df[abs(diff) > 1]

our_accuracy = accuracy_score(rep_df["reps"], rep_df["reps_pred"])
rep_df.to_pickle("../../data/external/counting_reps.pkl")
