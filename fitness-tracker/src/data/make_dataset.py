import pandas as pd
from glob import glob

# --------------------------------------------------------------
# Read single CSV file
# --------------------------------------------------------------

single_file_acc = pd.read_csv(
    "../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv"
)

single_file_gyro = pd.read_csv(
    "../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv"
)
# --------------------------------------------------------------
# List all data in data/raw/MetaMotion
# --------------------------------------------------------------
files = glob("../../data/raw/MetaMotion/*.csv")
# len(files)

# --------------------------------------------------------------
# Extract features from filename
# --------------------------------------------------------------
data_path = "../../data/raw/MetaMotion/"
f = files[0]
type(f)
participant = f.split("-")[0].replace(data_path, "")
label = f.split("-")[1]
category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")
df = pd.read_csv(f)

df["participant"] = participant
df["label"] = label
df["category"] = category
# --------------------------------------------------------------
# Read all files
# --------------------------------------------------------------
acc_df = pd.DataFrame()
gyro_df = pd.DataFrame()

acc_set = 1
gyro_set = 1

for f in files:
    participant = f.split("-")[0].replace(data_path, "")
    label = f.split("-")[1]
    category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

    df = pd.read_csv(f)
    df["participant"] = participant
    df["label"] = label
    df["category"] = category

    if "Accelerometer" in f:
        df["set"] = acc_set
        acc_df = pd.concat([acc_df, df], ignore_index=True)
        acc_set += 1

    if "Gyroscope" in f:
        df["set"] = gyro_set
        gyro_df = pd.concat([gyro_df, df], ignore_index=True)
        gyro_set += 1

# gyro collected data with higher frequency

# --------------------------------------------------------------
# Working with datetimes
# --------------------------------------------------------------

acc_df.info()

# making epoch and time in date time data type
df.columns

pd.to_datetime(df["epoch (ms)"], unit="ms")

pd.to_datetime(df["time (01:00)"])

"""we do this because we can extract the month, 
day of the week and many more using date time object"""

acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")

gyro_df.index = pd.to_datetime(gyro_df["epoch (ms)"], unit="ms")

# now lets remove the other time-series columns
del acc_df["epoch (ms)"]
del acc_df["time (01:00)"]
del acc_df["elapsed (s)"]

del gyro_df["epoch (ms)"]
del gyro_df["time (01:00)"]
del gyro_df["elapsed (s)"]


# --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------


def read_all_data(files, data_path):
    # --------------------------------------------------------------
    # Extract features from filename
    # --------------------------------------------------------------
    f = files[0]

    participant = f.split("-")[0].replace(data_path, "")[-1]
    label = f.split("-")[1]
    category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")
    df = pd.read_csv(f)

    df["participant"] = participant
    df["label"] = label
    df["category"] = category
    # --------------------------------------------------------------
    # Read all files
    # --------------------------------------------------------------
    acc_df = pd.DataFrame()
    gyro_df = pd.DataFrame()

    acc_set = 1
    gyro_set = 1

    for f in files:
        participant = f.split("-")[0].replace(data_path, "")[-1]
        label = f.split("-")[1]
        category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

        df = pd.read_csv(f)
        df["participant"] = participant
        df["label"] = label
        df["category"] = category

        if "Accelerometer" in f:
            df["set"] = acc_set
            acc_df = pd.concat([acc_df, df], ignore_index=True)
            acc_set += 1

        if "Gyroscope" in f:
            df["set"] = gyro_set
            gyro_df = pd.concat([gyro_df, df], ignore_index=True)
            gyro_set += 1

    # gyro collected data with higher frequency

    # --------------------------------------------------------------
    # Working with datetimes
    # --------------------------------------------------------------

    acc_df.info()

    # making epoch and time in date time data type
    df.columns

    pd.to_datetime(df["epoch (ms)"], unit="ms")

    pd.to_datetime(df["time (01:00)"])

    """we do this because we can extract the month, 
    day of the week and many more using date time object"""

    acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")

    gyro_df.index = pd.to_datetime(gyro_df["epoch (ms)"], unit="ms")

    # now lets remove the other time-series columns
    del acc_df["epoch (ms)"]
    del acc_df["time (01:00)"]
    del acc_df["elapsed (s)"]

    del gyro_df["epoch (ms)"]
    del gyro_df["time (01:00)"]
    del gyro_df["elapsed (s)"]
    return acc_df, gyro_df


files = glob("../../data/raw/MetaMotion/*.csv")
data_path = "../../data/raw/MetaMotion/"

acc_df, gyro_df = read_all_data(files, data_path)


# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------
data_merged = pd.concat([acc_df.iloc[:, :3], gyro_df], axis=1)
data_merged.columns = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "participant",
    "label",
    "category",
    "set"
]
# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz

sampling = {
    'acc_x' :"mean", 
    'acc_y' :"mean", 
    'acc_z' :"mean", 
    'gyro_x': "mean", 
    'gyro_y': "mean", 
    'gyro_z': "mean", 
    'participant':"last",
    'label' :"last",
    'category':"last",
    'set':"last"
}

data_merged.iloc[:1000].resample(rule='200ms').apply(sampling)

days = [g for n, g in data_merged.groupby(pd.Grouper(freq = "D"))]

data_resampled = pd.concat([df.resample(rule = "200ms").apply(sampling).dropna() for df in days])

data_resampled["set"] = data_resampled["set"].astype("int64")

data_resampled.info()

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

data_resampled.to_pickle("../../data/interim/01_data_processed.pkl")
data_resampled.to_csv("../../data/interim/01_data_processed.csv")

