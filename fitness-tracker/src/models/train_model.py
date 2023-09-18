import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from LearningAlgorithms import ClassificationAlgorithms
import seaborn as sns
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix

learner = ClassificationAlgorithms()

# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

df = pd.read_pickle("../../data/interim/03_data_features.pkl")

# --------------------------------------------------------------
# Create a training and test set
# --------------------------------------------------------------

df_train = df.drop(["participant", "set", "category"], axis=1)

X = df_train.drop("label", axis=1)

y = df_train["label"]

# stratify to have equal inputs of all labels in y
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

fig, ax = plt.subplots(figsize=(10, 5))

df_train["label"].value_counts().plot(
    kind="bar", ax=ax, color="lightblue", label="Total"
)

y_train.value_counts().plot(kind="bar", ax=ax, color="dodgerblue", label="Train")

y_test.value_counts().plot(kind="bar", ax=ax, color="royalblue", label="Test")
plt.legend()
plt.show()

print(X_train.columns)
X_train["gyro_r_freq_0.0_Hz_ws_10"][:25]
# --------------------------------------------------------------
# Split feature subsets
# --------------------------------------------------------------

basic_features = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
square_features = ["acc_r", "gyro_r"]
pca_features = ["pca_1", "pca_2", "pca_3"]

time_features = [f for f in df_train.columns if "_temp_" in f]
frequecy_features = [f for f in df_train.columns if ("_pse" in f) or ("_freq") in f]
cluster_features = ["cluster"]
len(frequecy_features)

feature_set1 = list(set(basic_features))
feature_set2 = list(set(basic_features + square_features + pca_features))
feature_set3 = list(set(feature_set2 + time_features))
feature_set4 = list(set(feature_set3 + frequecy_features + cluster_features))

# --------------------------------------------------------------
# Perform forward feature selection using simple decision tree
# --------------------------------------------------------------

# in the start the accuracy may increase as we add more features ....> but after somepoint we might see deminshing returns
learner = ClassificationAlgorithms()
max_features = 10
selected_features, ordered_features, ordered_scores = learner.forward_selection(
    max_features, X_train, y_train
)

selected_features = [
    "pca_1",
    "duration",
    "acc_x_freq_0.0_Hz_ws_10",
    "acc_z_temp_mean_ws_5",
    "acc_y_temp_mean_ws_5",
    "gyro_y_temp_mean_ws_5",
    "acc_z_pse",
    "gyro_z_freq_0.5_Hz_ws_10",
    "gyro_r_freq_2.5_Hz_ws_10",
    "gyro_y_freq_1.5_Hz_ws_10",
]
ordered_scores = [
    0.8926572275271649,
    0.9776094830424761,
    0.997365821534409,
    0.9993414553836022,
    0.9996707276918011,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
]
X_train.columns
"""new selected features
['pca_1',
 'duration',
 'acc_x_freq_0.0_Hz_ws_10',
 'acc_z_temp_mean_ws_5',
 'acc_y_temp_mean_ws_5',
 'gyro_y_temp_mean_ws_5',
 'acc_z_pse',
 'gyro_z_freq_0.5_Hz_ws_10',
 'gyro_r_freq_2.5_Hz_ws_10',
 'gyro_y_freq_1.5_Hz_ws_10']
"""

"""new ordered_scores
[0.8926572275271649,
 0.9776094830424761,
 0.997365821534409,
 0.9993414553836022,
 0.9996707276918011,
 1.0,
 1.0,
 1.0,
 1.0,
 1.0]
"""

"""selected features --->   ['acc_x',
 'duration',
 'pca_1',
 'acc_z',
 'cluster',
 'acc_r_max_freq',
 'gyro_r_max_freq',
 'acc_y',
 'gyro_y',
 'acc_r']"""
"""ordered scores --->  [0.8507992895204263,
 0.9659561870929544,
 0.9937833037300178,
 0.9973357015985791,
 0.9973357015985791,
 0.9973357015985791,
 0.9973357015985791,
 0.9973357015985791,
 0.9973357015985791,
 0.9970396684428656] """

plt.figure(figsize=(20, 15))
plt.plot(np.arange(1, max_features + 1), ordered_scores, label=selected_features)
plt.xlabel("number of features")
plt.ylabel("corresponding accuracies")
plt.show()

plt.savefig("../../src/visualization/accuracy_plot_today.png")
# --------------------------------------------------------------
# Grid search for best hyperparameters and model selection
# --------------------------------------------------------------

possible_feature_sets = [
    feature_set1,
    feature_set2,
    feature_set3,
    feature_set4,
    selected_features,
]
feature_names = [
    "feature_set1",
    "feature_set2",
    "feature_set3",
    "feature_set4",
    "selected_features",
]

iterations = 1
score_df = pd.DataFrame()
type(df.values)


for i, f in zip(range(len(possible_feature_sets)), feature_names):
    print("Feature set:", i)
    selected_train_X = X_train[possible_feature_sets[i]]
    selected_test_X = X_test[possible_feature_sets[i]]

    # First run non deterministic classifiers to average their score.
    performance_test_nn = 0
    performance_test_rf = 0

    for it in range(0, iterations):
        print("\tTraining neural network,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.feedforward_neural_network(
            selected_train_X,
            y_train,
            selected_test_X,
            gridsearch=False,
        )
        performance_test_nn += accuracy_score(y_test, class_test_y)

        print("\tTraining random forest,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.random_forest(
            selected_train_X, y_train, selected_test_X, gridsearch=True
        )
        performance_test_rf += accuracy_score(y_test, class_test_y)

    performance_test_nn = performance_test_nn / iterations
    performance_test_rf = performance_test_rf / iterations

    # And we run our deterministic classifiers:
    print("\tTraining KNN")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.k_nearest_neighbor(
        selected_train_X, y_train, selected_test_X, gridsearch=True
    )
    performance_test_knn = accuracy_score(y_test, class_test_y)

    print("\tTraining decision tree")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.decision_tree(
        selected_train_X, y_train, selected_test_X, gridsearch=True
    )
    performance_test_dt = accuracy_score(y_test, class_test_y)

    print("\tTraining naive bayes")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.naive_bayes(selected_train_X, y_train, selected_test_X)

    performance_test_nb = accuracy_score(y_test, class_test_y)

    # Save results to dataframe
    models = ["NN", "RF", "KNN", "DT", "NB"]
    new_scores = pd.DataFrame(
        {
            "model": models,
            "feature_set": f,
            "accuracy": [
                performance_test_nn,
                performance_test_rf,
                performance_test_knn,
                performance_test_dt,
                performance_test_nb,
            ],
        }
    )
    score_df = pd.concat([score_df, new_scores])

# score_df.to_csv("../../data/interim/score_data.csv")
# score_df.to_pickle("../../data/interim/score_data.pkl")
score_df = pd.read_pickle("../../data/interim/score_data.pkl")
score_df.sort_values("accuracy", ascending=False)
# saved ends here
"""	model	feature_set	accuracy
1	RF	feature_set2	0.964462
1	RF	feature_set1	0.960513
0	NN	feature_set2	0.941757
0	NN	feature_set1	0.939783
3	DT	feature_set2	0.938796
3	DT	feature_set1	0.937808
4	NB	feature_set1	0.875617
4	NB	feature_set2	0.873643
2	KNN	feature_set1	0.768016
2	KNN	feature_set2	0.766041"""


# --------------------------------------------------------------
# Create a grouped bar plot to compare the results
# --------------------------------------------------------------
score_df.sort_values("accuracy", ascending=False)
plt.figure(figsize=(15, 8))
sns.barplot(x="model", y="accuracy", hue="feature_set", data=score_df)
plt.xlabel("Models")
plt.ylabel("Their accuracies")
plt.ylim(0.7, 1)
plt.legend(loc="lower right")
plt.show()

plt.savefig("../../src/visualization/accuracies_sorted.png")
# --------------------------------------------------------------
# Select best model and evaluate results
# --------------------------------------------------------------
# lets pick random forest

(
    class_train_y,
    class_test_y,
    class_train_prob_y,
    class_test_prob_y,
) = learner.random_forest(
    X_train[feature_set4], y_train, X_test[feature_set4], gridsearch=False
)
accuracy = accuracy_score(y_test, class_test_y)
# accuracy = 0.9891411648568608

# classes = class_test_prob_y.columns
classes = ["bench", "dead", "ohp", "rest", "row", "squat"]

cm = confusion_matrix(y_test, class_test_y)

# create confusion matrix for cm
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show()
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import ConfusionMatrixDisplay
# disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=classes)
# disp.plot()
# plt.show()

# --------------------------------------------------------------
# Select train and test data based on participant
# --------------------------------------------------------------
participant_df = df.drop(["set", "category"], axis=1)

X_train = participant_df[participant_df["participant"] != "A"].drop("label", axis=1)
X_test = participant_df[participant_df["participant"] == "A"].drop("label", axis=1)
y_train = participant_df[participant_df["participant"] != "A"]["label"]
y_test = participant_df[participant_df["participant"] == "A"]["label"]

X_train = X_train.drop("participant", axis=1)
X_test = X_test.drop("participant", axis=1)


fig, ax = plt.subplots(figsize=(10, 5))

df_train["label"].value_counts().plot(
    kind="bar", ax=ax, color="lightblue", label="Total"
)

y_train.value_counts().plot(kind="bar", ax=ax, color="dodgerblue", label="Train")

y_test.value_counts().plot(kind="bar", ax=ax, color="royalblue", label="Test")
plt.legend()
plt.show()


# --------------------------------------------------------------
# Use best model again and evaluate results
# --------------------------------------------------------------

(
    class_train_y,
    class_test_y,
    class_train_prob_y,
    class_test_prob_y,
) = learner.random_forest(
    X_train[feature_set4], y_train, X_test[feature_set4], gridsearch=False
)
accuracy = accuracy_score(y_test, class_test_y)

# classes = class_test_prob_y.columns
classes = ["bench", "dead", "ohp", "rest", "row", "squat"]

cm = confusion_matrix(y_test, class_test_y)

# create confusion matrix for cm
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show()
len(feature_set4)
