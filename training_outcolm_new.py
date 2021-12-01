from config import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix, accuracy_score, make_scorer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import plot_tree
from collections import OrderedDict
import joblib

np.set_printoptions(threshold=sys.maxsize)  # print the whole array

# assembly all training csv files in one pandas data frame
files = "C:\\Users\\ricar\\MA\\training_csv\\out_colm"
files_list = glob.glob(files + "/*.csv")
df_train = pd.DataFrame()
for i, file in enumerate(files_list):
    df_temp = pd.read_csv(file)
    df_train = df_train.append(df_temp)

# transform name columns into string
df_train["name"] = df_train["name"].astype("string")

#define the lsit to be droped
list_drop = ["class", "x", "y", "name", "rgb"] #,"wa","tri","velo","lateral"

# Reclassify dataset with 3, 5 of 8 classes
df_train = correct_labels_outcolmation(df=df_train, modus_classes=3)
df_train = df_train[df_train["ddem"] > 0]  # temporal fix
df_train = df_train[df_train["velo"] > 0]
df_train = modify_predictors_labels(df=df_train)


# encode the labels
labels = np.sort(df_train["class"].unique().tolist()).tolist()
for i, x in enumerate(labels): labels[i] = str(x)  # convert labels to string
label_class = LabelEncoder()  # create labels for confusion matrix
df_train["class"] = label_class.fit_transform(df_train["class"])

# loop to perform sensitivity analysis
n_runs = 1
number_classes = len(labels)  # number of classes
precision_matrix = np.zeros((n_runs, number_classes))
bank_to_predict = None  # Put None to use all banks or the name of the bank to be predicted

for i in range(0, n_runs):

    # look if is training set uses all the data or if a specific bank should be predicted
    if bank_to_predict is None:
        df_train_set, X_test, y_test = split_test_samples_force(df=df_train,
                                                            split=0.2,
                                                            list_drop=list_drop)
    else:
        # remove test samples from training set
        df_train_set, X_test, y_test = split_test_samples_banktopredict(df=df_train,
                                                            riverbank=bank_to_predict,
                                                            list_drop=list_drop)
    # organize training dataset
    X = df_train_set.drop(list_drop,
                      axis=1)
    y = df_train_set["class"]
    X_train, Xdum_test, y_train, ydum_test = train_test_split(X, y,
                                                              test_size=0.01,  # size of testing set
                                                              shuffle=True,  # shuffle before splitting
                                                              )
    # applying standard scaling to normalize data
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    # training RF classifier
    rfc = RandomForestClassifier(n_estimators=100,  # number of trees
                                 criterion='gini',  # Gini impurity as quality measure
                                 bootstrap=True,  # random selection sample of dataset per tree
                                 # min_samples_split=50,  # minimum number of samples for split
                                 max_depth=6,  # maximum depth of each tree
                                 # warm_start=True,  # use previous solution to train next tree and enable keep track of obb
                                 # oob_score=True,
                                 max_features=3
                                 )

    # ________train model for one run___________
    rfc.fit(X_train, y_train)
    rfc_predict = rfc.predict(X_test)
    if n_runs != 1:
        results = classification_report(y_test, rfc_predict, output_dict=True)
        precision_list = []
        for cls in results:
            if cls.isnumeric():
                for att in results[cls]:
                    precision_list.append(float(results[cls][att])) if (att == "precision") else None
        precision_matrix[i, :] = np.array(precision_list)
    print()
if n_runs != 1:
    pd.DataFrame(precision_matrix).to_csv("training_csv/sensitivity/incolm_kb05_ddem_tri_wa_3classes_10depth.csv", index=False)

# # __________Search for the best model using OOB error__________
# parameters = {"max_depth": [4, 5,6,7,8,9,10],
#               "min_samples_leaf": [10, 20, 30, 40, 50, 60],
#               "min_samples_split": [2, 3, 4, 5],
#               "criterion": ["gini", "entropy"]}
# score = make_scorer(accuracy_score)
# rfcs_grid = GridSearchCV(rfc, parameters, scoring=score)
# rfcs_fit = rfcs_grid.fit(X_train,y_train)
# rfc_best = rfcs_fit.best_estimator_
# rfc_best_predict = rfc_best.predict(X_test)

# # ______________plot the importances of the variables_____________
# feature_names = list(X.columns)
# importances = rfc.feature_importances_
# std = np.std([
#     tree.feature_importances_ for tree in rfc.estimators_], axis=0)
#
# forest_importances = pd.Series(importances, index=feature_names)
#
# fig, ax = plt.subplots()
# forest_importances.plot.bar(yerr=std, ax=ax)
# # ax.set_title("Feature importances using MDI")
# ax.set_ylabel("Mean decrease in impurity")
# ax.set_xlabel("Predictors")
# fig.tight_layout()
# plt.show()

# __________________plot improvement of oob or test set increasing the number fo threes_______________________

# # Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
# error_rate = []
# error_rate_train = []
#
# ## Range of `n_estimators` values to explore.
# min_estimators = 2
# max_estimators = 150
#
# # loop and fit a new model for different number of tree estimators
# for i in range(min_estimators, max_estimators + 1):
#     rfc.set_params(n_estimators=i)
#     rfc.fit(X_train, y_train)
#
#     # Record the OOB error for each `n_estimators=i` setting.
#     # oob_error = 1 - rfc.oob_score_
#     # error_rate.append((i, oob_error))
#     test_error = 1 - accuracy_score(y_test, rfc.predict(X_test))
#     train_error = 1 - rfc.score(X_train, y_train)
#     error_rate.append((i, test_error))
#     error_rate_train.append((i, train_error))
#
# # devide x and y axes
# xs, ys = zip(*error_rate)
# xsdum, ys_train = zip(*error_rate_train)
# fig, ax1 = plt.subplots()
# ax1.plot(xs, ys, label="test error", marker="o")
# ax1.plot(xs, ys_train, label="train error", marker="o")
# # plt.xlim(min_depth, max_depth)
# plt.xlabel("number of trees")
# plt.ylabel("error rate")
# plt.legend(loc="upper right")
# plt.show()


# # ___________plot extend of pruning effect__________
# # range of pruning
# min_depth = 2
# max_depth = 20
#
# # Empty list to save oob error with pruning.
# error_rate = []
# error_rate_train = []
#
# # loop and fit a new model for different depth values
# for i in reversed(range(min_depth, max_depth + 1)):
#     rfc.set_params(max_depth=i)
#     rfc.fit(X_train, y_train)
#
#     # Record the OOB error for each `max_depth=i` setting.
#     # oob_error = 1 - rfc.oob_score_
#     test_error = 1 - accuracy_score(y_test, rfc.predict(X_test))
#     train_error = 1 - rfc.score(X_train, y_train)
#     error_rate.append((i, test_error))
#     error_rate_train.append((i, train_error))
# # devide x and y axes
# xs, ys = zip(*error_rate)
# xsdum, ys_train = zip(*error_rate_train)
# fig, ax1 = plt.subplots()
# ax1.plot(xs, ys, label="test error", marker="o")
# ax1.plot(xs, ys_train, label="train error", marker="o")
# ax1.invert_xaxis()
# # plt.xlim(min_depth, max_depth)
# plt.xlabel("depth of trees")
# plt.ylabel("error rate")
# plt.legend(loc="upper right")
# plt.show()

## _______Visualize one tree of the forest____________

# rfc_tree = rfc.estimators_[25]  # extract the 100th tree
# plt.figure(figsize=(18, 12))
# plot_tree(rfc_tree, filled=True, rounded=True,
#           class_names=labels, feature_names=X.columns)
# plt.show()



print(classification_report(y_test, rfc_predict))
print()
print(confusion_matrix(y_test, rfc_predict))
# print()
print(rfc.score(X_train, y_train))
# # print(rfc.oob_score_)
print(accuracy_score(y_test, rfc_predict))
# print()
# print(oob_best_iter)

## _____Plot confusion matrix_______
plot_confusion_matrix(rfc, X_test, y_test, display_labels=labels, normalize="true")
plt.show()



# __________ save model_________
# joblib.dump(rfc, "trained_rf/colm_out/wa_velo_la_ddem_out" + bankout + ".joblib")

# __________ save best model_________
# joblib.dump(rfc_best, "trained_rf/colm_in/best_model/wa_velo_la_ddem_in.joblib")

# # load the model from disk
# loaded_model = pickle.load(open(filename, 'rb'))
# result = loaded_model.score(X_test, Y_test)


# #____plot best model for the data____
# print(rfc_best.score(X_train,y_train))
# print(rfc_best.oob_score_)
# print(accuracy_score(y_test, rfc_best_predict))
# print()
# print(rfc_best)


# _____ Old visualization for training data_____________

# print(X_train[0:10])
# print(df_test.isnull().sum())
# print(df_test["class"].unique(), "\n")
# print(df_test.head(10))
# print(df_test["class"].value_counts())

# __________NOTES________
# best predictor for XTRain["dem","tri"]
# ' RandomForestClassifier(criterion='entropy', max_depth=5, min_samples_leaf=10,
# n_estimators=200, oob_score=True)'
