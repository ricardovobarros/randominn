from config import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix, accuracy_score, make_scorer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import plot_tree
from collections import OrderedDict

# assembly all training csv files in one pandas data frame
files = "C:\\Users\\ricar\\MA\\training_csv\\shovel_5classes"
files_list = glob.glob(files + "/*.csv")
df_train = pd.DataFrame()
for i, file in enumerate(files_list):
    df_temp = pd.read_csv(file)
    df_train = df_train.append(df_temp)

# encode the labels
label_class = LabelEncoder()
df_train["class"] = label_class.fit_transform(df_train["class"])

# organize dataset
X = df_train.drop(["class", "x", "y", "name", "rgb"],
                  axis=1)  # , "wa","vari","green","lateral", "longitudinal","tri"
y = df_train["class"]

# train and test splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.33,  # size of testing set
                                                    shuffle=True,  # shuffle before splitting
                                                    )
# # applying standard scaling to normalize data
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.fit_transform(X_test)

# training RF classifier
rfc = RandomForestClassifier(n_estimators=20,  # number of trees
                             criterion='gini',  # Gini impurity as quality measure
                             bootstrap=True,  # random selection sample of dataset per tree
                             min_samples_split=10,  # minimum number of samples for split
                             max_depth=5,  # maximum depth of each tree
                             # warm_start=True, # use previous solution to train next tree and enable keep track of obb
                             # oob_score=True
                             )
rfc.fit(X_train, y_train)
rfc_predict = rfc.predict(X_test)

# # __________Search for the best model using OOB error__________
# parameters = {"max_depth": [1, 2, 3, 4, 5],
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
# ax.set_title("Feature importances using MDI")
# ax.set_ylabel("Mean decrease in impurity")
# fig.tight_layout()
# plt.show()

# # __________________plot improvement of oob_______________________
#
# # Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
# error_rate = []
#
# ## Range of `n_estimators` values to explore.
# min_estimators = 1
# max_estimators = 8
#
# # loop and fit a new model for different number of estimares
# for i in range(min_estimators, max_estimators + 1):
#     rfc.set_params(n_estimators=i)
#     rfc.fit(X_train, y_train)
#
#     # Record the OOB error for each `n_estimators=i` setting.
#     oob_error = 1 - rfc.oob_score_
#     error_rate.append((i, oob_error))
#
# # devide x and y axes
# xs, ys = zip(*error_rate)
# plt.plot(xs, ys, label="OBB error", marker="o")
#
# plt.xlim(min_estimators, max_estimators)
# plt.xlabel("n_estimators")
# plt.ylabel("OOB error rate")
# plt.legend(loc="upper right")
# plt.show()


# ___________plot extend of pruning effect__________
# range of pronning
# min_depth = 4
# max_depth = 15
#
# # Empty list to save oob error with pruning.
# error_rate = []
#
# # loop and fit a new model for different depth values
# for i in reversed(range(min_depth, max_depth + 1)):
#     rfc.set_params(max_depth=i)
#     rfc.fit(X_train, y_train)
#
#     # Record the OOB error for each `max_depth=i` setting.
#     oob_error = 1 - rfc.oob_score_
#     error_rate.append((i, oob_error))
#
# # devide x and y axes
# xs, ys = zip(*error_rate)
# fig, ax1 = plt.subplots()
# ax1.plot(xs, ys, label="OBB error", marker="o")
# ax1.invert_xaxis()
# # plt.xlim(min_depth, max_depth)
# plt.xlabel("depth of trees")
# plt.ylabel("OOB error rate")
# plt.legend(loc="upper right")
# plt.show()


print(classification_report(y_test, rfc.predict(X_test)))
print()
print(confusion_matrix(y_test, rfc.predict(X_test)))
print()
print(rfc.score(X_train, y_train))
# print(rfc.oob_score_)
print(accuracy_score(y_test, rfc.predict(X_test)))
print()
# print(oob_best_iter)

# print(rfc_best.score(X_train,y_train))
# print(rfc_best.oob_score_)
# print(accuracy_score(y_test, rfc_best_predict))
# print()
# print(rfc_best)

# plot_confusion_matrix(rfc, X_test, y_test, display_labels=labels, normalize="true")
# plt.show()

## _____________Visualize one tree of the forest____________

# rfc_tree = rfc.estimators_[100]  # extract the 100th tree
# plt.figure(figsize=(18,12))
# plot_tree(rfc_tree, filled=True, rounded=True,
#           class_names=labels, feature_names=X.columns)
# plt.show()

# _____ Old visualization for training data_____________

# print(X_train[0:10])
# print(df_test.isnull().sum())
# print(df_test["class"].unique(), "\n")
# print(df_test.head(10))
# print(df_test["class"].value_counts())

#__________NOTES________
# best predictor for XTRain["dem","tri"]
# ' RandomForestClassifier(criterion='entropy', max_depth=5, min_samples_leaf=10,
                      # n_estimators=200, oob_score=True)'
