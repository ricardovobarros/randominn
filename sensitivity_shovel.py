from config import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# assembly all training csv files in one pandas data frame
files = "C:\\Users\\ricar\\MA\\training_csv\\shovel_4classes"
files_list = glob.glob(files + "/*.csv")
df_train = pd.DataFrame()
for i, file in enumerate(files_list):
    df_temp = pd.read_csv(file)
    df_train = df_train.append(df_temp)

# encode the labels
label_class = LabelEncoder()
df_train["class"] = label_class.fit_transform(df_train["class"])

# separate the dataset
X = df_train.drop(["class", "x", "y", "name", "rgb"], axis=1) # ,"vari"
y = df_train["class"]

# train the model many times
n_runs = 100
number_classes = len(np.unique(np.array(shovel_classes)))  # number of classes
precision_matrix = np.zeros((n_runs, number_classes))

for i in range(0, n_runs):
    # train and test splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.33,  # size of testing set
                                                        shuffle=True,  # shuffle before splitting
                                                        )
    # applying standard scaling to normalize data
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    # training RF classifier
    rfc = RandomForestClassifier(n_estimators=200  # number of trees
                                 )
    rfc.fit(X_train, y_train)
    rfc_predict = rfc.predict(X_test)

    results = classification_report(y_test, rfc_predict, output_dict=True)
    precision_list = []
    for cls in results:
        if cls.isnumeric():
            for att in results[cls]:
                precision_list.append(float(results[cls][att])) if (att == "precision") else None
    precision_matrix[i, :] = np.array(precision_list)
pd.DataFrame(precision_matrix).to_csv("training_csv/sensitivity/5classes_100_wvari.csv", index=False)







# # plot the importances of the variables
# feature_names = list(X.columns)
# importances = rfc.feature_importances_
# std = np.std([
#     tree.feature_importances_ for tree in rfc.estimators_], axis=0)

# forest_importances = pd.Series(importances, index=feature_names)
#
# fig, ax = plt.subplots()
# forest_importances.plot.bar(yerr=std, ax=ax)
# ax.set_title("Feature importances using MDI")
# ax.set_ylabel("Mean decrease in impurity")
# fig.tight_layout()
# plt.show()

# print(classification_report(y_test, rfc_predict))
# print()
# print(confusion_matrix(y_test, rfc_predict))

# get values of classification report



