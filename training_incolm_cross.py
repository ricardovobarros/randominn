from config import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix, accuracy_score, make_scorer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.tree import plot_tree
from collections import OrderedDict
import joblib
import random

np.set_printoptions(threshold=sys.maxsize)  # print the whole array

# assembly all training csv files in one pandas data frame
files = "C:\\Users\\ricar\\MA\\training_csv\\in_colm"
files_list = glob.glob(files + "/*.csv")
dataset = pd.DataFrame()
for i, file in enumerate(files_list):
    df_temp = pd.read_csv(file)
    dataset = dataset.append(df_temp)

# transform name columns into string
dataset["name"] = dataset["name"].astype("string")

# define the lsit to be droped and drop unecessary columns
list_drop = ["x", "y", "rgb"]  # ,"wa","velo","ddem","tri","lateral"
dataset.drop(list_drop, axis=1, inplace=True)

# Reclassify dataset with 3, 5 of 8 classes
dataset = correct_labels_incolmation(df=dataset, n_classes=4)

# dataset adjusts
dataset = dataset[dataset["ddem"] > 0]  # temporal fix
dataset = dataset[dataset["velo"] > 0]
dataset = modify_predictors_labels(df=dataset)
ids = dataset["name"].unique()
random.shuffle(ids)
dataset = dataset.set_index("name").loc[ids].reset_index()

# # test using group by and taking the means
# dataset = dataset.groupby("name").mean()


# encode the labels
labels = np.sort(dataset["class"].unique().tolist()).tolist()
for i, x in enumerate(labels): labels[i] = str(x)  # convert labels to string
label_class = LabelEncoder()  # create labels for confusion matrix
dataset["class"] = label_class.fit_transform(dataset["class"])

# Set Hyperparameters
rfc = RandomForestClassifier(n_estimators=90,  # number of trees
                             criterion='gini',  # Gini impurity as quality measure
                             bootstrap=True,  # random selection sample of dataset per tree
                             # min_samples_split=50,  # minimum number of samples for split
                             # max_depth=6,  # maximum depth of each tree
                             # warm_start=True,  # use previous solution to train next tree and enable keep track of obb
                             # oob_score=True,
                             max_features=3
                             )

# split testing set
train_inds, test_inds = next(GroupShuffleSplit(test_size=.25, n_splits=2)
                             .split(dataset, groups=dataset['name']))
df_train = dataset.iloc[train_inds]
df_test = dataset.iloc[test_inds]

# force class 3 of internal clogging to got o testing set if there are 3 on training
df_train, df_test = force_class_3(dftr=df_train, dft=df_test)

# create lists with the name of the folders for cross validation
t_set, v_set = create_folders(n_folder=5, dataset=df_train)

# k-fold cross-validation loop to find optimal depth
cross = True
if cross:
    depth_opt = k_fold_cross_validation(rfc=rfc, df_train=df_train,
                                        t_set=t_set, v_set=v_set, n_folder=5,
                                        validation_graphs=True,
                                        max_depth=10)
else:
    depth_opt = 5

# train RF with parameter optimal depth and whole training set
train_and_test(rfc=rfc, df_train=df_train, df_test=df_test,
               depth=depth_opt,labels=labels,
               confusion_matrix=False, save_model=True,
               plot_importance=True, evalute_n_trees=False,
               n_max_trees=20)
print(depth_opt)


