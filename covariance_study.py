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

# define the lsit to be droped
list_drop = ["class", "x", "y", "name", "rgb"]  # ,"wa","tri","velo","lateral"

# correct classes
# df_train = correct_labels_incolmation(df=df_train, n_classes=3)
df_train = correct_labels_outcolmation(df=df_train, modus_classes=1)
df_train = df_train[df_train["ddem"] > 0]

# encode the labels
labels = np.sort(df_train["class"].unique().tolist()).tolist()
for i, x in enumerate(labels): labels[i] = str(x)  # convert labels to string
label_class = LabelEncoder()  # create labels for confusion matrix
# df_train["class"] = label_class.fit_transform(df_train["class"])

# loop to print histogram of all banks for a chosen class
label = 1

predictor = "wa"
banks_list = ["kb05","kb07","kb08","kb13","kb19"]  #
df = df_train

for bank in banks_list:
    df_plot = df[df['class'] == label]
    df_plot = df_plot[df_plot['name'].str.match(bank)]
    array = np.array(df_plot[predictor])
    plot_distribution(array, label, bank, predictor)
