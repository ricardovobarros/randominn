from config import *
import scipy.cluster.hierarchy as sch
import pandas as pd
import matplotlib.pyplot as plt


# label fuction

def llf(xx):
    return "{}  {}".format(*temp[xx])


## This version gives you your label AND the count
# temp = {R["leaves"][ii]:(labels[ii], R["ivl"][ii]) for ii in range(len(R["leaves"]))}
# def llf(xx):
#     return "{} - {}".format(*temp[xx])

# Open data as pandas
columns = np.linspace(0, 18, 19, dtype=int)
df = pd.read_csv("data/grainsizes/shovel/overview_shovel_probes.csv",
                 usecols=columns)
df.drop(columns=["layer_type", "Kampagne"])  # remove unnecessary columns
df = df[df["layer_type"] == "OS"]  # filter only OS

# Create new selection columns
df[">125mm"] = df["250"] - df["125"]
df["125-63mm"] = df["125"] - df["63"]
df["63-8mm"] = df["63"] - df["8"]
df["<8mm"] = df["8"] - df["0.063"]

# Verify if 100% is reached
df["total_veri"] = df[[">125mm","125-63mm", "63-8mm", "<8mm"]].sum(axis=1)

# create column with the names of all measurements
df['name'] = df['kb'].map(str) + "_" + df['meas_number'].map(str)

# Create data point to be clustered
# df.filter(items=["pebble_coarse","pebble_finer","gravel","sand"])
df_dendrogram = df[["name", ">125mm","125-63mm", "63-8mm", "<8mm"]]
shovel_array = df_dendrogram.iloc[:, 1:5].to_numpy()
df_dendrogram.to_csv("Testfile/shovel_samples_clustering_corrected.csv", index=False)

# figure seetings
plt.figure(figsize=(10, 10))
# plt.title('Shovel Samples Hierarchical Clustering Dendrogram', fontsize=20)
plt.xlabel('Shovel Samples', fontsize=16)
plt.ylabel('Distance %', fontsize=16)
labels = list(df_dendrogram["name"])

# rename the samples from KB to GB
new_strings = []
for string in labels:
    new_string = string.replace("K", "G")
    new_strings.append(new_string)
labels = new_strings

# dendrogram
p = len(labels)
linked = sch.linkage(shovel_array, method="ward", metric="euclidean")
R = sch.dendrogram(
    linked,
    # truncate_mode='lastp',  # show only the last p merged clusters
    # p=p,  # show only the last p merged clusters
    no_plot=True,
)

# create a label dictionary
temp = {R["leaves"][ii]:(labels[int(R["ivl"][ii])], "") for ii in range(len(R["leaves"]))}
sch.set_link_color_palette(['m', 'c', 'y', 'b', 'k'])
demogram = sch.dendrogram(linked,
                          # truncate_mode='lastp',  # show only the last p merged clusters
                          color_threshold=0.4,
                          # p=p,  # show only the last p merged clusters
                          leaf_label_func=llf,
                          leaf_rotation=90.,
                          # show_contracted=True,  # to get a distribution impression in truncated branches
                          )
plt.tight_layout()
plt.show()
print(df)
