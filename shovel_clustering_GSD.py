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
# df["total_m"] = df[["125", "63", "31.5", "16",
#                     "8", "4", "2", "1", "0.5",
#                     "0.25", "0.125", "0.063"]].sum(axis=1)
# df["pebble_coarse:125mm"] = df[["125"]].sum(axis=1) / df["total_m"]
# df["pebble_finer:63mm"] = df[["63"]].sum(axis=1) / df["total_m"]
# df["gravel:2-31.5mm"] = df[["31.5", "16",
#                             "8", "4", "2"]].sum(axis=1) / df["total_m"]
# df["sand:0.063-1mm"] = df[["1", "0.5",
#                             "0.25", "0.125", "0.063"]].sum(axis=1) / df["total_m"]
df['name'] = df['kb'].map(str) + "_" + df['meas_number'].map(str)

# __Create data point to be clustered
# df.filter(items=["pebble_coarse","pebble_finer","gravel","sand"])
df_dendrogram = df  # [["name", "pebble_coarse:125mm", "pebble_finer:63mm", "gravel:2-31.5mm", "sand:0.063-1mm"]]
# shovel_array = df_dendrogram.iloc[:, 1:5].to_numpy()
shovel_array = df_dendrogram[["125", "63", "31.5", "16",
                    "8", "4", "2", "1", "0.5",
                    "0.25", "0.125", "0.063"]].to_numpy()
# df_dendrogram.to_csv("Testfile/shovel_samples_clustering.csv", index=False)
# figure seetings
plt.figure(figsize=(10, 10))
plt.title('Shovel Probes OS Hierarchical Clustering Dendrogram', fontsize=20)
plt.xlabel('Shovel probe locations', fontsize=16)
plt.ylabel('Distance %', fontsize=16)
labels2 = ["fine", "medium", "coarse", "very coarse"]
labels = list(df_dendrogram["name"])

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
                          color_threshold=0.07,
                          # p=p,  # show only the last p merged clusters
                          leaf_label_func=llf,
                          leaf_rotation=90.,
                          # show_contracted=True,  # to get a distribution impression in truncated branches
                          )
plt.tight_layout()
plt.show()
print(df)
