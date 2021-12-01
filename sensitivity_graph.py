from config import *

file = "training_csv/sensitivity_new/shovel_newparameters.csv"
df = pd.read_csv(file)
n_classes = len(df.columns)


in_colm = False
out_colm = False
shovel = True

if in_colm:
    df.rename(columns={"0":"2","1":"3","2":"4","3":"5" }, inplace=True)

if shovel:
    df.rename(columns={"0": "Courser Cobble", "1": "Gravel Pred.", "2": "Finer Cobble", "3": "Sand"}, inplace=True)

# example data
y = []
std = []
# x = np.array(range(1, n_classes+1, 1))
x = np.array(df.columns)
for columns in df.columns:
    y.append(np.mean(df[columns]))
    std.append(np.std(df[columns]))

# x = np.arange(0.1, 4, 0.5)
# y = np.exp(-x)
std = np.array(std)
y = np.array(y)

# plot sensibility model analysis
fig = plt.figure(figsize=(6.18, 3.82), dpi=150, facecolor="w", edgecolor="gray")
axes = fig.add_subplot(1, 1, 1)
axes.set_ylim([0, 1])
axes.set_xlim([-1, n_classes])
# axes.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel('Grain size class')
plt.ylabel('Sensitivity')
axes.errorbar(x, y, yerr= std, fmt='o')
for x, y in zip(x,y):
    label = "{:.2f}".format(y)

    plt.annotate(label, # this is the text
                 (x,y), # these are the coordinates to position the label
                 textcoords="offset points", # how to position the text
                 xytext=(15,-15), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
plt.show()
