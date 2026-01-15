import utils.importingfile as i

df = i.pd.read_csv("data/winequality-red.csv", sep=";")  # or white

features = [c for c in df.columns if c != "quality"]
n = len(features)

cols = 3
rows = i.math.ceil(n / cols)

fig, axes = i.plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
axes = axes.flatten()

y = df["quality"]

for ax, col in zip(axes, features):
    ax.scatter(df[col], y, alpha=0.4)
    ax.set_xlabel(col)
    ax.set_ylabel("quality")
    ax.set_title(col)

# Skjul tomme plott
for ax in axes[len(features):]:
    ax.axis("off")

i.plt.tight_layout()
i.plt.show()