
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/detect_dataset.csv")
df = df.dropna(axis=1, how="all")
df = df.rename(columns={"Output (S)": "fault"})

df = df.head(2000)

plt.figure(figsize=(14, 7))

plt.plot(df.index, df["Ia"], label="Ia")
plt.plot(df.index, df["Ib"], label="Ib")
plt.plot(df.index, df["Ic"], label="Ic")

fault_rows = df["fault"] == 1
plt.fill_between(
    df.index,
    0,
    1,
    where=fault_rows,
    transform=plt.gca().get_xaxis_transform(),
    alpha=0.2,
    color="red",
    label="Fault"
)

plt.xlabel("Row number")
plt.ylabel("Current (A)")
plt.title("Three-Phase Currents for First 2000 Rows")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/plot1_currents.png")
plt.show()
plt.figure(figsize=(8, 6))

plt.scatter(
    df["Ia"],
    df["Ib"],
    c=df["fault"],
    alpha=0.6
)

plt.xlabel("Ia (A)")
plt.ylabel("Ib (A)")
plt.title("Ia vs Ib (Colored by Fault)")
plt.colorbar(label="Fault (0 = No Fault, 1 = Fault)")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/plot2_scatter.png")
plt.show()

plt.figure(figsize=(10, 6))

plt.hist(
    df.loc[df["fault"] == 0, "Ia"],
    bins=40,
    alpha=0.6,
    label="No Fault (0)"
)

plt.hist(
    df.loc[df["fault"] == 1, "Ia"],
    bins=40,
    alpha=0.6,
    label="Fault (1)"
)

plt.xlabel("Ia (A)")
plt.ylabel("Frequency")
plt.title("Distribution of Ia: Fault vs No Fault")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/plot3_fault_distribution.png")
plt.show()
