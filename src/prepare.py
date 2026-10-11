import numpy as np
from load_data import load_raw

df = load_raw()

df = df.dropna(axis=1, how="all")
df = df.rename(columns={"Output (S)": "fault"})

print("Cleaned dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())
rng = np.random.default_rng(42)
idx = rng.permutation(len(df))

n_train = int(0.6 * len(df))
n_val = int(0.2 * len(df))

train_idx = idx[:n_train]
val_idx = idx[n_train:n_train + n_val]
test_idx = idx[n_train + n_val:]

train_df = df.iloc[train_idx].copy()
val_df = df.iloc[val_idx].copy()
test_df = df.iloc[test_idx].copy()

print("\nTrain rows:", len(train_df))
print("Validation rows:", len(val_df))
print("Test rows:", len(test_df))

print("\nFault fraction in each split:")
print("Train:", train_df["fault"].mean())
print("Validation:", val_df["fault"].mean())
print("Test:", test_df["fault"].mean())

features = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]

train_mean = train_df[features].mean()
train_std = train_df[features].std()
# Save Set R scaling parameters for the frontend
train_mean.to_csv("results/scaler_R_mean.csv")
train_std.to_csv("results/scaler_R_std.csv")

print("\nTraining means:")
print(train_mean)

print("\nTraining standard deviations:")
print(train_std)

train_scaled = train_df.copy()
val_scaled = val_df.copy()
test_scaled = test_df.copy()

train_scaled[features] = (train_df[features] - train_mean) / train_std
val_scaled[features] = (val_df[features] - train_mean) / train_std
test_scaled[features] = (test_df[features] - train_mean) / train_std

print("\nStandardized training features (first five rows):")
print(train_scaled[features].head())
print("\nVerification:")
print("Training means after standardization:")
print(train_scaled[features].mean().round(6))

print("\nTraining standard deviations after standardization:")
print(train_scaled[features].std().round(6))
train_scaled.to_csv("results/train.csv", index=False)
val_scaled.to_csv("results/validation.csv", index=False)
test_scaled.to_csv("results/test.csv", index=False)

print("\nPrepared datasets saved successfully.")

# Feature Set R: six raw features
features_R = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]

# Create squared features for Set Q
for feature in features_R:
    df[feature + "_sq"] = df[feature] ** 2

# Feature Set Q: raw features plus their squares
features_Q = features_R + [
    "Ia_sq", "Ib_sq", "Ic_sq",
    "Va_sq", "Vb_sq", "Vc_sq"
]

print("\nFeature Set R size:", len(features_R))
print("Feature Set Q size:", len(features_Q))
print("\nFirst five rows of Set Q:")
print(df[features_Q].head())

# Prepare Feature Set Q for each split
train_Q = train_df.copy()
val_Q = val_df.copy()
test_Q = test_df.copy()

# Create squared features in each split
for feature in features_R:
    square_feature = feature + "_sq"

    train_Q[square_feature] = train_Q[feature] ** 2
    val_Q[square_feature] = val_Q[feature] ** 2
    test_Q[square_feature] = test_Q[feature] ** 2

# Feature Set Q: six raw features plus six squared features
features_Q = features_R + [
    "Ia_sq", "Ib_sq", "Ic_sq",
    "Va_sq", "Vb_sq", "Vc_sq"
]

# Calculate statistics from training data only
train_mean_Q = train_Q[features_Q].mean()
train_std_Q = train_Q[features_Q].std()
train_std_Q = train_std_Q.replace(0, 1)
# Save Set Q scaling parameters for the frontend
train_mean_Q.to_csv("results/scaler_Q_mean.csv")
train_std_Q.to_csv("results/scaler_Q_std.csv")

# Standardize using training statistics
train_Q[features_Q] = (
    train_Q[features_Q] - train_mean_Q
) / train_std_Q

val_Q[features_Q] = (
    val_Q[features_Q] - train_mean_Q
) / train_std_Q

test_Q[features_Q] = (
    test_Q[features_Q] - train_mean_Q
) / train_std_Q

print("\nFeature Set R size:", len(features_R))
print("Feature Set Q size:", len(features_Q))
print("\nSet Q standardized successfully.")
print("Train Q shape:", train_Q.shape)
print("Validation Q shape:", val_Q.shape)
print("Test Q shape:", test_Q.shape)

print("\nStandardized Set Q training means:")
print(train_Q[features_Q].mean().round(6))

# Save standardized data for both feature sets
train_Q.to_csv("results/train_Q.csv", index=False)
val_Q.to_csv("results/validation_Q.csv", index=False)
test_Q.to_csv("results/test_Q.csv", index=False)

print("\nSet Q datasets saved successfully.")
