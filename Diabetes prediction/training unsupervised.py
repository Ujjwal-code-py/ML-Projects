import pandas as pd
import numpy as np
import pickle
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Create model folder if not exists
os.makedirs("model", exist_ok=True)

# Load dataset and sample for memory efficiency
df = pd.read_csv("diabetes_012_health_indicators_BRFSS2021.csv")
df = df.sample(n=25000, random_state=42)  # Full training limited to 25,000

# ===============================
# 🧠 CLUSTERING (KMeans)
# ===============================
features_for_clustering = ['BMI', 'Age', 'GenHlth', 'PhysHlth', 'MentHlth']
X_cluster = df[features_for_clustering]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# Train KMeans model
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(X_scaled)

# Save scaler and clustering model
with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("model/kmeans.pkl", "wb") as f:
    pickle.dump(kmeans, f)

print("✅ Clustering model and scaler saved.")

# ===============================
# 🔍 ASSOCIATION RULE MINING (Optimized)
# ===============================
# Use smaller sample and fewer binary features
sample_df = df.sample(n=5000, random_state=42)
binary_features = [
    'HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Fruits',
    'Veggies', 'PhysActivity', 'HvyAlcoholConsump', 'AnyHealthcare'
]

# Encode features to readable form
encoded_df = pd.DataFrame()
for col in binary_features:
    encoded_df[col] = sample_df[col].apply(lambda x: f"{col}_Yes" if x == 1 else f"{col}_No")

# Convert to transactions
transactions = encoded_df.astype(str).values.tolist()

# Encode for Apriori
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_trans = pd.DataFrame(te_ary, columns=te.columns_)

# Apply Apriori and extract rules
frequent_itemsets = apriori(df_trans, min_support=0.03, use_colnames=True)
if frequent_itemsets.empty:
    print("⚠️ No frequent itemsets found.")
else:
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
    if rules.empty:
        print("⚠️ No rules met the confidence threshold.")
    else:
        rules.to_csv("model/rules.csv", index=False)
        print("✅ Association rules saved to model/rules.csv.")
