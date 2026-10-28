
import pandas as pd

# Load the generated CSV files
hubs = pd.read_csv("data/hubs.csv")
raw = pd.read_csv("data/raw_shipments.csv")
new = pd.read_csv("data/new_shipments.csv")

# 1. Check dataset dimensions
print("\n--- DATASET SHAPES ---")
print("Hubs:", hubs.shape)
print("Raw shipments:", raw.shape)
print("New shipments:", new.shape)

# 2. Preview historical shipment records
print("\n--- RAW SHIPMENT SAMPLE ---")
print(raw.head())

# 3. Identify missing values
print("\n--- MISSING VALUES ---")
print(raw.isnull().sum())

# 4. Identify duplicate shipment IDs
print("\n--- DUPLICATE SHIPMENT IDS ---")
print(raw["shipment_id"].duplicated().sum())

# 5. Inspect column data types
print("\n--- COLUMN DATA TYPES ---")
print(raw.dtypes)
