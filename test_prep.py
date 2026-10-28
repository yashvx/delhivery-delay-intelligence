
import pandas as pd

from src.shipment_prep import clean_shipments, build_features

raw = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")

clean, medians = clean_shipments(raw)
features = build_features(clean, hubs)

print("Raw rows:", len(raw))
print("Clean rows:", len(clean))
print("Unique shipment IDs:", clean["shipment_id"].nunique())

print("\nMedians:")
print(medians)

print("\nMissing values in cleaned data:")
print(clean.isnull().sum().sum())

print("\nService types:")
print(sorted(clean["service_type"].unique()))

print("\nFeature columns:")
print(features.columns.tolist())

print("\nAll 17 required features exist:")
print(all(column in features.columns for column in [
    "promised_hours",
    "distance_km",
    "parcel_weight_kg",
    "log_declared_value",
    "n_stops",
    "rain_flag",
    "sale_event_flag",
    "is_cod",
    "is_weekend",
    "dispatch_hour",
    "hub_tier",
    "hub_daily_capacity",
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no",
]))

