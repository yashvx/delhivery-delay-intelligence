import pandas as pd

from src.shipment_prep import clean_shipments, build_features


# 1. Load the datasets
raw = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")

print("Raw shipments:", raw.shape)
print("Hubs:", hubs.shape)


# 2. Clean the shipment data
clean, medians = clean_shipments(raw)

print("\nCleaned shipments:", clean.shape)
print("Imputation medians:", medians)


# 3. Build features using shipment and hub data
modelling = build_features(clean, hubs)

# 4. Mark shipments as delayed if they exceed promised hours
modelling["is_delayed"] = (
    modelling["delivery_hours"] > modelling["promised_hours"]
).astype(int)


# 5. Save the modelling table
modelling.to_csv("data/modelling_table.csv", index=False)

print("\nModelling table:", modelling.shape)
print("Delayed shipments:", modelling["is_delayed"].sum())
print("Saved data/modelling_table.csv")



# 6. Calculate the required Part A checks

# Deduplicate before measuring the original data-quality issues.
before = raw.drop_duplicates().copy()

# Count implausibly high weights separately from missing weights.
sentinel_weight_rows = int((before["parcel_weight_kg"] > 100).sum())

# Calculate quartiles and outlier bounds from delivery hours.
q1 = modelling["delivery_hours"].quantile(0.25)
q3 = modelling["delivery_hours"].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

# The model's required features plus its target columns.
required_columns = [
    "promised_hours", "distance_km", "parcel_weight_kg",
    "log_declared_value", "n_stops", "rain_flag",
    "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity", "label_print_batch",
    "scanner_build_no", "tape_roll_no", "sticker_serial_mod",
    "shelf_slot_no", "delivery_hours", "is_delayed",
]

declared = modelling["declared_value_inr"]
log_declared = modelling["log_declared_value"]

checks = [
    ("raw_rows", len(raw)),
    ("duplicate_rows_removed", len(raw) - len(before)),
    ("rows_after_dedup", len(before)),
    ("missing_distance_km_before", int(before["distance_km"].isna().sum())),
    ("missing_parcel_weight_before", int(before["parcel_weight_kg"].isna().sum())),
    ("sentinel_weight_rows", sentinel_weight_rows),
    ("missing_declared_value_before", int(before["declared_value"].isna().sum())),
    ("missing_payment_mode_before", int(before["payment_mode"].isna().sum())),
    ("service_type_spellings_before", int(before["service_type"].nunique())),
    ("service_type_spellings_after", int(clean["service_type"].nunique())),
    ("median_distance_km", round(medians["distance_km"], 2)),
    ("median_parcel_weight_kg", round(medians["parcel_weight_kg"], 2)),
    ("median_declared_value_inr", round(medians["declared_value_inr"], 2)),
    ("rows_after_hub_merge", len(modelling)),
    (
        "nulls_in_features_and_target",
        int(modelling[required_columns].isna().sum().sum()),
    ),
    ("delayed_count", int(modelling["is_delayed"].sum())),
    ("delay_rate", round(modelling["is_delayed"].mean(), 4)),
    ("hours_q1", round(q1, 2)),
    ("hours_q3", round(q3, 2)),
    ("hours_lower_bound", round(lower_bound, 2)),
    ("hours_upper_bound", round(upper_bound, 2)),
    (
        "hours_outlier_count",
        int(
            (
                (modelling["delivery_hours"] < lower_bound)
                | (modelling["delivery_hours"] > upper_bound)
            ).sum()
        ),
    ),
    ("declared_value_mean_inr", round(declared.mean(), 2)),
    ("declared_value_median_inr", round(declared.median(), 2)),
    ("log_declared_value_mean", round(log_declared.mean(), 3)),
    ("log_declared_value_median", round(log_declared.median(), 3)),
]

checks_df = pd.DataFrame(checks, columns=["metric", "value"])
checks_df.to_csv("outputs/part_a_checks.csv", index=False)

print("\nSaved outputs/part_a_checks.csv")



# 7. Analyze delays by service type, hub tier, and distance band

# Define the required distance ranges.
modelling["distance_band"] = pd.cut(
    modelling["distance_km"],
    bins=[0, 200, 400, 600, 2000],
    labels=["0-200", "201-400", "401-600", "601+"],
    right=True,
    include_lowest=True,
)

# Build one summary row for each group.
group_reports = []

for dimension in ["service_type", "hub_tier", "distance_band"]:
    grouped = modelling.groupby(
        dimension, observed=True, dropna=False
    )

    summary = grouped.agg(
        shipments=("shipment_id", "size"),
        delayed=("is_delayed", "sum"),
        delay_rate=("is_delayed", "mean"),
        mean_hours=("delivery_hours", "mean"),
    ).reset_index()

    summary = summary.rename(columns={dimension: "group"})
    summary.insert(0, "dimension", dimension)
    group_reports.append(summary)

delay_by_group = pd.concat(group_reports, ignore_index=True)

delay_by_group["delay_rate"] = delay_by_group["delay_rate"].round(4)
delay_by_group["mean_hours"] = delay_by_group["mean_hours"].round(2)

delay_by_group.to_csv(
    "outputs/part_a_delay_by_group.csv", index=False
)

print("Saved outputs/part_a_delay_by_group.csv")



# 8. Calculate correlations between numeric columns

numeric_columns = modelling.select_dtypes(include="number")
correlations = numeric_columns.corr()

correlations.to_csv("outputs/part_a_correlations.csv", index=True)

print("Saved outputs/part_a_correlations.csv")
