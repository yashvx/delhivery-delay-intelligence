
import numpy as np
import pandas as pd

PROMISED_HOURS = {
    "Express": 36,
    "Standard": 60,
    "Economy": 96,
}

FEATURES = [
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
]

NOISE_COLUMNS = [
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no",
]


def clean_shipments(raw, medians=None):
    """Clean shipment records and return the cleaned data and fill medians."""
    df = raw.drop_duplicates().copy()

    # Normalize service names.
    df["service_type"] = df["service_type"].str.strip().str.title()

    # Represent missing payment modes explicitly.
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")

    # Convert declared values from text such as ₹1,290 to numbers.
    declared_value = (
        df["declared_value"]
        .str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False)
    )
    df["declared_value_inr"] = pd.to_numeric(
        declared_value, errors="coerce"
    )
    df = df.drop(columns=["declared_value"])

    # Convert dispatch timestamps into datetimes.
    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"])

    # Treat implausibly high parcel weights as scanner-fault sentinels.
    df.loc[df["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan

    # Compute medians from historical data, or reuse supplied medians.
    columns_to_fill = [
        "distance_km",
        "parcel_weight_kg",
        "declared_value_inr",
    ]

    if medians is None:
        medians = {
            column: float(df[column].median())
            for column in columns_to_fill
        }
    else:
        medians = {
            column: float(medians[column])
            for column in columns_to_fill
        }

    # Fill missing numeric values.
    for column in columns_to_fill:
        df[column] = df[column].fillna(medians[column])

    return df, medians


def build_features(clean, hubs):
    """Attach hub information and construct the model input features."""
    df = clean.merge(
        hubs[
            [
                "hub_id",
                "hub_city",
                "region",
                "hub_tier",
                "hub_daily_capacity",
            ]
        ],
        on="hub_id",
        how="left",
        validate="many_to_one",
    )

    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS).astype(int)
    df["is_cod"] = (df["payment_mode"] == "COD").astype(int)

    weekday = df["dispatch_time"].dt.dayofweek
    df["is_weekend"] = (weekday >= 5).astype(int)
    df["dispatch_hour"] = df["dispatch_time"].dt.hour.astype(int)

    df["log_declared_value"] = np.log(df["declared_value_inr"])

    return df
