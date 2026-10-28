## Part A — Data Cleaning and Exploratory Data Analysis

### Data preparation

The raw dataset contained 1,224 shipment rows. Removing 24 exact duplicate rows left 1,200 unique shipments. Missing numeric values were imputed using the historical medians: 314.95 km for distance, 1.83 kg for parcel weight, and INR 1,121.50 for declared value. Service-type text was normalized from 12 distinct spellings to three standardized categories. The modelling table contains 1,200 rows, and the required features and target columns have no missing values.

### Data findings

1. **Service type:** Express shipments had the highest delay rate at 48.97% (119 of 243 shipments), compared with 20.56% for Standard and 8.57% for Economy.

2. **Hub tier:** The delay rate increased from 4.00% at Tier 1 to 21.50% at Tier 2 and 44.00% at Tier 3. Each tier contained 400 shipments in this synthetic dataset.

3. **Distance:** Delay rates rose across the distance bands: 2.19% for 0–200 km, 13.68% for 201–400 km, 43.64% for 401–600 km, and 85.23% for shipments over 600 km.

4. **Correlation with delivery time:** The three strongest correlations with `delivery_hours`, excluding the target itself, were `promised_hours` (+0.8372), `distance_km` (+0.4013), and `n_stops` (+0.3247). These are associations, not proof of causation.

5. **Declared-value transformation:** The median-filled declared-value column had a mean of INR 1,533.05 and a median of INR 1,121.50. After the natural-log transformation, the mean was 7.028 and the median was 7.022. The log transformation reduces the influence of large values and makes this right-skewed variable more balanced.

### Reproducibility note

The dataset used in this project is synthetic and was generated for the capstone exercise. It does not contain real Delhivery shipment records, and this project is not affiliated with Delhivery.

