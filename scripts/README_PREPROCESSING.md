# UAH-DriveSet Data Preprocessing Pipeline

This preprocessing pipeline converts the UAH-DriveSet raw data into structured CSV files suitable for machine learning and analysis.

## 📁 Expected Data Structure

Your data should be organized as follows:
```
UAH-DRIVESET-v1/
├── D1/
│   ├── 20151111135612-13km-D1-DROWSY-SECONDARY/
│   │   ├── RAW_ACCELEROMETERS.txt
│   │   ├── RAW_GPS.txt
│   │   ├── PROC_LANE_DETECTION.txt
│   │   ├── PROC_VEHICLE_DETECTION.txt
│   │   ├── PROC_OPENSTREETMAP_DATA.txt
│   │   └── 20151111123123-25km-D1-NORMAL-M.MP4
│   ├── 20151111134545-16km-D1-AGGRESSIVE-SECONDARY/
│   └── ...
├── D2/
├── D3/
└── ...
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn --break-system-packages
```

### 2. Run Preprocessing

```bash
python3 preprocess_driveset.py
```

**You will be prompted for:**
- Path to your UAH-DRIVESET-v1 folder
- Output directory for processed files
- Sampling rate preference:
  - **1 Hz (recommended)**: Good balance for ML - 1 sample per second
  - **10 Hz**: Higher resolution - 10 samples per second
  - **Original**: Keep all data points (largest files)
  - **0.1 Hz**: Downsampled - 1 sample per 10 seconds

### 3. Analyze the Data

```bash
python3 analyze_data.py
```

This will generate:
- Comprehensive statistics report
- Summary CSV file
- Visualizations (optional)

## 📊 Output Files

The preprocessing creates the following files in `processed_data/`:

### Core Files

1. **master_routes.csv**
   - Metadata for each route
   - Columns: `route_id`, `driver_id`, `behavior_type`, `road_type`, `route_date`, `route_distance_km`, etc.

2. **raw_accelerometers.csv**
   - Accelerometer sensor data
   - Columns: `timestamp`, `route_id`, `accel_x_raw`, `accel_y_raw`, `accel_z_raw`, `accel_x_filtered`, `accel_y_filtered`, `accel_z_filtered`, `roll`, `pitch`, `yaw`

3. **raw_gps.csv**
   - GPS location and speed data
   - Columns: `timestamp`, `route_id`, `speed_kmh`, `latitude`, `longitude`, `altitude`, `course`, etc.

4. **proc_lane_detection.csv**
   - Lane position data
   - Columns: `timestamp`, `route_id`, `lane_position`, `phi`, `road_width`, `lane_state`

5. **proc_vehicle_detection.csv**
   - Vehicle detection data
   - Columns: `timestamp`, `route_id`, `distance_ahead`, `time_to_impact`, `num_vehicles`

6. **proc_openstreetmap_data.csv**
   - Road information from OpenStreetMap
   - Columns: `timestamp`, `route_id`, `max_speed_limit`, `road_type`, `num_lanes`, `current_lane`

### Optional Files

7. **merged_all_sensors.csv** (if requested)
   - All sensors merged by timestamp
   - Warning: Can be very large!

8. **data_summary_statistics.csv**
   - Statistical summary of all variables

## 🎯 Recommended Workflow for ML

### For Classification Models (Behavior Type Prediction)

```python
import pandas as pd

# Load metadata
routes = pd.read_csv('processed_data/master_routes.csv')

# Load sensor data
gps = pd.read_csv('processed_data/raw_gps.csv')
accel = pd.read_csv('processed_data/raw_accelerometers.csv')

# Merge with metadata to get labels
gps_labeled = gps.merge(routes[['route_id', 'behavior_type', 'driver_id']], on='route_id')

# Now you have features (sensor data) and labels (behavior_type)
```

### Feature Engineering Examples

```python
# Calculate aggregated features per route
features_per_route = gps_labeled.groupby('route_id').agg({
    'speed_kmh': ['mean', 'std', 'max', 'min'],
    'course_variation': ['mean', 'std'],
    # Add more features
})

# Merge with accelerometer features
accel_features = accel.groupby('route_id').agg({
    'accel_x_filtered': ['mean', 'std'],
    'accel_y_filtered': ['mean', 'std'],
    'accel_z_filtered': ['mean', 'std'],
})

# Combine all features
all_features = features_per_route.join(accel_features)

# Add labels
all_features = all_features.merge(
    routes[['route_id', 'behavior_type']], 
    on='route_id'
)
```

## 📈 Data Structure Overview

### Behavior Types
Based on the folder names, the dataset includes:
- **NORMAL**: Normal driving behavior
- **DROWSY**: Drowsy/fatigued driving
- **AGGRESSIVE**: Aggressive driving patterns

### Road Types
- **MOTORWAY**: Highway driving
- **SECONDARY**: Secondary roads
- **NORMAL**: Standard roads

### Drivers
- **D1** through **D6**: Six different drivers

## 🔍 Data Quality Notes

1. **Timestamps**: All sensor data uses synchronized timestamps (in seconds)
2. **Missing Values**: Some sensors may have gaps - check with `analyze_data.py`
3. **Sampling Rates**: Original data has varying sampling rates across sensors
4. **GPS Accuracy**: Check `horizontal_accuracy` and `vertical_accuracy` fields

## 💡 Tips for Analysis

### 1. Balance Your Dataset
```python
routes['behavior_type'].value_counts()
# Use stratified sampling if imbalanced
```

### 2. Create Time Windows
```python
# Extract 30-second windows for classification
window_size = 30  # seconds
gps['window_id'] = (gps['timestamp'] // window_size).astype(int)
```

### 3. Handle Multi-Driver Data
```python
# Leave-one-driver-out cross-validation
for test_driver in routes['driver_id'].unique():
    train = routes[routes['driver_id'] != test_driver]
    test = routes[routes['driver_id'] == test_driver]
    # Train and evaluate
```

## 🐛 Troubleshooting

### Issue: "No route folders found"
- Check that your base path contains D1, D2, etc. folders
- Ensure route folders follow the naming convention

### Issue: "Could not load XXX.txt"
- Some routes may be missing certain sensor files
- This is normal - the script will skip missing files

### Issue: "Memory error with merged dataset"
- Don't create the merged dataset (choose 'n' when prompted)
- Use individual sensor files instead
- Work with one route at a time

## 📝 Example: Complete ML Pipeline

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. Load data
routes = pd.read_csv('processed_data/master_routes.csv')
gps = pd.read_csv('processed_data/raw_gps.csv')
accel = pd.read_csv('processed_data/raw_accelerometers.csv')

# 2. Engineer features per route
features = gps.groupby('route_id').agg({
    'speed_kmh': ['mean', 'std', 'max'],
}).reset_index()

features.columns = ['route_id', 'speed_mean', 'speed_std', 'speed_max']

# 3. Add labels
data = features.merge(routes[['route_id', 'behavior_type']], on='route_id')

# 4. Prepare for ML
X = data[['speed_mean', 'speed_std', 'speed_max']]
y = data['behavior_type']

# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 7. Evaluate
accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2%}")
```

## 📄 License

This preprocessing tool is provided as-is for use with the UAH-DriveSet dataset.
Please refer to the original dataset license for data usage terms.

## 🤝 Contributing

Feel free to modify these scripts for your specific needs!

## ❓ Questions?

If you encounter issues or have questions about the preprocessing pipeline, please check:
1. The data structure matches the expected format
2. All dependencies are installed
3. File permissions allow reading the data files
