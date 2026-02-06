#!/usr/bin/env python3
"""
Enhanced Window Feature Generation Script
Creates windowed features for all drivers with comprehensive statistics
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class WindowFeatureGenerator:
    """Generate windowed features from time-series driving data"""
    
    def __init__(self, window_size=30, overlap=0):
        """
        Initialize feature generator
        
        Args:
            window_size: Window size in seconds (default: 30)
            overlap: Overlap between windows in seconds (default: 0)
        """
        self.window_size = window_size
        self.overlap = overlap
        
        # Define features to aggregate
        self.feature_config = {
            # GPS features
            'speed_kmh': ['mean', 'std', 'min', 'max', 'median'],
            'course': ['mean', 'std'],
            'difcourse': ['mean', 'std', 'max'],  # Steering activity
            
            # Accelerometer features
            'acc_x': ['mean', 'std', 'min', 'max'],
            'acc_y': ['mean', 'std', 'min', 'max'],
            'acc_z': ['mean', 'std', 'min', 'max'],
            'acc_x_kf': ['mean', 'std', 'min', 'max'],
            'acc_y_kf': ['mean', 'std', 'min', 'max'],
            'acc_z_kf': ['mean', 'std', 'min', 'max'],
            'roll': ['mean', 'std', 'min', 'max'],
            'pitch': ['mean', 'std', 'min', 'max'],
            'yaw': ['mean', 'std'],
            
            # Lane detection features
            'x_lane': ['mean', 'std', 'max'],  # Lane position
            'phi': ['mean', 'std', 'max'],  # Steering angle
            'road_width': ['mean'],
            
            # Vehicle detection features
            'dist_front': ['mean', 'std', 'min'],
            'ttc_front': ['mean', 'std', 'min'],
            'num_vehicles': ['mean', 'max'],
            
            # OSM features
            'max_speed': ['mean'],
            'num_lanes': ['mean'],
        }
    
    def load_driver_data(self, filepath):
        """Load time-series data for a driver"""
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} records from {Path(filepath).name}")
        return df
    
    def create_windows(self, df):
        """
        Create time windows from continuous data
        
        Args:
            df: DataFrame with 't_sec' column
            
        Returns:
            DataFrame with 'window_id' column added
        """
        df = df.copy()
        
        if self.overlap == 0:
            # Non-overlapping windows
            df['window_id'] = (df['t_sec'] // self.window_size).astype(int)
        else:
            # Overlapping windows
            step = self.window_size - self.overlap
            df['window_id'] = (df['t_sec'] // step).astype(int)
        
        return df
    
    def compute_derived_features(self, df):
        """
        Add derived features to time-series data
        
        These are computed BEFORE windowing
        """
        df = df.copy()
        
        # Speed features
        if 'speed_kmh' in df.columns and 'max_speed' in df.columns:
            df['speed_ratio'] = df['speed_kmh'] / df['max_speed'].replace(0, np.nan)
            df['speed_over_limit'] = (df['speed_kmh'] - df['max_speed']).clip(lower=0)
        
        # Acceleration derivatives (rate of change)
        if 't_sec' in df.columns:
            time_diff = df['t_sec'].diff().replace(0, np.nan)
            
            if 'speed_kmh' in df.columns:
                df['speed_change_rate'] = df['speed_kmh'].diff() / time_diff
            
            if 'phi' in df.columns:
                df['steering_rate'] = df['phi'].diff() / time_diff
            
            if 'x_lane' in df.columns:
                df['lane_drift_rate'] = df['x_lane'].diff() / time_diff
        
        # Lateral acceleration (cornering force)
        if 'acc_y_kf' in df.columns and 'speed_kmh' in df.columns:
            df['lateral_force'] = df['acc_y_kf'] * df['speed_kmh']
        
        # Braking intensity (negative acceleration only)
        if 'acc_x_kf' in df.columns:
            df['braking_intensity'] = df['acc_x_kf'].clip(upper=0).abs()
        
        return df
    
    def aggregate_window(self, window_df):
        """
        Compute aggregate statistics for a single window
        
        Args:
            window_df: DataFrame for one window
            
        Returns:
            Dictionary of aggregated features
        """
        features = {}
        
        # Apply configured aggregations
        for col, agg_funcs in self.feature_config.items():
            if col in window_df.columns:
                for func in agg_funcs:
                    if func == 'mean':
                        features[f'{col}_mean'] = window_df[col].mean()
                    elif func == 'std':
                        features[f'{col}_std'] = window_df[col].std()
                    elif func == 'min':
                        features[f'{col}_min'] = window_df[col].min()
                    elif func == 'max':
                        features[f'{col}_max'] = window_df[col].max()
                    elif func == 'median':
                        features[f'{col}_median'] = window_df[col].median()
        
        # Add derived feature aggregations
        derived_features = [
            'speed_ratio', 'speed_over_limit', 'speed_change_rate',
            'steering_rate', 'lane_drift_rate', 'lateral_force', 
            'braking_intensity'
        ]
        
        for feat in derived_features:
            if feat in window_df.columns:
                features[f'{feat}_mean'] = window_df[feat].mean()
                features[f'{feat}_std'] = window_df[feat].std()
        
        # Custom features
        
        # Lane crossing frequency (how often crosses center)
        if 'x_lane' in window_df.columns:
            x_lane_values = window_df['x_lane'].values
            sign_changes = np.sum(np.diff(np.sign(x_lane_values)) != 0)
            features['lane_crossing_freq'] = sign_changes / len(window_df)
        
        # Steering entropy (measure of unpredictability)
        if 'phi' in window_df.columns:
            phi_values = window_df['phi'].values
            if len(phi_values) > 0:
                # Discretize into bins and calculate entropy
                hist, _ = np.histogram(phi_values, bins=10)
                hist = hist / hist.sum()
                entropy = -np.sum(hist * np.log(hist + 1e-10))
                features['steering_entropy'] = entropy
        
        # Harsh braking events
        if 'acc_x_kf' in window_df.columns:
            harsh_braking = (window_df['acc_x_kf'] < -0.3).sum()
            features['harsh_braking_count'] = harsh_braking
        
        # Harsh acceleration events
        if 'acc_x_kf' in window_df.columns:
            harsh_accel = (window_df['acc_x_kf'] > 0.3).sum()
            features['harsh_accel_count'] = harsh_accel
        
        # Average reaction time to vehicle ahead
        if 'dist_front' in window_df.columns and 'ttc_front' in window_df.columns:
            # Only when vehicle detected
            vehicle_ahead = window_df[window_df['dist_front'] > 0]
            if len(vehicle_ahead) > 0:
                features['avg_reaction_time'] = vehicle_ahead['ttc_front'].mean()
            else:
                features['avg_reaction_time'] = np.nan
        
        return features
    
    def generate_window_features(self, df):
        """
        Generate windowed features from time-series data
        
        Args:
            df: DataFrame with time-series sensor data
            
        Returns:
            DataFrame with windowed features
        """
        print(f"\n{'='*60}")
        print("Generating Window Features")
        print(f"{'='*60}")
        
        # Compute derived features
        print("Computing derived features...")
        df = self.compute_derived_features(df)
        
        # Create windows
        print(f"Creating {self.window_size}-second windows...")
        df = self.create_windows(df)
        
        # Group by window and metadata
        groupby_cols = ['window_id']
        metadata_cols = ['driver', 'behavior', 'road_type']
        
        for col in metadata_cols:
            if col in df.columns:
                groupby_cols.append(col)
        
        # Aggregate each window
        print("Aggregating window statistics...")
        window_list = []
        
        for group_vals, window_df in df.groupby(groupby_cols):
            if len(window_df) < 3:  # Skip very small windows
                continue
            
            # Get metadata
            if isinstance(group_vals, tuple):
                window_id = group_vals[0]
                metadata = {groupby_cols[i]: group_vals[i] 
                           for i in range(len(groupby_cols))}
            else:
                window_id = group_vals
                metadata = {'window_id': window_id}
            
            # Aggregate features
            features = self.aggregate_window(window_df)
            features.update(metadata)
            
            window_list.append(features)
        
        # Create DataFrame
        window_features = pd.DataFrame(window_list)
        
        print(f"✓ Generated {len(window_features)} windows")
        
        return window_features
    
    def encode_labels(self, df):
        """
        Encode behavior labels as numeric
        
        Args:
            df: DataFrame with 'behavior' column
            
        Returns:
            DataFrame with 'label' column added
        """
        if 'behavior' not in df.columns:
            return df
        
        label_map = {
            'Normal': 0,
            'Drowsy': 1,
            'Aggressive': 2
        }
        
        df['label'] = df['behavior'].map(label_map)
        
        return df
    
    def save_features(self, window_features, output_path, split_by_road=True):
        """
        Save window features to CSV
        
        Args:
            window_features: DataFrame with features
            output_path: Output directory path
            split_by_road: If True, create separate files per road type
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Encode labels
        window_features = self.encode_labels(window_features)
        
        if split_by_road and 'road_type' in window_features.columns:
            # Save by road type
            for road_type in window_features['road_type'].unique():
                road_df = window_features[window_features['road_type'] == road_type]
                filename = f"{road_type}_window_features.csv"
                filepath = output_path / filename
                road_df.to_csv(filepath, index=False)
                print(f"✓ Saved {len(road_df)} windows to {filename}")
        
        # Save combined file
        combined_path = output_path / "all_window_features.csv"
        window_features.to_csv(combined_path, index=False)
        print(f"✓ Saved {len(window_features)} windows to all_window_features.csv")


def main():
    """Main execution"""
    print("\n" + "="*80)
    print(" Window Feature Generation Tool")
    print("="*80)
    
    # Configuration
    print("\nConfiguration:")
    window_size = int(input("Window size in seconds [30]: ").strip() or "30")
    overlap = int(input("Overlap in seconds [0]: ").strip() or "0")
    
    print(f"\n✓ Window size: {window_size} seconds")
    print(f"✓ Overlap: {overlap} seconds")
    
    # Input file
    print("\n" + "-"*80)
    input_file = input("Path to time-series CSV file: ").strip()
    
    if not Path(input_file).exists():
        print(f"ERROR: File not found: {input_file}")
        return
    
    # Output directory
    output_dir = input("Output directory [./window_features]: ").strip() or "./window_features"
    
    # Generate features
    generator = WindowFeatureGenerator(window_size=window_size, overlap=overlap)
    
    # Load data
    df = generator.load_driver_data(input_file)
    
    # Generate features
    window_features = generator.generate_window_features(df)
    
    # Save features
    split_by_road = input("\nSplit by road type? (y/n) [y]: ").lower().strip() != 'n'
    generator.save_features(window_features, output_dir, split_by_road)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")
    print(f"Total windows: {len(window_features)}")
    
    if 'behavior' in window_features.columns:
        print("\nClass distribution:")
        print(window_features['behavior'].value_counts())
    
    if 'road_type' in window_features.columns:
        print("\nRoad type distribution:")
        print(window_features['road_type'].value_counts())
    
    print("\n✓ Feature generation complete!")


if __name__ == '__main__':
    main()
