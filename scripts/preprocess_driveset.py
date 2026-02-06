#!/usr/bin/env python3
"""
UAH-DriveSet Data Preprocessing Script
Converts raw driver data folders into structured CSV files for analysis
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import re
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class DriveSetPreprocessor:
    """Preprocesses UAH-DriveSet data into structured CSV files"""
    
    def __init__(self, base_path: str, output_path: str = './processed_data'):
        """
        Initialize preprocessor
        
        Args:
            base_path: Path to UAH-DRIVESET-v1 folder containing driver folders (D1, D2, etc.)
            output_path: Path where processed CSV files will be saved
        """
        self.base_path = Path(base_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Define expected data files and their column mappings
        self.file_columns = {
            'RAW_ACCELEROMETERS.txt': [
                'timestamp', 'activation_flag', 'accel_x_raw', 'accel_y_raw', 'accel_z_raw',
                'accel_x_filtered', 'accel_y_filtered', 'accel_z_filtered',
                'roll', 'pitch', 'yaw'
            ],
            'RAW_GPS.txt': [
                'timestamp', 'speed_kmh', 'latitude', 'longitude', 'altitude',
                'vertical_accuracy', 'horizontal_accuracy', 'course', 'course_variation',
                'position_state', 'lanex_dist_state', 'lanex_history'
            ],
            'PROC_LANE_DETECTION.txt': [
                'timestamp', 'lane_position', 'phi', 'road_width', 'lane_state'
            ],
            'PROC_VEHICLE_DETECTION.txt': [
                'timestamp', 'distance_ahead', 'time_to_impact', 'num_vehicles', 'speed_kmh'
            ],
            'PROC_OPENSTREETMAP_DATA.txt': [
                'timestamp', 'max_speed_limit', 'maxspeed_reliability', 'road_type',
                'num_lanes', 'current_lane', 'latitude_osm', 'longitude_osm',
                'osm_delay', 'speed_kmh'
            ]
        }
    
    def extract_route_metadata(self, folder_name: str) -> Dict[str, str]:
        """
        Extract metadata from route folder name
        Format: YYYYMMDDHHMMSS-XXkm-DY-BEHAVIOR-ROADTYPE
        
        Example: 20151111135612-13km-D1-DROWSY-SECONDARY
        """
        parts = folder_name.split('-')
        
        metadata = {
            'route_folder': folder_name,
            'route_date': parts[0] if len(parts) > 0 else '',
            'route_distance_km': parts[1].replace('km', '') if len(parts) > 1 else '',
            'driver_id': parts[2] if len(parts) > 2 else '',
            'behavior_type': parts[3] if len(parts) > 3 else '',
            'road_type': parts[4] if len(parts) > 4 else ''
        }
        
        # Parse date if available
        if metadata['route_date']:
            try:
                dt = datetime.strptime(metadata['route_date'], '%Y%m%d%H%M%S')
                metadata['year'] = dt.year
                metadata['month'] = dt.month
                metadata['day'] = dt.day
                metadata['hour'] = dt.hour
                metadata['minute'] = dt.minute
            except:
                pass
        
        return metadata
    
    def load_sensor_file(self, filepath: Path, columns: List[str]) -> pd.DataFrame:
        """Load a sensor data file and assign column names"""
        try:
            # Load data with space delimiter
            data = np.genfromtxt(filepath, dtype=np.float64, delimiter=' ')
            
            # Handle single vs multiple columns
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
            
            # Create DataFrame with appropriate columns
            if data.shape[1] == len(columns):
                df = pd.DataFrame(data, columns=columns)
            else:
                # Use available columns or pad with generic names
                actual_cols = columns[:data.shape[1]]
                df = pd.DataFrame(data, columns=actual_cols)
            
            return df
        except Exception as e:
            print(f"  Warning: Could not load {filepath.name}: {e}")
            return pd.DataFrame()
    
    def process_route(self, route_path: Path, route_id: int) -> Dict[str, pd.DataFrame]:
        """
        Process a single route folder
        
        Returns:
            Dictionary with 'metadata' and sensor data DataFrames
        """
        print(f"\nProcessing route: {route_path.name}")
        
        # Extract metadata
        metadata = self.extract_route_metadata(route_path.name)
        metadata['route_id'] = route_id
        
        route_data = {'metadata': metadata}
        
        # Load each sensor file
        for filename, columns in self.file_columns.items():
            filepath = route_path / filename
            
            if filepath.exists():
                df = self.load_sensor_file(filepath, columns)
                
                if not df.empty:
                    # Add route_id to each dataframe
                    df['route_id'] = route_id
                    
                    # Store with key
                    key = filename.replace('.txt', '').lower()
                    route_data[key] = df
                    
                    print(f"  ✓ Loaded {filename}: {len(df)} records")
                else:
                    print(f"  ✗ Failed to load {filename}")
            else:
                print(f"  - {filename} not found")
        
        return route_data
    
    def find_all_routes(self) -> List[Tuple[Path, str]]:
        """
        Find all route folders across all drivers
        
        Returns:
            List of (route_path, driver_id) tuples
        """
        routes = []
        
        # Look for driver folders (D1, D2, D3, etc.)
        for driver_folder in sorted(self.base_path.glob('D*')):
            if driver_folder.is_dir():
                driver_id = driver_folder.name
                
                # Find all route folders within driver folder
                for route_folder in sorted(driver_folder.glob('*')):
                    if route_folder.is_dir():
                        routes.append((route_folder, driver_id))
        
        return routes
    
    def process_all_routes(self, sampling_rate_hz: float = 1.0) -> None:
        """
        Process all routes and create consolidated CSV files
        
        Args:
            sampling_rate_hz: Desired output sampling rate (1 Hz = 1 sample per second)
        """
        print("="*80)
        print("UAH-DriveSet Data Preprocessing")
        print("="*80)
        
        # Find all routes
        all_routes = self.find_all_routes()
        print(f"\nFound {len(all_routes)} routes across all drivers")
        
        if not all_routes:
            print("ERROR: No route folders found. Please check the base_path.")
            return
        
        # Initialize data collectors
        all_metadata = []
        all_sensor_data = {
            'raw_accelerometers': [],
            'raw_gps': [],
            'proc_lane_detection': [],
            'proc_vehicle_detection': [],
            'proc_openstreetmap_data': []
        }
        
        # Process each route
        for route_id, (route_path, driver_id) in enumerate(all_routes, start=1):
            route_data = self.process_route(route_path, route_id)
            
            # Collect metadata
            if 'metadata' in route_data:
                all_metadata.append(route_data['metadata'])
            
            # Collect sensor data
            for key in all_sensor_data.keys():
                if key in route_data:
                    df = route_data[key]
                    
                    # Resample if requested
                    if sampling_rate_hz and sampling_rate_hz != -1:
                        df = self.resample_data(df, sampling_rate_hz)
                    
                    all_sensor_data[key].append(df)
        
        # Save consolidated files
        print("\n" + "="*80)
        print("Saving consolidated CSV files...")
        print("="*80)
        
        self.save_consolidated_data(all_metadata, all_sensor_data)
        
        print("\n✓ Preprocessing complete!")
        print(f"Output files saved to: {self.output_path}")
    
    def resample_data(self, df: pd.DataFrame, target_hz: float) -> pd.DataFrame:
        """
        Resample data to target frequency
        
        Args:
            df: DataFrame with 'timestamp' column
            target_hz: Target sampling rate in Hz
        """
        if 'timestamp' not in df.columns or len(df) == 0:
            return df
        
        # Create time index
        df = df.copy()
        df['time_index'] = (df['timestamp'] / target_hz).astype(int)
        
        # Group by time index and take mean
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols.remove('time_index')
        
        if 'route_id' in numeric_cols:
            numeric_cols.remove('route_id')
        
        # Aggregate
        grouped = df.groupby('time_index')[numeric_cols].mean().reset_index(drop=True)
        
        # Add back route_id
        if 'route_id' in df.columns:
            grouped['route_id'] = df['route_id'].iloc[0]
        
        return grouped
    
    def save_consolidated_data(self, metadata_list: List[Dict], 
                               sensor_data: Dict[str, List[pd.DataFrame]]) -> None:
        """Save all processed data to CSV files"""
        
        # 1. Save master routes table
        if metadata_list:
            df_routes = pd.DataFrame(metadata_list)
            routes_file = self.output_path / 'master_routes.csv'
            df_routes.to_csv(routes_file, index=False)
            print(f"\n✓ Saved master_routes.csv ({len(df_routes)} routes)")
            print(f"  Columns: {list(df_routes.columns)}")
        
        # 2. Save each sensor data type
        for sensor_name, df_list in sensor_data.items():
            if df_list:
                # Concatenate all routes
                df_combined = pd.concat(df_list, ignore_index=True)
                
                # Save combined file
                sensor_file = self.output_path / f'{sensor_name}.csv'
                df_combined.to_csv(sensor_file, index=False)
                
                print(f"\n✓ Saved {sensor_name}.csv ({len(df_combined)} records)")
                print(f"  Columns: {list(df_combined.columns)}")
        
        # 3. Create a fully merged dataset (optional, can be large)
        print("\n" + "-"*80)
        create_merged = input("Create fully merged dataset? (y/n): ").lower().strip()
        
        if create_merged == 'y':
            self.create_merged_dataset(sensor_data)
    
    def create_merged_dataset(self, sensor_data: Dict[str, List[pd.DataFrame]]) -> None:
        """Create a single merged dataset with all sensor streams"""
        print("\nCreating merged dataset...")
        
        all_routes = []
        
        # Get unique route IDs
        route_ids = set()
        for df_list in sensor_data.values():
            for df in df_list:
                if 'route_id' in df.columns:
                    route_ids.update(df['route_id'].unique())
        
        # Merge data for each route separately
        for route_id in sorted(route_ids):
            route_dfs = []
            
            # Collect data for this route from each sensor
            for sensor_name, df_list in sensor_data.items():
                for df in df_list:
                    if 'route_id' in df.columns and route_id in df['route_id'].values:
                        route_df = df[df['route_id'] == route_id].copy()
                        
                        # Add sensor prefix to avoid column conflicts
                        cols_to_rename = [c for c in route_df.columns 
                                        if c not in ['timestamp', 'route_id']]
                        rename_dict = {c: f'{sensor_name}_{c}' for c in cols_to_rename}
                        route_df = route_df.rename(columns=rename_dict)
                        
                        route_dfs.append(route_df)
            
            if route_dfs:
                # Merge on timestamp
                merged = route_dfs[0]
                for df in route_dfs[1:]:
                    merged = pd.merge(merged, df, on=['timestamp', 'route_id'], 
                                    how='outer', suffixes=('', '_dup'))
                
                # Sort by timestamp
                merged = merged.sort_values('timestamp').reset_index(drop=True)
                all_routes.append(merged)
        
        if all_routes:
            df_merged = pd.concat(all_routes, ignore_index=True)
            merged_file = self.output_path / 'merged_all_sensors.csv'
            df_merged.to_csv(merged_file, index=False)
            print(f"✓ Saved merged_all_sensors.csv ({len(df_merged)} records)")
            print(f"  Warning: This file may be very large!")


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print(" UAH-DriveSet Preprocessing Tool")
    print("="*80)
    
    # Get input path
    print("\nPlease provide the path to your UAH-DRIVESET-v1 folder")
    print("(This folder should contain D1, D2, D3, etc. subdirectories)")
    default_path = "./UAH-DRIVESET-v1"
    base_path = input(f"\nBase path [{default_path}]: ").strip()
    
    if not base_path:
        base_path = default_path
    
    if not os.path.exists(base_path):
        print(f"\nERROR: Path does not exist: {base_path}")
        return
    
    # Get output path
    default_output = "./processed_data"
    output_path = input(f"Output path [{default_output}]: ").strip()
    
    if not output_path:
        output_path = default_output
    
    # Get sampling rate preference
    print("\n" + "-"*80)
    print("Sampling rate options:")
    print("  1. Keep original sampling rate (fastest, largest files)")
    print("  2. 10 Hz (10 samples per second)")
    print("  3. 1 Hz (1 sample per second - recommended for ML)")
    print("  4. 0.1 Hz (1 sample per 10 seconds - for overview)")
    
    sampling_choice = input("\nChoice [3]: ").strip()
    
    sampling_map = {
        '1': -1,      # Original
        '2': 10.0,
        '3': 1.0,
        '4': 0.1
    }
    
    sampling_rate = sampling_map.get(sampling_choice, 1.0)
    
    # Process data
    print("\n" + "="*80)
    print("Starting preprocessing...")
    print("="*80)
    
    preprocessor = DriveSetPreprocessor(base_path, output_path)
    preprocessor.process_all_routes(sampling_rate_hz=sampling_rate)
    
    print("\n" + "="*80)
    print("Done!")
    print("="*80)


if __name__ == '__main__':
    main()
