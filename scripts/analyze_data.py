#!/usr/bin/env python3
"""
Data Summary and Quality Check Script
Analyzes preprocessed UAH-DriveSet data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


class DataAnalyzer:
    """Analyzes preprocessed driver data"""
    
    def __init__(self, data_path: str = './processed_data'):
        self.data_path = Path(data_path)
        
    def load_data(self):
        """Load all processed CSV files"""
        self.routes = pd.read_csv(self.data_path / 'master_routes.csv')
        
        # Load sensor data if exists
        self.sensors = {}
        for csv_file in self.data_path.glob('*.csv'):
            if csv_file.name != 'master_routes.csv' and csv_file.name != 'merged_all_sensors.csv':
                sensor_name = csv_file.stem
                self.sensors[sensor_name] = pd.read_csv(csv_file)
        
        print(f"Loaded {len(self.sensors)} sensor data files")
    
    def print_summary(self):
        """Print comprehensive data summary"""
        print("\n" + "="*80)
        print(" DATA SUMMARY REPORT")
        print("="*80)
        
        # Routes summary
        print("\n1. ROUTES OVERVIEW")
        print("-"*80)
        print(f"Total routes: {len(self.routes)}")
        
        if 'driver_id' in self.routes.columns:
            print(f"\nRoutes per driver:")
            print(self.routes['driver_id'].value_counts().sort_index())
        
        if 'behavior_type' in self.routes.columns:
            print(f"\nRoutes by behavior type:")
            print(self.routes['behavior_type'].value_counts())
        
        if 'road_type' in self.routes.columns:
            print(f"\nRoutes by road type:")
            print(self.routes['road_type'].value_counts())
        
        # Sensor data summary
        print("\n\n2. SENSOR DATA OVERVIEW")
        print("-"*80)
        
        for sensor_name, df in self.sensors.items():
            print(f"\n{sensor_name.upper()}:")
            print(f"  Total records: {len(df):,}")
            print(f"  Columns: {df.shape[1]}")
            
            if 'route_id' in df.columns:
                print(f"  Routes covered: {df['route_id'].nunique()}")
            
            # Check for missing values
            missing = df.isnull().sum()
            if missing.any():
                print(f"  Missing values:")
                for col, count in missing[missing > 0].items():
                    pct = (count / len(df)) * 100
                    print(f"    - {col}: {count:,} ({pct:.1f}%)")
        
        # Data quality metrics
        print("\n\n3. DATA QUALITY METRICS")
        print("-"*80)
        
        if 'raw_gps' in self.sensors:
            gps = self.sensors['raw_gps']
            
            print("\nGPS Data Quality:")
            if 'speed_kmh' in gps.columns:
                print(f"  Speed range: {gps['speed_kmh'].min():.1f} - {gps['speed_kmh'].max():.1f} km/h")
                print(f"  Mean speed: {gps['speed_kmh'].mean():.1f} km/h")
            
            if 'horizontal_accuracy' in gps.columns:
                print(f"  Mean horizontal accuracy: {gps['horizontal_accuracy'].mean():.2f} m")
        
        if 'raw_accelerometers' in self.sensors:
            accel = self.sensors['raw_accelerometers']
            
            print("\nAccelerometer Data Quality:")
            for axis in ['accel_x_raw', 'accel_y_raw', 'accel_z_raw']:
                if axis in accel.columns:
                    print(f"  {axis} range: {accel[axis].min():.3f} - {accel[axis].max():.3f} G")
    
    def behavior_statistics(self):
        """Analyze behavior patterns per driver and route type"""
        print("\n\n4. BEHAVIOR ANALYSIS")
        print("-"*80)
        
        if 'behavior_type' not in self.routes.columns:
            print("Behavior type not found in metadata")
            return
        
        # Cross-tabulation
        if 'driver_id' in self.routes.columns:
            crosstab = pd.crosstab(
                self.routes['driver_id'], 
                self.routes['behavior_type'], 
                margins=True
            )
            print("\nRoutes by Driver and Behavior:")
            print(crosstab)
        
        # Statistics per behavior type
        if 'raw_gps' in self.sensors:
            print("\n\nSpeed Statistics by Behavior Type:")
            gps = self.sensors['raw_gps'].merge(
                self.routes[['route_id', 'behavior_type']], 
                on='route_id'
            )
            
            if 'speed_kmh' in gps.columns:
                stats = gps.groupby('behavior_type')['speed_kmh'].describe()
                print(stats)
    
    def export_summary_csv(self):
        """Export summary statistics to CSV"""
        output = []
        
        for sensor_name, df in self.sensors.items():
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            numeric_cols = [c for c in numeric_cols if c not in ['route_id', 'timestamp']]
            
            for col in numeric_cols:
                stats = {
                    'sensor': sensor_name,
                    'variable': col,
                    'count': df[col].count(),
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'q25': df[col].quantile(0.25),
                    'median': df[col].median(),
                    'q75': df[col].quantile(0.75),
                    'max': df[col].max(),
                    'missing': df[col].isnull().sum()
                }
                output.append(stats)
        
        df_summary = pd.DataFrame(output)
        summary_file = self.data_path / 'data_summary_statistics.csv'
        df_summary.to_csv(summary_file, index=False)
        print(f"\n✓ Saved summary statistics to: {summary_file}")
    
    def create_visualizations(self):
        """Create basic visualizations"""
        print("\n\n5. CREATING VISUALIZATIONS")
        print("-"*80)
        
        viz_path = self.data_path / 'visualizations'
        viz_path.mkdir(exist_ok=True)
        
        # 1. Routes per driver
        if 'driver_id' in self.routes.columns:
            plt.figure(figsize=(10, 6))
            self.routes['driver_id'].value_counts().sort_index().plot(kind='bar')
            plt.title('Number of Routes per Driver')
            plt.xlabel('Driver ID')
            plt.ylabel('Number of Routes')
            plt.tight_layout()
            plt.savefig(viz_path / 'routes_per_driver.png', dpi=150)
            plt.close()
            print("✓ Saved routes_per_driver.png")
        
        # 2. Behavior type distribution
        if 'behavior_type' in self.routes.columns:
            plt.figure(figsize=(10, 6))
            self.routes['behavior_type'].value_counts().plot(kind='bar', color='skyblue')
            plt.title('Distribution of Behavior Types')
            plt.xlabel('Behavior Type')
            plt.ylabel('Number of Routes')
            plt.tight_layout()
            plt.savefig(viz_path / 'behavior_distribution.png', dpi=150)
            plt.close()
            print("✓ Saved behavior_distribution.png")
        
        # 3. Speed distribution by behavior
        if 'raw_gps' in self.sensors and 'behavior_type' in self.routes.columns:
            gps = self.sensors['raw_gps'].merge(
                self.routes[['route_id', 'behavior_type']], 
                on='route_id'
            )
            
            if 'speed_kmh' in gps.columns:
                plt.figure(figsize=(12, 6))
                for behavior in gps['behavior_type'].unique():
                    data = gps[gps['behavior_type'] == behavior]['speed_kmh']
                    plt.hist(data, bins=50, alpha=0.5, label=behavior)
                
                plt.title('Speed Distribution by Behavior Type')
                plt.xlabel('Speed (km/h)')
                plt.ylabel('Frequency')
                plt.legend()
                plt.tight_layout()
                plt.savefig(viz_path / 'speed_by_behavior.png', dpi=150)
                plt.close()
                print("✓ Saved speed_by_behavior.png")
        
        print(f"\nVisualizations saved to: {viz_path}")


def main():
    """Main execution"""
    print("\n" + "="*80)
    print(" Data Summary and Analysis Tool")
    print("="*80)
    
    data_path = input("\nPath to processed data [./processed_data]: ").strip()
    if not data_path:
        data_path = './processed_data'
    
    if not Path(data_path).exists():
        print(f"ERROR: Path does not exist: {data_path}")
        return
    
    analyzer = DataAnalyzer(data_path)
    
    print("\nLoading data...")
    analyzer.load_data()
    
    # Print comprehensive summary
    analyzer.print_summary()
    analyzer.behavior_statistics()
    
    # Export summary
    analyzer.export_summary_csv()
    
    # Create visualizations
    create_viz = input("\n\nCreate visualizations? (y/n) [y]: ").lower().strip()
    if create_viz != 'n':
        try:
            analyzer.create_visualizations()
        except Exception as e:
            print(f"Warning: Could not create visualizations: {e}")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == '__main__':
    main()
