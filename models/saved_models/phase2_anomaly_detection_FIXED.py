#!/usr/bin/env python3
"""
Phase 2: Normal Profile & Anomaly Detection (FIXED)
Create a baseline "normal driving" profile and measure deviation from it
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope


class NormalProfileDetector:
    """
    Create a normal driving profile and detect anomalies
    """
    
    def __init__(self, method='mahalanobis'):
        """
        Initialize detector
        
        Args:
            method: 'mahalanobis', 'isolation_forest', or 'elliptic_envelope'
        """
        self.method = method
        self.normal_profile = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def load_data(self, motor_path, secondary_path):
        """Load window features"""
        print("Loading window features...")
        
        motor = pd.read_csv(motor_path)
        secondary = pd.read_csv(secondary_path)
        
        df = pd.concat([motor, secondary], ignore_index=True)
        print(f"Loaded {len(df)} windows")
        
        return df
    
    def create_normal_profile(self, df):
        """
        Create baseline profile from Normal driving windows
        
        Args:
            df: DataFrame with window features and labels
            
        Returns:
            Normal profile statistics
        """
        print("\n" + "="*60)
        print("Creating Normal Driving Profile")
        print("="*60)
        
        # Filter only Normal windows (label == 0)
        if 'label' not in df.columns:
            raise ValueError("No 'label' column found")
        
        normal_df = df[df['label'] == 0].copy()
        
        print(f"\nNormal windows: {len(normal_df)}")
        print(f"Drowsy windows: {len(df[df['label'] == 1])}")
        print(f"Aggressive windows: {len(df[df['label'] == 2])}")
        
        # Select features (only _mean features in your data)
        feature_cols = [col for col in df.columns 
                       if col not in ['label'] and df[col].dtype in ['float64', 'int64']]
        
        self.feature_columns = feature_cols
        print(f"\nUsing {len(feature_cols)} features")
        
        # Extract normal features
        X_normal = normal_df[feature_cols].copy()
        
        # Handle missing values
        X_normal = X_normal.replace([np.inf, -np.inf], np.nan)
        X_normal = X_normal.fillna(X_normal.median())
        
        # Create profile based on method
        if self.method == 'mahalanobis':
            # Calculate mean and covariance
            self.normal_profile = {
                'mean': X_normal.mean(),
                'cov': X_normal.cov(),
                'cov_inv': np.linalg.pinv(X_normal.cov()),  # Pseudo-inverse for stability
                'std': X_normal.std()
            }
            
        elif self.method == 'isolation_forest':
            # Train Isolation Forest on normal data
            self.scaler.fit(X_normal)
            X_scaled = self.scaler.transform(X_normal)
            
            self.normal_profile = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_jobs=-1
            )
            self.normal_profile.fit(X_scaled)
            
        elif self.method == 'elliptic_envelope':
            # Robust covariance estimation
            self.scaler.fit(X_normal)
            X_scaled = self.scaler.transform(X_normal)
            
            self.normal_profile = EllipticEnvelope(
                contamination=0.1,
                random_state=42
            )
            self.normal_profile.fit(X_scaled)
        
        print(f"\n✓ Normal profile created using {self.method}")
        
        # Print summary statistics
        if self.method == 'mahalanobis':
            print("\nNormal Profile Summary (first 5 features):")
            print("-" * 60)
            for i, col in enumerate(feature_cols[:5]):
                mean = self.normal_profile['mean'][col]
                std = self.normal_profile['std'][col]
                print(f"  {col[:40]:40s}: {mean:8.3f} ± {std:6.3f}")
            print("  ...")
        
        return self.normal_profile
    
    def calculate_anomaly_scores(self, df):
        """
        Calculate anomaly score for each window
        
        Returns:
            DataFrame with anomaly scores added
        """
        print("\n" + "="*60)
        print("Calculating Anomaly Scores")
        print("="*60)
        
        df = df.copy()
        
        # Extract features
        X = df[self.feature_columns].copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        if self.method == 'mahalanobis':
            # Calculate Mahalanobis distance for each window
            mean = self.normal_profile['mean'].values
            cov_inv = self.normal_profile['cov_inv']
            
            scores = []
            for idx, row in X.iterrows():
                diff = row.values - mean
                distance = np.sqrt(diff.T @ cov_inv @ diff)
                scores.append(distance)
            
            df['anomaly_score'] = scores
            
            # Normalize to 0-100 scale
            df['anomaly_score_normalized'] = (
                (df['anomaly_score'] - df['anomaly_score'].min()) / 
                (df['anomaly_score'].max() - df['anomaly_score'].min()) * 100
            )
            
        elif self.method in ['isolation_forest', 'elliptic_envelope']:
            X_scaled = self.scaler.transform(X)
            
            # Get anomaly scores (-1 = anomaly, 1 = normal)
            predictions = self.normal_profile.predict(X_scaled)
            decision_scores = self.normal_profile.decision_function(X_scaled)
            
            df['anomaly_prediction'] = predictions
            df['anomaly_score'] = -decision_scores  # Invert so higher = more anomalous
            
            # Normalize to 0-100
            df['anomaly_score_normalized'] = (
                (df['anomaly_score'] - df['anomaly_score'].min()) / 
                (df['anomaly_score'].max() - df['anomaly_score'].min()) * 100
            )
        
        print(f"✓ Calculated anomaly scores for {len(df)} windows")
        
        return df
    
    def analyze_by_behavior(self, df):
        """Analyze anomaly scores by behavior type"""
        print("\n" + "="*60)
        print("Anomaly Score Analysis by Behavior")
        print("="*60)
        
        # Map labels to behavior names
        behavior_map = {0: 'Normal', 1: 'Drowsy', 2: 'Aggressive'}
        df['behavior_name'] = df['label'].map(behavior_map)
        
        # Calculate statistics by behavior
        results = df.groupby('behavior_name')['anomaly_score_normalized'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(2)
        
        print(results)
        
        # Percentage of high anomaly scores by behavior
        print("\n% of Windows with High Anomaly (score > 70):")
        high_anomaly = df[df['anomaly_score_normalized'] > 70]
        if len(high_anomaly) > 0:
            pct = high_anomaly.groupby('behavior_name').size() / df.groupby('behavior_name').size() * 100
            print(pct.round(1))
        else:
            print("  No windows with score > 70")
        
        return results
    
    def visualize_anomalies(self, df, output_dir='results/phase2_results'):
        """Create visualizations"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Map labels to behavior names
        behavior_map = {0: 'Normal', 1: 'Drowsy', 2: 'Aggressive'}
        df['behavior_name'] = df['label'].map(behavior_map)
        
        # 1. Distribution of anomaly scores by behavior
        plt.figure(figsize=(12, 6))
        
        for behavior in df['behavior_name'].unique():
            data = df[df['behavior_name'] == behavior]['anomaly_score_normalized']
            plt.hist(data, bins=30, alpha=0.5, label=behavior)
        
        plt.xlabel('Anomaly Score (0-100)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Anomaly Scores by Behavior')
        plt.legend()
        plt.axvline(x=70, color='r', linestyle='--', label='Threshold (70)')
        plt.tight_layout()
        plt.savefig(output_dir / 'anomaly_distribution.png', dpi=150)
        plt.close()
        
        # 2. Box plot
        plt.figure(figsize=(10, 6))
        df.boxplot(column='anomaly_score_normalized', by='behavior_name')
        plt.xlabel('Behavior')
        plt.ylabel('Anomaly Score')
        plt.title('Anomaly Scores by Behavior Type')
        plt.suptitle('')  # Remove default title
        plt.tight_layout()
        plt.savefig(output_dir / 'anomaly_boxplot.png', dpi=150)
        plt.close()
        
        print(f"\n✓ Visualizations saved to {output_dir}")
    
    def get_top_anomalous_windows(self, df, n=10):
        """Get most anomalous windows"""
        top_anomalies = df.nlargest(n, 'anomaly_score_normalized')
        
        print(f"\n{n} Most Anomalous Windows:")
        print("-" * 80)
        
        behavior_map = {0: 'Normal', 1: 'Drowsy', 2: 'Aggressive'}
        
        for idx, row in top_anomalies.iterrows():
            label = row['label']
            behavior_name = behavior_map.get(label, f'Unknown({label})')
            score = row['anomaly_score_normalized']
            print(f"  Window {idx}: {behavior_name:12s} | Anomaly Score: {score:.1f}")
        
        return top_anomalies


def main():
    """Main execution"""
    print("\n" + "="*80)
    print(" PHASE 2: NORMAL PROFILE & ANOMALY DETECTION")
    print(" Create baseline and measure deviation from normal driving")
    print("="*80)
    
    # Configuration
    print("\nSelect anomaly detection method:")
    print("1. Mahalanobis Distance (recommended)")
    print("2. Isolation Forest")
    print("3. Elliptic Envelope")
    
    method_choice = input("Choice [1]: ").strip() or "1"
    
    method_map = {
        '1': 'mahalanobis',
        '2': 'isolation_forest',
        '3': 'elliptic_envelope'
    }
    
    method = method_map.get(method_choice, 'mahalanobis')
    
    # Load data
    motor_path = input("\nMotor window features CSV: ").strip()
    secondary_path = input("Secondary window features CSV: ").strip()
    
    # Check files exist
    if not Path(motor_path).exists():
        print(f"ERROR: File not found: {motor_path}")
        return
    if not Path(secondary_path).exists():
        print(f"ERROR: File not found: {secondary_path}")
        return
    
    # Initialize detector
    detector = NormalProfileDetector(method=method)
    
    # Load data
    df = detector.load_data(motor_path, secondary_path)
    
    # Create normal profile
    detector.create_normal_profile(df)
    
    # Calculate anomaly scores
    df_with_scores = detector.calculate_anomaly_scores(df)
    
    # Analyze by behavior
    detector.analyze_by_behavior(df_with_scores)
    
    # Get top anomalies
    detector.get_top_anomalous_windows(df_with_scores, n=10)
    
    # Visualize
    detector.visualize_anomalies(df_with_scores)
    
    # Save results
    output_dir = Path("results/phase2_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df_with_scores.to_csv(output_dir / 'windows_with_anomaly_scores.csv', index=False)
    print(f"\n✓ Results saved to {output_dir}")
    
    # Save detector
    import joblib
    Path("models/saved_models").mkdir(parents=True, exist_ok=True)
    joblib.dump(detector, "models/saved_models/phase2_normal_profile.pkl")
    print("✓ Normal profile saved to models/saved_models/phase2_normal_profile.pkl")
    
    print("\n" + "="*80)
    print("✓ Phase 2 Complete!")
    print("="*80)
    print("\nKey Findings:")
    print("- Normal windows should have LOW anomaly scores")
    print("- Drowsy/Aggressive windows should have HIGH anomaly scores")
    print("- Score > 70 typically indicates abnormal driving")
    print("\nNext: Run Phase 3 to identify which features cause the anomalies")


if __name__ == '__main__':
    main()
