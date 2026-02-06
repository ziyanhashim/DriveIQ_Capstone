#!/usr/bin/env python3
"""
Phase 3: Feature Attribution & Actionable Feedback
Identify which features cause anomalies and generate driver feedback
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


class FeatureAttributor:
    """
    Identify contributing features and generate feedback
    """
    
    def __init__(self, normal_profile):
        """
        Initialize with normal profile from Phase 2
        
        Args:
            normal_profile: NormalProfileDetector object or dict with normal stats
        """
        self.normal_profile = normal_profile
        
        # Define feature-to-behavior mappings
        self.feature_behaviors = {
            # Speed-related
            'speed_kmh_std': 'Erratic speed changes',
            'speed_kmh_mean': 'Average speed',
            'speed_change_rate_std': 'Inconsistent acceleration/deceleration',
            
            # Acceleration (braking/acceleration)
            'acc_x_kf_min': 'Harsh braking',
            'acc_x_kf_max': 'Harsh acceleration',
            'acc_x_kf_std': 'Inconsistent longitudinal control',
            'braking_intensity_mean': 'Heavy braking',
            'harsh_braking_count': 'Harsh braking events',
            'harsh_accel_count': 'Harsh acceleration events',
            
            # Lane position (weaving)
            'x_lane_std': 'Lane weaving',
            'x_lane_mean': 'Lane position deviation',
            'lane_crossing_freq': 'Frequent lane crossings',
            'lane_drift_rate_std': 'Erratic lane drift',
            
            # Steering
            'phi_std': 'Excessive steering corrections',
            'phi_mean': 'Steering angle',
            'steering_rate_std': 'Erratic steering',
            'steering_entropy': 'Unpredictable steering pattern',
            
            # Lateral forces
            'acc_y_kf_std': 'Inconsistent lateral control',
            'lateral_force_mean': 'High cornering forces',
            
            # Following distance
            'dist_front_mean': 'Following distance',
            'ttc_front_mean': 'Time to collision (reaction time)',
            'avg_reaction_time': 'Reaction time to vehicle ahead',
        }
        
        # Severity thresholds (in standard deviations from normal)
        self.severity_thresholds = {
            'low': 1.0,      # 1σ
            'medium': 2.0,   # 2σ
            'high': 3.0      # 3σ
        }
    
    def load_anomaly_data(self, csv_path):
        """Load windows with anomaly scores from Phase 2"""
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} windows with anomaly scores")
        return df
    
    def calculate_feature_contributions(self, window_row):
        """
        Calculate how much each feature contributes to anomaly
        
        Args:
            window_row: Single window (Series or dict) with features
            
        Returns:
            DataFrame with feature contributions
        """
        if isinstance(window_row, pd.Series):
            window_row = window_row.to_dict()
        
        contributions = []
        
        # Get normal mean and std
        mean = self.normal_profile['mean']
        std = self.normal_profile['std']
        
        for feature, value in window_row.items():
            # Skip non-feature columns
            if not (feature.endswith('_mean') or feature.endswith('_std') 
                   or feature.endswith('_min') or feature.endswith('_max')):
                continue
            
            if feature not in mean.index:
                continue
            
            # Calculate z-score (standard deviations from normal)
            normal_mean = mean[feature]
            normal_std = std[feature]
            
            if normal_std == 0:
                z_score = 0
            else:
                z_score = abs(value - normal_mean) / normal_std
            
            # Determine severity
            if z_score > self.severity_thresholds['high']:
                severity = 'high'
            elif z_score > self.severity_thresholds['medium']:
                severity = 'medium'
            elif z_score > self.severity_thresholds['low']:
                severity = 'low'
            else:
                severity = 'normal'
            
            # Get behavior description
            behavior = self.feature_behaviors.get(feature, 'Unknown behavior')
            
            contributions.append({
                'feature': feature,
                'value': value,
                'normal_mean': normal_mean,
                'normal_std': normal_std,
                'z_score': z_score,
                'severity': severity,
                'behavior': behavior
            })
        
        # Sort by z-score
        contrib_df = pd.DataFrame(contributions)
        contrib_df = contrib_df.sort_values('z_score', ascending=False)
        
        return contrib_df
    
    def generate_feedback(self, window_row, top_n=5):
        """
        Generate human-readable feedback for a window
        
        Args:
            window_row: Single window data
            top_n: Number of top contributing features to include
            
        Returns:
            Dict with feedback
        """
        # Get contributions
        contrib = self.calculate_feature_contributions(window_row)
        
        # Filter significant contributions (z_score > 1)
        significant = contrib[contrib['z_score'] > 1.0].head(top_n)
        
        if len(significant) == 0:
            return {
                'severity': 'normal',
                'message': '✓ Driving behavior is within normal range.',
                'details': []
            }
        
        # Determine overall severity
        max_z = significant['z_score'].max()
        if max_z > self.severity_thresholds['high']:
            overall_severity = 'high'
            emoji = '🔴'
        elif max_z > self.severity_thresholds['medium']:
            overall_severity = 'medium'
            emoji = '🟡'
        else:
            overall_severity = 'low'
            emoji = '🟢'
        
        # Build feedback message
        message_parts = []
        details = []
        
        for idx, row in significant.iterrows():
            behavior = row['behavior']
            z_score = row['z_score']
            feature = row['feature']
            
            # Create specific feedback
            if 'braking' in feature.lower():
                feedback = f"⚠️ Detected harsh braking (severity: {z_score:.1f}σ). Maintain gradual deceleration."
            elif 'lane' in feature.lower() and 'std' in feature:
                feedback = f"⚠️ Excessive lane weaving (severity: {z_score:.1f}σ). Maintain steady lane position."
            elif 'speed' in feature.lower() and 'std' in feature:
                feedback = f"⚠️ Erratic speed changes (severity: {z_score:.1f}σ). Maintain consistent speed."
            elif 'steering' in feature.lower() and 'entropy' in feature:
                feedback = f"⚠️ Unpredictable steering (severity: {z_score:.1f}σ). Smooth steering inputs."
            elif 'ttc' in feature.lower() or 'reaction' in feature.lower():
                feedback = f"⚠️ Delayed reaction time (severity: {z_score:.1f}σ). Increase following distance."
            else:
                feedback = f"⚠️ {behavior} (severity: {z_score:.1f}σ)"
            
            message_parts.append(feedback)
            details.append({
                'feature': feature,
                'behavior': behavior,
                'z_score': z_score,
                'severity': row['severity']
            })
        
        # Combine messages
        full_message = f"{emoji} Abnormal driving detected\n\n" + "\n".join(message_parts)
        
        # Add recommendations
        recommendations = self._generate_recommendations(significant)
        if recommendations:
            full_message += f"\n\nRecommendations:\n" + "\n".join(recommendations)
        
        return {
            'severity': overall_severity,
            'message': full_message,
            'details': details,
            'top_features': significant['feature'].tolist()
        }
    
    def _generate_recommendations(self, contributions):
        """Generate specific recommendations based on contributions"""
        recommendations = []
        
        features = contributions['feature'].tolist()
        
        # Braking-related
        if any('braking' in f.lower() or 'acc_x' in f for f in features):
            recommendations.append("• Anticipate stops earlier and brake more gradually")
        
        # Lane-related
        if any('lane' in f.lower() for f in features):
            recommendations.append("• Focus on maintaining steady lane position")
            recommendations.append("• Reduce distractions and stay centered in lane")
        
        # Speed-related
        if any('speed' in f.lower() and 'std' in f for f in features):
            recommendations.append("• Maintain more consistent speed using cruise control")
        
        # Steering-related
        if any('steering' in f.lower() or 'phi' in f for f in features):
            recommendations.append("• Make smoother, more deliberate steering inputs")
        
        # Following distance
        if any('ttc' in f.lower() or 'dist_front' in f.lower() for f in features):
            recommendations.append("• Increase following distance (3-second rule)")
            recommendations.append("• Stay more alert to traffic ahead")
        
        # If many issues, suggest break
        if len(contributions) >= 4:
            recommendations.append("• Consider taking a break - multiple abnormal behaviors detected")
        
        return recommendations
    
    def analyze_all_windows(self, df, min_anomaly_score=50):
        """
        Analyze all windows above anomaly threshold
        
        Args:
            df: DataFrame with windows and anomaly scores
            min_anomaly_score: Minimum score to analyze (0-100)
            
        Returns:
            DataFrame with feedback for each window
        """
        print(f"\nAnalyzing windows with anomaly score > {min_anomaly_score}")
        
        # Filter high-anomaly windows
        anomalous = df[df['anomaly_score_normalized'] > min_anomaly_score].copy()
        print(f"Found {len(anomalous)} anomalous windows")
        
        # Generate feedback for each
        feedbacks = []
        
        for idx, row in anomalous.iterrows():
            feedback = self.generate_feedback(row)
            
            feedbacks.append({
                'window_id': idx,
                'anomaly_score': row['anomaly_score_normalized'],
                'behavior': row.get('behavior', row.get('label', 'Unknown')),
                'severity': feedback['severity'],
                'top_feature': feedback['top_features'][0] if feedback['top_features'] else None,
                'message': feedback['message']
            })
        
        feedback_df = pd.DataFrame(feedbacks)
        
        return feedback_df
    
    def visualize_feature_contributions(self, window_row, output_path=None):
        """Create visualization of top contributing features"""
        contrib = self.calculate_feature_contributions(window_row)
        
        # Top 10 features
        top_contrib = contrib.head(10)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = ['red' if s == 'high' else 'orange' if s == 'medium' else 'yellow' 
                 for s in top_contrib['severity']]
        
        y_pos = range(len(top_contrib))
        ax.barh(y_pos, top_contrib['z_score'], color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_contrib['behavior'])
        ax.set_xlabel('Deviation from Normal (σ)')
        ax.set_title('Top Contributing Features to Anomaly')
        ax.axvline(x=1, color='green', linestyle='--', label='1σ (low)')
        ax.axvline(x=2, color='orange', linestyle='--', label='2σ (medium)')
        ax.axvline(x=3, color='red', linestyle='--', label='3σ (high)')
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def main():
    """Main execution"""
    print("\n" + "="*80)
    print(" PHASE 3: FEATURE ATTRIBUTION & FEEDBACK GENERATION")
    print(" Identify why windows are anomalous and generate actionable feedback")
    print("="*80)
    
    # Load Phase 2 results
    import joblib
    
    profile_path = input("\nPhase 2 normal profile (pkl): ").strip() or \
                   "models/saved_models/phase2_normal_profile.pkl"
    
    anomaly_data_path = input("Phase 2 anomaly scores (CSV): ").strip() or \
                        "results/phase2_results/windows_with_anomaly_scores.csv"
    
    # Load normal profile
    detector = joblib.load(profile_path)
    
    # Initialize attributor
    attributor = FeatureAttributor(detector.normal_profile)
    
    # Load anomaly data
    df = attributor.load_anomaly_data(anomaly_data_path)
    
    # Interactive mode: analyze specific window
    print("\n" + "="*60)
    print("Interactive Analysis")
    print("="*60)
    
    while True:
        choice = input("\n1. Analyze specific window\n2. Analyze all high-anomaly windows\n3. Exit\nChoice: ").strip()
        
        if choice == '1':
            window_idx = int(input("Enter window index: "))
            
            if window_idx not in df.index:
                print(f"Window {window_idx} not found")
                continue
            
            window = df.loc[window_idx]
            
            # Generate feedback
            feedback = attributor.generate_feedback(window)
            
            print("\n" + "="*80)
            print(f"FEEDBACK FOR WINDOW {window_idx}")
            print("="*80)
            print(f"\nAnomaly Score: {window['anomaly_score_normalized']:.1f}/100")
            print(f"Behavior: {window.get('behavior', window.get('label', 'Unknown'))}")
            print("\n" + feedback['message'])
            
            # Visualize
            output_dir = Path("results/phase3_results")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            attributor.visualize_feature_contributions(
                window,
                output_dir / f'window_{window_idx}_contributions.png'
            )
            
            print(f"\n✓ Visualization saved to results/phase3_results/window_{window_idx}_contributions.png")
        
        elif choice == '2':
            threshold = float(input("Minimum anomaly score [70]: ") or "70")
            
            feedback_df = attributor.analyze_all_windows(df, min_anomaly_score=threshold)
            
            # Save results
            output_dir = Path("results/phase3_results")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            feedback_df.to_csv(output_dir / 'all_window_feedback.csv', index=False)
            
            print(f"\n✓ Feedback generated for {len(feedback_df)} windows")
            print(f"✓ Saved to results/phase3_results/all_window_feedback.csv")
            
            # Summary
            print("\nFeedback Summary:")
            print(feedback_df.groupby('severity').size())
            
        elif choice == '3':
            break
    
    print("\n" + "="*80)
    print("✓ Phase 3 Complete!")
    print("="*80)
    print("\nAll three phases complete:")
    print("✓ Phase 1: Classified sessions")
    print("✓ Phase 2: Created normal profile and detected anomalies")
    print("✓ Phase 3: Generated actionable feedback")


if __name__ == '__main__':
    main()
