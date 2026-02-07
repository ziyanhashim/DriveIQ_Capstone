#!/usr/bin/env python3
"""
Phase 3: Feature Attribution & Actionable Feedback (NO PICKLE VERSION)
Identify which specific features cause anomalies and provide recommendations
Works directly from the CSV file - no pickle dependencies!
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


class FeatureAttributor:
    """
    Identify contributing features and generate actionable feedback
    """
    
    def __init__(self):
        """Initialize with empty profile - will build from CSV"""
        self.normal_profile = None
        
        # Comprehensive feature-to-behavior mappings
        self.feature_behaviors = {
            # Speed features
            'speed_kmh_mean': ('Speed', 'Average driving speed'),
            'gps_speed_mean': ('GPS Speed', 'Speed from GPS'),
            'gps_speed_osm_mean': ('Map Speed', 'Speed from map data'),
            'speed_ratio_mean': ('Speed Consistency', 'Speed matching expected speed'),
            
            # Acceleration features (X = forward/backward)
            'acc_x_mean': ('Longitudinal Acceleration', 'Forward/backward acceleration'),
            'acc_x_kf_mean': ('Filtered Acceleration', 'Smoothed acceleration pattern'),
            
            # Lateral features (Y = left/right)
            'acc_y_mean': ('Lateral Acceleration', 'Side-to-side movement'),
            'acc_y_kf_mean': ('Filtered Lateral Acceleration', 'Smoothed lateral movement'),
            
            # Vertical features (Z = up/down)
            'acc_z_mean': ('Vertical Acceleration', 'Up/down movement over bumps'),
            'acc_z_kf_mean': ('Filtered Vertical Acceleration', 'Smoothed vertical movement'),
            
            # Lane position and control
            'x_lane_mean': ('Lane Position', 'Distance from lane center'),
            'phi_mean': ('Steering Angle', 'Steering wheel position'),
            'lane_state_mean': ('Lane State', 'Lane keeping status'),
            
            # Vehicle detection and following
            'dist_front_mean': ('Following Distance', 'Distance to vehicle ahead'),
            'ttc_front_mean': ('Time to Collision', 'Reaction time buffer'),
            'num_vehicles_mean': ('Traffic Density', 'Number of nearby vehicles'),
            
            # Road characteristics
            'road_width_mean': ('Road Width', 'Width of current road'),
            'num_lanes_mean': ('Number of Lanes', 'Lane count'),
            'max_speed_mean': ('Speed Limit', 'Posted speed limit'),
            
            # Orientation and direction
            'roll_mean': ('Vehicle Roll', 'Side-to-side tilt'),
            'pitch_mean': ('Vehicle Pitch', 'Front-to-back tilt'),
            'yaw_mean': ('Vehicle Yaw', 'Rotation/heading angle'),
            'course_mean': ('Heading', 'Direction of travel'),
            'difcourse_mean': ('Heading Change', 'Change in direction'),
            
            # GPS quality
            'horiz_acc_mean': ('GPS Horizontal Accuracy', 'GPS position accuracy'),
            'vert_acc_mean': ('GPS Vertical Accuracy', 'GPS altitude accuracy'),
            'hdop_mean': ('Horizontal Precision', 'GPS horizontal precision'),
            'vdop_mean': ('Vertical Precision', 'GPS vertical precision'),
            'pdop_mean': ('Position Precision', 'GPS position precision'),
        }
        
        # Specific recommendations for each feature type
        self.feature_recommendations = {
            # Speed features
            'speed_kmh_mean': 'Adjust speed to match road conditions and speed limit',
            'gps_speed_mean': 'Verify speedometer accuracy - GPS shows different speed',
            'gps_speed_osm_mean': 'Your speed differs from expected map speed - adjust accordingly',
            'speed_ratio_mean': 'Maintain consistent speed relative to speed limit',
            'speed_rel_mean': 'Match your speed to surrounding traffic flow',
            
            # Acceleration (braking/throttle)
            'acc_x_mean': 'Smooth out acceleration and braking - avoid sudden changes',
            'acc_x_kf_mean': 'Apply gradual throttle and brake inputs',
            
            # Lane keeping
            'x_lane_mean': 'Center yourself in the lane - avoid drifting',
            'phi_mean': 'Reduce excessive steering corrections - relax your grip',
            'lane_state_mean': 'Stay within lane boundaries - check mirrors frequently',
            
            # Lateral control (swerving)
            'acc_y_mean': 'Reduce sharp turns and lane changes - plan maneuvers ahead',
            'acc_y_kf_mean': 'Make smoother steering inputs to reduce lateral forces',
            
            # Vertical (bumps/road surface)
            'acc_z_mean': 'Reduce speed over bumps and uneven road surfaces',
            'acc_z_kf_mean': 'Drive more gently over road irregularities',
            
            # Following distance & traffic
            'dist_front_mean': 'Increase following distance - maintain 3-second rule',
            'ttc_front_mean': 'Allow more reaction time - increase gap to vehicle ahead',
            'num_vehicles_mean': 'Adjust driving for current traffic density - stay alert',
            
            # Road characteristics awareness
            'road_width_mean': 'Adjust lane positioning for current road width',
            'num_lanes_mean': 'Be aware of multi-lane traffic - check blind spots',
            'max_speed_mean': 'Observe posted speed limit - currently exceeding safe speed',
            
            # Vehicle stability
            'roll_mean': 'Reduce sharp cornering and lateral movements',
            'pitch_mean': 'Smooth acceleration and braking to reduce pitch',
            'yaw_mean': 'Maintain steady heading - avoid weaving',
            
            # Direction changes
            'course_mean': 'Verify navigation - heading may be incorrect',
            'difcourse_mean': 'Make gradual direction changes - avoid sudden turns',
            
            # GPS/Location quality (often indicates erratic driving pattern)
            'lat_mean': 'Position tracking shows erratic path - drive more predictably',
            'lon_mean': 'Location variance high - maintain steady course',
            'alt_mean': 'Elevation changes rapid - adjust speed for terrain',
            'lat_osm_mean': 'GPS position unstable - ensure smooth driving for better tracking',
            'lon_osm_mean': 'Route deviation detected - stay on intended path',
            'osm_delay_mean': 'GPS sync issues - drive more consistently for stable tracking',
            
            # GPS precision metrics (indirect indicators of driving behavior)
            'horiz_acc_mean': 'GPS horizontal drift high - erratic driving affects tracking',
            'vert_acc_mean': 'GPS vertical drift high - smooth out speed changes',
            'hdop_mean': 'Position precision low - drive more predictably',
            'vdop_mean': 'Altitude precision low - maintain steady speed',
            'pdop_mean': 'Overall GPS precision low - stabilize driving pattern',
            
            # GPS activity
            'active_mean': 'GPS signal fluctuating - drive more consistently',
        }
        
        # Severity thresholds
        self.severity_thresholds = {
            'low': 1.5,
            'medium': 2.5,
            'high': 4.0
        }
    
    def build_normal_profile_from_csv(self, df):
        """
        Build normal profile directly from CSV data
        
        Args:
            df: DataFrame with all windows including label column
        """
        print("\nBuilding Normal profile from CSV...")
        
        # Filter only Normal windows (label == 0)
        normal_df = df[df['label'] == 0].copy()
        
        print(f"Normal windows: {len(normal_df)}")
        print(f"Drowsy windows: {len(df[df['label'] == 1])}")
        print(f"Aggressive windows: {len(df[df['label'] == 2])}")
        
        # Get feature columns (exclude metadata)
        exclude_cols = ['label', 'behavior_name', 'anomaly_score', 
                       'anomaly_score_normalized', 'anomaly_prediction']
        
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols and df[col].dtype in ['float64', 'int64']]
        
        # Calculate mean and std from normal windows
        X_normal = normal_df[feature_cols]
        
        self.normal_profile = {
            'mean': X_normal.mean(),
            'std': X_normal.std(),
            'feature_columns': feature_cols
        }
        
        print(f"✓ Normal profile created with {len(feature_cols)} features")
        
        return self.normal_profile
    
    def calculate_feature_contributions(self, window_row):
        """Calculate how much each feature contributes to anomaly"""
        if isinstance(window_row, pd.Series):
            window_row = window_row.to_dict()
        
        contributions = []
        
        mean = self.normal_profile['mean']
        std = self.normal_profile['std']
        
        for feature, value in window_row.items():
            # Skip metadata
            if feature in ['label', 'behavior_name', 'anomaly_score', 
                          'anomaly_score_normalized', 'anomaly_prediction']:
                continue
            
            if not isinstance(value, (int, float, np.number)):
                continue
            
            if feature not in mean.index:
                continue
            
            # Calculate z-score
            normal_mean = mean[feature]
            normal_std = std[feature]
            
            if normal_std == 0 or pd.isna(normal_std):
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
            if feature in self.feature_behaviors:
                behavior_short, behavior_desc = self.feature_behaviors[feature]
            else:
                behavior_short = feature.replace('_mean', '').replace('_', ' ').title()
                behavior_desc = behavior_short
            
            contributions.append({
                'feature': feature,
                'behavior_short': behavior_short,
                'value': value,
                'normal_mean': normal_mean,
                'normal_std': normal_std,
                'z_score': z_score,
                'severity': severity
            })
        
        contrib_df = pd.DataFrame(contributions)
        contrib_df = contrib_df.sort_values('z_score', ascending=False)
        
        return contrib_df
    
    def generate_feedback(self, window_row, top_n=5):
        """Generate human-readable feedback"""
        contrib = self.calculate_feature_contributions(window_row)
        
        significant = contrib[contrib['z_score'] > 1.5].head(top_n)
        
        if len(significant) == 0:
            return {
                'severity': 'normal',
                'message': '✅ Driving behavior is within normal range.',
                'details': [],
                'top_features': []
            }
        
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
        
        print(f"\n{emoji} ABNORMAL DRIVING DETECTED")
        print("=" * 80)
        
        message_parts = []
        details = []
        
        for idx, row in significant.iterrows():
            feature = row['feature']
            behavior_short = row['behavior_short']
            z_score = row['z_score']
            severity = row['severity']
            
            recommendation = self.feature_recommendations.get(
                feature,
                'Monitor this behavior and adjust driving accordingly'
            )
            
            feedback_line = (
                f"⚠️  {behavior_short}: {z_score:.1f}σ deviation ({severity} severity)\n"
                f"   💡 Recommendation: {recommendation}"
            )
            
            message_parts.append(feedback_line)
            print(f"\n{feedback_line}")
            
            details.append({
                'feature': feature,
                'behavior': behavior_short,
                'z_score': round(z_score, 2),
                'severity': severity,
                'recommendation': recommendation
            })
        
        print("\n" + "=" * 80)
        
        full_message = f"{emoji} Abnormal driving detected\n\n" + "\n\n".join(message_parts)
        
        return {
            'severity': overall_severity,
            'message': full_message,
            'details': details,
            'top_features': significant['feature'].tolist()
        }
    
    def visualize_feature_contributions(self, window_row, output_path=None):
        """Create visualization"""
        contrib = self.calculate_feature_contributions(window_row)
        top_contrib = contrib.head(10)
        
        if len(top_contrib) == 0:
            print("No significant features to visualize")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = []
        for severity in top_contrib['severity']:
            if severity == 'high':
                colors.append('red')
            elif severity == 'medium':
                colors.append('orange')
            elif severity == 'low':
                colors.append('yellow')
            else:
                colors.append('lightgreen')
        
        y_pos = range(len(top_contrib))
        ax.barh(y_pos, top_contrib['z_score'], color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_contrib['behavior_short'])
        ax.set_xlabel('Deviation from Normal (σ - standard deviations)')
        ax.set_title('Top Contributing Features to Anomaly')
        ax.axvline(x=1.5, color='green', linestyle='--', linewidth=1, label='1.5σ (low)')
        ax.axvline(x=2.5, color='orange', linestyle='--', linewidth=1, label='2.5σ (medium)')
        ax.axvline(x=4.0, color='red', linestyle='--', linewidth=1, label='4.0σ (high)')
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✓ Chart saved to {output_path}")
        else:
            plt.show()


def main():
    """Main execution"""
    print("\n" + "="*80)
    print(" PHASE 3: FEATURE ATTRIBUTION & FEEDBACK GENERATION")
    print(" Identify which features cause anomalies and provide recommendations")
    print("="*80)
    
    # Only need the CSV file!
    anomaly_data_path = input("\nPhase 2 anomaly scores CSV [results/phase2_results/windows_with_anomaly_scores.csv]: ").strip()
    if not anomaly_data_path:
        anomaly_data_path = "results/phase2_results/windows_with_anomaly_scores.csv"
    
    if not Path(anomaly_data_path).exists():
        print(f"ERROR: File not found: {anomaly_data_path}")
        return
    
    # Load data
    print(f"\nLoading {anomaly_data_path}...")
    df = pd.read_csv(anomaly_data_path)
    print(f"Loaded {len(df)} windows")
    
    # Initialize attributor
    attributor = FeatureAttributor()
    
    # Build normal profile from CSV
    attributor.build_normal_profile_from_csv(df)
    
    # Interactive mode
    print("\n" + "="*80)
    print("Interactive Analysis")
    print("="*80)
    
    while True:
        print("\nOptions:")
        print("1. Analyze specific window by index")
        print("2. Analyze highest anomaly window")
        print("3. Analyze all high-anomaly windows (score > 70)")
        print("4. Exit")
        
        choice = input("\nChoice: ").strip()
        
        if choice == '1':
            try:
                window_idx = int(input("Enter window index (row number): "))
                
                if window_idx not in df.index:
                    print(f"❌ Window {window_idx} not found. Valid range: 0 to {len(df)-1}")
                    continue
                
                window = df.loc[window_idx]
                
                behavior_map = {0: 'Normal', 1: 'Drowsy', 2: 'Aggressive'}
                behavior = behavior_map.get(window['label'], 'Unknown')
                score = window['anomaly_score_normalized']
                
                print(f"\n{'='*80}")
                print(f"WINDOW {window_idx} ANALYSIS")
                print(f"{'='*80}")
                print(f"Behavior: {behavior}")
                print(f"Anomaly Score: {score:.1f}/100")
                
                feedback = attributor.generate_feedback(window)
                
                output_dir = Path("results/phase3_results")
                output_dir.mkdir(parents=True, exist_ok=True)
                
                attributor.visualize_feature_contributions(
                    window,
                    output_dir / f'window_{window_idx}_contributions.png'
                )
                
            except ValueError:
                print("❌ Please enter a valid number")
                
        elif choice == '2':
            max_idx = df['anomaly_score_normalized'].idxmax()
            window = df.loc[max_idx]
            
            behavior_map = {0: 'Normal', 1: 'Drowsy', 2: 'Aggressive'}
            behavior = behavior_map.get(window['label'], 'Unknown')
            score = window['anomaly_score_normalized']
            
            print(f"\n{'='*80}")
            print(f"HIGHEST ANOMALY WINDOW {max_idx} ANALYSIS")
            print(f"{'='*80}")
            print(f"Behavior: {behavior}")
            print(f"Anomaly Score: {score:.1f}/100")
            
            feedback = attributor.generate_feedback(window)
            
            output_dir = Path("results/phase3_results")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            attributor.visualize_feature_contributions(
                window,
                output_dir / f'window_{max_idx}_highest_anomaly.png'
            )
            
        elif choice == '3':
            threshold = 70
            high_anomaly = df[df['anomaly_score_normalized'] > threshold]
            
            print(f"\nFound {len(high_anomaly)} windows with anomaly score > {threshold}")
            
            if len(high_anomaly) == 0:
                print("No high-anomaly windows found")
                continue
            
            all_feedback = []
            
            for idx, row in high_anomaly.iterrows():
                feedback = attributor.generate_feedback(row)
                
                all_feedback.append({
                    'window_index': idx,
                    'anomaly_score': row['anomaly_score_normalized'],
                    'label': row['label'],
                    'severity': feedback['severity'],
                    'top_feature': feedback['top_features'][0] if feedback['top_features'] else None,
                    'num_issues': len(feedback['details'])
                })
            
            output_dir = Path("results/phase3_results")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            feedback_df = pd.DataFrame(all_feedback)
            feedback_df.to_csv(output_dir / 'all_high_anomaly_feedback.csv', index=False)
            
            print(f"\n✓ Feedback saved to {output_dir / 'all_high_anomaly_feedback.csv'}")
            
        elif choice == '4':
            break
        else:
            print("Invalid choice")
    
    print("\n" + "="*80)
    print("✓ Phase 3 Complete!")
    print("="*80)


if __name__ == '__main__':
    main()
