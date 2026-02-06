#!/usr/bin/env python3
"""
Phase 1: Session-Level Classification
Classify entire driving sessions as Normal, Drowsy, or Aggressive
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


class SessionClassifier:
    """Classify entire driving sessions by behavior type"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.feature_columns = None
        
    def load_window_data(self, motor_path, secondary_path):
        """Load windowed features"""
        print("Loading window features...")
        
        motor = pd.read_csv(motor_path)
        secondary = pd.read_csv(secondary_path)
        
        df = pd.concat([motor, secondary], ignore_index=True)
        print(f"Loaded {len(df)} windows")
        
        return df
    
    def aggregate_to_session_level(self, df):
        """
        Aggregate windows to session level
        
        Groups by route/behavior and creates summary statistics
        """
        print("\nAggregating windows to session level...")
        
        # Determine grouping columns
        group_cols = []
        if 'behavior' in df.columns:
            group_cols.append('behavior')
        if 'road_type' in df.columns:
            group_cols.append('road_type')
        
        # If no clear session ID, create one based on consecutive windows
        if 'route_id' not in df.columns:
            # Create pseudo-session based on behavior + road_type combination
            df['session_id'] = df.groupby(group_cols).ngroup()
        else:
            df['session_id'] = df['route_id']
        
        group_cols.append('session_id')
        
        # Get feature columns (exclude metadata)
        feature_cols = [col for col in df.columns 
                       if col.endswith('_mean') or col.endswith('_std') 
                       or col.endswith('_min') or col.endswith('_max')]
        
        # Aggregate features across session
        agg_dict = {col: ['mean', 'std', 'min', 'max'] for col in feature_cols}
        
        # Add metadata (take first value)
        if 'behavior' in df.columns:
            agg_dict['behavior'] = 'first'
        if 'road_type' in df.columns:
            agg_dict['road_type'] = 'first'
        if 'label' in df.columns:
            agg_dict['label'] = 'first'
        
        session_df = df.groupby('session_id').agg(agg_dict).reset_index()
        
        # Flatten column names
        session_df.columns = ['_'.join(col).strip('_') for col in session_df.columns.values]
        
        print(f"Created {len(session_df)} sessions")
        
        return session_df
    
    def prepare_data(self, df):
        """Prepare features and labels"""
        
        # Get behavior label
        label_col = None
        for col in ['behavior_first', 'behavior', 'label_first', 'label']:
            if col in df.columns:
                label_col = col
                break
        
        if label_col is None:
            raise ValueError("No behavior/label column found")
        
        # Encode labels if needed
        if df[label_col].dtype == 'object':
            label_map = {'Normal': 0, 'Drowsy': 1, 'Aggressive': 2}
            y = df[label_col].map(label_map)
        else:
            y = df[label_col]
        
        # Get features
        exclude_cols = ['session_id', 'behavior', 'behavior_first', 
                       'label', 'label_first', 'road_type', 'road_type_first']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        X = df[feature_cols]
        
        # Handle missing values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        self.feature_columns = feature_cols
        
        print(f"\nFeatures: {len(feature_cols)}")
        print(f"Classes: {y.value_counts().to_dict()}")
        
        return X, y
    
    def train(self, X_train, y_train):
        """Train session classifier"""
        print("\nTraining session classifier...")
        
        # Standardize
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train Random Forest
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train,
            cv=cv, scoring='accuracy', n_jobs=-1
        )
        
        print(f"  CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        
        # Train on full training set
        self.model.fit(X_train_scaled, y_train)
        
        return self.model
    
    def evaluate(self, X_test, y_test):
        """Evaluate classifier"""
        print("\n" + "="*80)
        print("SESSION CLASSIFICATION RESULTS")
        print("="*80)
        
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        
        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nAccuracy: {accuracy:.3f}")
        
        # Classification report
        target_names = ['Normal', 'Drowsy', 'Aggressive']
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print("                Predicted")
        print("              Normal  Drowsy  Aggressive")
        print(f"Actual Normal    {cm[0,0]:3d}     {cm[0,1]:3d}      {cm[0,2]:3d}")
        print(f"       Drowsy    {cm[1,0]:3d}     {cm[1,1]:3d}      {cm[1,2]:3d}")
        print(f"       Aggr.     {cm[2,0]:3d}     {cm[2,1]:3d}      {cm[2,2]:3d}")
        
        return accuracy, cm
    
    def predict_session(self, session_features):
        """
        Predict behavior for a new session
        
        Args:
            session_features: Dict or DataFrame with session-level features
            
        Returns:
            Predicted class (0=Normal, 1=Drowsy, 2=Aggressive)
        """
        if isinstance(session_features, dict):
            X = pd.DataFrame([session_features])[self.feature_columns]
        else:
            X = session_features[self.feature_columns]
        
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_scaled = self.scaler.transform(X)
        
        return self.model.predict(X_scaled)


def main():
    """Main execution"""
    print("\n" + "="*80)
    print(" PHASE 1: SESSION-LEVEL CLASSIFICATION")
    print(" Classify entire driving sessions as Normal/Drowsy/Aggressive")
    print("="*80)
    
    # Load data
    motor_path = input("\nMotor window features CSV: ").strip()
    secondary_path = input("Secondary window features CSV: ").strip()
    
    classifier = SessionClassifier()
    
    # Load windows
    windows_df = classifier.load_window_data(motor_path, secondary_path)
    
    # Aggregate to session level
    session_df = classifier.aggregate_to_session_level(windows_df)
    
    # Prepare data
    X, y = classifier.prepare_data(session_df)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining sessions: {len(X_train)}")
    print(f"Test sessions: {len(X_test)}")
    
    # Train
    classifier.train(X_train, y_train)
    
    # Evaluate
    accuracy, cm = classifier.evaluate(X_test, y_test)
    
    # Save model (optional)
    save = input("\nSave model? (y/n): ").lower().strip()
    if save == 'y':
        import joblib
        Path("models/saved_models").mkdir(parents=True, exist_ok=True)
        joblib.dump(classifier, "models/saved_models/phase1_session_classifier.pkl")
        print("✓ Saved to models/saved_models/phase1_session_classifier.pkl")
    
    print("\n" + "="*80)
    print("✓ Phase 1 Complete!")
    print("="*80)
    print("\nNext: Run Phase 2 to create Normal profile and detect anomalies")


if __name__ == '__main__':
    main()
