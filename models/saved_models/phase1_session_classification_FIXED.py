#!/usr/bin/env python3
"""
Phase 1: Session-Level Classification (FIXED)
Classify entire driving sessions as Normal, Drowsy, or Aggressive
"""

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


class SessionClassifier:
    """Classify driving behavior using window features"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.feature_columns = None
        
    def load_window_data(self, motor_path, secondary_path):
        """Load windowed features"""
        print("Loading window features...")
        
        motor = pd.read_csv(motor_path)
        secondary = pd.read_csv(secondary_path)
        
        # Add road_type column to distinguish sources
        motor['road_type'] = 'motor'
        secondary['road_type'] = 'secondary'
        
        df = pd.concat([motor, secondary], ignore_index=True)
        print(f"Loaded {len(df)} windows")
        print(f"  Motor: {len(motor)}")
        print(f"  Secondary: {len(secondary)}")
        
        return df
    
    def prepare_data(self, df):
        """Prepare features and labels"""
        
        # Get label column
        if 'label' not in df.columns:
            raise ValueError("No 'label' column found in data")
        
        y = df['label']
        
        # Get feature columns (all numeric columns except label and metadata)
        exclude_cols = ['label', 'road_type']
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols and df[col].dtype in ['float64', 'int64']]
        
        X = df[feature_cols]
        
        # Handle missing values and inf
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        self.feature_columns = feature_cols
        
        print(f"\nFeatures: {len(feature_cols)}")
        print(f"\nClass distribution:")
        label_counts = y.value_counts().sort_index()
        label_map = {0: 'Normal', 1: 'Drowsy', 2: 'Aggressive'}
        for label, count in label_counts.items():
            print(f"  {label_map.get(label, label)}: {count}")
        
        return X, y
    
    def train(self, X_train, y_train):
        """Train classifier"""
        print("\nTraining Random Forest classifier...")
        
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
        print("  Running 5-fold cross-validation...")
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
        print("CLASSIFICATION RESULTS")
        print("="*80)
        
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        
        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nTest Accuracy: {accuracy:.3f}")
        
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
        
        # Feature importance (top 10)
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        
        print("\nTop 10 Most Important Features:")
        for i, idx in enumerate(indices, 1):
            print(f"  {i}. {self.feature_columns[idx][:40]:40s}: {importances[idx]:.4f}")
        
        return accuracy, cm
    
    def predict(self, features):
        """
        Predict behavior for new window(s)
        
        Args:
            features: DataFrame or dict with features
            
        Returns:
            Predicted class (0=Normal, 1=Drowsy, 2=Aggressive)
        """
        if isinstance(features, dict):
            X = pd.DataFrame([features])[self.feature_columns]
        else:
            X = features[self.feature_columns]
        
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_scaled = self.scaler.transform(X)
        
        return self.model.predict(X_scaled)


def main():
    """Main execution"""
    print("\n" + "="*80)
    print(" PHASE 1: BEHAVIOR CLASSIFICATION")
    print(" Classify driving windows as Normal/Drowsy/Aggressive")
    print("="*80)
    
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
    
    classifier = SessionClassifier()
    
    # Load windows
    df = classifier.load_window_data(motor_path, secondary_path)
    
    # Prepare data
    X, y = classifier.prepare_data(df)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining windows: {len(X_train)}")
    print(f"Test windows: {len(X_test)}")
    
    # Train
    classifier.train(X_train, y_train)
    
    # Evaluate
    accuracy, cm = classifier.evaluate(X_test, y_test)
    
    # Save model
    save = input("\nSave model? (y/n): ").lower().strip()
    if save == 'y':
        import joblib
        Path("models/saved_models").mkdir(parents=True, exist_ok=True)
        joblib.dump(classifier, "models/saved_models/phase1_classifier.pkl")
        print("✓ Saved to models/saved_models/phase1_classifier.pkl")
    
    print("\n" + "="*80)
    print("✓ Phase 1 Complete!")
    print("="*80)
    print("\nNext: Run Phase 2 to create Normal profile and detect anomalies")


if __name__ == '__main__':
    main()
