# 🎯 REVISED: Your Actual Three Phases

## Phase Goals (Corrected Understanding)

### **Phase 1: Session Classification** ✅
**Goal:** Identify if an entire driving session was Drowsy, Normal, or Aggressive

**Approach:**
- Aggregate window features across entire session/route
- Multi-class classification (3 classes)
- One prediction per session

**Your Data:** ✅ Ready to use

**Script:** `phase1_session_classification.py`

---

### **Phase 2: Normal Profile & Anomaly Detection** 🎯
**Goal:** Create a "normal driving" baseline and measure how far each window deviates

**Think of it as:**
- Normal driving = center point in feature space
- Each window = a point
- Distance from center = "weirdness score"

**Approach:**
1. Train only on **Normal** windows
2. Calculate mean, std, covariance (the "normal profile")
3. For any window, calculate distance from this profile
4. Output: Anomaly score (0-100) for each window

**Methods Available:**
- **Mahalanobis Distance** (recommended) - statistical distance
- **Isolation Forest** - ML-based anomaly detection
- **Elliptic Envelope** - robust covariance

**Script:** `phase2_anomaly_detection.py`

---

### **Phase 3: Feature Attribution & Feedback** 💡
**Goal:** Explain WHY a window is anomalous and give actionable feedback

**Example Output:**
```
Window #142 - Anomaly Score: 87/100

🔴 Abnormal driving detected

⚠️ Excessive lane weaving (severity: 3.2σ)
⚠️ Harsh braking (severity: 2.8σ)  
⚠️ Erratic speed changes (severity: 2.1σ)

Recommendations:
• Focus on maintaining steady lane position
• Anticipate stops earlier and brake gradually
• Consider taking a break - multiple issues detected
```

**Approach:**
1. For each feature, calculate z-score (standard deviations from normal)
2. Identify top contributing features
3. Map features to behaviors (e.g., acc_x_min → harsh braking)
4. Generate human-readable feedback

**Script:** `phase3_feature_attribution.py`

---

## 🚀 How to Run (Step by Step)

### **Prerequisites**
```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn joblib
```

### **Phase 1: Classify Sessions**

```bash
python3 phase1_session_classification.py
```

**Prompts:**
```
Motor features: motor_window_features.csv
Secondary features: secondary_window_features.csv
```

**Output:**
- Classification accuracy by session
- Confusion matrix
- Saved model: `models/saved_models/phase1_session_classifier.pkl`

**Expected Results:**
- Accuracy: 70-85% (depends on data quality and number of sessions)
- Shows which sessions are Normal/Drowsy/Aggressive

---

### **Phase 2: Create Normal Profile & Detect Anomalies**

```bash
python3 phase2_anomaly_detection.py
```

**Prompts:**
```
Select method: 1 (Mahalanobis - recommended)
Motor features: motor_window_features.csv
Secondary features: secondary_window_features.csv
```

**What it does:**
1. Filters only "Normal" windows
2. Calculates mean and covariance matrix
3. For each window (Normal, Drowsy, Aggressive), calculates distance from Normal
4. Outputs anomaly score (0-100)

**Output Files:**
- `results/phase2_results/windows_with_anomaly_scores.csv` - All windows with scores
- `results/phase2_results/anomaly_distribution.png` - Score distribution by behavior
- `results/phase2_results/anomaly_boxplot.png` - Box plots
- `models/saved_models/phase2_normal_profile.pkl` - Saved profile

**Expected Results:**
- Normal windows: Low scores (0-40)
- Drowsy windows: Medium-High scores (50-80)
- Aggressive windows: High scores (60-95)

---

### **Phase 3: Generate Feedback**

```bash
python3 phase3_feature_attribution.py
```

**Prompts:**
```
Normal profile: models/saved_models/phase2_normal_profile.pkl
Anomaly scores: results/phase2_results/windows_with_anomaly_scores.csv
```

**Interactive Menu:**
1. Analyze specific window (get detailed feedback)
2. Analyze all high-anomaly windows (batch processing)

**Example Session:**
```
Choice: 1
Window index: 142

FEEDBACK FOR WINDOW 142
Anomaly Score: 87.3/100
Behavior: Drowsy

🔴 Abnormal driving detected

⚠️ Lane weaving (severity: 3.2σ). Maintain steady lane position.
⚠️ Harsh braking (severity: 2.8σ). Maintain gradual deceleration.
⚠️ Erratic speed (severity: 2.1σ). Maintain consistent speed.

Recommendations:
• Focus on maintaining steady lane position
• Anticipate stops earlier and brake more gradually
• Consider taking a break
```

**Output:**
- Visual chart showing top contributing features
- CSV with feedback for all high-anomaly windows
- Severity classification (low/medium/high)

---

## 📊 Understanding the Results

### Phase 1: Session Classification
**Question Answered:** "Was this entire trip Normal, Drowsy, or Aggressive?"

**Use Case:**
- Post-trip analysis
- Driver performance evaluation
- Fleet management

---

### Phase 2: Anomaly Scores
**Question Answered:** "How weird is this 30-second window compared to normal driving?"

**Interpretation:**
- **0-30:** Normal driving
- **30-50:** Slightly abnormal
- **50-70:** Moderately abnormal  
- **70-85:** Highly abnormal
- **85-100:** Extremely abnormal

**Key Insight:** 
- Most Normal windows should score 0-40
- Most Drowsy/Aggressive windows should score 50-100
- If not → profile may need tuning

---

### Phase 3: Feature Attribution
**Question Answered:** "WHICH behaviors caused this window to be abnormal?"

**Feature → Behavior Mapping:**
| Feature | Behavior | Feedback |
|---------|----------|----------|
| `x_lane_std` | Lane weaving | "Maintain steady lane position" |
| `acc_x_kf_min` | Harsh braking | "Brake more gradually" |
| `speed_kmh_std` | Erratic speed | "Maintain consistent speed" |
| `phi_std` | Excessive steering | "Smooth steering inputs" |
| `ttc_front_mean` | Poor reaction time | "Increase following distance" |
| `steering_entropy` | Unpredictable driving | "Stay focused and alert" |

---

## 💡 Real-World Application

### **In-Vehicle Real-Time System:**

```
Every 30 seconds:
1. Calculate window features
2. Get anomaly score (Phase 2)
3. If score > 70:
   → Identify contributing features (Phase 3)
   → Display feedback to driver
   → "⚠️ Lane weaving detected. Focus on lane position."
```

### **Post-Trip Analysis:**

```
After trip:
1. Classify entire session (Phase 1)
2. Identify high-anomaly windows (Phase 2)
3. Generate detailed report (Phase 3)
   → "You had 8 high-anomaly windows during this trip"
   → "Main issues: lane weaving (6x), harsh braking (3x)"
```

### **Fleet Management Dashboard:**

```
For each driver:
- Overall behavior classification (Phase 1)
- Anomaly score trends over time (Phase 2)
- Top behavioral issues (Phase 3)
```

---

## 🔬 Technical Details

### How Mahalanobis Distance Works:

```python
# Normal profile
mean = [speed_mean=100, lane_std=0.2, ...]
covariance = [[...], [...], ...]  # Feature correlations

# New window
window = [speed_mean=85, lane_std=0.8, ...]

# Calculate distance
diff = window - mean
distance = sqrt(diff^T × Cov^-1 × diff)

# Higher distance = more abnormal
```

**Why it's good:**
- Accounts for feature correlations
- Handles multi-dimensional data well
- Statistically principled

---

### Z-Score (Feature Contribution):

```python
z_score = |value - normal_mean| / normal_std

# Example:
# Normal lane_std: mean=0.2, std=0.1
# Window lane_std: 0.5
# Z-score = |0.5 - 0.2| / 0.1 = 3.0

# Interpretation:
# 3σ from normal = HIGHLY abnormal
```

---

## 🎨 Customization Ideas

### Custom Feedback Messages:

```python
# In phase3_feature_attribution.py

# Add your own feature mappings:
self.feature_behaviors = {
    'custom_feature': 'Your custom behavior description',
    ...
}

# Add custom feedback logic:
if 'your_feature' in feature.lower():
    feedback = "Your custom feedback message"
```

### Adjust Severity Thresholds:

```python
# In phase3_feature_attribution.py

self.severity_thresholds = {
    'low': 1.5,      # 1.5σ instead of 1σ (less sensitive)
    'medium': 2.5,   # 2.5σ
    'high': 4.0      # 4σ (only extreme cases)
}
```

### Add New Features:

Use `generate_window_features.py` to add custom derived features like:
- Time since last brake
- Acceleration jerk (rate of change of acceleration)
- Steering wheel reversal rate
- etc.

---

## 🐛 Troubleshooting

### "All windows have high anomaly scores"
**Problem:** Normal profile is too narrow
**Solution:** 
- Check if you have enough Normal windows (need 50+)
- Try different anomaly detection method
- Adjust contamination parameter

### "Normal and Drowsy have similar scores"
**Problem:** Features don't differentiate well
**Solution:**
- Add more derived features (use enhanced window generator)
- Check feature importance from Phase 1
- Focus on top discriminative features

### "Feedback doesn't make sense"
**Problem:** Feature-behavior mapping is generic
**Solution:**
- Customize `feature_behaviors` dict in Phase 3
- Add domain-specific mappings
- Review actual feature values to understand patterns

---

## 📈 Next Steps

1. **Run all three phases** with your current data
2. **Review results:**
   - Does Phase 1 classify sessions correctly?
   - Do Normal windows have low Phase 2 scores?
   - Does Phase 3 feedback make sense?
3. **Add all 6 drivers** for better generalization
4. **Enhance features** using `generate_window_features.py`
5. **Customize feedback** based on your domain knowledge

---

## 🎯 Bottom Line

Your three phases are:

1. **Session Classification** - Classify entire trips
2. **Anomaly Detection** - Score each 30s window vs Normal
3. **Feature Attribution** - Explain why + give feedback

Your current data **works for all three phases**! Just need to:
- ✅ Run the scripts I provided
- ⚠️ Add D2-D6 for production quality
- 💡 Customize feedback messages for your use case

**This is actually MORE sophisticated than typical binary classification!** You're building an explainable anomaly detection system with actionable feedback. Very cool! 🚀
