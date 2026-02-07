# 🚀 Quick Start Guide - DriveIQ Capstone

Get up and running with the drowsiness detection pipeline in 5 minutes!

---

## 🎯 What This Project Does

This system detects and explains abnormal driving behavior in three phases:

1. **Phase 1:** "Was this trip Normal, Drowsy, or Aggressive?" → Session classification
2. **Phase 2:** "How weird is this 30-second window?" → Anomaly score (0-100)
3. **Phase 3:** "WHY is it weird?" → Feature attribution + actionable feedback

**Example Output:**
```
🔴 Window #142 - Anomaly Score: 87/100

⚠️ Lane weaving detected (3.2σ from normal)
⚠️ Harsh braking detected (2.8σ from normal)

Recommendation: Maintain steady lane position and anticipate stops earlier.
```

---

## ⚡ 5-Minute Quick Start

### **Step 1: Install Dependencies (1 minute)**

```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn joblib
```

### **Step 2: Run Phase 2 - Anomaly Detection (2 minutes)**

This is the **most important** phase - it creates the Normal baseline and scores all windows.

```bash
python scripts/phase2_anomaly_detection.py
```

**When prompted:**
```
Method: 1         # Press 1 for Mahalanobis (recommended)
Motor features: data/features/motor_window_features.csv
Secondary features: data/features/secondary_window_features.csv
```

**Output:**
- Creates Normal driving profile
- Scores every window (0-100)
- Shows distribution by behavior
- Saves to: `results/phase2_results/`

### **Step 3: Run Phase 3 - Get Feedback (2 minutes)**

```bash
python scripts/phase3_feature_attribution.py
```

**When prompted:**
```
Profile: models/saved_models/phase2_normal_profile.pkl
Scores: results/phase2_results/windows_with_anomaly_scores.csv
```

**Interactive menu:**
- Type `1` to analyze a specific window
- Enter a high anomaly score window number
- See detailed feedback and visualization!

**Example:**
```
Choice: 1
Window index: 142

FEEDBACK FOR WINDOW 142
Anomaly Score: 87.3/100

🔴 Abnormal driving detected
⚠️ Lane weaving (3.2σ) - Maintain steady position
⚠️ Harsh braking (2.8σ) - Brake more gradually
```

---

## 📊 Understanding Your Results

### Phase 2: Anomaly Scores

**What the numbers mean:**
- **0-30:** ✅ Normal driving (expected for Normal behavior)
- **30-50:** 🟡 Slightly unusual
- **50-70:** 🟠 Moderately abnormal
- **70-85:** 🔴 Highly abnormal (warning level)
- **85-100:** 🔴🔴 Extremely abnormal (critical)

**Expected Results:**
- Most **Normal** windows: 0-40
- Most **Drowsy** windows: 50-80
- Most **Aggressive** windows: 60-95

If you don't see this separation, you may need to add more derived features or tune the model.

---

### Phase 3: Feature Contributions

**What the symbols mean:**

**σ (sigma)** = Standard deviations from normal
- **1σ:** 1 in 3 chance (mild)
- **2σ:** 1 in 20 chance (moderate)
- **3σ:** 1 in 370 chance (severe)
- **4σ+:** 1 in 15,000+ chance (critical)

**Severity Levels:**
- 🟢 **Low:** 1-2σ (noticeable but not dangerous)
- 🟡 **Medium:** 2-3σ (concerning, needs attention)
- 🔴 **High:** 3+σ (dangerous, immediate action)

---

## 🎮 Full Three-Phase Walkthrough

### **Phase 1: Session Classification (Optional)**

This classifies entire driving sessions/routes.

```bash
python scripts/phase1_session_classification.py
```

**Input:** Same window features  
**Output:** One prediction per session (Normal/Drowsy/Aggressive)  
**Use Case:** Post-trip analysis, "How did the whole trip go?"

---

### **Phase 2: Anomaly Detection (Core)**

This creates the Normal baseline and scores each window.

```bash
python scripts/phase2_anomaly_detection.py
```

**What it does:**
1. Filters only Normal driving windows
2. Calculates mean and covariance (the "Normal profile")
3. For every window, measures distance from Normal
4. Outputs anomaly score (0-100)

**Files created:**
- `windows_with_anomaly_scores.csv` - All windows with scores
- `anomaly_distribution.png` - Histogram by behavior
- `anomaly_boxplot.png` - Box plots
- `phase2_normal_profile.pkl` - Saved model

---

### **Phase 3: Feature Attribution (Explanation)**

This explains WHY windows are abnormal.

```bash
python scripts/phase3_feature_attribution.py
```

**What it does:**
1. For each feature, calculates z-score (how many σ from normal)
2. Identifies top contributing features
3. Maps to behaviors (e.g., `x_lane_std` → "lane weaving")
4. Generates actionable feedback

**Interactive Options:**
- **Option 1:** Analyze specific window (detailed view + chart)
- **Option 2:** Batch analyze all high-anomaly windows (CSV report)

---

## 🔧 Common Workflows

### **Workflow 1: Real-Time Monitoring (Simulated)**

```bash
# Get anomaly score for a single window
python scripts/phase2_anomaly_detection.py
# → Load your live data as a single window
# → Get score
# → If score > 70: Run Phase 3 for feedback
```

### **Workflow 2: Post-Trip Analysis**

```bash
# 1. Score the entire trip
python scripts/phase2_anomaly_detection.py

# 2. Identify problem areas
# Look at windows_with_anomaly_scores.csv
# Find windows with score > 70

# 3. Get detailed explanations
python scripts/phase3_feature_attribution.py
# Analyze each high-score window
```

### **Workflow 3: Driver Performance Report**

```bash
# 1. Classify the session
python scripts/phase1_session_classification.py
# → Overall behavior: Drowsy

# 2. Find anomalies
python scripts/phase2_anomaly_detection.py
# → 8 windows with score > 70

# 3. Generate feedback
python scripts/phase3_feature_attribution.py
# → Main issues: Lane weaving (6x), Harsh braking (3x)
```

---

## 📁 Where Are My Files?

### **Input Data (What You Provide):**
```
data/features/
├── motor_window_features.csv      # Your window features (motorway)
└── secondary_window_features.csv  # Your window features (secondary roads)
```

### **Output Results (What Gets Created):**
```
results/
├── phase1_results/
│   └── model_comparison.csv       # Session classification results
│
├── phase2_results/
│   ├── windows_with_anomaly_scores.csv  # ⭐ Main output
│   ├── anomaly_distribution.png
│   └── anomaly_boxplot.png
│
└── phase3_results/
    ├── all_window_feedback.csv          # Batch feedback
    └── window_*_contributions.png       # Individual charts
```

### **Saved Models:**
```
models/saved_models/
├── phase1_session_classifier.pkl
└── phase2_normal_profile.pkl  # ⭐ The Normal baseline
```

---

## 🐛 Troubleshooting

### **"No such file or directory"**

Make sure you're in the right directory:
```bash
pwd  # Should show: .../DriveIQ_Capstone
ls   # Should show: scripts/, data/, models/, etc.
```

### **"All windows have high scores"**

Check you have enough Normal windows:
```python
import pandas as pd
motor = pd.read_csv('data/features/motor_window_features.csv')
print(motor['label'].value_counts())
# Should have label=0 (Normal) with decent count
```

### **"Module not found"**

Install missing dependencies:
```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn joblib
```

### **"Can't open file"**

Check your file paths are correct. Use relative paths from project root:
```bash
# If you see this error:
FileNotFoundError: [Errno 2] No such file or directory: 'motor_window_features.csv'

# Use full path:
data/features/motor_window_features.csv
```

---

## 🎓 Next Steps

### **Immediate (Now):**
- ✅ Run Phase 2 and Phase 3 with your current data
- ✅ Analyze a few high-anomaly windows
- ✅ Understand the feedback system

### **Short Term (This Week):**
- 📋 Add all 6 drivers (D1-D6) for better model
- 📊 Run Phase 1 for session-level classification
- 🎨 Customize feedback messages for your use case

### **Long Term (Later):**
- 🚀 Real-time implementation
- 📱 Mobile/web dashboard
- 🔧 Hyperparameter tuning
- 📈 Longitudinal analysis (track drivers over time)

---

## 💡 Pro Tips

### **Tip 1: Start with Phase 2**
Phase 2 is the heart of the system. Run this first to understand your data.

### **Tip 2: Focus on High-Anomaly Windows**
Don't analyze every window. Focus on score > 70 for actionable insights.

### **Tip 3: Customize Feature Mappings**
Edit `phase3_feature_attribution.py` to add your own feedback messages:
```python
self.feature_behaviors = {
    'your_feature': 'Your custom behavior description',
}
```

### **Tip 4: Visualize Your Results**
Phase 3 creates charts showing which features contribute most. These are great for presentations!

### **Tip 5: Iterate on Feedback**
After seeing results, refine your feedback messages to be more specific and actionable for your context.

---

## 📚 Learn More

- **[REVISED_THREE_PHASES_GUIDE.md](docs/REVISED_THREE_PHASES_GUIDE.md)** - Complete technical details
- **[README.md](README.md)** - Full project documentation
- **[DATA_ANALYSIS_AND_RECOMMENDATIONS.md](docs/DATA_ANALYSIS_AND_RECOMMENDATIONS.md)** - Data structure explained

---

## 🎯 Success Checklist

After completing the quick start, you should have:

- ✅ Normal profile created (`phase2_normal_profile.pkl`)
- ✅ All windows scored (`windows_with_anomaly_scores.csv`)
- ✅ Clear separation between Normal/Drowsy/Aggressive scores
- ✅ Feedback generated for at least one high-anomaly window
- ✅ Visualization showing feature contributions

If you have all of these, **you're ready to move forward!** 🚀

---

## ❓ Need Help?

Check the files in order:
1. This file (QUICK_START_GUIDE.md) - You're here!
2. REVISED_THREE_PHASES_GUIDE.md - Deep dive
3. README.md - Full documentation

Still stuck? Review the error message carefully and check file paths.

---

**Time to completion:** 5-10 minutes  
**Difficulty:** Beginner  
**Prerequisites:** Python installed, data files ready

**Happy analyzing!** 🎉
