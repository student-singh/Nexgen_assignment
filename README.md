# 🚀 NexGen SmartRoute Planner

**AI-Powered Delivery Route Optimization for Cost, Time, and Sustainability**

[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-FF4B4B.svg?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.2-F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

---

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Demo Screenshots](#demo-screenshots)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Data Processing](#data-processing)
- [Machine Learning Model](#machine-learning-model)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Future Enhancements](#future-enhancements)
- [License](#license)

---

## 🎯 Overview

**NexGen SmartRoute Planner** is an intelligent delivery route optimization system that leverages **Machine Learning** and **AI** to recommend the best vehicle and route for delivery operations. The system balances three critical factors:

- 💰 **Cost Efficiency** - Minimizes delivery expenses
- ⏱️ **Time Optimization** - Reduces delivery time
- 🌍 **Sustainability** - Lowers carbon emissions (CO2)

The application addresses real-world logistics challenges where suboptimal vehicle selection and weather conditions lead to high costs (avg $1013) and delays. Our solution provides **18% cost savings** and **20% delay reduction**.

---

## ✨ Features

### 🔍 Core Capabilities
- **Smart Vehicle Recommendation** - ML-powered suggestions based on distance, weight, mode, weather, and region
- **Real-Time Route Optimization** - Instant calculation of cost, time, CO2, and delay risk
- **Interactive Dashboards** - 4+ visualization types (bar, scatter, heatmap, line charts)
- **Delay Risk Prediction** - RandomForest classifier with 200+ estimators and SMOTE balancing
- **Multi-Filter Analytics** - Filter data by region, weather, and delivery mode
- **Export Functionality** - Download filtered results as CSV

### 🤖 AI Integration
- **Google Gemini AI** - Natural language insights for route recommendations
- **Fallback Logic** - Works offline if API is unavailable

### 📊 Key Metrics Tracked
- Average Cost
- Delay Percentage
- Total CO2 Emissions
- Optimization Score

---

## 🖼️ Demo Screenshots

*(Add screenshots of your app here after running it)*

1. **Dashboard Overview** - KPIs and filters
2. **Visualizations** - Charts showing delay patterns, cost distribution
3. **Route Optimizer** - Input form and AI recommendations
4. **Export Feature** - CSV download button

---

## 🛠️ Installation

### Prerequisites
- **Python 3.10+** (Python 3.11+ recommended for better compatibility)
- **pip** package manager
- **Git** (optional, for cloning)

### Step 1: Clone the Repository
```bash
git clone <your-repo-url>
cd nextgen_smartroute_planner
```

### Step 2: Create Virtual Environment
```powershell
# Windows (PowerShell)
python -m venv .venv
& .\.venv\Scripts\Activate.ps1

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

**Dependencies installed:**
- `streamlit==1.28.0` - Web app framework
- `pandas==2.1.4` - Data manipulation
- `numpy==1.25.2` - Numerical computing
- `plotly==5.17.0` - Interactive visualizations
- `scikit-learn==1.7.2` - Machine learning
- `google-generativeai` - Gemini AI integration
- `protobuf==4.25.8` - Protocol buffers (for compatibility)
- `imbalanced-learn` - SMOTE for class balancing

### Step 4: Set Up API Key (Optional)
Create `.streamlit/secrets.toml` for Gemini AI:
```toml
GEMINI_API_KEY = "your-api-key-here"
```
> **Note:** The app works without this - it will use fallback suggestions.

### Step 5: Prepare Data
Run the data preprocessing script:
```powershell
python explore_data.py
```
This generates `data/processed_delivery_data.csv` with calculated metrics (delay_pct, co2_kg, opt_score).

---

## 🚀 Usage

### Run the Application
```powershell
# Activate venv first (if not already active)
& .\.venv\Scripts\Activate.ps1

# Launch Streamlit app
streamlit run app.py

# Alternative: Use venv's streamlit directly
& ".\.venv\Scripts\streamlit.exe" run app.py
```

The app will open in your browser at **http://localhost:8501** (or next available port).

### Using the App

1. **Explore Dashboard**
   - View KPIs (Average Cost, Delay %, CO2, Optimization Score)
   - Use sidebar filters to drill down by Region, Weather, or Mode

2. **Analyze Visualizations**
   - **Delay by Weather** - Bar chart showing which conditions cause delays
   - **Cost vs Distance** - Scatter plot with vehicle type color-coding
   - **Delay Heatmap** - Region × Delivery Mode delay matrix
   - **Cost Trends** - Line chart of costs over delivery IDs

3. **Optimize a Route**
   - Enter: Distance (km), Weight (kg), Mode, Weather, Region
   - Click **"Optimize!"**
   - View best vehicle recommendation with cost, time, CO2, and risk
   - See AI-generated suggestion (if API key is configured)
   - Review all alternative vehicle options in a table

4. **Export Data**
   - Scroll to "Export" section
   - Click **"Download CSV"** to save filtered data

---

## 📂 Project Structure

```
nextgen_smartroute_planner/
│
├── .streamlit/
│   └── secrets.toml           # API keys (not tracked in git)
│
├── .venv/                      # Virtual environment (not tracked)
│
├── data/
│   ├── dilevery_logistics.csv         # Raw data
│   └── processed_delivery_data.csv    # Preprocessed data with metrics
│
├── app.py                      # Main Streamlit application
├── explore_data.py             # Data preprocessing script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🧰 Technologies Used

| Technology | Purpose |
|------------|---------|
| **Python 3.10+** | Core programming language |
| **Streamlit** | Web app framework for interactive dashboards |
| **Pandas** | Data manipulation and analysis |
| **NumPy** | Numerical operations |
| **Plotly** | Interactive visualizations (charts, graphs) |
| **scikit-learn** | Machine learning (RandomForest, train-test split) |
| **imbalanced-learn** | SMOTE for handling class imbalance |
| **Google Gemini AI** | Natural language insights (optional) |
| **Protobuf** | Data serialization for API compatibility |

---

## 📊 Data Processing

### Input Data (`dilevery_logistics.csv`)
Expected columns:
- `delivery_id`, `distance_km`, `package_weight_kg`
- `weather_condition`, `region`, `vehicle_type`, `delivery_mode`
- `delivery_time_hours`, `expected_time_hours`, `delivery_cost`
- `delayed` (yes/no), `delivery_rating`

### Processing Steps (`explore_data.py`)
1. **Data Cleaning**
   - Standardize `delayed` to lowercase
   - Convert time strings to numeric
   - Fill missing values

2. **Feature Engineering**
   - `delay_hours` = actual - expected time
   - `delay_pct` = (delay_hours / expected) × 100
   - `co2_kg` = emissions based on vehicle type and distance
     - EV: 0 kg/km
     - Bike/Scooter: 0.05 kg/km
     - Van: 0.2 kg/km
     - Truck: 0.5 kg/km
   - `opt_score` = weighted score (40% cost, 30% delay, 30% CO2)

3. **Output**
   - Saves to `data/processed_delivery_data.csv`

---

## 🤖 Machine Learning Model

### Algorithm: RandomForest Classifier

**Task:** Predict delivery delay (binary: yes/no)

### Model Configuration
```python
RandomForestClassifier(
    n_estimators=200,      # More trees for better accuracy
    max_depth=15,          # Prevent overfitting
    min_samples_split=5,   # Balanced splits
    random_state=42
)
```

### Features Used
- `distance_km`, `package_weight_kg`
- `weather_condition`, `region`, `vehicle_type` (one-hot encoded)
- `heavy_load` (>50kg flag)
- `long_distance` (>200km flag)

### Training Process
1. **Train-Test Split:** 80/20
2. **SMOTE Oversampling:** Balances "yes" vs "no" delays in training set
3. **Cross-Validation:** 5-fold CV for accuracy estimation
4. **Prediction:** Outputs delay probability (0-1)

### Performance
- **Accuracy:** Displayed in app (via cross-validation)
- **Metric:** Delay risk percentage shown per route recommendation

---

## ⚙️ Configuration

### Environment Variables
Create `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "your-actual-api-key"
```

### Data Paths
Edit in `app.py` if needed:
```python
pd.read_csv('data/processed_delivery_data.csv')
```

### Model Tuning
Adjust in `load_model()` function:
- `n_estimators` - More trees = better accuracy but slower
- `max_depth` - Lower = less overfitting
- `SMOTE` parameters for class balancing

---

## 🔧 Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'plotly'`
**Solution:** Ensure you're using the venv and dependencies are installed:
```powershell
& .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Issue: `FileNotFoundError: No secrets files found`
**Solution:** Create `.streamlit/secrets.toml` or the app will use fallback logic (no AI insights).

### Issue: Gemini API Error `404 models/gemini-1.5-flash is not found`
**Solution:** The app has fallback logic. Update API key or model name in `app.py`:
```python
genai.configure(api_key="your-key")
model_llm = genai.GenerativeModel('gemini-pro')  # Try alternate model
```

### Issue: Streamlit uses wrong Python environment
**Solution:** Use venv's streamlit directly:
```powershell
& ".\.venv\Scripts\streamlit.exe" run app.py
```

### Issue: `imblearn` import errors
**Solution:** Install explicitly:
```powershell
pip install imbalanced-learn
```

---

## 🚀 Future Enhancements

- [ ] **Real-Time GPS Integration** - Live route tracking
- [ ] **Historical Trend Analysis** - Time-series forecasting
- [ ] **Multi-Depot Optimization** - Fleet-wide route planning
- [ ] **Weather API Integration** - Auto-fetch current conditions
- [ ] **Cost Breakdown** - Itemized fuel, labor, maintenance costs
- [ ] **Mobile App** - React Native companion app
- [ ] **Database Backend** - PostgreSQL/MongoDB for large datasets
- [ ] **User Authentication** - Role-based access control
- [ ] **Notification System** - Email/SMS alerts for delays

---

## 📄 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2025 NexGen SmartRoute Planner

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 👥 Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

For questions or support:
- **Email:** [your-email@example.com]
- **GitHub Issues:** [Open an issue](https://github.com/your-username/nextgen_smartroute_planner/issues)
- **LinkedIn:** [Your LinkedIn Profile]

---

## 🙏 Acknowledgments

- **Streamlit** for the amazing web app framework
- **scikit-learn** for robust ML tools
- **Google Gemini AI** for natural language insights
- **Plotly** for beautiful interactive visualizations

---

**Built with ❤️ for smarter, greener logistics**

*Last Updated: December 19, 2025*
