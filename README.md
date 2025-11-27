🌾 Agriculture Crop Production Prediction (India)

This project predicts agricultural crop production in India using historical data (2006–2011).
A Random Forest Regressor model is trained using:

Crop

Year

Area

Yield

The deployed Streamlit application allows users to input these values and instantly receive production predictions along with estimated uncertainty.

🚀 Features
✔ Machine Learning Model (RandomForest)

MAE: 5.72

RMSE: 9.64

R² Score: 0.973

✔ Clean Preprocessing Pipeline

Year-wise Production, Area, and Yield unpivoting

Missing value handling

Label Encoding for Crop & Year

✔ Interactive Streamlit App

Crop & Year dropdowns

Numeric inputs for Area & Yield

Unit conversion (quintal → tons)

Result uncertainty (±MAE)

Clean prediction UI

📂 Project Structure
├── app.py
├── data/
│   ├── agri_model.joblib
│   ├── le_crop.joblib
│   ├── le_year.joblib
|   |── final_agri_dataset.csv
├── README.md
└── 
    ├── data_cleaning.ipynb
    ├── model_training.ipynb
    └── eda.ipynb

📝 Installation & Setup
1️⃣ Install Python libraries
pip install streamlit pandas numpy scikit-learn joblib

2️⃣ Run the Streamlit App
streamlit run app.py


This opens a browser window at:

http://localhost:8501/

🎯 Usage Instructions

Open the app

Select a Crop

Select a Historical Year (2006–2011)

Enter:

Area

Yield

Click Predict Production

The app displays:

Expected Production

Range (±MAE)

Tons + Quintal conversion

Inputs used

Model info

📈 Model Training Details

Algorithm: RandomForestRegressor

Training rows: 275

Features used:

Crop (encoded)

Year (encoded)

Area

Yield

Target variable: Production

📊 EDA Summary

Distribution plots for Production

Correlation Heatmap (Area–Yield–Production)

Value counts of Crops & Years

Missing value handling

Numeric summary (mean, std, min, max)

🧠 Future Enhancements

Add multi-year forecasting

Use LSTM or Prophet for time-series

Add cost-based yield optimization

Extend dataset beyond 2011

⚠ Dataset is not included due to licensing and size. 
Download CMAPSS FD001 from NASA PCoE:
([https://drive.google.com/file/d/1dgWM0KKOnoN9kVObbA-GahsgXPJBCT4c/view?usp=sharing](https://drive.google.com/file/d/1zfqvs8-mAO6E0JpgvhBdueNx8Th03pUp/view?usp=sharing ) )
and place files in /data before running the notebook.
