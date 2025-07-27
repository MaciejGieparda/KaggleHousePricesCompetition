# Kaggle House Prices Competition 🏠📈

This repository contains all files related to the Kaggle competition:  
[House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

---

## 📁 Structure

- `data/` – raw `.csv` files (train, test, submission)  
- `notebooks/` – EDA and modeling notebooks  
- `src/` – scripts for preprocessing, modeling, utils  
- `reports/` – plots and visual summaries  

---

## 🏡 House Prices – Advanced Regression Techniques  
### 🔍 Full ML Pipeline: EDA, Feature Engineering, Modeling & Explainability

This notebook presents an end-to-end solution for the **House Prices** Kaggle competition. The pipeline covers:

---

### 📊 1. Exploratory Data Analysis (EDA)  
- Initial inspection, missing value heatmaps, target variable distribution  
- Correlation matrix to identify top predictive features  
- Outlier analysis (analyzed, not dropped – they may reflect market dynamics)

---

### 🧱 2. Feature Engineering  
- Smart imputation based on feature meaning  
- Ordinal encoding of categorical variables (missing values preserved as signal)  
- Saved clean dataset for modeling

---

### 🤖 3. Modeling & Benchmarking  
- Benchmarked models using RMSE:  
  - Baselines: DummyRegressor, LinearRegression  
  - Regularized: Ridge, Lasso  
  - Tree-based: RandomForest, XGBoost, LightGBM  
- ✅ **LightGBM** delivered the best performance

---

### 📈 4. Model Selection & Explainability  
- Final model: LightGBM with tuned parameters  
- Used **SHAP** for global feature importance and local explanation  
- Saved predictions for submission

---

### 🧠 Key Takeaways  
- Missing values can carry economic signal  
- Tree models need encoding, but handle NA well  
- SHAP enables interpretation of complex models  

---

👨‍💻 Author: **Maciej Gieparda**  
🛠️ Tools: Python, Pandas, Scikit-learn, LightGBM, SHAP, Seaborn, Matplotlib  
🏷️ Scope: `EDA → Preprocessing → Modeling → SHAP → Submission`
