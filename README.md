# 🔥 Predictive Modeling of London Fire Brigade Response Times and Pump Deployment Patterns

This project investigates how machine learning can be used to **predict emergency response times** and **fire pump deployment** by the London Fire Brigade. By analyzing real historical incident data from early 2017, the project implements two end-to-end ML pipelines designed to assist in **resource allocation** and **emergency preparedness** at the ward level.

---

## 📌 Project Objectives

1. **Predict** the number of fire pumps dispatched at the time of an emergency.
2. **Estimate** the fire brigade's response time to an incident.
3. **Identify** critical spatial, temporal, and operational features affecting these outcomes.
4. **Evaluate** ensemble learning approaches for improved model accuracy.

---

## 📂 Dataset

- 📅 **Timeframe**: January to April 2017  
- 📍 **Source**: London Fire Brigade (via London Datastore)  
- 🧮 **Records**: ~32,000 intervention logs  
- 🔑 **Key Features**: 
  - Incident time, location, borough, property type
  - Response time
  - Number of pumps deployed
  - Station information and more

---

## 🧪 Methodology

### 📋 Data Preprocessing

- Cleaning, validation, and null handling
- Feature engineering:
  - Spatiotemporal grids
  - Cyclical encoding for time-based features
  - Distance from city center
  - Historical property density and risk categories
- Outlier handling and Winsorization

### 🔄 Pipeline 1: Pump Deployment Classification

- **Problem**: Multiclass classification (1, 2, 3, or 4+ pumps)
- **Models Used**:
  - Logistic Regression
  - Random Forest Classifier
  - XGBoost (🏆 Best performer)
- **Highlights**:
  - Target leakage prevention
  - Class balancing using stratified sampling
  - RandomisedSearchCV for hyperparameter tuning

| Model                | Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
|---------------------|----------|------------------|---------------|----------------|
| XGBoost             | 79.8%    | 0.712            | 0.523         | 0.559          |
| Random Forest       | 78.3%    | 0.584            | 0.573         | 0.568          |
| Logistic Regression | 62.9%    | 0.438            | 0.503         | 0.425          |

---

### ⏱ Pipeline 2: Response Time Regression

- **Problem**: Predicting attendance time in seconds
- **Models Used**:
  - Random Forest
  - Gradient Boosting
  - XGBoost
  - LightGBM
  - Stacking Ensemble (🏆 Best performer)
- **Transformations**:
  - Log1p transformation of target
  - Target encoding for high-cardinality features
- **Ensemble Meta-Learner**: Ridge Regression

| Model            | MAE (s) | RMSE (s) | CV RMSE (s) | R² Score |
|------------------|---------|----------|-------------|----------|
| Stacking Ensemble| 63.88   | 98.56    | 95.71       | 0.4575   |
| LightGBM         | 65.53   | 99.44    | 96.58       | 0.4478   |
| Gradient Boosting| 63.00   | 100.04   | 97.08       | 0.4411   |
| XGBoost          | 65.26   | 100.19   | 97.53       | 0.4394   |
| Random Forest    | 66.58   | 101.10   | 98.68       | 0.4292   |

---

## 💡 Key Insights

- Fire brigade response efficiency decreases with distance from central London.
- Incidents peak during late mornings and early evenings, especially weekends.
- Response times are significantly impacted by **incident location**, **day/hour**, and **property type**.
- Ensemble methods using stacked regressors significantly improve model generalisation and accuracy.

---

## 🔮 Future Work

- Integrate **live traffic**, **weather**, and **road network** data for more accurate real-time predictions.
- Explore deeper **spatiotemporal models** like LSTMs or Transformers for sequence modeling.
- Investigate **simulation-based resource allocation** using predictive outputs.

---

## 🤝 Team Contributions

- **Bradley Marimbire**: Abstract, Pipeline 2 (code, methods, results, discussion)
- **Jacob Abraham Palakunnathu**: Literature review, EDA, cleaning, clustering, feature engineering
- **Diana Muthoni**: Literature review, Pipeline 1 (code, methods, results, discussion)

---

## 📚 References

- [Fire Facts – Incident Response Times 2017 (London Datastore)](https://data.london.gov.uk/dataset/incident-response-times-fire-facts)
- [Grid Reference Finder](https://gridreferencefinder.com/)
- Hassler et al., Sahebi et al., Sharma et al., among others (see full report for academic references)
