## *Promotion Effectiveness at a Fashion Retail Chain*

A fashion retailer operates 50 stores across urban, semi-urban, and rural locations. Each month, the marketing team runs one of five promotions: Flat Discount, BOGO (Buy-One-Get-One), Free Gift with Purchase, Category-Specific Offer, and Loyalty Points Bonus. Stores vary in size, monthly footfall, local competition density, and customer demographics. The company wants to determine which promotion should be deployed in each store each month to maximise the number of items sold.


# B1. Problem Formulation
## (a) ML Formulation
- Target Variable (y): 
items_sold (monthly sales volume per store) 
Candidate Input Features (X):
Store-level: store_size, location_type (urban/semi-urban/rural), competition_density 
- Customer behavior: monthly footfall, demographics (if available)
Promotion-related: promotion_type (5 categories)
- Time features: month, year, seasonality, is_weekend_ratio, is_festival
- Historical features: lagged sales (previous month sales), rolling averages
- - Type of ML Problem:
Supervised Regression
Justification: The goal is to predict a continuous numeric value (items_sold) based on known inputs. Since historical labeled data exists (past promotions → resulting sales), this fits supervised learning. Regression is appropriate because the output is not categorical.

## (b) Why “Items Sold” > Revenue

- Using revenue can be misleading because:

Different promotions affect price, not just demand
Example: Flat discount may increase units sold but reduce revenue per item
Revenue mixes pricing effects + demand effects

- Using items sold:

Directly measures customer response to promotions
Is independent of price distortions
Better reflects promotion effectiveness

(c) Alternative to One Global Model

A single global model ignores store heterogeneity.

## Better approaches:

- Hierarchical / Mixed Models (Recommended)
Global model + store-level adjustments
Captures shared patterns + local variations
Cluster-based Models
Group stores (urban vs rural, high vs low competition)
Train one model per cluster
Store-specific Models (if enough data)
Separate model per store

- Justification:
Different stores respond differently due to: Customer demographics,
Competition,
Buying power,

- A hierarchical or segmented approach balances generalization + personalization.

# B2. Data and EDA Strategy
## (a) Data Joining & Aggregation

Tables:

Transactions
Store attributes
Promotion details
Calendar

Joins:

Transactions + Store → store_id
Transactions + Promotion → promotion_id
Transactions + Calendar → transaction_date

Final Grain: One row = Store × Month

Aggregations:

Total items_sold (target)
Total revenue (optional)
Avg footfall
Promotion used that month
% weekend days
Festival flag
Avg basket size
Competition density (static)

## (b) EDA Plan

Here are key analyses:

1. Promotion vs Sales (Boxplot / Bar chart)

Compare items_sold across promotion types
Detect which promotions perform better overall
Helps in baseline expectations

2. Time Series Trends (Line plots)

Sales over months/years
Identify:
Seasonality (festivals, holidays)
Trends (growth/decline)

➡ Helps create time-based features (month, seasonality)

3. Store Segmentation Analysis (Cluster or grouped plots)

Compare urban vs rural vs semi-urban
Analyze sales distribution per segment

➡ Justifies segmented modeling

4. Correlation Heatmap

Identify relationships between:
Footfall
Competition
Sales

➡ Helps feature selection & multicollinearity handling

5. Promotion Effect by Store Type (Interaction plots)

Example: BOGO works better in urban stores?

➡ Suggests interaction features like:
promotion_type × location_type

## (c) Handling Promotion Imbalance (80% No Promotion)

Problem:

Model may learn to predict “no promotion effect” most of the time
Weak learning for rare promotion types

Solutions:

Add promotion indicator feature (binary)
Use resampling techniques (oversample promotion cases)
Use stratified evaluation (check performance by promotion type)
Consider separate models:
Promotion vs no-promotion
Then promotion-type prediction

# B3. Model Evaluation and Deployment
## (a) Train-Test Split & Metrics

Data: 3 years monthly data

Split Strategy:

Sort by time
Train: First 80% (earlier months)
Test: Last 20% (recent months)

Why NOT random split:

Causes data leakage
Future data may influence past predictions
Unrealistic for real-world forecasting

Evaluation Metrics:

MAE (Mean Absolute Error)
Average error in units sold
Easy to interpret
 “On average, we miss by 120 items”
RMSE
Penalizes large errors
 Important for avoiding big mistakes in high-volume stores
MAPE
Percentage error
Useful for comparing across stores
Business Metric (Important):
Incremental sales vs baseline
Promotion uplift

## (b) Explaining Different Recommendations

Model suggests:

December → Loyalty Points
March → Flat Discount

Use Feature Importance / Explainability:

Tools:

SHAP values (best)
Feature importance plots

What to analyze:

December:
Festival season → high demand
Loyalty points may increase repeat purchases
March:
Low demand period
Discounts may stimulate purchases

How to communicate:

Show top contributing features per prediction
Example:
“Festival flag contributed +300 units”
“Low footfall reduced baseline demand”

👉 Translate model output into business reasoning

## (c) Deployment Pipeline

1. Model Saving

Save trained model using:
joblib or pickle

2. Monthly Prediction Workflow

At start of each month:

Collect latest data:
Store features
Calendar info
Planned promotions
Apply same preprocessing pipeline
Generate predictions for each promotion option
Select promotion with highest predicted sales

3. Automation

Schedule pipeline (e.g., monthly batch job)
Output recommendations for all 50 stores

4. Monitoring

Track:

Prediction vs Actuals
MAE over time
Data Drift
Changes in:
footfall
competition
customer behavior
Concept Drift
Promotions stop working as before

5. Retraining Strategy

Retrain:
Every 3–6 months OR
When performance drops beyond threshold

End-to-End Flow:
Data → Feature Engineering → Model → Prediction → Recommendation → Monitoring → Retraining