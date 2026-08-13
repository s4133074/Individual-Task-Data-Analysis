#!/usr/bin/env python
# coding: utf-8

# # Part 1.3 – Data Analysis
# Two machine learning algorithms, Logistic Regression and Random Forest are applied separately to the two datasets. The models are used to identify patterns and compare there effectiveness in supporting Business Analyst decision-making. Logistic Regression provides an interpretable linear model, while Random Forest can capture more complex relationships between variables.

# ### Dataset 1 – OFT Consumer Complaints
# 
# The first analysis uses the Office of Fair Trading consumer complaints dataset. The modelling task is to predict whether the complainant who reported belonged to a vulnerable group based on available complaint characteristics such as channel, industry, product, region and timing.

# In[1]:


# Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)


# In[2]:


# Load the consumer complaint dataset

complaints = pd.read_csv(
    r"Downloads\complaints-received-by-oft-01012026-to-31032026.csv"
)

complaints.head()


# In[3]:


print(complaints.shape)
print(complaints.info())
print(complaints.isnull().sum())


# In[4]:


print(complaints.columns.tolist())


# In[5]:


# Clean the complaints data


# In[6]:


complaints = complaints.drop(
    columns=["Case Number"]
)


# In[7]:


complaints["OFT Region"] = complaints["OFT Region"].fillna(
    "Unknown"
)


# In[8]:


complaints["Initiated Date"] = pd.to_datetime(
    complaints["Initiated Date"],
    format="%m/%d/%Y %H:%M"
)


# In[9]:


complaints["Month"] = complaints["Initiated Date"].dt.month
complaints["DayOfWeek"] = complaints["Initiated Date"].dt.dayofweek


# In[10]:


complaints = complaints.drop(
    columns=["Initiated Date"]
)


# ### Target Definition and Feature Preparation
# "Vulnerable Group Provided" is used as the target variable, with Yes encoded as 1 and No as 0. Categorical attributes are converted into numerical dummy variables so they can be used by the machine learning models.

# In[11]:


print(
    complaints["Vulnerable Group Provided"].unique()
)


# In[12]:


# Encode target variable as 1 for Yes and 0 for No

complaints["Target"] = complaints[
    "Vulnerable Group Provided"
].map({
    "Yes": 1,
    "No": 0
})


# In[13]:


complaints = complaints.drop(
    columns=["Vulnerable Group Provided"]
)


# In[14]:


#Prepare X and Y

X = complaints.drop(columns=["Target"])
Y = complaints["Target"]


# In[15]:


print(Y.value_counts())
print(Y.value_counts(normalize=True))


# In[16]:


X = pd.get_dummies(
    X,
    drop_first=True
)


# In[17]:


print(X.shape)
X.head()


# ### Training and Testing Data
# 
# The data is divided into 80% training and 20% testing sets. Stratified sampling is used to maintain the target class distribution in both sets. The test set is kept separate from training so that model effectiveness can be evaluated using unseen observations.

# In[18]:


X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.20,
    random_state=42,
    stratify=Y
)


# ### Logistic Regression

# In[19]:


# Scaling features for Logistic Regression

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[20]:


log_model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)

log_model.fit(
    X_train_scaled,
    Y_train
)


# In[21]:


log_pred = log_model.predict(
    X_test_scaled
)

log_prob = log_model.predict_proba(
    X_test_scaled
)[:, 1]


# In[22]:


# Evaluate logistic regression

print(
    classification_report(
        Y_test,
        log_pred
    )
)


# In[23]:


log_accuracy = accuracy_score(
    Y_test,
    log_pred
)

log_precision = precision_score(
    Y_test,
    log_pred
)

log_recall = recall_score(
    Y_test,
    log_pred
)

log_f1 = f1_score(
    Y_test,
    log_pred
)

log_auc = roc_auc_score(
    Y_test,
    log_prob
)


# In[24]:


print("Accuracy:", log_accuracy)
print("Precision:", log_precision)
print("Recall:", log_recall)
print("F1:", log_f1)
print("ROC-AUC:", log_auc)


# ### Random Forest

# In[25]:


RF_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

RF_model.fit(
    X_train,
    Y_train
)


# In[26]:


RF_pred = RF_model.predict(
    X_test
)

RF_prob = RF_model.predict_proba(
    X_test
)[:, 1]


# In[27]:


# Evaluate Random Forest

RF_accuracy = accuracy_score(
    Y_test,
    RF_pred
)

RF_precision = precision_score(
    Y_test,
    RF_pred
)

RF_recall = recall_score(
    Y_test,
    RF_pred
)

RF_f1 = f1_score(
    Y_test,
    RF_pred
)

RF_auc = roc_auc_score(
    Y_test,
    RF_prob
)


# In[28]:


print("Accuracy:", RF_accuracy)
print("Precision:", RF_precision)
print("Recall:", RF_recall)
print("F1:", RF_f1)
print("ROC-AUC:", RF_auc)


# ### Model Evaluation

# In[29]:


complaints_results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest"
    ],
    "Accuracy": [
        log_accuracy,
        RF_accuracy
    ],
    "Precision": [
        log_precision,
        RF_precision
    ],
    "Recall": [
        log_recall,
        RF_recall
    ],
    "F1": [
        log_f1,
        RF_f1
    ],
    "ROC-AUC": [
        log_auc,
        RF_auc
    ]
})

print(complaints_results)


# #### Complaints Model Results
# 
# Random Forest achieved higher overall accuracy than Logistic Regression, while Logistic Regression achieved higher recall for the vulnerable-group class. However, both models produced relatively low F1 and ROC-AUC scores, indicating limited ability to distinguish vulnerable-group cases.
# 
# The results show why accuracy should not be considered alone as the vulnerable-group class is less common, a model may achieve higher accuracy while failing to identify many actual vulnerable-group cases. Recall and F1-score are therefore important for evaluating this dataset.

# ### Feature Insights – Logistic Regression

# In[30]:


log_features = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": log_model.coef_[0]
})

log_features["Absolute"] = abs(
    log_features["Coefficient"]
)

log_features = log_features.sort_values(
    "Absolute",
    ascending=False
)

print(log_features.head(15))


# ### Feature Insights – Random Forest

# In[31]:


RF_features = pd.DataFrame({
    "Feature": X.columns,
    "Importance": RF_model.feature_importances_
})

RF_features = RF_features.sort_values(
    "Importance",
    ascending=False
)

print(RF_features.head(15))


# ### Complaints Model Insights
# 
# The two models provide complementary insights, as they identify some similar complaint characteristics as important while their feature rankings differ. Random Forest can capture more complex patterns in data, while Logistic Regression provide a better view of how individual features relate to the predicted outcome.
# 
# However, these findings should be interpreted carefully because the overall predictive performance is weak. Therefore, the identified features should be viewed as patterns associated with vulnerable group cases rather than factors that directly cause them.

# In[32]:


ConfusionMatrixDisplay.from_predictions(
    Y_test,
    log_pred
)

plt.title("Logistic Regression - Complaints")
plt.show()


# In[33]:


ConfusionMatrixDisplay.from_predictions(
    Y_test,
    RF_pred
)

plt.title("Random Forest - Complaints")
plt.show()


# ## Dataset 2 – Metro Median House Sales
# 
# The Metro Median House Sales dataset is analysed to check whether an area's median property price increased based on previous-period sales volume, median price and location.

# In[34]:


# Load the prorperty sales dataset

property_df = pd.read_excel(
    r"Downloads\lsg_stats_2026_q1.xlsx"
)

property_df.head()


# In[35]:


print(property_df.shape)
print(property_df.info())
print(property_df.isnull().sum())


# In[36]:


property_df.describe()


# In[37]:


print(
    property_df["City"].value_counts()
)


# In[38]:


# Visualise distribution of median price changes

property_df["Median Change"].hist(
    bins=20
)

plt.title(
    "Distribution of Median Price Change"
)

plt.xlabel(
    "Median Change"
)

plt.ylabel(
    "Frequency"
)

plt.show()


# ### Target Definition
# 
# A binary target, `Price_Increased`, is created from Median Change. A value of 1 represents an increase in median property price, while 0 represents no increase. Records without a known Median Change are removed because their target outcome cannot be determined.

# In[39]:


property_df = property_df.dropna(
    subset=["Median Change"]
)


# In[40]:


property_df["Price_Increased"] = (
    property_df["Median Change"] > 0
).astype(int)


# In[41]:


print(
    property_df["Price_Increased"]
    .value_counts()
)


# ### Feature Selection and Missing Values
# 
# City, Sales 1Q 2025 and Median 1Q 2025 are used as predictors. Median Change is excluded because it reveals information about the target and could cause data leakage. City is converted into numerical dummy variables for modelling.
# 
# The data is divided into training and testing sets before missing values are handled. Missing sales and price values are replaced using medians calculated from the training data, preventing information from the test set to influence preprocessing.

# In[42]:


X_property = property_df[
    [
        "City",
        "Sales 1Q 2025",
        "Median 1Q 2025"
    ]
].copy()

Y_property = property_df[
    "Price_Increased"
]


# In[43]:


X_property = pd.get_dummies(
    X_property,
    columns=["City"],
    drop_first=True
)


# In[44]:


Xp_train, Xp_test, Yp_train, Yp_test = train_test_split(
    X_property,
    Y_property,
    test_size=0.20,
    random_state=42,
    stratify=Y_property
)


# In[45]:


sales_median = Xp_train["Sales 1Q 2025"].median()
price_median = Xp_train["Median 1Q 2025"].median()


# In[46]:


Xp_test["Sales 1Q 2025"] = Xp_test[
    "Sales 1Q 2025"
].fillna(sales_median)

Xp_test["Median 1Q 2025"] = Xp_test[
    "Median 1Q 2025"
].fillna(price_median)


# In[47]:


print(Xp_train.isnull().sum())
print(Xp_test.isnull().sum())


# #### Logistic Regression

# In[48]:


property_scaler = StandardScaler()

Xp_train_scaled = property_scaler.fit_transform(Xp_train)
Xp_test_scaled = property_scaler.transform(Xp_test)


# In[49]:


property_log = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)

property_log.fit(
    Xp_train_scaled,
    Yp_train
)


# #### Logistic Regression Predictions

# In[50]:


property_log_pred = property_log.predict(
    Xp_test_scaled
)

property_log_prob = property_log.predict_proba(
    Xp_test_scaled
)[:, 1]


# #### Random Forest

# In[51]:


property_RF = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42
)

property_RF.fit(
    Xp_train,
    Yp_train
)


# In[52]:


property_RF_pred = property_RF.predict(
    Xp_test
)

property_RF_prob = property_RF.predict_proba(
    Xp_test
)[:, 1]


# ### Property Model Evaluation

# In[53]:


property_log_results = {
    "Accuracy": accuracy_score(
        Yp_test,
        property_log_pred
    ),
    "Precision": precision_score(
        Yp_test,
        property_log_pred
    ),
    "Recall": recall_score(
        Yp_test,
        property_log_pred
    ),
    "F1": f1_score(
        Yp_test,
        property_log_pred
    ),
    "ROC-AUC": roc_auc_score(
        Yp_test,
        property_log_prob
    )
}


# In[54]:


property_RF_results = {
    "Accuracy": accuracy_score(
        Yp_test,
        property_RF_pred
    ),
    "Precision": precision_score(
        Yp_test,
        property_RF_pred
    ),
    "Recall": recall_score(
        Yp_test,
        property_RF_pred
    ),
    "F1": f1_score(
        Yp_test,
        property_RF_pred
    ),
    "ROC-AUC": roc_auc_score(
        Yp_test,
        property_RF_prob
    )
}


# In[55]:


property_results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest"
    ],
    "Accuracy": [
        property_log_results["Accuracy"],
        property_RF_results["Accuracy"]
    ],
    "Precision": [
        property_log_results["Precision"],
        property_RF_results["Precision"]
    ],
    "Recall": [
        property_log_results["Recall"],
        property_RF_results["Recall"]
    ],
    "F1": [
        property_log_results["F1"],
        property_RF_results["F1"]
    ],
    "ROC-AUC": [
        property_log_results["ROC-AUC"],
        property_RF_results["ROC-AUC"]
    ]
})

print(property_results)


# #### Property Model Results
# 
# Both models performed better on the property dataset than on the complaints dataset. The results are compared using accuracy, precision, recall, F1-score and ROC-AUC rather than relying on a single metric. Differences between the models show that the preferred model depends on the business objective.

# ### Feature Insights – Random Forest

# In[56]:


property_importance = pd.DataFrame({
    "Feature": X_property.columns,
    "Importance": property_RF.feature_importances_
})

property_importance = (
    property_importance
    .sort_values(
        "Importance",
        ascending=False
    )
)

print(
    property_importance.head(10)
)


# ### Feature Insights – Logistic Regression

# In[57]:


property_coefficients = pd.DataFrame({
    "Feature": X_property.columns,
    "Coefficient": property_log.coef_[0]
})

property_coefficients["Absolute"] = abs(
    property_coefficients["Coefficient"]
)

property_coefficients = (
    property_coefficients
    .sort_values(
        "Absolute",
        ascending=False
    )
)

print(
    property_coefficients.head(10)
)


# ### Property Model Insights
# 
# The two models provide complementary information about the factors associated with median-price increases. Previous median price, sales activity and location can be compared across both models to determine which features are most useful for predicting whether prices increased.
# 
# However, these findings represent predictive associations rather than direct causes of price increase. The relatively small dataset and limited time period should also be considered when interpreting the results, as they may limit how widely the results can be applied.

# In[58]:


ConfusionMatrixDisplay.from_predictions(
    Yp_test,
    property_log_pred
)

plt.title("Logistic Regression - Property")
plt.show()


# In[59]:


ConfusionMatrixDisplay.from_predictions(
    Yp_test,
    property_RF_pred
)

plt.title("Random Forest - Property")
plt.show()


# ## Evaluation Metric Justification
# 
# Accuracy, precision, recall, F1-score and ROC-AUC are used to evaluate model effectiveness.
# 
# For the complaints dataset, recall and F1-score are particularly important because the vulnerable-group class is underrepresented. Accuracy alone can be misleading if a model performs well mainly on the majority class. Recall shows how many actual vulnerable-group cases are identified, while F1-score balances recall and precision.
# 
# For the property dataset, accuracy shows overall prediction performance, while precision and recall show how well teh models identify price increase without producing too many incorrect predictions. F1-score balances these measures, and ROC-AUC evaluates how effectively each model distinguishes between increasing and non-increasing prices overall.

# ## Comparison of Data Sources and Models
# 
# Both datasets provide useful insights, but their usefulness differs. The property dataset produces stronger predictive performance, suggesting that the available property features provide more useful information for its classification target. In contrast, the complaints dataset produces weaker predictive performance, indicating that the available complaint features alone may be insufficient for reliable vulnerability classification.
# 
# The findings are complementary rather than contradictory because the datasets address different Business Analyst problems. The complaints dataset provides customer and service-related insights, while the property dataset provides market and sales-related insights. Together, they demonstrate that machine learning performance depends not only on the model but also on the relevance and quality of the available data.

# ## Limitations
# 
# The complaints dataset covers only one quarter, and the vulnerable-group status is self-reported, limiting the conclusions that can be drawn. Its available attributes also provide relatively weak predictive information for the target.
# 
# The property dataset is relatively small and covers only a limited time period, which may reduce the reliability of the results to other periods or regions. In both analysis, model outputs identify predictive relationships rather than causal relationships.
