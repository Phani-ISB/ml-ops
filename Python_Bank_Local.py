# ========================================================================
# BANK‑MARKETING TERM‑DEPOSIT – END‑TO‑END PIPELINE (CRISP‑ML(Q))
# ========================================================================
# Phase 1.a – BUSINESS UNDERSTANDING
# ------------------------------------------------------------------------
#  • Business Problem:  Target only customers likely to subscribe so that
#    call‑centre / e‑mail spend is reduced without losing conversions.
#  • Objective:         Flag “hot leads” (y = "yes") with high recall
#                       while keeping false positives affordable.
#  • Constraints:       Contact‑cost budget, regulator’s contact limits.
#  • Success metrics:   – Business: ≥ 30 % lift vs random targeting
#                       – ML:       ≥ 0.85 ROC‑AUC on hold‑out
#                       – Economic: ≥ 15 % lower cost‑per‑conversion
#
# Phase 1.b – DATA UNDERSTANDING
# ------------------------------------------------------------------------
#  • Source:  `bank-additional.csv`  (UCI; semicolon‑delimited)
#  • Size:    41 188 rows × 21 columns
#  • Target:  `y`  {“yes”, “no”}
#  • Features: 10 numeric + 10 categorical (job, marital, …, month, …)
# ========================================================================

# --------------------  REQUIRED LIBS  -----------------------------------
# pip install shap feature-engine imbalanced-learn pymysql dtale
import os, pickle, joblib, shap, dtale
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt

from sqlalchemy import create_engine
from urllib.parse import quote

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from feature_engine.outliers import Winsorizer
from imblearn.over_sampling import SMOTE

# --------------------  PHASE 2 – DATA COLLECTION & LOADING  -------------
DATA_PATH = r"C:/path/to/bank-additional.csv"    # adjust
bank_df   = pd.read_csv(DATA_PATH, sep=';', na_values=['unknown'])

# optional – push to MySQL (governance demo)
user, pw, db = 'user1', quote('user1'), 'bank_mktg_db'
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")
bank_df.to_sql('raw_campaign', con=engine, if_exists='replace', index=False)

# --------------------  QUICK EDA SNAPSHOT -------------------------------
d = dtale.show(bank_df, ignore_duplicate=True)
# d.open_browser()

# --------------------  PHASE 3 – DATA PREPARATION  ----------------------
# ---- 3.1  Target recoding
bank_df['y'] = bank_df['y'].map({'no': 'Non-subscriber', 'yes': 'Subscriber'})

# ---- 3.2  Feature / target split
X = bank_df.drop(columns=['y'])
y = bank_df['y']

# ---- 3.3  Column groups
numeric_feats     = X.select_dtypes(exclude=['object']).columns.tolist()
categorical_feats = X.select_dtypes(include=['object']).columns.tolist()

# ---- 3.4  Pre‑processing pipelines
num_pipe = Pipeline([
    ('impute',    SimpleImputer(strategy='median')),
    ('winsorize', Winsorizer(capping_method='iqr', tail='both', fold=1.5)),
    ('scale',     MinMaxScaler())
])

cat_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('encode', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

preprocess = ColumnTransformer([
    ('num', num_pipe, numeric_feats),
    ('cat', cat_pipe, categorical_feats)
])

# ---- 3.5  Feature selection (retain 25 best)
selector = SelectKBest(score_func=f_classif, k=25)

full_pipe = Pipeline([
    ('prep', preprocess),
    ('fs',   selector)
])

X_selected = full_pipe.fit_transform(X, y)

# Persist for inference pipeline reuse
joblib.dump(full_pipe, 'bank_marketing_prep_pipeline.joblib')

# --------------------  PHASE 4 – DATA SPLIT & BALANCING -----------------
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.20, random_state=42, stratify=y
)

smote = SMOTE(random_state=42, sampling_strategy=0.4)   # keep class balance realistic
X_train, y_train = smote.fit_resample(X_train, y_train)

# --------------------  PHASE 5 – MODEL BUILDING & TUNING  ---------------
knn_base  = KNeighborsClassifier()
param_grid = {'n_neighbors': list(range(3, 40, 2))}

grid  = GridSearchCV(knn_base, param_grid, cv=5, scoring='roc_auc',
                     verbose=2, n_jobs=-1)
grid.fit(X_train, y_train)
best_knn = grid.best_estimator_
pickle.dump(best_knn, open('bank_marketing_knn.pkl', 'wb'))

# --------------------  PHASE 6 – EVALUATION -----------------------------
print(f"Best k: {grid.best_params_['n_neighbors']}")
print(f"CV ROC‑AUC: {grid.best_score_:.4f}")

y_proba = best_knn.predict_proba(X_test)[:, 1]
roc     = roc_auc_score(y_test.map({'Non-subscriber':0, 'Subscriber':1}), y_proba)
print(f"Hold‑out ROC‑AUC: {roc:.4f}")

y_pred  = best_knn.predict(X_test)
cm      = confusion_matrix(y_test, y_pred, labels=['Non-subscriber','Subscriber'])
ConfusionMatrixDisplay(cm,
                       display_labels=['Non-sub.', 'Sub.']).plot()
plt.title('Bank‑Marketing – Confusion Matrix')
plt.show()

# --------------------  PHASE 7 – EXPLAINABILITY (SHAP) ------------------
# Build post‑transform feature name list
prep_only   = full_pipe.named_steps['prep']
cat_names   = prep_only.named_transformers_['cat'] \
                 .named_steps['encode'].get_feature_names_out(categorical_feats)
num_names   = numeric_feats
feature_names = np.concatenate([num_names, cat_names])[selector.get_support()]

sample_X = pd.DataFrame(X_test, columns=feature_names)

explainer   = shap.KernelExplainer(best_knn.predict_proba, sample_X.sample(200, random_state=42))
shap_values = explainer.shap_values(sample_X)

shap.summary_plot(shap_values, sample_X, show=False)
plt.title('SHAP Summary – Drivers of Subscription')
plt.show()

# --------------------  OUTPUT ARTEFACTS ---------------------------------
#   • bank_marketing_prep_pipeline.joblib   (pre‑processing + FS)
#   • bank_marketing_knn.pkl                (optimised classifier)
#   • SHAP summary plot                     (feature importance)
# Ready for MLflow/DVC tracking and deployment exactly like your turbine
# example – e.g. wrap with FastAPI, containerise, push to AWS ECR/ECS.
