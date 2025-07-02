# ========================================================================
# BANK‑MARKETING TERM‑DEPOSIT – END‑TO‑END PIPELINE (CRISP‑ML(Q))
# ========================================================================
# Phase 1.a – BUSINESS UNDERSTANDING
# ------------------------------------------------------------------------
#  • Business Problem:  Identify the right customers to target to accept offers (like Term Deposit) 
#  • Objective:         Flag 'Subscribers' & 'Non subscribers' with probabilities
#  • Success metrics:   – ML:       ≥ 0.85 ROC‑AUC on hold‑out
#                       
#
# Phase 1.b – DATA UNDERSTANDING
# ------------------------------------------------------------------------
#  • Source:  `bank-additional.csv`
#  • Size:    4119 rows × 21 columns
#  • Target:  `Subscriber`  {“yes”, “no”}
#  • Features: 10 numeric + 10 categorical (job, marital, …, month, …)
# ========================================================================

# --------------------  REQUIRED LIBS  -----------------------------------
#pip install shap feature-engine imbalanced-learn pymysql dtale
import os, pickle, joblib, shap, dtale
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt

from sqlalchemy import create_engine
from urllib.parse import quote

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, make_scorer
from feature_engine.outliers import Winsorizer
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE


# --------------------  PHASE 2 – DATA COLLECTION & LOADING  -------------
DATA_PATH = "https://raw.githubusercontent.com/Phani-ISB/ml-ops/refs/heads/main/bank-additional.csv"    # Using dataset uploaded into github
bank_df   = pd.read_csv(DATA_PATH, sep=',', na_values=['unknown'])

# MySQL data

from urllib.parse import quote_plus
from sqlalchemy import create_engine
from sqlalchemy import types

user     = "root"
password = quote_plus("Manasa@123")
host     = "127.0.0.1"
port     = 3306
db       = "bank_mktg_db"

engine = create_engine(
    f"mysql+pymysql://{user}:{password}@{host}:{port}/{db}",
    pool_pre_ping=True,          # drops broken connections
    connect_args={"ssl": {"ssl_disabled": True}}  # toggle if TLS needed
)
bank_df.to_sql('raw_campaign', con=engine, if_exists='replace', index=False)

# --------------------  QUICK EDA SNAPSHOT -------------------------------
d = dtale.show(bank_df, ignore_duplicate=True)
d.open_browser()

# --------------------  PHASE 3 – DATA PREPARATION  ----------------------
# ---- 3.1  Target recoding
bank_df['y'] = bank_df['y'].map({'no': 'Non-subscriber', 'yes': 'Subscriber'})

# ---- 3.2  Feature / target split
X = bank_df.drop(columns=['y'])
y = bank_df['y'].dropna()

# ---- 3.3  Column groups
numeric_feats     = X.select_dtypes(exclude=['object']).columns.tolist()
numeric_feats = [col for col in numeric_feats 
                 if col not in ('pdays', 'previous')]  # Based on EDA, dropping columns with no variation
categorical_feats = X.select_dtypes(include=['object']).columns.tolist()


# ---- 3.4  Pre‑processing pipelines
num_pipe = Pipeline([
    ('impute',    SimpleImputer(strategy='mean')),
    ('winsorize', Winsorizer(capping_method='iqr', tail='both', fold=1.5)),
    ('scale',     StandardScaler())
])

cat_pipe = Pipeline([
        ('encode', OneHotEncoder(drop= None, handle_unknown='ignore', sparse_output=False))
])

preprocess = ColumnTransformer([
    ('num', num_pipe, numeric_feats),
    ('cat', cat_pipe, categorical_feats)
])

# ---- 3.5  Feature selection
selector = SelectKBest(score_func=f_classif, k=15)

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
#For keeeping class balance using SMOTE
smote = SMOTE(random_state=42, sampling_strategy='auto')
X_train, y_train = smote.fit_resample(X_train, y_train)


# --------------------  PHASE 5 – MODEL BUILDING & TUNING  ---------------
knn_base  = KNeighborsClassifier()
param_grid = {'n_neighbors': list(range(5, 31, 2))} 

# Custom scorers (let you tune pos_label / average / beta, etc.)
precision_scorer = make_scorer(precision_score, pos_label='Subscriber', average='binary')
recall_scorer    = make_scorer(recall_score,pos_label='Subscriber', average='binary')
f1_scorer        = make_scorer(f1_score,pos_label='Subscriber', average='binary')

scoring = {'accuracy' : 'accuracy' , 'precision' : precision_scorer, 
           'roc_auc' : 'roc_auc', 'recall' : recall_scorer, 'f1_score' : f1_scorer}

grid  = GridSearchCV(knn_base, param_grid, cv=5, scoring= scoring, refit = 'accuracy', n_jobs=-1)

grid.fit(X_train, y_train)
best_knn = grid.best_estimator_

results = pd.DataFrame(grid.cv_results_)
best = grid.best_index_
best_fit = results.loc[best,['params','mean_test_f1_score','mean_test_recall',
                             'mean_test_accuracy','mean_test_precision','mean_test_roc_auc']]

results

#Saving model into pickle file
pickle.dump(best_knn, open('bank_marketing_knn.pkl', 'wb'))

# --------------------  PHASE 6 – EVALUATION -----------------------------

print(best_fit)
y_proba = best_knn.predict_proba(X_test)[:, 1]
roc     = roc_auc_score(y_test.map({'Non-subscriber':0, 'Subscriber':1}), y_proba)
print(f"Hold‑out ROC‑AUC: {roc:.4f}")


y_pred  = best_knn.predict(X_test)
cm      = confusion_matrix(y_test, y_pred, labels=['Subscriber','Non-subscriber'])

# Confusion Matrix display
ConfusionMatrixDisplay(cm,
                       display_labels=['Sub.', 'Non-Sub.']).plot()
plt.title('Bank‑Marketing – Confusion Matrix')
plt.show()
