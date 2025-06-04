import json
import sys
from statistics import mean, stdev

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

try:
    with open('patientdata.json', 'r') as f:
        #json.load() reads the entire JSON document from the file, since the root is a JSON array, it returns a Python list
        data_list = json.load(f)
        print("Number of items in the array:", len(data_list))
except FileNotFoundError:
    print("No patientdata.json found, please go to the website and download the json, then place it in the root directory")
    sys.exit(1)

patient_records = []

i = 0;
for doc in data_list["documents"]:
    i+=1
    record = {
        "patient_id": i,
        "document_id": doc["document_id"],
        "date": doc["date"],
        "sleep_interruptions": doc["sleep"]["interruptions"],
        "sleep_duration": doc["sleep"]["duration_hours"],
        "sleep_quality": doc["sleep"]["quality"],
        "temp_mean": mean(doc["vitals"]["temperature"]),
        "bp_avg_sys": round(mean(int(bp.split('/')[0]) for bp in doc["vitals"]["blood_pressure"])),
        "bp_avg_dia": round(mean(int(bp.split('/')[1]) for bp in doc["vitals"]["blood_pressure"])),
        "heart_rate_mean": mean(doc["vitals"]["heart_rate"]),
        "heart_rate_std": stdev(doc["vitals"]["heart_rate"]),
        "calories": doc["nutrition"]["calories"],
        "protein": doc["nutrition"]["macros"]["protein_g"],
        "carbs": doc["nutrition"]["macros"]["carbs_g"],
        "fat": doc["nutrition"]["macros"]["fat_g"],
        "water": doc["nutrition"]["water_oz"],
        "steps": doc["activity"]["steps"],
        "active_minutes": doc["activity"]["active_minutes"],
        "sedentary_hours": doc["activity"]["sedentary_hours"]
    }
    patient_records.append(record)

#Convert to DataFrame
df = pd.DataFrame(patient_records)
print(df.head())

#One-hot encode sleep_quality
df = pd.concat([df.drop("sleep_quality", axis=1),
                #df.drop("document_id", axis=1),
                pd.get_dummies(df["sleep_quality"], prefix="sleep")], axis=1)

#Apply labeling function
def assign_label(row):
    good_conditions = [
        row["sleep_duration"] >= 7,
        #97.5 <= row["temp_mean"] <= 99.0,
        row["bp_avg_sys"] < 125,
        row["bp_avg_dia"] < 85,
        row["heart_rate_mean"] < 75,
        row["active_minutes"] >= 45,
        row["steps"] >= 8000
    ]
    violations = sum(not cond for cond in good_conditions)
    if violations == 0:
        return "Good"
    if violations <= 2:
        return "Moderate"
    return "Poor"

print(df.head())  #confirm df is defined
df['health_status'] = df.apply(assign_label, axis=1)
df.to_csv("patient_data.csv", index=False)

package = []

for i, doc in df.iterrows():
    #i += 1
    record = {
        "patient_id": i+1,
        "document_id": doc['document_id'],
        "health_status": doc['health_status'],
        "date": doc['date'],
    }
    package.append(record)

#Smaller csv file generated
pf = pd.DataFrame(package)
pf.to_csv("patient_health_label.csv", index=False)

features = df.drop(["date", "health_status", "document_id"], axis=1)
label_enc = LabelEncoder()
labels = label_enc.fit_transform(df["health_status"])  # 0=Good,1=Moderate,2=Poor

#Define Stratified K-Fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#Decision Tree Grid Search
param_grid_dt = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 7],
    'min_samples_split': [5, 10]
}
grid_dt = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid_dt,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1
)
grid_dt.fit(features, labels)
best_dt = grid_dt.best_estimator_

#Random Forest Grid Search
param_grid_rf = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10],
    'min_samples_split': [5, 10]
}
grid_rf = GridSearchCV(
    RandomForestClassifier(random_state=42, class_weight='balanced'),
    param_grid_rf,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1
)
grid_rf.fit(features, labels)
best_rf = grid_rf.best_estimator_

#Cross-validated predictions
y_pred_dt = cross_val_predict(best_dt, features, labels, cv=cv, n_jobs=-1)
y_pred_rf = cross_val_predict(best_rf, features, labels, cv=cv, n_jobs=-1)

#Evaluation
print("Decision Tree Classification Report:")
print(classification_report(labels, y_pred_dt, target_names=label_enc.classes_))
print("\nRandom Forest Classification Report:")
print(classification_report(labels, y_pred_rf, target_names=label_enc.classes_))