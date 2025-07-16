import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def create_features(df):
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    # lag features (admissions previous day)
    df['adm_lag1'] = df['admissions'].shift(1).fillna(method='bfill')
    return df

def train_icu_classifier(df):
    df = create_features(df)
    X = df[['admissions', 'day_of_week', 'month', 'adm_lag1']]
    y = (df['icu_admissions'] > 0).astype(int)  # ICU admission yes/no

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    print("Classification Report:")
    print(classification_report(y_test, preds))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))

    return clf
import joblib

if __name__ == "__main__":
    df = pd.read_csv('hospital_data.csv', parse_dates=['date'])
    clf = train_icu_classifier(df)

    # Save the trained model for later use
    joblib.dump(clf, 'icu_classifier.pkl')
    print("Saved ICU classifier model to icu_classifier.pkl")


if __name__ == "__main__":
    df = pd.read_csv('hospital_data.csv', parse_dates=['date'])
    clf = train_icu_classifier(df)