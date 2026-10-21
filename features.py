"""Load, clean and build features for the No-Show Prediction project."""
import pandas as pd
import numpy as np

RAW_PATH = "data/raw/KaggleV2-May-2016.csv"

# Features used by the model (same list is used in train.py and predict.py)
NUMERIC = ["age", "lead_days", "scheduled_hour", "prev_appointments",
           "prev_noshows", "prev_noshow_rate"]
BINARY = ["is_male", "scholarship", "hypertension", "diabetes", "alcoholism",
          "handicap", "sms_received", "is_first_visit", "same_day"]
CATEGORICAL = ["appt_weekday"]          # 0 = Monday ... 6 = Sunday
FEATURES = NUMERIC + BINARY + CATEGORICAL


def load_and_clean(path=RAW_PATH):
    df = pd.read_csv(path)

    # 1. Fix column names (the raw file has spelling mistakes)
    df = df.rename(columns={"Hipertension": "hypertension", "Handcap": "handicap",
                            "No-show": "no_show", "PatientId": "patient_id",
                            "AppointmentID": "appointment_id", "Age": "age",
                            "Gender": "gender", "Neighbourhood": "neighbourhood",
                            "Scholarship": "scholarship", "Diabetes": "diabetes",
                            "Alcoholism": "alcoholism", "SMS_received": "sms_received",
                            "ScheduledDay": "scheduled_day",
                            "AppointmentDay": "appointment_day"})
    df["patient_id"] = df["patient_id"].astype("int64")

    # 2. Dates
    df["scheduled_day"] = pd.to_datetime(df["scheduled_day"]).dt.tz_localize(None)
    df["appointment_day"] = pd.to_datetime(df["appointment_day"]).dt.tz_localize(None)

    # 3. Target: 1 = patient did NOT show up
    df["no_show"] = (df["no_show"] == "Yes").astype(int)

    # 4. Lead time in whole days (appointment time is not recorded, only the date)
    df["lead_days"] = (df["appointment_day"].dt.normalize()
                       - df["scheduled_day"].dt.normalize()).dt.days

    # 5. Remove impossible rows (negative age, appointment before booking)
    df = df[(df["age"] >= 0) & (df["lead_days"] >= 0)].copy()

    return df.reset_index(drop=True)


def build_features(df):
    df = df.copy()

    df["is_male"] = (df["gender"] == "M").astype(int)
    df["handicap"] = (df["handicap"] > 0).astype(int)      # 0-4 -> 0/1
    df["scheduled_hour"] = df["scheduled_day"].dt.hour
    df["appt_weekday"] = df["appointment_day"].dt.dayofweek
    df["same_day"] = (df["lead_days"] == 0).astype(int)

    # ---- Patient history (uses ONLY earlier appointments -> no data leakage) ----
    df = df.sort_values(["patient_id", "appointment_day", "scheduled_day"])
    g = df.groupby("patient_id")
    df["prev_appointments"] = g.cumcount()
    df["prev_noshows"] = g["no_show"].cumsum() - df["no_show"]
    df["prev_noshow_rate"] = (df["prev_noshows"] / df["prev_appointments"]).fillna(0)
    df["is_first_visit"] = (df["prev_appointments"] == 0).astype(int)
    df = df.sort_index()

    # Bucket for charts / EDA
    df["lead_bucket"] = pd.cut(df["lead_days"], [-1, 0, 2, 7, 14, 30, 400],
                               labels=["Same day", "1-2 days", "3-7 days",
                                       "8-14 days", "15-30 days", "30+ days"])
    return df
