"""Score ONE booking. Run from the project folder:  python src/predict.py
Use predict_no_show() inside your own app (for example a booking system backend).
"""
import sys
import joblib
import pandas as pd

sys.path.insert(0, "src")

_bundle = joblib.load("models/no_show_model.joblib")


def predict_no_show(booking: dict) -> dict:
    """booking keys: age, gender ('M'/'F'), booked_at, appointment_date (YYYY-MM-DD),
    scholarship, hypertension, diabetes, alcoholism, handicap, sms_received (0/1),
    prev_appointments, prev_noshows (the patient's history BEFORE this booking)."""
    booked = pd.Timestamp(booking["booked_at"])
    appt = pd.Timestamp(booking["appointment_date"])
    lead = (appt.normalize() - booked.normalize()).days
    prev_a = booking.get("prev_appointments", 0)
    prev_n = booking.get("prev_noshows", 0)

    row = {
        "age": booking["age"], "lead_days": lead, "scheduled_hour": booked.hour,
        "prev_appointments": prev_a, "prev_noshows": prev_n,
        "prev_noshow_rate": prev_n / prev_a if prev_a else 0,
        "is_male": int(booking["gender"] == "M"),
        "scholarship": booking.get("scholarship", 0),
        "hypertension": booking.get("hypertension", 0),
        "diabetes": booking.get("diabetes", 0),
        "alcoholism": booking.get("alcoholism", 0),
        "handicap": int(booking.get("handicap", 0) > 0),
        "sms_received": booking.get("sms_received", 0),
        "is_first_visit": int(prev_a == 0),
        "same_day": int(lead == 0),
        "appt_weekday": appt.dayofweek,
    }
    X = pd.DataFrame([row])[_bundle["features"]]
    score = float(_bundle["pipeline"].predict_proba(X)[0, 1])
    thr = _bundle["threshold"]

    if score >= thr + 0.10:
        level, action = "HIGH", "Call the patient and confirm; consider a back-up booking."
    elif score >= thr:
        level, action = "MEDIUM", "Send an extra SMS reminder 1 day before."
    else:
        level, action = "LOW", "Normal reminder is enough."
    return {"risk_score": round(score, 3), "risk_level": level, "suggested_action": action}


if __name__ == "__main__":
    example_high = {"age": 22, "gender": "M", "booked_at": "2016-04-01 10:15",
                    "appointment_date": "2016-05-20", "sms_received": 1,
                    "prev_appointments": 3, "prev_noshows": 2}
    example_low = {"age": 68, "gender": "F", "booked_at": "2016-05-20 08:30",
                   "appointment_date": "2016-05-20", "sms_received": 0,
                   "prev_appointments": 4, "prev_noshows": 0, "hypertension": 1}
    print("Booking 1 (young, booked 7 weeks ahead, 2 past no-shows):", predict_no_show(example_high))
    print("Booking 2 (68 yrs, same-day booking, no past no-shows):  ", predict_no_show(example_low))
