# No-Show Prediction for Appointments

Machine learning project that predicts which appointments are likely to be **no-shows**, so a clinic (or salon, tutor, etc.) can send extra reminders, call high-risk patients, or free the slot early.

**Data:** [Medical Appointment No-Shows (Kaggle)](https://www.kaggle.com/datasets/joniarroba/noshowappointments) - 110,527 appointments from Brazil, 2016 (110,521 after cleaning).
**Target:** `no_show` = 1 if the patient did not come.

## Business Problem
About **1 in 5 appointments (20.2%)** ends in a no-show. Every empty slot is lost money and wasted staff time. Sending an extra reminder to *everyone* costs time and money. A prediction model lets the clinic focus on the **riskiest bookings only**.

## Tools Used
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Joblib, Jupyter Notebook

## Project Structure
```
no-show-prediction/
├── data/
│   ├── raw/KaggleV2-May-2016.csv
│   └── processed/
├── notebooks/
│   └── 01_eda_and_model.ipynb    EDA + modeling + conclusions
├── src/
│   ├── features.py               cleaning + feature engineering
│   ├── eda.py                    saves EDA charts
│   ├── train.py                  trains 3 models, saves the best
│   └── predict.py                scores one new booking
├── models/                       no_show_model.joblib, metrics.json
├── images/                       charts
├── requirements.txt
└── README.md
```

## What I Did
1. **Cleaning** - fixed column spelling, converted dates, removed 6 impossible rows (negative age, appointment before booking date).
2. **Feature engineering** - lead time (days between booking and appointment), booking hour, weekday, same-day flag, and **patient history** (previous appointments, previous no-shows, previous no-show rate).
3. **No data leakage** - history features use **only earlier appointments** of the same patient.
4. **Split** - 70% train, 15% validation (choose model and threshold), 15% test (final score).
5. **Models** - Logistic Regression, Random Forest, Gradient Boosting, compared with simple baselines.
6. **Explain** - feature importance to show *why* a booking is risky.
7. **Threshold** - chose the risk cutoff using validation data.

## EDA Findings
| Finding | Number |
|---------|--------|
| Overall no-show rate | 20.2% |
| Same-day bookings | only 4.6% no-show |
| Booked 15-30 days ahead | 32.6% no-show |
| Booked 30+ days ahead | 33.0% no-show |
| Age 13-25 | 25.7% no-show |
| Age 60+ | 15.2% no-show |
| Patients with 0 previous no-shows | 18.7% |
| Patients with 3 previous no-shows | 36.5% |
| Patients with 4 previous no-shows | 43.0% |

![Lead time](images/01_noshow_by_lead_time.png)
![History](images/03_noshow_by_history.png)

**Note on SMS:** patients who got an SMS no-show *more* (27.6% vs 16.7%). This does not mean SMS causes no-shows. SMS is never sent for same-day bookings, which are the safest group. Proving the real effect of reminders needs an A/B test.

## Model Results
Comparison on the test set (threshold 0.5):

| Model | Precision | Recall | F1 | ROC-AUC |
|-------|-----------|--------|----|---------|
| Logistic Regression | 0.303 | 0.855 | 0.447 | 0.732 |
| Random Forest | 0.305 | 0.835 | 0.447 | 0.741 |
| **Gradient Boosting** | 0.309 | 0.825 | 0.450 | **0.742** |
| Baseline rule (lead time 8+ days) | 0.320 | 0.572 | 0.410 | - |
| Baseline (everyone shows up) | - | 0.000 | - | - |

**Chosen model:** Gradient Boosting with threshold **0.55** (best F1 on validation).

**Final test result:** Precision 0.330 | Recall 0.705 | F1 0.450 | ROC-AUC 0.742

![ROC](images/05_roc_curves.png)
![Feature importance](images/07_feature_importance.png)

**Most important features:** lead time (by far), age, previous no-show rate.

### How good is it?
- It catches about **70% of real no-shows**, but about 2 out of 3 flagged bookings are false alarms (precision 0.33). This is normal for this dataset.
- The **top 20% riskiest bookings have a 40.1% no-show rate, double the overall 20.2%**. So reminders can be focused on a small group.
- It beats the "everyone shows up" baseline (which catches none) and the simple lead-time rule.

## Business Recommendations
1. Send a **second reminder** for bookings made more than 2 weeks ahead.
2. **Call to confirm** patients with previous no-shows and a high risk score.
3. Consider **light overbooking** only for slots marked high risk.
4. Run an **A/B test** (reminder vs no reminder) to measure the true effect of SMS.

## Limitations
- Data is from one region in 2016, so results may not match other clinics.
- Only the appointment date is available, not the time of day.
- The score is a *risk ranking*, not an exact probability (class weights change the scale).
- Predictions should help staff decide on reminders, not be used to deny service to anyone.

## How to Run
```bash
git clone https://github.com/ahalya7259-lgtm/no-show-prediction.git
cd no-show-prediction
pip install -r requirements.txt
python src/eda.py        # EDA charts
python src/train.py      # train, compare, save model
python src/predict.py    # score two example bookings
```
Or open `notebooks/01_eda_and_model.ipynb` in Jupyter Notebook and run all cells.

## Example Prediction
```python
from src.predict import predict_no_show
predict_no_show({"age": 22, "gender": "M", "booked_at": "2016-04-01 10:15",
                 "appointment_date": "2016-05-20", "sms_received": 1,
                 "prev_appointments": 3, "prev_noshows": 2})
# {'risk_score': 0.837, 'risk_level': 'HIGH', 'suggested_action': 'Call the patient and confirm...'}
```

## Future Work
- Probability calibration and a cost-based threshold
- FastAPI endpoint `POST /predict-no-show`
- Integrate with an appointment booking system to flag high-risk bookings

## Author
**Ahalya** - BCA Graduate | Aspiring Data Analyst
LinkedIn: linkedin.com/in/ahalya-k-096a6a30a | GitHub: github.com/ahalya7259-lgtm
