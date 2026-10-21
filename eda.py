"""Exploratory charts. Run from the project folder:  python src/eda.py"""
import sys
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.insert(0, "src")
from features import load_and_clean, build_features

sns.set_theme(style="whitegrid")
df = build_features(load_and_clean())
overall = df["no_show"].mean() * 100
print(f"Appointments: {len(df):,} | Overall no-show rate: {overall:.1f}%")


def rate_chart(series, title, file, xlabel="", rot=0, color="#2E75B6"):
    r = df.groupby(series, observed=True)["no_show"].mean() * 100
    plt.figure(figsize=(7, 4.2))
    ax = r.plot(kind="bar", color=color)
    plt.axhline(overall, color="red", ls="--", lw=1, label=f"Overall {overall:.1f}%")
    for i, v in enumerate(r):
        ax.text(i, v + 0.4, f"{v:.0f}%", ha="center", fontsize=9)
    plt.title(title); plt.ylabel("No-show rate (%)"); plt.xlabel(xlabel)
    plt.xticks(rotation=rot); plt.legend()
    plt.tight_layout(); plt.savefig(f"images/{file}", dpi=150); plt.close()


rate_chart("lead_bucket", "No-show rate by lead time (booking to appointment)", "01_noshow_by_lead_time.png", "Lead time")
rate_chart(
    __import__("pandas").cut(df["age"], [-1, 12, 25, 40, 60, 120],
                             labels=["0-12", "13-25", "26-40", "41-60", "60+"]),
    "No-show rate by age group", "02_noshow_by_age.png", "Age group", color="#ED7D31")
rate_chart(df["prev_noshows"].clip(upper=3).map({0: "0", 1: "1", 2: "2", 3: "3+"}),
           "No-show rate by number of PREVIOUS no-shows", "03_noshow_by_history.png",
           "Previous no-shows", color="#C0392B")
rate_chart(df["sms_received"].map({0: "No SMS", 1: "SMS sent"}),
           "No-show rate: SMS reminder vs none", "04_noshow_by_sms.png", color="#70AD47")
print("EDA charts saved in images/")
