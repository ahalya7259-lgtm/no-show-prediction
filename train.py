"""Train 3 models, compare them, choose a threshold and save the best model.
Run from the project folder:  python src/train.py
"""
import sys, json
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score, roc_curve, confusion_matrix,
                             ConfusionMatrixDisplay)

sys.path.insert(0, "src")
from features import load_and_clean, build_features, FEATURES, NUMERIC, BINARY, CATEGORICAL

sns.set_theme(style="whitegrid")
SEED = 42

# ---------------- data ----------------
df = build_features(load_and_clean())
X, y = df[FEATURES], df["no_show"]

# 70% train / 15% validation (choose model + threshold) / 15% test (final report)
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)
print(f"Train {len(X_train):,} | Validation {len(X_val):,} | Test {len(X_test):,}")
print(f"No-show rate: train {y_train.mean():.3f}, test {y_test.mean():.3f}")


def make_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("bin", "passthrough", BINARY),
    ])


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=50, max_depth=12,
                                            class_weight="balanced_subsample", n_jobs=-1, random_state=SEED),
    "Gradient Boosting": HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=200,
                                                        class_weight="balanced", random_state=SEED),
}


def scores(y_true, prob, thr=0.5):
    pred = (prob >= thr).astype(int)
    return {"precision": precision_score(y_true, pred), "recall": recall_score(y_true, pred),
            "f1": f1_score(y_true, pred), "roc_auc": roc_auc_score(y_true, prob),
            "pr_auc": average_precision_score(y_true, prob)}


# ---------------- baselines ----------------
base_pred = np.zeros(len(y_test), dtype=int)
print("\nBaseline 1 (everyone shows up): accuracy "
      f"{(base_pred == y_test).mean():.3f}, recall 0.000  <- useless for catching no-shows")
rule = (X_test["lead_days"] >= 8).astype(int)
rule_res = {"precision": precision_score(y_test, rule), "recall": recall_score(y_test, rule),
            "f1": f1_score(y_test, rule)}
print("Baseline 2 (rule: booked 8+ days ahead = risky):", {k: round(v, 3) for k, v in rule_res.items()})

# ---------------- train + compare on validation ----------------
fitted, val_results = {}, {}
plt.figure(figsize=(6, 5))
for name, clf in models.items():
    pipe = Pipeline([("prep", make_preprocessor()), ("model", clf)]).fit(X_train, y_train)
    fitted[name] = pipe
    prob = pipe.predict_proba(X_val)[:, 1]
    val_results[name] = scores(y_val, prob)
    fpr, tpr, _ = roc_curve(y_val, prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC {val_results[name]['roc_auc']:.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlabel("False positive rate"); plt.ylabel("True positive rate (recall)")
plt.title("ROC curves (validation set)"); plt.legend()
plt.tight_layout(); plt.savefig("images/05_roc_curves.png", dpi=150); plt.close()

print("\nValidation results (threshold 0.5):")
for n, r in val_results.items():
    print(f"  {n:20s}", {k: round(v, 3) for k, v in r.items()})

best_name = max(val_results, key=lambda n: val_results[n]["roc_auc"])
best = fitted[best_name]
print("\nBest model (highest ROC-AUC):", best_name)

# ---------------- choose threshold on validation ----------------
val_prob = best.predict_proba(X_val)[:, 1]
rows = []
for thr in np.arange(0.30, 0.71, 0.05):
    pred = (val_prob >= thr).astype(int)
    rows.append((round(thr, 2), precision_score(y_val, pred), recall_score(y_val, pred),
                 f1_score(y_val, pred), pred.mean()))
print("\nThreshold | precision | recall | f1 | % flagged")
for r in rows:
    print(f"  {r[0]:.2f}   |  {r[1]:.3f}   | {r[2]:.3f}  | {r[3]:.3f} | {r[4]*100:.0f}%")
best_thr = max(rows, key=lambda r: r[3])[0]
print("Chosen threshold (best F1 on validation):", best_thr)

# ---------------- final test evaluation ----------------
test_prob = best.predict_proba(X_test)[:, 1]
final = scores(y_test, test_prob, best_thr)
print("\nFINAL TEST RESULTS:", {k: round(v, 3) for k, v in final.items()})

# test results of all models (fair comparison table for README)
test_all = {n: scores(y_test, p.predict_proba(X_test)[:, 1]) for n, p in fitted.items()}

cm = confusion_matrix(y_test, (test_prob >= best_thr).astype(int))
ConfusionMatrixDisplay(cm, display_labels=["Showed up", "No-show"]).plot(cmap="Blues", values_format="d")
plt.title(f"Confusion matrix - {best_name} (threshold {best_thr})")
plt.tight_layout(); plt.savefig("images/06_confusion_matrix.png", dpi=150); plt.close()

# lift: how much better than random are the top-risk bookings?
order = np.argsort(-test_prob)
top = int(len(order) * 0.2)
top_rate = y_test.iloc[order[:top]].mean()
print(f"Top 20% riskiest bookings: {top_rate*100:.1f}% are no-shows (overall {y_test.mean()*100:.1f}%)")

# feature importance (permutation, on a sample of the test set)
idx = np.random.RandomState(SEED).choice(len(X_test), 8000, replace=False)
imp = permutation_importance(best, X_test.iloc[idx], y_test.iloc[idx], scoring="roc_auc",
                             n_repeats=5, random_state=SEED, n_jobs=-1)
imp_s = (__import__("pandas").Series(imp.importances_mean, index=FEATURES).sort_values())
plt.figure(figsize=(7, 5))
imp_s.tail(10).plot(kind="barh", color="#2E75B6")
plt.title("Top 10 features (drop in ROC-AUC when shuffled)"); plt.xlabel("Importance")
plt.tight_layout(); plt.savefig("images/07_feature_importance.png", dpi=150); plt.close()
print("\nTop features:", list(imp_s.sort_values(ascending=False).head(6).round(4).items()))

# ---------------- save ----------------
joblib.dump({"pipeline": best, "threshold": best_thr, "features": FEATURES, "model_name": best_name},
            "models/no_show_model.joblib")
json.dump({"best_model": best_name, "threshold": best_thr,
           "test_results": {k: round(v, 4) for k, v in final.items()},
           "all_models_test": {n: {k: round(v, 4) for k, v in r.items()} for n, r in test_all.items()},
           "baseline_rule": {k: round(v, 4) for k, v in rule_res.items()},
           "top20pct_noshow_rate": round(float(top_rate), 4)},
          open("models/metrics.json", "w"), indent=2)
print("\nSaved: models/no_show_model.joblib, models/metrics.json, charts in images/")
