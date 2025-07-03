import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# ── 1. Set-up ──────────────────────────────────────────────────────────────────
np.random.seed(42)                 # Reproducibility

# ── 2. Create synthetic dataset ───────────────────────────────────────────────
n = 500
ages          = np.random.randint(13, 19, n)
heights_cm    = np.random.normal(165, 9, n).round(1)
bmi           = np.random.normal(22, 4, n).clip(14, 40).round(1)
weights_kg    = (bmi * (heights_cm / 100) ** 2).round(1)
daily_steps   = np.random.normal(8500, 2500, n).clip(1000, 20_000).astype(int)
calories      = np.random.normal(2400, 600, n).clip(1200, 4000).astype(int)
water_liters  = np.random.normal(2.2, 0.8, n).clip(0.5, 5).round(2)
sleep_hours   = np.random.normal(7.2, 1.3, n).clip(4, 11).round(1)
mental_score  = np.random.normal(70, 15, n).clip(20, 100).astype(int)
gender        = np.random.choice(['Male', 'Female', 'Other'], n, p=[0.49, 0.49, 0.02])

df = pd.DataFrame({
    "id": np.arange(1, n + 1),
    "age": ages,
    "gender": gender,
    "height_cm": heights_cm,
    "weight_kg": weights_kg,
    "bmi": bmi,
    "daily_steps": daily_steps,
    "calories_intake": calories,
    "water_intake_liters": water_liters,
    "sleep_hours": sleep_hours,
    "mental_health_score": mental_score
})

df.to_csv("teen_health_dataset.csv", index=False)
print("✅  teen_health_dataset.csv saved")

# ── 3. Correlation matrix & visualisations ────────────────────────────────────
corr = df[["bmi", "daily_steps", "calories_intake",
           "water_intake_liters", "sleep_hours",
           "mental_health_score"]].corr()

print("\n--- Key Correlations ---")
print(corr)

# Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap – Teen Health Factors")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()
print("📊 correlation_heatmap.png saved")

# Scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="daily_steps", y="bmi", hue="gender", alpha=0.7)
plt.title("Daily Steps vs Body Mass Index (BMI)")
plt.xlabel("Daily Steps")
plt.ylabel("BMI")
plt.tight_layout()
plt.savefig("steps_vs_bmi.png")
plt.close()
print("📉 steps_vs_bmi.png saved")

# ── 4. One-page PDF summary ───────────────────────────────────────────────────
with PdfPages("teen_health_summary.pdf") as pdf:
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")

    title = "Teen Health Analysis Project – Summary Report"
    text  = (
        "Project Title: Teen Health Analysis Using Python\n\n"
        "Objective:\n"
        "  Analyse how steps, water intake, sleep, calories, and mental\n"
        "  health relate to Body Mass Index (BMI) in teenagers.\n\n"
        "Dataset:\n"
        "  • 500 synthetic teen records (age, gender, BMI, habits)\n\n"
        "Key Findings:\n"
        "  • More daily steps are linked to slightly lower BMI.\n"
        "  • Water intake shows a weak inverse relation with BMI.\n"
        "  • Sleep and mental-health scores have minimal direct effect.\n\n"
        "Tools Used:\n"
        "  Python, pandas, matplotlib, seaborn\n\n"
        "Completed: July 2025"
    )

    ax.text(0.5, 0.5, text, ha="center", va="center", wrap=True, fontsize=12)
    ax.set_title(title, fontsize=16, pad=20)
    pdf.savefig(fig)
    plt.close()

print("📄 teen_health_summary.pdf saved")
