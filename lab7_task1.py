import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =========================
# 1) CONFIG
# =========================
CWD = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
DATA_TRAIN = os.path.join(CWD, "adult.data")
DATA_TEST = os.path.join(CWD, "adult.test")
PLOTS_DIR = os.path.join(CWD, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income"
]


# =========================
# 2) LOAD DATA
# =========================
df_train = pd.read_csv(
    DATA_TRAIN,
    header=None,
    names=COLUMNS,
    sep=", ",
    engine="python"
)

# adult.test: первая строка - служебная, пропускаем
df_test = None
if os.path.exists(DATA_TEST):
    df_test = pd.read_csv(
        DATA_TEST,
        header=None,
        names=COLUMNS,
        sep=", ",
        engine="python",
        skiprows=1
    )

print("=== TRAIN SHAPE ===", df_train.shape)
if df_test is not None:
    print("=== TEST SHAPE ===", df_test.shape)

df = df_train.copy()  # EDA делаем на train (можно и на объединении)


# =========================
# 3) DATA UNDERSTANDING
# =========================
print("\n=== HEAD ===")
print(df.head())

print("\n=== DTYPES ===")
print(df.dtypes)

# Пропуски обозначены "?"
missing_q = (df == "?").sum().sort_values(ascending=False)
print("\n=== MISSING VALUES (as '?') ===")
print(missing_q[missing_q > 0])

# Сколько всего строк содержат хотя бы один "?"
rows_with_q = (df == "?").any(axis=1).sum()
print(f"\nRows with at least one '?': {rows_with_q} ({rows_with_q / len(df) * 100:.2f}%)")

# Базовая статистика числовых
print("\n=== NUMERIC DESCRIBE ===")
print(df.select_dtypes(include="number").describe())

# Распределение target
print("\n=== TARGET DISTRIBUTION (income) ===")
print(df["income"].value_counts())


# =========================
# 4) OPTIONAL: CLEAN FOR SOME PLOTS
# =========================
# Для некоторых графиков удобнее заменить "?" на NaN
df_clean = df.replace("?", pd.NA).copy()


# =========================
# 5) VISUALIZATIONS (3–5)
# =========================

# 5.1 Age histogram
plt.figure()
plt.hist(df_clean["age"].dropna(), bins=30)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "01_age_hist.png"), dpi=200)
plt.show()

# 5.2 Target distribution (income)
plt.figure()
df_clean["income"].value_counts().plot(kind="bar")
plt.title("Income Distribution")
plt.xlabel("Income")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "02_income_bar.png"), dpi=200)
plt.show()

# 5.3 Hours per week vs income (boxplot)
plt.figure()
sns.boxplot(x="income", y="hours_per_week", data=df_clean)
plt.title("Hours per Week vs Income")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "03_hours_vs_income_box.png"), dpi=200)
plt.show()

# 5.4 Correlation heatmap (numeric only)
numeric_df = df_clean.select_dtypes(include="number")
corr = numeric_df.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap (Numeric Features)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "04_corr_heatmap.png"), dpi=200)
plt.show()

# 5.5 Education vs income (countplot)
# Чтобы график был читаемым — ограничим топ-10 education или отсортируем
plt.figure(figsize=(8, 6))
sns.countplot(y="education", hue="income", data=df_clean,
              order=df_clean["education"].value_counts().index)
plt.title("Education vs Income")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "05_education_vs_income.png"), dpi=200)
plt.show()


# =========================
# 6) SUMMARY TEXT TEMPLATE
# =========================
# Это просто подсказка — в отчёт вставишь и подправишь числа/наблюдения
train_rows, train_cols = df_train.shape
miss_cols = missing_q[missing_q > 0].index.tolist()

summary = f"""
The Adult Census Income dataset contains {train_rows} records and {train_cols} columns, with a binary target variable
('income') indicating whether an individual earns more than 50K USD per year. The dataset includes a mix of numerical
features (e.g., age, education_num, capital_gain, capital_loss, hours_per_week) and categorical attributes describing
demographics and employment characteristics (e.g., workclass, occupation, marital_status, relationship, race, sex, and
native_country). Missing values are present in several categorical columns and are represented by the '?' symbol,
most notably in: {", ".join(miss_cols[:5])}.

Exploratory analysis shows that the target classes are imbalanced, with a larger proportion of individuals earning
<=50K. The age distribution is concentrated around working-age adults, and boxplot results suggest that higher-income
individuals tend to work more hours per week on average. The correlation heatmap for numeric variables indicates that
education_num and capital_gain are among the strongest positively associated features with income, while most other
numeric correlations are relatively weak. Overall, the dataset exhibits clear socioeconomic patterns that are suitable
for a classification task predicting income level.
""".strip()

print("\n=== DRAFT SUMMARY (edit to 150–200 words) ===")
print(summary)
print(f"\nWord count (approx): {len(summary.split())}")

print(f"\nSaved plots to: {PLOTS_DIR}")
