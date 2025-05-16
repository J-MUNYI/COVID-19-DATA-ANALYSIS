import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set visualization styles
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# Dataset Source: Our World in Data
url = "https://covid.ourworldindata.org/data/owid-covid-data.csv"

try:
    df = pd.read_csv(url)
    print("✅ Data loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load data: {e}")

# Preview the dataset
df.head()

# Drop rows without continent info (they are aggregates like 'World')
df = df.dropna(subset=["continent"])

# Convert 'date' column to datetime format
df["date"] = pd.to_datetime(df["date"])

# Focus on relevant columns for our analysis
df = df[[
    "iso_code", "continent", "location", "date",
    "total_cases", "new_cases", "total_deaths", "new_deaths", "population"
]]

# Check for missing values
print("\n🔍 Missing values per column:")
print(df.isnull().sum())

# Fill missing new_cases and new_deaths with 0
df["new_cases"].fillna(0, inplace=True)
df["new_deaths"].fillna(0, inplace=True)

# =============================
# 📊 4. Basic Statistical Summary
# =============================
print("\n📈 Summary statistics for numerical columns:")
print(df.describe())


latest_date = df["date"].max()
latest_data = df[df["date"] == latest_date]
top_countries = latest_data.sort_values("total_cases", ascending=False).head(10)

print("\n🌍 Top 10 Countries by Total COVID-19 Cases:")
print(top_countries[["location", "total_cases"]])


 # Visualizations

# 6.1 Line Chart: Global New Cases Over Time
global_daily = df.groupby("date")["new_cases"].sum()

plt.figure(figsize=(14, 6))
global_daily.plot()
plt.title("🌐 Global Daily New COVID-19 Cases")
plt.xlabel("Date")
plt.ylabel("New Cases")
plt.tight_layout()
plt.show()

# 6.2 Bar Chart: Top 10 Countries by Total Cases
plt.figure(figsize=(12, 6))
sns.barplot(data=top_countries, x="total_cases", y="location", palette="Reds_r")
plt.title("Top 10 Countries by Total COVID-19 Cases")
plt.xlabel("Total Cases")
plt.ylabel("Country")
plt.tight_layout()
plt.show()

# 6.3 Histogram: Distribution of Total Deaths (Latest Day)
plt.figure(figsize=(10, 5))
sns.histplot(data=latest_data, x="total_deaths", bins=30, kde=True, color="purple")
plt.title("Distribution of Total Deaths Across Countries")
plt.xlabel("Total Deaths")
plt.tight_layout()
plt.show()

# 6.4 Scatter Plot: Population vs Total Cases
plt.figure(figsize=(10, 6))
sns.scatterplot(data=latest_data, x="population", y="total_cases", hue="continent", alpha=0.7)
plt.title("Total COVID-19 Cases vs Population")
plt.xlabel("Population")
plt.ylabel("Total Cases")
plt.xscale("log")
plt.yscale("log")
plt.tight_layout()
plt.show()

# Conclusion and Observations
print("""
✅ Observations:
- COVID-19 cases show clear spikes globally over time.
- Countries like the US, India, and Brazil report the highest total cases.
- Larger populations tend to have more total cases, but some outliers exist.
- The distribution of deaths is skewed, with most countries under 100K.

 Next Steps:
- Analyze vaccination rates.
- Study case fatality ratios per region.
- Explore policy impacts and lockdown measures.
""")

