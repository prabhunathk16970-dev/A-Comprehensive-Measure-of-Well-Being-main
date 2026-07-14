import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ===========================
# Load Dataset
# ===========================

df = pd.read_csv("dataset/hdi.csv")

# Remove duplicate header row
df = df[df["Country"] != "Country"]

# Convert columns to numeric
numeric_columns = [
    "HDI",
    "Life_Expectancy",
    "Expected_Schooling",
    "Mean_Schooling",
    "GNI_Per_Capita"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove missing values
df = df.dropna()

print(df.head())

print("\nDataset Shape:", df.shape)

# ===========================
# Visualization
# ===========================

plt.figure(figsize=(10,6))
sns.heatmap(df[numeric_columns].corr(),annot=True,cmap="Blues")
plt.title("Correlation Heatmap")
plt.savefig("visualization/heatmap.png")
plt.close()

plt.figure(figsize=(8,5))
sns.scatterplot(
    data=df,
    x="Life_Expectancy",
    y="HDI"
)
plt.savefig("visualization/scatter.png")
plt.close()

# ===========================
# Machine Learning
# ===========================

X = df[
    [
        "Life_Expectancy",
        "Expected_Schooling",
        "Mean_Schooling",
        "GNI_Per_Capita"
    ]
]

y = df["HDI"]

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()

model.fit(X_train,y_train)

pred = model.predict(X_test)

print("\nR2 Score :",r2_score(y_test,pred))
print("MAE :",mean_absolute_error(y_test,pred))
print("RMSE :",mean_squared_error(y_test,pred)**0.5)

pickle.dump(model,open("hdi_model.pkl","wb"))

print("\nModel Saved Successfully")