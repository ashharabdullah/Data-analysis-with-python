import pandas as pd

# Load the dataset
df = pd.read_csv("real_estate_dataset.csv")

# Display the first 10 rows
df.head(10)

# Dataset shape
print("Shape:", df.shape)

# Columns
print("Columns:", df.columns)

# Data types
print("Data Types:", df.dtypes)

# Filter properties priced above $500,000
high_priced = df[df['Price'] > 500000]
high_priced.head()

# Sort properties by price in descending order
sorted_df = df.sort_values(by='Price', ascending=False)
sorted_df.head()

# Check for missing values
missing_values = df.isnull().sum()
print("Missing Values per Column:", missing_values)

# Check missing values in Size_sqft
print("Missing values before filling:", df["Size_sqft"].isna().sum())

# Fill missing Size_sqft with median
median_size = df["Size_sqft"].median()
df["Size_sqft"].fillna(median_size, inplace=True)

# Verify missing values are filled
print("Missing values after filling:", df["Size_sqft"].isna().sum())

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Average price by city
avg_price_by_city = df.groupby('City')['Price'].mean()
print("Average Price by City:", avg_price_by_city)

# Create a new column PricePerSqft
df["PricePerSqft"] = df["Price"] / df["Size_sqft"]

# Show first 10 rows to verify
print(df[["PropertyID", "Price", "Size_sqft", "PricePerSqft"]])

import pandas as pd
import numpy as np

# Create a random listing date for each property
df["ListingDate"] = pd.to_datetime(
    np.random.choice(pd.date_range("2018-01-01", "2023-12-31"), size=len(df))
)

# Extract Month and Year into new columns
df["ListingMonth"] = df["ListingDate"].dt.month
df["ListingYear"] = df["ListingDate"].dt.year

# Show first 10 rows to verify
print(df[["PropertyID", "ListingDate", "ListingMonth", "ListingYear"]].head(10))

# Count number of properties in each city
listings_by_city = df["City"].value_counts()

# Show the result
print(listings_by_city)

# Group by City and Year, then calculate average Price
avg_price_city_year = df.groupby(["City", "Year"])["Price"].mean().unstack()

# Show the result
print(avg_price_city_year)

# Select relevant numeric columns
numeric_cols = ["Size_sqft", "Bedrooms", "Bathrooms", "Price"]

# Calculate correlation matrix
corr_matrix = df[numeric_cols].corr()

# Show correlation matrix
print(corr_matrix)

import matplotlib.pyplot as plt
import seaborn as sns

# Group by City and sum the Price
total_sales_by_city = df.groupby("City")["Price"].sum().sort_values(ascending=False)

# Take top 10 cities (if dataset has fewer than 10, it will take all)
top_cities = total_sales_by_city.head(10)

# Plot bar chart
plt.figure(figsize=(10,6))
sns.barplot(x=top_cities.index, y=top_cities.values, palette="viridis")
plt.title("Top 10 Cities by Total Sales")
plt.xlabel("City")
plt.ylabel("Total Sales ($)")
plt.xticks(rotation=45)
plt.show()

# Group by ListingYear and calculate average Price
yearly_trend = df.groupby("ListingYear")["Price"].mean()

# Plot line chart
plt.figure(figsize=(10,6))
plt.plot(yearly_trend.index, yearly_trend.values, marker='o', linestyle='-', color='blue')
plt.title("Yearly Average House Price Trend")
plt.xlabel("Year")
plt.ylabel("Average Price ($)")
plt.grid(True)
plt.show()

# Plot histogram of house sizes
plt.figure(figsize=(10,6))
plt.hist(df["Size_sqft"], bins=10, edgecolor="black", color="skyblue")
plt.title("Distribution of House Sizes (sqft)")
plt.xlabel("Size (sqft)")
plt.ylabel("Number of Properties")
plt.show()

# Boxplot of Price by City
plt.figure(figsize=(10,6))
sns.boxplot(x="City", y="Price", data=df, palette="Set2")
plt.title("House Prices by City")
plt.xlabel("City")
plt.ylabel("Price ($)")
plt.xticks(rotation=45)
plt.show()

# Count of each property type
property_counts = df["Type"].value_counts()

# Plot pie chart
plt.figure(figsize=(8,8))
plt.pie(property_counts, labels=property_counts.index, autopct="%1.1f%%", startangle=140, colors=sns.color_palette("pastel"))
plt.title("Distribution of Property Types")
plt.show()

# Ensure ListingDate exists and is datetime
df["ListingDate"] = pd.to_datetime(df["ListingDate"])

# Extract Year-Month for grouping
df["YearMonth"] = df["ListingDate"].dt.to_period("M")

# Group by YearMonth and calculate average Price
monthly_avg_price = df.groupby("YearMonth")["Price"].mean()

# Plot line chart
plt.figure(figsize=(12,6))
monthly_avg_price.plot(marker='o', linestyle='-')
plt.title("Average Monthly Housing Prices Over Time")
plt.xlabel("Year-Month")
plt.ylabel("Average Price ($)")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Select relevant numeric columns
numeric_cols = ["Size_sqft", "Bedrooms", "Bathrooms", "Price"]

# Calculate correlation matrix
corr_matrix = df[numeric_cols].corr()

# Plot heatmap
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap: Size, Bedrooms, Bathrooms, Price")
plt.show()