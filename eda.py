# Importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Load the first dataset
df1 = pd.read_csv(r"C:\Users\uchen\PycharmProjects\CoffeeShopAnalysis\coffee.csv")

# Load the second dataset
df2 = pd.read_csv(r"C:\Users\uchen\PycharmProjects\CoffeeShopAnalysis\coffee_clean.csv")

# Load the third dataset
df3 = pd.read_csv(r"C:\Users\uchen\PycharmProjects\CoffeeShopAnalysis\coffee_id.csv")

# Display the head of the data
print(df1.head(3))
print(df2.head(3))
print(df3.head(3))

# Display the tail of the data
print(df1.tail(3))
print(df2.tail(3))
print(df3.tail(3))

#Display data information
print(df1.info())
print(df2.info())
print(df3.info())

# Check duplicate rows across datasets
print(f'Duplicate rows in df1: {df1.duplicated().sum()}')
print(f'Duplicate rows in df2: {df2.duplicated().sum()}')
print(f'Duplicate rows in df3: {df3.duplicated().sum()}')

# Check missing values across datasets
print(f'Missing values in df1: \n{df1.isnull().sum()}')
print(f'Missing values in df2: \n{df2.isnull().sum()}')
print(f'Missing values in df3: \n{df3.isnull().sum()}')

# Identify categorical and numerical variables in df1
categorical_columns_df1 = df1.select_dtypes(include=['object']).columns
numerical_columns_df1 = df1.select_dtypes(include=['float64', 'int64']).columns

print(f'Categorical columns in df1: {categorical_columns_df1}')
print(f'Numerical columns in df1: {numerical_columns_df1}')

# Fill categorical columns with 'unknown'
df1[categorical_columns_df1] = df1[categorical_columns_df1].fillna('unknown')

# Fill numerical columns with the median
df1[numerical_columns_df1] = df1[numerical_columns_df1].fillna(df1[numerical_columns_df1].median())

# Check missing values in df1
print(f'Missing values in df1: \n{df1.isnull().sum()}')

# Covert review data to datetime format
df1['review_date'] = pd.to_datetime(df1['review_date'])

# Inspect column names in each dataset
print('Columns in Dataset 1:', df1.columns)
print('Columns in Dataset 2:', df2.columns)
print('Columns in Dataset 3:', df3.columns)

# To identify unique columns in df1, df2, and df3
# Get the columns set for each dataset
columns_df1 = set(df1.columns)
columns_df2 = set(df2.columns)
columns_df3 = set(df3.columns)

# Find unique and shared columns
unique_to_df1 = columns_df1 - (columns_df2 | columns_df3)
unique_to_df2 = columns_df2 - (columns_df1 | columns_df3)
unique_to_df3 = columns_df3 - (columns_df1 | columns_df2)

shared_columns = columns_df1 & columns_df2 & columns_df3

# Display the results
print('Unique columns in Dataset 1:')
print(unique_to_df1)

print('\nUnique columns in Dataset 2:')
print(unique_to_df2)

print('\nUnique columns in Dataset 3:')
print(unique_to_df3)

print('\nColumns shared across all datasets:')
print(shared_columns)

# Columns shared between any two datasets
shared_df1_df2 = columns_df1 & columns_df2
shared_df1_df3 = columns_df1 & columns_df3
shared_df2_df3 = columns_df2 & columns_df3

print('\nShared columns between Dataset 1 and Dataset 2:')
print(shared_df1_df2)

print('\nShared columns between Dataset 1 and Dataset 3:')
print(shared_df1_df3)

print('\nShared columns between Dataset 2 and Dataset 3:')
print(shared_df2_df3)

# Define common column
common_column = 'slug'

# Merge df1, df2, and df3 into a dataframe
merged_df = pd.merge(df1, df2, on=common_column, how='left')
merged_df1 = pd.merge(merged_df, df3, on=common_column, how='left')

# Display the merged dataframe
print(merged_df1.head(3))

# Merged dataset information
print(merged_df1.info())

# Check missing values of merged data
print(f'Missing values in merged_df1: \n{merged_df1.isnull().sum()}')

# Fill missing values in the merged dataset
merged_df1.fillna(0, inplace=True)

# Check missing values of merged data
print(f'Missing values in merged_df1: \n{merged_df1.isnull().sum()}')

# Combine duplicate columns
merged_df1['name'] = merged_df1['name_x'].combine_first(merged_df1['name_y'])
merged_df1['rating'] = merged_df1['rating_x'].combine_first(merged_df1['rating_y'])
merged_df1['roaster'] = merged_df1['roaster_x'].combine_first(merged_df1['roaster_y'])
merged_df1['region_africa_arabia'] = merged_df1['region_africa_arabia_x'].combine_first(merged_df1['region_africa_arabia_y'])
merged_df1['region_caribbean'] = merged_df1['region_caribbean_x'].combine_first(merged_df1['region_caribbean_y'])
merged_df1['region_central_america'] = merged_df1['region_central_america_x'].combine_first(merged_df1['region_central_america_y'])
merged_df1['region_hawaii'] = merged_df1['region_hawaii_x'].combine_first(merged_df1['region_hawaii_y'])
merged_df1['region_asia_pacific'] = merged_df1['region_asia_pacific_x'].combine_first(merged_df1['region_asia_pacific_y'])
merged_df1['region_south_america'] = merged_df1['region_south_america_x'].combine_first(merged_df1['region_south_america_y'])
merged_df1['type_espresso'] = merged_df1['type_espresso_x'].combine_first(merged_df1['type_espresso_y'])
merged_df1['type_organic'] = merged_df1['type_organic_x'].combine_first(merged_df1['type_organic_y'])
merged_df1['type_fair_trade'] = merged_df1['type_fair_trade_x'].combine_first(merged_df1['type_fair_trade_y'])
merged_df1['type_decaffeinated'] = merged_df1['type_decaffeinated_x'].combine_first(merged_df1['type_decaffeinated_y'])
merged_df1['type_pod_capsule'] = merged_df1['type_pod_capsule_x'].combine_first(merged_df1['type_pod_capsule_y'])
merged_df1['type_blend'] = merged_df1['type_blend_x'].combine_first(merged_df1['type_blend_y'])
merged_df1['type_estate'] = merged_df1['type_estate_x'].combine_first(merged_df1['type_estate_y'])
merged_df1['review_date'] = merged_df1['review_date_x'].combine_first(merged_df1['review_date_y'])
merged_df1['aroma'] = merged_df1['aroma_x'].combine_first(merged_df1['aroma_x'])
merged_df1['body'] = merged_df1['body_x'].combine_first(merged_df1['body_y'])
merged_df1['flavor'] = merged_df1['flavor_x'].combine_first(merged_df1['flavor_y'])

# Drop the duplicate columns
merged_df1.drop(columns=['name_x', 'name_y', 'rating_x', 'rating_y', 'roaster_x', 'roaster_y',
                        'region_africa_arabia_x', 'region_africa_arabia_y', 'region_caribbean_x', 'region_caribbean_y',
                        'region_central_america_x', 'region_central_america_y', 'region_hawaii_x', 'region_hawaii_y',
                        'region_asia_pacific_x', 'region_asia_pacific_y', 'region_south_america_x', 'region_south_america_y',
                        'type_espresso_x', 'type_espresso_y', 'type_organic_x', 'type_organic_y',
                        'type_fair_trade_x', 'type_fair_trade_y', 'type_decaffeinated_x', 'type_decaffeinated_y',
                        'type_pod_capsule_x', 'type_pod_capsule_y', 'type_blend_x', 'type_blend_y',
                        'type_estate_x', 'type_estate_y', 'review_date_x', 'review_date_y',
                        'aroma_x', 'aroma_y', 'body_x', 'body_y', 'flavor_x', 'flavor_y'], axis=1, inplace=True)

# Save the merged dataframe to a file using joblib
joblib.dump(merged_df1, 'merged_df.joblib')

# Merged data information
print(merged_df1.info())

# Display summary statistics
print(merged_df1.describe())

# Identify categorical and numerical columns
categorical_columns = merged_df1.select_dtypes(include=['object']).columns
numerical_columns = merged_df1.select_dtypes(include=['float64', 'int64']).columns

print(f'Categorical columns: {categorical_columns}')
print(f'Numerical columns: {numerical_columns}')

# Correlation matrix of numerical columns
correlation_matrix = merged_df1[numerical_columns].corr()
print(correlation_matrix)

# Plot heatmap for correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
print(plt.show())

# Pair plot for numerical relationship
sns.pairplot(merged_df1[numerical_columns], diag_kind='kde', corner=True)
print(plt.show())

# Count plot of roast type across region africa arabia
plt.figure(figsize=(12, 6))
sns.countplot(data=merged_df1, x='region_africa_arabia', hue='roast')
plt.title('Count Plot of Roast Type Across Region Africa Arabia')
plt.xlabel('Region Africa Arabia')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Roast Type', loc='upper right')
print(plt.show())

# Count plot of roast type across region caribbean
plt.figure(figsize=(12, 6))
sns.countplot(data=merged_df1, x='region_caribbean', hue='roast')
plt.title('Count Plot of Roast Type Across Region Caribbean')
plt.xlabel('Region Caribbean')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Roast Type', loc='upper right')
print(plt.show())

# Count plot of roast type across region central america
plt.figure(figsize=(12, 6))
sns.countplot(data=merged_df1, x='region_central_america', hue='roast')
plt.title('Count Plot of Roast Type Across Region Central America')
plt.xlabel('Region Central America')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Roast Type', loc='upper right')
print(plt.show())

# Count plot of roast type across region hawaii
plt.figure(figsize=(12, 6))
sns.countplot(data=merged_df1, x='region_hawaii', hue='roast')
plt.title('Count Plot of Roast Type Across Region Hawaii')
plt.xlabel('Region Hawaii')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Roast Type', loc='upper right')
print(plt.show())

# Count plot of roast type across region asia pacific
plt.figure(figsize=(12, 6))
sns.countplot(data=merged_df1, x='region_asia_pacific', hue='roast')
plt.title('Count Plot of Roast Type Across Region Asia Pacific')
plt.xlabel('Region Asia Pacific')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Roast Type', loc='upper right')
print(plt.show())

# Count plot of roast type across region south america
plt.figure(figsize=(12, 6))
sns.countplot(data=merged_df1, x='region_south_america', hue='roast')
plt.title('Count Plot of Roast Type Across Region South America')
plt.xlabel('Region South America')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Roast Type', loc='upper right')
plt.show()

# To verify unique identifiers like slug, name, and review_date
# Columns to check for uniqueness
columns_to_check = ['slug', 'name', 'review_date']

# Verify uniqueness for each column
for column in columns_to_check:
  print(f'Checking uniqueness for column: {column}')

  # Check if values are unique
  is_unique = merged_df1[column].is_unique
  print(f' All values unique: {is_unique}')

  if not is_unique:
    # Identify duplicate values
    duplicates = merged_df1[merged_df.duplicated(subset=[column], keep=False)]
    print(f" Duplicate values found in column '{column}':")
    print(duplicates[[column]].value_counts())
  else:
      print(f" No duplicate values found in column '{column}'.")

print('-' * 40)

# Check for missing values in rating and review_date columns
columns_to_check_missing_values = ['rating', 'review_date']

# Check for missing values
missing_values = merged_df1[columns_to_check_missing_values].isnull().sum()

# Print the result
print('Missing values in rating and review_date columns:')
print(missing_values)

# Duplicate entries in slug or name
# Check for duplicate entries in 'slug' and 'name'
duplicate_slug = merged_df1[merged_df1.duplicated(subset=['slug'], keep=False)]
duplicate_name = merged_df1[merged_df1.duplicated(subset=['name'], keep=False)]

# Display the duplicate rows
print("Duplicate entries in 'slug':")
print(duplicate_slug)

print("\nDuplicate entries in 'name':")
print(duplicate_name)

# Identify categorical variables
categorical_columns = merged_df1.select_dtypes(include=['object', 'category']).columns

print('Categorical variables:')
print(categorical_columns)

# To identify inconsistencies in categorical variables
# Columns to check for inconsistencies
categorical_columns = ['slug', 'location', 'origin', 'roast', 'roaster']

# Check unique value in each categorical column
for column in categorical_columns:
  print(f"Unique values in '{column}':")
  print(merged_df1[column].unique())
  print('-' * 40)

# Check for duplicates in slug and review date
duplicate_slug = merged_df1[merged_df1.duplicated(subset=['slug'], keep=False)]
duplicate_review_date = merged_df1[merged_df1.duplicated(subset=['review_date'], keep=False)]

print("Duplicate entries in 'slug':")
print(duplicate_slug)

print("\nDuplicate entries in 'review_date':")
print(duplicate_review_date)

# Display the number of duplicates slug
print(f"Number of duplicate entries in 'slug': {duplicate_slug.shape[0]}")

# Display the number of duplicates review date
print(f"Number of duplicate entries in 'review_date': {duplicate_review_date.shape[0]}")

# Remove duplicates, keeping first occurrence in review date
merged_df1.drop_duplicates(subset=['review_date'], keep='first', inplace=True)

# Display number of duplicates in review date after removal
duplicate_review_date = merged_df1[merged_df1.duplicated(subset=['review_date'], keep=False)]
print(f"Number of duplicate entries in 'review_date' after removal: {duplicate_review_date.shape[0]}")

# Save the cleaned merged dataframe to a file using joblib
joblib.dump(merged_df1, 'cleaned_merged_df.joblib')

# Read and view the head of the cleaned merged dataset
# Load the joblib file
cleaned_df = joblib.load('cleaned_merged_df.joblib')
print(cleaned_df.head(3))

# Normalize rating column
def normalize_column(series, min_value=0, max_value=1):
  # Convert the series to numeric, handling errors by coercing to NaN
  series = pd.to_numeric(series, errors='coerce')
  return (series - series.min()) / (series.max() - series.min()) * (max_value - min_value) + min_value # Changed missing_values to min_value

# Normalize the 'rating' column to a 1-5 scale
cleaned_df['normalized_rating_1_5'] = normalize_column(cleaned_df['rating'], min_value=1, max_value=5)

print(cleaned_df[['rating', 'normalized_rating_1_5']].head())

# To add derived columns like year and month from review_date
# Ensure 'review_date' is in datetime format
cleaned_df['review_date'] = pd.to_datetime(cleaned_df['review_date'], errors='coerce')

# Extract year and month into new columns
cleaned_df['year'] = cleaned_df['review_date'].dt.year
cleaned_df['month'] = cleaned_df['review_date'].dt.month

print(cleaned_df[['review_date', 'year', 'month']].head())

# Calculate the average rating for each region africa arabia and roast
average_rating = cleaned_df.groupby(['region_africa_arabia', 'roast'])['normalized_rating_1_5'].mean().reset_index()

# Visualize the average rating
plt.figure(figsize=(10, 6))
sns.barplot(x='region_africa_arabia', y='normalized_rating_1_5', hue='roast', data=average_rating)
plt.title('Average Rating by Region Africa Arabia')
plt.xlabel('Region Africa Arabia')
plt.ylabel('Average Rating')
plt.legend(title='Roast', loc='upper right')

print(plt.show())

# Calculate the average rating for each region caribbean and roast
average_rating = cleaned_df.groupby(['region_caribbean', 'roast'])['normalized_rating_1_5'].mean().reset_index()

# Visualize the average rating
plt.figure(figsize=(10, 6))
sns.barplot(x='region_caribbean', y='normalized_rating_1_5', hue='roast', data=average_rating)
plt.title('Average Rating by Region Caribbean')
plt.xlabel('Region Caribbean')
plt.ylabel('Average Rating')
plt.legend(title='Roast', loc='upper right')

print(plt.show())

# Calculate the average rating for each region central america and roast
average_rating = cleaned_df.groupby(['region_central_america', 'roast'])['normalized_rating_1_5'].mean().reset_index()

# Visualize the average rating
plt.figure(figsize=(10, 6))
sns.barplot(x='region_central_america', y='normalized_rating_1_5', hue='roast', data=average_rating)
plt.title('Average Rating by Region Central America')
plt.xlabel('Region Central America')
plt.ylabel('Average Rating')
plt.legend(title='Roast', loc='upper right')

print(plt.show())

# Calculate the average rating for each region hawaii and roast
average_rating = cleaned_df.groupby(['region_hawaii', 'roast'])['normalized_rating_1_5'].mean().reset_index()

# Visualize the average rating
plt.figure(figsize=(10, 6))
sns.barplot(x='region_hawaii', y='normalized_rating_1_5', hue='roast', data=average_rating)
plt.title('Average Rating by Region Hawaii')
plt.xlabel('Region Hawaii')
plt.ylabel('Average Rating')
plt.legend(title='Roast', loc='upper right')

print(plt.show())

# Calculate the average rating for each region asia pacific and roast
average_rating = cleaned_df.groupby(['region_asia_pacific', 'roast'])['normalized_rating_1_5'].mean().reset_index()

# Visualize the average rating
plt.figure(figsize=(10, 6))
sns.barplot(x='region_asia_pacific', y='normalized_rating_1_5', hue='roast', data=average_rating)
plt.title('Average Rating by Region Asia Pacific')
plt.xlabel('Region Asia Pacific')
plt.ylabel('Average Rating')
plt.legend(title='Roast', loc='upper right')

print(plt.show())

# Calculate the average rating for each region south america and roast
average_rating = cleaned_df.groupby(['region_south_america', 'roast'])['normalized_rating_1_5'].mean().reset_index()

# Visualize the average rating
plt.figure(figsize=(10, 6))
sns.barplot(x='region_south_america', y='normalized_rating_1_5', hue='roast', data=average_rating)
plt.title('Average Rating by Region South America')
plt.xlabel('Region South America')
plt.ylabel('Average Rating')
plt.legend(title='Roast', loc='upper right')

print(plt.show())

# To identify top-rated products
cleaned_df_sorted = cleaned_df.sort_values(by='normalized_rating_1_5', ascending=False)

# Display the top rated products
top_rated_products = cleaned_df_sorted.head(10)
print('Top Rated Products:')
print(top_rated_products[['name', 'roaster', 'normalized_rating_1_5', 'roast', 'review_date']])

# Plot top rated products
plt.figure(figsize=(12, 6))
plt.bar(top_rated_products['roaster'], top_rated_products['normalized_rating_1_5'])
plt.title('Top Rated Products')
plt.xlabel('Product Name')
plt.ylabel('Normalized Rating')
plt.xticks(rotation=90)

print(plt.show())

# To identify most reviewed roasters
# Count the number of reviews per roaster
roaster_review_counts = cleaned_df['roaster'].value_counts().reset_index()

# Rename the column for clarity
roaster_review_counts.columns = ['roaster', 'review_count']

# Sort the result in ascending order of review count
roaster_review_counts = roaster_review_counts.sort_values(by='review_count', ascending=False)

# Display the most-reviewd roasters
print('Most Reviewed Roasters:')
print(roaster_review_counts)

# Plot the top 10 most reviewed roaster
top_10_roasters = roaster_review_counts.head(10)
plt.figure(figsize=(12, 6))
plt.bar(top_10_roasters['roaster'], top_10_roasters['review_count'])
plt.title('Top 10 Most Reviewed Roasters')
plt.xlabel('Roaster')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=90)

print(plt.show())

# Plot the top 10 least reviewed roaster
least_10_roasters = roaster_review_counts.tail(10)
plt.figure(figsize=(12, 6))
plt.bar(least_10_roasters['roaster'], least_10_roasters['review_count'])
plt.title('Top 10 Least Reviewed Roasters')
plt.xlabel('Roaster')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=90)

print(plt.show())

# Analyze ratings and reviews by season or year
# Define a function to assign season
def get_season(month):
  if month in [12, 1, 2]:
    return 'Winter'
  elif month in [3, 4, 5]:
    return 'Spring'
  elif month in [6, 7, 8]:
    return 'Summer'
  elif month in [9, 10, 11]:
    return 'Autumn'

# Create a 'season' column
cleaned_df['season'] = cleaned_df['month'].apply(get_season)

# Group by 'year' and 'season' to analyze ratings and review count
seasonal_analysis = cleaned_df.groupby(['year', 'season']).agg(
    average_rating=('normalized_rating_1_5', 'mean'),
    review_count=('review_date', 'count')
).reset_index()

# Display the analysis
print('Seasonal Analysis and Review:')
print(seasonal_analysis)

# To analyze data for a specific year or season
specific_year = seasonal_analysis[seasonal_analysis['year'] == 2018]
specific_season = seasonal_analysis[seasonal_analysis['season'] == 'Winter']

# Plot average rating by season
plt.figure(figsize=(10, 6))
sns.barplot(x='season', y='average_rating', hue='year', data=seasonal_analysis) # Changed y to 'average_rating'
plt.title('Average Rating by Season')
plt.xlabel('season')
plt.ylabel('Rating Count')


print(plt.show())

# Review count by season
plt.figure(figsize=(10, 6))
sns.barplot(x='season', y='review_count', hue='year', data=seasonal_analysis)
plt.title('Review Count by Season')
plt.xlabel('season')
plt.ylabel('Review Count')

print(plt.show())

# To compare average ratings across regions
# Average rating by region_africa_arabia
average_rating_by_region_africa_arabia = cleaned_df.groupby('region_africa_arabia')['normalized_rating_1_5'].mean().reset_index()

# Rename column for clarity
average_rating_by_region_africa_arabia.rename(columns={'normalized_rating_1_5': 'average_rating'}, inplace=True)

# Sort in ascending order
average_rating_by_region_africa_arabia = average_rating_by_region_africa_arabia.sort_values(by='average_rating', ascending=False)

# Display the result
print('Average Rating by Region Africa Arabia:')
print(average_rating_by_region_africa_arabia)

# Plot average rating by region africa arabia
plt.figure(figsize=(10, 6))
sns.barplot(x='region_africa_arabia', y='average_rating', data=average_rating_by_region_africa_arabia)
plt.title('Average Rating by Region Africa Arabia')
plt.xlabel('Region Africa Arabia')
plt.ylabel('Average Rating')

print(plt.show())

# Average rating by region_africa_arabia
average_rating_by_region_caribbean = cleaned_df.groupby('region_caribbean')['normalized_rating_1_5'].mean().reset_index()

# Rename column for clarity
average_rating_by_region_caribbean.rename(columns={'normalized_rating_1_5': 'average_rating'}, inplace=True)

# Sort in ascending order
average_rating_by_region_caribbean = average_rating_by_region_caribbean.sort_values(by='average_rating', ascending=False)

# Display the result
print('Average Rating by Region Caribbean:')
print(average_rating_by_region_caribbean)

# Plot average rating by region caribbean
plt.figure(figsize=(10, 6))
sns.barplot(x='region_caribbean', y='average_rating', data=average_rating_by_region_caribbean)
plt.title('Average Rating by Region Caribbean')
plt.xlabel('Region Caribbean')
plt.ylabel('Average Rating')

print(plt.show())

# Average rating by region_region_central_america
average_rating_by_region_central_america = cleaned_df.groupby('region_central_america')['normalized_rating_1_5'].mean().reset_index()

# Rename column for clarity
average_rating_by_region_central_america.rename(columns={'normalized_rating_1_5': 'average_rating'}, inplace=True)

# Sort in ascending order
average_rating_by_region_central_america = average_rating_by_region_central_america.sort_values(by='average_rating', ascending=False)

# Display the result
print('Average Rating by Region Central America:')
print(average_rating_by_region_central_america)

# Plot average rating by region central america
sns.barplot(x='region_central_america', y='average_rating', data=average_rating_by_region_central_america)
plt.title('Average Rating by Region Central America')
plt.xlabel('Region Central America')
plt.ylabel('Average Rating')

print(plt.show())

# Average rating by region_region_hawaii
average_rating_by_region_hawaii = cleaned_df.groupby('region_hawaii')['normalized_rating_1_5'].mean().reset_index()

# Rename column for clarity
average_rating_by_region_hawaii.rename(columns={'normalized_rating_1_5': 'average_rating'}, inplace=True)

# Sort in ascending order
average_rating_by_region_hawaii = average_rating_by_region_hawaii.sort_values(by='average_rating', ascending=False)

# Display the result
print('Average Rating by Region Hawaii:')
print(average_rating_by_region_hawaii)

# Plot average rating by region hawaii
sns.barplot(x='region_hawaii', y='average_rating', data=average_rating_by_region_hawaii)
plt.title('Average Rating by Region Hawaii')
plt.xlabel('Region Hawaii')
plt.ylabel('Average Rating')

print(plt.show())

# Average rating by region_region_asia_pacific
average_rating_by_region_asia_pacific = cleaned_df.groupby('region_asia_pacific')['normalized_rating_1_5'].mean().reset_index()

# Rename column for clarity
average_rating_by_region_asia_pacific.rename(columns={'normalized_rating_1_5': 'average_rating'}, inplace=True)

# Sort in ascending order
average_rating_by_region_asia_pacific = average_rating_by_region_asia_pacific.sort_values(by='average_rating', ascending=False)

# Display the result
print('Average Rating by Region Asia Pacific:')
print(average_rating_by_region_asia_pacific)

# Plot average rating by region asia pacific
sns.barplot(x='region_asia_pacific', y='average_rating', data=average_rating_by_region_asia_pacific)
plt.title('Average Rating by Region Asia Pacific')
plt.xlabel('Region Asia Pacific')
plt.ylabel('Average Rating')

print(plt.show())

# Average rating by region_south_america
average_rating_by_region_south_america = cleaned_df.groupby('region_south_america')['normalized_rating_1_5'].mean().reset_index()

# Rename column for clarity
average_rating_by_region_south_america.rename(columns={'normalized_rating_1_5': 'average_rating'}, inplace=True)

# Sort in ascending order
average_rating_by_region_south_america = average_rating_by_region_south_america.sort_values(by='average_rating', ascending=False)

# Display the result
print('Average Rating by Region South America:')
print(average_rating_by_region_south_america)

# Plot average rating by region south america
sns.barplot(x='region_south_america', y='average_rating', data=average_rating_by_region_south_america)
plt.title('Average Rating by Region South America')
plt.xlabel('Region South America')
plt.ylabel('Average Rating')

print(plt.show())

# To analyze popular roast type and their regional distribution
roast_popularity = cleaned_df['roast'].value_counts().reset_index()
roast_popularity.columns = ['roast', 'count']
print('Most Popular Roast Type:')
print(roast_popularity)

# Plot roast popularity
plt.figure(figsize=(8, 6))
sns.barplot(x='roast', y='count', data=roast_popularity)
plt.title('Most Popular Roast Type')
plt.xlabel('Roast')
plt.ylabel('Count')

print(plt.show())

# Roast distribution by Region Africa Arabia
roast_distribution_africa_arabia = cleaned_df.groupby(['region_africa_arabia', 'roast']).size().reset_index(name='count')

# For better readability
roast_distribution_pivot = roast_distribution_africa_arabia.pivot(index='region_africa_arabia', columns='roast', values='count').fillna(0)
print('Roast Distribution by Region Africa Arabia:')
print(roast_distribution_pivot)

# To plot the chart of Region Africa Arabia
plt.figure(figsize=(10, 6))
sns.barplot(x='region_africa_arabia', y='count', hue='roast', data=roast_distribution_africa_arabia)
plt.title('Roast Distribution by Region Africa Arabia')
plt.xlabel('Region Africa Arabia')
plt.ylabel('Count')
plt.legend(title='Roast', loc='upper right')

print(plt.show())

# Roast distribution by Region Caribbean
roast_distribution_caribbean = cleaned_df.groupby(['region_caribbean', 'roast']).size().reset_index(name='count')

# For better readability
roast_distribution_pivot = roast_distribution_caribbean.pivot(index='region_caribbean', columns='roast', values='count').fillna(0)
print('Roast Distribution by Region Caribbean:')
print(roast_distribution_pivot)

# To plot the chart of Region Caribbean
plt.figure(figsize=(10, 6))
sns.barplot(x='region_caribbean', y='count', hue='roast', data=roast_distribution_caribbean)
plt.title('Roast Distribution by Region Caribbean')
plt.xlabel('Region Caribbean')
plt.ylabel('Count')
plt.legend(title='Roast', loc='upper right')

print(plt.show())

# Roast distribution by Region Central America
roast_distribution_central_america = cleaned_df.groupby(['region_central_america', 'roast']).size().reset_index(name='count')

# For better readability
roast_distribution_pivot = roast_distribution_central_america.pivot(index='region_central_america', columns='roast', values='count').fillna(0)
print('Roast Distribution by Region Central America:')
print(roast_distribution_pivot)

# To plot the chart of Region Central america
plt.figure(figsize=(10, 6))
sns.barplot(x='region_central_america', y='count', hue='roast', data=roast_distribution_central_america)
plt.title('Roast Distribution by Region Central America')
plt.xlabel('Region Central America')
plt.ylabel('Count')
plt.legend(title='Roast', loc='upper right')

print(plt.show())

# Roast distribution by Region Hawaii
roast_distribution_hawaii = cleaned_df.groupby(['region_hawaii', 'roast']).size().reset_index(name='count')

# For better readability
roast_distribution_pivot = roast_distribution_hawaii.pivot(index='region_hawaii', columns='roast', values='count').fillna(0)
print('Roast Distribution by Region Hawaii:')
print(roast_distribution_pivot)

# To plot the chart of Region Hawaii
plt.figure(figsize=(10, 6))
sns.barplot(x='region_hawaii', y='count', hue='roast', data=roast_distribution_hawaii)
plt.title('Roast Distribution by Region Hawaii')
plt.xlabel('Region Hawaii')
plt.ylabel('Count')
plt.legend(title='Roast', loc='upper right')

print(plt.show())

# Roast distribution by Region Asia Pacific
roast_distribution_asia_pacific = cleaned_df.groupby(['region_asia_pacific', 'roast']).size().reset_index(name='count')

# For better readability
roast_distribution_pivot = roast_distribution_asia_pacific.pivot(index='region_asia_pacific', columns='roast', values='count').fillna(0)
print('Roast Distribution by Region Asia Pacific:')
print(roast_distribution_pivot)

# To plot the chart of Region Asia Pacific
plt.figure(figsize=(10, 6))
sns.barplot(x='region_asia_pacific', y='count', hue='roast', data=roast_distribution_asia_pacific)
plt.title('Roast Distribution by Region Asia Pacific')
plt.xlabel('Region Asia Pacific')
plt.ylabel('Count')
plt.legend(title='Roast', loc='upper right')

print(plt.show())

# Roast distribution by Region South America
roast_distribution_south_america = cleaned_df.groupby(['region_south_america', 'roast']).size().reset_index(name='count')

# For better readability
roast_distribution_pivot = roast_distribution_south_america.pivot(index='region_south_america', columns='roast', values='count').fillna(0)
print('Roast Distribution by Region South America:')
print(roast_distribution_pivot)

# To plot the chart of Region South America
plt.figure(figsize=(10, 6))
sns.barplot(x='region_south_america', y='count', hue='roast', data=roast_distribution_south_america)
plt.title('Roast Distribution by Region South America')
plt.xlabel('Region South America')
plt.ylabel('Count')
plt.legend(title='Roast', loc='upper right')

print(plt.show())

# Correlation matrix between aroma, body, flavor
correlation_matrix = cleaned_df[['aroma', 'body', 'flavor']].corr()

# Plot correlation graph
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=" .2f")
plt.title('Correlation Matrix of Aroma, Body, and Flavor')

print(plt.show())

# To identify Attributes of Top-Rated Products
# Define top-rated products (Top 25% by rating)
top_rated_products = cleaned_df[cleaned_df['normalized_rating_1_5'] >= cleaned_df['normalized_rating_1_5'].quantile(0.75)]

# Compute average attributes for top-rated products
top_product_attributes = top_rated_products.select_dtypes(include=['number']).mean()

# Display top-rated product attributes
print("\n🔹 Top Attributes of High-Rated Products:")
print(top_product_attributes)

# Plot the top products attributes
plt.figure(figsize=(10, 6))
sns.barplot(x=top_product_attributes.index, y=top_product_attributes.values)
plt.title('Top Attributes of High-Rated Products')
plt.xlabel('Attribute')
plt.ylabel('Average Value')
plt.xticks(rotation=45)

print(plt.show())

# Identify Top-Performing Regions and Roasters
# Extracting the best regions
if 'region_africa_arabia' in cleaned_df.columns:
    # Identify region columns
    region_columns = [col for col in cleaned_df.columns if 'region_' in col]

    # Get the most relevant region per product
    cleaned_df['region'] = cleaned_df[region_columns].idxmax(axis=1).str.replace('region_', '').str.replace('_', ' ').str.title()

    # Compute region performance
    region_performance = cleaned_df.groupby('region')['normalized_rating_1_5'].agg(['mean', 'count']).reset_index()
    region_performance.rename(columns={'mean': 'average_rating', 'count': 'total_reviews'}, inplace=True)

    # Sort by highest rating and reviews
    top_regions = region_performance.sort_values(by=['average_rating', 'total_reviews'], ascending=[False, False])
    print("\n🔹 Top Performing Regions:")
    print(top_regions)

# Extracting best roasters
if 'roaster' in cleaned_df.columns:
    roaster_performance = cleaned_df.groupby('roaster')['normalized_rating_1_5'].agg(['mean', 'count']).reset_index()
    roaster_performance.rename(columns={'mean': 'average_rating', 'count': 'total_reviews'}, inplace=True)

    # Sort by highest average rating
    top_roasters = roaster_performance.sort_values(by=['average_rating', 'total_reviews'], ascending=[False, False])

    print("\n🔹 Top Performing Roasters:")
    print(top_roasters)

# Plot top performing regions
plt.figure(figsize=(10, 6))
sns.barplot(x='region', y='average_rating', data=top_regions.head(10))
plt.title('Top Performing Regions')
plt.xlabel('Region')
plt.ylabel('Average Rating')
plt.xticks(rotation=90)

print(plt.show())

# Plot top roasters
plt.figure(figsize=(10, 6))
sns.barplot(x='roaster', y='average_rating', data=top_roasters.head(10))
plt.title('Top Performing Roasters')
plt.xlabel('Roaster')
plt.ylabel('Average Rating')
plt.xticks(rotation=90)

print(plt.show())

# Identify Popular Roast Types and Regions for Promotions
# Identifying top roast types
if 'roast' in cleaned_df.columns:
    roast_popularity = cleaned_df['roast'].value_counts().reset_index()
    roast_popularity.columns = ['roast_type', 'count']

    print("\n🔹 Popular Roast Types:")
    print(roast_popularity)

# Identifying top regions for promotions
promotion_regions = top_regions[top_regions['total_reviews'] > top_regions['total_reviews'].median()]

print("\n🔹 Target Regions for Promotions:")
print(promotion_regions)

# Plot promotion regions
plt.figure(figsize=(10, 6))
sns.barplot(x='region', y='total_reviews', data=promotion_regions)
plt.title('Target Regions for Promotions')
plt.xlabel('Region')
plt.ylabel('Total Reviews')
plt.xticks(rotation=45)

print(plt.show())