# Pandas Study Guide

Pandas is essential for data manipulation in ML and robotics applications, especially for handling sensor data, logs, and datasets. Robotics interviews often test data manipulation skills and analytical thinking with real-world datasets.

## Key Concepts

### DataFrame Operations
- `df.mean()`: Column-wise mean
- `df[df['col'] > value]`: Filtering
- `df.groupby('col').sum()`: Group operations
- `df.fillna(value)`: Handle missing data
- `pd.merge(df1, df2, on='col')`: Join DataFrames

### Data Loading
- `pd.read_csv(file)`: Load CSV files
- `pd.read_json(file)`: Load JSON
- `pd.read_parquet(file)`: Load Parquet (efficient for large data)

### Data Cleaning
- `df.dropna()`: Remove NaN values
- `df.fillna(method='mean')`: Fill with mean
- `df.duplicated()`: Find duplicates

## Interview-Ready Concepts

### Time Series Analysis
- `df.resample()`: Resample time series
- `df.rolling()`: Rolling window operations
- `df.shift()`: Lag/lead operations
- Critical for robotics sensor data

### Advanced Grouping
- `df.groupby().agg()`: Multiple aggregations
- `df.groupby().transform()`: Group-wise transformations
- `df.groupby().apply()`: Custom functions per group

### Memory Optimization
- `df.dtypes`: Check data types
- `df.memory_usage()`: Memory consumption
- `pd.to_numeric()`: Convert to optimal types
- Important for large robotics datasets

## Worked Examples

### Problem 1: Load and Analyze CSV
```python
import pandas as pd

# Load CSV (assuming 'data.csv' exists)
df = pd.read_csv('data.csv')

# Compute mean of 'value' column
mean_value = df['value'].mean()
print(f"Mean value: {mean_value}")
```

### Problem 2: Filter DataFrame
```python
# Filter rows where 'score' > 80
filtered_df = df[df['score'] > 80]
print(filtered_df.head())
```

### Problem 3: Group and Aggregate
```python
# Group by 'category' and sum 'amount'
grouped = df.groupby('category')['amount'].sum()
print(grouped)
```

### Problem 4: Handle Missing Values
```python
# Fill NaN in 'price' column with mean
df['price'] = df['price'].fillna(df['price'].mean())
print(df.head())
```

### Problem 5: Merge DataFrames
```python
# Create sample DataFrames
df1 = pd.DataFrame({'id': [1, 2, 3], 'name': ['A', 'B', 'C']})
df2 = pd.DataFrame({'id': [1, 2, 3], 'value': [10, 20, 30]})

# Merge on 'id'
merged = pd.merge(df1, df2, on='id')
print(merged)
```

## Advanced Interview Topics

### Time Series Operations
```python
# Create time series data
dates = pd.date_range('2023-01-01', periods=100, freq='D')
ts_data = pd.DataFrame({
    'date': dates,
    'sensor_reading': np.random.randn(100).cumsum()
})
ts_data.set_index('date', inplace=True)

# Resample to weekly means
weekly_data = ts_data.resample('W').mean()
print(weekly_data.head())

# Rolling window operations
rolling_mean = ts_data.rolling(window=7).mean()
print(rolling_mean.head())
```

### Advanced Data Cleaning
```python
# Handle outliers
def remove_outliers(df, column, threshold=3):
    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
    return df[z_scores < threshold]

# Remove duplicates based on subset of columns
df.drop_duplicates(subset=['id', 'timestamp'], keep='first', inplace=True)

# Convert data types for memory efficiency
df['category'] = df['category'].astype('category')
df['timestamp'] = pd.to_datetime(df['timestamp'])
```

### Pivot Tables and Cross-Tabulations
```python
# Create pivot table
pivot = df.pivot_table(
    values='value', 
    index='category', 
    columns='month', 
    aggfunc='mean'
)
print(pivot)

# Cross-tabulation
cross_tab = pd.crosstab(df['category'], df['status'])
print(cross_tab)
```

### Apply and Transform Operations
```python
# Apply custom function to each group
def normalize_group(group):
    return (group - group.mean()) / group.std()

df['normalized_value'] = df.groupby('category')['value'].transform(normalize_group)

# Apply function to DataFrame rows
def complex_calculation(row):
    return row['a'] * row['b'] + row['c']

df['result'] = df.apply(complex_calculation, axis=1)
```

### Memory and Performance Optimization
```python
# Check memory usage
print(df.memory_usage(deep=True))

# Convert to categorical for memory savings
df['category'] = df['category'].astype('category')

# Use chunks for large files
chunk_size = 10000
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    process_chunk(chunk)
```

### MultiIndex Operations
```python
# Create MultiIndex DataFrame
arrays = [['A', 'A', 'B', 'B'], ['one', 'two', 'one', 'two']]
index = pd.MultiIndex.from_arrays(arrays, names=['first', 'second'])
df = pd.DataFrame(np.random.randn(4, 2), index=index, columns=['X', 'Y'])

# Access MultiIndex levels
print(df.loc['A'])  # All rows with first level 'A'
print(df.loc[('A', 'one')])  # Specific combination
```

## Practice Tips
- Use `df.head()` and `df.info()` for exploration
- Chain operations: `df.dropna().groupby('col').mean()`
- Index matters: use `df.reset_index()` when needed
- For large datasets, consider `chunksize` in `read_csv()`
- Use vectorized operations instead of loops
- Always check for missing values and data types
- Consider memory constraints for robotics applications
