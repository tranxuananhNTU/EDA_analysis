# Exploratory Data Analysis (EDA) Workflow

A comprehensive, beginner-friendly EDA workflow for small datasets (<10K rows) with a numeric target column (regression scenario).

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the notebook:**
   ```bash
   jupyter notebook eda_analysis.ipynb
   ```

3. **Configure for your data:**
   - Update `FILEPATH` to point to your CSV file
   - Set `TARGET_COLUMN` to your target variable name
   - Optionally set `EXCLUDE_COLS` and `DATE_COLUMN`

## Files

| File | Description |
|------|-------------|
| `eda_analysis.ipynb` | Main Jupyter notebook with complete EDA workflow |
| `eda_utils.py` | Reusable utility functions for EDA |
| `sample_data.csv` | Sample dataset for testing the workflow |
| `requirements.txt` | Python dependencies |
| `plots/` | Directory for saved visualizations |

## Workflow Structure

### 1. Setup
- Import libraries
- Configure display options
- Load dataset

### 2. Data Overview
- Shape and preview (head, tail, info)
- Missing values analysis
- Duplicate rows check
- Data types summary

### 3. Descriptive Statistics
- Numerical features summary
- Categorical features summary
- Statistical summary table

### 4. Data Cleaning
- Handle missing values
- Handle outliers
- Type conversions
- Save cleaned dataset

### 5. Univariate Analysis
- Numeric features: histograms, boxplots, violin plots
- Categorical features: bar plots, frequency tables

### 6. Bivariate Analysis
- Correlation matrix and heatmap
- Scatterplots vs target
- Pairplot of key features
- Numeric vs categorical analysis

### 7. Target Analysis
- Target distribution
- Top correlated features
- Target by categorical groups
- Transformation check

### 8. Advanced Analysis
- Distribution comparisons
- Time series (if applicable)
- Feature importance preview

### 9. Statistical Tests
- Normality test for target
- Group comparison tests (t-test, ANOVA)
- Chi-square tests

### 10. Summary and Findings
- Key insights documentation
- Recommendations for modeling

## Utility Functions

The `eda_utils.py` module provides reusable functions:

### Data Loading
```python
setup_environment()  # Configure pandas, matplotlib, numpy
load_data(filepath)  # Load dataset with validation
```

### Overview Functions
```python
data_overview(df)              # Comprehensive data overview
missing_values_analysis(df)    # Missing values with visualization
duplicate_analysis(df)         # Duplicate rows check
```

### Statistics
```python
numerical_summary(df)          # Numerical features statistics
categorical_summary(df)        # Categorical features statistics
outliers_summary(df)           # Outlier detection (IQR method)
```

### Visualization
```python
plot_numeric_distribution(df, column)       # Histogram, boxplot, violin
plot_categorical_distribution(df, column)   # Bar and pie charts
correlation_analysis(df, target)            # Correlation heatmap
pairplot_analysis(df, columns)              # Pairplot grid
```

### Target Analysis
```python
target_analysis(df, target)    # Comprehensive target analysis
plot_scatter_with_target(df, feature, target)  # Scatter with regression
numeric_vs_categorical(df, num_col, cat_col)   # Box/violin plots
```

### Statistical Tests
```python
normality_test(df, column)           # Shapiro-Wilk test
t_test_by_category(df, num, cat)     # Independent t-test
anova_test(df, num, cat)             # One-way ANOVA
chi_square_test(df, col1, col2)      # Chi-square test
```

### Data Cleaning
```python
clean_data(df, strategy='median')    # Basic cleaning pipeline
cap_outliers(df, column)             # Winsorization
save_cleaned_data(df, filepath)      # Save to CSV
```

## Sample Dataset

The included `sample_data.csv` contains 100 employee records with:
- **Target:** `salary` (numeric)
- **Features:** age, income, education, experience, job_title, location, department, gender, hiring_date, performance_score
- **Issues to explore:** Missing values, categorical variables, date column

## Customization

### Missing Value Strategies
Edit the imputation strategies in Section 4.1:
```python
imputation_strategies = {
    'column_name': 'mean',    # or 'median', 'mode', 'constant_value'
}
```

### Outlier Handling
Choose from options in Section 4.2:
- Remove outliers
- Cap at percentiles (winsorize)
- Transform (log, sqrt)
- Keep as-is

### Feature Engineering
Add custom features in Section 4 or modify analysis as needed.

## Output

The workflow generates:
- `data_cleaned.csv` - Cleaned dataset
- `plots/*.png` - Saved visualizations
- Summary statistics and insights in the notebook

## Checklist

Before modeling, ensure:
- [ ] Data loaded successfully
- [ ] Missing values handled
- [ ] Duplicates addressed
- [ ] Outliers documented
- [ ] Distributions understood
- [ ] Correlations identified
- [ ] Target analyzed
- [ ] Key findings documented

## Requirements

- Python 3.8+
- pandas >= 1.5.0
- numpy >= 1.21.0
- matplotlib >= 3.5.0
- seaborn >= 0.12.0
- scipy >= 1.7.0
- scikit-learn >= 1.0.0 (optional, for feature importance)

## License

This workflow is provided as-is for educational and analytical purposes.