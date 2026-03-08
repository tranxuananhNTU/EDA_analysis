# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an Exploratory Data Analysis (EDA) workflow toolkit for small datasets (<10K rows) with numeric target columns (regression scenarios). It provides a comprehensive, beginner-friendly analysis pipeline.

## Commands

### Setup
```bash
pip install -r requirements.txt
```

### Run the EDA Notebook
```bash
jupyter notebook eda_analysis.ipynb
```

## Architecture

### Core Files

| File | Purpose |
|------|---------|
| `eda_utils.py` | Reusable utility functions for EDA (20+ functions) |
| `eda_analysis.ipynb` | Main Jupyter notebook with 10-section workflow |
| `sample_data.csv` | Sample dataset (100 employee records) for testing |

### Workflow Structure (10 Sections)

1. **Setup** - Library imports, configuration, data loading
2. **Data Overview** - Shape, dtypes, missing values, duplicates
3. **Descriptive Statistics** - Numerical/categorical summaries
4. **Data Cleaning** - Missing value handling, outliers, type conversions
5. **Univariate Analysis** - Distribution plots for each feature
6. **Bivariate Analysis** - Correlations, scatterplots, pairplots
7. **Target Analysis** - Target distribution, correlations, transformations
8. **Advanced Analysis** - Distribution comparisons, time series, feature importance
9. **Statistical Tests** - Normality, t-test, ANOVA, chi-square
10. **Summary** - Key findings, modeling recommendations

### Key Function Categories (eda_utils.py)

**Data Loading & Setup:**
- `setup_environment()` - Configure pandas, matplotlib, random seed
- `load_data(filepath)` - Load CSV with validation

**Overview & Statistics:**
- `data_overview(df)` - Shape, dtypes, memory
- `missing_values_analysis(df)` - Missing values with heatmap visualization
- `numerical_summary(df)` / `categorical_summary(df)` - Detailed statistics
- `outliers_summary(df)` - IQR/Z-score outlier detection

**Visualization:**
- `plot_numeric_distribution(df, col)` - Histogram, boxplot, violin
- `plot_categorical_distribution(df, col)` - Bar chart, pie chart
- `correlation_analysis(df, target)` - Heatmap with target correlations
- `pairplot_analysis(df)` - Feature pairplot grid

**Target Analysis:**
- `target_analysis(df, target)` - Comprehensive target variable analysis
- `plot_scatter_with_target(df, feature, target)` - Scatter with regression line
- `numeric_vs_categorical(df, num, cat)` - Box/violin/bar plots

**Statistical Tests:**
- `normality_test(df, col)` - Shapiro-Wilk test
- `t_test_by_category(df, num, cat)` - Independent t-test
- `anova_test(df, num, cat)` - One-way ANOVA
- `chi_square_test(df, col1, col2)` - Chi-square independence test

**Data Cleaning:**
- `clean_data(df, strategy)` - Basic cleaning pipeline
- `cap_outliers(df, col)` - Winsorization
- `save_cleaned_data(df, filepath)` - Export cleaned CSV

## Configuration

In the notebook, update these variables for your dataset:
```python
FILEPATH = 'your_data.csv'      # Path to your CSV file
TARGET_COLUMN = 'target'         # Name of target variable
EXCLUDE_COLS = ['id']            # Columns to exclude from analysis
DATE_COLUMN = 'date'             # Optional date column for time series
```

## Output Files

- `data_cleaned.csv` - Cleaned dataset ready for modeling
- `plots/*.png` - Saved visualizations (numbered 01-06)
- Summary statistics in notebook cells

## Design Patterns

- All visualization functions accept `save=False` parameter to save to `plots/` directory
- Analysis functions return DataFrames or dicts for further programmatic use
- Statistical test functions print results and return dict with test statistics
- The notebook is designed to be run sequentially from top to bottom

## Dependencies

Core: pandas, numpy, matplotlib, seaborn, scipy
Optional: scikit-learn (for feature importance preview), plotly (interactive plots)