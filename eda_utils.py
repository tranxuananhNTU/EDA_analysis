"""
EDA Utilities - Reusable functions for Exploratory Data Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# 1. DATA LOADING AND SETUP
# =============================================================================

def setup_environment():
    """Configure pandas, numpy, and matplotlib settings for EDA."""
    # Pandas display options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.float_format', '{:.2f}'.format)

    # Random seed for reproducibility
    np.random.seed(42)

    # Plotting style
    sns.set_style('whitegrid')
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12

    print("Environment configured successfully!")


def load_data(filepath, **kwargs):
    """
    Load dataset from file with basic validation.

    Parameters:
    -----------
    filepath : str
        Path to the data file (CSV, Excel, etc.)
    **kwargs : dict
        Additional arguments passed to pd.read_csv()

    Returns:
    --------
    tuple : (DataFrame, DataFrame copy)
    """
    df = pd.read_csv(filepath, **kwargs)
    df_original = df.copy()
    print(f"Data loaded successfully!")
    print(f"File: {filepath}")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    return df, df_original


# =============================================================================
# 2. DATA OVERVIEW FUNCTIONS
# =============================================================================

def data_overview(df):
    """
    Display comprehensive overview of the dataset.

    Parameters:
    -----------
    df : DataFrame
        Input dataframe

    Returns:
    --------
    dict : Summary statistics
    """
    print("=" * 60)
    print("DATA OVERVIEW")
    print("=" * 60)

    # Shape
    print(f"\nShape: {df.shape[0]} rows, {df.shape[1]} columns")

    # Data types
    print("\nData Types:")
    print(df.dtypes.value_counts())

    # Memory usage
    print(f"\nMemory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    return {
        'shape': df.shape,
        'dtypes': df.dtypes.value_counts().to_dict(),
        'memory_mb': df.memory_usage(deep=True).sum() / 1024**2
    }


def missing_values_analysis(df, plot=True, figsize=(12, 6)):
    """
    Analyze and visualize missing values.

    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    plot : bool
        Whether to create visualization
    figsize : tuple
        Figure size for plot

    Returns:
    --------
    DataFrame : Missing values summary table
    """
    # Calculate missing values
    missing = pd.DataFrame({
        'column': df.columns,
        'missing_count': df.isnull().sum().values,
        'missing_pct': (df.isnull().sum() / len(df) * 100).values,
        'dtype': df.dtypes.values
    })
    missing = missing.sort_values('missing_pct', ascending=False)
    missing = missing[missing['missing_count'] > 0]

    if len(missing) == 0:
        print("No missing values found!")
        return None

    print("=" * 60)
    print("MISSING VALUES ANALYSIS")
    print("=" * 60)
    print(f"\nColumns with missing values: {len(missing)}")
    print(f"Total missing values: {missing['missing_count'].sum()}")
    print(f"\n{missing.to_string(index=False)}")

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Heatmap
        plt.subplot(1, 2, 1)
        cols_with_missing = missing['column'].head(20).tolist()
        sns.heatmap(df[cols_with_missing].isnull(), cbar=True, yticklabels=False)
        plt.title('Missing Values Heatmap (Top 20 Columns)')
        plt.xlabel('Columns')

        # Bar plot
        plt.subplot(1, 2, 2)
        data = missing.head(15)
        bars = plt.barh(data['column'], data['missing_pct'])
        plt.xlabel('Missing Percentage (%)')
        plt.title('Missing Values by Column (Top 15)')
        plt.tight_layout()

        for bar, pct in zip(bars, data['missing_pct']):
            plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{pct:.1f}%', va='center')

        plt.tight_layout()
        plt.savefig('plots/01_missing_values.png', dpi=150, bbox_inches='tight')
        plt.show()

    return missing


def duplicate_analysis(df):
    """
    Analyze duplicate rows in the dataset.

    Parameters:
    -----------
    df : DataFrame

    Returns:
    --------
    dict : Duplicate statistics
    """
    duplicates = df.duplicated().sum()
    print("=" * 60)
    print("DUPLICATE ANALYSIS")
    print("=" * 60)
    print(f"\nTotal duplicate rows: {duplicates}")
    print(f"Percentage of duplicates: {duplicates / len(df) * 100:.2f}%")

    return {
        'duplicate_count': duplicates,
        'duplicate_pct': duplicates / len(df) * 100
    }


# =============================================================================
# 3. DESCRIPTIVE STATISTICS
# =============================================================================

def numerical_summary(df, exclude_cols=None):
    """
    Generate comprehensive summary for numerical features.

    Parameters:
    -----------
    df : DataFrame
    exclude_cols : list
        Columns to exclude from analysis

    Returns:
    --------
    DataFrame : Summary statistics table
    """
    if exclude_cols is None:
        exclude_cols = []

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

    if len(numeric_cols) == 0:
        print("No numerical columns found!")
        return None

    summary = pd.DataFrame(index=numeric_cols)

    for col in numeric_cols:
        data = df[col].dropna()
        summary.loc[col, 'dtype'] = df[col].dtype
        summary.loc[col, 'count'] = len(data)
        summary.loc[col, 'missing_pct'] = df[col].isnull().sum() / len(df) * 100
        summary.loc[col, 'mean'] = data.mean()
        summary.loc[col, 'median'] = data.median()
        summary.loc[col, 'std'] = data.std()
        summary.loc[col, 'min'] = data.min()
        summary.loc[col, 'max'] = data.max()
        summary.loc[col, 'range'] = data.max() - data.min()
        summary.loc[col, 'q1'] = data.quantile(0.25)
        summary.loc[col, 'q3'] = data.quantile(0.75)
        summary.loc[col, 'iqr'] = data.quantile(0.75) - data.quantile(0.25)
        summary.loc[col, 'skewness'] = data.skew()
        summary.loc[col, 'kurtosis'] = data.kurtosis()
        summary.loc[col, 'cv'] = data.std() / data.mean() if data.mean() != 0 else np.nan

    print("=" * 60)
    print("NUMERICAL FEATURES SUMMARY")
    print("=" * 60)
    print(f"\nNumerical columns: {len(numeric_cols)}")
    print(f"\n{summary.round(2).to_string()}")

    return summary


def categorical_summary(df, exclude_cols=None, max_categories=10):
    """
    Generate comprehensive summary for categorical features.

    Parameters:
    -----------
    df : DataFrame
    exclude_cols : list
        Columns to exclude
    max_categories : int
        Maximum categories to display

    Returns:
    --------
    DataFrame : Summary table
    """
    if exclude_cols is None:
        exclude_cols = []

    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    cat_cols = [col for col in cat_cols if col not in exclude_cols]

    if len(cat_cols) == 0:
        print("No categorical columns found!")
        return None

    summary = pd.DataFrame(index=cat_cols)

    for col in cat_cols:
        data = df[col].dropna()
        summary.loc[col, 'dtype'] = df[col].dtype
        summary.loc[col, 'count'] = len(data)
        summary.loc[col, 'missing_pct'] = df[col].isnull().sum() / len(df) * 100
        summary.loc[col, 'unique'] = data.nunique()
        summary.loc[col, 'mode'] = data.mode().iloc[0] if len(data.mode()) > 0 else np.nan
        summary.loc[col, 'mode_freq'] = data.value_counts().iloc[0] if len(data) > 0 else np.nan
        summary.loc[col, 'mode_pct'] = (data.value_counts().iloc[0] / len(data) * 100) if len(data) > 0 else np.nan

    print("=" * 60)
    print("CATEGORICAL FEATURES SUMMARY")
    print("=" * 60)
    print(f"\nCategorical columns: {len(cat_cols)}")
    print(f"\n{summary.round(2).to_string()}")

    # Print value counts for each column
    print("\n" + "=" * 60)
    print("VALUE COUNTS")
    print("=" * 60)
    for col in cat_cols:
        print(f"\n--- {col} ---")
        vc = df[col].value_counts().head(max_categories)
        print(vc.to_string())
        if df[col].nunique() > max_categories:
            print(f"... and {df[col].nunique() - max_categories} more categories")

    return summary


# =============================================================================
# 4. OUTLIER DETECTION AND HANDLING
# =============================================================================

def detect_outliers_iqr(df, column):
    """
    Detect outliers using IQR method.

    Returns:
    --------
    dict : Outlier statistics
    """
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = df[(df[column] < lower) | (df[column] > upper)]

    return {
        'column': column,
        'lower_bound': lower,
        'upper_bound': upper,
        'outlier_count': len(outliers),
        'outlier_pct': len(outliers) / len(df) * 100,
        'outlier_indices': outliers.index.tolist()
    }


def detect_outliers_zscore(df, column, threshold=3):
    """
    Detect outliers using Z-score method.
    """
    z_scores = np.abs(stats.zscore(df[column].dropna()))
    outliers = df[column].dropna()[z_scores > threshold]

    return {
        'column': column,
        'threshold': threshold,
        'outlier_count': len(outliers),
        'outlier_pct': len(outliers) / len(df) * 100,
        'outlier_indices': outliers.index.tolist()
    }


def outliers_summary(df, method='iqr', exclude_cols=None):
    """
    Generate outliers summary for all numerical columns.

    Parameters:
    -----------
    df : DataFrame
    method : str
        'iqr' or 'zscore'
    exclude_cols : list

    Returns:
    --------
    DataFrame : Outliers summary
    """
    if exclude_cols is None:
        exclude_cols = []

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

    results = []
    for col in numeric_cols:
        if method == 'iqr':
            result = detect_outliers_iqr(df, col)
        else:
            result = detect_outliers_zscore(df, col)
        results.append(result)

    summary = pd.DataFrame(results)
    summary = summary[['column', 'outlier_count', 'outlier_pct', 'lower_bound', 'upper_bound']]
    summary = summary.sort_values('outlier_pct', ascending=False)

    print("=" * 60)
    f"OUTLIERS SUMMARY ({method.upper()} method)"
    print("=" * 60)
    print(f"\n{summary.round(2).to_string(index=False)}")

    return summary


def cap_outliers(df, column, lower_pct=0.01, upper_pct=0.99):
    """
    Cap outliers at specified percentiles (winsorization).
    """
    lower = df[column].quantile(lower_pct)
    upper = df[column].quantile(upper_pct)

    df_capped = df.copy()
    df_capped[column] = df_capped[column].clip(lower, upper)

    print(f"Capped {column}: [{lower:.2f}, {upper:.2f}]")
    return df_capped


# =============================================================================
# 5. UNIVARIATE ANALYSIS
# =============================================================================

def plot_numeric_distribution(df, column, figsize=(14, 4), save=False):
    """
    Create distribution plots for a numeric column.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Histogram with KDE
    plt.subplot(1, 3, 1)
    sns.histplot(df[column].dropna(), kde=True, bins=30)
    plt.title(f'Histogram: {column}')
    plt.xlabel(column)

    # Boxplot
    plt.subplot(1, 3, 2)
    sns.boxplot(x=df[column].dropna())
    plt.title(f'Boxplot: {column}')

    # Violin plot
    plt.subplot(1, 3, 3)
    sns.violinplot(x=df[column].dropna())
    plt.title(f'Violin: {column}')

    plt.tight_layout()

    if save:
        plt.savefig(f'plots/univariate_{column}.png', dpi=150, bbox_inches='tight')

    plt.show()


def plot_categorical_distribution(df, column, figsize=(14, 5), max_categories=15, save=False):
    """
    Create distribution plots for a categorical column.
    """
    value_counts = df[column].value_counts().head(max_categories)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Bar plot
    plt.subplot(1, 2, 1)
    sns.barplot(x=value_counts.values, y=value_counts.index)
    plt.title(f'Frequency: {column}')
    plt.xlabel('Count')

    # Pie chart (if reasonable number of categories)
    plt.subplot(1, 2, 2)
    if len(value_counts) <= 8:
        plt.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%')
        plt.title(f'Proportion: {column}')
    else:
        # For many categories, show top 8 and group rest
        top_8 = value_counts.head(8)
        other = value_counts[8:].sum()
        if other > 0:
            pie_data = pd.concat([top_8, pd.Series({'Other': other})])
        else:
            pie_data = top_8
        plt.pie(pie_data.values, labels=pie_data.index, autopct='%1.1f%%')
        plt.title(f'Proportion: {column} (Top 8)')

    plt.tight_layout()

    if save:
        plt.savefig(f'plots/univariate_{column}.png', dpi=150, bbox_inches='tight')

    plt.show()


def univariate_analysis(df, target=None, max_categories=15, save_plots=False):
    """
    Perform univariate analysis on all columns.

    Parameters:
    -----------
    df : DataFrame
    target : str
        Name of target column (to exclude from analysis)
    max_categories : int
        Maximum categories to show in plots
    save_plots : bool
        Whether to save plots to files
    """
    print("=" * 60)
    print("UNIVARIATE ANALYSIS")
    print("=" * 60)

    # Numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if target:
        numeric_cols = [col for col in numeric_cols if col != target]

    print(f"\nAnalyzing {len(numeric_cols)} numerical columns...")
    for col in numeric_cols:
        print(f"\n--- {col} ---")
        print(f"Mean: {df[col].mean():.2f}")
        print(f"Median: {df[col].median():.2f}")
        print(f"Std: {df[col].std():.2f}")
        print(f"Skewness: {df[col].skew():.2f}")
        plot_numeric_distribution(df, col, save=save_plots)

    # Categorical columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if target:
        cat_cols = [col for col in cat_cols if col != target]

    print(f"\nAnalyzing {len(cat_cols)} categorical columns...")
    for col in cat_cols:
        print(f"\n--- {col} ---")
        print(f"Unique values: {df[col].nunique()}")
        print(f"Mode: {df[col].mode().iloc[0]}")
        plot_categorical_distribution(df, col, max_categories=max_categories, save=save_plots)


# =============================================================================
# 6. BIVARIATE / MULTIVARIATE ANALYSIS
# =============================================================================

def correlation_analysis(df, target=None, figsize=(12, 10), save=False):
    """
    Create correlation matrix and heatmap.

    Parameters:
    -----------
    df : DataFrame
    target : str
        Target column name
    figsize : tuple
    save : bool

    Returns:
    --------
    DataFrame : Correlation matrix
    """
    numeric_df = df.select_dtypes(include=[np.number])

    if len(numeric_df.columns) < 2:
        print("Not enough numeric columns for correlation analysis!")
        return None

    corr_matrix = numeric_df.corr()

    print("=" * 60)
    print("CORRELATION ANALYSIS")
    print("=" * 60)

    # Plot heatmap
    plt.figure(figsize=figsize)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                cmap='coolwarm', center=0, square=True)
    plt.title('Correlation Matrix')
    plt.tight_layout()

    if save:
        plt.savefig('plots/02_correlation_matrix.png', dpi=150, bbox_inches='tight')

    plt.show()

    # Target correlations
    if target and target in corr_matrix.columns:
        print(f"\n--- Correlations with {target} ---")
        target_corr = corr_matrix[target].drop(target).sort_values(key=abs, ascending=False)
        print(target_corr.to_string())

        # Plot target correlations
        plt.figure(figsize=(10, max(6, len(target_corr) * 0.3)))
        colors = ['green' if x > 0 else 'red' for x in target_corr.values]
        plt.barh(target_corr.index, target_corr.values, color=colors)
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        plt.xlabel('Correlation Coefficient')
        plt.title(f'Feature Correlations with {target}')
        plt.tight_layout()

        if save:
            plt.savefig('plots/03_target_correlations.png', dpi=150, bbox_inches='tight')

        plt.show()

    return corr_matrix


def plot_scatter_with_target(df, feature, target, figsize=(8, 6), save=False):
    """
    Create scatter plot of feature vs target with regression line.
    """
    plt.figure(figsize=figsize)

    # Scatter + regression line
    sns.regplot(x=feature, y=target, data=df, scatter_kws={'alpha': 0.5})
    plt.title(f'{feature} vs {target}')

    # Calculate correlation
    corr = df[[feature, target]].corr().iloc[0, 1]
    plt.annotate(f'Correlation: {corr:.3f}', xy=(0.05, 0.95),
                 xycoords='axes fraction', fontsize=10)

    plt.tight_layout()

    if save:
        plt.savefig(f'plots/scatter_{feature}_vs_{target}.png', dpi=150, bbox_inches='tight')

    plt.show()


def pairplot_analysis(df, columns=None, target=None, corner=True, save=False):
    """
    Create pairplot for selected columns.
    """
    if columns is None:
        # Select top correlated features with target
        if target:
            numeric_df = df.select_dtypes(include=[np.number])
            if target in numeric_df.columns:
                corr = numeric_df.corr()[target].drop(target).abs().sort_values(ascending=False)
                columns = [target] + corr.head(6).index.tolist()
            else:
                columns = numeric_df.columns[:7].tolist()
        else:
            numeric_df = df.select_dtypes(include=[np.number])
            columns = numeric_df.columns[:7].tolist()

    print(f"Creating pairplot for {len(columns)} columns...")

    # Create pairplot
    g = sns.pairplot(df[columns].dropna(), corner=corner, diag_kind='kde')
    g.fig.suptitle('Pairplot of Key Features', y=1.02)

    if save:
        g.savefig('plots/04_pairplot.png', dpi=150, bbox_inches='tight')

    plt.show()


def numeric_vs_categorical(df, numeric_col, categorical_col, figsize=(12, 5), save=False):
    """
    Analyze relationship between numeric and categorical variables.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Boxplot
    plt.subplot(1, 3, 1)
    order = df.groupby(categorical_col)[numeric_col].median().sort_values().index
    sns.boxplot(x=categorical_col, y=numeric_col, data=df, order=order)
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Boxplot: {numeric_col} by {categorical_col}')

    # Violin plot
    plt.subplot(1, 3, 2)
    sns.violinplot(x=categorical_col, y=numeric_col, data=df, order=order)
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Violin: {numeric_col} by {categorical_col}')

    # Bar plot (mean with CI)
    plt.subplot(1, 3, 3)
    sns.barplot(x=categorical_col, y=numeric_col, data=df, order=order)
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Mean: {numeric_col} by {categorical_col}')

    plt.tight_layout()

    if save:
        plt.savefig(f'plots/num_vs_cat_{numeric_col}_{categorical_col}.png',
                    dpi=150, bbox_inches='tight')

    plt.show()

    # Group statistics
    print(f"\n--- {numeric_col} by {categorical_col} ---")
    group_stats = df.groupby(categorical_col)[numeric_col].agg(['mean', 'std', 'count'])
    print(group_stats.round(2).to_string())


def target_analysis(df, target, save_plots=False):
    """
    Comprehensive analysis of the target variable.

    Parameters:
    -----------
    df : DataFrame
    target : str
        Target column name
    save_plots : bool
    """
    print("=" * 60)
    print(f"TARGET ANALYSIS: {target}")
    print("=" * 60)

    if target not in df.columns:
        print(f"Error: Target column '{target}' not found!")
        return

    # Target distribution
    print(f"\nTarget Statistics:")
    print(f"  Mean: {df[target].mean():.2f}")
    print(f"  Median: {df[target].median():.2f}")
    print(f"  Std: {df[target].std():.2f}")
    print(f"  Min: {df[target].min():.2f}")
    print(f"  Max: {df[target].max():.2f}")
    print(f"  Skewness: {df[target].skew():.2f}")
    print(f"  Kurtosis: {df[target].kurtosis():.2f}")

    # Target distribution plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    plt.subplot(1, 3, 1)
    sns.histplot(df[target].dropna(), kde=True, bins=30)
    plt.title(f'Target Distribution: {target}')

    plt.subplot(1, 3, 2)
    sns.boxplot(x=df[target].dropna())
    plt.title(f'Target Boxplot')

    plt.subplot(1, 3, 3)
    # QQ plot for normality check
    from scipy.stats import probplot
    probplot(df[target].dropna(), dist="norm", plot=plt)
    plt.title('Q-Q Plot (Normality Check)')

    plt.tight_layout()

    if save_plots:
        plt.savefig('plots/05_target_distribution.png', dpi=150, bbox_inches='tight')

    plt.show()

    # Normality test
    stat, p_value = stats.shapiro(df[target].dropna().sample(min(5000, len(df))))
    print(f"\nShapiro-Wilk Normality Test:")
    print(f"  Statistic: {stat:.4f}")
    print(f"  P-value: {p_value:.4f}")
    if p_value < 0.05:
        print("  Result: Target is NOT normally distributed (p < 0.05)")
    else:
        print("  Result: Target appears normally distributed (p >= 0.05)")

    # Top correlated features with target
    numeric_df = df.select_dtypes(include=[np.number])
    if target in numeric_df.columns:
        corr = numeric_df.corr()[target].drop(target).sort_values(key=abs, ascending=False)
        print(f"\nTop 10 Correlated Features with {target}:")
        print(corr.head(10).round(3).to_string())


# =============================================================================
# 7. STATISTICAL TESTS
# =============================================================================

def normality_test(data, column):
    """
    Perform Shapiro-Wilk normality test.
    """
    sample = data[column].dropna().sample(min(5000, len(data)))
    stat, p_value = stats.shapiro(sample)

    return {
        'column': column,
        'statistic': stat,
        'p_value': p_value,
        'normal': p_value > 0.05
    }


def t_test_by_category(df, numeric_col, categorical_col):
    """
    Perform t-test comparing numeric values between two groups.
    """
    groups = df[categorical_col].unique()
    if len(groups) != 2:
        print(f"Warning: t-test requires exactly 2 groups. Found {len(groups)} groups.")
        return None

    group1 = df[df[categorical_col] == groups[0]][numeric_col].dropna()
    group2 = df[df[categorical_col] == groups[1]][numeric_col].dropna()

    stat, p_value = stats.ttest_ind(group1, group2)

    print(f"T-test: {numeric_col} by {categorical_col}")
    print(f"  Group 1 ({groups[0]}): mean={group1.mean():.2f}, n={len(group1)}")
    print(f"  Group 2 ({groups[1]}): mean={group2.mean():.2f}, n={len(group2)}")
    print(f"  T-statistic: {stat:.4f}")
    print(f"  P-value: {p_value:.4f}")
    print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")

    return {'statistic': stat, 'p_value': p_value}


def anova_test(df, numeric_col, categorical_col):
    """
    Perform one-way ANOVA test.
    """
    groups = df[categorical_col].unique()
    group_data = [df[df[categorical_col] == g][numeric_col].dropna() for g in groups]

    stat, p_value = stats.f_oneway(*group_data)

    print(f"ANOVA: {numeric_col} by {categorical_col}")
    print(f"  Groups: {len(groups)}")
    print(f"  F-statistic: {stat:.4f}")
    print(f"  P-value: {p_value:.4f}")
    print(f"  Significant difference: {'Yes' if p_value < 0.05 else 'No'}")

    return {'statistic': stat, 'p_value': p_value}


def chi_square_test(df, col1, col2):
    """
    Perform chi-square test of independence.
    """
    contingency = pd.crosstab(df[col1], df[col2])
    stat, p_value, dof, expected = stats.chi2_contingency(contingency)

    print(f"Chi-square test: {col1} vs {col2}")
    print(f"  Chi-square statistic: {stat:.4f}")
    print(f"  P-value: {p_value:.4f}")
    print(f"  Degrees of freedom: {dof}")
    print(f"  Independent: {'No' if p_value < 0.05 else 'Yes'}")

    return {'statistic': stat, 'p_value': p_value, 'dof': dof}


# =============================================================================
# 8. FEATURE IMPORTANCE (PREVIEW)
# =============================================================================

def feature_importance_preview(df, target, n_top=15, figsize=(10, 6)):
    """
    Quick feature importance using Random Forest.
    Note: This is exploratory; proper feature selection should be done in modeling.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder

    # Prepare data
    df_encoded = df.copy()

    # Encode categorical columns
    cat_cols = df_encoded.select_dtypes(include=['object', 'category']).columns
    le = LabelEncoder()
    for col in cat_cols:
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

    # Remove rows with missing target
    df_encoded = df_encoded.dropna(subset=[target])

    # Fill remaining missing values
    df_encoded = df_encoded.fillna(df_encoded.median())

    # Split features and target
    X = df_encoded.drop(columns=[target])
    y = df_encoded[target]

    # Fit model
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)

    # Get feature importance
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True)

    # Plot
    plt.figure(figsize=figsize)
    plt.barh(importance['feature'].tail(n_top), importance['importance'].tail(n_top))
    plt.xlabel('Importance')
    plt.title(f'Top {n_top} Feature Importance (Random Forest)')
    plt.tight_layout()
    plt.savefig('plots/06_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("Feature Importance Summary:")
    print(importance.sort_values('importance', ascending=False).head(n_top).to_string(index=False))

    return importance


# =============================================================================
# 9. SUMMARY REPORT
# =============================================================================

def generate_summary_report(df, target=None):
    """
    Generate a comprehensive summary report.
    """
    report = []
    report.append("=" * 60)
    report.append("EDA SUMMARY REPORT")
    report.append("=" * 60)

    # Basic info
    report.append(f"\nDataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    report.append(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # Missing values
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    report.append(f"\nColumns with Missing Values: {len(missing_cols)}")
    if len(missing_cols) > 0:
        for col, count in missing_cols.items():
            report.append(f"  - {col}: {count} ({count/len(df)*100:.1f}%)")

    # Duplicates
    duplicates = df.duplicated().sum()
    report.append(f"\nDuplicate Rows: {duplicates} ({duplicates/len(df)*100:.1f}%)")

    # Numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    report.append(f"\nNumeric Columns: {len(numeric_cols)}")

    # Categorical columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    report.append(f"Categorical Columns: {len(cat_cols)}")

    # High cardinality categorical
    high_card = [col for col in cat_cols if df[col].nunique() > 50]
    if high_card:
        report.append(f"\nHigh Cardinality Columns (>50 unique): {high_card}")

    # Skewed numeric
    skewed = [col for col in numeric_cols if abs(df[col].skew()) > 1]
    if skewed:
        report.append(f"\nHighly Skewed Columns (|skew| > 1):")
        for col in skewed:
            report.append(f"  - {col}: skew = {df[col].skew():.2f}")

    # Target info
    if target and target in df.columns:
        report.append(f"\n--- TARGET: {target} ---")
        report.append(f"  Mean: {df[target].mean():.2f}")
        report.append(f"  Std: {df[target].std():.2f}")
        report.append(f"  Range: [{df[target].min():.2f}, {df[target].max():.2f}]")
        report.append(f"  Skewness: {df[target].skew():.2f}")

    report_text = "\n".join(report)
    print(report_text)

    return report_text


# =============================================================================
# 10. DATA CLEANING UTILITIES
# =============================================================================

def clean_data(df, missing_strategy='median', drop_duplicates=True, verbose=True):
    """
    Basic data cleaning pipeline.

    Parameters:
    -----------
    df : DataFrame
    missing_strategy : str
        'median', 'mean', 'mode', or 'drop'
    drop_duplicates : bool
    verbose : bool

    Returns:
    --------
    DataFrame : Cleaned dataframe
    """
    df_clean = df.copy()

    if verbose:
        print("Starting data cleaning...")
        print(f"Initial shape: {df_clean.shape}")

    # Drop duplicates
    if drop_duplicates:
        before = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        if verbose:
            print(f"Dropped {before - len(df_clean)} duplicate rows")

    # Handle missing values
    if missing_strategy != 'drop':
        # Numeric columns
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].isnull().any():
                if missing_strategy == 'median':
                    fill_value = df_clean[col].median()
                elif missing_strategy == 'mean':
                    fill_value = df_clean[col].mean()
                else:
                    fill_value = 0
                df_clean[col].fillna(fill_value, inplace=True)
                if verbose:
                    print(f"Filled {col} missing values with {missing_strategy}: {fill_value:.2f}")

        # Categorical columns
        cat_cols = df_clean.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            if df_clean[col].isnull().any():
                fill_value = df_clean[col].mode().iloc[0]
                df_clean[col].fillna(fill_value, inplace=True)
                if verbose:
                    print(f"Filled {col} missing values with mode: {fill_value}")
    else:
        before = len(df_clean)
        df_clean = df_clean.dropna()
        if verbose:
            print(f"Dropped {before - len(df_clean)} rows with missing values")

    if verbose:
        print(f"Final shape: {df_clean.shape}")

    return df_clean


def save_cleaned_data(df, filepath='data_cleaned.csv'):
    """
    Save cleaned dataframe to CSV.
    """
    df.to_csv(filepath, index=False)
    print(f"Cleaned data saved to {filepath}")
    print(f"Shape: {df.shape}")