import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. BASIC DATA UNDERSTANDING
# ==========================================
def basic_overview(df):
    print("=" * 50)
    print("BASIC OVERVIEW")
    print("=" * 50)

    print("\n memory usage:")
    print(df.memory_usage(deep=True).sum() / 1024**2, "MB")

    print("\nShape:")
    print(df.shape)

    print("\nSize:")
    print(df.size)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData Types:")
    print(df.dtypes)

    print("\nInfo:")
    print(df.info())

    print("\nDescription:")
    print(df.describe())

    print("\nHead:")
    print(df.head())

    print("\nTail:")
    print(df.tail())


# ==========================================
# 2. DATA QUALITY CHECK
# ==========================================
def data_quality(df):
    print("=" * 50)
    print("DATA QUALITY CHECK")
    print("=" * 50)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nMissing Percentage:")
    print((df.isnull().sum() / len(df)) * 100)

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())

    print("\nDuplicate Percentage:")
    print((df.duplicated().sum() / len(df)) * 100)

    # Cardinality analaysis
    print("\nUnique Values(cardinality analysis):")
    print(df.nunique())

    print("\nUnique Value Counts:")
    for col in df.columns:
        vc = df[col].value_counts()
        vc_norm = df[col].value_counts(normalize=True)

        print(f"\n{col} - Unique Value Counts:")
        print(vc)

        print("\n percentage distribution:")
        print(vc_norm * 100)

        print("\nMost Frequent:", vc.idxmax())
        print("\nLeast Frequent:", vc.idxmin())

        imbalance_ratio = vc_norm.max() / vc_norm.min()
        print(f"\nImbalance Ratio for {col}:", imbalance_ratio)




# ==========================================
# 3. UNIVARIATE ANALYSIS
# ==========================================
def univariate_analysis(df):
    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        plt.figure(figsize=(8,4))
        plt.hist(df[col], bins=20)
        plt.title(f"Histogram - {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.show()


        plt.figure(figsize=(6,3))
        plt.boxplot(df[col].dropna(), vert=False)
        plt.title(f"Boxplot - {col}")
        plt.show()


# ==========================================
# 4. CATEGORICAL ANALYSIS
# ==========================================
def categorical_analysis(df):
    cat_cols = df.select_dtypes(include='object').columns

    for col in cat_cols:
        print(f"\nValue Counts for {col}")
        print(df[col].value_counts())

        plt.figure(figsize=(8,4))
        sns.countplot(x=df[col])
        plt.title(f"Countplot - {col}")
        plt.xticks(rotation=45)
        plt.show()


# ==========================================
# 5. BIVARIATE ANALYSIS
# ==========================================
def bivariate_analysis(df):
    numeric_cols = df.select_dtypes(include=np.number).columns

    if len(numeric_cols) >= 2:
        for i in range(len(numeric_cols)-1):
            x = numeric_cols[i]
            y = numeric_cols[i+1]

            plt.figure(figsize=(6,4))
            plt.scatter(df[x], df[y])
            plt.xlabel(x)
            plt.ylabel(y)
            plt.title(f"{x} vs {y}")
            plt.show()





# ==========================================
# 6. CORRELATION / FEATURE RELATIONSHIP
# ==========================================
def correlation_analysis(df):
    numeric_df = df.select_dtypes(include=np.number)

    corr = numeric_df.corr()

    print("\nCorrelation Matrix:")
    print(corr)

    plt.figure(figsize=(10,6))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()




# ==========================================
# 7. STATISTICAL ANALYSIS
# ==========================================
def statistical_analysis(df):
    numeric_df = df.select_dtypes(include=np.number)

    print("=" * 50)
    print("STATISTICAL ANALYSIS")
    print("=" * 50)


    # Measure of Central Tendency
    print("\nMean:")
    print(numeric_df.mean())

    print("\nMedian:")
    print(numeric_df.median())

    print("\nMode:")
    print(numeric_df.mode().iloc[0])


    # Measure of Spread
    print("\nVariance:")
    print(numeric_df.var())

    print("\nStandard Deviation:")
    print(numeric_df.std())


    # Measure of Distribution
    print("\nSkewness:")
    print(numeric_df.skew())

    print("\nKurtosis:")
    print(numeric_df.kurt())




# ==========================================
# 8. OUTLIER DETECTION (IQR)
# ==========================================
def outlier_detection(df):
    numeric_cols = df.select_dtypes(include=np.number).columns

    print("=" * 50)
    print("OUTLIER DETECTION")
    print("=" * 50)

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower) | (df[col] > upper)]

        print(f"{col}: {len(outliers)} outliers")






# ==========================================
# 9. FULL EDA PIPELINE
# ==========================================
def full_eda(df):
    basic_overview(df)
    data_quality(df)
    univariate_analysis(df)
    categorical_analysis(df)
    bivariate_analysis(df)
    correlation_analysis(df)
    statistical_analysis(df)
    outlier_detection(df)







