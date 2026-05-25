import pandas as pd
import numpy as np

from scipy.stats import (
    shapiro,
    pearsonr,
    spearmanr,
    chi2_contingency,
    ttest_ind,
    mannwhitneyu,
    f_oneway,
    kruskal,
    levene
)

# =========================================================
# 1. GET COLUMN TYPES
# =========================================================

def get_column_types(df):

    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_cols = df.select_dtypes(
        include=['object', 'category', 'bool']
    ).columns.tolist()

    return numeric_cols, categorical_cols


# =========================================================
# 2. NORMALITY TEST
# =========================================================

def normality_test(df, numeric_cols, alpha=0.05):

    print("\n" + "=" * 70)
    print("NORMALITY TESTS")
    print("=" * 70)

    normality_results = {}

    for col in numeric_cols:

        data = df[col].dropna()

        if len(data) < 3:
            continue

        stat, p = shapiro(data)

        normality_results[col] = p

        print(f"\nCOLUMN : {col}")

        print("H0 : Data is normally distributed")
        print("H1 : Data is NOT normally distributed")

        print(f"Statistic : {stat:.4f}")
        print(f"P-value   : {p:.6f}")

        if p < alpha:

            print("Decision  : Reject H0")
            print("Result    : NOT Normally Distributed")

        else:

            print("Decision  : Fail to Reject H0")
            print("Result    : Normally Distributed")

    return normality_results


# =========================================================
# 3. NUMERIC vs NUMERIC
# =========================================================

def numeric_vs_numeric_analysis(
    df,
    numeric_cols,
    normality_results,
    alpha=0.05
):

    print("\n" + "=" * 70)
    print("NUMERIC vs NUMERIC")
    print("=" * 70)

    for i in range(len(numeric_cols)):

        for j in range(i + 1, len(numeric_cols)):

            col1 = numeric_cols[i]
            col2 = numeric_cols[j]

            temp = df[[col1, col2]].dropna()

            if len(temp) < 3:
                continue

            x = temp[col1]
            y = temp[col2]

            # ---------------------------------------------
            # Choose Test Automatically
            # ---------------------------------------------

            if (
                normality_results.get(col1, 0) > alpha and
                normality_results.get(col2, 0) > alpha
            ):

                test_name = "Pearson Correlation"

                stat, p = pearsonr(x, y)

            else:

                test_name = "Spearman Correlation"

                stat, p = spearmanr(x, y)

            print("\n" + "-" * 50)

            print(f"{col1} vs {col2}")

            print(f"Selected Test : {test_name}")

            print("H0 : No relationship exists")
            print("H1 : Significant relationship exists")

            print(f"Statistic : {stat:.4f}")
            print(f"P-value   : {p:.6f}")

            if p < alpha:

                print("Decision  : Reject H0")
                print("Result    : Significant relationship")

            else:

                print("Decision  : Fail to Reject H0")
                print("Result    : No significant relationship")


# =========================================================
# 4. CATEGORICAL vs CATEGORICAL
# =========================================================

def categorical_vs_categorical_analysis(
    df,
    categorical_cols,
    alpha=0.05
):

    print("\n" + "=" * 70)
    print("CATEGORICAL vs CATEGORICAL")
    print("=" * 70)

    for i in range(len(categorical_cols)):

        for j in range(i + 1, len(categorical_cols)):

            col1 = categorical_cols[i]
            col2 = categorical_cols[j]

            table = pd.crosstab(
                df[col1],
                df[col2]
            )

            if table.shape[0] < 2 or table.shape[1] < 2:
                continue

            chi2, p, dof, expected = chi2_contingency(table)

            print("\n" + "-" * 50)

            print(f"{col1} vs {col2}")

            print("Selected Test : Chi-Square Test")

            print("H0 : Variables are independent")
            print("H1 : Variables are associated")

            print(f"Chi2 Statistic : {chi2:.4f}")
            print(f"P-value        : {p:.6f}")

            if p < alpha:

                print("Decision       : Reject H0")
                print("Result         : Variables are associated")

            else:

                print("Decision       : Fail to Reject H0")
                print("Result         : Variables are independent")


# =========================================================
# 5. TWO GROUP TEST
# =========================================================

def two_group_test(groups, alpha=0.05):

    g1 = groups[0]
    g2 = groups[1]

    g1_normal = (
        shapiro(g1)[1] > alpha
        if len(g1) >= 3 else False
    )

    g2_normal = (
        shapiro(g2)[1] > alpha
        if len(g2) >= 3 else False
    )

    # -------------------------------------------------
    # Both Normal
    # -------------------------------------------------

    if g1_normal and g2_normal:

        lev_stat, lev_p = levene(g1, g2)

        # Equal Variance

        if lev_p > alpha:

            test_name = "Independent T-Test"

            stat, p = ttest_ind(g1, g2)

        else:

            test_name = "Mann-Whitney U Test"

            stat, p = mannwhitneyu(g1, g2)

    else:

        test_name = "Mann-Whitney U Test"

        stat, p = mannwhitneyu(g1, g2)

    return test_name, stat, p


# =========================================================
# 6. MULTI GROUP TEST
# =========================================================

def multi_group_test(groups, alpha=0.05):

    all_normal = True

    for grp in groups:

        if len(grp) >= 3:

            if shapiro(grp)[1] < alpha:
                all_normal = False

    if all_normal:

        test_name = "ANOVA"

        stat, p = f_oneway(*groups)

    else:

        test_name = "Kruskal-Wallis Test"

        stat, p = kruskal(*groups)

    return test_name, stat, p


# =========================================================
# 7. CATEGORICAL vs NUMERIC
# =========================================================

def categorical_vs_numeric_analysis(
    df,
    categorical_cols,
    numeric_cols,
    alpha=0.05
):

    print("\n" + "=" * 70)
    print("CATEGORICAL vs NUMERIC")
    print("=" * 70)

    for cat_col in categorical_cols:

        unique_groups = df[cat_col].dropna().unique()

        if len(unique_groups) < 2:
            continue

        for num_col in numeric_cols:

            groups = []

            for grp in unique_groups:

                values = df[
                    df[cat_col] == grp
                ][num_col].dropna()

                if len(values) > 0:
                    groups.append(values)

            if len(groups) < 2:
                continue

            # -----------------------------------------
            # Choose Test
            # -----------------------------------------

            if len(groups) == 2:

                test_name, stat, p = two_group_test(
                    groups,
                    alpha
                )

            else:

                test_name, stat, p = multi_group_test(
                    groups,
                    alpha
                )

            print("\n" + "-" * 50)

            print(f"{cat_col} vs {num_col}")

            print(f"Selected Test : {test_name}")

            print("H0 : Groups are equal")
            print("H1 : Significant differences exist")

            print(f"Statistic : {stat:.4f}")
            print(f"P-value   : {p:.6f}")

            if p < alpha:

                print("Decision  : Reject H0")

                print(
                    f"Result    : {cat_col} significantly affects {num_col}"
                )

            else:

                print("Decision  : Fail to Reject H0")

                print(
                    f"Result    : No significant effect detected"
                )


# =========================================================
# 8. MASTER FUNCTION
# =========================================================

def inferential_analysis(df, alpha=0.05):

    print("=" * 70)
    print("AUTOMATED INFERENTIAL ANALYSIS")
    print("=" * 70)

    # ---------------------------------------------
    # Get Column Types
    # ---------------------------------------------

    numeric_cols, categorical_cols = get_column_types(df)

    # ---------------------------------------------
    # Normality Tests
    # ---------------------------------------------

    normality_results = normality_test(
        df,
        numeric_cols,
        alpha
    )

    # ---------------------------------------------
    # Numeric vs Numeric
    # ---------------------------------------------

    numeric_vs_numeric_analysis(
        df,
        numeric_cols,
        normality_results,
        alpha
    )

    # ---------------------------------------------
    # Categorical vs Categorical
    # ---------------------------------------------

    categorical_vs_categorical_analysis(
        df,
        categorical_cols,
        alpha
    )

    # ---------------------------------------------
    # Categorical vs Numeric
    # ---------------------------------------------

    categorical_vs_numeric_analysis(
        df,
        categorical_cols,
        numeric_cols,
        alpha
    )

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETED")
    print("=" * 70)




