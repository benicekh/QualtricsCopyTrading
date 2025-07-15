import pandas as pd
import numpy as np
from statsmodels.stats.power import TTestIndPower

#####################################################################################################
# Power Calculator
#####################################################################################################

def calculate_sample_size_table(df, variable, baseline_treatment, alpha=0.05, power=0.9, min_group_size=5):
    """
    Calculate required sample sizes to detect treatment differences vs. baseline.

    Parameters:
        df (pd.DataFrame): Input DataFrame
        variable (str): Outcome variable name (e.g., 'copiedCRRA_rank')
        baseline_treatment (tuple): (pathseries, nameTreatment, followersTreatment)
        alpha (float): Significance level (default 0.05)
        power (float): Desired power level (default 0.9)
        min_group_size (int): Minimum group size to include in calculation (default 5)

    Returns:
        pd.DataFrame: Pivoted sample size table
    """
    results = []
    phases = df['phase'].unique()
    pathseries_list = df['pathseries'].unique()
    name_treatments = df['nameTreatment'].unique()
    followers_treatments = df['followersTreatment'].unique()

    for phase in sorted(phases):
        group_base = df[
            (df['phase'] == phase) &
            (df['pathseries'] == baseline_treatment[0]) &
            (df['nameTreatment'] == baseline_treatment[1]) &
            (df['followersTreatment'] == baseline_treatment[2])
        ][variable].dropna()

        if len(group_base) < min_group_size:
            continue

        for path in pathseries_list:
            for nt in name_treatments:
                for ft in followers_treatments:
                    group_test = df[
                        (df['phase'] == phase) &
                        (df['pathseries'] == path) &
                        (df['nameTreatment'] == nt) &
                        (df['followersTreatment'] == ft)
                    ][variable].dropna()

                    if len(group_test) >= min_group_size:
                        mean_diff = group_test.mean() - group_base.mean()
                        pooled_sd = np.sqrt((group_test.var() + group_base.var()) / 2)
                        if pooled_sd > 0:
                            d = mean_diff / pooled_sd
                            try:
                                sample_size = TTestIndPower().solve_power(
                                    effect_size=abs(d), alpha=alpha, power=power
                                )
                            except:
                                sample_size = np.nan
                        else:
                            sample_size = np.nan
                    else:
                        sample_size = np.nan

                    results.append({
                        'phase': phase,
                        'pathseries': path,
                        'nameTreatment': nt,
                        'followersTreatment': ft,
                        'required_sample_size': sample_size
                    })

    sample_size_df = pd.DataFrame(results)

    table_sample_size = sample_size_df.pivot_table(
        values='required_sample_size',
        index='phase',
        columns=['pathseries', 'nameTreatment', 'followersTreatment'],
        aggfunc='first'
    )

    return table_sample_size

sample_size_table = calculate_sample_size_table(
    panel_df_clean,
    variable='copiedCRRA_rank',
    baseline_treatment=(2, 0, 0), # Baseline group: pathseries=2, nameTreatment=0, followersTreatment=0
    alpha=0.05,
    power=0.8
)

# Display the table
print(sample_size_table)

def calculate_sample_size_phase_comparison(df, variable, phase_0=0, phase_9=9, alpha=0.05, power=0.8, min_group_size=5):
    """
    For each treatment combination, calculate the required sample size to detect a difference
    between phase 0 and phase 9 using a two-sample t-test.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        variable (str): Outcome variable name (e.g., 'copiedCRRA_rank').
        phase_0 (int): First phase to compare (default 0).
        phase_9 (int): Second phase to compare (default 9).
        alpha (float): Significance level for power analysis (default 0.05).
        power (float): Desired power level (default 0.8).
        min_group_size (int): Minimum group size to compute effect size (default 5).

    Returns:
        pd.DataFrame: Required sample size per group for each treatment combination.
    """
    results = []

    combinations = df[['pathseries', 'nameTreatment', 'followersTreatment']].drop_duplicates()

    for _, row in combinations.iterrows():
        path, nt, ft = row['pathseries'], row['nameTreatment'], row['followersTreatment']

        group0 = df[
            (df['phase'] == phase_0) &
            (df['pathseries'] == path) &
            (df['nameTreatment'] == nt) &
            (df['followersTreatment'] == ft)
        ][variable].dropna()

        group9 = df[
            (df['phase'] == phase_9) &
            (df['pathseries'] == path) &
            (df['nameTreatment'] == nt) &
            (df['followersTreatment'] == ft)
        ][variable].dropna()

        if len(group0) >= min_group_size and len(group9) >= min_group_size:
            mean_diff = group9.mean() - group0.mean()
            pooled_sd = np.sqrt((group0.var(ddof=1) + group9.var(ddof=1)) / 2)

            if pooled_sd > 0:
                effect_size = mean_diff / pooled_sd
                try:
                    required_n = TTestIndPower().solve_power(
                        effect_size=abs(effect_size),
                        alpha=alpha,
                        power=power
                    )
                except:
                    required_n = np.nan
            else:
                required_n = np.nan
        else:
            required_n = np.nan

        results.append({
            'pathseries': path,
            'nameTreatment': nt,
            'followersTreatment': ft,
            'n_phase0': len(group0),
            'n_phase9': len(group9),
            'effect_size': effect_size if pooled_sd > 0 else np.nan,
            'required_sample_size_per_group': required_n
        })

    return pd.DataFrame(results)

sample_size_df = calculate_sample_size_phase_comparison(
    df=panel_df_clean,
    variable='copiedCRRA_rank',
    phase_0=0,
    phase_9=9,
    alpha=0.05,
    power=0.9
)

print(sample_size_df)