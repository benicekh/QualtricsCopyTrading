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
    baseline_treatment=(2, 0, 0),
    alpha=0.05,
    power=0.8
)

# Display the table
print(sample_size_table)