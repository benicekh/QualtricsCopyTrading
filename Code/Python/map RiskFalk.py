df = panel_df_clean_bot[panel_df_clean_bot['phase'] == 0].copy()

# Step 1: Check distribution of CRRA_rank (1 = highest risk, 5 = lowest risk)
riskrank_counts = df['CRRA_rank'].value_counts(normalize=True).sort_index()
df["RiskFalk"] = pd.to_numeric(df["RiskFalk"], errors="coerce")
print("\nCRRA_rank distribution (proportion per category):")
print(riskrank_counts)

# Step 2: Calculate cumulative quantiles
quantiles = riskrank_counts.cumsum().values
print("\nCumulative quantiles for binning:", quantiles)

# Step 3: Handle RiskFalk ties by ranking them
df['RiskFalk_ranked'] = df['RiskFalk'].rank(method='first')

# Step 4: Bin RiskFalk into 5 categories matching CRRA_rank distribution
# We use labels [5,4,3,2,1] so the highest RiskFalk gets category 1, lowest gets 5
df['RiskFalk_cat'] = pd.qcut(
    df['RiskFalk_ranked'],
    q=[0] + list(quantiles),
    labels=[5, 4, 3, 2, 1]  # reverse the labels here
)

# Step 5: Convert to integer
df['RiskFalk_cat'] = df['RiskFalk_cat'].astype(int)

# Step 6: Drop helper column
df = df.drop(columns=['RiskFalk_ranked'])

# Step 7: Check final distribution
print("\nMapped RiskFalk categories distribution (after reversing):")
print(df['RiskFalk_cat'].value_counts().sort_index())

print("\nFinal table comparing counts:")
comparison = pd.DataFrame({
    'CRRA_rank': df['CRRA_rank'].value_counts().sort_index(),
    'RiskFalk_cat': df['RiskFalk_cat'].value_counts().sort_index()
})
print(comparison)

# === Diagnostic bar plot comparing distributions ===
plt.figure(figsize=(8, 6))
df['CRRA_rank'].value_counts(normalize=True).sort_index().plot(kind='bar', alpha=0.6, label='CRRA_rank')
df['RiskFalk_cat'].value_counts(normalize=True).sort_index().plot(kind='bar', alpha=0.6, label='RiskFalk_cat', color='orange')
plt.xlabel('Category')
plt.ylabel('Proportion')
plt.title('Comparison of CRRA_rank and RiskFalk_cat distributions')
plt.legend()
plt.tight_layout()
plt.show()
# Calculate Pearson correlation
corr_pearson = df[['CRRA_rank', 'RiskFalk_cat']].corr(method='pearson').loc['CRRA_rank', 'RiskFalk_cat']
# Calculate Spearman correlation (rank-based, nonparametric)
corr_spearman = df[['CRRA_rank', 'RiskFalk_cat']].corr(method='spearman').loc['CRRA_rank', 'RiskFalk_cat']
# Print the results
print("\nCorrelation between CRRA_rank and RiskFalk_cat:")
print(f"Pearson correlation:  {corr_pearson:.3f}")
print(f"Spearman correlation: {corr_spearman:.3f}")