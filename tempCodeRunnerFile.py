    pd.DataFrame({
        "method":         ["Random Search","Cluster-Guided"],
        "mean_idi_ratio": [round(np.mean(rb),4), round(np.mean(rc),4)],
        "std_idi_ratio":  [round(np.std(rb),4),  round(np.std(rc),4)],
        "improvement_pct":[0.0, round(imp,1)],
        "wilcoxon_p":     ["N/A", p],
    }).to_csv("results/exp1_adult_summary.csv", index=False)
