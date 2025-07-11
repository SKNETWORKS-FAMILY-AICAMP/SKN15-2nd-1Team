import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 설정
target_col = 'MonthlyMinutes'
group_col = 'Churn'
bins = 40

# 구간 경계 계산
min_val = df[target_col].min()
max_val = df[target_col].max()
bin_edges = np.linspace(min_val, max_val, bins+1)

# 라벨 만들기
labels = [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(len(bin_edges)-1)]

# 구간화
df[f'{target_col}_bin'] = pd.cut(df[target_col], bins=bin_edges, labels=labels, include_lowest=True)

# 상대도수 계산
rel_freq_df = pd.crosstab(
    df[f'{target_col}_bin'], df[group_col], normalize='columns'
).reset_index()

# x축 위치 인덱스
x = np.arange(len(rel_freq_df[f'{target_col}_bin']))

# y값
yes_vals = rel_freq_df['Yes']
no_vals  = rel_freq_df['No']

width = 0.4  # 막대 너비

fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(x - width/2, yes_vals, width, label='Yes', alpha=0.8)
ax.bar(x + width/2, no_vals, width, label='No', alpha=0.8)

ax.set_title(f"Relative Frequency of {target_col} (binned) by Churn")
ax.set_ylabel("Proportion")
ax.set_xlabel(target_col)
ax.set_xticks(x)
ax.set_xticklabels(rel_freq_df[f'{target_col}_bin'], rotation=90, fontsize=8)
ax.legend(title=group_col)

plt.tight_layout()
plt.savefig(f"{target_col}_relative.png", dpi=300, bbox_inches='tight')
plt.show()
