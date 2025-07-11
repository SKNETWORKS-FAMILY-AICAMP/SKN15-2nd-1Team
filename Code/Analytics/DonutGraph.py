import matplotlib.pyplot as plt
import seaborn as sns

# 대상 컬럼
target_col = 'HandsetPrice'

# 데이터 준비
labels = df[target_col].value_counts().index
sizes = df[target_col].value_counts(normalize=True) * 100

# 색상 자동 생성
palette = sns.color_palette('pastel', n_colors=len(labels))
colors = palette.as_hex()  # matplotlib 호환 색상

# 도넛 차트
fig, ax = plt.subplots(figsize=(6,6))
wedges, texts, autotexts = ax.pie(
    sizes,
    labels=labels,
    colors=colors,
    autopct='%1.1f%%',
    startangle=90,
    pctdistance=0.75,
    wedgeprops=dict(width=0.8),
    shadow=True
)

for autotext in autotexts:
    autotext.set_color('black')
    autotext.set_fontsize(11)

plt.title(f"{target_col} Distribution", fontsize=14)
plt.tight_layout()
plt.savefig('my_plot.png', dpi=300, bbox_inches='tight')
plt.show()
