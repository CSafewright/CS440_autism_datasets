
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
from statsmodels.nonparametric.smoothers_lowess import lowess
import io

# Load cleaned data
df = pd.read_csv('autism_cleaned_updated.csv')

# Define bins for Sample Size
bins = [0, 10000, 500000, 5000000, np.inf]
labels = ['0-10,000', '10,001-500,000', '500,001-5,000,000', '5,000,001+']
df['Sample Size Bin'] = pd.cut(df['Sample Size'], bins=bins, labels=labels, include_lowest=True)

# Map bin labels to colors
color_map = {
    '0-10,000': 'green',
    '10,001-500,000': 'yellow',
    '500,001-5,000,000': 'blue',
    '5,000,001+': 'red'
}
colors = df['Sample Size Bin'].map(color_map)

# Create figure
fig, ax = plt.subplots(figsize=(12,7))

# Scatter plot
ax.scatter(
    df['Year Published'],
    df['ASD Prevalence Estimate per 1,000'],
    c=colors,
    alpha=0.7
)

lowess_fit = lowess(
    df['ASD Prevalence Estimate per 1,000'],
    df['Year Published'],
    frac=0.3
)
ax.plot(lowess_fit[:,0], lowess_fit[:,1], color='black', linewidth=2)

# Legend for Sample Size bins only
patches = [mpatches.Patch(color=color_map[label], label=label) for label in labels]
ax.legend(handles=patches, title='Sample Size', loc='best')

ax.set_xlabel("Year Published")
ax.set_ylabel("ASD Prevalence Estimate per 1,000")
ax.set_title("Global Autism Prevalence by Year (Sample Size Color)")
ax.grid(True)


fig.savefig("autism_plot.png", bbox_inches='tight')
#plt.show()
plt.close(fig)
