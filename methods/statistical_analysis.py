import sns
from matplotlib import pyplot as plt
import ast
import itertools
import seaborn as sns
import numpy
import numpy as np
from matplotlib import pyplot as plt

from scipy.stats import mannwhitneyu, friedmanchisquare

from Orange.evaluation import compute_CD, graph_ranks



import scipy.stats as stats
import scikit_posthocs as sp

hv_values = {
    "conST": [0.26, 0.37, 0.42, 0.33, 0.30, 0.23, 0.38, 0.38, 0.51, 0.45, 0.44, 0.39],
    "DeepST": [0.53, 0.44, 0.49, 0.46, 0.33, 0.33, 0.49, 0.50, 0.59, 0.43, 0.52, 0.49],
    "SpaceFlow": [0.31, 0.26, 0.21, 0.21, 0.20, 0.17, 0.26, 0.28, 0.35, 0.34, 0.25, 0.29],
    "STAGATE": [0.59, 0.43, 0.43, 0.43, 0.39, 0.51, 0.30, 0.54, 0.51, 0.54, 0.57, 0.51],
    "SpaGCN": [0.43, 0.38, 0.37, 0.40, 0.30, 0.21, 0.38, 0.51, 0.48, 0.41, 0.39, 0.32],
    "GraphST": [0.43, 0.49, 0.42, 0.51, 0.43, 0.38, 0.60, 0.61, 0.63, 0.43, 0.48, 0.57],
    "Method-G": [0.53, 0.00, 0.43, 0.47, 0.40, 0.39, 0.60, 0.62, 0.66, 0.59, 0.47, 0.64],
    "Method-GS": [0.55, 0.41, 0.47, 0.60, 0.41, 0.39, 0.61, 0.59, 0.62, 0.46, 0.58, 0.63],
    "Method-GSD": [0.48, 0.33, 0.45, 0.50, 0.54, 0.38, 0.59, 0.59, 0.60, 0.44, 0.42, 0.42],
    "Method-GD": [0.52, 0.49, 0.47, 0.42, 0.32, 0.39, 0.55, 0.61, 0.64, 0.43, 0.46, 0.64],
    "Scatter": [0.47, 0.40, 0.48, 0.38, 0.39, 0.38, 0.47, 0.73, 0.66, 0.55, 0.46, 0.42],
    "ACT": [0.41, 0.42, 0.53, 0.31, 0.34, 0.37, 0.44, 0.59, 0.67, 0.36, 0.34, 0.21],
    "FACT": [0.32, 0.34, 0.49, 0.34, 0.43, 0.29, 0.83, 0.51, 0.43, 0.30, 0.48, 0.24],
    "Mix": [0.41, 0.42, 0.53, 0.52, 0.43, 0.37, 0.60, 0.51, 0.67, 0.59, 0.47, 0.52]
}
algorithms = [
    "conST", "DeepST", "SpaceFlow", "STAGATE", "SpaGCN", "GraphST",
    "Method-G", "Method-GS", "Method-GSD", "Method-GD",
    "Scatter", "ACT", "FACT", "Mix"
]


stat1, p_value1 = friedmanchisquare(*[hv_values[alg] for alg in algorithms])
mean_ranks1 = np.argsort(np.argsort([-np.mean(hv_values[alg]) for alg in algorithms])) + 1

print(p_value1)
num_runs = 5  # Adjust if needed
cd1 = compute_CD(mean_ranks1, num_runs)


p=0
ranks2=[]
for_csv=''
# HV VALUES
for ijk in mean_ranks1:
    pa=algorithms[p]
    p=p+1
    ranks2.append((ijk,pa))

ranks2.sort(key=lambda x: x[0])
for ijk in ranks2:
    print(ijk)




data = np.array(list(hv_values.values()))

# Perform Nemenyi post-hoc test
result = sp.posthoc_nemenyi_friedman(data.T)
mask = np.triu(np.ones_like(result, dtype=bool))

# Plot the results as a heatmap
plt.figure(figsize=(10, 8),dpi=300)
sns.heatmap(result, annot=True, fmt='.1f', cmap='coolwarm', mask=mask)

# Adjust axis labels to start from 1
ax = plt.gca()
ax.set_xticklabels(np.arange(1, 14 +1))
ax.set_yticklabels(np.arange(1, 14 +1))

plt.title('Nemenyi Test Results')

plt.show()