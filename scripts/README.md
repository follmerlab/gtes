# Blog Post Python Scripts

This directory contains all Python scripts used to generate figures and demonstrations for blog posts on Ground Truths & Excited States.

## Organization

Scripts are organized by blog section and post:

```
scripts/
├── ground-state/
│   ├── understanding-linear-regression/
│   │   └── generate_plots.py
│   └── svd-pca-foundations/
│       └── generate_plots.py
└── excited-state/
    └── [future posts]
```

## Running Scripts

All scripts are designed to be run from the repository root:

```bash
# Activate Python environment
source .venv/bin/activate  # or: conda activate your-env

# Run a specific post's script
python scripts/ground-state/understanding-linear-regression/generate_plots.py
python scripts/ground-state/svd-pca-foundations/generate_plots.py
```

## Script Naming Convention

- **`generate_plots.py`**: Main script that generates all figures for the post
- **`example_*.py`**: Standalone examples or demonstrations
- **`analysis_*.py`**: Data analysis scripts (when applicable)

## Requirements

All scripts require the Python environment with these packages:
- numpy
- matplotlib
- scipy
- scikit-learn

Install via:
```bash
pip install numpy matplotlib scipy scikit-learn
```

## Output

Scripts save figures directly to the post's content directory:
- `content/ground-state/[post-name]/[figure-name].png`
- `content/excited-state/[post-name]/[figure-name].png`

This ensures figures are automatically included in the Hugo build.

## Using Scripts for Teaching/Sharing

Each script is:
1. **Self-contained**: All parameters and setup at the top
2. **Well-documented**: Clear comments explaining each section
3. **Reproducible**: Fixed random seeds where applicable
4. **Modifiable**: Easy to adjust parameters for different demonstrations

To share a script with collaborators or students:
1. Copy the specific script file
2. Update output paths as needed
3. The script will work standalone with standard scientific Python packages

## Best Practices for New Scripts

When adding scripts for new posts:

1. **Create a new directory** for each post:
   ```bash
   mkdir -p scripts/[section]/[post-slug]
   ```

2. **Name the main script** `generate_plots.py` for consistency

3. **Include a docstring** at the top with:
   - Purpose
   - Author
   - Related blog post title/link
   - Date

4. **Set random seeds** for reproducibility:
   ```python
   np.random.seed(42)
   ```

5. **Print progress** messages:
   ```python
   print("Generating figure 1...")
   print("  ✓ Saved figure-name.png")
   ```

6. **Save to post directory**:
   ```python
   output_dir = 'content/[section]/[post-name]/'
   ```

## Example Script Structure

```python
"""
[Post Title] - Figure Generation
================================

Description of what this script does.

Author: Your Name
Blog Post: [Post Title]
Date: [Month Year]
"""

import numpy as np
import matplotlib.pyplot as plt

# Set random seed
np.random.seed(42)

# Output directory
output_dir = 'content/ground-state/post-name/'

print("Generating figures...")
print("="*60)

# Figure 1
print("1. Generating [description]...")
# ... figure code ...
plt.savefig(f'{output_dir}figure-name.png', dpi=200, bbox_inches='tight')
plt.close()
print("  ✓ Saved figure-name.png")

print("="*60)
print("All figures generated successfully!")
```

## Current Scripts

### Ground State Section

#### Understanding Linear Regression
**Script**: `understanding-linear-regression/generate_plots.py`

Generates comprehensive regression diagnostic plot with:
- Scatter plot with fitted line and confidence intervals
- Residual plot
- Q-Q plot for normality testing
- Histogram of residuals

**Output**: `linear-regression-analysis.png` (7×5.25 inches @ 200 DPI)

**Parameters to adjust**:
- `n_points`: Number of data points (default: 25)
- `true_slope`: True slope of relationship (default: 2.5)
- `true_intercept`: True intercept (default: 1.0)
- `noise_std`: Standard deviation of noise (default: 2.0)

#### SVD and PCA Foundations
**Script**: `svd-pca-foundations/generate_plots.py`

Generates four demonstration figures:
1. **Eigenvalue decomposition**: Geometric visualization of eigenvectors
2. **SVD spectroscopy**: Synthetic time-resolved spectroscopy analysis
3. **Denoising**: Low-rank approximation for noise reduction
4. **PCA demo**: Comparison of covariance and SVD methods

**Outputs**:
- `eigenvalue-demo.png`
- `svd-spectroscopy-demo.png`
- `denoising-demo.png`
- `pca-demo.png`

**Parameters to adjust**:
- Spectroscopy: `n_energies`, `n_times`, `noise_level`
- Kinetics: `k1`, `k2` (rate constants)
- PCA: `n_samples` (number of data points)

---

*For questions or issues with scripts, contact [your email] or open an issue on GitHub.*
