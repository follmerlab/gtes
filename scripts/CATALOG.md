# Script Catalog

Quick reference for all blog post Python scripts.

## Ground State

### Understanding Linear Regression: The Foundation of Data Analysis
**Date**: November 2025  
**Script**: `scripts/ground-state/understanding-linear-regression/generate_plots.py`  
**Output**: `content/ground-state/understanding-linear-regression/linear-regression-analysis.png`  
**Description**: Generates 2×2 panel figure with regression fit, residual plot, Q-Q plot, and histogram  
**Key Parameters**:
- `n_points = 25` (number of data points)
- `true_slope = 2.5` (known slope for synthetic data)
- `noise_std = 2.0` (Gaussian noise level)

**Run**: 
```bash
python scripts/ground-state/understanding-linear-regression/generate_plots.py
```

---

### Singular Value Decomposition and PCA: Practical Foundations
**Date**: November 2025  
**Script**: `scripts/ground-state/svd-pca-foundations/generate_plots.py`  
**Outputs**:
- `eigenvalue-demo.png` (2D transformation visualization)
- `svd-spectroscopy-demo.png` (time-resolved spectroscopy analysis)
- `denoising-demo.png` (low-rank approximation examples)
- `pca-demo.png` (PCA via covariance vs SVD comparison)

**Description**: Comprehensive set of demonstrations for SVD/PCA concepts  
**Key Parameters**:
- Spectroscopy: `n_energies = 200`, `n_times = 50`, `noise_level = 0.05`
- Kinetics: `k1 = 0.1`, `k2 = 0.02` (rate constants in ps⁻¹)

**Run**: 
```bash
python scripts/ground-state/svd-pca-foundations/generate_plots.py
```

---

## Excited State

*(Scripts for excited-state posts will be added here)*

---

## Utilities

### Regenerate All Figures
**Script**: `scripts/regenerate_all.py`  
**Description**: Master script to regenerate all blog post figures  

**Usage**:
```bash
# Regenerate all figures
python scripts/regenerate_all.py

# Regenerate specific post
python scripts/regenerate_all.py --post understanding-linear-regression
python scripts/regenerate_all.py --post svd-pca-foundations
```

---

## Adding New Scripts

When creating a new blog post with Python figures:

1. **Create directory structure**:
   ```bash
   mkdir -p scripts/[section]/[post-slug]
   ```

2. **Create `generate_plots.py`** with standard structure (see README.md)

3. **Add entry to this catalog** with:
   - Post title and date
   - Script path
   - Output file(s)
   - Brief description
   - Key adjustable parameters
   - Run command

4. **Update `scripts/regenerate_all.py`** to include new post in `POSTS` dict

5. **Test the script**:
   ```bash
   python scripts/[section]/[post-slug]/generate_plots.py
   ```

---

**Last Updated**: November 19, 2025
