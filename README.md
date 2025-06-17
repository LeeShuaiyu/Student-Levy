# Student-t Lévy Increment Analysis

A Python toolkit for numerical analysis of Student-t–driven Lévy process increments:

* Fast & accurate computation of characteristic functions
* Numerical inversion to PDF via FFT or quadrature
* Quality metrics (KS, Wasserstein-1, average log-likelihood)
* Plotting helpers and demo script

---

## 📦 Installation

1. **Clone this repository**

   ```bash
   git clone https://github.com/your-username/student-levy.git
   cd student-levy
   ```

2. **Create a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   *requirements.txt* should include:

   ```
   numpy
   scipy
   matplotlib
   ```

---

## 🔧 Usage

### 1. Compute characteristic functions

```python
from cf import phi_student_unit, phi_levy
import numpy as np

u = np.linspace(-10, 10, 1001)
ν = 3.0
# unit-scale Student-t CF
φ1 = phi_student_unit(u, nu=ν)
# Lévy increment CF with h=0.1, μ=0, σ=1
φh = phi_levy(u, h=0.1, nu=ν)
```

### 2. Invert to PDF via FFT

```python
from inversion import density_fft
x, f = density_fft(h=0.05, nu=1.5, window='hann', n_grid=2**14)
```

### 3. Invert to PDF via quadrature

```python
import numpy as np
from inversion import density_quad
x = np.linspace(-5, 5, 501)
f_quad = density_quad(h=0.05, nu=1.5, x=x, U=100, N_u=8193)
```

### 4. Compute sample-vs-PDF metrics

```python
from metrics import ks_w1, avg_loglik
import numpy as np

samples = np.random.standard_t(df=ν, size=10000) * np.sqrt(0.05)
ks_stat, p_val, w1 = ks_w1(samples, x, f)
avg_ll = avg_loglik(samples, x, f)
print(f"KS={ks_stat:.4f}, p={p_val:.4f}, W1={w1:.4f}, avg_ll={avg_ll:.4f}")
```

### 5. Plotting helpers

```python
import matplotlib.pyplot as plt
from plotting import plot_density, compare_with_limit
from scipy.stats import cauchy

# density via FFT
plot_density(x, f, label='FFT density', adaptive=True)
# compare with Cauchy limit for ν=1
scale = 0.05 * np.sqrt(1)
compare_with_limit(x, f, pdf_limit=lambda z: cauchy.pdf(z, scale=scale))
plt.show()
```

### 6. Demo script

Run `demo.py` to see a complete end-to-end example:

```bash
python demo.py
```

This will:

1. Compute and plot the FFT-based density for ν=1, h=0.01
2. Overlay the Cauchy limit
3. Print the normalization ∫f dx

---

## 📁 Project Structure

```
.
├── cf.py              # Characteristic functions
├── inversion.py       # FFT & quadrature density inversion
├── metrics.py         # KS, Wasserstein, log-likelihood
├── plotting.py        # Matplotlib helpers
├── windows.py         # Frequency-domain window functions
├── demo.py            # End-to-end usage example
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

---

## ✍️ Contributing

1. Fork the repo
2. Create your feature branch: `git checkout -b feature/foo`
3. Commit your changes: `git commit -m "Add foo feature"`
4. Push to the branch: `git push origin feature/foo`
5. Open a Pull Request

---

## ⚖️ License

This project is released under the [MIT License](LICENSE).
