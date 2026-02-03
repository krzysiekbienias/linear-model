# LinearRegressionOLS

A minimal, educational implementation of **Ordinary Least Squares (OLS)** linear regression in pure NumPy.

The goal of this project is **clarity and correctness**, not feature completeness.  
It is intentionally built step by step, with tests and clean separation between:
- algebra / numerical core
- data ingestion (planned)
- diagnostics (planned)

---

## 📌 Current state

Implemented:
- `LinearRegressionOLS` class
- OLS fitting via **normal equations**
- optional intercept handling (`fit_intercept=True/False`)
- internal fitted state (`fitted_`)
- unit tests using **pytest**

Verified with exact, deterministic datasets (no noise).

---

## 🧠 Design philosophy

- **Numerical core uses NumPy only**
- No pandas inside the algebra engine
- Clean public API: `fit()`, later `predict()`
- Internal helpers kept private
- Intermediate matrices can be stored for debugging but are not part of the public API

This is closer to a *learning / reference implementation* than a production ML library.

---

## 🧪 Tests

Tests are written with **pytest** and cover:
- single-feature regression with intercept
- regression without intercept
- multi-feature regression
- refitting behavior (state overwrite)

Run tests with:
```bash
pytest -q