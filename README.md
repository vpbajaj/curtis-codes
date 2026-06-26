# Curtis Orbital Mechanics — algorithms in Python

Python ports of the MATLAB code in **Appendix C** (n-body numerical
integration) and **Appendix D** (the numbered algorithms) of Howard D. Curtis,
*Orbital Mechanics for Engineering Students*. Each script is a faithful,
standalone translation that uses only the standard library plus NumPy (Appendix
C also uses SciPy's ODE solver and matplotlib for plotting), and each
reproduces the corresponding worked example or figure printed in the book.

## Layout

```
curtis/
  constants.py                         Appendix A physical data (mu, radii, SOI, ...)
  algorithms/
    alg_3_1_kepler_E.py                Alg 3.1  Kepler's equation (Newton)
    alg_3_2_kepler_H.py                Alg 3.2  Kepler's equation, hyperbola
    stumpff.py                         D.4      Stumpff functions C(z), S(z)
    alg_3_3_kepler_U.py                Alg 3.3  Universal Kepler's equation
    lagrange_coefficients.py           D.6      Lagrange f, g and derivatives
    alg_3_4_rv_from_r0v0.py            Alg 3.4  State vector after time Δt
    alg_4_1_coe_from_sv.py             Alg 4.1  Orbital elements from state vector
    alg_4_2_sv_from_coe.py             Alg 4.2  State vector from orbital elements
    alg_5_1_gibbs.py                   Alg 5.1  Gibbs' method
    alg_5_2_lambert.py                 Alg 5.2  Lambert's problem
    julian_day.py                      D.12     Julian day number (J0)
    alg_5_3_lst.py                     Alg 5.3  Local sidereal time
    alg_5_4_rv_from_observe.py         Alg 5.4  State vector from radar observations
    alg_5_5_gauss.py                   Alg 5.5/5.6  Gauss' method + iterative improvement
    month_planet_names.py             D.16     Month / planet name lookup
    alg_8_1_planet_elements_and_sv.py  Alg 8.1  Planet state vector at an epoch
    alg_8_2_interplanetary.py          Alg 8.2  Interplanetary trajectory (patched conic)
  appendix_c/
    accel_3body.py                     C.1      Planar 3-body ODE right-hand side
    threebody.py                       C.2      Integrate (solve_ivp) + Figures 2.5/2.6
tests/
  test_algorithms.py                   Verifies every Appendix D algorithm vs the book
  test_appendix_c.py                   Checks COM motion / momentum for the 3-body solver
```

## Running

Activate the project's virtual environment, then run any algorithm's worked
example directly:

```bash
source /Users/vishalbajaj/vishal_codes/astronomy/venv/bin/activate
python -m curtis.algorithms.alg_3_4_rv_from_r0v0
python -m curtis.algorithms.alg_8_2_interplanetary
python -m curtis.appendix_c.threebody          # integrates and writes Figures 2.5/2.6
```

Run the test suite (each test checks the algorithm against the numbers printed
in the book):

```bash
python -m pytest tests/ -q
```

## Notes on book errata reproduced here

* **Algorithm 5.6 / Eq 5.114 (Gauss).** The printed appendix code has the
  `rho3` denominator as `(tau^2 - tau1^2)`, but the derivation in Section 5.10
  (Example 5.11, Step 9) and the appendix's own printed output both use
  `(tau^2 - tau3^2)`. The corrected form is used so the result matches the book;
  see the comment in `alg_5_5_gauss.py`.
* **Example 8.8 driver.** The book's MATLAB driver prints the arrival-planet
  position magnitude as `norm(R1)` (the departure planet) — a typo. The Python
  example prints the correct `norm(R2)`; all other values match the book.
