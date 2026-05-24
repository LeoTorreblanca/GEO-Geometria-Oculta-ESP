from classy import Class
import numpy as np

cosmo = Class()

params = {
    'h': 0.67,
    'omega_b': 0.022,
    'omega_cdm': 0.12,
    'A_s': 2.1e-9,
    'n_s': 0.965,
    'tau_reio': 0.054,
    'output': 'mPk',
    'z_max_pk': 2.5,
    'P_k_max_1/Mpc': 2.0
}

cosmo.set(params)
cosmo.compute()

sigma8_0 = cosmo.sigma8()

print("sigma8_0 =", sigma8_0)
print("z   D(z)   f(z)   fσ8(z)")

for z in [0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
    D = cosmo.scale_independent_growth_factor(z)
    f = cosmo.scale_independent_growth_factor_f(z)
    fs8 = f * sigma8_0 * D
    print(f"{z:4.2f}  {D:.6f}  {f:.6f}  {fs8:.6f}")

cosmo.struct_cleanup()
cosmo.empty()
