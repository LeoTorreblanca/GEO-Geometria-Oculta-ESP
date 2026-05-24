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
    'z_max_pk': 2.0
}

cosmo.set(params)
cosmo.compute()

z_vals = np.linspace(0, 1.5, 10)

print("z   fσ8")
for z in z_vals:
    print(z, cosmo.sigma8() * cosmo.scale_independent_growth_factor_f(z))

cosmo.struct_cleanup()
cosmo.empty()
