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

    # outputs que nos interesan
    'output': 'mPk',
    'P_k_max_1/Mpc': 2.0,
    'z_max_pk': 2.0
}

cosmo.set(params)
cosmo.compute()

# ejemplo: P(k) a z=0
k = 0.1
z = 0.0
pk = cosmo.pk(k, z)

print("P(k=0.1, z=0) =", pk)

cosmo.struct_cleanup()
cosmo.empty()
