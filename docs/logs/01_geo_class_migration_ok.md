# 01 — GEO_CLASS migration OK

Se migró la nomenclatura histórica GDDv2 hacia nomenclatura pública GEO_CLASS.

Cambios principales:

- xi_gdd   -> geo_xi
- mu_gdd   -> geo_mu
- gdd_mode -> geo_mode
- test_gdd.ini -> geo_test.ini
- class_gddv2_* -> geo_growth_*

Validación realizada:

- CLASS recompila.
- classy importa correctamente.
- CLASS acepta parámetros geo_xi, geo_mu, geo_mode.
- compute() ejecuta correctamente.
- sigma8 responde.

Estado: OK.
