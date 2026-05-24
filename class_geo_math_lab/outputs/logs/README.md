# GEO_CLASS outputs

This folder stores generated outputs from local GEO_CLASS validation scripts.

## Structure

- `csv/`: numerical outputs in CSV format.
- `plots/`: generated plots.
- `logs/`: execution logs and notes.

## First validated script

`geo_class_internal_smoke_test.py`

Purpose:

Validate that the modified CLASS engine accepts the GEO_CLASS parameter `geo_xi`
and produces a measurable response in `sigma8` and matter power spectrum `P(k)`.

Observed result:

- `geo_xi = 1.00` -> high growth response.
- `geo_xi = 0.07` -> suppressed growth response.
- `geo_xi = 0.00` -> stronger suppression.

Status: OK.
