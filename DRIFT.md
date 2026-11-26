# Drift Report

## Population Stability Index (PSI)

The drift report compares `data/reference.csv` against `data/current.csv`.

**Drift threshold:** 0.2

| Column | PSI | Missing (Reference) | Missing (Current) | Drifted |
|---|---:|---:|---:|---|
| weeks_of_cover | 1.217307 | 0.00% | 0.00% | Yes |
| supplier_lead_time_days | 3.317172 | 0.00% | 0.00% | Yes |
| orders_last_4w | 0.003736 | 0.00% | 0.00% | No |
| supplier_fill_rate | 3.714538 | 3.10% | 15.67% | Yes |
| therapeutic_class | 0.002271 | 0.00% | 0.00% | No |

## Interpretation

- `weeks_of_cover`: Significant distribution drift, indicating a substantial change in inventory coverage patterns.
- `supplier_lead_time_days`: Severe distribution drift, suggesting substantial changes in supplier delivery lead times.
- `orders_last_4w`: No meaningful distribution drift detected.
- `supplier_fill_rate`: Severe distribution drift, accompanied by a significant increase in missing values from 3.10% to 15.67%.
- `therapeutic_class`: No meaningful distribution drift detected.

## Recommendations for Dr. Mona

1. The weeks_of_cover distribution has changed significantly, suggesting that inventory coverage patterns may have shifted and should be investigated.
2. Supplier lead times show severe drift, which may indicate changes in supplier performance or delivery conditions.
3. Orders over the last four weeks and therapeutic class distributions remain stable, so these features currently show no significant distribution changes.
4. Supplier fill rate has the highest PSI (3.7145), and its missing rate increased from 3.10% to 15.67%, requiring investigation of data quality and supplier reporting.
5. The service should monitor drift regularly, investigate affected features, validate incoming data, and trigger model review or retraining when necessary rather than relying on potentially outdated predictions.
