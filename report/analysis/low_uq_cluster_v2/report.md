# Low-Uncertainty Hallucination: Clustering Analysis

## Data and label-blind input
- Valid joined records: 6662
- Inputs: within-model-dataset empirical percentiles of PPL, semantic entropy, and UMPIRE.
- Clustering never receives correctness, hallucination, rating, or hallucination type labels.

## K selection
- K=4 is the primary resolution: it separates consensus-low, consensus-high, and two semantic-entropy disagreement profiles.
- K=2--6 quality metrics are reported. K=2 is a coarser low/high split; K=4 has the lowest Davies-Bouldin index among the tested resolutions.

## Selected consensus-low cluster
- n=1820 (27.3% of valid records).
- Centroid percentiles: PPL=0.222, SE=0.221, UMPIRE=0.167.
- Hallucination rate=31.8%; error rate=25.1%.
- Low-UQ hallucination subset: 579 (8.7% of all valid records).
- Severe subset (also incorrect): 361 (5.4% of all valid records).

## Group bootstrap stability
- Replicates: 200; resampling unit: group_id within each model-dataset stratum.
- Low-cluster Jaccard median=0.912 (95% percentile interval 0.871--0.950).
- Full partition ARI median=0.856 (95% percentile interval 0.791--0.921).

## Artifacts
- sample_level_analysis.csv: percentiles, label-blind cluster profile, and extraction flags.
- cluster_centers.csv, cluster_summary.csv, cluster_quality.csv, bootstrap_stability.csv.
- low_uq_hallucinations.csv and low_uq_hallucinations_severe.csv for manual audit.
