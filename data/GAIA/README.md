# GAIA MicroSS Data

Raw GAIA data is not redistributed.

Download GAIA-DataSet from:

https://github.com/CloudWise-OpenSource/GAIA-DataSet

For a full rerun of the integrated Stage B experiment, arrange the July 2021
inputs as:

```text
data/GAIA/
  GAIA_extracted/
    trace/trace/
    metric/metric/
  gaia_run/
    run/run_table_2021-07.csv
```

Most Stage B follow-up analyses do not require raw GAIA data. They read the
bundled aggregate outputs under:

```text
results/stage_b/gaia_integrated_experiment/
```

