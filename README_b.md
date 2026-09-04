# Data provenance

The real-data analysis uses the **El Niño sea-surface temperature dataset**
(monthly SST, 1950–2010, 12 monthly columns per year) as distributed with
the `statsmodels` Python package:

```python
import statsmodels.api as sm
d = sm.datasets.elnino.load_pandas().data
```

Original source: NOAA/NCEP (public domain). No raw data files are stored in
this repository; the dataset is fetched from the installed `statsmodels`
distribution, which guarantees that every user runs the analysis on exactly
the same data.
