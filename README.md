# Code files for Chen, Glaeser, and Wessel (2019)

## Dependencies
- All code is written on Python 3.7 (Anaconda distribution) and R 3.5.1
- Non-standard python packages:
    - cenpy (https://cenpy-devs.github.io/cenpy/index.html)
    - pyjanitor (https://pyjanitor.readthedocs.io/)
    - geopandas (http://geopandas.org/)
    - rpy2 (https://rpy2.github.io/doc/v3.0.x/html/index.html)
- R packages:
    - plm (https://cran.r-project.org/web/packages/plm/plm.pdf)
    - drdid (https://pedrohcgs.github.io/DRDID/)
        - Doubly-robust difference-in-differences
    - did (https://bcallaway11.github.io/did/)
        - Callaway--Sant'Anna difference-in-differences

## Data
The only datasets that are not automatically retrieved are
- ZIP-tract crosswalk (https://www.huduser.gov/portal/datasets/usps_crosswalk.html)
- ZCTA-level population (https://factfinder.census.gov/faces/tableservices/jsf/pages/productview.xhtml?pid=ACS_16_5YR_DP05&prodType=table)
