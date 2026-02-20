# SOC Change Signal-to-Noise (SNR) Analysis Framework

This repository contains the full analytical and modelling pipeline used to quantify **Soil Organic Carbon (SOC) change detectability** under different sampling designs, with explicit treatment of temporal uncertainty and spatial aggregation effects.

The workflow is designed to address a central inference problem in SOC monitoring:

> Under realistic sampling constraints and measurement noise, to what extent can SOC change be distinguished from background variability?

To that end, the pipeline integrates:

- Signal reconstruction at specific depth (e.g. PCHIP interpolation)
- Sampling design organization (point / paired / SRS)
- Model-based SOC change estimation with uncertainty (via Random forest model trained)
- Signal-to-noise ratio (SNR) diagnostics
- Spatial aggregation scaling
- SHAP-based interpretability

All analytical artefacts used in the final manuscript are tracked using Data Version Control (DVC) metadata to ensure full provenance of input material, derived evaluation metrics, trained model artefacts and manuscript figures.

## Repository Structure

- `model_fit.py`  
  Core model fitting routines implemented as callable functions for SOC change estimation.

- `trees_rf.py`  
  Random Forest implementation used for SOC change inference.

- `*.ipynb`  
  Analysis and modelling workflows:

  - **Indexed notebooks** with a numeric prefix (e.g. `01_`, `07a_`, `13d_`)
    constitute the formal analytical pipeline and are ordered according to
    execution sequence. Outputs generated from these notebooks are used in
    the final manuscript.

  - **Non-indexed notebooks** (e.g. `test_*`, scenario or comparison
    notebooks) are used for exploratory analyses, methodological diagnostics,
    or intermediate model checks. Results produced from these notebooks are
    not included in the final manuscript and are retained solely for
    transparency and development traceability.


## Reproducibility

The analytical pipeline integrates SOC observations from the following inventories:

- LUCAS Topsoil Survey ([2009 and 2012](https://esdac.jrc.ec.europa.eu/content/lucas-2009-topsoil-data), [2015](https://esdac.jrc.ec.europa.eu/content/lucas-2015-topsoil-data), [2018](https://esdac.jrc.ec.europa.eu/content/lucas-2018-topsoil-data))
- [Parcelas (COS and INES)](https://www.miteco.gob.es/content/dam/miteco/es/biodiversidad/servicios/banco-datos-naturaleza/2-cos/bbdd-cos.zip)
- German Agricultural Soil Inventory ([BZE-LW](https://doi.org/10.3220/DATA20200203151139))

Environmental covariates used for SOC change modelling are listed in **full covariate list - soc_change_snr.csv**. All listed covariates are openly accessible via the https://ecodatacube.eu, with the specific data version indicated in the corresponding file names.

The computational workflow begins with spatial–temporal overlay of SOC observations and covariates, as implemented in the `01_*` notebooks. Execution of the indexed notebook sequence reproduces the full analytical pipeline used in the manuscript.

