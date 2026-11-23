# Astrocyte Network Simulation
This python project simulates ion currents in astrocyte networks and is
based on [the computational modeling by Freund et al., 2024](https://doi.org/10.3389/fncel.2024.1474948).

## Installation
This project is tested with specific python and package versions:

| python       | 3.9.23 |
|--------------|--------|
| numpy        | 1.23.5 |
| matplotlib   | 3.9.4  |
| pandas       | 2.3.3  |
| pymatreader  | 1.1.0  |
| scikit-learn | 1.6.1  |
| seaborn      | 0.13.2 |
| matlabengine | 24.1.4 |
| pylatex      | 1.4.2  |
| kneed        | 0.8.5  |
| yellowbrick  | 1.5    |
| morphio      | 3.4.2  |
| networkx     | 3.2.1  |
| neurots      | 3.7.0  |

## Converting MATLAB files to .h5
The code uses MorphIO to read and write HDF5 files. If you have MATLAB data from [Rafaeli et al., 2021](https://doi.org/10.1002/glia.24044), you can convert them to HDF5 files:
1. Generate a directory structure like `{dataset_name}_matlab/{dataset_name}.mat`
2. In MATLAB, run
```
load('{dataset_name}_matlab/{dataset_name}.mat');
% pymatreader can't read tables, so we need a csv for that
writetable(positionData, '{dataset_name}_matlab/positionData.csv');
writetable(diameterData, '{dataset_name}_matlab/diameterData.csv');
```
3. In python, run:
```
from astropyte.dataset import Dataset
Dataset(name="{dataset_name}").from_matlab(f"{dataset_name}_matlab/").save("{dataset_name}")
```