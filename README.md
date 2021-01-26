## Installation
Cola requires an MPI installation compatible with mpi4py.

Clone the repository and navigate to the directory. 
Install the python package locally:
```
pip3 install -r requirements.txt
pip3 install -e .
```

## Partitioning and Running

From the `cola-report-plots` directory: Use colatools to load the dataset and split across ranks
```
colatools load inv split --train 24 --seed 42 inv
```

To generate report plots:
```
export JOBLIB_CACHE_DIR='./cache'
mpiexec -n 16 python3 scripts/run_cola_experiments.py inv
```
