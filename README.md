## Installation
Cola requires an MPI installation compatible with mpi4py.

Clone the repository and navigate to the directory. 
Install the python package locally:
```
pip3 install -r requirements.txt
pip3 install -e .
```

## Partitioning and Running

From the cola-master directory: Use colatools to load the dataset and split across ranks
```
svm_source_dir=../data
local_data_dir=./data
dataset=mg
colatools --indir $svm_source_dir --outdir $local_data_dir \
    load mg \
    split --train 0.7 --seed 42 mg
```

To generate report plots:
```
export JOBLIB_CACHE_DIR='./cache'
mpiexec -n 6 python3 scripts/run_cola_experiments.py mg
```
