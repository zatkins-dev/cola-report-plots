import click
import numpy as np
import pandas as pd
import cola.communication as comm

from cola.dataset import load_dataset, load_dataset_by_rank
from cola.graph import define_graph_topology
from cola.cocoasolvers import configure_solver
from cola.algo import Cola
from cola.monitor import Monitor
import pickle
import os



@click.command()
@click.option('--dataset', type=click.STRING, help='The type of dataset.')
@click.option('--solvername', type=click.STRING, help='The name of solvers.')
@click.option('--algoritmname', type=click.STRING, help='The name of algorithm')
@click.option('--output_dir', type=click.STRING, default=None, help='Save metrics in the training.')
@click.option('--dataset_size', default='small', type=click.Choice(['small', 'all']), help='Size of dataset',)
@click.option('--datapoints', default=-1, type=click.INT, help='Number of datapoints of dataset to load',)
@click.option('--use_split_dataset', is_flag=True)
@click.option('--logmode', default='local', type=click.Choice(['local', 'global', 'all']),
              help='Log local or global information.')
@click.option('--split_by', default='samples', type=click.Choice(['samples', 'features']),
              help='Split data matrix by samples or features.')
@click.option('--max_global_steps', default=100, help='Maximum number of global steps.')
@click.option('--theta', type=float, help='Theta-approximate solution (if local_iters is not specified)')
@click.option('--local_iters', default=1.0, help='Theta-approximate solution in terms of local data pass')
@click.option('--random_state', default=42, help='Random state')
@click.option('--dataset_path', default=None, help='Path to dataset')
@click.option('--graph_topology', type=str, help='Graph topology of the network.')
@click.option('--n_connectivity', type=int, help='Connected Cycle.')
@click.option('--l1_ratio', type=float, default=1,help='l1 ratio in the ElasticNet')
@click.option('--lambda_', type=float, default=1e-5,help='Size of regularizer')
@click.option('--c', type=float, help='Constant in the LinearSVM.')
@click.option('--ckpt_freq', type=int, default=10, help='')
@click.option('--exit_time', default=1000.0, help='The maximum running time of a node.')
@click.option('--fit-intercept/--no-fit-intercept', is_flag=True)
@click.option('--normalize/--no-normalize', is_flag=True)
@click.option('--verbose', default=1, type=click.IntRange(0, 3, clamp=True), help='Verbosity level, 0 is silent, 3 is full debug')
def main(dataset, dataset_path, dataset_size, datapoints, use_split_dataset, split_by, random_state,
         algoritmname, max_global_steps, local_iters, solvername, output_dir, exit_time, lambda_, l1_ratio, theta,
         graph_topology, c, logmode, ckpt_freq, n_connectivity, fit_intercept, normalize, verbose):

    # Fix gamma = 1.0 according to:
    #   Adding vs. Averaging in Distributed Primal-Dual Optimization
    gamma = 1.0

    # Initialize process group
    comm.init_process_group('mpi')

    # Get rank of current process
    rank = comm.get_rank()
    world_size = comm.get_world_size()

    # Create graph with specified topology
    graph = define_graph_topology(
        world_size, graph_topology, n_connectivity=n_connectivity, verbose=verbose)

    if use_split_dataset:
        if not dataset_path:
            dataset_path = os.path.join('data', dataset, split_by, f'{world_size}')
        X, y, X_test, y_test = load_dataset_by_rank(dataset, rank, world_size, dataset_size, datapoints, split_by,
                                    dataset_path=dataset_path, random_state=random_state, verbose=verbose)
    else:
        X, y = load_dataset(dataset, rank, world_size, dataset_size, datapoints, split_by,
                            dataset_path=dataset_path, random_state=random_state, verbose=verbose)

    # Define subproblem
    solver = configure_solver(name=solvername, split_by=split_by, l1_ratio=l1_ratio,
                              lambda_=lambda_, C=c, random_state=random_state)

    # Add hooks to log and save metrics.
    if algoritmname != 'cola':
        output_dir = os.path.join(output_dir, algoritmname)
    if dataset:
        output_dir = os.path.join(output_dir, dataset, f'{world_size:0>2}', graph_topology)
    monitor = Monitor(output_dir, ckpt_freq=ckpt_freq, exit_time=exit_time, split_by=split_by, mode=logmode, verbose=verbose, Ak=X, Ak_test=X_test, y_test=y_test)
    
    # Run CoLA
    comm.barrier()
    if algoritmname == 'cola':
        model = Cola(gamma, solver, theta, fit_intercept, normalize)
        monitor.init(model, graph)
        model = model.fit(X, y, graph, monitor, max_global_steps, local_iters)
    else:
        raise NotImplementedError()
    
    # Show test stats
    if X_test is not None:
        monitor.show_test_statistics()
    
    # Save final model
    monitor.save(modelname='model.pickle', logname=f'result.csv')


if __name__ == '__main__':
    main()
