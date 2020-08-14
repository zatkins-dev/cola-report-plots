from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds
import numpy as np
import os
import click


class MatrixEditor(object):
    def __init__(self, indir=None, outdir=None, seed=None, debug=False):
        self.debug = debug
        self.indir = os.path.abspath(indir or '../data')
        self.outdir = os.path.abspath(outdir or 'data')
        self.data = None
        self.y = None
        self.index = None
        self.s = None
        self.Vh = None
        self.seed = seed

    @property
    def n_features(self):
        if self.data is None:
            return None 
        return self.data.shape[1]
    
    @property
    def n_points(self):
        if self.data is None:
            return None 
        return self.data.shape[0]
        

pass_editor = click.make_pass_decorator(MatrixEditor)

@click.group(chain=True)
@click.option('--basepath', default='.')
@click.option('--indir', default=os.path.join('..', 'data'))
@click.option('--outdir', default='data')
@click.option('--seed', type=click.INT, default=None)
@click.option('-v', is_flag=True)
@click.pass_context
def cli(ctx, basepath, indir, outdir, seed, v):
    indir = os.path.join(basepath, indir)
    outdir = os.path.join(basepath, outdir)
    ctx.obj = MatrixEditor(indir, outdir, seed, v) 

@cli.command('load')
@click.argument('dataset')
@click.option('--confirm', is_flag=True)
@pass_editor
def load(editor, dataset, confirm):
    if editor.data is not None and not confirm:
        print("There is existing data, pass --confirm flag to load anyway")
        return False
    
    if '.svm' not in dataset:
        dataset += '.svm'
    path = os.path.join(editor.indir, dataset)
    assert os.path.exists(path), f"SVM file '{path}' not found"
    old_data = editor.data.copy() if editor.data is not None else None
    old_y = editor.y.copy() if editor.y is not None else None
    try:
        editor.data, editor.y = load_svmlight_file(path)
        editor.index = np.asarray(range(len(editor.y)))
        editor.data = editor.data.tocsc()
        if editor.debug:
            print(f"Loaded '{dataset}', shape {editor.data.shape}")
        return
    except Exception as e:
        print(e)
        editor.data = old_data
        editor.y = old_y
        exit(0)

@cli.command('scale')
@click.option('--center', is_flag=True)
@click.option('--norm', is_flag=True)
@pass_editor
def scale(editor, center, norm):
    # from sklearn.preprocessing import scale
    # editor.data = csc_matrix(scale(editor.data.todense(), with_mean=center, with_std=scale_var))
    editor.data = editor.data.todense()
    if center:
        editor.data -= np.average(editor.data, axis=0)
    if norm:
        editor.data = normalize(editor.data, axis=0, copy=False)
    editor.data = csc_matrix(editor.data)

@cli.command('replace-column')
@click.argument('col', type=click.INT)
@click.argument('scheme', type=click.Choice(['uniform', 'scale', 'weights']))
@click.option('--scale-col', default=0, help="scale specified column (default 0)")
@click.option('--scale-by', default=1.0, help="scale factor for vector specified by `--scale-col` (default 1)")
@click.option('--weights', type=click.STRING, default=None,
    help="string containing python array with length n_col."
         "values in array correspond weights of each remaining column for replacement linear combination.")
@pass_editor
def replace_column(editor, col, scheme, scale_col, scale_by, weights):
    assert editor.data is not None, "load data before attempting to edit"
    assert not(weights is None and scheme == 'weights'), "specify weighting scheme"
    
    n_row, n_col = editor.data.shape
    if scheme == 'weights':
        weights = np.fromstring(weights, sep=' ')
        debug_str=f'A*{np.array2string(weights)}^T'
    elif scheme == 'scale':
        weights = np.zeros((n_col-1,))
        weights[scale_col] = scale_by
        debug_str=f'{scale_by}*A[:,{scale_col}]'
    elif scheme == 'uniform':
        weights = np.array([1/(n_col-1)]*(n_col-1))
        debug_str=f'average of other columns'
    else:
        return NotImplementedError

    weights = np.insert(weights, col, 0)

    new_col = editor.data * weights
    from scipy.sparse import csc_matrix
    editor.data[:,col] = csc_matrix(new_col.reshape((n_row,1)))
    if editor.debug:
        print(f"Replaced A[:,{col}] with "+debug_str)
    return

@cli.command('insert-columns')
@click.argument('n', type=click.INT)
@click.option('--weights', type=click.STRING, default=None)
@pass_editor
def insert_columns(editor, n, weights):
    assert editor.data is not None, "load data before attempting to edit"
    # assert weights is not None or uniform, "either specify weights or use the `--uniform` flag"
    if weights:
        from json import loads
        weights = loads(weights)
        for spec in weights:
            _insert_column(editor, spec.get('scheme'), spec.get('scale_col', 0), spec.get('scale_by', 1), spec.get('weights'))
        return
    
    for i in range(n):
        _insert_column(editor, 'uniform', 0, 1, None)    

    return

@cli.command('insert-column')
@click.argument('scheme', type=click.Choice(['uniform', 'scale', 'weights', 'ones']))
@click.option('--scale-col', default=0, help="scale specified column (default 0)")
@click.option('--scale-by', default=1.0, help="scale factor for vector specified by `--scale-col` (default 1)")
@click.option('--weights', type=click.STRING, default=None,
    help="string containing python array with length n_col."
         "values in array correspond weights of each remaining column for replacement linear combination.")
@pass_editor
def insert_column(editor, scheme, scale_col, scale_by, weights):
    _insert_column(editor, scheme, scale_col, scale_by, weights)

def _insert_column(editor, scheme, scale, scale_by, weights):
    assert editor.data is not None, "load data before attempting to edit"
    assert not (weights is None and scheme == 'weights'), "specify weighting scheme"
    
    n_row, n_col = editor.data.shape
    debug_str=''
    if scheme == 'weights':
        weights = np.fromstring(weights, sep=' ')
        debug_str=f'A*{np.array2string(weights)}^T'
    elif scheme == 'scale':
        weights = np.zeros((n_col,))
        weights[scale] = scale_by
        debug_str=f'{scale_by}*A[:,{scale}]'
    elif scheme == 'uniform':
        weights = np.array([1/n_col]*n_col)
        debug_str=f'average of existing columns'
    elif scheme == 'ones':
        editor.data = np.column_stack((editor.data.todense(), np.ones(n_row)))
        editor.data = csc_matrix(editor.data)
        return
    else:
        raise NotImplementedError()

    if weights.shape[0] != n_col:
        weights = np.concatenate((weights, [0]*(n_col - weights.shape[0])))
    weights = weights.reshape((n_col,1))
    
    new_col = editor.data * weights
    editor.data = csc_matrix(np.column_stack((editor.data.todense(), new_col)))
    if editor.debug:
        print("Inserted new column as "+debug_str)
    return

@cli.command('remove-column')
@click.argument('col', type=click.INT)
@pass_editor
def remove_column(editor, col):
    editor.data = csc_matrix(np.delete(editor.data.todense(), col, 1))

@cli.command('dump-svm')
@click.argument('filename', type=click.STRING)
@click.option('--overwrite', is_flag=True)
@pass_editor
def dump_svm(editor, filename, overwrite):
    assert editor.data is not None, "no data is loaded"
    if editor.Vh is not None:
        filenamew = filename.replace('.svm', '') + '-w.svm'
        pathw = os.path.join(editor.indir, filenamew)
    if not '.svm' in filename:
        filename += '.svm'
    path = os.path.join(editor.indir, filename)
    if os.path.exists(path) and not overwrite:
        print(f"Error: '{path}' already exists, use `--overwrite` to save anyway")
        return
    elif os.path.exists(path):
        if editor.debug:
            print(f"Warning: '{path}' already exists, overwriting")
        os.remove(path)
        if editor.Vh is not None and os.path.exists(pathw):
            os.remove(pathw)
    dump_svmlight_file(editor.data, editor.y, path)
    if editor.Vh is not None:
        dump_svmlight_file(editor.Vh, editor.s, pathw)
    if editor.debug:
        print(f"Data with shape {editor.data.shape} saved to '{path}'")
    return

@cli.command('info-rank')
@pass_editor
def info_rank(editor):
    rank = np.linalg.matrix_rank(editor.data.todense())
    print(f"rank(A) = {rank}")

@cli.command('info-cond')
@click.option('--p', type=click.STRING, help="Norm for use for condition number, either int or 'fro'")
@pass_editor
def info_cond(editor, p):
    try:
        new_p = int(p)
        p = new_p
    except:
        pass
    A_pinv = np.linalg.pinv(editor.data.todense())
    norm = np.linalg.norm(editor.data.todense(), p)
    norm_pinv = np.linalg.norm(A_pinv, p)
    print(f"cond(A, p={p}) = {norm*norm_pinv}")

@cli.command('split')
@click.argument('dataset_name', type=click.STRING)
@click.option('--K', type=click.INT, default=0, help="if provided, data is only split for K nodes (default: all possible splits)")
@click.option('--seed', type=click.INT, default=None, help="random shuffling seed (default: no shuffle)")
@click.option('--train', type=click.FLOAT, default=1)
@pass_editor
def split(editor, dataset_name, k, seed, train):
    if train > 1:
        train = int(np.floor(train))
        if train > editor.n_points:
            train = editor.n_points
    elif train >= 0:
        train = int(np.ceil(train * editor.n_points))
    else: 
        raise ValueError('`train` must be a float in [0,1] or a positive integer')
    if k:
        _split_dataset(editor, dataset_name, k, seed, train)
        return
    for k in range(1, editor.data.shape[1]+1):
        _split_dataset(editor, dataset_name, k, seed, train)

def _split_dataset(editor, dataset, K, seed, train):
    import joblib
    output_folder = os.path.join(editor.outdir, dataset, 'features', str(K))
    os.makedirs(os.path.join(output_folder, 'X'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'y'), exist_ok=True)
    X_test = y_test = None

    n_train = np.floor(train)
    issplit = train < editor.n_points
    print(f'Train/Test/Total: {train}/{editor.n_points-train}/{editor.n_points}')
    if issplit:
        os.makedirs(os.path.join(output_folder, 'X_test'), exist_ok=True)
        os.makedirs(os.path.join(output_folder, 'y_test'), exist_ok=True)
        X, X_test, y, y_test, index, index_test = train_test_split(editor.data, editor.y, editor.index,
            shuffle=seed is not None, random_state=seed, train_size=train)
    else:
        X, y = editor.data, editor.y
        index = editor.index
    indices = np.arange(editor.n_features)
    # if seed is not None:
    #     np.random.seed(seed)
    #     np.random.shuffle(indices)
    #     if editor.debug:
    #         print(f"randomized indices: {indices}")
    block_size = int(editor.n_features // K)
    extra = int(editor.n_features % K)

    beg = 0
    for k in range(K):
        e = int(extra>0)
        file_x_train = os.path.join(output_folder, 'X', str(k))
        file_y_train = os.path.join(output_folder, 'y', str(k))
        if editor.debug:
            print(f"Node {k}: {indices[beg:beg+block_size+e]}")
        joblib.dump(X[:,indices[beg:beg+block_size+e]], file_x_train)
        joblib.dump(y, file_y_train)
        if issplit:
            file_x_test = os.path.join(output_folder, 'X_test', str(k))
            file_y_test = os.path.join(output_folder, 'y_test', str(k))
            joblib.dump(X_test[:,indices[beg:beg+block_size+e]], file_x_test)
            joblib.dump(y_test, file_y_test)
        beg += block_size+e
        extra -= e
    index.dump(os.path.join(output_folder, 'index.npy'))
    index_test.dump(os.path.join(output_folder, 'index_test.npy'))
    indices.dump(os.path.join(output_folder, 'col_index.npy'))
    if editor.debug:
        print(f"Train splits for world_size {K} stored in '{output_folder}'")
        if issplit:
            print(f"Test splits for world_size {K} stored in '{output_folder}'")        

if __name__ == "__main__":
    cli()