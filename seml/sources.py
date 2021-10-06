import sys
import os
from pathlib import Path
import logging
import importlib
import gridfs

from seml.database import get_collection, upload_file
from seml.errors import ExecutableError, MongoDBError


def is_local_file(filename, root_dir):
    """
    See https://github.com/IDSIA/sacred/blob/master/sacred/dependencies.py
    Parameters
    ----------
    filename
    root_dir

    Returns
    -------

    """
    file_path = Path(filename).expanduser().resolve()
    root_path = Path(root_dir).expanduser().resolve()
    return root_path in file_path.parents


def import_exe(executable, conda_env):
    """Import the given executable file.

    Parameters
    ----------
    executable: str
        The Python file containing the experiment.
    conda_env: str
        The experiment's Anaconda environment.

    Returns
    -------
    The module of the imported executable.

    """
    # Check if current environment matches experiment environment
    if conda_env is not None and conda_env != os.environ.get('CONDA_DEFAULT_ENV'):
        logging.warning(f"Current Anaconda environment does not match the experiment's environment ('{conda_env}').")

    # Get experiment as module (which causes Sacred not to start ex.automain)
    exe_path = str(Path(executable).expanduser().resolve())
    sys.path.insert(0, os.path.dirname(exe_path))
    orig_handlers = logging.root.handlers
    orig_loglevel = logging.root.level
    exe_module = importlib.import_module(os.path.splitext(os.path.basename(executable))[0])
    logging.root.handlers = orig_handlers
    logging.root.setLevel(orig_loglevel)
    del sys.path[0]

    return exe_module


def get_imported_sources(executable, root_dir, conda_env):
    import_exe(executable, conda_env)
    root_path = str(Path(root_dir).expanduser().resolve())

    sources = set()
    for name, mod in sys.modules.items():
        if mod is None:
            continue
        if not getattr(mod, "__file__", False):
            continue
        filename = os.path.abspath(mod.__file__)
        if filename not in sources and is_local_file(filename, root_path):
            sources.add(filename)

    return sources


def upload_sources(seml_config, collection, batch_id):
    root_dir = str(Path(seml_config['working_dir']).expanduser().resolve())

    sources = get_imported_sources(seml_config['executable'], root_dir=root_dir,
                                   conda_env=seml_config['conda_environment'])
    executable_abs = str(Path(seml_config['executable']).expanduser().resolve())

    if executable_abs not in sources:
        raise ExecutableError(f"Executable {executable_abs} was not found in the source code files to upload.")

    uploaded_files = []
    for s in sources:
        file_id = upload_file(s, collection, batch_id, 'source_file')
        source_path = Path(s)
        uploaded_files.append((str(source_path.relative_to(root_dir)), file_id))
    return uploaded_files


def get_git_info(filename):
    """
    Get the git commit info.
    See https://github.com/IDSIA/sacred/blob/c1c19a58332368da5f184e113252b6b0abc8e33b/sacred/dependencies.py#L400

    Parameters
    ----------
    filename: str

    Returns
    -------
    path: str
        The base path of the repository
    commit: str
        The commit hash
    is_dirty: bool
        True if there are uncommitted changes in the repository
    """

    try:
        from git import Repo, InvalidGitRepositoryError
    except ImportError:
        logging.warning("Cannot import git (pip install GitPython). "
                        "Not saving git status.")

    directory = os.path.dirname(filename)
    try:
        repo = Repo(directory, search_parent_directories=True)
    except InvalidGitRepositoryError:
        return None, None, None
    try:
        path = repo.remote().url
    except ValueError:
        path = "git:/" + repo.working_dir
    commit = repo.head.commit.hexsha
    return path, commit, repo.is_dirty()


def load_sources_from_db(exp, collection, to_directory):
    db = collection.database
    fs = gridfs.GridFS(db)
    if 'source_files' not in exp['seml']:
        raise MongoDBError(f'No source files found for experiment with ID {exp["_id"]}.')
    source_files = exp['seml']['source_files']
    for path, _id in source_files:
        _dir = f"{to_directory}/{os.path.dirname(path)}"
        if not os.path.exists(_dir):
            os.makedirs(_dir, mode=0o700)  # only current user can read, write, or execute
        with open(f'{to_directory}/{path}', 'wb') as f:
            file = fs.find_one(_id)
            if file is None:
                raise MongoDBError(f"Could not find source file with ID '{_id}' for experiment with ID {exp['_id']}.")
            f.write(file.read())


def delete_orphaned_sources(collection, batch_ids=None):
    if batch_ids is not None:
        # check for empty batches within list of batch ids
        filter_dict = {'batch_id': {'$in': list(batch_ids)}}
    else:
        # check for any empty batches
        filter_dict = {}
    db_results = collection.find(filter_dict, {'batch_id'})
    remaining_batch_ids = set([x['batch_id'] for x in db_results])
    empty_batch_ids = set(batch_ids) - remaining_batch_ids
    for b_id in empty_batch_ids:
        delete_batch_sources(collection, b_id)


def delete_batch_sources(collection, batch_id):
    db = collection.database
    fs = gridfs.GridFS(db)
    filter_dict = {'metadata.batch_id': batch_id,
                   'metadata.collection_name': f'{collection.name}'}
    source_files = db['fs.files'].find(filter_dict, {'_id'})
    source_files = [x['_id'] for x in source_files]
    if len(source_files) > 0:
        print(f"Deleting {len(source_files)} source files corresponding "
              f"to batch {batch_id} in collection {collection.name}.")
        for to_delete in source_files:
            fs.delete(to_delete)


def reload_sources(db_collection_name, batch_ids=None, keep_old=False):
    collection = get_collection(db_collection_name)
    
    if batch_ids is not None:
        filter_dict = {'batch_id': {'$in': list(batch_ids)}}
    else:
        filter_dict = {}
    db_results = collection.find(filter_dict, {'batch_id', 'seml'})
    id_to_seml = {x['batch_id']: x['seml'] for x in db_results}
    for batch_id, seml_config in id_to_seml.items():
        if 'working_dir' not in seml_config or not seml_config['working_dir']:
            print(f'Batch {batch_id}: No source files to refresh.')
        # Find the currently used source files
        db = collection.database
        fs = gridfs.GridFS(db)
        fs_filter_dict = {
            'metadata.batch_id': batch_id,
            'metadata.collection_name': f'{collection.name}',
            'metadata.deprecated': {'$exists': False}
        }
        current_source_files = db['fs.files'].find(filter_dict, '_id')
        current_ids = [x['_id'] for x in current_source_files]
        fs_filter_dict = {
            '_id': {'$in': current_ids}
        }
        # Deprecate them
        db['fs.files'].update_many(fs_filter_dict, {'$set': {'metadata.deprecated': True}})

        try:
            # Try to upload the new ones
            source_files = upload_sources(seml_config, collection, batch_id)
        except Exception:
            # If it fails we reconstruct the old ones
            print(f"Batch {batch_id}: Please navigate to the executable '{seml_config['executable']}'.")
            db['fs.files'].update_many(fs_filter_dict, {'$unset': {'metadata.deprecated': ""}})
        else:
            try:
                # Try to assign the new ones to the experiments
                filter_dict = {
                    'batch_id': batch_id
                }
                collection.update_many(filter_dict, {
                    '$set': {
                        'seml.source_files': source_files
                    }
                })
            except:
                print(f'Batch {batch_id}: Failed to set new source files.')
                for to_delete in source_files:
                    fs.delete(to_delete[1])
        
        # Delete the old source files
        if not keep_old:
            fs_filter_dict = {
                'metadata.batch_id': batch_id,
                'metadata.collection_name': f'{collection.name}',
                'metadata.deprecated': True
            }
            source_files = [x['_id'] for x in db['fs.files'].find(fs_filter_dict, {'_id'})]
            for to_delete in source_files:
                fs.delete(to_delete)
