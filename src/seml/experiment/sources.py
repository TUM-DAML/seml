import importlib
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Set, Union, cast

from seml.database import delete_files, upload_file
from seml.settings import SETTINGS
from seml.utils import is_local_file, src_layout_to_flat_layout, working_directory
from seml.utils.errors import ExecutableError, MongoDBError

if TYPE_CHECKING:
    from pymongo.collection import Collection

States = SETTINGS.STATES


def import_exe(executable, conda_env, working_dir):
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
        logging.warning(
            f"Current Anaconda environment does not match the experiment's environment ('{conda_env}')."
        )

    with working_directory(working_dir):
        # Get experiment as module (which causes Sacred not to start ex.automain)
        exe_path = str(Path(executable).expanduser().resolve())
        sys.path.insert(0, os.path.dirname(exe_path))
        orig_handlers = logging.root.handlers
        orig_loglevel = logging.root.level
        exe_module = importlib.import_module(
            os.path.splitext(os.path.basename(executable))[0]
        )

    if exe_module.__file__ != exe_path:
        logging.error(
            f'Imported module path\n"{exe_module.__file__}" does not match executable path\n'
            f'"{exe_path}".\nIs the executable file name "{os.path.basename(executable)}" '
            f'a package name required by seml, e.g., "numpy.py"? '
            f'If yes, this case it not supported; please rename your script.\n'
            f'Otherwise, you can also skip source file uploading and configuration sanity checking '
            f'py passing "--no-code-checkpoint" and "--no-sanity-check" to seml.'
        )
        exit(1)
    logging.root.handlers = orig_handlers
    logging.root.setLevel(orig_loglevel)
    del sys.path[0]

    return exe_module


def get_imported_sources(
    executable, root_dir, conda_env, working_dir, stash_all_py_files: bool
) -> Set[str]:
    """Get the sources imported by the given executable.

    Args:
        executable (_type_): The Python file containing the experiment.
        root_dir (_type_): The root directory of the experiment.
        conda_env (_type_): The experiment's Anaconda environment.
        working_dir (_type_): The working directory of the experiment.
        stash_all_py_files (_type_): Whether to stash all .py files in the working directory.

    Returns:
        List[str]: The sources imported by the given executable.
    """
    import_exe(executable, conda_env, working_dir)
    root_path = str(Path(root_dir).expanduser().resolve())

    sources = set()
    source_added = True
    while source_added:
        source_added = False
        for name, mod in list(sys.modules.items()):
            if mod is None:
                continue
            if not getattr(mod, '__file__', False):
                continue
            filename = os.path.abspath(mod.__file__)
            if filename not in sources and is_local_file(filename, root_path):
                sources.add(filename)
                source_added = True

    if stash_all_py_files:
        for file in Path(working_dir).glob('**/*.py'):
            sources.add(str(file))

    return sources


def upload_sources(seml_config, collection, batch_id):
    with working_directory(seml_config['working_dir']):
        root_dir = str(Path(seml_config['working_dir']).expanduser().resolve())

        sources = get_imported_sources(
            seml_config['executable'],
            root_dir=root_dir,
            conda_env=seml_config['conda_environment'],
            working_dir=seml_config['working_dir'],
            stash_all_py_files=seml_config.get('stash_all_py_files', False),
        )
        executable_abs = str(Path(seml_config['executable']).expanduser().resolve())

        if executable_abs not in sources:
            raise ExecutableError(
                f'Executable {executable_abs} was not found in the source code files to upload.'
            )

        uploaded_files = []
        for s in sources:
            file_id = upload_file(s, collection, batch_id, 'source_file')
            source_path = Path(s)
            uploaded_files.append((str(source_path.relative_to(root_dir)), file_id))
    return uploaded_files


def get_git_info(filename, working_dir):
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
        from git import InvalidGitRepositoryError, Repo
    except ImportError:
        logging.warning(
            'Cannot import git (pip install GitPython). ' 'Not saving git status.'
        )

    with working_directory(working_dir):
        directory = os.path.dirname(filename)

        try:
            repo = Repo(directory, search_parent_directories=True)
        except InvalidGitRepositoryError:
            return None, None, None
        try:
            path = repo.remote().url
        except ValueError:
            path = 'git:/' + repo.working_dir
        commit = repo.head.commit.hexsha
    return path, commit, repo.is_dirty()


def load_sources_from_db(
    experiment: Dict,
    collection: 'Collection',
    to_directory: Union[str, Path],
    remove_src_directory: bool = SETTINGS.CODE_CHECKPOINT_REMOVE_SRC_DIRECTORY,
):
    import gridfs

    db = collection.database
    fs = gridfs.GridFS(db)
    if 'source_files' not in experiment['seml']:
        raise MongoDBError(
            f'No source files found for experiment with ID {experiment["_id"]}.'
        )
    source_files = experiment['seml']['source_files']
    target_directory = Path(to_directory)
    for path, _id in source_files:
        path = cast(str, path)
        # For the imports to prefer our loaded seml version, we need to convert the src-layout to the flat-layout.
        # https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/
        if remove_src_directory:
            path = src_layout_to_flat_layout(path)
        out_path = target_directory / path
        # only current user can read, write, or execute
        out_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        db_file = fs.find_one(_id)
        if db_file is None:
            raise MongoDBError(
                f"Could not find source file with ID '{_id}' for experiment with ID {experiment['_id']}."
            )
        with open(out_path, 'wb') as f:
            f.write(db_file.read())


def delete_batch_sources(collection: 'Collection', batch_id: int):
    db = collection.database
    filter_dict = {
        'metadata.batch_id': batch_id,
        'metadata.collection_name': f'{collection.name}',
    }
    source_files = db['fs.files'].find(filter_dict, {'_id'})
    source_files = [x['_id'] for x in source_files]
    if len(source_files) > 0:
        logging.info(
            f'Deleting {len(source_files)} source files corresponding '
            f'to batch {batch_id} in collection {collection.name}.'
        )
        delete_files(db, source_files)


def delete_orphaned_sources(collection: 'Collection', batch_ids=None):
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
