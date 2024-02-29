import logging
import os
from pathlib import Path
from typing import Optional, Union


def init_project(
    directory: Union[str, Path] = '.',
    project_name: Optional[str] = None,
    user_name: Optional[str] = None,
    user_mail: Optional[str] = None,
    template: str = 'default',
    yes: bool = False,
):
    """
    Initialize a new project in the given directory.

    Parameters
    ----------
    directory : Union[str, Path]
        The directory to initialize the project in.
    project_name : Optional[str]
        The name of the project. If not given, the name of the directory is used.
    user_name : Optional[str]
        The name of the user. If not given, the environment variable USER is used.
    user_mail : Optional[str]
        The email of the user. If not given, ''.
    template : str
        The template to use for the project.
    yes : bool
        If True, no confirmation is asked before initializing the project.
    """
    import importlib.resources
    from click import prompt
    from gitignore_parser import parse_gitignore

    if directory is None:
        directory = Path()
    directory = Path(directory).absolute()
    # Ensure that the directory exists
    if not directory.exists():
        directory.mkdir(parents=True)

    # Ensure that its state is okay
    if any(directory.glob('**/*')) and not yes:
        if not prompt(
            f'Directory "{directory}" is not empty. Are you sure you want to initialize a new project here? (y/n)',
            type=bool,
        ):
            exit(1)
    logging.info(f'Initializing project in "{directory}" using template "{template}"')

    template_path = (
        Path(str(importlib.resources.files('seml')))
        / 'templates'
        / 'project'
        / template
    )
    if not template_path.exists():
        logging.error(f'Template "{template}" does not exist')
        exit(1)

    if project_name is None:
        project_name = directory.name
    if user_name is None:
        user_name = os.getenv('USER', os.getenv('USERNAME', 'user'))
    if user_mail is None:
        user_mail = 'my@mail.com'
    format_map = dict(
        project_name=project_name, user_name=user_name, user_mail=user_mail
    )

    gitignore_path = template_path / '.gitignore'
    if gitignore_path.exists():
        ignore_file = parse_gitignore(gitignore_path)
    else:

        def ignore_file(file_path: str):
            return False

    # Copy files one-by-one
    for src in template_path.glob('**/*'):
        # skip files ignored by .gitignore
        if ignore_file(str(src)):
            continue
        # construct destination
        file_name = src.relative_to(template_path)
        target_file_name = Path(str(file_name).format_map(format_map))
        dst = directory / target_file_name
        # Create directories
        if src.is_dir():
            if not dst.exists():
                dst.mkdir()
        elif not dst.exists():
            # For templates fill in variables
            if src.suffix.endswith('.template'):
                dst = dst.with_suffix(src.suffix.removesuffix('.template'))
                dst.write_text(src.read_text().format_map(format_map))
            else:
                # Other files copy directly
                dst.write_bytes(src.read_bytes())
    logging.info('Project initialized successfully')
