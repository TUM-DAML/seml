import logging
from typing import TYPE_CHECKING, Protocol
from seml.database import get_collection
from seml.settings import SETTINGS
from seml.utils import s_if, version_filter
from seml.utils.slurm import get_cluster_name


if TYPE_CHECKING:
    from pymongo.collection import Collection

States = SETTINGS.STATES


def migrate_collection(db_collection_name: str, skip: bool, backup: bool):
    """
    Migrate the given collection to be compatible with the current SEML version.

    Parameters
    ----------
    db_collection_name : str
        The name of the collection to migrate.
    skip : bool
        Whether to skip the migration.
    backup : bool
        Whether to create a backup of the collection before migrating.
    """
    from seml.console import prompt

    collection = get_collection(db_collection_name)

    if skip:
        return

    migrations = [migration_cls(collection) for migration_cls in _MIGRATIONS]
    backed_up = False
    for migration in migrations:
        if not migration.requires_migration():
            continue

        if not migration.is_silent():
            logging.warning(
                f"The collection '{db_collection_name}' needs a {migration.name()} migration to work with newer SEML versions.\n"
                'If you wish to make backups, do not proceed and call the command with the `--migration-backup` flag.\n'
                'To skip migration, call the command with the `--migration-skip` flag.'
            )
            if not prompt('Do you want to proceed? (y/n)', type=bool):
                logging.error('Aborted migration.')
                exit(1)

        if not backed_up and backup and not migration.is_silent():
            backup_name = f'{db_collection_name}_backup'
            if backup_name in collection.database.list_collection_names():
                logging.error(f'Backup collection {backup_name} already exists.')
                exit(1)
            collection.aggregate([{'$match': {}}, {'$out': backup_name}])
            logging.info(
                f"Backed up collection '{db_collection_name}' to '{backup_name}'."
            )
            backed_up = True

        n_updated = migration.migrate()
        if not migration.is_silent():
            logging.info(
                f'Successfully migrated {n_updated} experiment{s_if(n_updated)}.'
            )


class Migration(Protocol):
    def __init__(self, collection: 'Collection'): ...
    def requires_migration(self) -> bool: ...
    def migrate(self) -> int: ...
    def name(self) -> str: ...
    def is_silent(self) -> bool: ...


class Migration04To05Slurm(Migration):
    collection: 'Collection'
    db_filter = {'slurm': {'$not': {'$type': 'array'}}}

    def __init__(self, collection: 'Collection'):
        self.collection = collection

    def is_silent(self):
        return False

    def requires_migration(self):
        return self.collection.count_documents(self.db_filter, limit=1) > 0

    def name(self):
        return 'SLURM config'

    def migrate(self):
        # Check if there are still experiments running
        # If so, we cannot migrate the SLURM configuration
        db_filter_running = self.db_filter | {
            'status': {'$in': [*States.PENDING, *States.RUNNING]}
        }
        if self.collection.count_documents(db_filter_running) > 0:
            logging.error(
                'Cannot migrate SLURM configuration while there are still experiments running.\n'
                'Please wait until all experiments have finished.'
            )
            exit(1)

        # Move slurm array to execution field
        db_filter_executed = self.db_filter | {'slurm.array_id': {'$exists': True}}
        self.collection.update_many(
            db_filter_executed,
            {
                '$set': {
                    'execution': {
                        'cluster': get_cluster_name(),
                        'slurm.output_file': '$slurm.output_file',
                        'array_id': '$slurm.array_id',
                        'task_id': '$slurm.task_id',
                    }
                }
            },
        )
        # Convert slurm field to array
        n_updated = self.collection.update_many(
            self.db_filter,
            [{'$set': {'slurm': ['$slurm']}}],
        ).modified_count
        return n_updated


class Migration05Version(Migration):
    collection: 'Collection'
    db_filter = {
        '$or': [
            {'seml.version': {'$not': {'$type': 'array'}}},
            version_filter((0, 5, 0)),
        ]
    }

    def __init__(self, collection: 'Collection'):
        self.collection = collection

    def is_silent(self):
        return True

    def requires_migration(self):
        return self.collection.count_documents(self.db_filter, limit=1) > 0

    def name(self):
        return 'Version'

    def migrate(self):
        n_updated = self.collection.update_many(
            self.db_filter,
            {'$set': {'seml.version': [0, 5, 0]}},
        ).modified_count
        return n_updated


_MIGRATIONS = [
    Migration04To05Slurm,
    Migration05Version,
]
