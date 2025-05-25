import logging
from logging.config import fileConfig

from flask import current_app
from alembic import context

# Alembic Config object, provides access to .ini file values
config = context.config

# Set up loggers
fileConfig(config.config_file_name)
logger = logging.getLogger('alembic.env')

# Enable debug logging for Alembic
logging.getLogger('alembic').setLevel(logging.DEBUG)

def get_engine():
    try:
        # Works with Flask-SQLAlchemy<3 and Alchemical
        return current_app.extensions['migrate'].db.get_engine()
    except (TypeError, AttributeError):
        # Works with Flask-SQLAlchemy>=3
        return current_app.extensions['migrate'].db.engine

def get_engine_url():
    try:
        return get_engine().url.render_as_string(hide_password=False).replace('%', '%%')
    except AttributeError:
        return str(get_engine().url).replace('%', '%%')

# Set SQLAlchemy URL and database
config.set_main_option('sqlalchemy.url', get_engine_url())
target_db = current_app.extensions['migrate'].db

def get_metadata():
    if hasattr(target_db, 'metadatas'):
        return target_db.metadatas[None]
    return target_db.metadata

def run_migrations_offline():
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=get_metadata(),
        literal_binds=True
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    """Run migrations in 'online' mode."""
    def process_revision_directives(context, revision, directives):
        if getattr(config.cmd_opts, 'autogenerate', False):
            script = directives[0]
            if script.upgrade_ops.is_empty():
                directives[:] = []
                logger.info('No changes in schema detected.')

    # Prioritize key tables to resolve foreign key dependencies
    def table_sort_key(table):
        priority_tables = {
            'drug': 0,
            'drug_detail': 1,
            'side_effect': 2,
            'detail_side_effect': 3,
            'clinical_annotation': 4,
            'clinical_ann_allele': 5,
            'clinical_ann_evidence': 6,  # Added to ensure it comes after clinical_annotation
            'publication': 7,
            'clinical_ann_evidence_publication': 8,
            'drug_salt': 9,
            'pathway': 10,
            'pathway_drug': 11
        }
        return (priority_tables.get(table.name, 12), table.name)

    conf_args = current_app.extensions['migrate'].configure_args
    if conf_args.get("process_revision_directives") is None:
        conf_args["process_revision_directives"] = process_revision_directives

    connectable = get_engine()

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=get_metadata(),
            include_schemas=True,  # Support explicit schema definitions
            table_sort_key=table_sort_key,  # Custom table sorting
            **conf_args
        )

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()