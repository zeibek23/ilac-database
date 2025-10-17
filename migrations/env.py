import logging
from logging.config import fileConfig
from flask import current_app
from alembic import context
from sqlalchemy.pool import NullPool

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

# Define schemas to include and exclude
INCLUDE_SCHEMAS = ['public']  # Only manage the 'public' schema
EXCLUDE_SCHEMAS = ['realtime', 'auth', 'storage', 'extensions', 'pgbouncer', 'information_schema', 'pg_catalog']

def run_migrations_offline():
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=get_metadata(),
        literal_binds=True,
        include_schemas=True,
        version_table_schema='public',  # Ensure alembic_version table is in 'public'
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

    def table_sort_key(table):
        priority_tables = {
            'drug': 0,
            'drug_detail': 1,
            'side_effect': 2,
            'detail_side_effect': 3,
            'clinical_annotation': 4,
            'clinical_ann_allele': 5,
            'clinical_ann_evidence': 6,
            'publication': 7,
            'clinical_ann_evidence_publication': 8,
            'drug_salt': 9,
            'pathway': 10,
            'pathway_drug': 11
        }
        return (priority_tables.get(table.name, 12), table.name)

    def include_object(object, name, type_, reflected, compare_to):
        if type_ == "table":
            # Explicitly exclude alembic_version table
            if name == 'alembic_version':
                return False
            schema = object.schema or 'public'
            return schema in INCLUDE_SCHEMAS and schema not in EXCLUDE_SCHEMAS
        return True

    conf_args = current_app.extensions['migrate'].configure_args
    if conf_args.get("process_revision_directives") is None:
        conf_args["process_revision_directives"] = process_revision_directives
    conf_args["include_object"] = include_object

    connectable = get_engine().execution_options(poolclass=NullPool)  # Use NullPool for Supabase

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=get_metadata(),
            include_schemas=True,
            version_table_schema='public',
            table_sort_key=table_sort_key,
            **conf_args
        )

        logger.info(f"Processing schemas: {INCLUDE_SCHEMAS}")
        logger.info(f"Excluding schemas: {EXCLUDE_SCHEMAS}")

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()