from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'fa8a734a6251'
down_revision = '1a3a08ec5b7e'
branch_labels = None
depends_on = None

def upgrade():
    op.add_column('drug_route', sa.Column('therapeutic_unit', sa.String(), nullable=True))

def downgrade():
    op.drop_column('drug_route', 'therapeutic_unit')