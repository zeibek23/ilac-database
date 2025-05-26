"""Add therapeutic_min and therapeutic_max to DrugRoute

Revision ID: 1a3a08ec5b7e
Revises: 89d5e6ac15d9
Create Date: <some-date>
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '1a3a08ec5b7e'
down_revision = '89d5e6ac15d9'
branch_labels = None
depends_on = None

def upgrade():
    # Only modify the drug_route table in the 'public' schema
    with op.batch_alter_table('drug_route', schema='public') as batch_op:
        batch_op.add_column(sa.Column('therapeutic_min', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('therapeutic_max', sa.Float(), nullable=True))

def downgrade():
    # Reverse the changes for drug_route
    with op.batch_alter_table('drug_route', schema='public') as batch_op:
        batch_op.drop_column('therapeutic_max')
        batch_op.drop_column('therapeutic_min')