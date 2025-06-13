"""References column added to DrugDetail table

Revision ID: 8f45327198d2
Revises: 5c1c8ee12bf9
Create Date: 2025-06-13 12:40:02.636374

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '8f45327198d2'
down_revision = '5c1c8ee12bf9'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('drug_detail', sa.Column('references', sa.Text(), nullable=True), schema='public')

def downgrade():
    op.drop_column('drug_detail', 'references', schema='public')
    # ### end Alembic commands ###
