"""Increase target name_tr and name_en length to 255

Revision ID: 89d5e6ac15d9
Revises: 4f72b01a33d9
Create Date: 2025-05-24 23:42:18.927449
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '89d5e6ac15d9'
down_revision = '4f72b01a33d9'
branch_labels = None
depends_on = None

def upgrade():
    # Alter target table columns to increase length to 255
    with op.batch_alter_table('target', schema='public') as batch_op:
        batch_op.alter_column('name_tr',
                             existing_type=sa.VARCHAR(length=100),
                             type_=sa.String(length=255),
                             existing_nullable=False)
        batch_op.alter_column('name_en',
                             existing_type=sa.VARCHAR(length=100),
                             type_=sa.String(length=255),
                             existing_nullable=False)

def downgrade():
    # Revert target table columns to length 100
    with op.batch_alter_table('target', schema='public') as batch_op:
        batch_op.alter_column('name_en',
                             existing_type=sa.String(length=255),
                             type_=sa.VARCHAR(length=100),
                             existing_nullable=False)
        batch_op.alter_column('name_tr',
                             existing_type=sa.String(length=255),
                             type_=sa.VARCHAR(length=100),
                             existing_nullable=False)