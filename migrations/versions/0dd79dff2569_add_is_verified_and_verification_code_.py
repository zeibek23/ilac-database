"""Add is_verified and verification_code to User

Revision ID: 0dd79dff2569
Revises: fa8a734a6251
Create Date: 2025-06-01 10:34:11.093860

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '0dd79dff2569'
down_revision = 'fa8a734a6251'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('user', sa.Column('is_verified', sa.Boolean(), nullable=False, server_default='false'))
    op.add_column('user', sa.Column('verification_code', sa.String(length=6), nullable=True))

def downgrade():
    op.drop_column('user', 'verification_code')
    op.drop_column('user', 'is_verified')
