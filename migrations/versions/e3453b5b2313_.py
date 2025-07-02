"""empty message

Revision ID: e3453b5b2313
Revises: b71bf3254bff
Create Date: 2025-06-27 22:00:07.063184

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'e3453b5b2313'
down_revision = 'b71bf3254bff'
branch_labels = None
depends_on = None


def upgrade():
    # Create association table
    op.create_table(
        'drug_category_association',
        sa.Column('drug_id', sa.Integer, sa.ForeignKey('public.drug.id'), primary_key=True),
        sa.Column('category_id', sa.Integer, sa.ForeignKey('public.drug_category.id'), primary_key=True),
        schema='public'
    )
    # Migrate existing category_id data
    op.execute("""
        INSERT INTO public.drug_category_association (drug_id, category_id)
        SELECT id, category_id FROM public.drug WHERE category_id IS NOT NULL
    """)
    # Drop category_id column
    op.drop_column('drug', 'category_id', schema='public')

def downgrade():
    # Recreate category_id column
    op.add_column('drug', sa.Column('category_id', sa.Integer, sa.ForeignKey('public.drug_category.id'), nullable=True), schema='public')
    # Restore data (pick first category for simplicity)
    op.execute("""
        UPDATE public.drug
        SET category_id = (
            SELECT category_id
            FROM public.drug_category_association
            WHERE drug_id = drug.id
            LIMIT 1
        )
    """)
    # Drop association table
    op.drop_table('drug_category_association', schema='public')