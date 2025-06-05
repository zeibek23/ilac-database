"""some changes made

Revision ID: 5c1c8ee12bf9
Revises: 0dd79dff2569
Create Date: 2025-06-05 16:26:42.027101

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '5c1c8ee12bf9'
down_revision = '0dd79dff2569'
branch_labels = None
depends_on = None


def upgrade():
    # Create severity table (if not already created)
    op.create_table(
        'severity',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=20), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name'),
        schema='public'
    )
    # Add severity_id to drug_interaction
    op.add_column('drug_interaction', sa.Column('severity_id', sa.Integer(), nullable=True), schema='public')
    op.create_foreign_key(
        None, 'drug_interaction', 'severity', ['severity_id'], ['id'], source_schema='public', referent_schema='public'
    )
    # Populate severity table
    op.execute("INSERT INTO public.severity (name, description) VALUES ('Hafif', 'Minor side effects, minimal risk')")
    op.execute("INSERT INTO public.severity (name, description) VALUES ('Orta', 'Moderate side effects, requires monitoring')")
    op.execute("INSERT INTO public.severity (name, description) VALUES ('Şiddetli', 'Severe side effects, high risk')")
    op.execute("INSERT INTO public.severity (name, description) VALUES ('Kritik', 'Life-threatening, avoid combination')")
    # Map existing severity values to severity_id
    op.execute("""
        UPDATE public.drug_interaction
        SET severity_id = (
            CASE severity
                WHEN 'düşük' THEN (SELECT id FROM public.severity WHERE name = 'Hafif')
                WHEN 'orta' THEN (SELECT id FROM public.severity WHERE name = 'Orta')
                WHEN 'yüksek' THEN (SELECT id FROM public.severity WHERE name = 'Şiddetli')
                WHEN 'hayati tehlike' THEN (SELECT id FROM public.severity WHERE name = 'Kritik')
                ELSE (SELECT id FROM public.severity WHERE name = 'Hafif') -- Default to Hafif
            END
        )
    """)
    # Make severity_id non-nullable
    op.alter_column('drug_interaction', 'severity_id', nullable=False, schema='public')
    # Drop old severity column
    op.drop_column('drug_interaction', 'severity', schema='public')

def downgrade():
    op.add_column('drug_interaction', sa.Column('severity', sa.String(length=20), nullable=False, server_default='Hafif'), schema='public')
    op.execute("""
        UPDATE public.drug_interaction
        SET severity = (
            SELECT name FROM public.severity WHERE id = drug_interaction.severity_id
        )
    """)
    op.drop_column('drug_interaction', 'severity_id', schema='public')
    op.drop_table('severity', schema='public')