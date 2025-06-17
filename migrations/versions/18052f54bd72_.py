"""Add severity_id and reference to drug_lab_test_interaction

Revision ID: 18052f54bd72
Revises: 8f45327198d2
Create Date: 2025-06-16 09:42:51.379325

"""
from alembic import op
import sqlalchemy as sa

# Revision identifiers, used by Alembic
revision = '18052f54bd72'
down_revision = '8f45327198d2'
branch_labels = None
depends_on = None

def upgrade():
    # Add severity_id column (nullable initially to allow data migration)
    op.add_column('drug_lab_test_interaction', sa.Column('severity_id', sa.Integer(), nullable=True))
    
    # Add reference column
    op.add_column('drug_lab_test_interaction', sa.Column('reference', sa.Text(), nullable=True))
    
    # Map existing severity strings to severity_id based on Severity table
    op.execute("""
        UPDATE public.drug_lab_test_interaction
        SET severity_id = CASE
            WHEN severity = 'Mild' THEN 1  -- Hafif
            WHEN severity = 'Moderate' THEN 2  -- Orta
            WHEN severity = 'Severe' THEN 3  -- Şiddetli
            WHEN severity = 'Critical' THEN 4  -- Kritik
            ELSE NULL
        END
    """)
    
    # Check for unmapped severities
    conn = op.get_bind()
    result = conn.execute(
        sa.text("SELECT COUNT(*) FROM public.drug_lab_test_interaction WHERE severity_id IS NULL AND severity IS NOT NULL")
    ).scalar()
    if result > 0:
        raise Exception(f"Found {result} rows with unmapped severity values in drug_lab_test_interaction. Please review the severity column data.")
    
    # Make severity_id non-nullable and add foreign key
    op.alter_column('drug_lab_test_interaction', 'severity_id', nullable=False)
    op.create_foreign_key(
        'drug_lab_test_interaction_severity_id_fkey',
        'drug_lab_test_interaction',
        'severity',
        ['severity_id'],
        ['id'],
        referent_schema='public'
    )
    
    # Drop old severity column
    op.drop_column('drug_lab_test_interaction', 'severity')

def downgrade():
    # Add back severity column
    op.add_column('drug_lab_test_interaction', sa.Column('severity', sa.String(20), nullable=True))
    
    # Map severity_id back to severity strings
    op.execute("""
        UPDATE public.drug_lab_test_interaction
        SET severity = CASE
            WHEN severity_id = 1 THEN 'Mild'
            WHEN severity_id = 2 THEN 'Moderate'
            WHEN severity_id = 3 THEN 'Severe'
            WHEN severity_id = 4 THEN 'Critical'
            ELSE NULL
        END
    """)
    
    # Make severity non-nullable
    op.alter_column('drug_lab_test_interaction', 'severity', nullable=False)
    
    # Drop foreign key and severity_id column
    op.drop_constraint('drug_lab_test_interaction_severity_id_fkey', 'drug_lab_test_interaction', type_='foreignkey')
    op.drop_column('drug_lab_test_interaction', 'severity_id')
    
    # Drop reference column
    op.drop_column('drug_lab_test_interaction', 'reference')