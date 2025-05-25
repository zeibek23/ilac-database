"""Fix clinical_ann_evidence foreign key and initialize schema

Revision ID: 4f72b01a33d9
Revises: None
Create Date: 2025-05-24 20:30:00
"""

from alembic import op
import sqlalchemy as sa
from datetime import datetime

# Revision identifiers, used by Alembic.
revision = '4f72b01a33d9'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # Create drug_category table
    op.create_table(
        'drug_category',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('name', sa.String(50), nullable=False, unique=True),
        sa.Column('parent_id', sa.Integer, sa.ForeignKey('public.drug_category.id'), nullable=True),
        schema='public'
    )

    # Create salt table
    op.create_table(
        'salt',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('name_tr', sa.String(100), nullable=False),
        sa.Column('name_en', sa.String(100), nullable=False),
        schema='public'
    )

    # Create indication table
    op.create_table(
        'indication',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('foundation_uri', sa.String(255), unique=True, nullable=True),
        sa.Column('disease_id', sa.String(255), unique=True, nullable=True),
        sa.Column('name_en', sa.String(255), nullable=False),
        sa.Column('name_tr', sa.String(255), nullable=True),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('synonyms', sa.Text, nullable=True),
        sa.Column('species', sa.String(50), nullable=True, default="Human"),
        sa.Column('code', sa.String(50), nullable=True),
        sa.Column('class_kind', sa.String(50), nullable=True),
        sa.Column('depth', sa.Integer, nullable=True),
        sa.Column('parent_id', sa.Integer, sa.ForeignKey('public.indication.id'), nullable=True),
        sa.Column('is_residual', sa.Boolean, nullable=False, default=False),
        sa.Column('is_leaf', sa.Boolean, nullable=False, default=False),
        sa.Column('chapter_no', sa.String(10), nullable=True),
        sa.Column('created_at', sa.DateTime, nullable=False, default=datetime.utcnow),
        sa.Column('updated_at', sa.DateTime, nullable=False, default=datetime.utcnow),
        sa.Column('BlockId', sa.String(50), nullable=True),
        schema='public'
    )

    # Create target table
    op.create_table(
        'target',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('name_tr', sa.String(100), nullable=False),
        sa.Column('name_en', sa.String(100), nullable=False),
        schema='public'
    )

    # Create route_of_administration table
    op.create_table(
        'route_of_administration',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('type', sa.String(50), nullable=False),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('parent_id', sa.Integer, sa.ForeignKey('public.route_of_administration.id'), nullable=True),
        schema='public'
    )

    # Create receptor table
    op.create_table(
        'receptor',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('type', sa.String(100), nullable=False),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('molecular_weight', sa.String(50), nullable=True),
        sa.Column('length', sa.Integer, nullable=True),
        sa.Column('gene_name', sa.String(100), nullable=True),
        sa.Column('subcellular_location', sa.Text, nullable=True),
        sa.Column('function', sa.Text, nullable=True),
        sa.Column('iuphar_id', sa.String(50), nullable=True),
        sa.Column('pdb_ids', sa.Text, nullable=True),
        sa.Column('binding_site_x', sa.Float, nullable=True),
        sa.Column('binding_site_y', sa.Float, nullable=True),
        sa.Column('binding_site_z', sa.Float, nullable=True),
        schema='public'
    )

    # Create pathway table
    op.create_table(
        'pathway',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('pathway_id', sa.String(50), unique=True, nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('organism', sa.String(50), nullable=False),
        sa.Column('url', sa.String(255), nullable=True),
        schema='public'
    )

    # Create metabolism_organ table
    op.create_table(
        'metabolism_organ',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('name', sa.String(100), nullable=False, unique=True),
        schema='public'
    )

    # Create metabolism_enzyme table
    op.create_table(
        'metabolism_enzyme',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('name', sa.String(100), nullable=False, unique=True),
        schema='public'
    )

    # Create metabolite table
    op.create_table(
        'metabolite',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('name', sa.String(200), nullable=False),
        sa.Column('parent_id', sa.Integer, sa.ForeignKey('public.metabolite.id'), nullable=True),
        sa.Column('drug_id', sa.Integer, nullable=True),
        schema='public'
    )

    # Create gene table
    op.create_table(
        'gene',
        sa.Column('gene_id', sa.String(50), primary_key=True),
        sa.Column('gene_symbol', sa.String(100), unique=True, nullable=False),
        schema='public'
    )

    # Create phenotype table
    op.create_table(
        'phenotype',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('name', sa.String(200), unique=True, nullable=False),
        schema='public'
    )

    # Create variant table
    op.create_table(
        'variant',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('pharmgkb_id', sa.String(50), unique=True, nullable=True),
        sa.Column('name', sa.String(200), unique=True, nullable=False),
        schema='public'
    )

    # Create drug table
    op.create_table(
        'drug',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('name_tr', sa.String(255), nullable=False),
        sa.Column('name_en', sa.String(255), nullable=False),
        sa.Column('alternative_names', sa.Text, nullable=True),
        sa.Column('fda_approved', sa.Boolean, nullable=False, default=False),
        sa.Column('indications', sa.Text, nullable=True),
        sa.Column('chembl_id', sa.String(50), nullable=True),
        sa.Column('category_id', sa.Integer, sa.ForeignKey('public.drug_category.id'), nullable=True),
        sa.Column('pharmgkb_id', sa.String(50), unique=True, nullable=True),
        schema='public'
    )

    # Create drug_salt table
    op.create_table(
        'drug_salt',
        sa.Column('drug_id', sa.Integer, sa.ForeignKey('public.drug.id'), primary_key=True),
        sa.Column('salt_id', sa.Integer, sa.ForeignKey('public.salt.id'), primary_key=True),
        schema='public'
    )

    # Create drug_detail table
    op.create_table(
        'drug_detail',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('drug_id', sa.Integer, sa.ForeignKey('public.drug.id'), nullable=False),
        sa.Column('salt_id', sa.Integer, sa.ForeignKey('public.salt.id'), nullable=True),
        sa.Column('molecular_formula', sa.String(100), nullable=True),
        sa.Column('synthesis', sa.Text, nullable=True),
        sa.Column('structure', sa.String(200), nullable=True),
        sa.Column('structure_3d', sa.String(200), nullable=True),
        sa.Column('mechanism_of_action', sa.Text, nullable=True),
        sa.Column('iupac_name', sa.String(200), nullable=True),
        sa.Column('smiles', sa.String(200), nullable=True),
        sa.Column('inchikey', sa.String(200), nullable=True),
        sa.Column('pubchem_cid', sa.String(50), nullable=True),
        sa.Column('pubchem_sid', sa.String(50), nullable=True),
        sa.Column('cas_id', sa.String(50), nullable=True),
        sa.Column('ec_number', sa.String(50), nullable=True),
        sa.Column('nci_code', sa.String(50), nullable=True),
        sa.Column('rxcui', sa.String(50), nullable=True),
        sa.Column('snomed_id', sa.String(50), nullable=True),
        sa.Column('molecular_weight', sa.Float, nullable=True),
        sa.Column('pharmacodynamics', sa.Text, nullable=True),
        sa.Column('indications', sa.Text, nullable=True),
        sa.Column('target_molecules', sa.Text, nullable=True),
        sa.Column('pharmacokinetics', sa.Text, nullable=True),
        sa.Column('boiling_point', sa.String(100), nullable=True),
        sa.Column('melting_point', sa.String(100), nullable=True),
        sa.Column('density', sa.String(100), nullable=True),
        sa.Column('solubility', sa.String(200), nullable=True),
        sa.Column('flash_point', sa.String(100), nullable=True),
        sa.Column('fda_approved', sa.Boolean, nullable=False, default=False),
        sa.Column('ema_approved', sa.Boolean, nullable=False, default=False),
        sa.Column('titck_approved', sa.Boolean, nullable=False, default=False),
        sa.Column('black_box_warning', sa.Boolean, nullable=False, default=False),
        sa.Column('black_box_details', sa.Text, nullable=True),
        sa.Column('half_life', sa.Float, nullable=True),
        sa.Column('clearance_rate', sa.Float, nullable=True),
        sa.Column('bioavailability', sa.Float, nullable=True),
        schema='public'
    )

    # Create side_effect table
    op.create_table(
        'side_effect',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('name_en', sa.String(100), nullable=False),
        sa.Column('name_tr', sa.String(100), nullable=True),
        schema='public'
    )

    # Create detail_side_effect table
    op.create_table(
        'detail_side_effect',
        sa.Column('detail_id', sa.Integer, sa.ForeignKey('public.drug_detail.id'), primary_key=True),
        sa.Column('side_effect_id', sa.Integer, sa.ForeignKey('public.side_effect.id'), primary_key=True),
        schema='public'
    )

    # Create drug_interaction table
    op.create_table(
        'drug_interaction',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('drug1_id', sa.Integer, sa.ForeignKey('public.drug.id'), nullable=False),
        sa.Column('drug2_id', sa.Integer, sa.ForeignKey('public.drug.id'), nullable=False),
        sa.Column('route_id', sa.Integer, sa.ForeignKey('public.route_of_administration.id'), nullable=True),
        sa.Column('interaction_type', sa.String(50), nullable=False),
        sa.Column('interaction_description', sa.Text, nullable=False),
        sa.Column('severity', sa.String(20), nullable=False, default="Hafif"),
        sa.Column('mechanism', sa.Text, nullable=True),
        sa.Column('pharmacokinetics', sa.Text, nullable=True),
        sa.Column('monitoring', sa.Text, nullable=True),
        sa.Column('alternatives', sa.Text, nullable=True),
        sa.Column('reference', sa.Text, nullable=True),
        sa.Column('predicted_severity', sa.String(50), nullable=True),
        sa.Column('prediction_confidence', sa.Float, nullable=True),
        sa.Column('processed', sa.Boolean, default=False),
        sa.Column('time_to_peak', sa.Float, nullable=True),
        schema='public'
    )

    # Create drug_receptor_interaction table
    op.create_table(
        'drug_receptor_interaction',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('drug_id', sa.Integer, sa.ForeignKey('public.drug.id'), nullable=False),
        sa.Column('receptor_id', sa.Integer, sa.ForeignKey('public.receptor.id'), nullable=False),
        sa.Column('affinity', sa.Float, nullable=True),
        sa.Column('interaction_type', sa.String(50), nullable=True),
        sa.Column('mechanism', sa.Text, nullable=True),
        sa.Column('pdb_file', sa.String(200), nullable=True),
        sa.Column('units', sa.String, nullable=True),
        sa.Column('affinity_parameter', sa.String(50), nullable=True),
        schema='public'
    )

    # Create drug_disease_interaction table
    op.create_table(
        'drug_disease_interaction',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('drug_id', sa.Integer, sa.ForeignKey('public.drug.id'), nullable=False),
        sa.Column('indication_id', sa.Integer, sa.ForeignKey('public.indication.id'), nullable=False),
        sa.Column('interaction_type', sa.String(50), nullable=False),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('severity', sa.String(20), default="Moderate"),
        sa.Column('recommendation', sa.Text, nullable=True),
        schema='public'
    )

    # Create drug_route table
    op.create_table(
        'drug_route',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('drug_detail_id', sa.Integer, sa.ForeignKey('public.drug_detail.id', ondelete="CASCADE"), nullable=False),
        sa.Column('route_id', sa.Integer, sa.ForeignKey('public.route_of_administration.id', ondelete="CASCADE"), nullable=False),
        sa.Column('pharmacodynamics', sa.Text, nullable=True),
        sa.Column('pharmacokinetics', sa.Text, nullable=True),
        sa.Column('absorption_rate_min', sa.Float, nullable=True),
        sa.Column('absorption_rate_max', sa.Float, nullable=True),
        sa.Column('vod_rate_min', sa.Float, nullable=True),
        sa.Column('vod_rate_max', sa.Float, nullable=True),
        sa.Column('protein_binding_min', sa.Float, nullable=True),
        sa.Column('protein_binding_max', sa.Float, nullable=True),
        sa.Column('half_life_min', sa.Float, nullable=True),
        sa.Column('half_life_max', sa.Float, nullable=True),
        sa.Column('clearance_rate_min', sa.Float, nullable=True),
        sa.Column('clearance_rate_max', sa.Float, nullable=True),
        sa.Column('bioavailability_min', sa.Float, nullable=True),
        sa.Column('bioavailability_max', sa.Float, nullable=True),
        sa.Column('tmax_min', sa.Float, nullable=True),
        sa.Column('tmax_max', sa.Float, nullable=True),
        sa.Column('cmax_min', sa.Float, nullable=True),
        sa.Column('cmax_max', sa.Float, nullable=True),
        schema='public'
    )

    # Create route_indication table
    op.create_table(
        'route_indication',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('drug_detail_id', sa.Integer, sa.ForeignKey('public.drug_detail.id'), nullable=False),
        sa.Column('route_id', sa.Integer, sa.ForeignKey('public.route_of_administration.id'), nullable=False),
        sa.Column('indication_id', sa.Integer, sa.ForeignKey('public.indication.id'), nullable=False),
        sa.Column('drug_route_id', sa.Integer, sa.ForeignKey('public.drug_route.id', ondelete="CASCADE"), nullable=True),
        schema='public'
    )

    # Create pathway_drug table
    op.create_table(
        'pathway_drug',
        sa.Column('pathway_id', sa.Integer, sa.ForeignKey('public.pathway.id'), primary_key=True),
        sa.Column('drug_id', sa.Integer, sa.ForeignKey('public.drug.id'), primary_key=True),
        schema='public'
    )

    # Create drug_route_metabolism_organ table
    op.create_table(
        'drug_route_metabolism_organ',
        sa.Column('drug_route_id', sa.Integer, sa.ForeignKey('public.drug_route.id'), primary_key=True),
        sa.Column('metabolism_organ_id', sa.Integer, sa.ForeignKey('public.metabolism_organ.id'), primary_key=True),
        schema='public'
    )

    # Create drug_route_metabolism_enzyme table
    op.create_table(
        'drug_route_metabolism_enzyme',
        sa.Column('drug_route_id', sa.Integer, sa.ForeignKey('public.drug_route.id'), primary_key=True),
        sa.Column('metabolism_enzyme_id', sa.Integer, sa.ForeignKey('public.metabolism_enzyme.id'), primary_key=True),
        schema='public'
    )

    # Create drug_route_metabolite table
    op.create_table(
        'drug_route_metabolite',
        sa.Column('drug_route_id', sa.Integer, sa.ForeignKey('public.drug_route.id'), primary_key=True),
        sa.Column('metabolite_id', sa.Integer, sa.ForeignKey('public.metabolite.id'), primary_key=True),
        schema='public'
    )

    # Create publication table
    op.create_table(
        'publication',
        sa.Column('pmid', sa.String, primary_key=True),
        sa.Column('title', sa.Text, nullable=True),
        sa.Column('year', sa.String(4), nullable=True),
        sa.Column('journal', sa.Text, nullable=True),
        schema='public'
    )

    # Create clinical_annotation table
    op.create_table(
        'clinical_annotation',
        sa.Column('clinical_annotation_id', sa.String, primary_key=True),
        sa.Column('level_of_evidence', sa.String, nullable=True),
        sa.Column('phenotype_category', sa.String, nullable=True),
        sa.Column('url', sa.String, nullable=True),
        sa.Column('latest_history_date', sa.Date, nullable=True),
        sa.Column('specialty_population', sa.String, nullable=True),
        sa.Column('level_override', sa.String, nullable=True),
        sa.Column('level_modifiers', sa.String, nullable=True),
        sa.Column('score', sa.Float, nullable=True),
        sa.Column('pmid_count', sa.Integer, nullable=True),
        sa.Column('evidence_count', sa.Integer, nullable=True),
        schema='public'
    )

    # Create clinical_ann_allele table
    op.create_table(
        'clinical_ann_allele',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('clinical_annotation_id', sa.String, sa.ForeignKey('public.clinical_annotation.clinical_annotation_id'), nullable=False),
        sa.Column('genotype_allele', sa.String, nullable=True),
        sa.Column('annotation_text', sa.Text, nullable=True),
        sa.Column('allele_function', sa.String, nullable=True),
        schema='public'
    )

    # Create clinical_ann_evidence table
    op.create_table(
        'clinical_ann_evidence',
        sa.Column('evidence_id', sa.String, primary_key=True),
        sa.Column('clinical_annotation_id', sa.String, sa.ForeignKey('public.clinical_annotation.clinical_annotation_id'), nullable=False),
        sa.Column('evidence_type', sa.String, nullable=True),
        sa.Column('evidence_url', sa.String, nullable=True),
        sa.Column('summary', sa.Text, nullable=True),
        sa.Column('score', sa.Float, nullable=True),
        schema='public'
    )

    # Create clinical_ann_evidence_publication table
    op.create_table(
        'clinical_ann_evidence_publication',
        sa.Column('evidence_id', sa.String, sa.ForeignKey('public.clinical_ann_evidence.evidence_id'), primary_key=True),
        sa.Column('pmid', sa.String, sa.ForeignKey('public.publication.pmid'), primary_key=True),
        schema='public'
    )

    # Create clinical_ann_history table
    op.create_table(
        'clinical_ann_history',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('clinical_annotation_id', sa.String, sa.ForeignKey('public.clinical_annotation.clinical_annotation_id'), nullable=False),
        sa.Column('date', sa.Date, nullable=True),
        sa.Column('type', sa.String, nullable=True),
        sa.Column('comment', sa.Text, nullable=True),
        schema='public'
    )

    # Create clinical_annotation_drug table
    op.create_table(
        'clinical_annotation_drug',
        sa.Column('clinical_annotation_id', sa.String, sa.ForeignKey('public.clinical_annotation.clinical_annotation_id'), primary_key=True),
        sa.Column('drug_id', sa.Integer, sa.ForeignKey('public.drug.id'), primary_key=True),
        schema='public'
    )

    # Create clinical_annotation_gene table
    op.create_table(
        'clinical_annotation_gene',
        sa.Column('clinical_annotation_id', sa.String, sa.ForeignKey('public.clinical_annotation.clinical_annotation_id'), primary_key=True),
        sa.Column('gene_id', sa.String, sa.ForeignKey('public.gene.gene_id'), primary_key=True),
        schema='public'
    )

    # Create clinical_annotation_phenotype table
    op.create_table(
        'clinical_annotation_phenotype',
        sa.Column('clinical_annotation_id', sa.String, sa.ForeignKey('public.clinical_annotation.clinical_annotation_id'), primary_key=True),
        sa.Column('phenotype_id', sa.Integer, sa.ForeignKey('public.phenotype.id'), primary_key=True),
        schema='public'
    )

    # Create clinical_annotation_variant table
    op.create_table(
        'clinical_annotation_variant',
        sa.Column('clinical_annotation_id', sa.String, sa.ForeignKey('public.clinical_annotation.clinical_annotation_id'), primary_key=True),
        sa.Column('variant_id', sa.Integer, sa.ForeignKey('public.variant.id'), primary_key=True),
        schema='public'
    )

    # Create variant_annotation table
    op.create_table(
        'variant_annotation',
        sa.Column('variant_annotation_id', sa.String, primary_key=True),
        schema='public'
    )

    # Create study_parameters table
    op.create_table(
        'study_parameters',
        sa.Column('study_parameters_id', sa.String, primary_key=True),
        sa.Column('variant_annotation_id', sa.String, sa.ForeignKey('public.variant_annotation.variant_annotation_id'), nullable=True),
        sa.Column('study_type', sa.String, nullable=True),
        sa.Column('study_cases', sa.Integer, nullable=True),
        sa.Column('study_controls', sa.Integer, nullable=True),
        sa.Column('characteristics', sa.Text, nullable=True),
        sa.Column('characteristics_type', sa.String, nullable=True),
        sa.Column('frequency_in_cases', sa.Float, nullable=True),
        sa.Column('allele_of_frequency_in_cases', sa.String, nullable=True),
        sa.Column('frequency_in_controls', sa.Float, nullable=True),
        sa.Column('allele_of_frequency_in_controls', sa.String, nullable=True),
        sa.Column('p_value', sa.String, nullable=True),
        sa.Column('ratio_stat_type', sa.String, nullable=True),
        sa.Column('ratio_stat', sa.Float, nullable=True),
        sa.Column('confidence_interval_start', sa.Float, nullable=True),
        sa.Column('confidence_interval_stop', sa.Float, nullable=True),
        sa.Column('biogeographical_groups', sa.String, nullable=True),
        schema='public'
    )

    # Create variant_fa_ann table
    op.create_table(
        'variant_fa_ann',
        sa.Column('variant_annotation_id', sa.String, sa.ForeignKey('public.variant_annotation.variant_annotation_id'), primary_key=True),
        sa.Column('significance', sa.String, nullable=True),
        sa.Column('notes', sa.Text, nullable=True),
        sa.Column('sentence', sa.Text, nullable=True),
        sa.Column('alleles', sa.String, nullable=True),
        sa.Column('specialty_population', sa.String, nullable=True),
        sa.Column('assay_type', sa.String, nullable=True),
        sa.Column('metabolizer_types', sa.String, nullable=True),
        sa.Column('is_plural', sa.String, nullable=True),
        sa.Column('is_associated', sa.String, nullable=True),
        sa.Column('direction_of_effect', sa.String, nullable=True),
        sa.Column('functional_terms', sa.String, nullable=True),
        sa.Column('gene_product', sa.String, nullable=True),
        sa.Column('when_treated_with', sa.String, nullable=True),
        sa.Column('multiple_drugs', sa.String, nullable=True),
        sa.Column('cell_type', sa.String, nullable=True),
        sa.Column('comparison_alleles', sa.String, nullable=True),
        sa.Column('comparison_metabolizer_types', sa.String, nullable=True),
        schema='public'
    )

    # Create variant_drug_ann table
    op.create_table(
        'variant_drug_ann',
        sa.Column('variant_annotation_id', sa.String, sa.ForeignKey('public.variant_annotation.variant_annotation_id'), primary_key=True),
        sa.Column('significance', sa.String, nullable=True),
        sa.Column('notes', sa.Text, nullable=True),
        sa.Column('sentence', sa.Text, nullable=True),
        sa.Column('alleles', sa.String, nullable=True),
        sa.Column('specialty_population', sa.String, nullable=True),
        sa.Column('metabolizer_types', sa.String, nullable=True),
        sa.Column('is_plural', sa.String, nullable=True),
        sa.Column('is_associated', sa.String, nullable=True),
        sa.Column('direction_of_effect', sa.String, nullable=True),
        sa.Column('pd_pk_terms', sa.String, nullable=True),
        sa.Column('multiple_drugs', sa.String, nullable=True),
        sa.Column('population_types', sa.String, nullable=True),
        sa.Column('population_phenotypes_diseases', sa.String, nullable=True),
        sa.Column('multiple_phenotypes_diseases', sa.String, nullable=True),
        sa.Column('comparison_alleles', sa.String, nullable=True),
        sa.Column('comparison_metabolizer_types', sa.String, nullable=True),
        schema='public'
    )

    # Create variant_pheno_ann table
    op.create_table(
        'variant_pheno_ann',
        sa.Column('variant_annotation_id', sa.String, sa.ForeignKey('public.variant_annotation.variant_annotation_id'), primary_key=True),
        sa.Column('significance', sa.String, nullable=True),
        sa.Column('notes', sa.Text, nullable=True),
        sa.Column('sentence', sa.Text, nullable=True),
        sa.Column('alleles', sa.String, nullable=True),
        sa.Column('specialty_population', sa.String, nullable=True),
        sa.Column('metabolizer_types', sa.String, nullable=True),
        sa.Column('is_plural', sa.String, nullable=True),
        sa.Column('is_associated', sa.String, nullable=True),
        sa.Column('direction_of_effect', sa.String, nullable=True),
        sa.Column('side_effect_efficacy_other', sa.String, nullable=True),
        sa.Column('phenotype', sa.String, nullable=True),
        sa.Column('multiple_phenotypes', sa.String, nullable=True),
        sa.Column('when_treated_with', sa.String, nullable=True),
        sa.Column('multiple_drugs', sa.String, nullable=True),
        sa.Column('population_types', sa.String, nullable=True),
        sa.Column('population_phenotypes_diseases', sa.String, nullable=True),
        sa.Column('multiple_phenotypes_diseases', sa.String, nullable=True),
        sa.Column('comparison_alleles', sa.String, nullable=True),
        sa.Column('comparison_metabolizer_types', sa.String, nullable=True),
        schema='public'
    )

    # Create variant_annotation_drug table
    op.create_table(
        'variant_annotation_drug',
        sa.Column('variant_annotation_id', sa.String, sa.ForeignKey('public.variant_annotation.variant_annotation_id'), primary_key=True),
        sa.Column('drug_id', sa.Integer, sa.ForeignKey('public.drug.id'), primary_key=True),
        schema='public'
    )

    # Create variant_annotation_gene table
    op.create_table(
        'variant_annotation_gene',
        sa.Column('variant_annotation_id', sa.String, sa.ForeignKey('public.variant_annotation.variant_annotation_id'), primary_key=True),
        sa.Column('gene_id', sa.String, sa.ForeignKey('public.gene.gene_id'), primary_key=True),
        schema='public'
    )

    # Create variant_annotation_variant table
    op.create_table(
        'variant_annotation_variant',
        sa.Column('variant_annotation_id', sa.String, sa.ForeignKey('public.variant_annotation.variant_annotation_id'), primary_key=True),
        sa.Column('variant_id', sa.Integer, sa.ForeignKey('public.variant.id'), primary_key=True),
        schema='public'
    )

    # Create drug_label table
    op.create_table(
        'drug_label',
        sa.Column('pharmgkb_id', sa.String, primary_key=True),
        sa.Column('name', sa.String, nullable=True),
        sa.Column('source', sa.String, nullable=True),
        sa.Column('biomarker_flag', sa.String, nullable=True),
        sa.Column('testing_level', sa.String, nullable=True),
        sa.Column('has_prescribing_info', sa.String, nullable=True),
        sa.Column('has_dosing_info', sa.String, nullable=True),
        sa.Column('has_alternate_drug', sa.String, nullable=True),
        sa.Column('has_other_prescribing_guidance', sa.String, nullable=True),
        sa.Column('cancer_genome', sa.String, nullable=True),
        sa.Column('prescribing', sa.String, nullable=True),
        sa.Column('latest_history_date', sa.Date, nullable=True),
        schema='public'
    )

    # Create drug_label_drug table
    op.create_table(
        'drug_label_drug',
        sa.Column('pharmgkb_id', sa.String, sa.ForeignKey('public.drug_label.pharmgkb_id'), primary_key=True),
        sa.Column('drug_id', sa.Integer, sa.ForeignKey('public.drug.id'), primary_key=True),
        schema='public'
    )

    # Create drug_label_gene table
    op.create_table(
        'drug_label_gene',
        sa.Column('pharmgkb_id', sa.String, sa.ForeignKey('public.drug_label.pharmgkb_id'), primary_key=True),
        sa.Column('gene_id', sa.String, sa.ForeignKey('public.gene.gene_id'), primary_key=True),
        schema='public'
    )

    # Create drug_label_variant table
    op.create_table(
        'drug_label_variant',
        sa.Column('pharmgkb_id', sa.String, sa.ForeignKey('public.drug_label.pharmgkb_id'), primary_key=True),
        sa.Column('variant_id', sa.Integer, sa.ForeignKey('public.variant.id'), primary_key=True),
        schema='public'
    )

    # Create clinical_variants table
    op.create_table(
        'clinical_variants',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('variant_type', sa.String, nullable=True),
        sa.Column('level_of_evidence', sa.String, nullable=True),
        sa.Column('gene_id', sa.String, sa.ForeignKey('public.gene.gene_id'), nullable=True),
        schema='public'
    )

    # Create clinical_variant_drug table
    op.create_table(
        'clinical_variant_drug',
        sa.Column('clinical_variant_id', sa.Integer, sa.ForeignKey('public.clinical_variants.id'), primary_key=True),
        sa.Column('drug_id', sa.Integer, sa.ForeignKey('public.drug.id'), primary_key=True),
        schema='public'
    )

    # Create clinical_variant_phenotype table
    op.create_table(
        'clinical_variant_phenotype',
        sa.Column('clinical_variant_id', sa.Integer, sa.ForeignKey('public.clinical_variants.id'), primary_key=True),
        sa.Column('phenotype_id', sa.Integer, sa.ForeignKey('public.phenotype.id'), primary_key=True),
        schema='public'
    )

    # Create clinical_variant_variant table
    op.create_table(
        'clinical_variant_variant',
        sa.Column('clinical_variant_id', sa.Integer, sa.ForeignKey('public.clinical_variants.id'), primary_key=True),
        sa.Column('variant_id', sa.Integer, sa.ForeignKey('public.variant.id'), primary_key=True),
        schema='public'
    )

    # Create relationships table
    op.create_table(
        'relationships',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('entity1_id', sa.String, nullable=True),
        sa.Column('entity1_name', sa.String, nullable=True),
        sa.Column('entity1_type', sa.String, nullable=True),
        sa.Column('entity2_id', sa.String, nullable=True),
        sa.Column('entity2_name', sa.String, nullable=True),
        sa.Column('entity2_type', sa.String, nullable=True),
        sa.Column('evidence', sa.String, nullable=True),
        sa.Column('association', sa.String, nullable=True),
        sa.Column('pk', sa.String, nullable=True),
        sa.Column('pd', sa.String, nullable=True),
        sa.Column('pmids', sa.String, nullable=True),
        schema='public'
    )

    # Create occurrences table
    op.create_table(
        'occurrences',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('source_type', sa.String, nullable=True),
        sa.Column('source_id', sa.String, nullable=True),
        sa.Column('source_name', sa.Text, nullable=True),
        sa.Column('object_type', sa.String, nullable=True),
        sa.Column('object_id', sa.String, nullable=True),
        sa.Column('object_name', sa.String, nullable=True),
        schema='public'
    )

    # Create automated_annotations table
    op.create_table(
        'automated_annotations',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('chemical_id', sa.String, nullable=True),
        sa.Column('chemical_name', sa.String, nullable=True),
        sa.Column('chemical_in_text', sa.String, nullable=True),
        sa.Column('variation_id', sa.String, nullable=True),
        sa.Column('variation_name', sa.String, nullable=True),
        sa.Column('variation_type', sa.String, nullable=True),
        sa.Column('variation_in_text', sa.String, nullable=True),
        sa.Column('gene_ids', sa.Text, nullable=True),
        sa.Column('gene_symbols', sa.Text, nullable=True),
        sa.Column('gene_in_text', sa.Text, nullable=True),
        sa.Column('literature_id', sa.String, nullable=True),
        sa.Column('literature_title', sa.Text, nullable=True),
        sa.Column('publication_year', sa.String, nullable=True),
        sa.Column('journal', sa.Text, nullable=True),
        sa.Column('sentence', sa.Text, nullable=True),
        sa.Column('source', sa.String, nullable=True),
        sa.Column('pmid', sa.String, sa.ForeignKey('public.publication.pmid'), nullable=True),
        schema='public'
    )

    # Create news table
    op.create_table(
        'news',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('title', sa.String(200), nullable=False),
        sa.Column('description', sa.Text, nullable=False),
        sa.Column('category', sa.String(50), nullable=False),
        sa.Column('publication_date', sa.Date, nullable=False, default=datetime.utcnow),
        sa.Column('created_at', sa.DateTime, default=datetime.utcnow),
        sa.Column('updated_at', sa.DateTime, default=datetime.utcnow),
        schema='public'
    )

    # Create occupation table
    op.create_table(
        'occupation',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('name', sa.String(100), unique=True, nullable=False),
        schema='public'
    )

    # Create user table
    op.create_table(
        'user',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('email', sa.String(120), unique=True, nullable=False),
        sa.Column('password', sa.String(128), nullable=False),
        sa.Column('name', sa.String(50), nullable=False),
        sa.Column('surname', sa.String(50), nullable=False),
        sa.Column('date_of_birth', sa.Date, nullable=False),
        sa.Column('occupation', sa.String(100), nullable=True),
        sa.Column('is_admin', sa.Boolean, default=False),
        schema='public'
    )

def downgrade():
    op.drop_table('user', schema='public')
    op.drop_table('occupation', schema='public')
    op.drop_table('news', schema='public')
    op.drop_table('automated_annotations', schema='public')
    op.drop_table('occurrences', schema='public')
    op.drop_table('relationships', schema='public')
    op.drop_table('clinical_variant_variant', schema='public')
    op.drop_table('clinical_variant_phenotype', schema='public')
    op.drop_table('clinical_variant_drug', schema='public')
    op.drop_table('clinical_variants', schema='public')
    op.drop_table('drug_label_variant', schema='public')
    op.drop_table('drug_label_gene', schema='public')
    op.drop_table('drug_label_drug', schema='public')
    op.drop_table('drug_label', schema='public')
    op.drop_table('variant_annotation_variant', schema='public')
    op.drop_table('variant_annotation_gene', schema='public')
    op.drop_table('variant_annotation_drug', schema='public')
    op.drop_table('variant_pheno_ann', schema='public')
    op.drop_table('variant_drug_ann', schema='public')
    op.drop_table('variant_fa_ann', schema='public')
    op.drop_table('study_parameters', schema='public')
    op.drop_table('variant_annotation', schema='public')
    op.drop_table('clinical_annotation_variant', schema='public')
    op.drop_table('clinical_annotation_phenotype', schema='public')
    op.drop_table('clinical_annotation_gene', schema='public')
    op.drop_table('clinical_annotation_drug', schema='public')
    op.drop_table('clinical_ann_history', schema='public')
    op.drop_table('clinical_ann_evidence_publication', schema='public')
    op.drop_table('clinical_ann_evidence', schema='public')
    op.drop_table('clinical_ann_allele', schema='public')
    op.drop_table('clinical_annotation', schema='public')
    op.drop_table('publication', schema='public')
    op.drop_table('drug_route_metabolite', schema='public')
    op.drop_table('drug_route_metabolism_enzyme', schema='public')
    op.drop_table('drug_route_metabolism_organ', schema='public')
    op.drop_table('pathway_drug', schema='public')
    op.drop_table('route_indication', schema='public')
    op.drop_table('drug_route', schema='public')
    op.drop_table('drug_disease_interaction', schema='public')
    op.drop_table('drug_receptor_interaction', schema='public')
    op.drop_table('drug_interaction', schema='public')
    op.drop_table('detail_side_effect', schema='public')
    op.drop_table('side_effect', schema='public')
    op.drop_table('drug_detail', schema='public')
    op.drop_table('drug_salt', schema='public')
    op.drop_table('drug', schema='public')
    op.drop_table('variant', schema='public')
    op.drop_table('phenotype', schema='public')
    op.drop_table('gene', schema='public')
    op.drop_table('metabolite', schema='public')
    op.drop_table('metabolism_enzyme', schema='public')
    op.drop_table('metabolism_organ', schema='public')
    op.drop_table('pathway', schema='public')
    op.drop_table('receptor', schema='public')
    op.drop_table('route_of_administration', schema='public')
    op.drop_table('target', schema='public')
    op.drop_table('indication', schema='public')
    op.drop_table('salt', schema='public')
    op.drop_table('drug_category', schema='public')