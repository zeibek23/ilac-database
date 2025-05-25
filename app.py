from flask import Flask, render_template, render_template_string, request, session, redirect, url_for, jsonify, make_response, flash, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from dotenv import load_dotenv
from openpyxl import load_workbook
from flask_migrate import Migrate
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for Matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import seaborn as sns
import requests
import smtplib
from email.mime.text import MIMEText
import uuid  # Import the uuid library for generating unique file names
import subprocess
import os
os.environ["TRANSFORMERS_BACKEND"] = "torch"

from transformers import pipeline

import urllib.parse  # For proper URL encoding
import time
import logging
import nltk
import traceback
import json
import xml.etree.ElementTree as ET
import csv
import ssl
import certifi
import numpy as np
from itertools import combinations
import spacy
import nltk
import os
# SSL sertifikası sorunlarını atlatmak için
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# NLTK veri seti dizini
NLTK_DATA_DIR = os.environ.get('NLTK_DATA', '/tmp/nltk_data' if os.environ.get('RENDER') == 'true' else 'C:\\Users\\ENES\\nltk_data')
nltk.data.path.append(NLTK_DATA_DIR)

# Render ortamında NLTK veri setlerini indirme, yerel ortamda indir
if os.environ.get('RENDER') != 'true':
    nltk.download('brown', download_dir=NLTK_DATA_DIR)
    nltk.download('punkt', download_dir=NLTK_DATA_DIR)

# Çalışma dizinini kontrol et (yerel ortam için)
if os.getcwd() != 'C:\\Users\\ENES\\Desktop\\ilac-database' and os.environ.get('RENDER') != 'true':
    os.chdir('C:\\Users\\ENES\\Desktop\\ilac-database')

from textblob import TextBlob
from dash import Dash, dcc, html, Input, Output
from sqlalchemy.sql import text, func
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, Date, ForeignKey, or_, nullslast, outerjoin, func, and_, select
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from io import BytesIO
from Bio import Entrez
from flask_login import UserMixin
from flask_login import current_user
from pydantic import BaseModel, field_validator, ValidationError
from typing import List
from http import HTTPStatus
from Bio.PDB import PDBParser
from werkzeug.utils import secure_filename
from datetime import datetime, date
from scipy.integrate import odeint
from functools import wraps
import sqlalchemy
from sqlalchemy import and_, or_, nullslast, extract, inspect
from rdkit import Chem
from rdkit.Chem import AllChem


load_dotenv()
app = Flask(__name__)
# Flask uygulamasını oluştur
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'postgresql://postgres.enrowjcfkauuutemluhd:Alnesuse200824_@aws-0-eu-central-1.pooler.supabase.com:5432/postgres')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'super-secret-key-123')

# Debug: Kullanılan DATABASE_URI’yi yazdır
print("DATABASE_URI:", app.config['SQLALCHEMY_DATABASE_URI'])

# SQLAlchemy nesnesini oluştur ve uygulamaya bağla
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
migrate = Migrate(app, db)  # Initialize Flask-Migrate

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_FOLDER = "uploaded_files"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# Define junction tables globally
drug_route_metabolism_organ = db.Table(
    'drug_route_metabolism_organ',
    db.Column('drug_route_id', db.Integer, db.ForeignKey('public.drug_route.id'), primary_key=True),
    db.Column('metabolism_organ_id', db.Integer, db.ForeignKey('public.metabolism_organ.id'), primary_key=True),
    schema='public',
    extend_existing=True
)

drug_route_metabolism_enzyme = db.Table(
    'drug_route_metabolism_enzyme',
    db.Column('drug_route_id', db.Integer, db.ForeignKey('public.drug_route.id'), primary_key=True),
    db.Column('metabolism_enzyme_id', db.Integer, db.ForeignKey('public.metabolism_enzyme.id'), primary_key=True),
    schema='public',
    extend_existing=True
)

drug_route_metabolite = db.Table(
    'drug_route_metabolite',
    db.Column('drug_route_id', db.Integer, db.ForeignKey('public.drug_route.id'), primary_key=True),
    db.Column('metabolite_id', db.Integer, db.ForeignKey('public.metabolite.id'), primary_key=True),
    schema='public',
    extend_existing=True
)

# Many-to-Many İlişkisi için Ara Tablo
drug_salt = db.Table(
    'drug_salt',
    db.Column('drug_id', db.Integer, db.ForeignKey('public.drug.id'), primary_key=True),
    db.Column('salt_id', db.Integer, db.ForeignKey('public.salt.id'), primary_key=True),
    schema='public',
    extend_existing=True
)

# Many-to-Many ilişki tablosu
detail_side_effect = db.Table(
    'detail_side_effect',
    db.Column('detail_id', db.Integer, db.ForeignKey('public.drug_detail.id'), primary_key=True),
    db.Column('side_effect_id', db.Integer, db.ForeignKey('public.side_effect.id'), primary_key=True),
    schema='public',
    extend_existing=True
)

pathway_drug = db.Table(
    'pathway_drug',
    db.Column('pathway_id', db.Integer, db.ForeignKey('public.pathway.id'), primary_key=True),
    db.Column('drug_id', db.Integer, db.ForeignKey('public.drug.id'), primary_key=True),
    schema='public',
    extend_existing=True
)

# Veritabanı Modeli
class DrugCategory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    name = db.Column(db.String(50), nullable=False, unique=True)  # e.g., "Cholinergic Agonists"
    parent_id = db.Column(db.Integer, db.ForeignKey('public.drug_category.id'), nullable=True)  # Self-referential FK
    parent = db.relationship('DrugCategory', remote_side=[id], backref='children')  # Parent-child relationship

class Salt(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    name_tr = db.Column(db.String(100), nullable=False)
    name_en = db.Column(db.String(100), nullable=False)    

class Drug(db.Model):
    __tablename__ = 'drug'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    id = db.Column(db.Integer, primary_key=True)
    name_tr = db.Column(db.String(255), nullable=False)
    name_en = db.Column(db.String(255), nullable=False)
    alternative_names = db.Column(db.Text, nullable=True)
    salts = db.relationship('Salt', secondary=drug_salt, backref='drugs', lazy='dynamic')
    fda_approved = db.Column(db.Boolean, nullable=False, default=False)
    indications = db.Column(db.Text, nullable=True)
    chembl_id = db.Column(db.String(50), nullable=True)
    category_id = db.Column(db.Integer, db.ForeignKey('public.drug_category.id'), nullable=True)
    category = db.relationship('DrugCategory', backref='drugs')
    drug_details = db.relationship(
        'DrugDetail',
        back_populates='parent_drug',
        cascade="all, delete-orphan"
    )
    # Added relationships for PharmGKB integration
    pharmgkb_id = db.Column(db.String(50), unique=True, nullable=True)  # To store PharmGKB chemical IDs (e.g., PA449412)
    clinical_annotations = db.relationship('ClinicalAnnotationDrug', back_populates='drug')
    drug_labels = db.relationship('DrugLabelDrug', back_populates='drug')
    clinical_variants = db.relationship('ClinicalVariantDrug', back_populates='drug')
    variant_annotations = db.relationship('VariantAnnotationDrug', back_populates='drug')



class DrugDetail(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    drug_id = db.Column(db.Integer, db.ForeignKey('public.drug.id'), nullable=False)
    parent_drug = db.relationship(
        'Drug',
        back_populates='drug_details',
        overlaps="details,drug"
    ) 
    routes = db.relationship("DrugRoute", back_populates="drug_detail", cascade="all, delete-orphan")
    salt_id = db.Column(db.Integer, db.ForeignKey('public.salt.id'), nullable=True)  # Opsiyonel
    molecular_formula = db.Column(db.String(100), nullable=True)
    synthesis = db.Column(db.Text, nullable=True)
    structure = db.Column(db.String(200), nullable=True)     # manuel yüklenilen
    #structure_2d = db.Column(db.String(200), nullable=True)  # 2D yapı
    structure_3d = db.Column(db.String(200), nullable=True)  # 3D yapı
    mechanism_of_action = db.Column(db.Text, nullable=True)  # Add this line
    iupac_name = db.Column(db.String(200), nullable=True)
    smiles = db.Column(db.String(200), nullable=True)
    inchikey = db.Column(db.String(200), nullable=True)
    pubchem_cid = db.Column(db.String(50), nullable=True)
    pubchem_sid = db.Column(db.String(50), nullable=True)
    cas_id = db.Column(db.String(50), nullable=True)
    ec_number = db.Column(db.String(50), nullable=True)
    nci_code = db.Column(db.String(50), nullable=True)
    rxcui = db.Column(db.String(50), nullable=True)
    snomed_id = db.Column(db.String(50), nullable=True)
    molecular_weight = db.Column(db.Float, nullable=True)
    #solubility = db.Column(db.Text, nullable=True)
    pharmacodynamics = db.Column(db.Text, nullable=True)
    indications = db.Column(db.Text, nullable=True)
    target_molecules = db.Column(db.Text, nullable=True)
    pharmacokinetics = db.Column(db.Text, nullable=True)
    boiling_point = db.Column(db.String(100), nullable=True)  # Kaynama Noktası
    melting_point = db.Column(db.String(100), nullable=True)  # Erime Noktası
    density = db.Column(db.String(100), nullable=True)        # Yoğunluk
    solubility = db.Column(db.String(200), nullable=True)     # Çözünürlük
    flash_point = db.Column(db.String(100), nullable=True)    # Alevlenme Noktası
    fda_approved = db.Column(db.Boolean, nullable=False, default=False)
    ema_approved = db.Column(db.Boolean, nullable=False, default=False)
    titck_approved = db.Column(db.Boolean, nullable=False, default=False)
    black_box_warning = db.Column(db.Boolean, nullable=False, default=False)  # Checkbox for black box warning
    black_box_details = db.Column(db.Text, nullable=True)  # Text box for details if the warning exists
    half_life = db.Column(db.Float, nullable=True)  # Hours
    clearance_rate = db.Column(db.Float, nullable=True)  # mL/min
    bioavailability = db.Column(db.Float, nullable=True)  # Fraction (0-1)

    # İlişkiler
    drug = db.relationship('Drug', backref=db.backref('details', lazy=True))
    salt = db.relationship('Salt', backref=db.backref('details', lazy=True))

# Database Model for Indication
class Indication(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    foundation_uri = db.Column(db.String(255), nullable=True, unique=True, index=True)
    disease_id = db.Column(db.String(255), nullable=True, unique=True, index=True)
    name_en = db.Column(db.String(255), nullable=False)
    name_tr = db.Column(db.String(255), nullable=True)
    description = db.Column(db.Text, nullable=True)
    synonyms = db.Column(db.Text, nullable=True)
    species = db.Column(db.String(50), nullable=True, default="Human")
    code = db.Column(db.String(50), nullable=True, index=True)
    class_kind = db.Column(db.String(50), nullable=True)
    depth = db.Column(db.Integer, nullable=True)
    parent_id = db.Column(db.Integer, db.ForeignKey('public.indication.id'), nullable=True, index=True)
    parent = db.relationship('Indication', remote_side=[id], backref=db.backref('children', lazy='select'))
    is_residual = db.Column(db.Boolean, nullable=False, default=False)
    is_leaf = db.Column(db.Boolean, nullable=False, default=False)
    chapter_no = db.Column(db.String(10), nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    BlockId = db.Column(db.String(50), nullable=True, index=True)

    def validate(self):
        if not self.name_en:
            raise ValueError("name_en is required")

    @property
    def has_children(self):
        count = Indication.query.filter_by(parent_id=self.id).count()
        return count > 0

class Target(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    name_tr = db.Column(db.String(255), nullable=False)  # Increased to 255
    name_en = db.Column(db.String(255), nullable=False)  # Increased to 255

class DrugInteraction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    drug1_id = db.Column(db.Integer, db.ForeignKey('public.drug.id'), nullable=False)
    drug2_id = db.Column(db.Integer, db.ForeignKey('public.drug.id'), nullable=False)
    route_id = db.Column(db.Integer, db.ForeignKey('public.route_of_administration.id'), nullable=True)  # Uygulama yolu
    interaction_type = db.Column(db.String(50), nullable=False)
    interaction_description = db.Column(db.Text, nullable=False)
    severity = db.Column(db.String(20), nullable=False, default="Hafif")  # Şiddet kategorisi
    mechanism = db.Column(db.Text, nullable=True)  # Mekanizma açıklaması
    pharmacokinetics = db.Column(db.Text, nullable=True)  # Farmakokinetik bilgi
    monitoring = db.Column(db.Text, nullable=True)  # İzleme önerileri
    alternatives = db.Column(db.Text, nullable=True)  # Alternatif ilaçlar
    reference = db.Column(db.Text, nullable=True)  # Kaynak/referans
    predicted_severity = db.Column(db.String(50), nullable=True) #AI'nin şiddet tahmini
    prediction_confidence = db.Column(db.Float)  # Confidence score from AI prediction
    processed = db.Column(db.Boolean, default=False)  # Track processing status
    time_to_peak = db.Column(db.Float, nullable=True)  # Hours until max risk
    
    # İlişkiler
    drug1 = db.relationship("Drug", foreign_keys=[drug1_id])
    drug2 = db.relationship("Drug", foreign_keys=[drug2_id])
    route = db.relationship("RouteOfAdministration", backref="interactions")  # Uygulama yolu ilişkisi


class RouteOfAdministration(db.Model):
    __tablename__ = 'route_of_administration'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    type = db.Column(db.String(50), nullable=False)  # Sistemik, Lokal veya Hem Sistemik Hem Lokal
    description = db.Column(db.Text, nullable=True)  # e.g., "Absorbed via GI tract"
    parent_id = db.Column(db.Integer, db.ForeignKey('public.route_of_administration.id'), nullable=True)
    parent = db.relationship(
        "RouteOfAdministration",
        remote_side=[id],
        backref=db.backref("children", cascade="all, delete-orphan")
    )


class Receptor(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    name = db.Column(db.String(100), nullable=False)
    type = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=True)
    molecular_weight = db.Column(db.String(50), nullable=True)  # Molecular Weight sütunu
    length = db.Column(db.Integer, nullable=True)              # Length sütunu
    gene_name = db.Column(db.String(100), nullable=True)       # Gene Name sütunu
    subcellular_location = db.Column(db.Text, nullable=True)   # Subcellular Location sütunu
    function = db.Column(db.Text, nullable=True)               # Function sütunu
    iuphar_id = db.Column(db.String(50))  # Yeni Alan
    pdb_ids = db.Column(db.Text, nullable=True)  # Add this field
    binding_site_x = db.Column(db.Float, nullable=True)
    binding_site_y = db.Column(db.Float, nullable=True)
    binding_site_z = db.Column(db.Float, nullable=True)




class DrugReceptorInteraction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    drug_id = db.Column(db.Integer, db.ForeignKey('public.drug.id'), nullable=False)
    receptor_id = db.Column(db.Integer, db.ForeignKey('public.receptor.id'), nullable=False)
    affinity = db.Column(db.Float, nullable=True)  # e.g., Kd or Ki value
    interaction_type = db.Column(db.String(50), nullable=True)  # Agonist, antagonist, etc.
    mechanism = db.Column(db.Text, nullable=True)  # Interaction mechanism
    pdb_file = db.Column(db.String(200), nullable=True)  # Optional PDB structure file
    units = db.Column(db.String, nullable=True)  # Add this line
    affinity_parameter = db.Column(db.String(50), nullable=True)  # New column


    drug = db.relationship('Drug', backref=db.backref('interactions', lazy=True))
    receptor = db.relationship('Receptor', backref=db.backref('interactions', lazy=True))


class DrugDiseaseInteraction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    drug_id = db.Column(db.Integer, db.ForeignKey('public.drug.id'), nullable=False)
    indication_id = db.Column(db.Integer, db.ForeignKey('public.indication.id'), nullable=False)
    interaction_type = db.Column(db.String(50), nullable=False)  # e.g., "Contraindication", "Caution"
    description = db.Column(db.Text, nullable=True)              # Why it’s a problem
    severity = db.Column(db.String(20), default="Moderate")      # e.g., "Mild", "Moderate", "Severe"
    recommendation = db.Column(db.Text, nullable=True)           # What to do about it

    drug = db.relationship("Drug", backref="disease_interactions")
    indication = db.relationship("Indication", backref="drug_interactions")      


class DrugRoute(db.Model):
    __tablename__ = 'drug_route'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    id = db.Column(db.Integer, primary_key=True)
    drug_detail_id = db.Column(db.Integer, db.ForeignKey('public.drug_detail.id', ondelete="CASCADE"), nullable=False)
    route_id = db.Column(db.Integer, db.ForeignKey('public.route_of_administration.id', ondelete="CASCADE"), nullable=False)
    pharmacodynamics = db.Column(db.Text, nullable=True)  # Uygulama yoluna özel farmakodinami Genel bilgi
    pharmacokinetics = db.Column(db.Text, nullable=True)  # Uygulama yoluna özel farmakokinetik Genel bilgi
    absorption_rate_min = db.Column(db.Float, nullable=True)
    absorption_rate_max = db.Column(db.Float, nullable=True)
    vod_rate_min = db.Column(db.Float, nullable=True)
    vod_rate_max = db.Column(db.Float, nullable=True)
    protein_binding_min = db.Column(db.Float, nullable=True)
    protein_binding_max = db.Column(db.Float, nullable=True)
    half_life_min = db.Column(db.Float, nullable=True)
    half_life_max = db.Column(db.Float, nullable=True)
    clearance_rate_min = db.Column(db.Float, nullable=True)
    clearance_rate_max = db.Column(db.Float, nullable=True)
    bioavailability_min = db.Column(db.Float, nullable=True)  # Fraction (0-1)
    bioavailability_max = db.Column(db.Float, nullable=True)
    tmax_min = db.Column(db.Float, nullable=True)  # Hours
    tmax_max = db.Column(db.Float, nullable=True)
    cmax_min = db.Column(db.Float, nullable=True)  # mg/L
    cmax_max = db.Column(db.Float, nullable=True)

    route = db.relationship("RouteOfAdministration")
    drug_detail = db.relationship("DrugDetail", back_populates="routes")
    metabolism_organs = db.relationship('MetabolismOrgan', secondary=drug_route_metabolism_organ, backref='drug_routes')
    metabolism_enzymes = db.relationship('MetabolismEnzyme', secondary=drug_route_metabolism_enzyme, backref='drug_routes')
    metabolites = db.relationship('Metabolite', secondary=drug_route_metabolite, backref='routes')    

    route_indications = db.relationship(
        "RouteIndication",
        backref="drug_route",
        cascade="all, delete-orphan"
    )


class RouteIndication(db.Model):
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    id = db.Column(db.Integer, primary_key=True)
    drug_detail_id = db.Column(db.Integer, db.ForeignKey('public.drug_detail.id'), nullable=False)
    route_id = db.Column(db.Integer, db.ForeignKey('public.route_of_administration.id'), nullable=False)
    indication_id = db.Column(db.Integer, db.ForeignKey('public.indication.id'), nullable=False)

    drug_detail = db.relationship('DrugDetail', backref='route_indications')
    route = db.relationship('RouteOfAdministration')
    indication = db.relationship('Indication')
    drug_route_id = db.Column(db.Integer, db.ForeignKey('public.drug_route.id', ondelete="CASCADE"), nullable=True)


class SideEffect(db.Model):
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    id = db.Column(db.Integer, primary_key=True)
    name_en = db.Column(db.String(100), nullable=False)  # İngilizce adı
    name_tr = db.Column(db.String(100), nullable=True)   # Türkçe adı (opsiyonel)
    details = db.relationship('DrugDetail', secondary=detail_side_effect, backref='side_effects', lazy='dynamic')

class Pathway(db.Model):
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    id = db.Column(db.Integer, primary_key=True)
    pathway_id = db.Column(db.String(50), unique=True, nullable=False)  # KEGG pathway ID (örneğin: hsa00010)
    name = db.Column(db.String(255), nullable=False)  # Pathway adı
    description = db.Column(db.Text, nullable=True)  # Pathway açıklaması
    organism = db.Column(db.String(50), nullable=False)  # Organism (örn: hsa, mmu)
    url = db.Column(db.String(255), nullable=True)  # KEGG pathway görseli URL'si
    drugs = db.relationship('Drug', secondary=pathway_drug, backref='pathways', lazy='dynamic')



class MetabolismOrgan(db.Model):
    __tablename__ = 'metabolism_organ'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique=True)  # e.g., "liver", "kidneys"

class MetabolismEnzyme(db.Model):
    __tablename__ = 'metabolism_enzyme'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique=True)  # e.g., "CYP2E1", "UGT"

class Metabolite(db.Model):
    __tablename__ = 'metabolite'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)  # e.g., "Acetaminophen", "NAPQI"
    parent_id = db.Column(db.Integer, db.ForeignKey('public.metabolite.id'), nullable=True)  # For hierarchy
    drug_id = db.Column(db.Integer, db.ForeignKey('public.drug.id'), nullable=True)  # Link to the parent drug
    parent = db.relationship("Metabolite", remote_side=[id], backref="children")
    drug = db.relationship("Drug", backref="metabolites")  # Link to the drug
    drug_routes = db.relationship('DrugRoute', secondary=drug_route_metabolite, backref='metabolite_routes')  

# PharmGKB için database modelleri:
# Clinical Annotations
class Gene(db.Model):
    __tablename__ = 'gene'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    gene_id = Column(String(50), primary_key=True)  # e.g., PA126 for CYP2C9
    gene_symbol = Column(String(100), unique=True, nullable=False)
    clinical_annotations = relationship('ClinicalAnnotationGene', back_populates='gene')
    variant_annotations = relationship('VariantAnnotationGene', back_populates='gene')
    drug_labels = relationship('DrugLabelGene', back_populates='gene')
    clinical_variants = relationship('ClinicalVariant', back_populates='gene')

class Phenotype(db.Model):
    __tablename__ = 'phenotype'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    id = Column(Integer, primary_key=True)
    name = Column(String(200), unique=True, nullable=False)
    clinical_annotations = relationship('ClinicalAnnotationPhenotype', back_populates='phenotype')
    clinical_variants = relationship('ClinicalVariantPhenotype', back_populates='phenotype')

class Variant(db.Model):
    __tablename__ = 'variant'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    id = db.Column(db.Integer, primary_key=True)
    pharmgkb_id = db.Column(db.String(50), unique=True, nullable=True)
    name = db.Column(db.String(200), unique=True, nullable=False)
    clinical_annotations = db.relationship('ClinicalAnnotationVariant', back_populates='variant')
    variant_annotations = db.relationship('VariantAnnotationVariant', back_populates='variant')
    drug_labels = db.relationship('DrugLabelVariant', back_populates='variant')
    clinical_variants = db.relationship('ClinicalVariantVariant', back_populates='variant')

class Publication(db.Model):
    __tablename__ = 'publication'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    pmid = Column(String(50), primary_key=True)
    title = Column(Text, nullable=True)
    year = Column(String(4), nullable=True)
    journal = Column(Text, nullable=True)
    clinical_evidence = relationship('ClinicalAnnEvidencePublication', back_populates='publication')
    automated_annotations = relationship('AutomatedAnnotation', back_populates='publication')

# Clinical Annotations
class ClinicalAnnotation(db.Model):
    __tablename__ = 'clinical_annotation'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    clinical_annotation_id = Column(String, primary_key=True)
    level_of_evidence = Column(String)
    phenotype_category = Column(String)
    url = Column(String)
    latest_history_date = Column(Date)
    specialty_population = Column(String, nullable=True)
    level_override = Column(String, nullable=True)
    level_modifiers = Column(String, nullable=True)
    score = Column(Float)
    pmid_count = Column(Integer)
    evidence_count = Column(Integer)
    alleles = relationship('ClinicalAnnAllele', back_populates='annotation')
    history = relationship('ClinicalAnnHistory', back_populates='annotation')
    evidence = relationship('ClinicalAnnEvidence', back_populates='annotation')
    drugs = relationship('ClinicalAnnotationDrug', back_populates='annotation')
    genes = relationship('ClinicalAnnotationGene', back_populates='annotation')
    phenotypes = relationship('ClinicalAnnotationPhenotype', back_populates='annotation')
    variants = relationship('ClinicalAnnotationVariant', back_populates='annotation')

class ClinicalAnnAllele(db.Model):
    __tablename__ = 'clinical_ann_allele'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    id = Column(Integer, primary_key=True)
    clinical_annotation_id = Column(String, ForeignKey('public.clinical_annotation.clinical_annotation_id'))
    genotype_allele = Column(String, index=True)
    annotation_text = Column(Text)
    allele_function = Column(String, nullable=True)
    annotation = relationship('ClinicalAnnotation', back_populates='alleles')

class ClinicalAnnHistory(db.Model):
    __tablename__ = 'clinical_ann_history'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    id = Column(Integer, primary_key=True)
    clinical_annotation_id = Column(String, ForeignKey('public.clinical_annotation.clinical_annotation_id'))
    date = Column(Date, index=True)
    type = Column(String)
    comment = Column(Text)
    annotation = relationship('ClinicalAnnotation', back_populates='history')

class ClinicalAnnEvidence(db.Model):
    __tablename__ = 'clinical_ann_evidence'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    evidence_id = Column(String, primary_key=True)
    clinical_annotation_id = Column(String, ForeignKey('public.clinical_annotation.clinical_annotation_id'))
    evidence_type = Column(String, index=True)
    evidence_url = Column(String)
    summary = Column(Text)
    score = Column(Float)
    annotation = relationship('ClinicalAnnotation', back_populates='evidence')
    publications = relationship('ClinicalAnnEvidencePublication', back_populates='evidence')

# Junction Tables for ClinicalAnnotation
class ClinicalAnnotationDrug(db.Model):
    __tablename__ = 'clinical_annotation_drug'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    clinical_annotation_id = Column(String, ForeignKey('public.clinical_annotation.clinical_annotation_id'), primary_key=True)
    drug_id = Column(Integer, ForeignKey('public.drug.id'), primary_key=True)
    annotation = relationship('ClinicalAnnotation', back_populates='drugs')
    drug = relationship('Drug', back_populates='clinical_annotations')

class ClinicalAnnotationGene(db.Model):
    __tablename__ = 'clinical_annotation_gene'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    clinical_annotation_id = Column(String, ForeignKey('public.clinical_annotation.clinical_annotation_id'), primary_key=True)
    gene_id = Column(String, ForeignKey('public.gene.gene_id'), primary_key=True)
    annotation = relationship('ClinicalAnnotation', back_populates='genes')
    gene = relationship('Gene', back_populates='clinical_annotations')

class ClinicalAnnotationPhenotype(db.Model):
    __tablename__ = 'clinical_annotation_phenotype'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    clinical_annotation_id = Column(String, ForeignKey('public.clinical_annotation.clinical_annotation_id'), primary_key=True)
    phenotype_id = Column(Integer, ForeignKey('public.phenotype.id'), primary_key=True)
    annotation = relationship('ClinicalAnnotation', back_populates='phenotypes')
    phenotype = relationship('Phenotype', back_populates='clinical_annotations')

class ClinicalAnnotationVariant(db.Model):
    __tablename__ = 'clinical_annotation_variant'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    clinical_annotation_id = Column(String, ForeignKey('public.clinical_annotation.clinical_annotation_id'), primary_key=True)
    variant_id = Column(Integer, ForeignKey('public.variant.id'), primary_key=True)
    annotation = relationship('ClinicalAnnotation', back_populates='variants')
    variant = relationship('Variant', back_populates='clinical_annotations')

class ClinicalAnnEvidencePublication(db.Model):
    __tablename__ = 'clinical_ann_evidence_publication'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    evidence_id = Column(String, ForeignKey('public.clinical_ann_evidence.evidence_id'), primary_key=True)
    pmid = Column(String, ForeignKey('public.publication.pmid'), primary_key=True)
    evidence = relationship('ClinicalAnnEvidence', back_populates='publications')
    publication = relationship('Publication', back_populates='clinical_evidence')

# Variant Annotations
class VariantAnnotation(db.Model):
    __tablename__ = 'variant_annotation'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    variant_annotation_id = Column(String, primary_key=True)
    study_parameters = relationship('StudyParameters', back_populates='variant_annotation')
    fa_annotations = relationship('VariantFAAnn', back_populates='variant_annotation')
    drug_annotations = relationship('VariantDrugAnn', back_populates='variant_annotation')
    pheno_annotations = relationship('VariantPhenoAnn', back_populates='variant_annotation')
    genes = relationship('VariantAnnotationGene', back_populates='variant_annotation')
    variants = relationship('VariantAnnotationVariant', back_populates='variant_annotation')
    drugs = relationship('VariantAnnotationDrug', back_populates='variant_annotation')

class StudyParameters(db.Model):
    __tablename__ = 'study_parameters'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    study_parameters_id = Column(String, primary_key=True)
    variant_annotation_id = Column(String, ForeignKey('public.variant_annotation.variant_annotation_id'))
    study_type = Column(String, nullable=True)
    study_cases = Column(Integer, nullable=True)
    study_controls = Column(Integer, nullable=True)
    characteristics = Column(Text, nullable=True)
    characteristics_type = Column(String, nullable=True)
    frequency_in_cases = Column(Float, nullable=True)
    allele_of_frequency_in_cases = Column(String, nullable=True)
    frequency_in_controls = Column(Float, nullable=True)
    allele_of_frequency_in_controls = Column(String, nullable=True)
    p_value = Column(String, nullable=True)
    ratio_stat_type = Column(String, nullable=True)
    ratio_stat = Column(Float, nullable=True)
    confidence_interval_start = Column(Float, nullable=True)
    confidence_interval_stop = Column(Float, nullable=True)
    biogeographical_groups = Column(String, nullable=True)
    variant_annotation = relationship('VariantAnnotation', back_populates='study_parameters')

class VariantFAAnn(db.Model):
    __tablename__ = 'variant_fa_ann'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    variant_annotation_id = Column(String, ForeignKey('public.variant_annotation.variant_annotation_id'), primary_key=True)
    significance = Column(String, nullable=True)
    notes = Column(Text, nullable=True)
    sentence = Column(Text)
    alleles = Column(String)
    specialty_population = Column(String, nullable=True)
    assay_type = Column(String, nullable=True)
    metabolizer_types = Column(String, nullable=True)
    is_plural = Column(String, nullable=True)
    is_associated = Column(String)
    direction_of_effect = Column(String, nullable=True)
    functional_terms = Column(String, nullable=True)
    gene_product = Column(String, nullable=True)
    when_treated_with = Column(String, nullable=True)
    multiple_drugs = Column(String, nullable=True)
    cell_type = Column(String, nullable=True)
    comparison_alleles = Column(String, nullable=True)
    comparison_metabolizer_types = Column(String, nullable=True)
    variant_annotation = relationship('VariantAnnotation', back_populates='fa_annotations')

class VariantDrugAnn(db.Model):
    __tablename__ = 'variant_drug_ann'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    variant_annotation_id = Column(String, ForeignKey('public.variant_annotation.variant_annotation_id'), primary_key=True)
    significance = Column(String, nullable=True)
    notes = Column(Text, nullable=True)
    sentence = Column(Text)
    alleles = Column(String)
    specialty_population = Column(String, nullable=True)
    metabolizer_types = Column(String, nullable=True)
    is_plural = Column(String, nullable=True)
    is_associated = Column(String)
    direction_of_effect = Column(String, nullable=True)
    pd_pk_terms = Column(String, nullable=True)
    multiple_drugs = Column(String, nullable=True)
    population_types = Column(String, nullable=True)
    population_phenotypes_diseases = Column(String, nullable=True)
    multiple_phenotypes_diseases = Column(String, nullable=True)
    comparison_alleles = Column(String, nullable=True)
    comparison_metabolizer_types = Column(String, nullable=True)
    variant_annotation = relationship('VariantAnnotation', back_populates='drug_annotations')

class VariantPhenoAnn(db.Model):
    __tablename__ = 'variant_pheno_ann'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    variant_annotation_id = Column(String, ForeignKey('public.variant_annotation.variant_annotation_id'), primary_key=True)
    significance = Column(String, nullable=True)
    notes = Column(Text, nullable=True)
    sentence = Column(Text)
    alleles = Column(String)
    specialty_population = Column(String, nullable=True)
    metabolizer_types = Column(String, nullable=True)
    is_plural = Column(String, nullable=True)
    is_associated = Column(String)
    direction_of_effect = Column(String, nullable=True)
    side_effect_efficacy_other = Column(String, nullable=True)
    phenotype = Column(String, nullable=True)
    multiple_phenotypes = Column(String, nullable=True)
    when_treated_with = Column(String, nullable=True)
    multiple_drugs = Column(String, nullable=True)
    population_types = Column(String, nullable=True)
    population_phenotypes_diseases = Column(String, nullable=True)
    multiple_phenotypes_diseases = Column(String, nullable=True)
    comparison_alleles = Column(String, nullable=True)
    comparison_metabolizer_types = Column(String, nullable=True)
    variant_annotation = relationship('VariantAnnotation', back_populates='pheno_annotations')

# Junction Tables for VariantAnnotation
class VariantAnnotationDrug(db.Model):
    __tablename__ = 'variant_annotation_drug'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    variant_annotation_id = Column(String, ForeignKey('public.variant_annotation.variant_annotation_id'), primary_key=True)
    drug_id = Column(Integer, ForeignKey('public.drug.id'), primary_key=True)
    variant_annotation = relationship('VariantAnnotation', back_populates='drugs')
    drug = relationship('Drug', back_populates='variant_annotations')

class VariantAnnotationGene(db.Model):
    __tablename__ = 'variant_annotation_gene'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    variant_annotation_id = Column(String, ForeignKey('public.variant_annotation.variant_annotation_id'), primary_key=True)
    gene_id = Column(String, ForeignKey('public.gene.gene_id'), primary_key=True)
    variant_annotation = relationship('VariantAnnotation', back_populates='genes')
    gene = relationship('Gene', back_populates='variant_annotations')

class VariantAnnotationVariant(db.Model):
    __tablename__ = 'variant_annotation_variant'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    variant_annotation_id = Column(String, ForeignKey('public.variant_annotation.variant_annotation_id'), primary_key=True)
    variant_id = Column(Integer, ForeignKey('public.variant.id'), primary_key=True)
    variant_annotation = relationship('VariantAnnotation', back_populates='variants')
    variant = relationship('Variant', back_populates='variant_annotations')

# Relationships
class Relationship(db.Model):
    __tablename__ = 'relationships'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    id = Column(Integer, primary_key=True)
    entity1_id = Column(String, index=True)
    entity1_name = Column(String)
    entity1_type = Column(String)
    entity2_id = Column(String, index=True)
    entity2_name = Column(String)
    entity2_type = Column(String)
    evidence = Column(String)
    association = Column(String)
    pk = Column(String, nullable=True)
    pd = Column(String, nullable=True)
    pmids = Column(String, nullable=True)

# Drug Labels
class DrugLabel(db.Model):
    __tablename__ = 'drug_label'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    pharmgkb_id = Column(String, primary_key=True)
    name = Column(String)
    source = Column(String)
    biomarker_flag = Column(String, nullable=True)
    testing_level = Column(String, nullable=True)
    has_prescribing_info = Column(String, nullable=True)
    has_dosing_info = Column(String, nullable=True)
    has_alternate_drug = Column(String, nullable=True)
    has_other_prescribing_guidance = Column(String, nullable=True)
    cancer_genome = Column(String, nullable=True)
    prescribing = Column(String, nullable=True)
    latest_history_date = Column(Date)
    drugs = relationship('DrugLabelDrug', back_populates='drug_label')
    genes = relationship('DrugLabelGene', back_populates='drug_label')
    variants = relationship('DrugLabelVariant', back_populates='drug_label')

class DrugLabelDrug(db.Model):
    __tablename__ = 'drug_label_drug'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    pharmgkb_id = Column(String, ForeignKey('public.drug_label.pharmgkb_id'), primary_key=True)
    drug_id = Column(Integer, ForeignKey('public.drug.id'), primary_key=True)
    drug_label = relationship('DrugLabel', back_populates='drugs')
    drug = relationship('Drug', back_populates='drug_labels')

class DrugLabelGene(db.Model):
    __tablename__ = 'drug_label_gene'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    pharmgkb_id = Column(String, ForeignKey('public.drug_label.pharmgkb_id'), primary_key=True)
    gene_id = Column(String, ForeignKey('public.gene.gene_id'), primary_key=True)
    drug_label = relationship('DrugLabel', back_populates='genes')
    gene = relationship('Gene', back_populates='drug_labels')

class DrugLabelVariant(db.Model):
    __tablename__ = 'drug_label_variant'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    pharmgkb_id = Column(String, ForeignKey('public.drug_label.pharmgkb_id'), primary_key=True)
    variant_id = Column(Integer, ForeignKey('public.variant.id'), primary_key=True)
    drug_label = relationship('DrugLabel', back_populates='variants')
    variant = relationship('Variant', back_populates='drug_labels')

# Clinical Variants
class ClinicalVariant(db.Model):
    __tablename__ = 'clinical_variants'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    id = db.Column(db.Integer, primary_key=True)
    variant_type = db.Column(db.String)
    level_of_evidence = db.Column(db.String)
    gene_id = db.Column(db.String, db.ForeignKey('public.gene.gene_id'), index=True)
    gene = db.relationship('Gene', back_populates='clinical_variants')
    drugs = db.relationship('ClinicalVariantDrug', back_populates='clinical_variant')
    phenotypes = db.relationship('ClinicalVariantPhenotype', back_populates='clinical_variant')
    variants = db.relationship('ClinicalVariantVariant', back_populates='clinical_variant')

class ClinicalVariantDrug(db.Model):
    __tablename__ = 'clinical_variant_drug'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    clinical_variant_id = Column(Integer, ForeignKey('public.clinical_variants.id'), primary_key=True)
    drug_id = Column(Integer, ForeignKey('public.drug.id'), primary_key=True)
    clinical_variant = relationship('ClinicalVariant', back_populates='drugs')
    drug = relationship('Drug', back_populates='clinical_variants')

class ClinicalVariantPhenotype(db.Model):
    __tablename__ = 'clinical_variant_phenotype'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    clinical_variant_id = Column(Integer, ForeignKey('public.clinical_variants.id'), primary_key=True)
    phenotype_id = Column(Integer, ForeignKey('public.phenotype.id'), primary_key=True)
    clinical_variant = relationship('ClinicalVariant', back_populates='phenotypes')
    phenotype = relationship('Phenotype', back_populates='clinical_variants')

class ClinicalVariantVariant(db.Model):
    __tablename__ = 'clinical_variant_variant'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    clinical_variant_id = db.Column(db.Integer, db.ForeignKey('public.clinical_variants.id'), primary_key=True)
    variant_id = db.Column(db.Integer, db.ForeignKey('public.variant.id'), primary_key=True)
    clinical_variant = db.relationship('ClinicalVariant', back_populates='variants')
    variant = db.relationship('Variant', back_populates='clinical_variants')

# Occurrences
class Occurrence(db.Model):
    __tablename__ = 'occurrences'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    id = Column(Integer, primary_key=True)
    source_type = Column(String)
    source_id = Column(String, index=True)
    source_name = Column(Text)
    object_type = Column(String)
    object_id = Column(String, index=True)
    object_name = Column(String)

# Automated Annotations
class AutomatedAnnotation(db.Model):
    __tablename__ = 'automated_annotations'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    id = Column(Integer, primary_key=True)
    chemical_id = Column(String, index=True, nullable=True)
    chemical_name = Column(String, nullable=True)
    chemical_in_text = Column(String, nullable=True)
    variation_id = Column(String, index=True, nullable=True)
    variation_name = Column(String, nullable=True)
    variation_type = Column(String, nullable=True)
    variation_in_text = Column(String, nullable=True)
    gene_ids = Column(Text, nullable=True)
    gene_symbols = Column(Text, index=True, nullable=True)
    gene_in_text = Column(Text, nullable=True)
    literature_id = Column(String, nullable=True)
    literature_title = Column(Text)
    publication_year = Column(String, nullable=True)
    journal = Column(Text, nullable=True)
    sentence = Column(Text)
    source = Column(String)
    pmid = Column(String, ForeignKey('public.publication.pmid'), index=True, nullable=True)
    publication = relationship('Publication', back_populates='automated_annotations')
# PharmGKB için database sonu.......





class News(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    title = db.Column(db.String(200), nullable=False)  # Title of the news
    description = db.Column(db.Text, nullable=False)   # Detailed description
    category = db.Column(db.String(50), nullable=False)  # "Announcement" or "Update"
    publication_date = db.Column(db.Date, nullable=False, default=datetime.utcnow)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, onupdate=datetime.utcnow)


bcrypt = Bcrypt()
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(128), nullable=False)
    name = db.Column(db.String(50), nullable=False)
    surname = db.Column(db.String(50), nullable=False)
    date_of_birth = db.Column(db.Date, nullable=False)
    occupation = db.Column(db.String(100), nullable=True)  # Optional field
    is_admin = db.Column(db.Boolean, default=False)

    def set_password(self, password):
        self.password = bcrypt.generate_password_hash(password).decode('utf-8')

    def check_password(self, password):
        return bcrypt.check_password_hash(self.password, password)

class Occupation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    name = db.Column(db.String(100), unique=True, nullable=False)


with app.app_context():
    #print("Registered Models:")
    for subclass in db.Model.__subclasses__():
        print(f"- {subclass.__name__}")

print("Current Working Directory:", os.getcwd())
print("Database URI:", app.config['SQLALCHEMY_DATABASE_URI'])




#with app.app_context():
#    print("Creating tables...")
#    db.create_all()
#    print("Tables created successfully.")


#with app.app_context():
#    inspector = inspect(db.engine)
#    tables = inspector.get_table_names()
#    print("Existing Tables:", tables)


# Dekoratörler
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash("Please log in to access this page.", "warning")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash("Giriş yap!", "warning")
            return redirect(url_for('login'))
        user = User.query.get(session['user_id'])
        if not user or not user.is_admin:
            flash("Admin değilsin!", "danger")
            return redirect(url_for('home'))
        return f(*args, **kwargs)
    return decorated_function

#Anasayfa
@app.route('/')
def home():
    try:
        announcements = News.query.filter(News.category.ilike('Announcement')).order_by(News.publication_date.desc()).all()
        updates = News.query.filter(News.category.ilike('Update')).order_by(News.publication_date.desc()).all()
    except Exception as e:
        print(f"Oops, toy box problem: {e}")
        announcements = []
        updates = []
    user = None
    user_email = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        if user:
            user_email = user.email
    return render_template(
        'home.html',
        announcements=announcements,
        updates=updates,
        user_email=user_email,
        user=user
    )

# Backend rotası
@app.route('/backend')
@admin_required
def backend_index():
    query = request.args.get('q', '').strip()
    page = request.args.get('page', 1, type=int)
    per_page = 20
    if query:
        drugs = Drug.query.filter(
            db.or_(
                Drug.name_en.ilike(f'%{query}%'),
                Drug.name_tr.ilike(f'%{query}%'),
                Drug.alternative_names.ilike(f'%{query}%')
            )
        ).paginate(page=page, per_page=per_page)
    else:
        drugs = Drug.query.paginate(page=page, per_page=per_page)
    salts = Salt.query.all()
    details = DrugDetail.query.all()
    return render_template('index.html', drugs=drugs, salts=salts, details=details, query=query)

# Updated Contact route
@app.route('/contact', methods=['GET', 'POST'])
def contact():
    user = None
    user_email = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        if user:
            user_email = user.email
    return render_template('contact.html', user_email=user_email, user=user)

# Updated About route
@app.route('/about')
def about():
    user = None
    user_email = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        if user:
            user_email = user.email
    return render_template('about.html', user_email=user_email, user=user)

# Updated Terms route
@app.route('/terms')
def terms():
    user = None
    user_email = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        if user:
            user_email = user.email
    return render_template('terms.html', user_email=user_email, user=user)

@app.route('/register', methods=['GET', 'POST'])
def register():
    occupations = Occupation.query.order_by(Occupation.name).all()  # Fetch occupations

    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        name = request.form['name']
        surname = request.form['surname']
        date_of_birth = request.form['date_of_birth']
        occupation = request.form['occupation']

        # Check if passwords match
        if password != confirm_password:
            flash("Passwords do not match.", "danger")
            return redirect(url_for('register'))

        # Check if email already exists
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash("Email already registered.", "danger")
            return redirect(url_for('register'))

        # Create new user
        new_user = User(
            email=email,
            name=name,
            surname=surname,
            date_of_birth=datetime.strptime(date_of_birth, '%Y-%m-%d'),
            occupation=occupation
        )
        new_user.set_password(password)  # Hash the password
        db.session.add(new_user)
        db.session.commit()

        flash("Registration successful. Please log in.", "success")
        return redirect(url_for('login'))

    return render_template('register.html', occupations=occupations)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Find user by email
        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            session['user_id'] = user.id
            flash("Welcome back!", "success")
            return redirect(url_for('home'))
        else:
            flash("Invalid email or password.", "danger")

    return render_template('login.html')


@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    user = User.query.get(session['user_id'])
    occupations = Occupation.query.order_by(Occupation.name).all()  # Fetch occupations dynamically

    if request.method == 'POST':
        # Update user information
        user.name = request.form['name']
        user.surname = request.form['surname']
        user.date_of_birth = datetime.strptime(request.form['date_of_birth'], '%Y-%m-%d')
        user.occupation = request.form['occupation']
        db.session.commit()

        flash("Profile updated successfully!", "success")
        return redirect(url_for('profile'))

    return render_template('profile.html', user=user, occupations=occupations)



@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash("You have been logged out.", "info")
    return redirect(url_for('home'))



# Yeni ilaç ekleme
@app.route('/add', methods=['GET', 'POST'])
def add_drug():
    if request.method == 'POST':
        name_tr = request.form['name_tr']
        name_en = request.form['name_en']
        alternative_names = request.form.getlist('alternative_names[]')  # Liste olarak alınır

        # Yeni ilaç ekleme
        new_drug = Drug(
            name_tr=name_tr,
            name_en=name_en,
            alternative_names="|".join(alternative_names)  # Alternatif isimleri " | " ile birleştir
        )
        db.session.add(new_drug)
        db.session.commit()
        return redirect(url_for('index'))
    return render_template('add_drug.html')




# Tuzlar için listeleme ve ekleme
@app.route('/salts', methods=['GET', 'POST'])
def manage_salts():
    if request.method == 'POST':
        name_tr = request.form['name_tr']
        name_en = request.form['name_en']
        new_salt = Salt(name_tr=name_tr, name_en=name_en)
        db.session.add(new_salt)
        db.session.commit()
        return redirect(url_for('manage_salts'))
    salts = Salt.query.all()
    return render_template('salts.html', salts=salts)

#Etken Madde ve Tuz eşleşmesi
@app.route('/matches', methods=['GET', 'POST'])
def manage_matches():
    drugs = Drug.query.all()  # Etken maddeleri al
    salts = Salt.query.all()  # Tuzları al
    if request.method == 'POST':
        drug_id = request.form['drug_id']
        salt_id = request.form['salt_id']
        drug = Drug.query.get(drug_id)
        salt = Salt.query.get(salt_id)
        drug.salts.append(salt)  # İlişkiyi ekle
        db.session.commit()
        return redirect(url_for('manage_matches'))
    return render_template('matches.html', drugs=drugs, salts=salts)

def generate_3d_structure(smiles, output_filename):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, "Invalid SMILES string"
        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
            return None, "Failed to generate 3D coordinates"
        AllChem.MMFFOptimizeMolecule(mol)
        output_path = os.path.join('static', 'uploads', '3d_structures', output_filename).replace('\\', '/')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        Chem.MolToPDBFile(mol, output_path)
        return output_path, None
    except Exception as e:
        return None, f"Error generating 3D structure: {str(e)}"
    

@app.route('/details/add', methods=['GET', 'POST'])
def add_detail():
    drugs = Drug.query.all()
    salts = Salt.query.all()
    indications = Indication.query.all()
    targets = Target.query.all()
    routes = RouteOfAdministration.query.all()
    side_effects = SideEffect.query.all()
    metabolites = Metabolite.query.all()
    metabolism_organs = MetabolismOrgan.query.all()
    metabolism_enzymes = MetabolismEnzyme.query.all()

    if request.method == 'POST':
        print("DEBUG: Entering POST block")
        print("DEBUG: Form data:", request.form)

        drug_id = request.form.get('drug_id')
        salt_id = request.form.get('salt_id', None)
        mechanism_of_action = request.form.get('mechanism_of_action')
        smiles = request.form.get('smiles')
        print(f"DEBUG: SMILES received: {smiles}")

        if not drug_id:
            return render_template(
                'add_detail.html', drugs=drugs, salts=salts, indications=indications,
                targets=targets, routes=routes, side_effects=side_effects, metabolites=metabolites,
                metabolism_organs=metabolism_organs, metabolism_enzymes=metabolism_enzymes,
                error_message="Drug ID is required!"
            )

        existing_detail = DrugDetail.query.filter_by(drug_id=drug_id, salt_id=salt_id).first()
        if existing_detail:
            return render_template(
                'add_detail.html', drugs=drugs, salts=salts, indications=indications,
                targets=targets, routes=routes, side_effects=side_effects, metabolites=metabolites,
                metabolism_organs=metabolism_organs, metabolism_enzymes=metabolism_enzymes,
                error_message="Bu etken madde ve tuz kombinasyonu için zaten detaylı bilgi mevcut!"
            )

        try:
            selected_routes = request.form.getlist('route_id[]')
            selected_side_effects = request.form.getlist('side_effects[]')
            fda_approved = 'fda_approved' in request.form
            ema_approved = 'ema_approved' in request.form
            titck_approved = 'titck_approved' in request.form
            molecular_formula = request.form.get('molecular_formula')
            synthesis = request.form.get('synthesis')
            structure = request.files.get('structure')
            structure_filename = None
            if structure:
                structure_filename = secure_filename(structure.filename)
                structure.save(os.path.join('static', 'Uploads', structure_filename))
                structure_filename = os.path.join('Uploads', structure_filename).replace('\\', '/')
                print(f"DEBUG: Structure file saved as {structure_filename}")
            structure_3d_filename = None
            if smiles:
                output_filename = f"drug_{drug_id}_salt_{salt_id or 'none'}_3d.pdb"
                structure_3d_path, error = generate_3d_structure(smiles, output_filename)
                if error:
                    print(f"DEBUG: 3D structure generation failed: {error}")
                else:
                    structure_3d_filename = os.path.join('Uploads', '3d_structures', output_filename).replace('\\', '/')
                    print(f"DEBUG: 3D structure saved as {structure_3d_filename}")
            else:
                print("DEBUG: No SMILES provided, skipping 3D structure generation")
            iupac_name = request.form.get('iupac_name')
            inchikey = request.form.get('inchikey')
            pubchem_cid = request.form.get('pubchem_cid')
            pubchem_sid = request.form.get('pubchem_sid')
            cas_id = request.form.get('cas_id')
            ec_number = request.form.get('ec_number')
            nci_code = request.form.get('nci_code')
            rxcui = request.form.get('rxcui')
            snomed_id = request.form.get('snomed_id')
            molecular_weight = request.form.get('molecular_weight', type=float)
            solubility = request.form.get('solubility')
            pharmacodynamics = request.form.get('pharmacodynamics')
            black_box_warning = 'black_box_warning' in request.form
            black_box_details = request.form.get('black_box_details') if black_box_warning else None
            indications = ','.join(request.form.getlist('indications[]')) if request.form.getlist('indications[]') else None
            selected_target_molecules = request.form.getlist('target_molecules[]')
            target_molecules = ','.join(selected_target_molecules) if selected_target_molecules else None
            pharmacokinetics = request.form.get('pharmacokinetics')
            boiling_point = request.form.get('boiling_point')
            melting_point = request.form.get('melting_point')
            density = request.form.get('density')
            flash_point = request.form.get('flash_point')

            # Create new DrugDetail
            new_detail = DrugDetail(
                drug_id=drug_id, salt_id=salt_id, mechanism_of_action=mechanism_of_action,
                molecular_formula=molecular_formula, synthesis=synthesis, structure=structure_filename,
                structure_3d=structure_3d_filename, iupac_name=iupac_name, smiles=smiles, inchikey=inchikey, pubchem_cid=pubchem_cid,
                pubchem_sid=pubchem_sid, cas_id=cas_id, ec_number=ec_number, nci_code=nci_code,
                rxcui=rxcui, snomed_id=snomed_id, molecular_weight=molecular_weight, solubility=solubility,
                pharmacodynamics=pharmacodynamics, black_box_warning=black_box_warning,
                black_box_details=black_box_details, indications=indications, target_molecules=target_molecules,
                pharmacokinetics=pharmacokinetics, boiling_point=boiling_point, melting_point=melting_point,
                density=density, flash_point=flash_point, fda_approved=fda_approved,
                ema_approved=ema_approved, titck_approved=titck_approved
            )
            db.session.add(new_detail)
            db.session.commit()
            #print(f"DEBUG: DrugDetail added with ID: {new_detail.id}")

            # Parse range helper
            def parse_range(value):
                if not value:
                    return None, None
                if '-' in value:
                    try:
                        min_val, max_val = map(float, value.split('-'))
                        return min_val, max_val
                    except ValueError:
                        print(f"DEBUG: Failed to parse range '{value}'")
                        return None, None
                try:
                    val = float(value)
                    return val, val
                except ValueError:
                    print(f"DEBUG: Failed to parse single value '{value}'")
                    return None, None

            # Process DrugRoute entries
            if not selected_routes:
                print("WARNING: No routes selected")
            for route_id in selected_routes:
                #print(f"DEBUG: Processing route_id={route_id}")
                pd = request.form.get(f'route_pharmacodynamics_{route_id}', '')  # Fixed: Removed $
                pk = request.form.get(f'route_pharmacokinetics_{route_id}', '')  # Fixed: Removed $

                # PK parameters - Fixed field names
                absorption_rate = request.form.get(f'absorption_rate_{route_id}', '')
                vod_rate = request.form.get(f'vod_rate_{route_id}', '')
                protein_binding = request.form.get(f'protein_binding_{route_id}', '')
                half_life = request.form.get(f'half_life_{route_id}', '')
                clearance_rate = request.form.get(f'clearance_rate_{route_id}', '')
                bioavailability = request.form.get(f'bioavailability_{route_id}', '')
                tmax = request.form.get(f'tmax_{route_id}', '')
                cmax = request.form.get(f'cmax_{route_id}', '')

                # Metabolism parameters - Fixed field names
                metabolism_organs_ids = request.form.getlist(f'metabolism_organs_{route_id}[]')  # Fixed: Removed $
                metabolism_enzymes_ids = request.form.getlist(f'metabolism_enzymes_{route_id}[]')  # Fixed: Removed $
                metabolite_ids = request.form.getlist(f'metabolites_{route_id}[]')  # Fixed: Removed $

                # Parse PK ranges
                absorption_min, absorption_max = parse_range(absorption_rate)
                vod_min, vod_max = parse_range(vod_rate)
                protein_min, protein_max = parse_range(protein_binding)
                if protein_min is not None and protein_max is not None:
                    protein_min /= 100
                    protein_max /= 100
                half_life_min, half_life_max = parse_range(half_life)
                clearance_min, clearance_max = parse_range(clearance_rate)
                bio_min, bio_max = parse_range(bioavailability)
                if bio_min is not None and bio_max is not None:
                    bio_min /= 100
                    bio_max /= 100
                tmax_min, tmax_max = parse_range(tmax)
                cmax_min, cmax_max = parse_range(cmax)

                print(f"DEBUG: Route ID {route_id} -> "
                      f"Absorption: {absorption_min}-{absorption_max}, "
                      f"VoD: {vod_min}-{vod_max}, "
                      f"Protein Binding: {protein_min}-{protein_max}, "
                      f"Half-Life: {half_life_min}-{half_life_max}, "
                      f"Clearance: {clearance_min}-{clearance_max}, "
                      f"Bioavailability: {bio_min}-{bio_max}, "
                      f"Tmax: {tmax_min}-{tmax_max}, "
                      f"Cmax: {cmax_min}-{cmax_max}, "
                      f"Organs IDs: {metabolism_organs_ids}, "
                      f"Enzymes IDs: {metabolism_enzymes_ids}, "
                      f"Metabolite IDs: {metabolite_ids}, "
                      f"PD: {pd}, PK: {pk}")

                # Validate route_id
                if not RouteOfAdministration.query.get(route_id):
                    print(f"ERROR: Invalid route_id {route_id}")
                    continue

                # Create DrugRoute entry
                new_drug_route = DrugRoute(
                    drug_detail_id=new_detail.id,
                    route_id=route_id,
                    pharmacodynamics=pd,
                    pharmacokinetics=pk,
                    absorption_rate_min=absorption_min,
                    absorption_rate_max=absorption_max,
                    vod_rate_min=vod_min,
                    vod_rate_max=vod_max,
                    protein_binding_min=protein_min,
                    protein_binding_max=protein_max,
                    half_life_min=half_life_min,
                    half_life_max=half_life_max,
                    clearance_rate_min=clearance_min,
                    clearance_rate_max=clearance_max,
                    bioavailability_min=bio_min,
                    bioavailability_max=bio_max,
                    tmax_min=tmax_min,
                    tmax_max=tmax_max,
                    cmax_min=cmax_min,
                    cmax_max=cmax_max
                )
                #print("DEBUG: Adding DrugRoute to session")
                db.session.add(new_drug_route)
                #print("DEBUG: Flushing session for DrugRoute")
                db.session.flush()  # Get ID for relationships
                #print(f"DEBUG: DrugRoute added with ID: {new_drug_route.id}")

                # Link metabolism data via relationships
                if metabolism_organs_ids:
                    organs = MetabolismOrgan.query.filter(MetabolismOrgan.id.in_([int(id) for id in metabolism_organs_ids])).all()
                    new_drug_route.metabolism_organs.extend(organs)
                    #print(f"DEBUG: Linked {len(organs)} metabolism organs to DrugRoute {new_drug_route.id}")
                if metabolism_enzymes_ids:
                    enzymes = MetabolismEnzyme.query.filter(MetabolismEnzyme.id.in_([int(id) for id in metabolism_enzymes_ids])).all()
                    new_drug_route.metabolism_enzymes.extend(enzymes)
                    #print(f"DEBUG: Linked {len(enzymes)} metabolism enzymes to DrugRoute {new_drug_route.id}")
                if metabolite_ids:
                    metabolites = Metabolite.query.filter(Metabolite.id.in_([int(id) for id in metabolite_ids])).all()
                    new_drug_route.metabolites.extend(metabolites)
                    #print(f"DEBUG: Linked {len(metabolites)} metabolites to DrugRoute {new_drug_route.id}")

                # Route-specific indications
                selected_route_indications = request.form.getlist(f'route_indications_{route_id}[]')  # Fixed: Removed $
                for indication_id in selected_route_indications:
                    new_route_indication = RouteIndication(
                        drug_detail_id=new_detail.id,
                        route_id=route_id,
                        indication_id=indication_id
                    )
                    #print("DEBUG: Adding RouteIndication to session")
                    db.session.add(new_route_indication)
                    #print(f"DEBUG: Added RouteIndication for indication_id={indication_id}")

            # Add side effects
            for side_effect_id in selected_side_effects:
                side_effect = SideEffect.query.get(side_effect_id)
                if side_effect:
                    new_detail.side_effects.append(side_effect)
                    #print(f"DEBUG: Added side_effect_id={side_effect_id} to DrugDetail {new_detail.id}")

            #print("DEBUG: Committing all changes")
            db.session.commit()
            #print("DEBUG: All records saved successfully")
            return redirect(url_for('view_details'))

        except Exception as e:
            db.session.rollback()
            #print(f"ERROR: Exception occurred: {str(e)}")
            return render_template(
                'add_detail.html', drugs=drugs, salts=salts, indications=indications,
                targets=targets, routes=routes, side_effects=side_effects, metabolites=metabolites,
                metabolism_organs=metabolism_organs, metabolism_enzymes=metabolism_enzymes,
                error_message=f"Error saving details: {str(e)}"
            )

    return render_template(
        'add_detail.html', drugs=drugs, salts=salts, indications=indications,
        targets=targets, routes=routes, side_effects=side_effects, metabolites=metabolites,
        metabolism_organs=metabolism_organs, metabolism_enzymes=metabolism_enzymes
    )



@app.route('/details', methods=['GET'])
@login_required
def view_details():
    details = DrugDetail.query.all()

    enriched_details = []
    for detail in details:
        # Prepare routes with route-specific indications
        routes_info = []
        for route in detail.routes:
            route_indications = [
                f"{ri.indication.name_en} ({ri.indication.name_tr})" if ri.indication.name_tr else ri.indication.name_en
                for ri in detail.route_indications
                if ri.route_id == route.route_id
            ]
            routes_info.append({
                'name': route.route.name,
                'pharmacodynamics': route.pharmacodynamics,
                'pharmacokinetics': route.pharmacokinetics,
                'indications': route_indications
            })


        # Indications isimlerini çözümle
        indications_list = []
        if detail.indications:
            #print(f"DEBUG: Indications (raw) for detail {detail.id} -> {detail.indications}")
            indication_ids = [int(ind_id) for ind_id in detail.indications.split(',') if ind_id.isdigit()]
            indications = Indication.query.filter(Indication.id.in_(indication_ids)).all()
            indications_list = [f"{ind.name_en} ({ind.name_tr})" if ind.name_tr else ind.name_en for ind in indications]
            #print(f"DEBUG: Retrieved Indications for detail {detail.id} -> {indications_list}")



        # Target Molecules isimlerini çözümle
        if detail.target_molecules:
            target_ids = detail.target_molecules.split(',')  # ID'leri ayır
            target_molecules_list = [
                db.session.get(Target, int(target_id)).name_en
                for target_id in target_ids
                if db.session.get(Target, int(target_id))
            ]
        else:
            target_molecules_list = []


        # Yan etkileri çözümle
        side_effects_list = [
            {"id": se.id, "name_en": se.name_en, "name_tr": se.name_tr or 'N/A'}
            for se in detail.side_effects
        ]

        # Detayı zenginleştir ve listeye ekle
        enriched_details.append({
            'id': detail.id,
            'drug_name': detail.drug.name_en,
            'salt_name': detail.salt.name_en if detail.salt else None,
            'fda_approved': detail.fda_approved,
            'ema_approved': detail.ema_approved,
            'titck_approved': detail.titck_approved,
            'molecular_formula': detail.molecular_formula,
            'synthesis': detail.synthesis,
            'structure': detail.structure,
            'structure_3d': detail.structure_3d,
            'iupac_name': detail.iupac_name,
            'smiles': detail.smiles,
            'inchikey': detail.inchikey,
            'pubchem_cid': detail.pubchem_cid,
            'pubchem_sid': detail.pubchem_sid,
            'cas_id': detail.cas_id,
            'ec_number': detail.ec_number,
            'nci_code': detail.nci_code,
            'rxcui': detail.rxcui,
            'snomed_id': detail.snomed_id,
            'molecular_weight': detail.molecular_weight,
            'solubility': detail.solubility,
            'side_effects': side_effects_list,  # Yan etkiler listesi
            'pharmacodynamics': detail.pharmacodynamics,
            'indications': indications_list,  # İsimleri şablona gönder
            'target_molecules': target_molecules_list,  # İsimleri şablona gönder
            'pharmacokinetics': detail.pharmacokinetics,
            'boiling_point': detail.boiling_point,
            'melting_point': detail.melting_point,
            'density': detail.density,
            'flash_point': detail.flash_point,
            'routes': routes_info,
            'black_box_warning': detail.black_box_warning,
            'black_box_details': detail.black_box_details
        })

    # Detayları şablona gönder
    return render_template('details_list.html', details=enriched_details)

#ICD-11 Başlangıç
# Helper Functions
def clean_title(title):
    return title.strip().lstrip('- ').strip()

def find_column(df, possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    return None

def import_icd11_mms_from_file(file_path):
    file_ext = os.path.splitext(file_path)[1].lower()
    chunk_size = 500   # Process 500 rows at a time to minimize memory
    batch_size = 250   # Commit categories every 250 records to reduce transaction size

    # Initialize reader for chunked processing
    if file_ext == '.xlsx':
        # Use openpyxl to read Excel iteratively
        wb = load_workbook(file_path, read_only=True, data_only=True)
        ws = wb.active
        rows = list(ws.iter_rows(values_only=True))
        header = rows[0]  # Assume first row is header
        data_rows = rows[1:]  # Data starts from second row
        total_rows_excel = len(data_rows)
        # Convert chunks to DataFrames
        chunks = [pd.DataFrame(data_rows[i:i+chunk_size], columns=header) for i in range(0, total_rows_excel, chunk_size)]
        wb.close()  # Free memory
    elif file_ext in ['.txt', '.tsv']:
        reader = pd.read_csv(file_path, sep='\t', dtype=str, encoding='utf-8', chunksize=chunk_size)
        chunks = reader
    else:
        raise ValueError(f"Unsupported file type: {file_ext}")

    # Column mapping (unchanged)
    required_columns = {
        'Foundation URI': ['Foundation URI', 'FoundationURI', 'Foundation_URI'],
        'Linearization URI': ['Linearization URI', 'LinearizationURI', 'Linearization_URI'],
        'Code': ['Code'],
        'Title': ['Title'],
        'ClassKind': ['ClassKind', 'Class Kind'],
        'DepthInKind': ['DepthInKind', 'Depth In Kind'],
        'BlockId': ['BlockId'],
        'ChapterNo': ['ChapterNo', 'Chapter No'],
        'Grouping1': ['Grouping1'],
        'Grouping2': ['Grouping2'],
        'Grouping3': ['Grouping3'],
        'Grouping4': ['Grouping4'],
        'Grouping5': ['Grouping5'],
        'IsResidual': ['IsResidual'],
        'isLeaf': ['isLeaf'],
    }

    # Initialize tracking variables
    uri_to_id = {}
    code_to_id = {}
    block_to_id = {}
    chapter_to_id = {}
    added_count = {'chapter': 0, 'block': 0, 'category': 0, 'other': 0}
    skipped_rows = []
    total_rows = 0
    parent_id_cache = {}  # Cache parent IDs to reduce database queries

    # Process each chunk
    for chunk_idx, df in enumerate(chunks):
        df = df.fillna('')
        total_rows += len(df)
        print(f"Processing chunk {chunk_idx + 1} (rows {chunk_idx * chunk_size + 1} to {chunk_idx * chunk_size + len(df)})")

        # Column validation (unchanged)
        found_columns = {}
        missing_columns = []
        for col, variations in required_columns.items():
            found_col = find_column(df, variations)
            if found_col:
                found_columns[col] = found_col
            else:
                missing_columns.append(col)

        if missing_columns:
            if 'Linearization URI' in missing_columns and 'Foundation URI' not in missing_columns:
                missing_columns.remove('Linearization URI')
            if missing_columns:
                raise ValueError(f"Missing columns: {missing_columns}")

        # Separate entries into chapters, blocks, and categories (unchanged)
        chapters = []
        blocks = []
        categories = []
        for index, row in df.iterrows():
            class_kind = row[found_columns['ClassKind']] or 'unknown'
            name_en = clean_title(row[found_columns['Title']])
            code = row[found_columns['Code']] or None
            safe_name_en = name_en.encode('ascii', 'replace').decode('ascii')
            print(f"Row {index}: class_kind={class_kind}, name_en={safe_name_en}, code={code}")
            if class_kind == 'chapter':
                chapters.append((index, row))
            elif class_kind == 'block':
                blocks.append((index, row))
            elif class_kind == 'category':
                categories.append((index, row))
            else:
                skipped_rows.append(f"Row {index}: Unknown class_kind '{class_kind}'")

        print(f"Chunk {chunk_idx + 1}: {len(chapters)} chapters, {len(blocks)} blocks, {len(categories)} categories")

        # Process chapters (unchanged)
        for index, row in chapters:
            foundation_uri = row[found_columns['Foundation URI']] or None
            disease_id_base = row.get(found_columns.get('Linearization URI'), foundation_uri) or None
            code = row[found_columns['Code']] or None
            disease_id = f"{index}_{disease_id_base}" if disease_id_base else str(index)
            block_id = row.get(found_columns['BlockId'], '') or None
            name_en = clean_title(row[found_columns['Title']])
            class_kind = row[found_columns['ClassKind']] or 'unknown'
            depth = int(row[found_columns['DepthInKind']])
            is_residual = row.get(found_columns['IsResidual'], 'False') == 'True'
            is_leaf = row.get(found_columns['isLeaf'], 'False') == 'True'
            chapter_no = row.get(found_columns['ChapterNo'], '') or None

            safe_name_en = name_en.encode('ascii', 'replace').decode('ascii')
            print(f"Processing chapter: {safe_name_en} (disease_id: {disease_id}, chapter_no: {chapter_no})")

            if not name_en:
                skipped_rows.append(f"Row {index}: Missing name_en")
                continue
            if class_kind == 'chapter' and not chapter_no:
                skipped_rows.append(f"Row {index}: Missing ChapterNo for chapter")
                continue

            parent_id = None  # Chapters have no parent

            existing = Indication.query.filter_by(disease_id=disease_id).first()
            if not existing:
                new_indication = Indication(
                    foundation_uri=foundation_uri,
                    disease_id=disease_id,
                    name_en=name_en,
                    code=code,
                    class_kind=class_kind,
                    depth=depth,
                    parent_id=parent_id,
                    is_residual=is_residual,
                    is_leaf=is_leaf,
                    chapter_no=chapter_no,
                    BlockId=block_id
                )
                db.session.add(new_indication)
                db.session.flush()
                uri_to_id[disease_id] = new_indication.id
                if code:
                    code_to_id[code] = new_indication.id
                if class_kind == 'chapter' and chapter_no:
                    chapter_to_id[chapter_no] = new_indication.id
                if block_id:
                    block_to_id[block_id] = new_indication.id
                added_count['chapter'] += 1
            else:
                uri_to_id[disease_id] = existing.id
                if code:
                    code_to_id[code] = existing.id
                if class_kind == 'chapter' and chapter_no:
                    chapter_to_id[chapter_no] = existing.id
                if block_id:
                    block_to_id[block_id] = existing.id

        # Process blocks in hierarchical order (unchanged)
        block_depths = {}
        for index, row in blocks:
            depth = int(row[found_columns['DepthInKind']])
            if depth not in block_depths:
                block_depths[depth] = []
            block_depths[depth].append((index, row))

        for depth in sorted(block_depths.keys()):
            for index, row in block_depths[depth]:
                foundation_uri = row[found_columns['Foundation URI']] or None
                disease_id_base = row.get(found_columns.get('Linearization URI'), foundation_uri) or None
                code = row[found_columns['Code']] or None
                disease_id = f"{index}_{disease_id_base}" if disease_id_base else str(index)
                block_id = row.get(found_columns['BlockId'], '') or None
                name_en = clean_title(row[found_columns['Title']])
                class_kind = row[found_columns['ClassKind']] or 'unknown'
                depth = int(row[found_columns['DepthInKind']])
                is_residual = row.get(found_columns['IsResidual'], 'False') == 'True'
                is_leaf = row.get(found_columns['isLeaf'], 'False') == 'True'
                chapter_no = row.get(found_columns['ChapterNo'], '') or None

                safe_name_en = name_en.encode('ascii', 'replace').decode('ascii')
                print(f"Processing block (depth {depth}): {safe_name_en} (block_id: {block_id}, chapter_no: {chapter_no})")

                if not name_en:
                    skipped_rows.append(f"Row {index}: Missing name_en")
                    continue

                parent_id = get_parent_id(row, depth, chapter_no, code_to_id, block_to_id, chapter_to_id, found_columns)

                if parent_id is None and depth > 1:
                    skipped_rows.append(f"Row {index}: No parent found for block '{safe_name_en}' (depth {depth})")
                    continue

                existing = Indication.query.filter_by(disease_id=disease_id).first()
                if not existing:
                    new_indication = Indication(
                        foundation_uri=foundation_uri,
                        disease_id=disease_id,
                        name_en=name_en,
                        code=code,
                        class_kind=class_kind,
                        depth=depth,
                        parent_id=parent_id,
                        is_residual=is_residual,
                        is_leaf=is_leaf,
                        chapter_no=chapter_no,
                        BlockId=block_id
                    )
                    db.session.add(new_indication)
                    db.session.flush()
                    uri_to_id[disease_id] = new_indication.id
                    if code:
                        code_to_id[code] = new_indication.id
                    if class_kind == 'chapter' and chapter_no:
                        chapter_to_id[chapter_no] = new_indication.id
                    if block_id:
                        block_to_id[block_id] = new_indication.id
                    added_count['block'] += 1
                else:
                    uri_to_id[disease_id] = existing.id
                    if code:
                        code_to_id[code] = existing.id
                    if class_kind == 'chapter' and chapter_no:
                        chapter_to_id[chapter_no] = existing.id
                    if block_id:
                        block_to_id[block_id] = existing.id

        print(f"Chunk {chunk_idx + 1} block_to_id: {block_to_id}")

        # Process categories in hierarchical order (unchanged)
        category_depths = {}
        for index, row in categories:
            depth = int(row[found_columns['DepthInKind']])
            if depth not in category_depths:
                category_depths[depth] = []
            category_depths[depth].append((index, row))

        for depth in sorted(category_depths.keys()):
            sorted_categories = sorted(category_depths[depth], key=lambda x: x[1][found_columns['Code']] or '')
            for index, row in sorted_categories:
                foundation_uri = row[found_columns['Foundation URI']] or None
                disease_id_base = row.get(found_columns.get('Linearization URI'), foundation_uri) or None
                code = row[found_columns['Code']] or None
                disease_id = f"{index}_{disease_id_base}" if disease_id_base else str(index)
                block_id = row.get(found_columns['BlockId'], '') or None
                name_en = clean_title(row[found_columns['Title']])
                class_kind = row[found_columns['ClassKind']] or 'unknown'
                depth = int(row[found_columns['DepthInKind']])
                is_residual = row.get(found_columns['IsResidual'], 'False') == 'True'
                is_leaf = row.get(found_columns['isLeaf'], 'False') == 'True'
                chapter_no = row.get(found_columns['ChapterNo'], '') or None

                safe_name_en = name_en.encode('ascii', 'replace').decode('ascii')
                print(f"Processing category (depth {depth}): {safe_name_en} (code: {code}, chapter_no: {chapter_no})")

                if not name_en:
                    skipped_rows.append(f"Row {index}: Missing name_en")
                    continue

                # Use cached parent_id if available
                cache_key = f"{code}_{depth}_{chapter_no}"
                if cache_key in parent_id_cache:
                    parent_id = parent_id_cache[cache_key]
                else:
                    parent_id = get_parent_id(row, depth, chapter_no, code_to_id, block_to_id, chapter_to_id, found_columns)
                    parent_id_cache[cache_key] = parent_id

                if parent_id is None and depth > 1:
                    skipped_rows.append(f"Row {index}: No parent found for category '{safe_name_en}' (depth {depth})")
                    continue

                existing = Indication.query.filter_by(disease_id=disease_id).first()
                if not existing:
                    new_indication = Indication(
                        foundation_uri=foundation_uri,
                        disease_id=disease_id,
                        name_en=name_en,
                        code=code,
                        class_kind=class_kind,
                        depth=depth,
                        parent_id=parent_id,
                        is_residual=is_residual,
                        is_leaf=is_leaf,
                        chapter_no=chapter_no,
                        BlockId=block_id
                    )
                    db.session.add(new_indication)
                    db.session.flush()
                    uri_to_id[disease_id] = new_indication.id
                    if code:
                        code_to_id[code] = new_indication.id
                    if class_kind == 'chapter' and chapter_no:
                        chapter_to_id[chapter_no] = new_indication.id
                    if block_id:
                        block_to_id[block_id] = new_indication.id
                    added_count['category'] += 1
                    if (added_count['category'] % batch_size) == 0:
                        db.session.commit()
                else:
                    uri_to_id[disease_id] = existing.id
                    if code:
                        code_to_id[code] = existing.id
                    if class_kind == 'chapter' and chapter_no:
                        chapter_to_id[chapter_no] = existing.id
                    if block_id:
                        block_to_id[block_id] = existing.id

        db.session.commit()
        print(f"Committed chunk {chunk_idx + 1}")

    # Final commit (unchanged)
    db.session.commit()

    # Validate hierarchy (unchanged)
    total_added = sum(added_count.values())
    orphaned = Indication.query.filter(Indication.parent_id.is_(None), Indication.depth > 1, Indication.class_kind != 'chapter').count()
    if orphaned > 0:
        print(f"Warning: {orphaned} non-root entries have no parent!")

    if total_added < total_rows:
        print(f"Processed {total_added} out of {total_rows} rows. Skipped: {len(skipped_rows)}")
    print(f"Skipped rows: {skipped_rows}")
    return total_added, skipped_rows


def get_parent_id(row, depth, chapter_no, code_to_id, block_to_id, chapter_to_id, found_columns):
    code = row[found_columns['Code']] or None
    block_id = row.get(found_columns['BlockId'], '') or None
    class_kind = row[found_columns['ClassKind']]

    if class_kind == 'chapter':
        return None  # Chapters are root nodes

    # For blocks, use Grouping columns
    if class_kind == 'block':
        for i in range(1, 6):
            grouping_key = f'Grouping{i}'
            if grouping_key in found_columns and row[found_columns[grouping_key]]:
                parent_block = row[found_columns[grouping_key]].strip()
                if parent_block in block_to_id:
                    return block_to_id[parent_block]
        if chapter_no in chapter_to_id:
            return chapter_to_id[chapter_no]
        return None

    # For categories, first check code hierarchy for subcategories
    if class_kind == 'category':
        # Check code hierarchy for subcategories (e.g., KA00.0 should link to KA00)
        if code and '.' in code:
            code_parts = code.split('.')
            for i in range(len(code_parts) - 1, 0, -1):
                parent_code = '.'.join(code_parts[:i])
                if parent_code in code_to_id:
                    parent = Indication.query.get(code_to_id[parent_code])
                    if parent.class_kind == 'category':
                        return code_to_id[parent_code]

        # If no code hierarchy match (e.g., base category like KA00), use Grouping columns
        deepest_block_id = None
        for i in range(1, 6):  # Start from Grouping1 to Grouping5
            grouping_key = f'Grouping{i}'
            if grouping_key in found_columns and row[found_columns[grouping_key]]:
                parent_block = row[found_columns[grouping_key]].strip()
                if parent_block in block_to_id:
                    deepest_block_id = block_to_id[parent_block]
                    # Don't break; continue to find the deepest block

        if deepest_block_id:
            return deepest_block_id

        # Fallback to chapter if no block found
        if chapter_no in chapter_to_id:
            return chapter_to_id[chapter_no]

    return None

# Routes
@app.route('/indications', methods=['GET', 'POST'])
def manage_indications():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash("No file part in request!", "error")
            return redirect(url_for('manage_indications'))
        file = request.files['file']
        if file.filename == '':
            flash("No file selected!", "error")
            return redirect(url_for('manage_indications'))

        if '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in ['txt', 'tsv', 'xlsx']:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
            file.save(file_path)
            try:
                added_count, skipped_rows = import_icd11_mms_from_file(file_path)
                flash(f"Imported {added_count} indications successfully!", "success")
                if skipped_rows:
                    flash(f"Skipped {len(skipped_rows)} rows. Check logs for details.", "warning")
            except Exception as e:
                flash(f"Error importing ICD-11 data: {e}", "error")
                app.logger.error(f"Import error: {e}")
            finally:
                if os.path.exists(file_path):
                    os.remove(file_path)
            return redirect(url_for('manage_indications'))
        else:
            flash("Invalid file type! Please upload a .txt, .tsv, or .xlsx file.", "error")
            return redirect(url_for('manage_indications'))

    root_nodes = Indication.query.filter_by(parent_id=None).order_by(Indication.chapter_no.asc()).all()
    return render_template('indications.html', indications=root_nodes)

@app.route('/indications/children/<int:parent_id>', methods=['GET'])
def get_children(parent_id):
    try:
        app.logger.debug(f"Fetching children for parent_id: {parent_id}")
        page = request.args.get('page', 1, type=int)
        per_page = 50
        parent = db.session.get(Indication, parent_id) if parent_id != 0 else None
        if not parent and parent_id != 0:
            app.logger.debug(f"Parent with ID {parent_id} not found")
            return jsonify({'error': f'Parent with ID {parent_id} not found'}), 404

        if parent_id == 0:
            expected_class_kind = 'chapter'
        elif parent.class_kind == 'chapter':
            expected_class_kind = 'block'
        elif parent.class_kind == 'block':
            has_sub_blocks = Indication.query.filter_by(parent_id=parent_id, class_kind='block').count() > 0
            expected_class_kind = 'block' if has_sub_blocks else 'category'
        elif parent.class_kind == 'category':
            expected_class_kind = 'category'
        else:
            expected_class_kind = None

        children_query = Indication.query.filter_by(parent_id=parent_id if parent_id != 0 else None)
        if expected_class_kind:
            children_query = children_query.filter_by(class_kind=expected_class_kind)
        
        # Sort chapters by chapter_no for parent_id=0, otherwise sort by code and name_en
        if parent_id == 0:
            children_query = children_query.order_by(Indication.chapter_no.asc())
        else:
            children_query = children_query.order_by(
                func.coalesce(Indication.code, ''),
                Indication.code.asc(),
                Indication.name_en.asc()
            )

        children_paginated = children_query.paginate(page=page, per_page=per_page, error_out=False)
        children = children_paginated.items

        app.logger.debug(f"Found {len(children)} children for parent_id: {parent_id} on page {page} (expected class_kind: {expected_class_kind})")
        for child in children:
            app.logger.debug(f"Child: {child.name_en} (ID: {child.id}, Code: {child.code}, Parent ID: {child.parent_id}, ClassKind: {child.class_kind})")

        response = [{
            'id': ind.id,
            'name_en': str(ind.name_en),
            'chapter_no': str(ind.chapter_no) if ind.chapter_no else '',
            'class_kind': str(ind.class_kind),
            'code': str(ind.code) if ind.code else '',
            'has_children': ind.has_children,
            'is_leaf': ind.is_leaf
        } for ind in children]
        app.logger.debug(f"Returning response for parent_id: {parent_id}: {response}")
        return jsonify({
            'children': response,
            'has_next': children_paginated.has_next,
            'page': page,
            'total_pages': children_paginated.pages
        })
    except Exception as e:
        app.logger.error(f"Error in get_children for parent_id {parent_id}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/indications/search', methods=['GET'])
def search_indications():
    query = request.args.get('search', '').strip()
    page = request.args.get('page', 1, type=int)
    limit = request.args.get('limit', 10, type=int)
    drug_id = request.args.get('drug_id', None)

    if not query:
        return jsonify({"error": "Search parameter is required"}), 400

    indications_query = Indication.query.filter(
        or_(
            Indication.name_en.ilike(f'%{query}%'),
            Indication.code.ilike(f'%{query}%')
        )
    )

    # Apply drug_id filter with a left outer join
    if drug_id:
        indications_query = indications_query.outerjoin(
            DrugDetail,
            and_(
                DrugDetail.drug_id == drug_id,
                or_(
                    DrugDetail.indications == f"{Indication.id}",
                    DrugDetail.indications.like(f"{Indication.id},%"),
                    DrugDetail.indications.like(f"%,{Indication.id},%"),
                    DrugDetail.indications.like(f"%,{Indication.id}")
                )
            )
        )
        # Include indications even if no DrugDetail match (i.e., allow null joins)
        indications_query = indications_query.filter(
            or_(
                DrugDetail.id.is_(None),  # No DrugDetail record
                DrugDetail.drug_id == drug_id  # Matching DrugDetail record
            )
        )

    indications_paginated = indications_query.order_by(Indication.code.asc(), Indication.name_en.asc()).paginate(
        page=page, per_page=limit, error_out=False
    )

    results = [{
        "id": ind.id,
        "text": f"{ind.name_en} ({ind.code or 'N/A'})",
        "depth": ind.depth,
        "class_kind": ind.class_kind
    } for ind in indications_paginated.items]

    return jsonify({
        "results": results,
        "has_next": indications_paginated.has_next
    })

@app.route('/indications/<int:id>/edit', methods=['GET'])
def edit_indication(id):
    indication = Indication.query.get_or_404(id)
    return render_template('edit_indication.html', indication=indication)

@app.route('/indications/<int:id>/update', methods=['POST'])
def update_indication(id):
    indication = Indication.query.get_or_404(id)
    try:
        indication.name_en = request.form.get('name_en')
        indication.code = request.form.get('code')
        db.session.commit()
        flash("Indication updated successfully!", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"Error updating indication: {e}", "error")
    return redirect(url_for('manage_indications'))

@app.route('/indications/<int:id>/delete', methods=['POST'])
def delete_indication(id):
    indication = Indication.query.get_or_404(id)
    try:
        db.session.delete(indication)
        db.session.commit()
        flash("Indication deleted successfully!", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"Error deleting indication: {e}", "error")
    return redirect(url_for('manage_indications'))

@app.route('/clear-indications', methods=['POST'])
def clear_indications():
    try:
        db.session.query(Indication).delete()
        db.session.commit()
        flash("Indication table cleared successfully!", "success")
        app.logger.info("Indication table cleared by user request.")
    except Exception as e:
        db.session.rollback()
        flash(f"Error clearing indication table: {e}", "error")
        app.logger.error(f"Error clearing indication table: {e}")
    return redirect(url_for('manage_indications'))
#ICD-11 Son...

@app.route('/targets', methods=['GET', 'POST'])
def manage_targets():
    # Mevcut sayfa numarasını al (Varsayılan: 1)
    page = request.args.get('page', 1, type=int)

    # Her sayfada gösterilecek target sayısı
    per_page = 20

    # Sayfalama işlemi
    targets_paginated = Target.query.paginate(page=page, per_page=per_page)

    if request.method == 'POST':
        name_tr = request.form.get('name_tr')
        name_en = request.form.get('name_en')

        # Yeni Target ekleme
        new_target = Target(name_tr=name_tr, name_en=name_en)
        db.session.add(new_target)
        db.session.commit()
        return redirect(url_for('manage_targets'))

    # targets_paginated değişkenini şablona gönder
    return render_template('targets.html', targets_paginated=targets_paginated)



@app.route('/search', methods=['GET', 'POST'])
def search():
    query = request.form.get('query', '').strip()  # Get the search query from the form

    drugs = set()  # Use a set to avoid duplicates
    diseases = []
    salts = []
    target_molecules = []
    side_effects = []

    if query:  # Only search if there's a query
        # Search drugs by name
        drugs.update(Drug.query.filter(
            (Drug.name_en.ilike(f'%{query}%')) | (Drug.name_tr.ilike(f'%{query}%'))
        ).all())

        # Search salts
        salts = Salt.query.filter(
            (Salt.name_en.ilike(f'%{query}%')) | (Salt.name_tr.ilike(f'%{query}%'))
        ).all()

        # Search diseases (indications)
        indications = Indication.query.filter(
            (Indication.name_en.ilike(f'%{query}%')) | (Indication.name_tr.ilike(f'%{query}%'))
        ).all()

        # Find drugs related to the indications
        for indication in indications:
            related_details = DrugDetail.query.filter(
                DrugDetail.indications.contains(str(indication.id))
            ).all()
            for detail in related_details:
                drugs.add(detail.drug)  # Add the parent drug

            diseases.append({
                'indication': indication,
                'related_drugs': [detail.drug for detail in related_details]
            })

        # Search target molecules
        target_molecules = Target.query.filter(
            Target.name_en.ilike(f'%{query}%')
        ).all()

        for target in target_molecules:
            related_details = DrugDetail.query.filter(
                DrugDetail.target_molecules.contains(str(target.id))
            ).all()
            for detail in related_details:
                drugs.add(detail.drug)  # Add the parent drug

        # Search side effects
        side_effects = SideEffect.query.filter(
            (SideEffect.name_en.ilike(f'%{query}%')) | (SideEffect.name_tr.ilike(f'%{query}%'))
        ).all()

        for side_effect in side_effects:
            related_details = side_effect.details.all()  # Fetch related DrugDetails
            for detail in related_details:
                drugs.add(detail.drug)  # Add the parent drug

    return render_template(
        'search.html',
        query=query,
        drugs=list(drugs),  # Convert set back to list for rendering
        diseases=diseases,
        salts=salts,
        target_molecules=target_molecules,
        side_effects=side_effects
    )





@app.route('/details', methods=['GET'])
def manage_details():
    details = DrugDetail.query.all()
    return render_template('details_list.html', details=details)


@app.route('/interactions', methods=['GET', 'POST'])
@login_required
def check_interactions():
    drugs = Drug.query.all()
    interaction_results = []

    if request.method == 'POST':
        drug1_id = request.form.get('drug1_id')
        drug2_id = request.form.get('drug2_id')

        interactions = DrugInteraction.query.filter(
            db.or_(
                db.and_(DrugInteraction.drug1_id == drug1_id, DrugInteraction.drug2_id == drug2_id),
                db.and_(DrugInteraction.drug1_id == drug2_id, DrugInteraction.drug2_id == drug1_id)
            )
        ).all()

        for interaction in interactions:
            # Eğer predicted_severity bir yapay zeka modelinden veya başka bir yerden çekiliyorsa
            # Burada onun değerini almak için kontrol eklenmelidir.
            predicted_severity = getattr(interaction, 'predicted_severity', 'Not Available')  # Varsayılan değer ekler

            interaction_results.append({
                'drug1': interaction.drug1.name_en if interaction.drug1 else "Unknown",
                'drug2': interaction.drug2.name_en if interaction.drug2 else "Unknown",
                'route': interaction.route.name if interaction.route else "General",
                'interaction_type': interaction.interaction_type,
                'interaction_description': interaction.interaction_description,
                'severity': interaction.severity,
                'predicted_severity': predicted_severity,
                'mechanism': interaction.mechanism or "Not Provided",
                'monitoring': interaction.monitoring or "Not Provided",
                'alternatives': interaction.alternatives or "Not Provided",
                'reference': interaction.reference or "Not Provided",
            })

    return render_template('interactions.html', drugs=drugs, interaction_results=interaction_results)




@app.route('/interactions/manage', methods=['GET', 'POST'])
def manage_interactions():
    drugs = Drug.query.all()
    routes = RouteOfAdministration.query.all()

    # Sayfa numarasını al ve sayfa başına kayıt sayısını belirle
    page = request.args.get('page', 1, type=int)
    per_page = 10

    # Sıralama parametrelerini al
    order_by_column = request.args.get('order_by', 'id')  # Varsayılan: 'id'
    order_direction = request.args.get('direction', 'desc')  # Varsayılan: 'desc'

    # Etkileşimleri al ve sıralama yap
    interactions_query = DrugInteraction.query.order_by(
        getattr(getattr(DrugInteraction, order_by_column), order_direction)()
    )
    interactions = interactions_query.paginate(page=page, per_page=per_page)

    if request.method == 'POST':
        # Formdan gelen verileri al
        drug1_id = request.form.get('drug1_id')
        drug2_id = request.form.get('drug2_id')
        route_id = request.form.get('route_id')
        interaction_type = request.form.get('interaction_type')
        interaction_description = request.form.get('interaction_description')
        severity = request.form.get('severity')
        reference = request.form.get('reference')
        mechanism = request.form.get('mechanism')
        monitoring = request.form.get('monitoring')
        alternatives = request.form.get('alternatives')

        # Yeni etkileşim kaydet
        new_interaction = DrugInteraction(
            drug1_id=drug1_id,
            drug2_id=drug2_id,
            route_id=route_id if route_id else None,
            interaction_type=interaction_type,
            interaction_description=interaction_description,
            severity=severity,
            reference=reference,
            mechanism=mechanism,
            monitoring=monitoring,
            alternatives=alternatives
        )
        db.session.add(new_interaction)
        db.session.commit()

        return redirect(url_for('manage_interactions'))

    return render_template(
        'manage_interactions.html',
        drugs=drugs,
        routes=routes,
        interactions=interactions
    )





#İlaç Veriliş Yolu ile ilgili olaylar
@app.route('/routes/manage', methods=['GET', 'POST'])
def manage_routes():
    routes = RouteOfAdministration.query.all()

    if request.method == 'POST':
        name = request.form.get('name')
        route_type = request.form.get('type')  # "Sistemik", "Lokal", vb.
        parent_id = request.form.get('parent_id', None)
        description = request.form.get('description')  # Add this line

        new_route = RouteOfAdministration(
            name=name,
            type=route_type,
            parent_id=parent_id if parent_id else None,
            description=description  # Add this
        )
        db.session.add(new_route)
        db.session.commit()
        return redirect(url_for('manage_routes'))

    return render_template('manage_routes.html', routes=routes)

@app.route('/routes/filter', methods=['POST'])
def filter_routes():
    selected_type = request.json.get('type')  # Selected category
    filtered_routes = RouteOfAdministration.query.filter_by(type=selected_type).all()
    return jsonify([{'id': route.id, 'name': route.name} for route in filtered_routes])

@app.route('/routes/delete/<int:route_id>', methods=['POST'])
def delete_route(route_id):
    route = RouteOfAdministration.query.get_or_404(route_id)
    db.session.delete(route)
    db.session.commit()
    return redirect(url_for('manage_routes'))

@app.route('/routes/update/<int:route_id>', methods=['POST'])
def update_route(route_id):
    route = RouteOfAdministration.query.get_or_404(route_id)
    route.name = request.form.get('name')
    route.type = request.form.get('type')
    route.description = request.form.get('description')
    parent_id = request.form.get('parent_id')
    route.parent_id = parent_id if parent_id else None
    db.session.commit()
    return redirect(url_for('manage_routes'))
#İlaç veriliş SON...


@app.route('/drug/update/<int:drug_id>', methods=['GET', 'POST'])
def update_drug(drug_id):
    drug = Drug.query.get_or_404(drug_id)
    categories = DrugCategory.query.order_by(DrugCategory.name).all()  # Fetch all categories dynamically

    if request.method == 'POST':
        try:
            # Update existing fields with your custom logic
            drug.name_tr = request.form['name_tr']
            drug.name_en = request.form['name_en']
            drug.alternative_names = request.form['alternative_names'].replace('\n', '|')  # Newlines to pipes
            
            # Update category
            category_id = request.form.get('category_id')  # Get category_id from form
            drug.category_id = int(category_id) if category_id else None  # Allow clearing category
            
            db.session.commit()
            flash('Drug updated successfully!', 'success')
            return redirect(url_for('index'))  # Keep your redirect to index
        except Exception as e:
            db.session.rollback()
            flash(f'Error updating drug: {e}', 'error')

    # Prepare alternative_names for display (your existing logic)
    alternative_names = '\n'.join(drug.alternative_names.replace('|', ',').split(',')) if drug.alternative_names else ''
    
    return render_template('update_drug.html', drug=drug, alternative_names=alternative_names, categories=categories)

@app.route('/drug/delete/<int:drug_id>', methods=['POST'])
def delete_drug(drug_id):
    drug = Drug.query.get_or_404(drug_id)
    db.session.delete(drug)
    db.session.commit()
    return redirect(url_for('index'))


#with app.app_context():
 #   drug_routes = DrugRoute.query.all()
 #   if not drug_routes:
 #       print("DrugRoute tablosunda hiç kayıt yok.")
 #   else:
 #       for route in drug_routes:
 #           print(f"DrugRoute ID: {route.id}, DrugDetail ID: {route.drug_detail_id}, Route ID: {route.route_id}")

@app.route('/drug/<int:drug_id>')
def drug_detail(drug_id):
    drug = Drug.query.get_or_404(drug_id)
    details = DrugDetail.query.filter_by(drug_id=drug_id).all()
    salts = drug.salts.all()

    enriched_details = []
    for detail in details:
        print(f"DEBUG: Enriching detail ID {detail.id}, structure_3d: {detail.structure_3d}")
        routes_info = []
        for route in detail.routes:
            route_indications = [
                f"{ri.indication.name_en} ({ri.indication.name_tr})" if ri.indication.name_tr else ri.indication.name_en
                for ri in route.route_indications
            ]
            routes_info.append({
                'name': route.route.name,
                'pharmacodynamics': route.pharmacodynamics,
                'pharmacokinetics': route.pharmacokinetics,
                'indications': route_indications,
                'absorption_rate': {
                    'min': route.absorption_rate_min,
                    'max': route.absorption_rate_max
                },
                'volume_of_distribution': {
                    'min': route.vod_rate_min,
                    'max': route.vod_rate_max
                },
                'protein_binding': {
                    'min': route.protein_binding_min,
                    'max': route.protein_binding_max
                },
                'half_life': {
                    'min': route.half_life_min,
                    'max': route.half_life_max
                },
                'clearance_rate': {
                    'min': route.clearance_rate_min,
                    'max': route.clearance_rate_max
                },
                'bioavailability': {
                    'min': route.bioavailability_min,
                    'max': route.bioavailability_max
                },
                'tmax': {
                    'min': route.tmax_min,
                    'max': route.tmax_max
                },
                'cmax': {
                    'min': route.cmax_min,
                    'max': route.cmax_max
                },
                'metabolism_organs': [organ.name for organ in route.metabolism_organs] if route.metabolism_organs else [],
                'metabolism_enzymes': [enzyme.name for enzyme in route.metabolism_enzymes] if route.metabolism_enzymes else [],
                'metabolites': [
                    {'id': met.id, 'name': met.name, 'parent_id': met.parent_id} for met in route.metabolites
                ] if route.metabolites else []
            })

        indications_list = []
        if detail.indications:
            indication_ids = [int(ind_id) for ind_id in detail.indications.split(',') if ind_id.isdigit()]
            indications = Indication.query.filter(Indication.id.in_(indication_ids)).all()
            indications_list = [f"{ind.name_en} ({ind.name_tr})" if ind.name_tr else ind.name_en for ind in indications]

        target_molecules_list = []
        if detail.target_molecules:
            target_ids = [tid for tid in detail.target_molecules.split(',') if tid.isdigit()]
            targets = Target.query.filter(Target.id.in_([int(tid) for tid in target_ids])).all()
            target_molecules_list = [t.name_en for t in targets]

        side_effects_list = [
            {"name_en": se.name_en, "name_tr": se.name_tr or 'N/A'}
            for se in detail.side_effects
        ]

        enriched_details.append({
            'id': detail.id,  # Added to ensure detail['id'] works
            'drug_name': drug.name_en,
            'mechanism_of_action': detail.mechanism_of_action,
            'salt_name': detail.salt.name_en if detail.salt else None,
            'fda_approved': detail.fda_approved,
            'ema_approved': detail.ema_approved,
            'titck_approved': detail.titck_approved,
            'molecular_formula': detail.molecular_formula,
            'synthesis': detail.synthesis,
            'structure': detail.structure,
            'structure_3d': detail.structure_3d,
            'iupac_name': detail.iupac_name,
            'smiles': detail.smiles,
            'inchikey': detail.inchikey,
            'pubchem_cid': detail.pubchem_cid,
            'pubchem_sid': detail.pubchem_sid,
            'cas_id': detail.cas_id,
            'ec_number': detail.ec_number,
            'nci_code': detail.nci_code,
            'rxcui': detail.rxcui,
            'snomed_id': detail.snomed_id,
            'molecular_weight': detail.molecular_weight,
            'solubility': detail.solubility,
            'pharmacodynamics': detail.pharmacodynamics,
            'pharmacokinetics': detail.pharmacokinetics,
            'boiling_point': detail.boiling_point,
            'melting_point': detail.melting_point,
            'density': detail.density,
            'flash_point': detail.flash_point,
            'routes': routes_info,
            'indications': indications_list,
            'target_molecules': target_molecules_list,
            'side_effects': side_effects_list,
            'black_box_warning': detail.black_box_warning,
            'black_box_details': detail.black_box_details,
        })

    return render_template('drug_detail.html', drug=drug, salts=salts, details=enriched_details)


import qrcode
from flask import send_file
import io

@app.route('/generate_qr/<int:drug_id>')
def generate_qr(drug_id):
    drug_detail_url = url_for('drug_detail', drug_id=drug_id, _external=True)
    qr = qrcode.make(drug_detail_url)
    qr_io = io.BytesIO()
    qr.save(qr_io, 'PNG')
    qr_io.seek(0)
    return send_file(qr_io, mimetype='image/png')

from flask import request




@app.route('/side_effects', methods=['GET', 'POST'])
def manage_side_effects():
    if request.method == 'POST':
        name_en = request.form.get('name_en')
        name_tr = request.form.get('name_tr')
        new_side_effect = SideEffect(name_en=name_en, name_tr=name_tr)
        db.session.add(new_side_effect)
        db.session.commit()
        return redirect(url_for('manage_side_effects'))

    side_effects = SideEffect.query.all()
    return render_template('side_effects.html', side_effects=side_effects)

@app.route('/side_effects/edit/<int:side_effect_id>', methods=['GET', 'POST'])
def edit_side_effect(side_effect_id):
    side_effect = SideEffect.query.get_or_404(side_effect_id)
    if request.method == 'POST':
        side_effect.name_en = request.form.get('name_en', side_effect.name_en)
        side_effect.name_tr = request.form.get('name_tr', side_effect.name_tr)
        db.session.commit()
        return redirect(url_for('manage_side_effects'))
    return render_template('edit_side_effect.html', side_effect=side_effect)

@app.route('/side_effects/delete/<int:side_effect_id>', methods=['POST'])
def delete_side_effect(side_effect_id):
    side_effect = SideEffect.query.get_or_404(side_effect_id)
    db.session.delete(side_effect)
    db.session.commit()
    return redirect(url_for('manage_side_effects'))


@app.route('/details/update/<int:detail_id>', methods=['GET', 'POST'])
def update_detail(detail_id):
    detail = DrugDetail.query.get_or_404(detail_id)
    drugs = Drug.query.all()
    salts = Salt.query.all()
    indications = Indication.query.all()
    targets = Target.query.all()
    routes = RouteOfAdministration.query.all()
    side_effects = SideEffect.query.all()

    if request.method == 'POST':
        try:
            # Debugging: Log incoming form data
            print(f"DEBUG: Received Form Data -> {request.form}")

            # Update fields
            detail.drug_id = request.form.get('drug_id')
            detail.salt_id = request.form.get('salt_id', None)
            detail.molecular_formula = request.form.get('molecular_formula')
            detail.synthesis = request.form.get('synthesis')
            detail.iupac_name = request.form.get('iupac_name')
            detail.smiles = request.form.get('smiles')
            detail.inchikey = request.form.get('inchikey')
            detail.pubchem_cid = request.form.get('pubchem_cid')
            detail.pubchem_sid = request.form.get('pubchem_sid')
            detail.cas_id = request.form.get('cas_id')
            detail.ec_number = request.form.get('ec_number')
            detail.nci_code = request.form.get('nci_code')
            detail.rxcui = request.form.get('rxcui')
            detail.snomed_id = request.form.get('snomed_id')
            detail.molecular_weight = request.form.get('molecular_weight')
            detail.solubility = request.form.get('solubility')
            detail.pharmacodynamics = request.form.get('pharmacodynamics')
            detail.pharmacokinetics = request.form.get('pharmacokinetics')
            detail.boiling_point = request.form.get('boiling_point')
            detail.melting_point = request.form.get('melting_point')
            detail.density = request.form.get('density')
            detail.flash_point = request.form.get('flash_point')
            detail.fda_approved = 'fda_approved' in request.form
            detail.ema_approved = 'ema_approved' in request.form
            detail.titck_approved = 'titck_approved' in request.form

            # Update multi-select fields
            detail.indications = ','.join(request.form.getlist('indications[]')) or None
            detail.target_molecules = ','.join(request.form.getlist('target_molecules[]')) or None

            # Update side effects
            selected_side_effects = request.form.getlist('side_effects[]')
            detail.side_effects = [SideEffect.query.get(se_id) for se_id in selected_side_effects]

            # Update routes and route-specific indications
            selected_routes = request.form.getlist('route_id[]')
            existing_routes = {route.route_id: route for route in detail.routes}

            # Add or update routes
            for route_id in selected_routes:
                pharmacodynamics = request.form.get(f'route_pharmacodynamics_{route_id}', '')
                pharmacokinetics = request.form.get(f'route_pharmacokinetics_{route_id}', '')

                if int(route_id) in existing_routes:
                    # Update existing route
                    existing_route = existing_routes[int(route_id)]
                    existing_route.pharmacodynamics = pharmacodynamics or existing_route.pharmacodynamics
                    existing_route.pharmacokinetics = pharmacokinetics or existing_route.pharmacokinetics

                    # Update route-specific indications
                    selected_route_indications = request.form.getlist(f'route_indications_{route_id}[]')
                    if selected_route_indications:
                        existing_route.route_indications = [
                            RouteIndication(
                                drug_detail_id=detail.id,
                                route_id=route_id,
                                indication_id=int(indication_id)
                            )
                            for indication_id in selected_route_indications
                        ]
                else:
                    # Add a new route
                    new_route = DrugRoute(
                        drug_detail_id=detail.id,
                        route_id=route_id,
                        pharmacodynamics=pharmacodynamics,
                        pharmacokinetics=pharmacokinetics
                    )

                    # Add route-specific indications
                    selected_route_indications = request.form.getlist(f'route_indications_{route_id}[]')
                    for indication_id in selected_route_indications:
                        new_route.route_indications.append(
                            RouteIndication(
                                drug_detail_id=detail.id,
                                route_id=route_id,
                                indication_id=int(indication_id)
                            )
                        )
                    detail.routes.append(new_route)

            # Remove routes that are no longer selected
            for existing_route_id in list(existing_routes.keys()):
                if str(existing_route_id) not in selected_routes:
                    route_to_remove = existing_routes[existing_route_id]
                    db.session.delete(route_to_remove)



            db.session.commit()  # Save updates
            print("DEBUG: Updated details saved successfully")
            return redirect(url_for('view_details'))

        except Exception as e:
            print(f"ERROR: {e}")

    return render_template(
        'update_detail.html',
        detail=detail,
        drugs=drugs,
        salts=salts,
        indications=indications,
        targets=targets,
        routes=routes,
        side_effects=side_effects,
        selected_indications=detail.indications.split(',') if detail.indications else [],
        selected_targets=detail.target_molecules.split(',') if detail.target_molecules else [],
        selected_side_effects=[se.id for se in detail.side_effects] if detail.side_effects else [],
        selected_routes=[
            {'route_id': route.route_id, 'pharmacodynamics': route.pharmacodynamics, 'pharmacokinetics': route.pharmacokinetics, 'indications': [ri.indication_id for ri in route.route_indications]}
            for route in detail.routes
        ]
    )







@app.route('/cdss/advanced', methods=['GET', 'POST'])
def cdss_advanced():
    drugs = Drug.query.all()
    routes = RouteOfAdministration.query.all()
    indications = Indication.query.all()
    interaction_results = None

    if request.method == 'POST':
        age = request.form.get('age', type=int, default=30)
        weight = request.form.get('weight', type=float, default=70.0)
        crcl = request.form.get('crcl', type=float, default=None)
        gender = request.form.get('gender', default='M')
        selected_indications = request.form.getlist('indications')  # Condition IDs
        selected_drugs = request.form.getlist('drugs')
        selected_route = request.form.get('route_id')

        if age < 0 or weight <= 0 or (crcl is not None and crcl < 0):
            return render_template('cdss_advanced.html', drugs=drugs, routes=routes, 
                                 indications=indications, interaction_results=None, 
                                 error="Invalid input—check age, weight, or CrCl!")

        if crcl is None and age and weight and gender:
            crcl = calculate_crcl(age, weight, gender)

        interaction_results = analyze_interactions(
            selected_drugs, selected_route, age, weight, crcl, 
            [int(i) for i in selected_indications]
        )

    return render_template('cdss_advanced.html', drugs=drugs, routes=routes, 
                         indications=indications, interaction_results=interaction_results)

def calculate_crcl(age, weight, gender, serum_creatinine=1.0):
    """Calculate CrCl using Cockcroft-Gault formula."""
    if gender.upper() == 'F':
        factor = 0.85
    else:
        factor = 1.0
    return ((140 - age) * weight * factor) / (72 * serum_creatinine)

def analyze_interactions(drug_ids, route_id, age, weight, crcl, conditions):
    results = []
    drug_ids = [int(d) for d in drug_ids]

    # 1. Multi-Drug Interactions
    for drug1_id, drug2_id in combinations(drug_ids, 2):
        interactions = DrugInteraction.query.filter(
            db.or_(
                db.and_(DrugInteraction.drug1_id == drug1_id, DrugInteraction.drug2_id == drug2_id),
                db.and_(DrugInteraction.drug1_id == drug2_id, DrugInteraction.drug2_id == drug1_id)
            )
        ).filter(
            db.or_(DrugInteraction.route_id == route_id, DrugInteraction.route_id == None)
        ).all()

        for interaction in interactions:
            drug1 = Drug.query.get(drug1_id)
            drug2 = Drug.query.get(drug2_id)
            detail1 = DrugDetail.query.filter_by(drug_id=drug1_id).first() or DrugDetail()
            detail2 = DrugDetail.query.filter_by(drug_id=drug2_id).first() or DrugDetail()

            time_points, risk_profile = simulate_pk_interaction(detail1, detail2, interaction, age, weight, crcl)

            result = {
                'type': 'Drug-Drug Interaction',
                'drug1': drug1.name_en,
                'drug2': drug2.name_en,
                'route': RouteOfAdministration.query.get(route_id).name if route_id else "General",
                'interaction_type': interaction.interaction_type,
                'description': interaction.interaction_description,
                'severity': interaction.severity,
                'predicted_severity': adjust_severity(interaction.severity, age, crcl),
                'peak_risk': interaction.time_to_peak or time_points[risk_profile.argmax()],
                'risk_profile': list(zip(time_points, risk_profile)),
                'mechanism': interaction.mechanism or "Not Provided",
                'monitoring': interaction.monitoring or "Not Provided",
                'alternatives': interaction.alternatives or "Not Provided",
                'reference': interaction.reference or "Not Provided"
            }
            results.append(result)

    # 2. Condition Checks (Only this remains!)
    condition_risks = {
        "renal failure": {"severity_boost": 1, "desc": "Worsened by renal impairment", "monitoring": "Monitor renal function"},
        "liver disease": {"severity_boost": 1, "desc": "Risk of hepatotoxicity", "monitoring": "Check LFTs"},
        "heart failure": {"severity_boost": 0.5, "desc": "May exacerbate fluid retention", "monitoring": "Monitor BP and weight"}
    }

    if conditions:
        for drug_id in drug_ids:
            drug = Drug.query.get(drug_id)
            for condition_id in conditions:
                condition = Indication.query.get(condition_id)
                if condition and condition.name_en.lower() in condition_risks:  # Fixed to name_en
                    risk_info = condition_risks[condition.name_en.lower()]
                    base_severity = "Moderate"
                    adjusted_severity = adjust_severity(base_severity, age, crcl, boost=risk_info["severity_boost"])
                    results.append({
                        'type': 'Drug-Condition Alert',
                        'drug1': drug.name_en,
                        'drug2': condition.name_en,  # Fixed to name_en
                        'route': "N/A",
                        'interaction_type': "Potential Risk",
                        'description': f"Caution: {drug.name_en} with {condition.name_en}. {risk_info['desc']}",
                        'severity': base_severity,
                        'predicted_severity': adjusted_severity,
                        'peak_risk': 0,
                        'risk_profile': [(0, 2 + risk_info["severity_boost"])],
                        'mechanism': "Condition-Drug Interaction",
                        'monitoring': risk_info["monitoring"],
                        'alternatives': "Consider alternative therapy",
                        'reference': "Clinical guideline"
                    })

    return results

def simulate_pk_interaction(detail1, detail2, interaction, age, weight, crcl):
    half_life1 = detail1.half_life or 4.0
    half_life2 = detail2.half_life or 4.0
    clearance1 = detail1.clearance_rate or 100.0
    clearance2 = detail2.clearance_rate or 100.0
    bioavail1 = detail1.bioavailability or 1.0
    bioavail2 = detail2.bioavailability or 1.0

    if age > 65:
        clearance1 *= 0.75
        clearance2 *= 0.75
    elif age < 18:
        clearance1 *= 1.2
        clearance2 *= 1.2

    if crcl:
        renal_factor = min(crcl / 100, 1.0)
        clearance1 *= renal_factor
        clearance2 *= renal_factor

    if weight:
        clearance1 *= (weight / 70)
        clearance2 *= (weight / 70)

    time_points = np.linspace(0, 24, 100)
    conc1 = bioavail1 * np.exp(-np.log(2) / half_life1 * time_points) * clearance1
    conc2 = bioavail2 * np.exp(-np.log(2) / half_life2 * time_points) * clearance2
    base_risk = {'Hafif': 1, 'Moderate': 2, 'Severe': 3}.get(interaction.severity, 1)
    risk_profile = base_risk * conc1 * conc2 * (1 if crcl > 30 else 1.5)
    return time_points, risk_profile

def adjust_severity(base_severity, age, crcl, boost=0):
    severity_map = {'Hafif': 1, 'Moderate': 2, 'Severe': 3}
    score = severity_map.get(base_severity, 2) + boost
    if age > 75 or crcl < 30:
        score += 1
    elif age < 18:
        score += 0.5
    if score >= 3:
        return "Severe"
    elif score >= 2:
        return "Moderate"
    return "Hafif"



@app.route('/upload', methods=['GET', 'POST'])
def upload_drugs():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            return "No file uploaded", 400

        # Check file extension
        if file.filename.endswith('.xlsx'):
            try:
                data = pd.read_excel(file, dtype=str)
                logger.info(f"Loaded Excel file with {len(data)} rows")
                if len(data) > 20000:
                    return "Excel file exceeds 20,000 rows. Please split into smaller files.", 400
            except Exception as e:
                logger.error(f"Error reading Excel file: {e}")
                return f"Error reading Excel file: {e}", 400
        else:
            return "Unsupported file format. Please upload an Excel (.xlsx) file.", 400

        # Clean column names
        data.columns = data.columns.str.strip()

        # Check if 'name_en' column exists
        if 'name_en' not in data.columns:
            return f"The file must contain a 'name_en' column. Found columns: {data.columns.tolist()}", 400

        # Process rows in batches
        batch_size = 50
        chunk_size = 500
        new_drugs = []
        total_processed = 0

        for start in range(0, len(data), chunk_size):
            chunk = data[start:start + chunk_size]
            logger.info(f"Processing chunk of {len(chunk)} rows (rows {start + 1} to {start + len(chunk)})")

            # Batch check for duplicates
            name_en_list = chunk['name_en'].dropna().str.strip().tolist()
            with db.session.no_autoflush:
                existing_drugs = {d.name_en for d in Drug.query.filter(Drug.name_en.in_(name_en_list)).all()}

            for index, row in chunk.iterrows():
                name_en = row.get('name_en')
                
                if pd.isna(name_en) or not str(name_en).strip():
                    logger.error(f"Invalid 'name_en' in row {index + 2}")
                    return f"Invalid 'name_en' value in row {index + 2}.", 400

                name_en = str(name_en).strip()
                if len(name_en) > 255:
                    logger.error(f"Drug name too long in row {index + 2}: {name_en}")
                    return f"Drug name '{name_en}' in row {index + 2} exceeds maximum length of 255 characters.", 400

                name_tr = row.get('name_tr', name_en)
                name_tr = str(name_tr).strip() if pd.notna(name_tr) else name_en
                if len(name_tr) > 255:
                    logger.error(f"Drug name (TR) too long in row {index + 2}: {name_tr}")
                    return f"Drug name (TR) '{name_tr}' in row {index + 2} exceeds maximum length of 255 characters.", 400

                # Skip duplicates
                if name_en in existing_drugs:
                    logger.debug(f"Skipping duplicate drug: {name_en}")
                    continue

                # Handle optional fields
                alt_names = row.get('alternative_names', None)
                if pd.notna(alt_names):
                    alt_names = str(alt_names).split(" | ")
                    alt_names_str = ", ".join(alt_names)
                else:
                    alt_names_str = None

                new_drug = Drug(
                    name_en=name_en,
                    name_tr=name_tr,
                    alternative_names=alt_names_str,
                    fda_approved=False
                )
                new_drugs.append(new_drug)

                if len(new_drugs) >= batch_size:
                    logger.info(f"Committing batch of {len(new_drugs)} drugs")
                    db.session.add_all(new_drugs)
                    try:
                        db.session.commit()
                        logger.info(f"Committed {len(new_drugs)} drugs")
                        new_drugs = []
                    except Exception as e:
                        db.session.rollback()
                        logger.error(f"Error committing drugs: {e}")
                        return f"Error saving drugs: {e}", 500

            total_processed += len(chunk)

        if new_drugs:
            logger.info(f"Committing final batch of {len(new_drugs)} drugs")
            db.session.add_all(new_drugs)
            try:
                db.session.commit()
                logger.info(f"Committed {len(new_drugs)} drugs")
            except Exception as e:
                db.session.rollback()
                logger.error(f"Error committing drugs: {e}")
                return f"Error saving drugs: {e}", 500

        logger.info(f"Upload completed successfully. Processed {total_processed} rows.")
        return f"Drugs uploaded successfully, duplicates were skipped! Processed {total_processed} rows.", 200

    return render_template('upload.html')






@app.route('/side_effects/upload', methods=['GET', 'POST'])
def upload_side_effects():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            return "No file uploaded", 400

        try:
            # Hatalı satırları atla ve tırnakları dikkate al
            data = pd.read_csv(file, sep=',', quotechar='"', on_bad_lines='skip')
        except Exception as e:
            return f"Error processing file: {str(e)}", 400

        # Gerekli sütunu kontrol et
        if 'name_en' not in data.columns:
            return "The file must contain a column named 'name_en'.", 400

        # Veritabanına ekle
        for _, row in data.iterrows():
            new_side_effect = SideEffect(name_en=row['name_en'], name_tr=None)
            db.session.add(new_side_effect)
        db.session.commit()
        return "Side effects uploaded successfully!", 200

    return render_template('side_effects_upload.html')






@app.route('/api/active_ingredients', methods=['GET'])
def get_active_ingredients():
    search = request.args.get('q', '').strip()  # Changed from 'search' to 'q' for consistency
    limit = request.args.get('limit', 10, type=int)
    page = request.args.get('page', 1, type=int)

    query = Drug.query.filter(
        db.or_(
            Drug.name_en.ilike(f"%{search}%"),
            Drug.name_tr.ilike(f"%{search}%"),
            Drug.alternative_names.ilike(f"%{search}%")
        )
    ) if search else Drug.query

    paginated_query = query.paginate(page=page, per_page=limit)

    results = [
        {"id": drug.id, "text": f"{drug.name_en} ({drug.name_tr})"}
        for drug in paginated_query.items
    ]

    return jsonify({
        "results": results,
        "pagination": {"more": paginated_query.has_next}
    })



@app.route('/api/salts', methods=['GET'])
def get_salts():
    search = request.args.get('search', '').strip()
    limit = request.args.get('limit', 10, type=int)

    query = Salt.query.filter(
        db.or_(
            Salt.name_en.ilike(f"%{search}%"),
            Salt.name_tr.ilike(f"%{search}%")
        )
    ) if search else Salt.query

    query = query.limit(limit)
    results = query.all()

    return jsonify([
        {"id": salt.id, "text": f"{salt.name_en} ({salt.name_tr})"}
        for salt in results
    ])


@app.route('/interactions/update/<int:interaction_id>', methods=['GET', 'POST'])
def update_interaction(interaction_id):
    interaction = DrugInteraction.query.get_or_404(interaction_id)
    drugs = Drug.query.all()
    routes = RouteOfAdministration.query.all()

    if request.method == 'POST':
        # Form verilerini al
        interaction.drug1_id = request.form.get('drug1_id')
        interaction.drug2_id = request.form.get('drug2_id')
        interaction.route_id = request.form.get('route_id') if request.form.get('route_id') else None
        interaction.interaction_type = request.form.get('interaction_type')
        interaction.interaction_description = request.form.get('interaction_description')
        interaction.severity = request.form.get('severity')
        interaction.reference = request.form.get('reference')
        interaction.mechanism = request.form.get('mechanism')
        interaction.monitoring = request.form.get('monitoring')
        interaction.alternatives = request.form.get('alternatives')

        db.session.commit()
        return redirect(url_for('manage_interactions'))

    return render_template('update_interaction.html', 
                           interaction=interaction, 
                           drugs=drugs, 
                           routes=routes)

@app.route('/interactions/delete/<int:interaction_id>', methods=['POST'])
def delete_interaction(interaction_id):
    interaction = DrugInteraction.query.get_or_404(interaction_id)
    db.session.delete(interaction)
    db.session.commit()
    return redirect(url_for('manage_interactions'))





import joblib

# Model ve vektörleştiriciyi yükleyin
#model = joblib.load('interaction_model.pkl')
#vectorizer = joblib.load('interaction_vectorizer.pkl')


from flask import Flask, request, jsonify

@app.route('/predict_severity', methods=['GET', 'POST'])
def predict_severity():
    if request.method == 'POST':
        # Formdan etkileşim açıklamasını al
        description = request.form.get('interaction_description')

        # TF-IDF ile açıklamayı vektörleştir
        description_vector = vectorizer.transform([description])

        # Şiddet seviyesini tahmin et
        severity = model.predict(description_vector)[0]

        # Tahmini JSON olarak döndür
        return jsonify({"interaction_description": description, "predicted_severity": severity})

    # GET isteği için bir form döndür
    return render_template('predict_severity.html')


@app.route('/view_predictions')
def view_predictions():
    import pandas as pd
    # Tahmin edilen dosyayı oku
    data = pd.read_csv('predicted_interactions.csv')
    
    # Veriyi HTML tablosu olarak gönder
    return data.to_html()

@app.route('/check_predictions')
def check_predictions():
    interactions = DrugInteraction.query.all()
    results = [
        {
            "drug1_id": interaction.drug1_id,
            "drug2_id": interaction.drug2_id,
            "description": interaction.interaction_description,
            "predicted_severity": interaction.predicted_severity,
        }
        for interaction in interactions
    ]
    return jsonify(results)

@app.route('/api/interactions', methods=['POST'])
def get_interactions():
    draw = request.form.get('draw', type=int)
    start = request.form.get('start', type=int)
    length = request.form.get('length', type=int)
    search_value = request.form.get('search[value]', '')
    order_column_index = request.form.get('order[0][column]', type=int)
    order_direction = request.form.get('order[0][dir]', 'asc')

    column_map = {
        0: DrugInteraction.id,
        1: Drug.name_en,  # drug1_name
        2: Drug.name_en,  # drug2_name
        3: RouteOfAdministration.name,
        4: DrugInteraction.interaction_type,
        5: DrugInteraction.interaction_description,
        6: DrugInteraction.severity,
        7: DrugInteraction.predicted_severity,
        8: DrugInteraction.mechanism,
        9: DrugInteraction.monitoring,
        10: DrugInteraction.alternatives,
        11: DrugInteraction.reference,
        12: DrugInteraction.id,  # "actions" kolonu; ID baz alabiliriz
    }
    order_column = column_map.get(order_column_index, DrugInteraction.id)

    total_records = DrugInteraction.query.count()

    # JOIN
    query = DrugInteraction.query \
        .join(Drug, DrugInteraction.drug1_id == Drug.id) \
        .outerjoin(RouteOfAdministration, DrugInteraction.route_id == RouteOfAdministration.id)

    if search_value:
        query = query.filter(
            db.or_(
                Drug.name_en.ilike(f"%{search_value}%"),
                DrugInteraction.interaction_description.ilike(f"%{search_value}%")
            )
        )

    filtered_records = query.count()

    if order_direction == 'desc':
        query = query.order_by(order_column.desc())
    else:
        query = query.order_by(order_column.asc())

    interactions = query.offset(start).limit(length).all()

    data = []
    for interaction in interactions:
        # Edit ve Delete linklerini/formlarını HTML olarak döndürüyoruz
        edit_url = url_for('update_interaction', interaction_id=interaction.id)
        delete_url = url_for('delete_interaction', interaction_id=interaction.id)
        actions_html = f"""
        <div class="d-flex">
            <a href="{edit_url}" class="btn btn-sm btn-info me-1">Düzenle</a>
            <form method="POST" action="{delete_url}" onsubmit="return confirm('Silmek istediğinize emin misiniz?');">
                <button type="submit" class="btn btn-sm btn-danger">Sil</button>
            </form>
        </div>
        """

        data.append({
            "id": interaction.id,
            "drug1_name": interaction.drug1.name_en if interaction.drug1 else "N/A",
            "drug2_name": interaction.drug2.name_en if interaction.drug2 else "N/A",
            "route": interaction.route.name if interaction.route else "Genel",
            "interaction_type": interaction.interaction_type,
            "interaction_description": interaction.interaction_description,
            "severity": interaction.severity,
            "predicted_severity": interaction.predicted_severity or "Not Available",
            "mechanism": interaction.mechanism or "Not Provided",
            "monitoring": interaction.monitoring or "Not Provided",
            "alternatives": interaction.alternatives or "Not Provided",
            "reference": interaction.reference or "Not Provided",
            "actions": actions_html
        })

    return jsonify({
        "draw": draw,
        "recordsTotal": total_records,
        "recordsFiltered": filtered_records,
        "data": data
    })




# Model ve scaler'ı yükle
#model = joblib.load('lab_model.pkl')
#scaler = joblib.load('lab_scaler.pkl')

@app.route('/predict_disease', methods=['GET', 'POST'])
def predict_disease():
    if request.method == 'GET':
        # Tahmin formunu göstermek için
        return render_template('predict_disease.html')
    
    if request.method == 'POST':
        try:
            # Formdan verileri al
            data = request.json
            input_features = [
                data['Gender'],  # 1 for Male, 0 for Female
                data['Age'],
                data['Hemoglobin'],
                data['RBC'],
                data['WBC'],
                data['AST (aspartate aminotransferase)'],
                data['ALT (alanine aminotransferase)'],
                data['Cholestrol'],
                data['Spirometry'],
                data['Creatinine'],
                data['Glucose'],
                data['Lipase'],
                data['Troponin']
            ]

            # Veriyi ölçeklendir
            input_scaled = scaler.transform([input_features])

            # Tahmin yap
            predicted_disease = model.predict(input_scaled)[0]

            return jsonify({"predicted_disease": predicted_disease})
        except Exception as e:
            print(f"ERROR: {e}")
            return jsonify({"error": "An error occurred during prediction"}), 500



@app.route('/api/interactions/network', methods=['GET'])
def interaction_network():
    interactions = DrugInteraction.query.all()
    print("DEBUG: Interactions Retrieved ->", interactions)

    nodes = {}
    links = []

    for interaction in interactions:
        # Check if drug1 and drug2 relationships are valid
        print(f"DEBUG: Interaction -> {interaction.drug1_id}, {interaction.drug2_id}")
        if interaction.drug1 and interaction.drug2:
            # Add nodes
            if interaction.drug1_id not in nodes:
                nodes[interaction.drug1_id] = {"id": interaction.drug1_id, "name": interaction.drug1.name_en}
            if interaction.drug2_id not in nodes:
                nodes[interaction.drug2_id] = {"id": interaction.drug2_id, "name": interaction.drug2.name_en}

            # Add links
            links.append({
                "source": interaction.drug1_id,
                "target": interaction.drug2_id,
                "description": interaction.interaction_description,
                "severity": interaction.predicted_severity
            })

    print("DEBUG: Nodes ->", nodes)
    print("DEBUG: Links ->", links)
    return jsonify({"nodes": list(nodes.values()), "links": links})



@app.route('/interactions/network')
def network_view():
    return render_template('network.html')

@app.route('/api/interactions/network/filtered', methods=['POST'])
def filtered_interaction_network():
    filters = request.json
    severity_filter = filters.get('severity', None)
    mechanism_filter = filters.get('mechanism', None)

    print("DEBUG: Severity Filter ->", severity_filter)
    print("DEBUG: Mechanism Filter ->", mechanism_filter)

    query = DrugInteraction.query
    if severity_filter:
        query = query.filter(
            db.func.lower(DrugInteraction.predicted_severity).contains(severity_filter.lower())
        )
    if mechanism_filter:
        query = query.filter(DrugInteraction.mechanism.ilike(f"%{mechanism_filter}%"))

    interactions = query.all()
    print("DEBUG: Number of Interactions Found ->", len(interactions))

    nodes = {}
    links = []
    for interaction in interactions:
        if interaction.drug1_id not in nodes:
            nodes[interaction.drug1_id] = {"id": interaction.drug1_id, "name": interaction.drug1.name_en}
        if interaction.drug2_id not in nodes:
            nodes[interaction.drug2_id] = {"id": interaction.drug2_id, "name": interaction.drug2.name_en}

        links.append({
            "source": interaction.drug1_id,
            "target": interaction.drug2_id,
            "description": interaction.interaction_description,
            "severity": interaction.predicted_severity,
            "mechanism": interaction.mechanism
        })

    print("DEBUG: Nodes ->", nodes)
    print("DEBUG: Links ->", links)
    return jsonify({"nodes": list(nodes.values()), "links": links})



# Reseptör Yönetimi
@app.route('/receptors/manage', methods=['GET', 'POST'])
def manage_receptors():
    if request.method == 'POST':
        name = request.form.get('name')
        receptor_type = request.form.get('type')
        description = request.form.get('description')
        new_receptor = Receptor(name=name, type=receptor_type, description=description)
        db.session.add(new_receptor)
        db.session.commit()
        return redirect(url_for('manage_receptors'))
    receptors = Receptor.query.all()
    return render_template('manage_receptors.html', receptors=receptors)



from Bio.SeqUtils import molecular_weight
import re  # Regex ile geçerli karakterleri kontrol etmek için

# Moleküler Ağırlık Hesaplama Fonksiyonu
def calculate_molecular_weight(sequence):
    if not sequence:
        return "No Sequence Provided"
    valid_protein_chars = re.match("^[ARNDCEQGHILKMFPSTWYV]+$", sequence)
    if not valid_protein_chars:
        print(f"Invalid sequence for molecular weight calculation: {sequence}")
        return "Invalid Sequence"
    return molecular_weight(sequence, seq_type='protein')


#UNIPROT API'DEN RESEPTÖRLER İLE İLGİLİ BİLGİLERİ ALMA...
@app.route('/api/uniprot', methods=['GET', 'POST'])
def fetch_uniprot_data():
    if request.method == 'POST':
        receptor_name = request.form.get('receptor_name')

        if not receptor_name or receptor_name.strip() == "":
            return "Receptor name cannot be empty. Please provide a valid name."

        # UniProt API URL with binding site features
        url = f"https://rest.uniprot.org/uniprotkb/search?query={receptor_name}&fields=accession,protein_name,organism_name,gene_names,length,cc_subcellular_location,cc_function,sequence,xref_pdb,ft_binding"
        print(f"Request URL: {url}")
        response = requests.get(url)

        if response.status_code != 200:
            print(f"API Error: {response.text}")
            return f"Failed to fetch data from UniProt. Status Code: {response.status_code}"

        data = response.json()
        logging.info(f"UniProt API Response: {data}")
        results = data.get('results', [])
        if not results:
            print(f"No results found for receptor: {receptor_name}")
            return f"No UniProt entries found for {receptor_name}"

        for result in results:
            # Organism Name
            organism_name = result.get('organism', {}).get('scientificName', 'Unknown Organism')
            if organism_name.lower() != "homo sapiens":
                print(f"Skipping receptor as it is not from Homo sapiens: {organism_name}")
                continue

            # Accession (UniProt ID)
            accession = result.get('primaryAccession', 'Unknown Accession')

            # Protein Name
            protein_description = (
                result.get('proteinDescription', {})
                .get('recommendedName', {})
                .get('fullName', {})
                .get('value', 'Unknown Protein Name')
            )

            # Gene Names
            gene_primary = "Unknown Gene"
            genes = result.get('genes', [])
            if genes:
                gene_primary = genes[0].get('geneName', {}).get('value', 'Unknown Gene')

            # Sequence Length
            sequence_info = result.get('sequence', {})
            length = sequence_info.get('length', 'Unknown Length')

            # Molecular Weight
            molecular_weight_val = sequence_info.get('molWeight', 'Unknown Molecular Weight')

            # Subcellular Location
            subcellular_location = 'Unknown Location'
            for comment in result.get('comments', []):
                if comment.get('commentType') == 'SUBCELLULAR LOCATION':
                    subcell_locations = comment.get('subcellularLocations', [])
                    if subcell_locations:
                        subcellular_location = subcell_locations[0].get('location', {}).get('value', 'Unknown Location')

            # Function
            function = 'Unknown Function'
            for comment in result.get('comments', []):
                if comment.get('commentType') == 'FUNCTION':
                    function_texts = comment.get('texts', [])
                    if function_texts:
                        function = function_texts[0].get('value', 'Unknown Function')

            # Fetch PDB IDs
            pdb_ids = []
            for xref in result.get('uniProtKBCrossReferences', []):
                if xref.get('database') == 'PDB':
                    pdb_ids.append(xref.get('id'))

            # Fetch Binding Sites
            binding_sites = []
            features = result.get('features', [])
            print(f"Features for {accession}: {features}")
            for feature in features:
                if feature.get('type') == 'BINDING':
                    pos = feature['location']['start']['value']
                    ligand = feature.get('description', 'Unknown')
                    binding_sites.append({"residue": pos, "ligand": ligand})
            print(f"Binding sites found for {accession}: {binding_sites}")

            # Get 3D coordinates from PDB (if available)
            binding_site_coords = None
            if pdb_ids and binding_sites:
                pdb_id = pdb_ids[0]
                pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
                pdb_content = requests.get(pdb_url).text

                # Parse PDB
                parser = PDBParser(QUIET=True)  # Suppress warnings
                structure = parser.get_structure(pdb_id, io.StringIO(pdb_content))
                chain = list(structure[0].get_chains())[0]  # Assume first chain

                # Map first binding site to coordinates
                residue_num = binding_sites[0]["residue"]
                try:
                    residue = chain[residue_num]
                    ca = residue["CA"]  # Alpha carbon
                    binding_site_coords = {
                        "x": float(ca.get_coord()[0]),
                        "y": float(ca.get_coord()[1]),
                        "z": float(ca.get_coord()[2])
                    }
                    print(f"Binding site coords for residue {residue_num} in {pdb_id}: {binding_site_coords}")
                except KeyError:
                    print(f"Residue {residue_num} not found in PDB {pdb_id}")
                    binding_site_coords = {"x": 0, "y": 0, "z": 0}
            else:
                print(f"No PDB or binding sites for {accession}")
                binding_site_coords = {"x": 0, "y": 0, "z": 0}

            # Log the filtered receptor data
            print(f"""
                Fetched receptor (Homo sapiens):
                Accession: {accession},
                Protein Name: {protein_description},
                Organism: {organism_name},
                Length: {length},
                Gene: {gene_primary},
                Location: {subcellular_location},
                Function: {function},
                Molecular Weight: {molecular_weight_val},
                PDB IDs: {', '.join(pdb_ids)},
                Binding Site Coords: {binding_site_coords}
            """)

            # Save to database
            new_receptor = Receptor(
                name=protein_description,
                type="Protein",
                description=f"Organism: {organism_name}",
                molecular_weight=molecular_weight_val,
                length=length,
                gene_name=gene_primary,
                subcellular_location=subcellular_location,
                function=function,
                pdb_ids=",".join(pdb_ids),
                binding_site_x=binding_site_coords["x"],
                binding_site_y=binding_site_coords["y"],
                binding_site_z=binding_site_coords["z"]
            )
            db.session.add(new_receptor)

        db.session.commit()
        return "Receptor data successfully fetched and saved!"
    return render_template('fetch_uniprot.html')






# IUPHAR'dan Ligand - Reseptör etkileşmesini alıyoruz... 
@app.route('/api/iuphar', methods=['GET', 'POST'])
def fetch_iuphar_interactions():
    if request.method == 'GET':
        receptors = Receptor.query.filter(Receptor.iuphar_id.isnot(None)).all()  # IUPHAR ID atanmış reseptörler
        return render_template('fetch_iuphar.html', receptors=receptors)

    elif request.method == 'POST':
        try:
            # Kullanıcıdan seçilen IUPHAR ID'yi al
            receptor_id = request.form.get('receptor_id', '').strip()
            receptor = Receptor.query.get(receptor_id)

            if not receptor or not receptor.iuphar_id:
                return "Invalid receptor or missing IUPHAR ID. Please select a valid receptor.", 400

            target_id = receptor.iuphar_id
            interaction_url = f"https://www.guidetopharmacology.org/services/targets/{target_id}/interactions"
            response = requests.get(interaction_url)

            if response.status_code != 200:
                return f"Failed to fetch interactions from IUPHAR. Status Code: {response.status_code}", 500

            interactions = response.json()

            if not isinstance(interactions, list) or not interactions:
                return f"No interactions found for Target ID: {target_id}.", 404

            saved_interactions = 0
            for interaction in interactions:
                # **Target Species Kontrolü (Sadece Human)**
                species = interaction.get('targetSpecies', 'Unknown Species')
                if species.lower() != "human":
                    continue

                # **Ligand İsmini Normalize Et ve Eşleştir**
                ligand_name = interaction.get('ligandName', '')
                drug = Drug.query.filter(db.func.lower(Drug.name_en) == db.func.lower(ligand_name)).first()

                if not drug:
                    logging.warning(f"Ligand not found in database: {ligand_name}")
                    continue

                # **Affinity ve Diğer Bilgileri Kaydet**
                affinity_value = parse_affinity(interaction.get('affinity'))
                if affinity_value is None:
                    continue

                affinity_parameter = interaction.get('affinityParameter', 'N/A')
                interaction_type = interaction.get('type', 'N/A')
                mechanism = interaction.get('action', 'N/A')

                # **Daha Önce Kaydedilmiş mi?**
                existing_interaction = DrugReceptorInteraction.query.filter_by(
                    drug_id=drug.id,
                    receptor_id=receptor.id,
                    interaction_type=interaction_type,
                    affinity=affinity_value,
                    affinity_parameter=affinity_parameter
                ).first()

                if not existing_interaction:
                    new_interaction = DrugReceptorInteraction(
                        drug_id=drug.id,
                        receptor_id=receptor.id,
                        affinity=affinity_value,
                        interaction_type=interaction_type,
                        mechanism=mechanism,
                        affinity_parameter=affinity_parameter
                    )
                    db.session.add(new_interaction)
                    saved_interactions += 1

            # Veritabanına kaydet
            db.session.commit()
            return f"Successfully saved {saved_interactions} new interactions for Target ID {target_id}.", 200

        except Exception as e:
            logging.error(f"Error: {e}")
            return jsonify({"error": "An error occurred while processing interactions.", "details": str(e)}), 500

def parse_affinity(affinity_value):
    try:
        if isinstance(affinity_value, str):
            # Örneğin: "4.0 - 5.0" -> 4.0
            return float(affinity_value.split(" - ")[0]) if " - " in affinity_value else float(affinity_value)
        return float(affinity_value) if affinity_value else None
    except ValueError:
        logging.warning(f"Unable to parse affinity value: {affinity_value}")
        return None



@app.route('/receptors/map', methods=['GET', 'POST'])
def map_receptor_iuphar():
    if request.method == 'POST':
        receptor_id = request.form.get('receptor_id')
        iuphar_id = request.form.get('iuphar_id')

        # Veritabanında reseptörü bul
        receptor = Receptor.query.get(receptor_id)
        if receptor:
            receptor.iuphar_id = iuphar_id
            db.session.commit()
            return redirect(url_for('map_receptor_iuphar'))
        else:
            return "Receptor not found.", 404

    # GET metodu: Mevcut eşleşmeleri ve yeni eşleştirme formunu gösterir
    receptors = Receptor.query.all()
    return render_template('map_receptors.html', receptors=receptors)




@app.route('/interactions/drug-receptor', methods=['GET', 'POST'])
def manage_drug_receptor_interactions():
    if request.method == 'POST':
        # Interaction eklemek için işlemleri burada gerçekleştir
        drug_id = request.form.get('drug_id')
        receptor_id = request.form.get('receptor_id')
        affinity = request.form.get('affinity')
        interaction_type = request.form.get('interaction_type')
        mechanism = request.form.get('mechanism')
        # Veritabanına kaydetme işlemleri
        # ...
        return redirect(url_for('manage_drug_receptor_interactions'))

    interactions = DrugReceptorInteraction.query.all()
    drugs = Drug.query.all()
    receptors = Receptor.query.all()
    enriched_interactions = []

    for interaction in interactions:
        drug = Drug.query.get(interaction.drug_id)
        receptor = Receptor.query.get(interaction.receptor_id)
        if not drug or not receptor:
            print(f"Missing data for Interaction ID: {interaction.id}")
            continue
        enriched_interactions.append({
            "id": interaction.id,
            "drug_name": drug.name_tr or drug.name_en,
            "receptor_name": receptor.name,
            "affinity": interaction.affinity,
            "affinity_parameter": interaction.affinity_parameter or "N/A",
            "interaction_type": interaction.interaction_type,
            "mechanism": interaction.mechanism or "N/A",
        })

    return render_template('drug_receptor_interactions.html', interactions=enriched_interactions, drugs=drugs, receptors=receptors)

@app.route('/interactions/drug-receptor/delete/<int:id>', methods=['POST'])
def delete_drug_receptor_interaction(id):
    interaction = DrugReceptorInteraction.query.get(id)
    if interaction:
        db.session.delete(interaction)
        db.session.commit()
    return redirect(url_for('manage_drug_receptor_interactions'))

@app.route('/interactions/drug-receptor/edit/<int:id>', methods=['POST'])
def edit_drug_receptor_interaction(id):
    interaction = DrugReceptorInteraction.query.get(id)
    if interaction:
        interaction.affinity = request.form.get('affinity')
        interaction.interaction_type = request.form.get('interaction_type')
        interaction.mechanism = request.form.get('mechanism')
        db.session.commit()
    return redirect(url_for('manage_drug_receptor_interactions'))



@app.route('/api/receptor-ligand-dashboard', methods=['GET'])
def receptor_ligand_dashboard():
    try:
        draw = request.args.get('draw', type=int, default=1)
        start = request.args.get('start', 0, type=int)
        length = request.args.get('length', 10, type=int)
        search_value = request.args.get('search[value]', '', type=str)
        table_id = request.args.get('table_id', '')

        order_column_index = request.args.get('order[0][column]', type=int, default=0)
        order_direction = request.args.get('order[0][dir]', default='asc')

        if table_id == "interaction-table":
            query = DrugReceptorInteraction.query.join(
                Receptor, DrugReceptorInteraction.receptor_id == Receptor.id
            ).join(
                Drug, DrugReceptorInteraction.drug_id == Drug.id
            )

            if search_value:
                query = query.filter(
                    db.or_(
                        Drug.name_en.ilike(f"%{search_value}%"),
                        Receptor.name.ilike(f"%{search_value}%"),
                        DrugReceptorInteraction.interaction_type.ilike(f"%{search_value}%"),
                        DrugReceptorInteraction.mechanism.ilike(f"%{search_value}%")
                    )
                )

            columns = [
                Drug.name_en,
                Receptor.name,
                DrugReceptorInteraction.affinity,
                DrugReceptorInteraction.affinity_parameter,
                DrugReceptorInteraction.interaction_type,
                DrugReceptorInteraction.mechanism
            ]

            if 0 <= order_column_index < len(columns):
                column = columns[order_column_index]
                query = query.order_by(db.desc(column)) if order_direction == 'desc' else query.order_by(column)

            total_count = DrugReceptorInteraction.query.count()
            filtered_count = query.count()
            paginated_items = query.offset(start).limit(length).all()

            data = [
                {
                    "Ligand": interaction.drug.name_en if interaction.drug else f"Unknown Ligand {interaction.drug_id}",
                    "Receptor": interaction.receptor.name if interaction.receptor else f"Unknown Receptor {interaction.receptor_id}",
                    "Affinity": interaction.affinity or "N/A",
                    "Affinity Parameter": interaction.affinity_parameter or "N/A",
                    "Interaction Type": interaction.interaction_type or "N/A",
                    "Mechanism": interaction.mechanism or "N/A"
                }
                for interaction in paginated_items
            ]

        elif table_id == "receptor-table":
            query = Receptor.query

            if search_value:
                query = query.filter(
                    db.or_(
                        Receptor.name.ilike(f"%{search_value}%"),
                        Receptor.type.ilike(f"%{search_value}%"),
                        Receptor.gene_name.ilike(f"%{search_value}%")
                    )
                )

            columns = [
                Receptor.name,
                Receptor.type,
                Receptor.molecular_weight,
                Receptor.length,
                Receptor.gene_name,
                Receptor.subcellular_location,
                Receptor.function
            ]

            if 0 <= order_column_index < len(columns):
                column = columns[order_column_index]
                query = query.order_by(db.desc(column)) if order_direction == 'desc' else query.order_by(column)

            total_count = Receptor.query.count()
            filtered_count = query.count()
            paginated_items = query.offset(start).limit(length).all()

            data = [
                {
                    "Name": receptor.name or "Unknown",
                    "Type": receptor.type or "Unknown",
                    "Molecular Weight": receptor.molecular_weight or "N/A",
                    "Length": receptor.length or "N/A",
                    "Gene Name": receptor.gene_name or "N/A",
                    "Localization": receptor.subcellular_location or "N/A",
                    "Function": receptor.function or "N/A"
                }
                for receptor in paginated_items
            ]

        else:
            return jsonify({"error": "Invalid table_id"}), 400

        return jsonify({
            "draw": draw,
            "recordsTotal": total_count,
            "recordsFiltered": filtered_count,
            "data": data
        })

    except Exception as e:
        logging.error(f"Error: {e}")
        return jsonify({"error": "Server Error", "message": str(e)}), 500



# Dashboard View
@app.route('/receptor-ligand-dashboard', methods=['GET'])
@login_required
def receptor_ligand_dashboard_view():
    return render_template('receptor_ligand_dashboard.html')


# Helper Function: Fetch Receptor Name by Target ID
def fetch_receptor_name(target_id):
    try:
        target_url = f"https://www.guidetopharmacology.org/services/targets/{target_id}"
        logging.info(f"Fetching receptor name for Target ID: {target_id}")
        response = requests.get(target_url)

        if response.status_code == 200:
            data = response.json()
            receptor_name = data.get('name', f"Unknown Target {target_id}")
            logging.info(f"Fetched receptor name: {receptor_name} for Target ID: {target_id}")
            return receptor_name
        else:
            logging.error(f"Failed to fetch receptor name: {response.status_code} - {response.text}")
            return f"Target {target_id}"
    except Exception as e:
        logging.error(f"Failed to fetch receptor name for Target ID {target_id}: {e}")
        return f"Target {target_id}"




@app.route('/api/affinity-data', methods=['GET'])
def affinity_data():
    interactions = DrugReceptorInteraction.query.all()
    chart_data = {
        "labels": [],
        "values": []
    }

    for interaction in interactions:
        receptor_name = interaction.receptor.name if interaction.receptor else "Unknown Receptor"
        ligand_name = interaction.drug.name_en if interaction.drug else "Unknown Ligand"
        chart_data["labels"].append(f"{ligand_name} ({receptor_name})")
        chart_data["values"].append(interaction.affinity or 0)

    return jsonify(chart_data)

@app.route('/api/interaction-type-distribution', methods=['GET'])
def interaction_type_distribution():
    interactions = DrugReceptorInteraction.query.with_entities(
        DrugReceptorInteraction.interaction_type,
        db.func.count(DrugReceptorInteraction.id)
    ).group_by(DrugReceptorInteraction.interaction_type).all()

    chart_data = {
        "labels": [interaction[0] for interaction in interactions],
        "values": [interaction[1] for interaction in interactions]
    }
    return jsonify(chart_data)



@app.route('/api/side_effects', methods=['GET'])
def search_side_effects():
    search = request.args.get('search', '').strip()
    limit = request.args.get('limit', 10, type=int)  # Varsayılan olarak 10 sonuç döner

    query = SideEffect.query.filter(
        SideEffect.name_en.ilike(f"%{search}%") | SideEffect.name_tr.ilike(f"%{search}%")
    ) if search else SideEffect.query

    query = query.limit(limit)
    results = query.all()

    return jsonify([
        {"id": side_effect.id, "text": f"{side_effect.name_en} ({side_effect.name_tr or 'N/A'})"}
        for side_effect in results
    ])



@app.route('/api/targets', methods=['GET'])
def get_targets():
    query = request.args.get('query', '')
    targets = Target.query.filter(Target.name_en.ilike(f"%{query}%")).all()
    target_list = [{'id': target.id, 'name_en': target.name_en} for target in targets]
    return jsonify(target_list)


@app.route('/api/indications', methods=['GET'])
def get_indications():
    search = request.args.get('search', '').strip()  # Get the search term
    limit = request.args.get('limit', 10, type=int)  # Limit the number of results
    page = request.args.get('page', 1, type=int)  # Get the current page

    # Filter by search term if provided
    if search:
        query = Indication.query.filter(
            db.or_(
                Indication.name_en.ilike(f"%{search}%"),  # Search in English name
                Indication.name_tr.ilike(f"%{search}%"),  # Search in Turkish name
                Indication.synonyms.ilike(f"%{search}%")  # Search in synonyms
            )
        )
    else:
        query = Indication.query

    # Paginate results
    paginated_query = query.paginate(page=page, per_page=limit)

    # Prepare the results for the dropdown
    results = [
        {
            "id": indication.id,
            "text": f"{indication.name_en} ({indication.name_tr})" if indication.name_tr else indication.name_en
        }
        for indication in paginated_query.items
    ]

    # Return the results to the frontend
    return jsonify({
        "results": results,  # The list of items for the dropdown
        "has_next": paginated_query.has_next  # If there are more pages
    })




@app.route('/upload_targets', methods=['GET', 'POST'])
def upload_targets():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            return "No file uploaded", 400

        # File format check
        if file.filename.endswith('.csv'):
            import pandas as pd
            data = pd.read_csv(file)  # Read the CSV file
        elif file.filename.endswith('.xlsx'):
            import pandas as pd
            data = pd.read_excel(file)  # Read the Excel file
        else:
            return "Unsupported file format. Please upload a CSV or Excel file.", 400

        # Check if the required column exists
        if 'name_en' not in data.columns:
            return "The file must contain a column named 'name_en'.", 400

        # Insert targets into the database
        for _, row in data.iterrows():
            new_target = Target(
                name_en=row['name_en'],
                name_tr=row['name_en']  # Default Turkish name to English name
            )
            db.session.add(new_target)
        db.session.commit()
        return "Targets uploaded successfully!", 200

    return render_template('upload_targets.html')  # Create a simple HTML form for uploading



# PHARMGKB KODLARI
# Helper Functions
def parse_date(date_str, row_id=""):
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        logger.warning(f"Invalid date format '{date_str}' for row {row_id}")
        return None

def save_uploaded_file(file_storage):
    if not file_storage.filename.endswith('.tsv'):
        raise ValueError("Only .tsv files are supported")
    filename = secure_filename(file_storage.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    file_storage.save(file_path)
    return file_path

def check_drugs_exist(drug_names, ca_id=""):
    missing = set()
    found_drugs = []
    for dname in drug_names:
        found = db.session.query(Drug).filter(
            (Drug.name_en.ilike(dname)) | 
            (Drug.name_tr.ilike(dname)) | 
            (Drug.pharmgkb_id.ilike(dname))
        ).first()
        if found:
            found_drugs.append(found)
        else:
            missing.add(dname)
            logger.warning(f"DRUG {dname} not found in database for CA {ca_id}, will skip linking this drug")
    logger.debug(f"Checked {len(drug_names)} drugs for CA {ca_id}, found {len(found_drugs)}, missing {len(missing)}")
    return found_drugs, missing

def safe_split(text, delimiter=';'):
    return [t.strip() for t in (text or '').split(delimiter) if t.strip()] if text else []

# ETL Functions
def load_clinical_annotations_tsv(filepath):
    missing_drugs = set()
    inserted = 0
    skipped = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            ca_id = row.get("Clinical Annotation ID", "").strip()
            if not ca_id:
                logger.warning(f"Skipping row with empty Clinical Annotation ID")
                skipped += 1
                continue
            raw_drugs = row.get("Drug(s)", "").strip()
            drug_names = [d.strip().lower() for d in raw_drugs.split(';') if d.strip()]
            found_drugs, missing = check_drugs_exist(drug_names, ca_id)
            missing_drugs.update(missing)
            # Process row even if no drugs are found, as Drug(s) can be empty
            try:
                # Get or create ClinicalAnnotation
                ca = db.session.get(ClinicalAnnotation, ca_id)
                if not ca:
                    ca = ClinicalAnnotation(clinical_annotation_id=ca_id)
                    db.session.add(ca)
                
                # Update ClinicalAnnotation fields
                ca.level_of_evidence = row.get("Level of Evidence", "")
                ca.level_override = row.get("Level Override", "")
                ca.level_modifiers = row.get("Level Modifiers", "")
                ca.score = float(row["Score"]) if row.get("Score") else None
                ca.pmid_count = int(row["PMID Count"]) if row.get("PMID Count") else None
                ca.evidence_count = int(row["Evidence Count"]) if row.get("Evidence Count") else None
                ca.phenotype_category = row.get("Phenotype Category", "")
                ca.url = row.get("URL", "")
                ca.latest_history_date = parse_date(row.get("Latest History Date (YYYY-MM-DD)", ""), ca_id)
                ca.specialty_population = row.get("Specialty Population", "")

                # Clear existing relationships with no_autoflush to prevent premature flush
                with db.session.no_autoflush:
                    db.session.query(ClinicalAnnotationDrug).filter_by(clinical_annotation_id=ca_id).delete()
                    db.session.query(ClinicalAnnotationGene).filter_by(clinical_annotation_id=ca_id).delete()
                    db.session.query(ClinicalAnnotationPhenotype).filter_by(clinical_annotation_id=ca_id).delete()
                    db.session.query(ClinicalAnnotationVariant).filter_by(clinical_annotation_id=ca_id).delete()

                # Add drugs (only those found)
                for drug in found_drugs:
                    ca_drug = ClinicalAnnotationDrug(clinical_annotation_id=ca_id, drug_id=drug.id)
                    db.session.add(ca_drug)

                # Add genes
                raw_genes = row.get("Gene", "").strip()
                for gene_name in safe_split(raw_genes):
                    gene = db.session.query(Gene).filter_by(gene_symbol=gene_name).first()
                    if not gene:
                        gene = Gene(gene_id=f"PA{gene_name}", gene_symbol=gene_name)
                        db.session.add(gene)
                        db.session.flush()  # Ensure gene.id is available
                    ca_gene = ClinicalAnnotationGene(clinical_annotation_id=ca_id, gene_id=gene.gene_id)
                    db.session.add(ca_gene)

                # Add phenotypes
                raw_phenotypes = row.get("Phenotype(s)", "").strip()
                for pheno_name in safe_split(raw_phenotypes):
                    pheno = db.session.query(Phenotype).filter_by(name=pheno_name).first()
                    if not pheno:
                        pheno = Phenotype(name=pheno_name)
                        db.session.add(pheno)
                        db.session.flush()  # Ensure phenotype.id is available
                    ca_pheno = ClinicalAnnotationPhenotype(clinical_annotation_id=ca_id, phenotype_id=pheno.id)
                    db.session.add(ca_pheno)

                # Add variants (only if non-empty)
                raw_variants = row.get("Variant/Haplotypes", "").strip()
                if raw_variants:  # Skip if empty to avoid invalid insertions
                    for var_name in safe_split(raw_variants):
                        var = db.session.query(Variant).filter_by(name=var_name).first()
                        if not var:
                            var = Variant(name=var_name, pharmgkb_id=None)
                            db.session.add(var)
                            db.session.flush()  # Ensure variant.id is available
                        ca_var = ClinicalAnnotationVariant(clinical_annotation_id=ca_id, variant_id=var.id)
                        db.session.add(ca_var)

                inserted += 1
                if inserted % 100 == 0:
                    db.session.commit()
            except (ValueError, KeyError, sqlalchemy.exc.IntegrityError) as e:
                logger.error(f"Row CA {ca_id} failed: {str(e)}")
                db.session.rollback()
                skipped += 1
        db.session.commit()
    if missing_drugs:
        logger.warning(f"[clinical_annotations] Missing drugs: {', '.join(missing_drugs)}")
    logger.info(f"[clinical_annotations] Inserted: {inserted}, Skipped: {skipped}")
    return inserted, skipped, missing_drugs

def load_clinical_ann_history_tsv(filepath):
    inserted = 0
    skipped = 0
    missing_drugs = set()  # No drugs involved
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            ca_id = row.get("Clinical Annotation ID", "").strip()
            if not ca_id or not db.session.get(ClinicalAnnotation, ca_id):
                logger.warning(f"Skipping history for non-existent Clinical Annotation ID: {ca_id}")
                skipped += 1
                continue
            try:
                hist = ClinicalAnnHistory(
                    clinical_annotation_id=ca_id,
                    date=parse_date(row.get("Date (YYYY-MM-DD)", ""), ca_id),
                    type=row.get("Type", ""),
                    comment=row.get("Comment", "")
                )
                db.session.add(hist)
                inserted += 1
                if inserted % 100 == 0:
                    db.session.commit()
            except (ValueError, sqlalchemy.exc.IntegrityError) as e:
                logger.error(f"History row for CA {ca_id} failed: {str(e)}")
                db.session.rollback()
                skipped += 1
        db.session.commit()
    logger.info(f"[clinical_ann_history] Inserted: {inserted}, Skipped: {skipped}")
    return inserted, skipped, missing_drugs

def load_clinical_ann_alleles_tsv(filepath):
    inserted = 0
    skipped = 0
    missing_drugs = set()  # No drugs involved
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            ca_id = row.get("Clinical Annotation ID", "").strip()
            genotype = row.get("Genotype/Allele", "").strip()
            if not ca_id or not db.session.get(ClinicalAnnotation, ca_id):
                logger.warning(f"Skipping allele for non-existent Clinical Annotation ID: {ca_id}")
                skipped += 1
                continue
            if not genotype:
                logger.warning(f"Skipping allele with empty Genotype/Allele for CA {ca_id}")
                skipped += 1
                continue
            try:
                allele = ClinicalAnnAllele(
                    clinical_annotation_id=ca_id,
                    genotype_allele=genotype,
                    annotation_text=row.get("Annotation Text", ""),
                    allele_function=row.get("Allele Function", "")
                )
                db.session.add(allele)
                inserted += 1
                if inserted % 100 == 0:
                    db.session.commit()
            except (ValueError, sqlalchemy.exc.IntegrityError) as e:
                logger.error(f"Allele row for CA {ca_id} failed: {str(e)}")
                db.session.rollback()
                skipped += 1
        db.session.commit()
    logger.info(f"[clinical_ann_alleles] Inserted: {inserted}, Skipped: {skipped}")
    return inserted, skipped, missing_drugs

def load_clinical_ann_evidence_tsv(filepath):
    inserted = 0
    skipped = 0
    missing_drugs = set()  # No drugs involved
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            ev_id = row.get("Evidence ID", "").strip()
            ca_id = row.get("Clinical Annotation ID", "").strip()
            pmid = row.get("PMID", "").strip()
            if not ev_id:
                logger.warning(f"Skipping evidence with empty Evidence ID")
                skipped += 1
                continue
            if not ca_id or not db.session.get(ClinicalAnnotation, ca_id):
                logger.warning(f"Skipping evidence for non-existent Clinical Annotation ID: {ca_id}")
                skipped += 1
                continue
            try:
                # Get or create ClinicalAnnEvidence
                ev = db.session.get(ClinicalAnnEvidence, ev_id)
                if not ev:
                    ev = ClinicalAnnEvidence(evidence_id=ev_id)
                    db.session.add(ev)
                
                ev.clinical_annotation_id = ca_id
                ev.evidence_type = row.get("Evidence Type", "")
                ev.evidence_url = row.get("Evidence URL", "")
                ev.summary = row.get("Summary", "")
                ev.score = float(row["Score"]) if row.get("Score") else None

                # Handle Publication
                if pmid:
                    pub = db.session.get(Publication, pmid)
                    if not pub:
                        pub = Publication(pmid=pmid, title="Unknown", year=None, journal=None)
                        db.session.add(pub)
                        db.session.flush()  # Ensure publication.pmid is available
                    # Link via ClinicalAnnEvidencePublication
                    ev_pub = db.session.query(ClinicalAnnEvidencePublication).filter_by(
                        evidence_id=ev_id, pmid=pmid
                    ).first()
                    if not ev_pub:
                        ev_pub = ClinicalAnnEvidencePublication(evidence_id=ev_id, pmid=pmid)
                        db.session.add(ev_pub)

                inserted += 1
                if inserted % 100 == 0:
                    db.session.commit()
            except (ValueError, KeyError, sqlalchemy.exc.IntegrityError) as e:
                logger.error(f"Evidence {ev_id} failed: {str(e)}")
                db.session.rollback()
                skipped += 1
        db.session.commit()
    logger.info(f"[clinical_ann_evidence] Inserted: {inserted}, Skipped: {skipped}")
    return inserted, skipped, missing_drugs

# Route
@app.route("/upload_clinical_annotations", methods=["GET", "POST"])
def upload_clinical_annotations():
    with app.app_context():
        if request.method == "GET":
            return render_template("upload_clinical_annotations.html")
        
        files = {
            "clinical_annotations": load_clinical_annotations_tsv,
            "clinical_ann_history": load_clinical_ann_history_tsv,
            "clinical_ann_alleles": load_clinical_ann_alleles_tsv,
            "clinical_ann_evidence": load_clinical_ann_evidence_tsv
        }
        for key, func in files.items():
            file = request.files.get(key)
            if file and file.filename:
                try:
                    path = save_uploaded_file(file)
                    inserted, skipped, missing_drugs = func(path)
                    logger.info(f"Processed {key}.tsv: Inserted={inserted}, Skipped={skipped}, Missing drugs={missing_drugs}")
                    flash(f"{key}.tsv processed successfully! Inserted: {inserted}, Skipped: {skipped}, Missing drugs: {', '.join(missing_drugs) if missing_drugs else 'None'}", "success")
                    os.remove(path)
                except Exception as e:
                    logger.error(f"Error processing {key}.tsv: {str(e)}")
                    flash(f"Error processing {key}.tsv: {str(e)}", "danger")
                    db.session.rollback()
                    if os.path.exists(path):
                        os.remove(path)
            else:
                logger.warning(f"No file uploaded for {key}")
                flash(f"No file uploaded for {key}", "warning")
        return redirect(url_for("upload_clinical_annotations"))
# clinical annotations için yükleme sonu

#variant annotations klasörü için yükleme route'ları
# ETL Functions for Variant Annotations
# ETL Functions for Variant Annotations
def load_study_parameters_tsv(filepath):
    inserted = 0
    skipped = 0
    missing_drugs = set()  # No drugs involved
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            sp_id = row.get("Study Parameters ID", "").strip()
            va_id = row.get("Variant Annotation ID", "").strip()
            if not sp_id:
                logger.warning(f"Skipping row with empty Study Parameters ID")
                skipped += 1
                continue
            if not va_id or not db.session.get(VariantAnnotation, va_id):
                logger.warning(f"Skipping row with invalid Variant Annotation ID: {va_id}")
                skipped += 1
                continue
            try:
                sp = db.session.get(StudyParameters, sp_id)
                if not sp:
                    sp = StudyParameters(study_parameters_id=sp_id)
                    db.session.add(sp)
                sp.variant_annotation_id = va_id
                sp.study_type = row.get("Study Type", "")
                sp.study_cases = int(row["Study Cases"]) if row.get("Study Cases") else None
                sp.study_controls = int(row["Study Controls"]) if row.get("Study Controls") else None
                sp.characteristics = row.get("Characteristics", "")
                sp.characteristics_type = row.get("Characteristics Type", "")
                sp.frequency_in_cases = float(row["Frequency In Cases"]) if row.get("Frequency In Cases") else None
                sp.allele_of_frequency_in_cases = row.get("Allele Of Frequency In Cases", "")
                sp.frequency_in_controls = float(row["Frequency In Controls"]) if row.get("Frequency In Controls") else None
                sp.allele_of_frequency_in_controls = row.get("Allele Of Frequency In Controls", "")
                sp.p_value = row.get("P Value", "")
                sp.ratio_stat_type = row.get("Ratio Stat Type", "")
                sp.ratio_stat = float(row["Ratio Stat"]) if row.get("Ratio Stat") else None
                sp.confidence_interval_start = float(row["Confidence Interval Start"]) if row.get("Confidence Interval Start") else None
                sp.confidence_interval_stop = float(row["Confidence Interval Stop"]) if row.get("Confidence Interval Stop") else None
                sp.biogeographical_groups = row.get("Biogeographical Groups", "")
                inserted += 1
                if inserted % 100 == 0:
                    db.session.commit()
            except (ValueError, KeyError, sqlalchemy.exc.IntegrityError) as e:
                logger.error(f"Study Parameters {sp_id} failed: {str(e)}")
                db.session.rollback()
                skipped += 1
        db.session.commit()
    logger.info(f"[study_parameters] Inserted: {inserted}, Skipped: {skipped}")
    return inserted, skipped, missing_drugs

def load_var_fa_ann_tsv(filepath):
    missing_drugs = set()
    inserted = 0
    skipped = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            va_id = row.get("Variant Annotation ID", "").strip()
            if not va_id:
                logger.warning(f"Skipping row with empty Variant Annotation ID")
                skipped += 1
                continue
            raw_drugs = row.get("Drug(s)", "").strip()
            drug_names = [d.strip().lower() for d in raw_drugs.split(';') if d.strip()]
            found_drugs, missing = check_drugs_exist(drug_names, va_id)
            missing_drugs.update(missing)
            try:
                # Ensure VariantAnnotation exists
                va = db.session.get(VariantAnnotation, va_id)
                if not va:
                    va = VariantAnnotation(variant_annotation_id=va_id)
                    db.session.add(va)
                    db.session.flush()
                
                # Update VariantFAAnn
                vfa = db.session.get(VariantFAAnn, va_id)
                if not vfa:
                    vfa = VariantFAAnn(variant_annotation_id=va_id)
                    db.session.add(vfa)
                vfa.significance = row.get("Significance", "")
                vfa.notes = row.get("Notes", "")
                vfa.sentence = row.get("Sentence", "")
                vfa.alleles = row.get("Alleles", "")
                vfa.specialty_population = row.get("Specialty Population", "")
                vfa.assay_type = row.get("Assay type", "")
                vfa.metabolizer_types = row.get("Metabolizer types", "")
                vfa.is_plural = row.get("isPlural", "")
                vfa.is_associated = row.get("Is/Is Not associated", "")
                vfa.direction_of_effect = row.get("Direction of effect", "")
                vfa.functional_terms = row.get("Functional terms", "")
                vfa.gene_product = row.get("Gene/gene product", "")
                vfa.when_treated_with = row.get("When treated with/exposed to/when assayed with", "")
                vfa.multiple_drugs = row.get("Multiple drugs And/or", "")
                vfa.cell_type = row.get("Cell type", "")
                vfa.comparison_alleles = row.get("Comparison Allele(s) or Genotype(s)", "")
                vfa.comparison_metabolizer_types = row.get("Comparison Metabolizer types", "")

                # Clear and update VariantAnnotationDrug
                with db.session.no_autoflush:
                    db.session.query(VariantAnnotationDrug).filter_by(variant_annotation_id=va_id).delete()
                for drug in found_drugs:
                    va_drug = VariantAnnotationDrug(variant_annotation_id=va_id, drug_id=drug.id)
                    db.session.add(va_drug)

                # Add genes to VariantAnnotationGene
                with db.session.no_autoflush:
                    db.session.query(VariantAnnotationGene).filter_by(variant_annotation_id=va_id).delete()
                raw_genes = row.get("Gene", "").strip()
                for gene_name in safe_split(raw_genes):
                    gene = db.session.query(Gene).filter_by(gene_symbol=gene_name).first()
                    if not gene:
                        gene = Gene(gene_id=f"PA{gene_name}", gene_symbol=gene_name)
                        db.session.add(gene)
                        db.session.flush()
                    va_gene = VariantAnnotationGene(variant_annotation_id=va_id, gene_id=gene.gene_id)
                    db.session.add(va_gene)

                # Add variants to VariantAnnotationVariant
                with db.session.no_autoflush:
                    db.session.query(VariantAnnotationVariant).filter_by(variant_annotation_id=va_id).delete()
                raw_variants = row.get("Variant/Haplotypes", "").strip()
                if raw_variants:
                    for var_name in safe_split(raw_variants):
                        var = db.session.query(Variant).filter_by(name=var_name).first()
                        if not var:
                            var = Variant(name=var_name, pharmgkb_id=None)
                            db.session.add(var)
                            db.session.flush()
                        va_var = VariantAnnotationVariant(variant_annotation_id=va_id, variant_id=var.id)
                        db.session.add(va_var)

                inserted += 1
                if inserted % 100 == 0:
                    db.session.commit()
            except (ValueError, KeyError, sqlalchemy.exc.IntegrityError) as e:
                logger.error(f"Variant FA Ann {va_id} failed: {str(e)}")
                db.session.rollback()
                skipped += 1
        db.session.commit()
    if missing_drugs:
        logger.warning(f"[var_fa_ann] Missing drugs: {', '.join(missing_drugs)}")
    logger.info(f"[var_fa_ann] Inserted: {inserted}, Skipped: {skipped}")
    return inserted, skipped, missing_drugs

def load_var_drug_ann_tsv(filepath):
    missing_drugs = set()
    inserted = 0
    skipped = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            va_id = row.get("Variant Annotation ID", "").strip()
            if not va_id:
                logger.warning(f"Skipping row with empty Variant Annotation ID")
                skipped += 1
                continue
            raw_drugs = row.get("Drug(s)", "").strip()
            drug_names = [d.strip().lower() for d in raw_drugs.split(';') if d.strip()]
            found_drugs, missing = check_drugs_exist(drug_names, va_id)
            missing_drugs.update(missing)
            # Process row even if no drugs are found, as Drug(s) can be empty
            try:
                # Ensure VariantAnnotation exists
                va = db.session.get(VariantAnnotation, va_id)
                if not va:
                    va = VariantAnnotation(variant_annotation_id=va_id)
                    db.session.add(va)
                    db.session.flush()
                
                # Update VariantDrugAnn
                vda = db.session.get(VariantDrugAnn, va_id)
                if not vda:
                    vda = VariantDrugAnn(variant_annotation_id=va_id)
                    db.session.add(vda)
                vda.significance = row.get("Significance", "")
                vda.notes = row.get("Notes", "")
                vda.sentence = row.get("Sentence", "")
                vda.alleles = row.get("Alleles", "")
                vda.specialty_population = row.get("Specialty Population", "")
                vda.metabolizer_types = row.get("Metabolizer types", "")
                vda.is_plural = row.get("isPlural", "")
                vda.is_associated = row.get("Is/Is Not associated", "")
                vda.direction_of_effect = row.get("Direction of effect", "")
                vda.pd_pk_terms = row.get("PD/PK terms", "")
                vda.multiple_drugs = row.get("Multiple drugs And/or", "")
                vda.population_types = row.get("Population types", "")
                vda.population_phenotypes_diseases = row.get("Population Phenotypes or diseases", "")
                vda.multiple_phenotypes_diseases = row.get("Multiple phenotypes or diseases And/or", "")
                vda.comparison_alleles = row.get("Comparison Allele(s) or Genotype(s)", "")
                vda.comparison_metabolizer_types = row.get("Comparison Metabolizer types", "")

                # Clear and update VariantAnnotationDrug
                with db.session.no_autoflush:
                    db.session.query(VariantAnnotationDrug).filter_by(variant_annotation_id=va_id).delete()
                for drug in found_drugs:
                    va_drug = VariantAnnotationDrug(variant_annotation_id=va_id, drug_id=drug.id)
                    db.session.add(va_drug)

                inserted += 1
                if inserted % 100 == 0:
                    db.session.commit()
            except (ValueError, KeyError, sqlalchemy.exc.IntegrityError) as e:
                logger.error(f"Variant Drug Ann {va_id} failed: {str(e)}")
                db.session.rollback()
                skipped += 1
        db.session.commit()
    if missing_drugs:
        logger.warning(f"[var_drug_ann] Missing drugs: {', '.join(missing_drugs)}")
    logger.info(f"[var_drug_ann] Inserted: {inserted}, Skipped: {skipped}")
    return inserted, skipped, missing_drugs

def load_var_pheno_ann_tsv(filepath):
    missing_drugs = set()
    inserted = 0
    skipped = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            va_id = row.get("Variant Annotation ID", "").strip()
            if not va_id:
                logger.warning(f"Skipping row with empty Variant Annotation ID")
                skipped += 1
                continue
            raw_drugs = row.get("Drug(s)", "").strip()
            drug_names = [d.strip().lower() for d in raw_drugs.split(';') if d.strip()]
            found_drugs, missing = check_drugs_exist(drug_names, va_id)
            missing_drugs.update(missing)
            # Process row even if no drugs are found, as Drug(s) can be empty
            try:
                # Ensure VariantAnnotation exists
                va = db.session.get(VariantAnnotation, va_id)
                if not va:
                    va = VariantAnnotation(variant_annotation_id=va_id)
                    db.session.add(va)
                    db.session.flush()
                
                # Update VariantPhenoAnn
                vpa = db.session.get(VariantPhenoAnn, va_id)
                if not vpa:
                    vpa = VariantPhenoAnn(variant_annotation_id=va_id)
                    db.session.add(vpa)
                vpa.significance = row.get("Significance", "")
                vpa.notes = row.get("Notes", "")
                vpa.sentence = row.get("Sentence", "")
                vpa.alleles = row.get("Alleles", "")
                vpa.specialty_population = row.get("Specialty Population", "")
                vpa.metabolizer_types = row.get("Metabolizer types", "")
                vpa.is_plural = row.get("isPlural", "")
                vpa.is_associated = row.get("Is/Is Not associated", "")
                vpa.direction_of_effect = row.get("Direction of effect", "")
                vpa.side_effect_efficacy_other = row.get("Side effect/efficacy/other", "")
                vpa.phenotype = row.get("Phenotype", "")
                vpa.multiple_phenotypes = row.get("Multiple phenotypes", "")
                vpa.when_treated_with = row.get("When treated with/exposed to/when assayed with", "")
                vpa.multiple_drugs = row.get("Multiple drugs And/or", "")
                vpa.population_types = row.get("Population types", "")
                vpa.population_phenotypes_diseases = row.get("Population Phenotypes or diseases", "")
                vpa.multiple_phenotypes_diseases = row.get("Multiple phenotypes or diseases And/or", "")
                vpa.comparison_alleles = row.get("Comparison Allele(s) or Genotype(s)", "")
                vpa.comparison_metabolizer_types = row.get("Comparison Metabolizer types", "")

                # Clear and update VariantAnnotationDrug
                with db.session.no_autoflush:
                    db.session.query(VariantAnnotationDrug).filter_by(variant_annotation_id=va_id).delete()
                for drug in found_drugs:
                    va_drug = VariantAnnotationDrug(variant_annotation_id=va_id, drug_id=drug.id)
                    db.session.add(va_drug)

                inserted += 1
                if inserted % 100 == 0:
                    db.session.commit()
            except (ValueError, KeyError, sqlalchemy.exc.IntegrityError) as e:
                logger.error(f"Variant Pheno Ann {va_id} failed: {str(e)}")
                db.session.rollback()
                skipped += 1
        db.session.commit()
    if missing_drugs:
        logger.warning(f"[var_pheno_ann] Missing drugs: {', '.join(missing_drugs)}")
    logger.info(f"[var_pheno_ann] Inserted: {inserted}, Skipped: {skipped}")
    return inserted, skipped, missing_drugs

# Route for Variant Annotations
@app.route("/upload_variant_annotations", methods=["GET", "POST"])
def upload_variant_annotations():
    with app.app_context():
        if request.method == "GET":
            return render_template("upload_variant_annotations.html")
        
        files = {
            "study_parameters": load_study_parameters_tsv,
            "var_fa_ann": load_var_fa_ann_tsv,
            "var_drug_ann": load_var_drug_ann_tsv,
            "var_pheno_ann": load_var_pheno_ann_tsv
        }
        for key, func in files.items():
            file = request.files.get(key)
            if file and file.filename:
                try:
                    path = save_uploaded_file(file)
                    inserted, skipped, missing_drugs = func(path)
                    logger.info(f"Processed {key}.tsv: Inserted={inserted}, Skipped={skipped}, Missing drugs={missing_drugs}")
                    flash(f"{key}.tsv processed successfully! Inserted: {inserted}, Skipped: {skipped}, Missing drugs: {', '.join(missing_drugs) if missing_drugs else 'None'}", "success")
                    os.remove(path)
                except Exception as e:
                    logger.error(f"Error processing {key}.tsv: {str(e)}")
                    flash(f"Error processing {key}.tsv: {str(e)}", "danger")
                    db.session.rollback()
                    if os.path.exists(path):
                        os.remove(path)
            else:
                logger.warning(f"No file uploaded for {key}")
                flash(f"No file uploaded for {key}", "warning")
        return redirect(url_for("upload_variant_annotations"))
#Variant Annotations için son


#Relationships yükleme route'u
# ETL Function for Relationships
def save_uploaded_file(file_storage):
    if not file_storage.filename.endswith('.tsv'):
        raise ValueError("Only .tsv files are supported")
    filename = secure_filename(file_storage.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    file_storage.save(file_path)
    return file_path

def check_drugs_exist(drug_names, rel_id=""):
    missing = set()
    found_drugs = []
    for dname in drug_names:
        found = db.session.query(Drug).filter(
            (Drug.name_en.ilike(dname)) | 
            (Drug.name_tr.ilike(dname)) | 
            (Drug.pharmgkb_id.ilike(dname))
        ).first()
        if found:
            found_drugs.append(found)
        else:
            missing.add(dname)
            logger.warning(f"DRUG {dname} not found in database for relationship {rel_id}, will skip linking this drug")
    logger.debug(f"Checked {len(drug_names)} drugs for relationship {rel_id}, found {len(found_drugs)}, missing {len(missing)}")
    return found_drugs, missing

def load_relationships_tsv(filepath):
    inserted = 0
    skipped = 0
    missing_entities = set()
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            entity1_id = row.get("Entity1_id", "").strip()
            entity1_name = row.get("Entity1_name", "").strip()
            entity1_type = row.get("Entity1_type", "").strip().lower()
            entity2_id = row.get("Entity2_id", "").strip()
            entity2_name = row.get("Entity2_name", "").strip()
            entity2_type = row.get("Entity2_type", "").strip().lower()
            evidence = row.get("Evidence", "").strip()
            association = row.get("Association", "").strip()
            pk = row.get("PK", "").strip()
            pd = row.get("PD", "").strip()
            pmids = row.get("PMIDs", "").strip()
            rel_id = f"{entity1_id}-{entity2_id}"

            if not entity1_id or not entity2_id:
                logger.warning(f"Skipping row with missing Entity1_id or Entity2_id for {rel_id}")
                skipped += 1
                continue

            try:
                # Check for drugs if entity1_type or entity2_type is chemical
                drug_names = []
                if entity1_type == "chemical" and entity1_name:
                    drug_names.append(entity1_name.lower())
                if entity2_type == "chemical" and entity2_name:
                    drug_names.append(entity2_name.lower())
                if drug_names:
                    found_drugs, missing = check_drugs_exist(drug_names, rel_id)
                    missing_entities.update(missing)

                # Create Relationship record
                rel = Relationship(
                    entity1_id=entity1_id,
                    entity1_name=entity1_name,
                    entity1_type=entity1_type,
                    entity2_id=entity2_id,
                    entity2_name=entity2_name,
                    entity2_type=entity2_type,
                    evidence=evidence,
                    association=association,
                    pk=pk,
                    pd=pd,
                    pmids=pmids
                )
                db.session.add(rel)

                # Process VariantAnnotation if Evidence contains VariantAnnotation
                if "VariantAnnotation" in evidence.split(','):
                    # Generate synthetic source_id for VariantAnnotation
                    source_id = f"VA_{entity1_id}_{entity2_id}"
                    
                    # Ensure VariantAnnotation exists
                    va = db.session.get(VariantAnnotation, source_id)
                    if not va:
                        va = VariantAnnotation(variant_annotation_id=source_id)
                        db.session.add(va)
                        db.session.flush()

                    # Process Gene (entity1 or entity2)
                    if entity1_type == "gene":
                        gene = db.session.query(Gene).filter_by(gene_id=entity1_id).first()
                        if not gene:
                            logger.warning(f"Gene {entity1_id} not found for {rel_id}, skipping gene link")
                            missing_entities.add(entity1_id)
                        else:
                            va_gene = db.session.query(VariantAnnotationGene).filter_by(
                                variant_annotation_id=source_id, gene_id=entity1_id
                            ).first()
                            if not va_gene:
                                va_gene = VariantAnnotationGene(variant_annotation_id=source_id, gene_id=entity1_id)
                                db.session.add(va_gene)
                    elif entity2_type == "gene":
                        gene = db.session.query(Gene).filter_by(gene_id=entity2_id).first()
                        if not gene:
                            logger.warning(f"Gene {entity2_id} not found for {rel_id}, skipping gene link")
                            missing_entities.add(entity2_id)
                        else:
                            va_gene = db.session.query(VariantAnnotationGene).filter_by(
                                variant_annotation_id=source_id, gene_id=entity2_id
                            ).first()
                            if not va_gene:
                                va_gene = VariantAnnotationGene(variant_annotation_id=source_id, gene_id=entity2_id)
                                db.session.add(va_gene)

                    # Process Variant or Haplotype (entity1 or entity2)
                    if entity1_type in ["variant", "haplotype"]:
                        var = db.session.query(Variant).filter_by(pharmgkb_id=entity1_id).first()
                        if not var:
                            # Create Variant if not found
                            var = Variant(pharmgkb_id=entity1_id, name=entity1_name)
                            db.session.add(var)
                            db.session.flush()
                        va_var = db.session.query(VariantAnnotationVariant).filter_by(
                            variant_annotation_id=source_id, variant_id=var.id
                        ).first()
                        if not va_var:
                            va_var = VariantAnnotationVariant(variant_annotation_id=source_id, variant_id=var.id)
                            db.session.add(va_var)
                    elif entity2_type in ["variant", "haplotype"]:
                        var = db.session.query(Variant).filter_by(pharmgkb_id=entity2_id).first()
                        if not var:
                            # Create Variant if not found
                            var = Variant(pharmgkb_id=entity2_id, name=entity2_name)
                            db.session.add(var)
                            db.session.flush()
                        va_var = db.session.query(VariantAnnotationVariant).filter_by(
                            variant_annotation_id=source_id, variant_id=var.id
                        ).first()
                        if not va_var:
                            va_var = VariantAnnotationVariant(variant_annotation_id=source_id, variant_id=var.id)
                            db.session.add(va_var)

                inserted += 1
                if inserted % 100 == 0:
                    db.session.commit()
            except sqlalchemy.exc.IntegrityError as e:
                logger.error(f"Relationship {rel_id} failed: {str(e)}")
                db.session.rollback()
                skipped += 1
        db.session.commit()
    if missing_entities:
        logger.warning(f"[relationships] Missing entities: {', '.join(missing_entities)}")
    logger.info(f"[relationships] Inserted: {inserted}, Skipped: {skipped}")
    return inserted, skipped, missing_entities

@app.route("/upload_relationships", methods=["GET", "POST"])
def upload_relationships():
    with app.app_context():
        if request.method == "GET":
            return render_template("upload_relationships.html")
        
        rel_file = request.files.get("relationships_file")
        if rel_file and rel_file.filename:
            try:
                path = save_uploaded_file(rel_file)
                inserted, skipped, missing_entities = load_relationships_tsv(path)
                logger.info(f"Processed relationships.tsv: Inserted={inserted}, Skipped={skipped}, Missing entities={missing_entities}")
                flash(f"relationships.tsv processed successfully! Inserted: {inserted}, Skipped: {skipped}, Missing entities: {', '.join(missing_entities) if missing_entities else 'None'}", "success")
                os.remove(path)
            except Exception as e:
                logger.error(f"Error processing relationships.tsv: {str(e)}")
                flash(f"Error processing relationships.tsv: {str(e)}", "danger")
                db.session.rollback()
                if os.path.exists(path):
                    os.remove(path)
        else:
            logger.warning("No file selected or invalid file for relationships_file")
            flash("No file selected or invalid file!", "danger")
        return redirect(url_for("upload_relationships"))
#Relationships için son....

#Drug Labels Başlangıç
def save_uploaded_file(file_storage):
    if not file_storage.filename.endswith('.tsv'):
        raise ValueError("Only .tsv files are supported")
    filename = secure_filename(file_storage.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    file_storage.save(file_path)
    return file_path

def check_drugs_exist(drug_names, label_id=""):
    missing = set()
    found_drugs = []
    for dname in drug_names:
        found = db.session.query(Drug).filter(
            (Drug.name_en.ilike(dname)) | 
            (Drug.name_tr.ilike(dname)) | 
            (Drug.pharmgkb_id.ilike(dname))
        ).first()
        if found:
            found_drugs.append(found)
        else:
            missing.add(dname)
            logger.warning(f"DRUG {dname} not found in database for DrugLabel {label_id}, will skip linking this drug")
    logger.debug(f"Checked {len(drug_names)} drugs for DrugLabel {label_id}, found {len(found_drugs)}, missing {len(missing)}")
    return found_drugs, missing

def parse_date(date_str, row_id=""):
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        logger.warning(f"Invalid date format '{date_str}' for row {row_id}")
        return None

def safe_split(text, delimiter=';'):
    return [t.strip() for t in (text or '').split(delimiter) if t.strip()] if text else []

# ETL Functions for Drug Labels
def load_drug_labels_tsv(filepath):
    missing_drugs = set()
    inserted = 0
    skipped = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            pharmgkb_id = row.get("PharmGKB ID", "").strip()
            if not pharmgkb_id:
                logger.warning(f"Skipping row with empty PharmGKB ID")
                skipped += 1
                continue
            raw_chems = row.get("Chemicals", "").strip()
            chem_list = [c.strip().lower() for c in raw_chems.replace('/', ';').split(';') if c.strip()]
            found_drugs, missing = check_drugs_exist(chem_list, pharmgkb_id)
            missing_drugs.update(missing)
            # Process row even if no drugs are found, as Chemicals can be empty
            try:
                # Get or create DrugLabel
                dl = db.session.get(DrugLabel, pharmgkb_id)
                if not dl:
                    dl = DrugLabel(pharmgkb_id=pharmgkb_id)
                    db.session.add(dl)
                
                # Update DrugLabel fields
                dl.name = row.get("Name", "")
                dl.source = row.get("Source", "")
                dl.biomarker_flag = row.get("Biomarker Flag", "") or None
                dl.testing_level = row.get("Testing Level", "") or None
                dl.has_prescribing_info = row.get("Has Prescribing Info", "") or None
                dl.has_dosing_info = row.get("Has Dosing Info", "") or None
                dl.has_alternate_drug = row.get("Has Alternate Drug", "") or None
                dl.has_other_prescribing_guidance = row.get("Has Other Prescribing Guidance", "") or None
                dl.cancer_genome = row.get("Cancer Genome", "") or None
                dl.prescribing = row.get("Prescribing", "") or None
                dl.latest_history_date = parse_date(row.get("Latest History Date (YYYY-MM-DD)", ""), pharmgkb_id)

                # Clear existing relationships with no_autoflush
                with db.session.no_autoflush:
                    db.session.query(DrugLabelDrug).filter_by(pharmgkb_id=pharmgkb_id).delete()
                    db.session.query(DrugLabelGene).filter_by(pharmgkb_id=pharmgkb_id).delete()
                    db.session.query(DrugLabelVariant).filter_by(pharmgkb_id=pharmgkb_id).delete()

                # Add drugs (only those found)
                for drug in found_drugs:
                    dl_drug = DrugLabelDrug(pharmgkb_id=pharmgkb_id, drug_id=drug.id)
                    db.session.add(dl_drug)

                # Add genes
                raw_genes = row.get("Genes", "").strip()
                for gene_name in safe_split(raw_genes):
                    # Check by gene_symbol first to avoid duplicates
                    gene = db.session.query(Gene).filter_by(gene_symbol=gene_name).first()
                    if not gene:
                        gene = Gene(gene_id=f"PA{gene_name}", gene_symbol=gene_name)
                        db.session.add(gene)
                        db.session.flush()  # Ensure gene_id is available
                    else:
                        logger.debug(f"Reusing existing Gene {gene.gene_id} for symbol {gene_name}")
                    dl_gene = DrugLabelGene(pharmgkb_id=pharmgkb_id, gene_id=gene.gene_id)
                    db.session.add(dl_gene)

                # Add variants (only if non-empty)
                raw_variants = row.get("Variants/Haplotypes", "").strip()
                if raw_variants:  # Skip if empty to avoid invalid insertions
                    for var_name in safe_split(raw_variants):
                        var = db.session.query(Variant).filter_by(name=var_name).first()
                        if not var:
                            var = Variant(name=var_name, pharmgkb_id=None)
                            db.session.add(var)
                            db.session.flush()  # Ensure variant_id is available
                        dl_var = DrugLabelVariant(pharmgkb_id=pharmgkb_id, variant_id=var.id)
                        db.session.add(dl_var)

                inserted += 1
                if inserted % 100 == 0:
                    db.session.commit()
            except (ValueError, KeyError, sqlalchemy.exc.IntegrityError) as e:
                logger.error(f"Drug Label {pharmgkb_id} failed: {str(e)}")
                db.session.rollback()
                skipped += 1
        db.session.commit()
    if missing_drugs:
        logger.warning(f"[drugLabels] Missing drugs: {', '.join(missing_drugs)}")
    logger.info(f"[drugLabels] Inserted: {inserted}, Skipped: {skipped}")
    return inserted, skipped, missing_drugs

def load_drug_labels_byGene_tsv(filepath):
    inserted = 0
    skipped = 0
    missing_drugs = set()  # No drugs involved
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            gene_id = row.get("Gene ID", "").strip()
            gene_symbol = row.get("Gene Symbol", "").strip()
            label_ids = row.get("Label IDs", "").strip()
            if not gene_id or not label_ids:
                logger.warning(f"Skipping row with empty Gene ID or Label IDs")
                skipped += 1
                continue
            pharmgkb_ids = [lid.strip() for lid in label_ids.split(';') if lid.strip()]
            try:
                # Ensure Gene exists, check by gene_symbol first
                gene = db.session.query(Gene).filter_by(gene_symbol=gene_symbol).first()
                if not gene:
                    gene = Gene(gene_id=gene_id, gene_symbol=gene_symbol)
                    db.session.add(gene)
                    db.session.flush()  # Ensure gene_id is available
                elif gene.gene_id != gene_id:
                    logger.warning(f"Gene symbol {gene_symbol} already exists with gene_id {gene.gene_id}, using it instead of {gene_id}")
                
                # Add DrugLabelGene relationships
                for pharmgkb_id in pharmgkb_ids:
                    dl = db.session.get(DrugLabel, pharmgkb_id)
                    if not dl:
                        logger.warning(f"DrugLabel {pharmgkb_id} not found for Gene {gene_id}")
                        continue  # Skip individual ID, don't skip row
                    dlg = db.session.query(DrugLabelGene).filter_by(
                        pharmgkb_id=pharmgkb_id, gene_id=gene.gene_id
                    ).first()
                    if not dlg:
                        dlg = DrugLabelGene(pharmgkb_id=pharmgkb_id, gene_id=gene.gene_id)
                        db.session.add(dlg)
                        inserted += 1
                if inserted % 100 == 0:
                    db.session.commit()
            except sqlalchemy.exc.IntegrityError as e:
                logger.error(f"DrugLabelGene {gene_id} ({gene_symbol}) failed: {str(e)}")
                db.session.rollback()
                skipped += 1
        db.session.commit()
    logger.info(f"[drugLabels_byGene] Inserted: {inserted}, Skipped: {skipped}")
    return inserted, skipped, missing_drugs

# Route for Drug Labels
@app.route("/upload_drug_labels", methods=["GET", "POST"])
def upload_drug_labels():
    with app.app_context():
        if request.method == "GET":
            return render_template("upload_drug_labels.html")
        
        files = {
            "drug_labels": load_drug_labels_tsv,
            "drug_labels_byGene": load_drug_labels_byGene_tsv
        }
        for key, func in files.items():
            file = request.files.get(key)
            if file and file.filename:
                try:
                    path = save_uploaded_file(file)
                    inserted, skipped, missing_drugs = func(path)
                    logger.info(f"Processed {key}.tsv: Inserted={inserted}, Skipped={skipped}, Missing drugs={missing_drugs}")
                    flash(f"{key.replace('_', '.')}.tsv processed successfully! Inserted: {inserted}, Skipped: {skipped}, Missing drugs: {', '.join(missing_drugs) if missing_drugs else 'None'}", "success")
                    os.remove(path)
                except Exception as e:
                    logger.error(f"Error processing {key}.tsv: {str(e)}")
                    flash(f"Error processing {key}.tsv: {str(e)}", "danger")
                    db.session.rollback()
                    if os.path.exists(path):
                        os.remove(path)
            else:
                logger.warning(f"No file uploaded for {key}")
                flash(f"No file uploaded for {key}", "warning")
        return redirect(url_for("upload_drug_labels"))
    
#drug labels yükleme route sonu...
    
#Clinical Variant için yükleme route'u
# ETL Function for Clinical Variants
# ETL Function for Clinical Variants
def save_uploaded_file(file_storage):
    if not file_storage.filename.endswith('.tsv'):
        raise ValueError("Only .tsv files are supported")
    filename = secure_filename(file_storage.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    file_storage.save(file_path)
    return file_path

def check_drugs_exist(drug_names, cv_id=""):
    missing = set()
    found_drugs = []
    for dname in drug_names:
        found = db.session.query(Drug).filter(
            (Drug.name_en.ilike(dname)) | 
            (Drug.name_tr.ilike(dname)) | 
            (Drug.pharmgkb_id.ilike(dname))
        ).first()
        if found:
            found_drugs.append(found)
        else:
            missing.add(dname)
            logger.warning(f"DRUG {dname} not found in database for Clinical Variant {cv_id}, will skip linking this drug")
    logger.debug(f"Checked {len(drug_names)} drugs for Clinical Variant {cv_id}, found {len(found_drugs)}, missing {len(missing)}")
    return found_drugs, missing

def safe_split(text, delimiter=','):
    return [t.strip() for t in (text or '').split(delimiter) if t.strip()] if text else []

# ETL Function for Clinical Variants
def load_clinical_variants_tsv(filepath):
    missing_drugs = set()
    inserted = 0
    skipped = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            variant_name = row.get("variant", "").strip()
            gene_symbol = row.get("gene", "").strip()
            cv_id = f"{variant_name}-{gene_symbol}"
            if not variant_name or not gene_symbol:
                logger.warning(f"Skipping row with empty variant or gene for {cv_id}")
                skipped += 1
                continue
            raw_chems = row.get("chemicals", "").strip()
            drug_list = [d.strip().lower() for d in raw_chems.replace('/', ',').replace(';', ',').split(',') if d.strip()]
            found_drugs, missing = check_drugs_exist(drug_list, cv_id)
            missing_drugs.update(missing)
            # Process row even if no drugs are found, as chemicals can be empty
            try:
                # Validate Gene
                gene = db.session.query(Gene).filter_by(gene_symbol=gene_symbol).first()
                if not gene:
                    gene = Gene(gene_id=f"PA{gene_symbol}", gene_symbol=gene_symbol)
                    db.session.add(gene)
                    db.session.flush()  # Ensure gene_id is available
                else:
                    logger.debug(f"Reusing existing Gene {gene.gene_id} for symbol {gene_symbol}")

                # Create ClinicalVariant
                cv = ClinicalVariant(
                    variant_type=row.get("Type", ""),
                    level_of_evidence=row.get("Level of Evidence", ""),
                    gene_id=gene.gene_id
                )
                db.session.add(cv)
                db.session.flush()  # Get cv.id for relationships

                # Clear existing relationships with no_autoflush
                with db.session.no_autoflush:
                    db.session.query(ClinicalVariantDrug).filter_by(clinical_variant_id=cv.id).delete()
                    db.session.query(ClinicalVariantPhenotype).filter_by(clinical_variant_id=cv.id).delete()
                    db.session.query(ClinicalVariantVariant).filter_by(clinical_variant_id=cv.id).delete()

                # Add drugs (only those found)
                for drug in found_drugs:
                    cv_drug = ClinicalVariantDrug(clinical_variant_id=cv.id, drug_id=drug.id)
                    db.session.add(cv_drug)

                # Add phenotypes
                raw_phenotypes = row.get("phenotypes", "").strip()
                for pheno_name in safe_split(raw_phenotypes):
                    pheno = db.session.query(Phenotype).filter_by(name=pheno_name).first()
                    if not pheno:
                        pheno = Phenotype(name=pheno_name)
                        db.session.add(pheno)
                        db.session.flush()  # Ensure phenotype_id is available
                    else:
                        logger.debug(f"Reusing existing Phenotype {pheno.id} for name {pheno_name}")
                    cv_pheno = ClinicalVariantPhenotype(clinical_variant_id=cv.id, phenotype_id=pheno.id)
                    db.session.add(cv_pheno)

                # Add variant (only if non-empty)
                if variant_name:  # Ensure variant_name is valid
                    var = db.session.query(Variant).filter_by(name=variant_name).first()
                    if not var:
                        var = Variant(name=variant_name, pharmgkb_id=None)
                        db.session.add(var)
                        db.session.flush()  # Ensure variant_id is available
                    else:
                        logger.debug(f"Reusing existing Variant {var.id} for name {variant_name}")
                    cv_var = ClinicalVariantVariant(clinical_variant_id=cv.id, variant_id=var.id)
                    db.session.add(cv_var)

                inserted += 1
                if inserted % 100 == 0:
                    db.session.commit()
            except (ValueError, KeyError, sqlalchemy.exc.IntegrityError) as e:
                logger.error(f"Clinical Variant {cv_id} failed: {str(e)}")
                db.session.rollback()
                skipped += 1
        db.session.commit()
    if missing_drugs:
        logger.warning(f"[clinicalVariants] Missing drugs: {', '.join(missing_drugs)}")
    logger.info(f"[clinicalVariants] Inserted: {inserted}, Skipped: {skipped}")
    return inserted, skipped, missing_drugs

# Route for Clinical Variants
@app.route("/upload_clinical_variants", methods=["GET", "POST"])
def upload_clinical_variants():
    with app.app_context():
        if request.method == "GET":
            return render_template("upload_clinical_variants.html")
        
        f_cv = request.files.get("clinical_variants_file")  # Match likely HTML input name
        if f_cv and f_cv.filename:
            try:
                logger.debug(f"Received file: {f_cv.filename}")
                path = save_uploaded_file(f_cv)
                inserted, skipped, missing_drugs = load_clinical_variants_tsv(path)
                logger.info(f"Processed clinicalVariants.tsv: Inserted={inserted}, Skipped={skipped}, Missing drugs={missing_drugs}")
                flash(f"clinicalVariants.tsv processed successfully! Inserted: {inserted}, Skipped: {skipped}, Missing drugs: {', '.join(missing_drugs) if missing_drugs else 'None'}", "success")
                os.remove(path)
            except Exception as e:
                logger.error(f"Error processing clinicalVariants.tsv: {str(e)}")
                flash(f"Error processing clinicalVariants.tsv: {str(e)}", "danger")
                db.session.rollback()
                if os.path.exists(path):
                    os.remove(path)
        else:
            logger.warning("No file selected or invalid file for clinical_variants_file")
            flash("No file selected or invalid file!", "danger")
        return redirect(url_for("upload_clinical_variants"))
#clinical variants için  yükleme sonu...


# ETL Function for Occurrences
def save_uploaded_file(file_storage):
    if not file_storage.filename.endswith('.tsv'):
        raise ValueError("Only .tsv files are supported")
    filename = secure_filename(file_storage.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    file_storage.save(file_path)
    return file_path

def check_drugs_exist(drug_names, occ_id=""):
    missing = set()
    found_drugs = []
    for dname in drug_names:
        found = db.session.query(Drug).filter(
            (Drug.name_en.ilike(dname)) | 
            (Drug.name_tr.ilike(dname)) | 
            (Drug.pharmgkb_id.ilike(dname))
        ).first()
        if found:
            found_drugs.append(found)
        else:
            missing.add(dname)
            logger.warning(f"DRUG {dname} not found in database for Occurrence {occ_id}, will skip linking this drug")
    logger.debug(f"Checked {len(drug_names)} drugs for Occurrence {occ_id}, found {len(found_drugs)}, missing {len(missing)}")
    return found_drugs, missing

# ETL Function for Occurrences
def load_occurrences_tsv(filepath):
    missing_drugs = set()
    inserted = 0
    skipped = 0
    total_rows = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            total_rows += 1
            source_id = row.get("Source ID", "").strip()
            object_id = row.get("Object ID", "").strip()
            occ_id = f"{source_id}-{object_id}"
            if not source_id or not object_id:
                logger.warning(f"Skipping row with empty Source ID or Object ID for {occ_id}")
                skipped += 1
                continue
            source_type = row.get("Source Type", "").strip()
            source_name = row.get("Source Name", "").strip()
            object_type = row.get("Object Type", "").strip()
            object_name = row.get("Object Name", "").strip()
            
            # Check drugs for both source_type and object_type = "chemical"
            drug_names = []
            if source_type.lower() == "chemical" and source_name:
                drug_names.append(source_name.lower())
            if object_type.lower() == "chemical" and object_name:
                drug_names.append(object_name.lower())
            
            found_drugs, missing = check_drugs_exist(drug_names, occ_id)
            missing_drugs.update(missing)
            # Process row even if no drugs are found, as types may not be chemical

            # Validate source_type and object_type
            valid_types = {"chemical", "gene", "variant", "haplotype", "disease", "phenotype", "literature", "pathway"}
            if source_type.lower() not in valid_types:
                logger.warning(f"Invalid Source Type: {source_type} for {occ_id}, proceeding anyway")
            if object_type.lower() not in valid_types:
                logger.warning(f"Invalid Object Type: {object_type} for {occ_id}, proceeding anyway")
            # Only skip if both types are invalid and critical fields are missing
            if (source_type.lower() not in valid_types and object_type.lower() not in valid_types and
                not (source_name or object_name)):
                logger.warning(f"Skipping row with both invalid types: {source_type}, {object_type} and no names for {occ_id}")
                skipped += 1
                continue

            # Optional: Warn about non-existent entity IDs (no skipping)
            if source_type.lower() == "chemical" and source_id and not db.session.query(Drug).filter_by(pharmgkb_id=source_id).first():
                logger.warning(f"Non-existent Drug {source_id} for {occ_id}, proceeding anyway")
            if object_type.lower() == "chemical" and object_id and not db.session.query(Drug).filter_by(pharmgkb_id=object_id).first():
                logger.warning(f"Non-existent Drug {object_id} for {occ_id}, proceeding anyway")
            if source_type.lower() == "gene" and source_id and not db.session.query(Gene).filter_by(gene_id=source_id).first():
                logger.warning(f"Non-existent Gene {source_id} for {occ_id}, proceeding anyway")
            if object_type.lower() == "gene" and object_id and not db.session.query(Gene).filter_by(gene_id=object_id).first():
                logger.warning(f"Non-existent Gene {object_id} for {occ_id}, proceeding anyway")
            if source_type.lower() == "variant" and source_id and not db.session.query(Variant).filter_by(pharmgkb_id=source_id).first():
                logger.warning(f"Non-existent Variant {source_id} for {occ_id}, proceeding anyway")
            if object_type.lower() == "variant" and object_id and not db.session.query(Variant).filter_by(pharmgkb_id=object_id).first():
                logger.warning(f"Non-existent Variant {object_id} for {occ_id}, proceeding anyway")

            try:
                logger.debug(f"Processing valid row for Occurrence {occ_id} with types {source_type}, {object_type}")
                occ = Occurrence(
                    source_type=source_type,
                    source_id=source_id,
                    source_name=source_name,
                    object_type=object_type,
                    object_id=object_id,
                    object_name=object_name
                )
                db.session.add(occ)
                inserted += 1
                if inserted % 100 == 0:
                    db.session.commit()
            except (ValueError, KeyError, sqlalchemy.exc.IntegrityError) as e:
                logger.error(f"Occurrence {occ_id} failed: {str(e)}")
                db.session.rollback()
                skipped += 1
        db.session.commit()
    if missing_drugs:
        logger.warning(f"[occurrences] Missing drugs: {', '.join(missing_drugs)}")
    logger.info(f"[occurrences] Total rows: {total_rows}, Inserted: {inserted}, Skipped: {skipped}")
    return inserted, skipped, missing_drugs

# Route for Occurrences
@app.route("/upload_occurrences", methods=["GET", "POST"])
def upload_occurrences():
    with app.app_context():
        if request.method == "GET":
            return render_template("upload_occurrences.html")
        
        occ_file = request.files.get("occurrences_file")  # Match HTML input name
        if occ_file and occ_file.filename:
            try:
                logger.debug(f"Received file: {occ_file.filename}")
                path = save_uploaded_file(occ_file)
                inserted, skipped, missing_drugs = load_occurrences_tsv(path)
                logger.info(f"Processed occurrences.tsv: Inserted={inserted}, Skipped={skipped}, Missing drugs={missing_drugs}")
                flash(f"occurrences.tsv processed successfully! Inserted: {inserted}, Skipped: {skipped}, Missing drugs: {', '.join(missing_drugs) if missing_drugs else 'None'}", "success")
                os.remove(path)
            except Exception as e:
                logger.error(f"Error processing occurrences.tsv: {str(e)}")
                flash(f"Error processing occurrences.tsv: {str(e)}", "danger")
                db.session.rollback()
                if os.path.exists(path):
                    os.remove(path)
        else:
            logger.warning("No file selected or invalid file for occurrences_file")
            flash("No file selected or invalid file!", "danger")
        return redirect(url_for("upload_occurrences"))
    
#occurrences için yükleme sonu
    
#automated annotations için yükleme
# ETL Function for Automated Annotations
def save_uploaded_file(file_storage):
    if not file_storage.filename.endswith('.tsv'):
        raise ValueError("Only .tsv files are supported")
    filename = secure_filename(file_storage.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    file_storage.save(file_path)
    return file_path

def check_drugs_exist(drug_names, ann_id=""):
    missing = set()
    found_drugs = []
    for dname in drug_names:
        found = db.session.query(Drug).filter(
            (Drug.name_en.ilike(dname)) | 
            (Drug.name_tr.ilike(dname)) | 
            (Drug.pharmgkb_id.ilike(dname))
        ).first()
        if found:
            found_drugs.append(found)
        else:
            missing.add(dname)
            logger.warning(f"DRUG {dname} not found in database for Automated Annotation {ann_id}, will skip linking this drug")
    logger.debug(f"Checked {len(drug_names)} drugs for Automated Annotation {ann_id}, found {len(found_drugs)}, missing {len(missing)}")
    return found_drugs, missing

# ETL Function for Automated Annotations (unchanged)
def load_automated_annotations_tsv(filepath):
    missing_drugs = set()
    inserted = 0
    skipped = 0
    row_number = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            row_number += 1
            chem_name = row.get("Chemical Name", "").strip()
            pmid = row.get("PMID", "").strip()
            lit_title = row.get("Literature Title", "").strip()
            sentence = row.get("Sentence", "").strip()
            ann_id = f"Row {row_number} (PMID: {pmid or 'Unknown'})"
            if not (pmid or lit_title or sentence):
                logger.warning(f"Skipping row {row_number} with no PMID, title, or sentence")
                skipped += 1
                continue
            # Check drug if chem_name is provided, but don't skip row
            if chem_name:
                found_drugs, missing = check_drugs_exist([chem_name.lower()], ann_id)
                missing_drugs.update(missing)
            try:
                # Validate and create Publication if PMID is provided
                if pmid:
                    pub = db.session.get(Publication, pmid)
                    if not pub:
                        pub = Publication(
                            pmid=pmid,
                            title=lit_title or "Unknown",
                            year=row.get("Publication Year", "") or None,
                            journal=row.get("Journal", "") or None
                        )
                        db.session.add(pub)
                        db.session.flush()  # Ensure pmid is available
                    logger.debug(f"Using Publication {pmid} for {ann_id}")
                
                # Create AutomatedAnnotation
                logger.debug(f"Processing valid row for Automated Annotation {ann_id}")
                ann = AutomatedAnnotation(
                    chemical_id=row.get("Chemical ID", "") or None,
                    chemical_name=chem_name or None,
                    chemical_in_text=row.get("Chemical in Text", "") or None,
                    variation_id=row.get("Variation ID", "") or None,
                    variation_name=row.get("Variation Name", "") or None,
                    variation_type=row.get("Variation Type", "") or None,
                    variation_in_text=row.get("Variation in Text", "") or None,
                    gene_ids=row.get("Gene IDs", "") or None,
                    gene_symbols=row.get("Gene Symbols", "") or None,
                    gene_in_text=row.get("Gene in Text", "") or None,
                    literature_id=row.get("Literature ID", "") or None,
                    pmid=pmid or None,  # Links to Publication
                    literature_title=lit_title or None,
                    publication_year=row.get("Publication Year", "") or None,
                    journal=row.get("Journal", "") or None,
                    sentence=sentence or None,
                    source=row.get("Source", "") or None
                )
                db.session.add(ann)
                inserted += 1
                if inserted % 100 == 0:
                    db.session.commit()
            except (ValueError, KeyError, sqlalchemy.exc.IntegrityError) as e:
                logger.error(f"Automated Annotation {ann_id} failed: {str(e)}")
                db.session.rollback()
                skipped += 1
        db.session.commit()
    if missing_drugs:
        logger.warning(f"[automated_annotations] Missing drugs: {', '.join(missing_drugs)}")
    logger.info(f"[automated_annotations] Total rows: {row_number}, Inserted: {inserted}, Skipped: {skipped}")
    return inserted, skipped, missing_drugs

# Route for Automated Annotations
@app.route("/upload_automated_annotations", methods=["GET", "POST"])
def upload_automated_annotations():
    with app.app_context():
        if request.method == "GET":
            return render_template("upload_automated_annotations.html")
        
        # Log all file keys received for debugging
        logger.debug(f"Received file keys in request.files: {list(request.files.keys())}")
        
        # Use the correct input name from the form
        f_auto = request.files.get("automated_file")
        
        if f_auto and f_auto.filename:
            try:
                logger.debug(f"Found file with key 'automated_file': {f_auto.filename}")
                path = save_uploaded_file(f_auto)
                inserted, skipped, missing_drugs = load_automated_annotations_tsv(path)
                logger.info(f"Processed automated_annotations.tsv: Inserted={inserted}, Skipped={skipped}, Missing drugs={missing_drugs}")
                flash(f"automated_annotations.tsv processed successfully! Inserted: {inserted}, Skipped: {skipped}, Missing drugs: {', '.join(missing_drugs) if missing_drugs else 'None'}", "success")
                os.remove(path)
            except Exception as e:
                logger.error(f"Error processing automated_annotations.tsv: {str(e)}")
                flash(f"Error processing automated_annotations.tsv: {str(e)}", "danger")
                db.session.rollback()
                if os.path.exists(path):
                    os.remove(path)
        else:
            logger.warning(f"No valid file found for key 'automated_file', request.files: {list(request.files.keys())}")
            flash("No file selected or invalid file! Ensure the form uses input name 'automated_file' and the file is a valid TSV.", "danger")
        return redirect(url_for("upload_automated_annotations"))
#automated annotations için yükleme sonu    

# Pharmacogenomics Module:
# Pharmacogenomics Module
# API Endpoints for Search Suggestions
# Helper Functions
def safe_split(text, delimiter=';'):
    if not text or not isinstance(text, str):
        return []
    return [t.strip() for t in text.split(delimiter) if t.strip()]

def matches_any(text, query, case_sensitive=False):
    if not text or not query:
        return False
    text = text.upper() if case_sensitive else text.lower()
    query = query.upper() if case_sensitive else query.lower()
    return query in text or any(re.search(rf'\b{re.escape(query)}\b', text, re.IGNORECASE))

# Existing Search Endpoints (unchanged)
# Search Variants
# Search Variants
@app.route("/search_variants", methods=["GET"])
def search_variants():
    query = request.args.get("q", "").strip().lower()
    page = request.args.get("page", 1, type=int)
    limit = request.args.get("limit", 10, type=int)
    offset = (page - 1) * limit

    with app.app_context():
        variants = set()
        # Query ClinicalAnnotation via ClinicalAnnotationVariant
        cas = db.session.query(ClinicalAnnotation).join(ClinicalAnnotationVariant).join(Variant).filter(
            or_(
                Variant.name.ilike(f"%{query}%"),
                Variant.pharmgkb_id.ilike(f"%{query}%")
            )
        ).all()
        variants.update(v.variant.name for ca in cas for v in ca.variants if v.variant.name)
        # Query ClinicalAnnAllele
        ca_alleles = db.session.query(ClinicalAnnAllele).filter(
            ClinicalAnnAllele.genotype_allele.ilike(f"%{query}%")
        ).all()
        variants.update(ca.genotype_allele for ca in ca_alleles if ca.genotype_allele)
        # Query VariantFAAnn, VariantDrugAnn, VariantPhenoAnn
        for va_type in [VariantFAAnn, VariantDrugAnn, VariantPhenoAnn]:
            vas = db.session.query(va_type).filter(
                or_(
                    va_type.variant_annotation_id.ilike(f"%{query}%"),
                    va_type.alleles.ilike(f"%{query}%")
                )
            ).all()
            variants.update(va.variant_annotation_id for va in vas if va.variant_annotation_id)
        # Query ClinicalVariant
        cvs = db.session.query(ClinicalVariant).join(ClinicalVariantVariant).join(Variant).filter(
            or_(
                Variant.name.ilike(f"%{query}%"),
                Variant.pharmgkb_id.ilike(f"%{query}%")
            )
        ).all()
        variants.update(v.variant.name for cv in cvs for v in cv.variants if v.variant.name)
        # Query AutomatedAnnotation
        autos = db.session.query(AutomatedAnnotation).filter(
            or_(
                AutomatedAnnotation.variation_id.ilike(f"%{query}%"),
                AutomatedAnnotation.variation_name.ilike(f"%{query}%")
            )
        ).all()
        variants.update(auto.variation_name for auto in autos if auto.variation_name)

        variant_list = list(variants)
        paginated = variant_list[offset:offset + limit]
        has_more = len(variant_list) > offset + limit

        logger.debug(f"search_variants: query={query}, found={len(variant_list)}, returned={len(paginated)}")
        return jsonify({
            "results": [{"id": v, "text": v} for v in paginated],
            "pagination": {"more": has_more}
        })

# Search Phenotypes
@app.route("/search_phenotypes", methods=["GET"])
def search_phenotypes():
    query = request.args.get("q", "").strip().lower()
    page = request.args.get("page", 1, type=int)
    limit = request.args.get("limit", 10, type=int)
    offset = (page - 1) * limit

    with app.app_context():
        phenotypes = set()
        # Query ClinicalAnnotation via ClinicalAnnotationPhenotype
        cas = db.session.query(ClinicalAnnotation).join(ClinicalAnnotationPhenotype).join(Phenotype).filter(
            Phenotype.name.ilike(f"%{query}%")
        ).all()
        phenotypes.update(p.phenotype.name for ca in cas for p in ca.phenotypes if p.phenotype.name and query in p.phenotype.name.lower())
        # Query VariantPhenoAnn
        vpas = db.session.query(VariantPhenoAnn).filter(
            VariantPhenoAnn.phenotype.ilike(f"%{query}%")
        ).all()
        phenotypes.update(vpa.phenotype for vpa in vpas if vpa.phenotype and query in vpa.phenotype.lower())
        # Query Relationship
        rels = db.session.query(Relationship).filter(
            or_(
                Relationship.entity1_name.ilike(f"%{query}%"),
                Relationship.entity2_name.ilike(f"%{query}%")
            ),
            or_(
                Relationship.entity1_type.ilike("disease"),
                Relationship.entity2_type.ilike("disease")
            )
        ).all()
        phenotypes.update(
            rel.entity1_name if rel.entity1_type.lower() == "disease" else rel.entity2_name
            for rel in rels
            if (rel.entity1_name or rel.entity2_name) and query in (rel.entity1_name or rel.entity2_name or "").lower()
        )
        # Query ClinicalVariant
        cvs = db.session.query(ClinicalVariant).join(ClinicalVariantPhenotype).join(Phenotype).filter(
            Phenotype.name.ilike(f"%{query}%")
        ).all()
        phenotypes.update(p.phenotype.name for cv in cvs for p in cv.phenotypes if p.phenotype.name and query in p.phenotype.name.lower())

        phenotype_list = list(phenotypes)
        paginated = phenotype_list[offset:offset + limit]
        has_more = len(phenotype_list) > offset + limit

        logger.debug(f"search_phenotypes: query={query}, found={len(phenotype_list)}, returned={len(paginated)}")
        return jsonify({
            "results": [{"id": p, "text": p} for p in paginated],
            "pagination": {"more": has_more}
        })

# Search Genes
@app.route("/search_genes", methods=["GET"])
def search_genes():
    with app.app_context():
        try:
            query = request.args.get("q", "").strip().lower()
            limit = request.args.get("limit", 10, type=int)
            page = request.args.get("page", 1, type=int)
            offset = (page - 1) * limit

            genes = set()
            # Query ClinicalAnnotation via ClinicalAnnotationGene
            cas = db.session.query(ClinicalAnnotation).join(ClinicalAnnotationGene).join(Gene).filter(
                or_(
                    Gene.gene_symbol.ilike(f"%{query}%"),
                    Gene.gene_id.ilike(f"%{query}%")
                )
            ).all()
            genes.update(g.gene.gene_symbol for ca in cas for g in ca.genes if g.gene.gene_symbol and query in g.gene.gene_symbol.lower())
            # Query AutomatedAnnotation
            autos = db.session.query(AutomatedAnnotation).filter(
                AutomatedAnnotation.gene_ids.ilike(f"%{query}%")
            ).all()
            genes.update(g.strip() for auto in autos for g in safe_split(auto.gene_ids, ',') if g and query in g.lower())
            # Query DrugLabelGene
            dlgs = db.session.query(DrugLabelGene).join(Gene).filter(
                or_(
                    Gene.gene_symbol.ilike(f"%{query}%"),
                    Gene.gene_id.ilike(f"%{query}%")
                )
            ).all()
            genes.update(dlg.gene.gene_symbol for dlg in dlgs if dlg.gene.gene_symbol and query in dlg.gene.gene_symbol.lower())
            # Query Relationship
            rels = db.session.query(Relationship).filter(
                or_(
                    Relationship.entity1_name.ilike(f"%{query}%"),
                    Relationship.entity2_name.ilike(f"%{query}%")
                ),
                or_(
                    Relationship.entity1_type.ilike("gene"),
                    Relationship.entity2_type.ilike("gene")
                )
            ).all()
            genes.update(
                rel.entity1_name if rel.entity1_type.lower() == "gene" else rel.entity2_name
                for rel in rels
                if (rel.entity1_name or rel.entity2_name) and query in (rel.entity1_name or rel.entity2_name or "").lower()
            )

            gene_list = list(genes)
            paginated = gene_list[offset:offset + limit]
            has_more = len(gene_list) > offset + limit

            logger.debug(f"search_genes: query={query}, found={len(gene_list)}, returned={len(paginated)}")
            return jsonify({
                "results": [{"id": g, "text": g} for g in paginated],
                "pagination": {"more": has_more}
            })
        except Exception as e:
            logger.error(f"Search genes error: {str(e)}")
            return jsonify({"error": f"Server error: {str(e)}"}), 500


# Pharmacogenomics Route (Enhanced)
# Helper Functions
def safe_split(text, delimiter=';'):
    if not text or not isinstance(text, str):
        return []
    return [t.strip() for t in text.split(delimiter) if t.strip()]

def matches_any(text, query, case_sensitive=False):
    if not text or not query or not isinstance(text, str) or not isinstance(query, str):
        return False
    text = text.upper() if case_sensitive else text.lower()
    query = query.upper() if case_sensitive else query.lower()
    try:
        return query in text or bool(re.search(rf'\b{re.escape(query)}\b', text, re.IGNORECASE))
    except TypeError:
        logger.debug(f"TypeError in matches_any: text={text}, query={query}")
        return False

def normalize_score(score, source):
    weights = {
        "ClinicalAnnotation": 1.0,
        "ClinicalVariant": 0.9,
        "DrugLabel": 0.8,
        "VariantFAAnn": 0.7,
        "VariantDrugAnn": 0.7,
        "VariantPhenoAnn": 0.7,
        "StudyParameters": 0.6,
        "Relationship": 0.4,
        "AutomatedAnnotation": 0.5,
        "Occurrence": 0.3,
        "DrugInteraction": 0.6,
        "DrugReceptorInteraction": 0.5,
        "Pathway": 0.4
    }
    return score * weights.get(source, 0.5)

def has_meaningful_data(prediction):
    """Check if a prediction has at least one meaningful field (not 'N/A' or empty)."""
    fields = [prediction.get("drug"), prediction.get("phenotype"), prediction.get("gene"), prediction.get("variant")]
    return any(field and field != "N/A" for field in fields)

@app.route("/pharmacogenomics", methods=["GET", "POST"])
def pharmacogenomics():
    with app.app_context():
        try:
            if request.method == "GET":
                return render_template("pharmacogenomics.html")

            # Validate JSON input
            data = request.get_json(silent=True) or {}
            if not isinstance(data, dict):
                return jsonify({"error": "Invalid JSON input"}), 400

            # Sanitize inputs
            variant = (data.get("variant") or "").strip()[:100] if data.get("variant") else ""
            drug = (data.get("drug") or "").strip().lower()[:100] if data.get("drug") else ""
            phenotype = (data.get("phenotype") or "").strip().lower()[:100] if data.get("phenotype") else ""
            gene = (data.get("gene") or "").strip().upper()[:50] if data.get("gene") else ""

            # Require at least one input
            if not (variant or drug or phenotype or gene):
                return jsonify({"error": "At least one of variant, drug, phenotype, or gene is required"}), 400

            logger.debug(f"Pharmacogenomics input: variant={variant}, drug={drug}, phenotype={phenotype}, gene={gene}")

            results = {
                "summary": "",
                "predictions": [],
                "evidence": [],
                "confidence_score": 0.0,
                "drug_details": [],
                "gene_details": [],
                "variant_details": [],
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "query": {"variant": variant, "drug": drug, "phenotype": phenotype, "gene": gene},
                    "result_count": 0
                }
            }
            evidence_weight = 0.0
            evidence_count = 0

            variant_id = variant.split(" ")[0] if variant and " " in variant else variant

            # 1. Clinical Annotations
            conditions = []
            if variant:
                conditions.append(func.lower(Variant.name).ilike(f"%{variant}%"))
                conditions.append(func.lower(Variant.pharmgkb_id).ilike(f"%{variant}%"))
            if drug:
                conditions.append(func.lower(Drug.name_en).ilike(f"%{drug}%"))
                conditions.append(func.lower(Drug.name_tr).ilike(f"%{drug}%"))
                conditions.append(func.lower(Drug.alternative_names).ilike(f"%{drug}%"))
            if phenotype:
                conditions.append(func.lower(Phenotype.name).ilike(f"%{phenotype}%"))
            if gene:
                conditions.append(func.lower(Gene.gene_symbol).ilike(f"%{gene}%"))

            if conditions:
                query = (
                    db.session.query(ClinicalAnnotation)
                    .outerjoin(ClinicalAnnotationVariant)
                    .outerjoin(Variant)
                    .outerjoin(ClinicalAnnotationDrug)
                    .outerjoin(Drug)
                    .outerjoin(ClinicalAnnotationPhenotype)
                    .outerjoin(Phenotype)
                    .outerjoin(ClinicalAnnotationGene)
                    .outerjoin(Gene)
                    .filter(or_(False, *conditions))
                    .order_by(nullslast(ClinicalAnnotation.score.desc()))
                    .limit(5)
                    .all()
                )
                for ca in query:
                    drugs = [d.drug.name_en for d in ca.drugs if d and d.drug and d.drug.name_en and isinstance(d.drug.name_en, str)] if ca.drugs and isinstance(ca.drugs, list) else []
                    phenotypes = [p.phenotype.name for p in ca.phenotypes if p and p.phenotype and p.phenotype.name] if ca.phenotypes and isinstance(ca.phenotypes, list) else []
                    genes = [
                        g.gene.gene_symbol
                        for g in ca.genes
                        if g and g.gene and g.gene.gene_symbol and isinstance(g.gene.gene_symbol, str) and g.gene.gene_symbol.strip()
                    ] if ca.genes and isinstance(ca.genes, list) else []
                    variants = [v.variant.name for v in ca.variants if v and v.variant and v.variant.name] if ca.variants and isinstance(ca.variants, list) else []
                    if (not variant or (variants and any(matches_any(v, variant, False) for v in variants))) and \
                       (not drug or (drugs and any(matches_any(d, drug, False) for d in drugs))) and \
                       (not phenotype or (phenotypes and any(matches_any(p, phenotype, False) for p in phenotypes))) and \
                       (not gene or (genes and any(matches_any(g, gene, True) for g in genes))):
                        ca_allele = db.session.query(ClinicalAnnAllele).filter_by(clinical_annotation_id=ca.clinical_annotation_id).first()
                        score = float(ca.score or 0) * (1.0 if ca.level_of_evidence in ["1A", "1B"] else 0.8 if ca.level_of_evidence in ["2A", "2B"] else 0.6)
                        score = normalize_score(score, "ClinicalAnnotation")
                        prediction = {
                            "effect": ca_allele.annotation_text or "Potential drug-gene interaction" if ca_allele else "No effect specified",
                            "drug": ", ".join(drugs) or "N/A",
                            "phenotype": ", ".join(phenotypes) or "N/A",
                            "gene": ", ".join(genes) or "N/A",
                            "variant": ", ".join(variants) or "N/A",
                            "source": "ClinicalAnnotation",
                            "confidence": round(score, 2),
                            "evidence_link": ca.url or "N/A",
                            "details": {
                                "clinical_annotation_id": ca.clinical_annotation_id,
                                "level_of_evidence": ca.level_of_evidence or "N/A",
                                "phenotype_category": ca.phenotype_category or "N/A"
                            }
                        }
                        if has_meaningful_data(prediction):
                            results["predictions"].append(prediction)
                            evidence_weight += score
                            evidence_count += ca.evidence_count or 1
                            evidence_text = f"Clinical Annotation: {ca_allele.annotation_text if ca_allele else 'No summary'} (PMID: {ca.url or 'N/A'})"
                            results["evidence"].append({"text": evidence_text, "source": "ClinicalAnnotation"})
                        else:
                            logger.debug(f"Skipping ClinicalAnnotation prediction with no meaningful data: {prediction}")

            # 2. Clinical Variants
            conditions = []
            if variant:
                conditions.append(func.lower(Variant.name).ilike(f"%{variant}%"))
                conditions.append(func.lower(Variant.pharmgkb_id).ilike(f"%{variant}%"))
            if drug:
                conditions.append(func.lower(Drug.name_en).ilike(f"%{drug}%"))
                conditions.append(func.lower(Drug.name_tr).ilike(f"%{drug}%"))
                conditions.append(func.lower(Drug.alternative_names).ilike(f"%{drug}%"))
            if phenotype:
                conditions.append(func.lower(Phenotype.name).ilike(f"%{phenotype}%"))
            if gene:
                conditions.append(func.lower(Gene.gene_symbol).ilike(f"%{gene}%"))

            if conditions:
                query = (
                    db.session.query(ClinicalVariant)
                    .outerjoin(ClinicalVariantVariant)
                    .outerjoin(Variant)
                    .outerjoin(ClinicalVariantDrug)
                    .outerjoin(Drug)
                    .outerjoin(ClinicalVariantPhenotype)
                    .outerjoin(Phenotype)
                    .outerjoin(Gene)
                    .filter(or_(False, *conditions))
                    .order_by(nullslast(ClinicalVariant.level_of_evidence.desc()))
                    .limit(5)
                    .all()
                )
                for cv in query:
                    drugs = [d.drug.name_en for d in cv.drugs if d and d.drug and d.drug.name_en and isinstance(d.drug.name_en, str)] if cv.drugs and isinstance(cv.drugs, list) else []
                    phenotypes = [p.phenotype.name for p in cv.phenotypes if p and p.phenotype and p.phenotype.name] if cv.phenotypes and isinstance(cv.phenotypes, list) else []
                    variants = [v.variant.name for v in cv.variants if v and v.variant and v.variant.name] if cv.variants and isinstance(cv.variants, list) else []
                    if (not variant or (variants and any(matches_any(v, variant, False) for v in variants))) and \
                       (not drug or (drugs and any(matches_any(d, drug, False) for d in drugs))) and \
                       (not phenotype or (phenotypes and any(matches_any(p, phenotype, False) for p in phenotypes))) and \
                       (not gene or (cv.gene and cv.gene.gene_symbol and isinstance(cv.gene.gene_symbol, str) and matches_any(cv.gene.gene_symbol, gene, True))):
                        score = 3.0 if cv.level_of_evidence == "1A" else 2.0 if cv.level_of_evidence == "1B" else 1.0
                        score = normalize_score(score, "ClinicalVariant")
                        prediction = {
                            "effect": f"{cv.variant_type or 'Unknown'} impact",
                            "drug": ", ".join(drugs) or "N/A",
                            "phenotype": ", ".join(phenotypes) or "N/A",
                            "gene": cv.gene.gene_symbol or "N/A" if cv.gene else "N/A",
                            "variant": ", ".join(variants) or "N/A",
                            "source": "ClinicalVariant",
                            "confidence": round(score, 2),
                            "evidence_link": "N/A",
                            "details": {
                                "id": cv.id,
                                "variant_type": cv.variant_type or "N/A",
                                "level_of_evidence": cv.level_of_evidence or "N/A"
                            }
                        }
                        if has_meaningful_data(prediction):
                            results["predictions"].append(prediction)
                            evidence_weight += score
                            evidence_count += 1
                            evidence_text = f"Clinical Variant: {', '.join(variants) or 'No variants'} linked to {', '.join(drugs) or 'No drugs'}"
                            results["evidence"].append({"text": evidence_text, "source": "ClinicalVariant"})
                        else:
                            logger.debug(f"Skipping ClinicalVariant prediction with no meaningful data: {prediction}")

            # 3. Drug Labels
            conditions = []
            if drug:
                conditions.append(func.lower(Drug.name_en).ilike(f"%{drug}%"))
                conditions.append(func.lower(Drug.name_tr).ilike(f"%{drug}%"))
                conditions.append(func.lower(Drug.alternative_names).ilike(f"%{drug}%"))
            if phenotype:
                conditions.append(func.lower(DrugLabel.name).ilike(f"%{phenotype}%"))
            if gene:
                conditions.append(func.lower(Gene.gene_symbol).ilike(f"%{gene}%"))
            if variant:
                conditions.append(func.lower(Variant.name).ilike(f"%{variant}%"))
                conditions.append(func.lower(Variant.pharmgkb_id).ilike(f"%{variant}%"))

            if conditions:
                query = (
                    db.session.query(DrugLabel)
                    .outerjoin(DrugLabelGene)
                    .outerjoin(Gene)
                    .outerjoin(DrugLabelDrug)
                    .outerjoin(Drug)
                    .outerjoin(DrugLabelVariant)
                    .outerjoin(Variant)
                    .filter(or_(False, *conditions))
                    .order_by(nullslast(DrugLabel.testing_level.desc()))
                    .limit(5)
                    .all()
                )
                for dl in query:
                    drugs = [d.drug.name_en for d in dl.drugs if d and d.drug and d.drug.name_en and isinstance(d.drug.name_en, str)] if dl.drugs and isinstance(dl.drugs, list) else []
                    variants = [v.variant.name for v in dl.variants if v and v.variant and v.variant.name] if dl.variants and isinstance(dl.variants, list) else []
                    genes = [g.gene.gene_symbol for g in dl.genes if g and g.gene and g.gene.gene_symbol and isinstance(g.gene.gene_symbol, str)] if dl.genes and isinstance(dl.genes, list) else []
                    if (not drug or (drugs and any(matches_any(d, drug, False) for d in drugs))) and \
                       (not phenotype or (dl.name and matches_any(dl.name, phenotype, False))) and \
                       (not gene or (genes and any(matches_any(g, gene, True) for g in genes))) and \
                       (not variant or (variants and any(matches_any(v, variant, False) for v in variants))):
                        score = 2.0 if dl.testing_level == "Testing Required" else 1.5 if dl.testing_level == "Testing Recommended" else 1.0
                        score = normalize_score(score, "DrugLabel")
                        prediction = {
                            "effect": dl.name or "Label-based guidance",
                            "drug": ", ".join(drugs) or "N/A",
                            "phenotype": dl.name or "N/A",
                            "gene": ", ".join(genes) or "N/A",
                            "variant": ", ".join(variants) or "N/A",
                            "source": "DrugLabel",
                            "confidence": round(score, 2),
                            "evidence_link": dl.source or "N/A",
                            "details": {
                                "pharmgkb_id": dl.pharmgkb_id or "N/A",
                                "testing_level": dl.testing_level or "N/A",
                                "has_prescribing_info": dl.has_prescribing_info or "N/A"
                            }
                        }
                        if has_meaningful_data(prediction):
                            results["predictions"].append(prediction)
                            evidence_weight += score
                            evidence_count += 1
                            evidence_text = f"Drug Label: {dl.name or 'No label info'} ({dl.source or 'N/A'})"
                            results["evidence"].append({"text": evidence_text, "source": "DrugLabel"})
                        else:
                            logger.debug(f"Skipping DrugLabel prediction with no meaningful data: {prediction}")

            # 4. Variant Annotations
            va_types = [VariantFAAnn, VariantDrugAnn, VariantPhenoAnn]
            for va_type in va_types:
                conditions = []
                if variant_id:
                    conditions.append(func.lower(va_type.variant_annotation_id).ilike(f"%{variant_id}%"))
                if variant:
                    conditions.append(func.lower(va_type.alleles).ilike(f"%{variant}%"))
                if drug and va_type == VariantDrugAnn:
                    conditions.append(func.lower(va_type.sentence).ilike(f"%{drug}%"))
                if phenotype and va_type == VariantPhenoAnn:
                    conditions.append(func.lower(va_type.phenotype).ilike(f"%{phenotype}%"))
                if gene:
                    conditions.append(func.lower(va_type.notes).ilike(f"%{gene}%"))
                    conditions.append(VariantAnnotation.genes.any(Gene.gene_symbol.ilike(f"%{gene}%")))

                query = (
                    db.session.query(va_type)
                    .join(VariantAnnotation, va_type.variant_annotation_id == VariantAnnotation.variant_annotation_id)
                    .outerjoin(VariantAnnotationGene)
                    .outerjoin(Gene)
                    .outerjoin(VariantAnnotationDrug)
                    .outerjoin(Drug)
                    .outerjoin(VariantAnnotationVariant)
                    .outerjoin(Variant)
                    .filter(or_(False, *conditions))
                    .order_by(nullslast(va_type.significance.desc()))
                    .limit(5)
                    .all()
                )
                for va in query:
                    if not va.variant_annotation:
                        logger.warning(f"Skipping VariantAnnotation with null variant_annotation_id: {va.variant_annotation_id}")
                        continue
                    drugs = [d.drug.name_en for d in va.variant_annotation.drugs if d and d.drug and d.drug.name_en and isinstance(d.drug.name_en, str)] if va.variant_annotation.drugs and isinstance(va.variant_annotation.drugs, list) else []
                    variants = [v.variant.name for v in va.variant_annotation.variants if v and v.variant and v.variant.name] if va.variant_annotation.variants and isinstance(va.variant_annotation.variants, list) else []
                    genes = [g.gene.gene_symbol for g in va.variant_annotation.genes if g and g.gene and g.gene.gene_symbol and isinstance(g.gene.gene_symbol, str)] if va.variant_annotation.genes and isinstance(va.variant_annotation.genes, list) else []
                    notes = "" if va.notes is None or not isinstance(va.notes, str) else va.notes.strip()
                    if va.notes is None:
                        logger.warning(f"Found NULL notes for variant_annotation_id: {va.variant_annotation_id}, table: {va_type.__tablename__}")
                        continue  # Skip records with NULL notes
                    elif not isinstance(va.notes, str):
                        logger.warning(f"Found non-string notes for variant_annotation_id: {va.variant_annotation_id}, table: {va_type.__tablename__}, type: {type(va.notes)}")
                        continue  # Skip records with non-string notes
                    elif not notes:
                        logger.debug(f"Empty notes for variant_annotation_id: {va.variant_annotation_id}, table: {va_type.__tablename__}")
                    if (not variant or (va.alleles and matches_any(va.alleles, variant, False)) or (variants and any(matches_any(v, variant, False) for v in variants))) and \
                       (not drug or (va_type == VariantDrugAnn and va.sentence and matches_any(va.sentence, drug, False)) or (drugs and any(matches_any(d, drug, False) for d in drugs))) and \
                       (not phenotype or (va_type == VariantPhenoAnn and va.phenotype and matches_any(va.phenotype, phenotype, False))) and \
                       (not gene or (notes and matches_any(notes, gene, True)) or (genes and any(matches_any(g, gene, True) for g in genes))):
                        score = 2.0 if va.significance == "yes" else 1.0
                        score = normalize_score(score, va_type.__tablename__)
                        prediction = {
                            "effect": va.sentence or "No effect specified",
                            "drug": ", ".join(drugs) or (va.sentence.split(" ")[0] if va_type == VariantDrugAnn and va.sentence else "N/A"),
                            "phenotype": va.phenotype if va_type == VariantPhenoAnn else "N/A",
                            "gene": ", ".join(genes) or (notes or "N/A"),
                            "variant": va.alleles or ", ".join(variants) or "N/A",
                            "source": va_type.__tablename__,
                            "confidence": round(score, 2),
                            "evidence_link": "N/A",
                            "details": {
                                "variant_annotation_id": va.variant_annotation_id,
                                "significance": va.significance or "N/A"
                            }
                        }
                        if has_meaningful_data(prediction):
                            results["predictions"].append(prediction)
                            evidence_weight += score
                            evidence_count += 1
                            evidence_text = f"{va_type.__tablename__}: {va.sentence or 'No evidence'}"
                            results["evidence"].append({"text": evidence_text, "source": va_type.__tablename__})
                        else:
                            logger.debug(f"Skipping {va_type.__tablename__} prediction with no meaningful data: {prediction}")

            # 5. Study Parameters
            conditions = []
            if variant_id:
                conditions.append(func.lower(StudyParameters.variant_annotation_id).ilike(f"%{variant_id}%"))
            if phenotype:
                conditions.append(func.lower(StudyParameters.characteristics).ilike(f"%{phenotype}%"))

            if conditions:
                query = (
                    db.session.query(StudyParameters)
                    .filter(or_(False, *conditions))
                    .order_by(nullslast(StudyParameters.p_value.asc()))
                    .limit(3)
                    .all()
                )
                for sp in query:
                    if (not variant or (sp.variant_annotation_id and matches_any(sp.variant_annotation_id, variant_id, False))) and \
                       (not phenotype or (sp.characteristics and matches_any(sp.characteristics, phenotype, False))):
                        p_value = float(sp.p_value.strip("= ") or 0) if sp.p_value and sp.p_value.strip("= ") else 0
                        score = 1.5 if p_value < 0.05 else 1.0
                        score = normalize_score(score, "StudyParameters")
                        prediction = {
                            "effect": f"Study: {sp.study_type or 'Unknown'}, P-value: {sp.p_value or 'N/A'}",
                            "drug": "N/A",
                            "phenotype": sp.characteristics or "N/A",
                            "gene": "N/A",
                            "variant": sp.variant_annotation_id or "N/A",
                            "source": "StudyParameters",
                            "confidence": round(score, 2),
                            "evidence_link": "N/A",
                            "details": {
                                "study_parameters_id": sp.study_parameters_id,
                                "study_type": sp.study_type or "N/A",
                                "p_value": sp.p_value or "N/A"
                            }
                        }
                        if has_meaningful_data(prediction):
                            results["predictions"].append(prediction)
                            evidence_weight += score
                            evidence_count += 1
                            evidence_text = f"Study: {sp.characteristics or 'No characteristics'} (P-value: {sp.p_value or 'N/A'})"
                            results["evidence"].append({"text": evidence_text, "source": "StudyParameters"})
                        else:
                            logger.debug(f"Skipping StudyParameters prediction with no meaningful data: {prediction}")

            # 6. Relationships
            conditions = []
            if variant_id:
                conditions.extend([
                    func.lower(Relationship.entity1_id).ilike(f"%{variant_id}%"),
                    func.lower(Relationship.entity2_id).ilike(f"%{variant_id}%")
                ])
            if drug:
                conditions.extend([
                    (func.lower(Relationship.entity1_name).ilike(f"%{drug}%") & Relationship.entity1_type.ilike("chemical")),
                    (func.lower(Relationship.entity2_name).ilike(f"%{drug}%") & Relationship.entity2_type.ilike("chemical"))
                ])
            if phenotype:
                conditions.extend([
                    (func.lower(Relationship.entity1_name).ilike(f"%{phenotype}%") & Relationship.entity1_type.ilike("disease")),
                    (func.lower(Relationship.entity2_name).ilike(f"%{phenotype}%") & Relationship.entity2_type.ilike("disease"))
                ])
            if gene:
                conditions.extend([
                    (func.lower(Relationship.entity1_name).ilike(f"%{gene}%") & Relationship.entity1_type.ilike("gene")),
                    (func.lower(Relationship.entity2_name).ilike(f"%{gene}%") & Relationship.entity2_type.ilike("gene"))
                ])

            if conditions:
                query = (
                    db.session.query(Relationship)
                    .filter(or_(False, *conditions))
                    .order_by(nullslast(Relationship.association.desc()))
                    .limit(5)
                    .all()
                )
                for rel in query:
                    chem_name = rel.entity2_name if rel.entity2_type.lower() == "chemical" else rel.entity1_name if rel.entity1_type.lower() == "chemical" else None
                    pheno_name = rel.entity2_name if rel.entity2_type.lower() == "disease" else rel.entity1_name if rel.entity1_type.lower() == "disease" else None
                    gene_name = rel.entity2_name if rel.entity2_type.lower() == "gene" else rel.entity1_name if rel.entity1_type.lower() == "gene" else None
                    variant_name = rel.entity2_id if rel.entity2_type.lower() in ["variant", "haplotype"] else rel.entity1_id if rel.entity1_type.lower() in ["variant", "haplotype"] else None
                    if (not variant or (variant_name and matches_any(variant_name, variant_id, False))) and \
                       (not drug or (chem_name and matches_any(chem_name, drug, False))) and \
                       (not phenotype or (pheno_name and matches_any(pheno_name, phenotype, False))) and \
                       (not gene or (gene_name and matches_any(gene_name, gene, True))):
                        score = 1.0 if rel.association == "associated" else 0.5
                        score = normalize_score(score, "Relationship")
                        prediction = {
                            "effect": rel.association or "No association specified",
                            "drug": chem_name or "N/A",
                            "phenotype": pheno_name or "N/A",
                            "gene": gene_name or "N/A",
                            "variant": variant_name or "N/A",
                            "source": "Relationship",
                            "confidence": round(score, 2),
                            "evidence_link": f"PMIDs: {rel.pmids}" if rel.pmids else "N/A",
                            "details": {
                                "id": rel.id,
                                "association": rel.association or "N/A",
                                "evidence": rel.evidence or "N/A"
                            }
                        }
                        if has_meaningful_data(prediction):
                            results["predictions"].append(prediction)
                            evidence_weight += score
                            evidence_count += len(safe_split(rel.pmids)) if rel.pmids else 1
                            evidence_text = f"Relationship: {rel.evidence or 'No evidence'} (PMIDs: {rel.pmids or 'N/A'})"
                            results["evidence"].append({"text": evidence_text, "source": "Relationship"})
                        else:
                            logger.debug(f"Skipping Relationship prediction with no meaningful data: {prediction}")

            # 7. Automated Annotations
            conditions = []
            if variant_id:
                conditions.append(func.lower(AutomatedAnnotation.variation_id).ilike(f"%{variant_id}%"))
                conditions.append(func.lower(AutomatedAnnotation.variation_name).ilike(f"%{variant}%"))
            if drug:
                conditions.append(func.lower(AutomatedAnnotation.chemical_name).ilike(f"%{drug}%"))
                conditions.append(func.lower(AutomatedAnnotation.chemical_in_text).ilike(f"%{drug}%"))
            if phenotype:
                conditions.append(func.lower(AutomatedAnnotation.sentence).ilike(f"%{phenotype}%"))
            if gene:
                conditions.append(func.lower(AutomatedAnnotation.gene_ids).ilike(f"%{gene}%"))
                conditions.append(func.lower(AutomatedAnnotation.gene_symbols).ilike(f"%{gene}%"))

            if conditions:
                query = (
                    db.session.query(AutomatedAnnotation)
                    .filter(or_(False, *conditions))
                    .order_by(nullslast(AutomatedAnnotation.publication_year.desc()))
                    .limit(5)
                    .all()
                )
                for auto in query:
                    if (not variant or (auto.variation_id and matches_any(auto.variation_id, variant_id, False)) or (auto.variation_name and matches_any(auto.variation_name, variant, False))) and \
                       (not drug or (auto.chemical_name and matches_any(auto.chemical_name, drug, False)) or (auto.chemical_in_text and matches_any(auto.chemical_in_text, drug, False))) and \
                       (not phenotype or (auto.sentence and matches_any(auto.sentence, phenotype, False))) and \
                       (not gene or (auto.gene_ids and any(matches_any(g, gene, True) for g in safe_split(auto.gene_ids, ','))) or (auto.gene_symbols and any(matches_any(g, gene, True) for g in safe_split(auto.gene_symbols, ',')))):
                        score = 1.0 if auto.publication_year and int(auto.publication_year or 0) >= 2020 else 0.8
                        score = normalize_score(score, "AutomatedAnnotation")
                        prediction = {
                            "effect": auto.sentence or "No effect specified",
                            "drug": auto.chemical_name or "N/A",
                            "phenotype": "N/A",
                            "gene": auto.gene_symbols or auto.gene_ids or "N/A",
                            "variant": auto.variation_name or auto.variation_id or "N/A",
                            "source": "AutomatedAnnotation",
                            "confidence": round(score, 2),
                            "evidence_link": f"PMID: {auto.pmid}" if auto.pmid else "N/A",
                            "details": {
                                "id": auto.id,
                                "pmid": auto.pmid or "N/A",
                                "publication_year": auto.publication_year or "N/A"
                            }
                        }
                        if has_meaningful_data(prediction):
                            results["predictions"].append(prediction)
                            evidence_weight += score
                            evidence_count += 1
                            evidence_text = f"Automated Annotation: {auto.sentence or 'No evidence'} (PMID: {auto.pmid or 'N/A'})"
                            results["evidence"].append({"text": evidence_text, "source": "AutomatedAnnotation"})
                        else:
                            logger.debug(f"Skipping AutomatedAnnotation prediction with no meaningful data: {prediction}")

            # 8. Occurrences
            conditions = []
            if variant_id:
                conditions.extend([
                    func.lower(Occurrence.object_id).ilike(f"%{variant_id}%"),
                    func.lower(Occurrence.object_name).ilike(f"%{variant}%")
                ])
            if drug:
                conditions.append(func.lower(Occurrence.object_name).ilike(f"%{drug}%") & Occurrence.object_type.ilike("chemical"))
            if phenotype:
                conditions.append(func.lower(Occurrence.source_name).ilike(f"%{phenotype}%") & Occurrence.source_type.ilike("disease"))
            if gene:
                conditions.append(func.lower(Occurrence.object_name).ilike(f"%{gene}%") & Occurrence.object_type.ilike("gene"))

            if conditions:
                query = (
                    db.session.query(Occurrence)
                    .filter(or_(False, *conditions))
                    .limit(3)
                    .all()
                )
                for occ in query:
                    if (not variant or (occ.object_id and matches_any(occ.object_id, variant_id, False)) or (occ.object_name and matches_any(occ.object_name, variant, False))) and \
                       (not drug or (occ.object_type.lower() == "chemical" and occ.object_name and matches_any(occ.object_name, drug, False))) and \
                       (not phenotype or (occ.source_type.lower() == "disease" and occ.source_name and matches_any(occ.source_name, phenotype, False))) and \
                       (not gene or (occ.object_type.lower() == "gene" and occ.object_name and matches_any(occ.object_name, gene, True))):
                        score = 0.8
                        score = normalize_score(score, "Occurrence")
                        prediction = {
                            "effect": f"Mentioned in {occ.source_type or 'Unknown'}",
                            "drug": occ.object_name if occ.object_type.lower() == "chemical" else "N/A",
                            "phenotype": occ.source_name if occ.source_type.lower() == "disease" else "N/A",
                            "gene": occ.object_name if occ.object_type.lower() == "gene" else "N/A",
                            "variant": occ.object_name if occ.object_type.lower() in ["variant", "haplotype"] else "N/A",
                            "source": "Occurrence",
                            "confidence": round(score, 2),
                            "evidence_link": "N/A",
                            "details": {
                                "id": occ.id,
                                "source_type": occ.source_type or "N/A"
                            }
                        }
                        if has_meaningful_data(prediction):
                            results["predictions"].append(prediction)
                            evidence_weight += score
                            evidence_count += 1
                            evidence_text = f"Occurrence: {occ.source_name or 'No source'} mentions {occ.object_name or 'No object'}"
                            results["evidence"].append({"text": evidence_text, "source": "Occurrence"})
                        else:
                            logger.debug(f"Skipping Occurrence prediction with no meaningful data: {prediction}")

            # 9. Drug Interactions
            if drug:
                drug_ids = select(Drug.id).where(
                    or_(
                        func.lower(Drug.name_en).ilike(f"%{drug}%"),
                        func.lower(Drug.name_tr).ilike(f"%{drug}%"),
                        func.lower(Drug.alternative_names).ilike(f"%{drug}%")
                    )
                )
                query = (
                    db.session.query(DrugInteraction)
                    .filter(or_(
                        DrugInteraction.drug1_id.in_(drug_ids),
                        DrugInteraction.drug2_id.in_(drug_ids)
                    ))
                    .limit(3)
                    .all()
                )
                for di in query:
                    drug1 = di.drug1.name_en if di.drug1 else "Unknown"
                    drug2 = di.drug2.name_en if di.drug2 else "Unknown"
                    if matches_any(drug1, drug, False) or matches_any(drug2, drug, False):
                        score = 2.0 if di.severity == "Severe" else 1.5 if di.severity == "Moderate" else 1.0
                        score = normalize_score(score, "DrugInteraction")
                        prediction = {
                            "effect": f"Drug interaction: {di.interaction_type or 'Unknown'}",
                            "drug": f"{drug1} with {drug2}",
                            "phenotype": "N/A",
                            "gene": "N/A",
                            "variant": "N/A",
                            "source": "DrugInteraction",
                            "confidence": round(score, 2),
                            "evidence_link": di.reference or "N/A",
                            "details": {
                                "id": di.id,
                                "severity": di.severity or "N/A",
                                "interaction_type": di.interaction_type or "N/A",
                                "mechanism": di.mechanism or "N/A"
                            }
                        }
                        if has_meaningful_data(prediction):
                            results["predictions"].append(prediction)
                            evidence_weight += score
                            evidence_count += 1
                            evidence_text = f"Drug Interaction: {drug1} and {drug2} ({di.interaction_description or 'No description'})"
                            results["evidence"].append({"text": evidence_text, "source": "DrugInteraction"})
                        else:
                            logger.debug(f"Skipping DrugInteraction prediction with no meaningful data: {prediction}")

            # 10. Drug Receptor Interactions
            if drug or gene:
                drug_ids = select(Drug.id).where(
                    or_(
                        func.lower(Drug.name_en).ilike(f"%{drug}%"),
                        func.lower(Drug.name_tr).ilike(f"%{drug}%"),
                        func.lower(Drug.alternative_names).ilike(f"%{drug}%")
                    )
                ) if drug else None
                conditions = []
                if drug:
                    conditions.append(DrugReceptorInteraction.drug_id.in_(drug_ids))
                if gene:
                    conditions.append(func.lower(Receptor.name).ilike(f"%{gene}%"))
                if conditions:
                    query = (
                        db.session.query(DrugReceptorInteraction)
                        .outerjoin(Drug)
                        .outerjoin(Receptor)
                        .filter(or_(False, *conditions))
                        .limit(3)
                        .all()
                    )
                    for dri in query:
                        drug_name = dri.drug.name_en if dri.drug else "Unknown"
                        receptor_name = dri.receptor.name if dri.receptor else "Unknown"
                        if (not drug or matches_any(drug_name, drug, False)) and \
                           (not gene or matches_any(receptor_name, gene, True)):
                            score = 1.5 if dri.affinity else 1.0
                            score = normalize_score(score, "DrugReceptorInteraction")
                            prediction = {
                                "effect": f"Receptor interaction: {dri.interaction_type or 'Unknown'}",
                                "drug": drug_name,
                                "phenotype": "N/A",
                                "gene": receptor_name,
                                "variant": "N/A",
                                "source": "DrugReceptorInteraction",
                                "confidence": round(score, 2),
                                "evidence_link": dri.pdb_file or "N/A",
                                "details": {
                                    "id": dri.id,
                                    "interaction_type": dri.interaction_type or "N/A",
                                    "affinity": dri.affinity or "N/A"
                                }
                            }
                            if has_meaningful_data(prediction):
                                results["predictions"].append(prediction)
                                evidence_weight += score
                                evidence_count += 1
                                evidence_text = f"Drug-Receptor Interaction: {drug_name} with {receptor_name} ({dri.mechanism or 'No mechanism'})"
                                results["evidence"].append({"text": evidence_text, "source": "DrugReceptorInteraction"})
                            else:
                                logger.debug(f"Skipping DrugReceptorInteraction prediction with no meaningful data: {prediction}")

            # 11. Pathways
            if drug or gene:
                drug_ids = select(Drug.id).where(
                    or_(
                        func.lower(Drug.name_en).ilike(f"%{drug}%"),
                        func.lower(Drug.name_tr).ilike(f"%{drug}%"),
                        func.lower(Drug.alternative_names).ilike(f"%{drug}%")
                    )
                ) if drug else None
                conditions = []
                if drug:
                    conditions.append(Pathway.drugs.any(Drug.id.in_(drug_ids)))
                if gene:
                    conditions.append(func.lower(Pathway.description).ilike(f"%{gene}%"))
                if conditions:
                    query = (
                        db.session.query(Pathway)
                        .outerjoin(Pathway.drugs)
                        .filter(or_(False, *conditions))
                        .limit(3)
                        .all()
                    )
                    for path in query:
                        drugs = [d.name_en for d in path.drugs if d and d.name_en and isinstance(d.name_en, str)] if path.drugs and isinstance(path.drugs, list) else []
                        if (not drug or (drugs and any(matches_any(d, drug, False) for d in drugs))) and \
                           (not gene or (path.description and matches_any(path.description, gene, True))):
                            score = 1.0
                            score = normalize_score(score, "Pathway")
                            prediction = {
                                "effect": f"Pathway involvement: {path.name or 'Unknown'}",
                                "drug": ", ".join(drugs) or "N/A",
                                "phenotype": "N/A",
                                "gene": "N/A",
                                "variant": "N/A",
                                "source": "Pathway",
                                "confidence": round(score, 2),
                                "evidence_link": path.url or "N/A",
                                "details": {
                                    "pathway_id": path.pathway_id or "N/A",
                                    "name": path.name or "N/A",
                                    "organism": path.organism or "N/A"
                                }
                            }
                            if has_meaningful_data(prediction):
                                results["predictions"].append(prediction)
                                evidence_weight += score
                                evidence_count += 1
                                evidence_text = f"Pathway: {path.name or 'No name'} ({path.description or 'No description'})"
                                results["evidence"].append({"text": evidence_text, "source": "Pathway"})
                            else:
                                logger.debug(f"Skipping Pathway prediction with no meaningful data: {prediction}")

            # 12. Drug Details
            if drug:
                drug_query = (
                    db.session.query(Drug)
                    .outerjoin(DrugDetail)
                    .filter(or_(
                        func.lower(Drug.name_en).ilike(f"%{drug}%"),
                        func.lower(Drug.name_tr).ilike(f"%{drug}%"),
                        func.lower(Drug.alternative_names).ilike(f"%{drug}%")
                    ))
                    .limit(3)
                    .all()
                )
                for d in drug_query:
                    detail = d.details[0] if d.details and isinstance(d.details, list) and d.details else None
                    indications = [i.indication.name_en for i in d.disease_interactions if i and i.indication and i.indication.name_en] if d.disease_interactions and isinstance(d.disease_interactions, list) else []
                    results["drug_details"].append({
                        "name_en": d.name_en or "N/A",
                        "name_tr": d.name_tr or "N/A",
                        "alternative_names": d.alternative_names or "N/A",
                        "fda_approved": d.fda_approved or False,
                        "indications": indications or ["N/A"],
                        "mechanism_of_action": detail.mechanism_of_action or "N/A" if detail else "N/A",
                        "molecular_formula": detail.molecular_formula or "N/A" if detail else "N/A",
                        "pharmacodynamics": detail.pharmacodynamics or "N/A" if detail else "N/A",
                        "pharmacokinetics": detail.pharmacokinetics or "N/A" if detail else "N/A",
                        "black_box_warning": detail.black_box_warning or False if detail else False,
                        "black_box_details": detail.black_box_details or "N/A" if detail else "N/A"
                    })

            # 13. Gene Details
            if gene:
                gene_query = (
                    db.session.query(Gene)
                    .filter(func.lower(Gene.gene_symbol).ilike(f"%{gene}%"))
                    .limit(3)
                    .all()
                )
                for g in gene_query:
                    clinical_annotations = [ca.clinical_annotation_id for ca in g.clinical_annotations if ca] if g.clinical_annotations and isinstance(g.clinical_annotations, list) else []
                    variant_annotations = [va.variant_annotation_id for va in g.variant_annotations if va] if g.variant_annotations and isinstance(g.variant_annotations, list) else []
                    drug_labels = [dl.pharmgkb_id for dl in g.drug_labels if dl] if g.drug_labels and isinstance(g.drug_labels, list) else []
                    results["gene_details"].append({
                        "gene_symbol": g.gene_symbol or "N/A",
                        "gene_id": g.gene_id or "N/A",
                        "clinical_annotations": clinical_annotations,
                        "variant_annotations": variant_annotations,
                        "drug_labels": drug_labels
                    })

            # 14. Variant Details
            if variant:
                variant_query = (
                    db.session.query(Variant)
                    .filter(or_(
                        func.lower(Variant.name).ilike(f"%{variant}%"),
                        func.lower(Variant.pharmgkb_id).ilike(f"%{variant}%")
                    ))
                    .limit(3)
                    .all()
                )
                for v in variant_query:
                    clinical_annotations = [ca.clinical_annotation_id for ca in v.clinical_annotations if ca] if v.clinical_annotations and isinstance(v.clinical_annotations, list) else []
                    variant_annotations = [va.variant_annotation_id for va in v.variant_annotations if va] if v.variant_annotations and isinstance(v.variant_annotations, list) else []
                    drug_labels = [dl.pharmgkb_id for dl in v.drug_labels if dl] if v.drug_labels and isinstance(v.drug_labels, list) else []
                    results["variant_details"].append({
                        "name": v.name or "N/A",
                        "pharmgkb_id": v.pharmgkb_id or "N/A",
                        "clinical_annotations": clinical_annotations,
                        "variant_annotations": variant_annotations,
                        "drug_labels": drug_labels
                    })

            # Summarize and Finalize
            results["predictions"] = sorted(results["predictions"], key=lambda x: x["confidence"], reverse=True)[:10]
            
            # Debug: Log evidence before deduplication
            logger.debug(f"Evidence before deduplication: {results['evidence']}")
            
            # Deduplicate evidence while preserving the object structure
            seen_texts = {}
            deduplicated_evidence = []
            for evidence_item in results["evidence"]:
                text = evidence_item["text"].lower()
                if text not in seen_texts:
                    seen_texts[text] = True
                    deduplicated_evidence.append(evidence_item)
            # Sort by text (lowercase) and limit to 10
            deduplicated_evidence = sorted(deduplicated_evidence, key=lambda x: x["text"].lower())[:10]
            results["evidence"] = deduplicated_evidence
            
            # Debug: Log evidence after deduplication
            logger.debug(f"Evidence after deduplication: {results['evidence']}")
            
            results["confidence_score"] = round(min(evidence_weight / max(evidence_count, 1), 1.0), 2) if evidence_count > 0 else 0.0
            results["metadata"]["result_count"] = len(results["predictions"])

            if results["predictions"]:
                top_pred = results["predictions"][0]
                results["summary"] = (
                    f"High-confidence interaction: {top_pred['effect']} (Source: {top_pred['source']}, Confidence: {top_pred['confidence']})"
                    if top_pred["confidence"] >= 0.8 else
                    f"Moderate-confidence interaction: {top_pred['effect']} (Source: {top_pred['source']}, Confidence: {top_pred['confidence']})"
                    if top_pred["confidence"] >= 0.5 else
                    f"Low-confidence interaction: {top_pred['effect']} (Source: {top_pred['source']}, Confidence: {top_pred['confidence']}). Consult a healthcare provider."
                )
            else:
                results["summary"] = "No pharmacogenomic interactions found. Try broadening your search terms."

            logger.info(f"Pharmacogenomics query completed: {results['metadata']['query']}, predictions: {len(results['predictions'])}, evidence: {len(results['evidence'])}, confidence: {results['confidence_score']}")
            return jsonify(results)

        except sqlalchemy.exc.DatabaseError as e:
            logger.error(f"Database error in pharmacogenomics: {str(e)}\n{traceback.format_exc()}")
            return jsonify({"error": "Database error. Please try again later.", "details": str(e)}), 500
        except Exception as e:
            logger.error(f"Pharmacogenomics error: {str(e)}\n{traceback.format_exc()}")
            return jsonify({"error": "Server error. Please try again or contact support.", "details": str(e)}), 500
#Pharmacogenomics Module son...

#PGx Dashboard Başlangıç:
@app.route("/pharmacogenomics/dashboard", methods=["GET", "POST"])
def pharmacogenomics_dashboard():
    with app.app_context():
        try:
            if request.method == "GET":
                return render_template("pharmacogenomics_dashboard.html")

            # Validate JSON input
            data = request.get_json(silent=True) or {}
            if not isinstance(data, dict):
                return jsonify({"error": "Invalid JSON input"}), 400

            # Sanitize and validate inputs
            variant = (data.get("variant") or "").strip()[:100] if data.get("variant") else ""
            drug = (data.get("drug") or "").strip().lower()[:100] if data.get("drug") else ""
            phenotype = (data.get("phenotype") or "").strip().lower()[:100] if data.get("phenotype") else ""
            gene = (data.get("gene") or "").strip().upper()[:50] if data.get("gene") else ""
            level_of_evidence = (data.get("level_of_evidence") or "").strip()[:50] if data.get("level_of_evidence") else ""

            # Validate input lengths
            if len(variant) > 100 or len(drug) > 100 or len(phenotype) > 100 or len(gene) > 50 or len(level_of_evidence) > 50:
                return jsonify({"error": "Input length exceeds maximum allowed characters"}), 400

            results = {
                "status": "success",
                "stats": {
                    "counts": {},
                    "top_entities": {},
                    "trends": {},
                    "child_counts": {},
                    "child_top_entities": {},
                    "child_trends": {},
                    "drug_categories": [],  # New section for drug category distribution
                },
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "query": {"variant": variant, "drug": drug, "phenotype": phenotype, "gene": gene, "level_of_evidence": level_of_evidence}
                }
            }

            def apply_filters(query, model, has_joins=False):
                conditions = []
                # For ClinicalAnnotation, join with related tables to apply filters
                if model == ClinicalAnnotation:
                    if variant and not has_joins:
                        query = query.outerjoin(ClinicalAnnotationVariant).outerjoin(Variant)
                        conditions.append(or_(
                            Variant.name.ilike(f"%{variant}%"),
                            Variant.pharmgkb_id.ilike(f"%{variant}%")
                        ))
                    if drug and not has_joins:
                        query = query.outerjoin(ClinicalAnnotationDrug).outerjoin(Drug)
                        conditions.append(or_(
                            Drug.name_en.ilike(f"%{drug}%"),
                            Drug.name_tr.ilike(f"%{drug}%"),
                            Drug.alternative_names.ilike(f"%{drug}%")
                        ))
                    if phenotype and not has_joins:
                        query = query.outerjoin(ClinicalAnnotationPhenotype).outerjoin(Phenotype)
                        conditions.append(Phenotype.name.ilike(f"%{phenotype}%"))
                    if gene and not has_joins:
                        query = query.outerjoin(ClinicalAnnotationGene).outerjoin(Gene)
                        conditions.append(Gene.gene_symbol.ilike(f"%{gene}%"))
                    if level_of_evidence:
                        conditions.append(ClinicalAnnotation.level_of_evidence.ilike(f"%{level_of_evidence}%"))
                # For DrugLabel
                elif model == DrugLabel:
                    if drug and not has_joins:
                        query = query.outerjoin(DrugLabelDrug, DrugLabelDrug.pharmgkb_id == DrugLabel.pharmgkb_id).outerjoin(Drug, DrugLabelDrug.drug_id == Drug.id)
                        conditions.append(or_(
                            Drug.name_en.ilike(f"%{drug}%"),
                            Drug.name_tr.ilike(f"%{drug}%"),
                            Drug.alternative_names.ilike(f"%{drug}%")
                        ))
                    if phenotype:
                        conditions.append(DrugLabel.name.ilike(f"%{phenotype}%"))
                    if gene and not has_joins:
                        query = query.outerjoin(DrugLabelGene).outerjoin(Gene)
                        conditions.append(Gene.gene_symbol.ilike(f"%{gene}%"))
                    if variant and not has_joins:
                        query = query.outerjoin(DrugLabelVariant).outerjoin(Variant)
                        conditions.append(or_(
                            Variant.name.ilike(f"%{variant}%"),
                            Variant.pharmgkb_id.ilike(f"%{variant}%")
                        ))
                # For ClinicalVariant
                elif model == ClinicalVariant:
                    if variant and not has_joins:
                        query = query.outerjoin(ClinicalVariantVariant).outerjoin(Variant)
                        conditions.append(or_(
                            Variant.name.ilike(f"%{variant}%"),
                            Variant.pharmgkb_id.ilike(f"%{variant}%")
                        ))
                    if drug and not has_joins:
                        query = query.outerjoin(ClinicalVariantDrug).outerjoin(Drug)
                        conditions.append(or_(
                            Drug.name_en.ilike(f"%{drug}%"),
                            Drug.name_tr.ilike(f"%{drug}%"),
                            Drug.alternative_names.ilike(f"%{drug}%")
                        ))
                    if phenotype and not has_joins:
                        query = query.outerjoin(ClinicalVariantPhenotype).outerjoin(Phenotype)
                        conditions.append(Phenotype.name.ilike(f"%{phenotype}%"))
                    if gene and not has_joins:
                        query = query.outerjoin(Gene)
                        conditions.append(Gene.gene_symbol.ilike(f"%{gene}%"))
                # For VariantAnnotation (used indirectly via child tables)
                elif model in [VariantFAAnn, VariantDrugAnn, VariantPhenoAnn]:
                    if not has_joins:
                        query = query.join(VariantAnnotation, model.variant_annotation_id == VariantAnnotation.variant_annotation_id)
                    if variant:
                        conditions.append(or_(
                            model.variant_annotation_id.ilike(f"%{variant}%"),
                            model.alleles.ilike(f"%{variant}%")
                        ))
                    if drug and model in [VariantDrugAnn, VariantFAAnn]:
                        conditions.append(model.sentence.ilike(f"%{drug}%"))
                    if phenotype and model == VariantPhenoAnn:
                        conditions.append(model.phenotype.ilike(f"%{phenotype}%"))
                    if gene and not has_joins:
                        query = query.outerjoin(VariantAnnotationGene).outerjoin(Gene)
                        conditions.append(or_(
                            model.notes.ilike(f"%{gene}%"),
                            Gene.gene_symbol.ilike(f"%{gene}%")
                        ))
                # For StudyParameters
                elif model == StudyParameters:
                    if variant:
                        conditions.append(StudyParameters.variant_annotation_id.ilike(f"%{variant}%"))
                    if phenotype:
                        conditions.append(StudyParameters.characteristics.ilike(f"%{phenotype}%"))
                # For Relationship
                elif model == Relationship:
                    if variant:
                        conditions.extend([
                            Relationship.entity1_id.ilike(f"%{variant}%"),
                            Relationship.entity2_id.ilike(f"%{variant}%")
                        ])
                    if drug:
                        conditions.extend([
                            (func.lower(Relationship.entity1_name).ilike(f"%{drug}%") & Relationship.entity1_type.ilike("chemical")),
                            (func.lower(Relationship.entity2_name).ilike(f"%{drug}%") & Relationship.entity2_type.ilike("chemical"))
                        ])
                    if phenotype:
                        conditions.extend([
                            (func.lower(Relationship.entity1_name).ilike(f"%{phenotype}%") & Relationship.entity1_type.ilike("disease")),
                            (func.lower(Relationship.entity2_name).ilike(f"%{phenotype}%") & Relationship.entity2_type.ilike("disease"))
                        ])
                    if gene:
                        conditions.extend([
                            (func.lower(Relationship.entity1_name).ilike(f"%{gene}%") & Relationship.entity1_type.ilike("gene")),
                            (func.lower(Relationship.entity2_name).ilike(f"%{gene}%") & Relationship.entity2_type.ilike("gene"))
                        ])
                # For AutomatedAnnotation
                elif model == AutomatedAnnotation:
                    if variant:
                        conditions.append(or_(
                            AutomatedAnnotation.variation_id.ilike(f"%{variant}%"),
                            AutomatedAnnotation.variation_name.ilike(f"%{variant}%")
                        ))
                    if drug:
                        conditions.append(or_(
                            AutomatedAnnotation.chemical_name.ilike(f"%{drug}%"),
                            AutomatedAnnotation.chemical_in_text.ilike(f"%{drug}%")
                        ))
                    if phenotype:
                        conditions.append(AutomatedAnnotation.sentence.ilike(f"%{phenotype}%"))
                    if gene:
                        conditions.append(or_(
                            AutomatedAnnotation.gene_ids.ilike(f"%{gene}%"),
                            AutomatedAnnotation.gene_symbols.ilike(f"%{gene}%")
                        ))
                # For Occurrence
                elif model == Occurrence:
                    if variant:
                        conditions.extend([
                            Occurrence.object_id.ilike(f"%{variant}%"),
                            Occurrence.object_name.ilike(f"%{variant}%")
                        ])
                    if drug:
                        conditions.append(func.lower(Occurrence.object_name).ilike(f"%{drug}%") & Occurrence.object_type.ilike("chemical"))
                    if phenotype:
                        conditions.append(func.lower(Occurrence.source_name).ilike(f"%{phenotype}%") & Occurrence.source_type.ilike("disease"))
                    if gene:
                        conditions.append(func.lower(Occurrence.object_name).ilike(f"%{gene}%") & Occurrence.object_type.ilike("gene"))
                return query.filter(*conditions) if conditions else query

            # Original Counts
            results["stats"]["counts"]["clinical_annotations"] = apply_filters(db.session.query(ClinicalAnnotation), ClinicalAnnotation).count()
            results["stats"]["counts"]["variant_annotations"] = apply_filters(db.session.query(VariantAnnotation), VariantAnnotation).count()
            rel_query = apply_filters(db.session.query(Relationship), Relationship)
            total_relationships = rel_query.count()
            gene_gene = rel_query.filter(Relationship.entity1_type.ilike("gene"), Relationship.entity2_type.ilike("gene")).count()
            gene_drug = rel_query.filter(or_(
                and_(Relationship.entity1_type.ilike("gene"), Relationship.entity2_type.ilike("chemical")),
                and_(Relationship.entity1_type.ilike("chemical"), Relationship.entity2_type.ilike("gene"))
            )).count()
            drug_phenotype = rel_query.filter(or_(
                and_(Relationship.entity1_type.ilike("chemical"), Relationship.entity2_type.ilike("disease")),
                and_(Relationship.entity1_type.ilike("disease"), Relationship.entity2_type.ilike("chemical"))
            )).count()
            results["stats"]["counts"]["relationships"] = {
                "total": total_relationships,
                "gene_gene": gene_gene,
                "gene_drug": gene_drug,
                "drug_phenotype": drug_phenotype,
                "other": total_relationships - (gene_gene + gene_drug + drug_phenotype)
            }
            results["stats"]["counts"]["drug_labels"] = apply_filters(db.session.query(DrugLabel), DrugLabel).count()
            results["stats"]["counts"]["clinical_variants"] = apply_filters(db.session.query(ClinicalVariant), ClinicalVariant).count()
            results["stats"]["counts"]["occurrences"] = apply_filters(db.session.query(Occurrence), Occurrence).count()
            results["stats"]["counts"]["automated_annotations"] = apply_filters(db.session.query(AutomatedAnnotation), AutomatedAnnotation).count()

            # Child Table Counts
            results["stats"]["child_counts"]["study_parameters"] = apply_filters(db.session.query(StudyParameters), StudyParameters).count()
            results["stats"]["child_counts"]["variant_fa_ann"] = apply_filters(db.session.query(VariantFAAnn), VariantFAAnn, has_joins=True).count()
            results["stats"]["child_counts"]["variant_drug_ann"] = apply_filters(db.session.query(VariantDrugAnn), VariantDrugAnn, has_joins=True).count()
            results["stats"]["child_counts"]["variant_pheno_ann"] = apply_filters(db.session.query(VariantPhenoAnn), VariantPhenoAnn, has_joins=True).count()

            # Original Top Entities
            # Top Genes (Join with ClinicalAnnotationGene and Gene)
            base_query = (db.session.query(Gene.gene_symbol, func.count(ClinicalAnnotation.clinical_annotation_id).label('count'))
                         .join(ClinicalAnnotationGene, ClinicalAnnotationGene.gene_id == Gene.gene_id)
                         .join(ClinicalAnnotation, ClinicalAnnotationGene.clinical_annotation_id == ClinicalAnnotation.clinical_annotation_id))
            top_genes_query = apply_filters(base_query, ClinicalAnnotation, has_joins=True)
            top_genes = top_genes_query.group_by(Gene.gene_symbol).order_by(func.count(ClinicalAnnotation.clinical_annotation_id).desc()).limit(5).all()
            results["stats"]["top_entities"]["genes"] = [{"name": g[0], "count": g[1]} for g in top_genes if g[0]]

            # Top Drugs (Join with DrugLabelDrug and Drug)
            base_query = (db.session.query(Drug.name_en, func.count(DrugLabel.pharmgkb_id).label('count'))
                         .join(DrugLabelDrug, DrugLabelDrug.drug_id == Drug.id)
                         .join(DrugLabel, DrugLabelDrug.pharmgkb_id == DrugLabel.pharmgkb_id))
            top_drugs_query = apply_filters(base_query, DrugLabel, has_joins=True)
            top_drugs = top_drugs_query.group_by(Drug.name_en).order_by(func.count(DrugLabel.pharmgkb_id).desc()).limit(5).all()
            results["stats"]["top_entities"]["drugs"] = [{"name": d[0], "count": d[1]} for d in top_drugs if d[0]]

            # Top Variants (Join with ClinicalAnnotationVariant and Variant)
            base_query = (db.session.query(Variant.name, func.count(ClinicalAnnotation.clinical_annotation_id).label('count'))
                         .join(ClinicalAnnotationVariant, ClinicalAnnotationVariant.variant_id == Variant.pharmgkb_id)
                         .join(ClinicalAnnotation, ClinicalAnnotationVariant.clinical_annotation_id == ClinicalAnnotation.clinical_annotation_id))
            top_variants_query = apply_filters(base_query, ClinicalAnnotation, has_joins=True)
            top_variants = top_variants_query.group_by(Variant.name).order_by(func.count(ClinicalAnnotation.clinical_annotation_id).desc()).limit(5).all()
            results["stats"]["top_entities"]["variants"] = [{"name": v[0], "count": v[1]} for v in top_variants if v[0]]

            # Top Phenotypes (Join with ClinicalAnnotationPhenotype and Phenotype)
            base_query = (db.session.query(Phenotype.name, func.count(ClinicalAnnotation.clinical_annotation_id).label('count'))
                         .join(ClinicalAnnotationPhenotype, ClinicalAnnotationPhenotype.phenotype_id == Phenotype.id)
                         .join(ClinicalAnnotation, ClinicalAnnotationPhenotype.clinical_annotation_id == ClinicalAnnotation.clinical_annotation_id))
            phenotypes_query = apply_filters(base_query, ClinicalAnnotation, has_joins=True)
            top_phenotypes = phenotypes_query.group_by(Phenotype.name).order_by(func.count(ClinicalAnnotation.clinical_annotation_id).desc()).limit(5).all()
            results["stats"]["top_entities"]["phenotypes"] = [{"name": p[0], "count": p[1]} for p in top_phenotypes if p[0]]

            # Top Sources (from Occurrences)
            top_sources = apply_filters(db.session.query(Occurrence.source_type, func.count(Occurrence.source_type).label('count')), Occurrence)\
                .group_by(Occurrence.source_type).order_by(func.count(Occurrence.source_type).desc()).limit(5).all()
            results["stats"]["top_entities"]["sources"] = [{"name": s[0] or "Unknown", "count": s[1]} for s in top_sources]

            # Top Chemicals (from Automated Annotations)
            top_chemicals = apply_filters(db.session.query(AutomatedAnnotation.chemical_name, func.count(AutomatedAnnotation.chemical_name).label('count')), AutomatedAnnotation)\
                .group_by(AutomatedAnnotation.chemical_name).order_by(func.count(AutomatedAnnotation.chemical_name).desc()).limit(5).all()
            results["stats"]["top_entities"]["chemicals"] = [{"name": c[0] or "Unknown", "count": c[1]} for c in top_chemicals]

            # New: Top Alleles (from ClinicalAnnAllele)
            base_query = (db.session.query(ClinicalAnnAllele.genotype_allele, func.count(ClinicalAnnAllele.genotype_allele).label('count'))
                         .join(ClinicalAnnotation, ClinicalAnnAllele.clinical_annotation_id == ClinicalAnnotation.clinical_annotation_id))
            top_alleles_query = apply_filters(base_query, ClinicalAnnotation, has_joins=True)
            top_alleles = top_alleles_query.group_by(ClinicalAnnAllele.genotype_allele).order_by(func.count(ClinicalAnnAllele.genotype_allele).desc()).limit(5).all()
            results["stats"]["top_entities"]["alleles"] = [{"name": a[0], "count": a[1]} for a in top_alleles if a[0]]

            # New: Drug Category Distribution
            base_query = (db.session.query(DrugCategory.name, func.count(Drug.id).label('count'))
                         .join(Drug, Drug.category_id == DrugCategory.id))
            if drug:
                base_query = base_query.join(DrugLabelDrug, DrugLabelDrug.drug_id == Drug.id).join(DrugLabel, DrugLabelDrug.pharmgkb_id == DrugLabel.pharmgkb_id)
                base_query = base_query.filter(or_(
                    Drug.name_en.ilike(f"%{drug}%"),
                    Drug.name_tr.ilike(f"%{drug}%"),
                    Drug.alternative_names.ilike(f"%{drug}%")
                ))
            if gene:
                base_query = base_query.join(ClinicalAnnotationDrug, ClinicalAnnotationDrug.drug_id == Drug.id)\
                                      .join(ClinicalAnnotationGene, ClinicalAnnotationGene.clinical_annotation_id == ClinicalAnnotation.clinical_annotation_id)\
                                      .join(Gene, ClinicalAnnotationGene.gene_id == Gene.gene_id)
                base_query = base_query.filter(Gene.gene_symbol.ilike(f"%{gene}%"))
            drug_categories = base_query.group_by(DrugCategory.name).order_by(func.count(Drug.id).desc()).all()
            results["stats"]["drug_categories"] = [{"name": c[0], "count": c[1]} for c in drug_categories if c[0]]

            # Trends for Main Tables
            if hasattr(ClinicalAnnotation, 'created_at'):
                trends_ca = apply_filters(db.session.query(extract('year', ClinicalAnnotation.created_at).label('year'), func.count().label('count')), ClinicalAnnotation)\
                    .group_by('year').order_by('year').all()
                results["stats"]["trends"]["annotations_by_year"] = [{"year": int(t[0]), "count": t[1]} for t in trends_ca if t[0]]

            if hasattr(DrugLabel, 'created_at'):
                trends_dl = apply_filters(db.session.query(extract('year', DrugLabel.created_at).label('year'), func.count().label('count')), DrugLabel)\
                    .group_by('year').order_by('year').all()
                results["stats"]["trends"]["drug_labels_by_year"] = [{"year": int(t[0]), "count": t[1]} for t in trends_dl if t[0]]

            if hasattr(Relationship, 'created_at'):
                trends_rel = apply_filters(db.session.query(extract('year', Relationship.created_at).label('year'), func.count().label('count')), Relationship)\
                    .group_by('year').order_by('year').all()
                results["stats"]["trends"]["relationships_by_year"] = [{"year": int(t[0]), "count": t[1]} for t in trends_rel if t[0]]

            # Child Table Top Entities
            # Top variant_annotation_ids (from StudyParameters)
            top_variants_child = apply_filters(db.session.query(StudyParameters.variant_annotation_id, func.count(StudyParameters.variant_annotation_id).label('count')), StudyParameters)\
                .group_by(StudyParameters.variant_annotation_id).order_by(func.count(StudyParameters.variant_annotation_id).desc()).limit(5).all()
            results["stats"]["child_top_entities"]["variants"] = [{"name": v[0], "count": v[1]} for v in top_variants_child if v[0]]

            # Top Study Types (from StudyParameters)
            top_study_types = apply_filters(db.session.query(StudyParameters.study_type, func.count(StudyParameters.study_type).label('count')), StudyParameters)\
                .group_by(StudyParameters.study_type).order_by(func.count(StudyParameters.study_type).desc()).limit(5).all()
            results["stats"]["child_top_entities"]["study_types"] = [{"name": s[0] or "Unknown", "count": s[1]} for s in top_study_types]

            # Top Sentences (from VariantFAAnn)
            base_query = db.session.query(VariantFAAnn.sentence, func.count(VariantFAAnn.sentence).label('count')).join(VariantAnnotation, VariantFAAnn.variant_annotation_id == VariantAnnotation.variant_annotation_id)
            top_fa_sentences_query = apply_filters(base_query, VariantFAAnn, has_joins=True)
            top_fa_sentences = top_fa_sentences_query.group_by(VariantFAAnn.sentence).order_by(func.count(VariantFAAnn.sentence).desc()).limit(5).all()
            results["stats"]["child_top_entities"]["fa_sentences"] = [{"name": s[0] or "Unknown", "count": s[1]} for s in top_fa_sentences]

            # Top Sentences (from VariantDrugAnn)
            base_query = db.session.query(VariantDrugAnn.sentence, func.count(VariantDrugAnn.sentence).label('count')).join(VariantAnnotation, VariantDrugAnn.variant_annotation_id == VariantAnnotation.variant_annotation_id)
            top_drug_sentences_query = apply_filters(base_query, VariantDrugAnn, has_joins=True)
            top_drug_sentences = top_drug_sentences_query.group_by(VariantDrugAnn.sentence).order_by(func.count(VariantDrugAnn.sentence).desc()).limit(5).all()
            results["stats"]["child_top_entities"]["drug_sentences"] = [{"name": s[0] or "Unknown", "count": s[1]} for s in top_drug_sentences]

            # Top Phenotypes (from VariantPhenoAnn)
            base_query = db.session.query(VariantPhenoAnn.phenotype, func.count(VariantPhenoAnn.phenotype).label('count')).join(VariantAnnotation, VariantPhenoAnn.variant_annotation_id == VariantAnnotation.variant_annotation_id)
            top_phenotypes_query = apply_filters(base_query, VariantPhenoAnn, has_joins=True)
            top_phenotypes_child = top_phenotypes_query.group_by(VariantPhenoAnn.phenotype).order_by(func.count(VariantPhenoAnn.phenotype).desc()).limit(5).all()
            results["stats"]["child_top_entities"]["phenotypes"] = [{"name": p[0] or "Unknown", "count": p[1]} for p in top_phenotypes_child]

            # Child Table Trends
            if hasattr(StudyParameters, 'created_at'):
                trends_sp = apply_filters(db.session.query(extract('year', StudyParameters.created_at).label('year'), func.count().label('count')), StudyParameters)\
                    .group_by('year').order_by('year').all()
                results["stats"]["child_trends"]["studies_by_year"] = [{"year": int(t[0]), "count": t[1]} for t in trends_sp if t[0]]

            if hasattr(VariantFAAnn, 'created_at'):
                base_query = db.session.query(extract('year', VariantFAAnn.created_at).label('year'), func.count().label('count')).join(VariantAnnotation, VariantFAAnn.variant_annotation_id == VariantAnnotation.variant_annotation_id)
                trends_fa_query = apply_filters(base_query, VariantFAAnn, has_joins=True)
                trends_fa = trends_fa_query.group_by('year').order_by('year').all()
                results["stats"]["child_trends"]["fa_annotations_by_year"] = [{"year": int(t[0]), "count": t[1]} for t in trends_fa if t[0]]

            if hasattr(VariantDrugAnn, 'created_at'):
                base_query = db.session.query(extract('year', VariantDrugAnn.created_at).label('year'), func.count().label('count')).join(VariantAnnotation, VariantDrugAnn.variant_annotation_id == VariantAnnotation.variant_annotation_id)
                trends_da_query = apply_filters(base_query, VariantDrugAnn, has_joins=True)
                trends_da = trends_da_query.group_by('year').order_by('year').all()
                results["stats"]["child_trends"]["drug_annotations_by_year"] = [{"year": int(t[0]), "count": t[1]} for t in trends_da if t[0]]

            if hasattr(VariantPhenoAnn, 'created_at'):
                base_query = db.session.query(extract('year', VariantPhenoAnn.created_at).label('year'), func.count().label('count')).join(VariantAnnotation, VariantPhenoAnn.variant_annotation_id == VariantAnnotation.variant_annotation_id)
                trends_pa_query = apply_filters(base_query, VariantPhenoAnn, has_joins=True)
                trends_pa = trends_pa_query.group_by('year').order_by('year').all()
                results["stats"]["child_trends"]["pheno_annotations_by_year"] = [{"year": int(t[0]), "count": t[1]} for t in trends_pa if t[0]]

            logger.info(f"Dashboard stats computed: {results['metadata']['query']}")
            return jsonify(results)

        except sqlalchemy.exc.DatabaseError as e:
            logger.error(f"Database error in pharmacogenomics dashboard: {str(e)}\n{traceback.format_exc()}")
            return jsonify({"status": "error", "error": "Database error. Please try again later.", "details": str(e)}), 500
        except Exception as e:
            logger.error(f"Pharmacogenomics dashboard error: {str(e)}\n{traceback.format_exc()}")
            return jsonify({"status": "error", "error": "Server error. Please try again or contact support.", "details": str(e)}), 500
#PGx Dashboard SON...        

@app.route('/fetch_kegg_pathways', methods=['POST'])
def fetch_kegg_pathways_route():
    try:
        print("Fetching KEGG Pathways...")  # Debugging statement
        save_kegg_pathways_to_db()
        flash("KEGG pathways başarıyla güncellendi.", "success")
        print("KEGG Pathways updated successfully!")  # Debugging statement
    except Exception as e:
        flash(f"Bir hata oluştu: {e}", "danger")
        print(f"Error: {e}")  # Debugging statement
    return redirect(url_for('pathways'))


def fetch_kegg_pathways(organism='hsa'):
    """Fetch pathway data from KEGG API."""
    url = f"https://rest.kegg.jp/list/pathway/{organism}"
    response = requests.get(url)
    
    if response.status_code == 200:
        pathways = []
        for line in response.text.strip().split('\n'):
            try:
                # Split line into ID and name
                parts = line.split('\t')
                if len(parts) != 2:
                    print(f"Skipping invalid line (no tab separator): {line}")
                    continue
                
                pathway_id = parts[0].strip()  # Use the raw ID
                name = parts[1].strip()  # Pathway name
                
                print(f"Processing pathway ID: {pathway_id}, Name: {name}")  # Debugging
                
                pathways.append({
                    'pathway_id': pathway_id,  # No splitting needed
                    'name': name,
                    'organism': organism,
                    'url': f"https://www.kegg.jp/kegg-bin/show_pathway?{pathway_id}"
                })
            except Exception as e:
                print(f"Error processing line: {line}. Error: {e}")
        return pathways
    else:
        print(f"API Error: {response.status_code}")
        return []



def save_kegg_pathways_to_db():
    print("Fetching pathways from KEGG API...")  # Debugging statement
    pathways = fetch_kegg_pathways()
    print(f"Fetched {len(pathways)} pathways.")  # Debugging statement

    for pathway in pathways:
        print(f"Processing pathway: {pathway['name']}")  # Debugging statement
        existing_pathway = Pathway.query.filter_by(pathway_id=pathway['pathway_id']).first()
        if not existing_pathway:
            new_pathway = Pathway(
                pathway_id=pathway['pathway_id'],
                name=pathway['name'],
                organism=pathway['organism'],
                url=pathway['url']
            )
            db.session.add(new_pathway)
    db.session.commit()
    print("KEGG pathways veritabanına kaydedildi.")  # Debugging statement


@app.route('/pathways', methods=['GET'])
def pathways():
    pathways = Pathway.query.all()
    return render_template('pathways.html', pathways=pathways)


@app.route('/pathways/add_drug', methods=['POST'])
def add_drug_to_pathway():
    pathway_id = request.form.get('pathway_id')
    drug_id = request.form.get('drug_id')

    pathway = Pathway.query.get(pathway_id)
    drug = Drug.query.get(drug_id)

    if pathway and drug:
        pathway.drugs.append(drug)
        db.session.commit()
        flash(f"{drug.name_en} ilacı {pathway.name} pathway'ine eklendi.", "success")
    else:
        flash("Pathway veya ilaç bulunamadı.", "danger")

    return redirect(url_for('pathways'))



@app.route('/pathways/<int:pathway_id>', methods=['GET'])
def pathway_details(pathway_id):
    pathway = Pathway.query.get_or_404(pathway_id)

    # Fetch pathway details from KEGG
    kegg_response = requests.get(f"https://rest.kegg.jp/get/{pathway.pathway_id}")
    kegg_details = (
        parse_kegg_details(kegg_response.text)
        if kegg_response.status_code == 200
        else {"Error": "KEGG data not available"}
    )

    return render_template(
        'pathway_details.html',
        pathway=pathway,
        kegg_details=kegg_details
    )





@app.route('/unlink_drug_from_pathway', methods=['POST'])
def unlink_drug_from_pathway():
    pathway_id = request.form.get('pathway_id')
    drug_id = request.form.get('drug_id')

    pathway = Pathway.query.get(pathway_id)
    drug = Drug.query.get(drug_id)

    if pathway and drug:
        pathway.drugs.remove(drug)
        db.session.commit()
        flash(f"{drug.name_en} successfully unlinked from {pathway.name}.", "success")
    else:
        flash("Pathway or drug not found.", "danger")

    return redirect(url_for('pathways'))

def parse_kegg_details(raw_data):
    """Parse KEGG pathway details, highlighting references and genes."""
    details = {}
    current_key = None
    genes = []
    references = []

    for line in raw_data.split("\n"):
        if line.startswith(" "):  # Continuation of the previous key
            if current_key == "GENE" and genes:
                # Extract genes from continuation lines
                genes[-1] += f" {line.strip()}"
            elif current_key == "REFERENCE" and references:
                # Extract references from continuation lines
                references[-1] += f" {line.strip()}"
            elif current_key in details:
                details[current_key] += f" {line.strip()}"
        else:
            parts = line.split("  ", 1)
            if len(parts) == 2:
                current_key = parts[0].strip()
                if current_key == "GENE":
                    # Add genes
                    genes.extend([gene.strip() for gene in parts[1].split(",")])
                elif current_key == "REFERENCE":
                    # Add references
                    references.append(parts[1].strip())
                elif current_key != "DRUG":  # Exclude the DRUG section
                    details[current_key] = parts[1].strip()

    details["GENES"] = genes  # Store genes separately
    details["REFERENCES"] = references  # Store references separately
    return details



@app.route('/news/manage', methods=['GET', 'POST'])
@admin_required
def manage_news():
    if request.method == 'POST':
        # Add new news entry
        title = request.form['title']
        description = request.form['description']
        category = request.form['category']
        publication_date = request.form['publication_date']
        news_item = News(
            title=title,
            description=description,
            category=category,
            publication_date=datetime.strptime(publication_date, '%Y-%m-%d')
        )
        db.session.add(news_item)
        db.session.commit()
        flash("News item added successfully.", "success")
        return redirect(url_for('manage_news'))

    # Fetch announcements and updates
    announcements = News.query.filter_by(category='Announcement').order_by(News.publication_date.desc()).all()
    updates = News.query.filter_by(category='Update').order_by(News.publication_date.desc()).all()
    return render_template('manage_news.html', announcements=announcements, updates=updates)

# Route for editing news
@app.route('/news/edit/<int:news_id>', methods=['GET', 'POST'])
@admin_required
def edit_news(news_id):
    news_item = News.query.get_or_404(news_id)
    if request.method == 'POST':
        # Update the news item
        news_item.title = request.form['title']
        news_item.description = request.form['description']
        news_item.category = request.form['category']
        news_item.publication_date = datetime.strptime(request.form['publication_date'], '%Y-%m-%d')
        db.session.commit()
        flash("News item updated successfully.", "success")
        return redirect(url_for('manage_news'))

    return render_template('edit_news.html', news_item=news_item)

# Route for deleting news
@app.route('/news/delete/<int:news_id>', methods=['POST'])
@admin_required
def delete_news(news_id):
    news_item = News.query.get_or_404(news_id)
    db.session.delete(news_item)
    db.session.commit()
    flash("News item deleted successfully.", "success")
    return redirect(url_for('manage_news'))



# Doz - Cevap Simülasyonu....
# Request model (Pydantic V2)
class DoseResponseRequest(BaseModel):
    emax: float
    ec50: float
    n: float  # Hill coefficient
    concentrations: List[float]

    @field_validator('emax', 'ec50', 'n')
    @classmethod
    def check_positive(cls, v):
        if v <= 0:
            raise ValueError("Value must be positive")
        return v

    @field_validator('concentrations')
    @classmethod
    def check_concentrations(cls, v):
        if not v:
            raise ValueError("Concentrations list cannot be empty")
        if any(c < 0 for c in v):
            raise ValueError("Concentrations must be non-negative")
        return v

# Response model
class DoseResponsePoint(BaseModel):
    concentration: float
    effect: float

class DoseResponseResponse(BaseModel):
    data: List[DoseResponsePoint]
    metadata: dict

@app.route('/simulate-dose-response', methods=['POST'])
def simulate_dose_response():
    """
    Simulates a dose-response curve based on the Hill equation.
    Returns a list of concentration-effect pairs with metadata.
    """
    try:
        # Parse and validate request data
        request_data = request.get_json()
        if not request_data:
            logger.error("No JSON data provided in request")
            return jsonify({"error": "No data provided"}), HTTPStatus.BAD_REQUEST

        input_data = DoseResponseRequest.model_validate(request_data)
        logger.info(f"Received valid request: {input_data.dict()}")

        # Calculate dose-response using Hill equation
        effects = [
            (input_data.emax * (c ** input_data.n)) / 
            ((input_data.ec50 ** input_data.n) + (c ** input_data.n))
            for c in input_data.concentrations
        ]

        # Handle numerical stability
        effects = [min(max(e, 0.0), input_data.emax) for e in effects]

        # Prepare response data
        results = [
            DoseResponsePoint(concentration=c, effect=e)
            for c, e in zip(input_data.concentrations, effects)
        ]

        # Add metadata
        metadata = {
            "model": "Hill Equation",
            "parameters": {
                "Emax": input_data.emax,
                "EC50": input_data.ec50,
                "Hill_Coefficient": input_data.n
            },
            "concentration_range": {
                "min": min(input_data.concentrations),
                "max": max(input_data.concentrations)
            },
            "timestamp": "2025-02-26T00:00:00Z"
        }

        response = DoseResponseResponse(data=results, metadata=metadata)
        logger.info("Simulation completed successfully")

        return jsonify(response.dict()), HTTPStatus.OK

    except ValidationError as ve:
        logger.error(f"Validation error: {ve.errors()}")
        return jsonify({"error": "Invalid input", "details": ve.errors()}), HTTPStatus.BAD_REQUEST

    except Exception as e:
        logger.exception(f"Unexpected error: {str(e)}")
        return jsonify({"error": "Internal server error", "details": str(e)}), HTTPStatus.INTERNAL_SERVER_ERROR

@app.route('/simulate-dose-response', methods=['GET'])
def serve_form():
    """
    Serve the static HTML form for dose-response simulation.
    """
    try:
        return send_from_directory(app.template_folder, 'simulate_dose_response.html')
    except Exception as e:
        logger.error(f"Error serving HTML: {str(e)}")
        return jsonify({"error": "Could not load form"}), HTTPStatus.NOT_FOUND
#Doz - Cevap Son...



# Reseptör - Ligand Etkileşim Kodları...
@app.route('/api/receptors', methods=['GET'])
def get_receptors():
    search = request.args.get('search', '').strip()
    limit = request.args.get('limit', 10, type=int)
    page = request.args.get('page', 1, type=int)

    query = Receptor.query.filter(
        Receptor.name.ilike(f"%{search}%")
    ) if search else Receptor.query

    paginated_query = query.paginate(page=page, per_page=limit)
    results = [{"id": receptor.id, "text": receptor.name} for receptor in paginated_query.items]

    return jsonify({
        "results": results,
        "has_next": paginated_query.has_next
    })

@app.route('/receptor-ligand-simulator', methods=['GET', 'POST'])
def receptor_ligand_simulator():
    if request.method == 'GET':
        return render_template('receptor_ligand_simulator.html')
    
    if request.method == 'POST':
        receptor = request.form.get('receptor')
        ligand = request.form.get('ligand')
        
        if ligand:
            session['ligand'] = ligand
        else:
            ligand = session.get('ligand')

        if not receptor or not ligand:
            return jsonify({"error": "Receptor and ligand data are required."}), 400

        try:
            receptor_file_path = os.path.join('static', f'receptor_{uuid.uuid4().hex}.pdb')
            with open(receptor_file_path, "w") as rec_file:
                rec_file.write(receptor)

            ligand_file_url = session.get('ligand_file_url')
            if not ligand_file_url:
                ligand_file_path = os.path.join('static', f'ligand_{uuid.uuid4().hex}.pdb')
                with open(ligand_file_path, "w") as lig_file:
                    lig_file.write(ligand)
                session['ligand_file_url'] = f"/{ligand_file_path}"
                ligand_file_url = session['ligand_file_url']

            return jsonify({
                "message": "Receptor and ligand processed successfully.",
                "receptor_file_url": f"/{receptor_file_path}",
                "ligand_file_url": ligand_file_url
            }), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

@app.route('/api/convert_ligand', methods=['GET'])
def convert_ligand():
    drug_id = request.args.get('drug_id')
    if not drug_id:
        return jsonify({"error": "Drug ID is required."}), 400

    try:
        drug_id = int(drug_id)
    except ValueError:
        return jsonify({"error": "Invalid Drug ID format."}), 400

    drug_detail = DrugDetail.query.filter_by(drug_id=drug_id).first()
    if not drug_detail or not drug_detail.smiles:
        return jsonify({"error": "SMILES not available for this DrugDetail."}), 404

    try:
        smiles_file = os.path.abspath(f"static/smiles_{uuid.uuid4().hex}.smi")
        pdb_file = os.path.abspath(f"static/ligand_{uuid.uuid4().hex}.pdb")

        with open(smiles_file, "w") as file:
            file.write(drug_detail.smiles)

        subprocess.run(
            ['/opt/homebrew/bin/obabel', smiles_file, '-O', pdb_file, '--gen3d'],
            check=True
        )

        if not os.path.exists(pdb_file):
            return jsonify({"error": "Failed to generate 3D PDB file."}), 500

        with open(pdb_file, 'r') as pdb:
            pdb_content = pdb.read()

        return jsonify({"pdb": pdb_content}), 200
    except subprocess.CalledProcessError as e:
        return jsonify({"error": "Open Babel conversion failed.", "details": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {e}"}), 500

# Updated endpoint with fpocket for binding site prediction
@app.route('/api/get_receptor_structure', methods=['GET'])
def get_receptor_structure():
    receptor_id = request.args.get('receptor_id')
    if not receptor_id:
        return jsonify({"error": "Receptor ID is required."}), 400

    receptor = Receptor.query.get(receptor_id)
    if not receptor or not receptor.pdb_ids:
        return jsonify({"error": "Receptor not found or no PDB IDs available."}), 404

    pdb_id = receptor.pdb_ids.split(",")[0]
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    response = requests.get(url)

    if response.status_code != 200:
        return jsonify({"error": f"Failed to fetch PDB structure for {pdb_id}."}), 500

    pdb_content = response.text

    # Predict binding site with fpocket
    binding_site_coords = get_pocket_coords(pdb_content, pdb_id)
    print(f"Binding site for {pdb_id}: {binding_site_coords}")

    # Update receptor with binding site coords (optional, for persistence)
    receptor.binding_site_x = binding_site_coords["x"]
    receptor.binding_site_y = binding_site_coords["y"]
    receptor.binding_site_z = binding_site_coords["z"]
    db.session.commit()

    return jsonify({
        "pdb": pdb_content,
        "binding_site": binding_site_coords
    }), 200

def get_pocket_coords(pdb_content, pdb_id):
    """Run fpocket to predict binding site from PDB."""
    try:
        temp_pdb = f"temp_{pdb_id}.pdb"
        with open(temp_pdb, "w") as f:
            f.write(pdb_content)

        subprocess.run(["fpocket", "-f", temp_pdb], check=True)

        pocket_file = f"temp_{pdb_id}_out/pockets/pocket1_atm.pdb"
        if not os.path.exists(pocket_file):
            raise FileNotFoundError("fpocket output not found")

        with open(pocket_file, "r") as f:
            lines = [l for l in f if l.startswith("ATOM")]
            if not lines:
                raise ValueError("No atoms found in pocket file")
            coords = [list(map(float, l[30:54].split())) for l in lines]
            x = sum(c[0] for c in coords) / len(coords)
            y = sum(c[1] for c in coords) / len(coords)
            z = sum(c[2] for c in coords) / len(coords)
            return {"x": x, "y": y, "z": z}
    except Exception as e:
        print(f"fpocket failed for {pdb_id}: {e}")
        return {"x": 0, "y": 0, "z": 0}
    finally:
        for file in [temp_pdb, f"temp_{pdb_id}_out"]:
            if os.path.exists(file):
                if os.path.isdir(file):
                    import shutil
                    shutil.rmtree(file)
                else:
                    os.remove(file)


@app.route('/api/get_interaction_data', methods=['GET'])
def get_interaction_data():
    drug_id = request.args.get('drug_id')
    receptor_id = request.args.get('receptor_id')

    if not drug_id or not receptor_id:
        return jsonify({"error": "Both drug_id and receptor_id are required."}), 400

    try:
        interaction = DrugReceptorInteraction.query.filter_by(
            drug_id=drug_id,
            receptor_id=receptor_id
        ).first()

        if not interaction:
            return jsonify({"error": "No interaction data found for the given drug and receptor."}), 404

        receptor = Receptor.query.get(receptor_id)
        if not receptor:
            return jsonify({"error": "Receptor not found."}), 404

        drug = Drug.query.get(drug_id)
        if not drug:
            return jsonify({"error": "Drug not found."}), 404

        interaction_data = {
            "ligand": drug.name_en,
            "receptor": receptor.name,
            "affinity": interaction.affinity,
            "affinity_parameter": interaction.affinity_parameter,
            "interaction_type": interaction.interaction_type,
            "mechanism": interaction.mechanism,
            "units": interaction.units,
            "pdb_file": interaction.pdb_file,
            "receptor_details": {
                "type": receptor.type,
                "description": receptor.description,
                "molecular_weight": receptor.molecular_weight,
                "length": receptor.length,
                "gene_name": receptor.gene_name,
                "subcellular_location": receptor.subcellular_location,
                "function": receptor.function,
                "iuphar_id": receptor.iuphar_id,
                "pdb_ids": receptor.pdb_ids
            }
        }
        return jsonify(interaction_data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


#Code for drug - disease interactions:

@app.route('/drug_disease/add', methods=['GET', 'POST'])
def add_drug_disease_interaction():
    drugs = Drug.query.all()
    indications = Indication.query.all()

    if request.method == 'POST':
        drug_id = request.form.get('drug_id', type=int)
        indication_id = request.form.get('indication_id', type=int)
        interaction_type = request.form.get('interaction_type')
        description = request.form.get('description')
        severity = request.form.get('severity')
        recommendation = request.form.get('recommendation')

        if not (drug_id and indication_id and interaction_type):
            flash("Drug, indication, and interaction type are required!", "error")
            return redirect(url_for('add_drug_disease_interaction'))

        interaction = DrugDiseaseInteraction(
            drug_id=drug_id,
            indication_id=indication_id,
            interaction_type=interaction_type,
            description=description,
            severity=severity,
            recommendation=recommendation
        )
        db.session.add(interaction)
        db.session.commit()
        flash("Drug-disease interaction added!", "success")
        return redirect(url_for('list_drug_disease_interactions'))

    return render_template('add_drug_disease.html', drugs=drugs, indications=indications)

@app.route('/drug_disease', methods=['GET'])
def list_drug_disease_interactions():
    interactions = DrugDiseaseInteraction.query.all()
    return render_template('list_drug_disease.html', interactions=interactions)




# İlaç Kategorileri
@app.route('/categories', methods=['GET', 'POST'])
def manage_categories():
    if request.method == 'POST':
        name = request.form.get('name').strip()
        parent_id = request.form.get('parent_id')

        if not name:
            flash('Category name is required!', 'error')
        elif DrugCategory.query.filter_by(name=name).first():
            flash('This category name already exists!', 'error')
        else:
            try:
                category = DrugCategory(
                    name=name,
                    parent_id=int(parent_id) if parent_id else None
                )
                db.session.add(category)
                db.session.commit()
                flash('Category added successfully!', 'success')
            except Exception as e:
                db.session.rollback()
                flash(f'Error adding category: {e}', 'error')
        return redirect(url_for('manage_categories'))

    # Fetch all categories
    categories = DrugCategory.query.order_by(DrugCategory.name).all()

    # For the parent dropdown
    def get_category_name(category, depth=0):
        prefix = "  " * depth
        return f"{prefix}{category.name}"

    def build_flat_list(cats, depth=0):
        result = []
        for cat in cats:
            if not cat.parent_id or depth == 0:
                result.append((cat.id, get_category_name(cat, depth)))
                result.extend(build_flat_list(cat.children, depth + 1))
        return result

    flat_categories = build_flat_list(categories)

    # For the tree
    top_categories = [
        {
            'id': cat.id,
            'name': cat.name,
            'drug_count': db.session.query(Drug).filter_by(category_id=cat.id).count(),  # Added comma here
            'children': cat.children
        }
        for cat in categories if not cat.parent_id
    ]

    return render_template('manage_categories.html', top_categories=top_categories, flat_categories=flat_categories)

@app.route('/categories/edit/<int:cat_id>', methods=['GET', 'POST'])
def edit_category(cat_id):
    category = DrugCategory.query.get_or_404(cat_id)
    
    if request.method == 'POST':
        new_name = request.form.get('name').strip()
        new_parent_id = request.form.get('parent_id')

        if not new_name:
            flash('Category name is required!', 'error')
        elif new_name != category.name and DrugCategory.query.filter_by(name=new_name).first():
            flash('This category name already exists!', 'error')
        elif new_parent_id and int(new_parent_id) == cat_id:
            flash('A category cannot be its own parent!', 'error')
        else:
            try:
                category.name = new_name
                category.parent_id = int(new_parent_id) if new_parent_id else None
                db.session.commit()
                flash('Category updated successfully!', 'success')
                return redirect(url_for('manage_categories'))
            except Exception as e:
                db.session.rollback()
                flash(f'Error updating category: {e}', 'error')

    categories = DrugCategory.query.order_by(DrugCategory.name).all()
    flat_categories = [(cat.id, f"  " * (1 if cat.parent_id else 0) + cat.name) for cat in categories if cat.id != cat_id]
    return render_template('edit_category.html', category=category, flat_categories=flat_categories)

@app.route('/categories/delete/<int:cat_id>', methods=['POST'])
def delete_category(cat_id):
    category = DrugCategory.query.get_or_404(cat_id)
    drug_count = db.session.query(Drug).filter_by(category_id=cat_id).count()
    
    if drug_count > 0:
        flash(f'Cannot delete "{category.name}" - it’s used by {drug_count} drugs!', 'error')
    elif category.children:
        flash(f'Cannot delete "{category.name}" - it has subcategories!', 'error')
    else:
        try:
            db.session.delete(category)
            db.session.commit()
            flash('Category deleted successfully!', 'success')
        except Exception as e:
            db.session.rollback()
            flash(f'Error deleting category: {e}', 'error')
    
    return redirect(url_for('manage_categories'))
#İlaç Kategorileri SON...

#PK modülü...
@app.route('/metabolism', methods=['GET', 'POST'])
def manage_metabolism():
    organs = MetabolismOrgan.query.all()
    enzymes = MetabolismEnzyme.query.all()
    metabolites = Metabolite.query.all()
    drugs = Drug.query.all()  # Add drugs to the template

    if request.method == 'POST':
        if 'organ_name' in request.form:
            name = request.form.get('organ_name').strip()
            if name and not MetabolismOrgan.query.filter_by(name=name).first():
                new_organ = MetabolismOrgan(name=name)
                db.session.add(new_organ)
                db.session.commit()
                return redirect(url_for('manage_metabolism'))
        elif 'enzyme_name' in request.form:
            name = request.form.get('enzyme_name').strip()
            if name and not MetabolismEnzyme.query.filter_by(name=name).first():
                new_enzyme = MetabolismEnzyme(name=name)
                db.session.add(new_enzyme)
                db.session.commit()
                return redirect(url_for('manage_metabolism'))
        elif 'metabolite_name' in request.form:
            name = request.form.get('metabolite_name').strip()
            parent_id = request.form.get('parent_id', type=int)
            drug_id = request.form.get('drug_id', type=int)  # New: Get drug ID
            if not drug_id or not name:
                flash("Drug ID and metabolite name are required.", "error")
                return redirect(url_for('manage_metabolism'))

            # Check if the parent drug exists as a top-level metabolite
            drug = Drug.query.get(drug_id)
            if not drug:
                flash(f"Drug with ID {drug_id} not found.", "error")
                return redirect(url_for('manage_metabolism'))

            # Check if the parent drug (e.g., "Acetaminophen") is already a top-level metabolite
            parent_metabolite = Metabolite.query.filter_by(name=drug.name_en, parent_id=None, drug_id=drug_id).first()
            if not parent_metabolite:
                # Add the parent drug as a top-level metabolite
                parent_metabolite = Metabolite(name=drug.name_en, parent_id=None, drug_id=drug_id)
                db.session.add(parent_metabolite)
                db.session.flush()  # Ensure parent_metabolite gets an ID before using it

            # Update parent_id if it's the parent drug (None or empty means top-level)
            if not parent_id or parent_id is None:
                parent_id = None  # Ensure parent_id is explicitly None for top-level
            else:
                # Verify the parent metabolite exists
                parent_metabolite_check = Metabolite.query.get(parent_id)
                if not parent_metabolite_check or parent_metabolite_check.drug_id != drug_id:
                    flash(f"Invalid parent metabolite for drug ID {drug_id}.", "error")
                    return redirect(url_for('manage_metabolism'))

            # Add or update the new metabolite (e.g., "NAPQI" or "Acetaminophen cysteine")
            existing_metabolite = Metabolite.query.filter_by(name=name, parent_id=parent_id, drug_id=drug_id).first()
            if existing_metabolite:
                flash(f"Metabolite '{name}' already exists for this drug and parent.", "error")
            else:
                new_metabolite = Metabolite(name=name, parent_id=parent_id or None, drug_id=drug_id)
                db.session.add(new_metabolite)
                db.session.commit()
                flash(f"Metabolite '{name}' added successfully for {drug.name_en}.", "success")

            return redirect(url_for('manage_metabolism'))

    return render_template('metabolism.html', organs=organs, enzymes=enzymes, metabolites=metabolites, drugs=drugs)

@app.route('/api/metabolism/organs', methods=['GET'])
def get_metabolism_organs():
    search = request.args.get('q', '').strip()
    limit = request.args.get('limit', 10, type=int)
    page = request.args.get('page', 1, type=int)

    query = MetabolismOrgan.query
    if search:
        query = query.filter(MetabolismOrgan.name.ilike(f'%{search}%'))
    paginated_query = query.paginate(page=page, per_page=limit)

    results = [{'id': organ.id, 'text': organ.name} for organ in paginated_query.items]
    return jsonify({'results': results, 'pagination': {'more': paginated_query.has_next}})

@app.route('/api/metabolism/enzymes', methods=['GET'])
def get_metabolism_enzymes():
    search = request.args.get('q', '').strip()
    limit = request.args.get('limit', 10, type=int)
    page = request.args.get('page', 1, type=int)

    query = MetabolismEnzyme.query
    if search:
        query = query.filter(MetabolismEnzyme.name.ilike(f'%{search}%'))
    paginated_query = query.paginate(page=page, per_page=limit)

    results = [{'id': enzyme.id, 'text': enzyme.name} for enzyme in paginated_query.items]
    return jsonify({'results': results, 'pagination': {'more': paginated_query.has_next}})

@app.route('/api/metabolites', methods=['GET'])
def get_metabolites():
    search = request.args.get('q', '').strip()
    drug_id = request.args.get('drug_id', '').strip()

    query = Metabolite.query
    if search:
        query = query.filter(Metabolite.name.ilike(f'%{search}%'))
    if drug_id:
        try:
            # Only filter by drug_id directly on Metabolite (ignore drug_route_metabolite until DrugRoute exists)
            query = query.filter(Metabolite.drug_id == drug_id)
            print(f"Querying metabolites for drug_id={drug_id}, search={search}")
        except Exception as e:
            print(f"Error in metabolite query for drug_id={drug_id}: {str(e)}")
            return jsonify({'results': []}), 500

    results = query.all()
    return jsonify({'results': [{'id': m.id, 'text': m.name} for m in results]})

@app.route('/api/metabolites/full', methods=['GET'])
def get_metabolites_full():
    ids = request.args.get('ids', '').split(',')
    drug_id = request.args.get('drug_id', '').strip()
    print(f"Fetching metabolites: ids={ids}, drug_id={drug_id}")  # Debug log
    
    if ids == ['']:
        return jsonify([])
    
    try:
        # Ensure ids are valid integers
        ids = [int(id) for id in ids if id]  # Filter out empty strings and convert to integers
        if not ids:
            return jsonify([]), 400  # No valid IDs provided
        
        query = Metabolite.query.filter(Metabolite.id.in_(ids))
        if drug_id:
            try:
                drug_id = int(drug_id)  # Ensure it's an integer
                if not Drug.query.get(drug_id):
                    return jsonify([]), 404  # Drug not found
                query = query.join(drug_route_metabolite, Metabolite.id == drug_route_metabolite.c.metabolite_id, isouter=True)\
                             .join(DrugRoute, DrugRoute.id == drug_route_metabolite.c.drug_route_id, isouter=True)\
                             .join(DrugDetail, DrugDetail.id == DrugRoute.drug_detail_id)\
                             .filter(DrugDetail.drug_id == drug_id)
            except (ValueError, AttributeError) as e:
                print(f"Error with drug_id {drug_id}: {str(e)}")
                return jsonify([]), 400  # Bad request if drug_id is invalid

        metabolites = query.all()
        return jsonify([{'id': m.id, 'name': m.name, 'parent_id': m.parent_id} for m in metabolites])
    except Exception as e:
        print(f"Error fetching metabolites: {str(e)}")
        return jsonify([]), 500  # Internal server error with empty response

@app.route('/api/metabolites/add', methods=['POST'])
def add_metabolite():
    data = request.get_json()
    name = data['name'].strip()
    parent_id = data.get('parent_id')
    drug_id = data.get('drug_id')
    print(f"Adding metabolite: name={name}, parent_id={parent_id}, drug_id={drug_id}")  # Debug log
    if not name or Metabolite.query.filter_by(name=name, parent_id=parent_id).first():
        return jsonify({'error': 'Invalid or duplicate name'}), 400
    new_metabolite = Metabolite(name=name, parent_id=parent_id)
    db.session.add(new_metabolite)
    db.session.commit()
    return jsonify({'id': new_metabolite.id, 'name': new_metabolite.name, 'parent_id': new_metabolite.parent_id}), 201

@app.route('/api/drug_routes', methods=['GET'])
def get_drug_routes():
    drug_id = request.args.get('drug_id', type=int)
    if not drug_id:
        return jsonify([])
    routes = DrugRoute.query.join(DrugDetail).filter(DrugDetail.drug_id == drug_id).all()
    result = [{
        'route_id': route.route_id,
        'metabolites': route.metabolites or '{}'
    } for route in routes]
    return jsonify(result)


@app.route('/pharmacokinetics', methods=['GET'])
def pharmacokinetics():
    selected_drug_id = request.args.get('drug_id', type=int)
    pk_data = None
    selected_drug = None
    
    if selected_drug_id:
        selected_drug = Drug.query.get(selected_drug_id)
        if not selected_drug:
            return render_template('pharmacokinetics.html', error="Drug not found", pk_data=None, selected_drug_id=selected_drug_id), 404
        
        details = DrugDetail.query.filter_by(drug_id=selected_drug_id).all()
        pk_data = []
        for detail in details:
            for route in detail.routes:
                pk_entry = {
                    'route_name': route.route.name,
                    'absorption_rate_min': route.absorption_rate_min or 0,
                    'absorption_rate_max': route.absorption_rate_max or 0,
                    'vod_rate_min': route.vod_rate_min or 0,
                    'vod_rate_max': route.vod_rate_max or 0,
                    'protein_binding_min': (route.protein_binding_min or 0) * 100,
                    'protein_binding_max': (route.protein_binding_max or 0) * 100,
                    'half_life_min': route.half_life_min or 0,
                    'half_life_max': route.half_life_max or 0,
                    'clearance_rate_min': route.clearance_rate_min or 0,
                    'clearance_rate_max': route.clearance_rate_max or 0,
                    'bioavailability_min': (route.bioavailability_min or 0) * 100,  # Fixed: 'route' instead of 'entry'
                    'bioavailability_max': (route.bioavailability_max or 0) * 100,  # Fixed: 'route' instead of 'entry'
                    'tmax_min': route.tmax_min or 0,
                    'tmax_max': route.tmax_max or 0,
                    'cmax_min': route.cmax_min or 0,
                    'cmax_max': route.cmax_max or 0,
                    'pharmacodynamics': route.pharmacodynamics or "N/A",
                    'pharmacokinetics': route.pharmacokinetics or "N/A",
                    'metabolites': [
                        {'id': met.id, 'name': met.name, 'parent_id': met.parent_id}
                        for met in route.metabolites
                    ] if route.metabolites else []
                }
                pk_data.append(pk_entry)
    
    return render_template('pharmacokinetics.html', pk_data=pk_data, selected_drug_id=selected_drug_id, selected_drug=selected_drug)
#PK modülü son...



if __name__ == "__main__":
    with app.app_context():  # Enter the app context
        db.create_all()  # Create tables in the database
        print("Database tables created successfully.")
    app.run(debug=True)
