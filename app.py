from flask import Flask, render_template, render_template_string, request, session, redirect, url_for, jsonify, make_response, flash, send_from_directory, Blueprint, current_app, send_file, Response
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from user_agents import parse
from dotenv import load_dotenv
from flask.cli import FlaskGroup
from openpyxl import load_workbook
import bleach
from bleach.sanitizer import Cleaner
from contextlib import contextmanager
from flask_migrate import Migrate
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for Matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import anthropic
import seaborn as sns
import threading
import psutil
import requests
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
import smtplib
from email.mime.text import MIMEText
import shutil
import uuid  # Import the uuid library for generating unique file names
import subprocess
import torch
import os
os.environ["TRANSFORMERS_BACKEND"] = "torch"

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from flask_caching import Cache
cache = Cache()
import urllib.parse  # For proper URL encoding
import time
import logging
import nltk
import hashlib
import traceback
from io import StringIO
from docx import Document
import math
import json
import xml.etree.ElementTree as ET
import tempfile
import csv
import ssl
import random
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
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, Date, ForeignKey, or_, nullslast, outerjoin, func, and_, select, DateTime, Boolean, Index
from sqlalchemy.orm import relationship, sessionmaker, joinedload, aliased
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.types import JSON, DateTime
from io import BytesIO
from Bio import Entrez
from flask_login import UserMixin
from flask_login import current_user, login_required
from pydantic import BaseModel, field_validator, ValidationError, model_validator
from typing import List, Optional
from http import HTTPStatus
from Bio.PDB import PDBParser, PDBIO
from werkzeug.utils import secure_filename
from datetime import datetime, date, timedelta
from scipy.integrate import odeint
from functools import wraps, lru_cache
from markupsafe import Markup
import sqlalchemy
from sqlalchemy import and_, or_, nullslast, extract, inspect
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from prody import parsePDB, calcCenter
import platform
from pathlib import Path


# Get the directory where app.py is located
BASE_DIR = Path(__file__).resolve().parent

# Load .env if it exists (local development), otherwise ignore (production)
dotenv_path = BASE_DIR / '.env'
load_dotenv(dotenv_path=dotenv_path)   # This line is completely safe even if file doesn't exist

# OPTIONAL: Only show the success message during local development
import os
if os.getenv('RENDER'):  # Render sets this env var automatically
    print("✓ Running on Render – using dashboard environment variables")
elif dotenv_path.exists():
    print(f"✓ Loaded .env from: {dotenv_path}")
else:
    print("⚠️  No .env file found – relying on system environment variables")


# Flask uygulamasını oluştur
app = Flask(__name__)
# Load and validate environment variables
DATABASE_URL = os.environ.get('DATABASE_URL')
SECRET_KEY = os.environ.get('SECRET_KEY')

if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable not set! Check your .env file.")

if not SECRET_KEY:
    raise ValueError("SECRET_KEY environment variable not set! Check your .env file.")


app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = SECRET_KEY

# SQLAlchemy nesnesini oluştur ve uygulamaya bağla
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
migrate = Migrate(app, db)  # Initialize Flask-Migrate

# Logging setup
logging.basicConfig(
    level=logging.DEBUG,  # Use DEBUG for development, INFO for production
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Set up upload folder
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploaded_files")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
logger.info(f"Upload folder set to: {UPLOAD_FOLDER}")

BATCH_SIZE = 1000  # Bulk insert batch size
COMMIT_FREQUENCY = 100  # Commit every N batches
MAX_RETRIES = 3



def parse_user_agent(user_agent_string):
    """Parse user agent to extract device, browser, and OS info"""
    try:
        from user_agents import parse
        user_agent = parse(user_agent_string)
        
        return {
            'device_type': 'mobile' if user_agent.is_mobile else 'tablet' if user_agent.is_tablet else 'desktop',
            'browser': f"{user_agent.browser.family} {user_agent.browser.version_string}",
            'os': f"{user_agent.os.family} {user_agent.os.version_string}",
            'device_brand': user_agent.device.brand or 'Unknown',
            'device_model': user_agent.device.model or 'Unknown',
            'is_bot': user_agent.is_bot,
            'is_touch_capable': user_agent.is_touch_capable,
            'browser_family': user_agent.browser.family,
            'browser_version': user_agent.browser.version_string,
            'os_family': user_agent.os.family,
            'os_version': user_agent.os.version_string
        }
    except Exception as e:
        logger.error(f"Error parsing user agent: {str(e)}")
        return {
            'device_type': 'unknown',
            'browser': 'unknown',
            'os': 'unknown',
            'device_brand': 'unknown',
            'device_model': 'unknown',
            'is_bot': False,
            'is_touch_capable': False,
            'browser_family': 'unknown',
            'browser_version': 'unknown',
            'os_family': 'unknown',
            'os_version': 'unknown'
        }

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

drug_salt = db.Table(
    'drug_salt',
    db.Column('drug_id', db.Integer, db.ForeignKey('public.drug.id'), primary_key=True),
    db.Column('salt_id', db.Integer, db.ForeignKey('public.salt.id'), primary_key=True),
    schema='public',
    extend_existing=True
)

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

interaction_route = db.Table(
    'interaction_route',
    db.Column('interaction_id', db.Integer, db.ForeignKey('public.drug_interaction.id'), primary_key=True),
    db.Column('route_id', db.Integer, db.ForeignKey('public.route_of_administration.id'), primary_key=True),
    schema='public',
    extend_existing=True
)

drug_category_association = db.Table(
    'drug_category_association',
    db.Column('drug_id', db.Integer, db.ForeignKey('public.drug.id'), primary_key=True),
    db.Column('category_id', db.Integer, db.ForeignKey('public.drug_category.id'), primary_key=True),
    schema='public'
)

# Veritabanı Modeli - DATABASE SCHEMA
class DrugCategory(db.Model):
    __tablename__ = 'drug_category'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique=True)
    parent_id = db.Column(db.Integer, db.ForeignKey('public.drug_category.id'), nullable=True)
    parent = db.relationship('DrugCategory', remote_side=[id], backref='children')

class Salt(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    name_tr = db.Column(db.String(100), nullable=False)
    name_en = db.Column(db.String(100), nullable=False)    

# ATC Models
class ATCLevel1(db.Model):
    __tablename__ = 'atc_level1'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.String(1), nullable=False, unique=True, index=True)
    name = db.Column(db.String(255), nullable=False)
    
    level2_children = db.relationship('ATCLevel2', back_populates='level1_parent', cascade='all, delete-orphan')

class ATCLevel2(db.Model):
    __tablename__ = 'atc_level2'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.String(3), nullable=False, unique=True, index=True)
    name = db.Column(db.String(255), nullable=False)
    level1_id = db.Column(db.Integer, db.ForeignKey('public.atc_level1.id'), nullable=False)
    
    level1_parent = db.relationship('ATCLevel1', back_populates='level2_children')
    level3_children = db.relationship('ATCLevel3', back_populates='level2_parent', cascade='all, delete-orphan')

class ATCLevel3(db.Model):
    __tablename__ = 'atc_level3'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.String(4), nullable=False, unique=True, index=True)
    name = db.Column(db.String(255), nullable=False)
    level2_id = db.Column(db.Integer, db.ForeignKey('public.atc_level2.id'), nullable=False)
    
    level2_parent = db.relationship('ATCLevel2', back_populates='level3_children')
    level4_children = db.relationship('ATCLevel4', back_populates='level3_parent', cascade='all, delete-orphan')

class ATCLevel4(db.Model):
    __tablename__ = 'atc_level4'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.String(5), nullable=False, unique=True, index=True)
    name = db.Column(db.String(255), nullable=False)
    level3_id = db.Column(db.Integer, db.ForeignKey('public.atc_level3.id'), nullable=False)
    
    level3_parent = db.relationship('ATCLevel3', back_populates='level4_children')
    level5_children = db.relationship('ATCLevel5', back_populates='level4_parent', cascade='all, delete-orphan')

class ATCLevel5(db.Model):
    __tablename__ = 'atc_level5'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.String(7), nullable=False, index=True)
    name = db.Column(db.String(255), nullable=False)
    level4_id = db.Column(db.Integer, db.ForeignKey('public.atc_level4.id'), nullable=False)
    ddd = db.Column(db.String(50), nullable=True)
    uom = db.Column(db.String(50), nullable=True)
    adm_r = db.Column(db.String(50), nullable=True)  # Changed from 10 to 50
    note = db.Column(db.Text, nullable=True)
    
    level4_parent = db.relationship('ATCLevel4', back_populates='level5_children')
    drug_mappings = db.relationship('DrugATCMapping', back_populates='atc_level5', cascade='all, delete-orphan')

class DrugATCMapping(db.Model):
    __tablename__ = 'drug_atc_mapping'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    id = db.Column(db.Integer, primary_key=True)
    drug_id = db.Column(db.Integer, db.ForeignKey('public.drug.id'), nullable=False)
    atc_level5_id = db.Column(db.Integer, db.ForeignKey('public.atc_level5.id'), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    
    drug = db.relationship('Drug', backref=db.backref('atc_mappings', lazy='dynamic'))
    atc_level5 = db.relationship('ATCLevel5', back_populates='drug_mappings')

# Updated Drug model
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
    # Removed category_id and category relationship
    categories = db.relationship(
        'DrugCategory',
        secondary=drug_category_association,
        backref=db.backref('drugs', lazy='dynamic')
    )
    drug_details = db.relationship(
        'DrugDetail',
        back_populates='parent_drug',
        cascade="all, delete-orphan"
    )
    pharmgkb_id = db.Column(db.String(50), unique=True, nullable=True)
    clinical_annotations = db.relationship('ClinicalAnnotationDrug', back_populates='drug')
    drug_labels = db.relationship('DrugLabelDrug', back_populates='drug')
    clinical_variants = db.relationship('ClinicalVariantDrug', back_populates='drug')
    variant_annotations = db.relationship('VariantAnnotationDrug', back_populates='drug')

class SafetyCategory(db.Model):
    __tablename__ = 'safety_category'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(20), nullable=False, unique=True)
    
    def __repr__(self):
        return f'<SafetyCategory {self.name}>'

class DrugDetail(db.Model):
    __tablename__ = 'drug_detail'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    id = db.Column(db.Integer, primary_key=True)
    drug_id = db.Column(db.Integer, db.ForeignKey('public.drug.id'), nullable=False)
    parent_drug = db.relationship(
        'Drug',
        back_populates='drug_details',
        overlaps="details,drug"
    )
    routes = db.relationship("DrugRoute", back_populates="drug_detail", cascade="all, delete-orphan")
    salt_id = db.Column(db.Integer, db.ForeignKey('public.salt.id'), nullable=True)
    molecular_formula = db.Column(db.String(100), nullable=True)
    synthesis = db.Column(db.Text, nullable=True)
    structure = db.Column(db.Text, nullable=True)
    structure_3d = db.Column(db.Text, nullable=True)
    mechanism_of_action = db.Column(db.Text, nullable=True)
    iupac_name = db.Column(db.Text, nullable=True)
    smiles = db.Column(db.Text, nullable=True)
    inchikey = db.Column(db.Text, nullable=True)
    pubchem_cid = db.Column(db.String(50), nullable=True)
    pubchem_sid = db.Column(db.String(50), nullable=True)
    cas_id = db.Column(db.String(50), nullable=True)
    ec_number = db.Column(db.String(50), nullable=True)
    nci_code = db.Column(db.String(50), nullable=True)
    rxcui = db.Column(db.String(50), nullable=True)
    snomed_id = db.Column(db.String(50), nullable=True)
    molecular_weight = db.Column(db.Float, nullable=True)
    molecular_weight_unit_id = db.Column(db.Integer, db.ForeignKey('public.unit.id'), nullable=True)
    pharmacodynamics = db.Column(db.Text, nullable=True)
    indications = db.Column(db.Text, nullable=True)
    target_molecules = db.Column(db.Text, nullable=True)
    pharmacokinetics = db.Column(db.Text, nullable=True)
    boiling_point = db.Column(db.Float, nullable=True)
    boiling_point_unit_id = db.Column(db.Integer, db.ForeignKey('public.unit.id'), nullable=True)
    melting_point = db.Column(db.Float, nullable=True)
    melting_point_unit_id = db.Column(db.Integer, db.ForeignKey('public.unit.id'), nullable=True)
    density = db.Column(db.Float, nullable=True)
    density_unit_id = db.Column(db.Integer, db.ForeignKey('public.unit.id'), nullable=True)
    solubility = db.Column(db.Float, nullable=True)
    solubility_unit_id = db.Column(db.Integer, db.ForeignKey('public.unit.id'), nullable=True)
    flash_point = db.Column(db.Float, nullable=True)
    flash_point_unit_id = db.Column(db.Integer, db.ForeignKey('public.unit.id'), nullable=True)
    fda_approved = db.Column(db.Boolean, nullable=False, default=False)
    ema_approved = db.Column(db.Boolean, nullable=False, default=False)
    titck_approved = db.Column(db.Boolean, nullable=False, default=False)
    black_box_warning = db.Column(db.Boolean, nullable=False, default=False)
    black_box_details = db.Column(db.Text, nullable=True)
    half_life = db.Column(db.Float, nullable=True)
    half_life_unit_id = db.Column(db.Integer, db.ForeignKey('public.unit.id'), nullable=True)
    clearance_rate = db.Column(db.Float, nullable=True)
    clearance_rate_unit_id = db.Column(db.Integer, db.ForeignKey('public.unit.id'), nullable=True)
    bioavailability = db.Column(db.Float, nullable=True)
    pregnancy_safety_trimester1_id = db.Column(db.Integer, db.ForeignKey('public.safety_category.id'), nullable=True)
    pregnancy_details_trimester1 = db.Column(db.Text)
    pregnancy_safety_trimester2_id = db.Column(db.Integer, db.ForeignKey('public.safety_category.id'), nullable=True)
    pregnancy_details_trimester2 = db.Column(db.Text)
    pregnancy_safety_trimester3_id = db.Column(db.Integer, db.ForeignKey('public.safety_category.id'), nullable=True)
    pregnancy_details_trimester3 = db.Column(db.Text)
    lactation_safety_id = db.Column(db.Integer, db.ForeignKey('public.safety_category.id'), nullable=True)
    lactation_details = db.Column(db.Text, nullable=True)
    references = db.Column(db.Text, nullable=True)
    drug = db.relationship('Drug', backref=db.backref('details', lazy=True))
    salt = db.relationship('Salt', backref=db.backref('details', lazy=True))
    molecular_weight_unit = db.relationship('Unit', foreign_keys=[molecular_weight_unit_id])
    boiling_point_unit = db.relationship('Unit', foreign_keys=[boiling_point_unit_id])
    melting_point_unit = db.relationship('Unit', foreign_keys=[melting_point_unit_id])
    density_unit = db.relationship('Unit', foreign_keys=[density_unit_id])
    solubility_unit = db.relationship('Unit', foreign_keys=[solubility_unit_id])
    flash_point_unit = db.relationship('Unit', foreign_keys=[flash_point_unit_id])
    half_life_unit = db.relationship('Unit', foreign_keys=[half_life_unit_id])
    clearance_rate_unit = db.relationship('Unit', foreign_keys=[clearance_rate_unit_id])
    pregnancy_safety_trimester1 = db.relationship('SafetyCategory', foreign_keys=[pregnancy_safety_trimester1_id])
    pregnancy_safety_trimester2 = db.relationship('SafetyCategory', foreign_keys=[pregnancy_safety_trimester2_id])
    pregnancy_safety_trimester3 = db.relationship('SafetyCategory', foreign_keys=[pregnancy_safety_trimester3_id])
    lactation_safety = db.relationship('SafetyCategory', foreign_keys=[lactation_safety_id])

# Database Model for Indications
class Indication(db.Model):
    __tablename__ = 'indication'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    id = db.Column(db.Integer, primary_key=True)
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
    __tablename__ = 'target'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    id = db.Column(db.Integer, primary_key=True)
    name_tr = db.Column(db.String(255), nullable=False)
    name_en = db.Column(db.String(255), nullable=False)

# Food Model
class Food(db.Model):
    __tablename__ = 'food'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    id = db.Column(db.Integer, primary_key=True)
    name_en = db.Column(db.String(255), nullable=False)
    name_tr = db.Column(db.String(255), nullable=True)
    category = db.Column(db.String(100), nullable=True)
    description = db.Column(db.Text, nullable=True)
    
    def __repr__(self):
        return f'<Food {self.name_en}>'

# DrugFoodInteraction Model
class DrugFoodInteraction(db.Model):
    __tablename__ = 'drug_food_interaction'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    id = db.Column(db.Integer, primary_key=True)
    drug_id = db.Column(db.Integer, db.ForeignKey('public.drug.id'), nullable=False)
    food_id = db.Column(db.Integer, db.ForeignKey('public.food.id'), nullable=False)
    interaction_type = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=False)
    severity_id = db.Column(db.Integer, db.ForeignKey('public.severity.id'), nullable=False)
    timing_instruction = db.Column(db.Text, nullable=True)
    recommendation = db.Column(db.Text, nullable=True)
    reference = db.Column(db.Text, nullable=True)
    predicted_severity = db.Column(db.String(50), nullable=True)
    prediction_confidence = db.Column(db.Float, nullable=True)
    
    drug = db.relationship('Drug', backref='food_interactions')
    food = db.relationship('Food', backref='drug_interactions')
    severity = db.relationship('Severity', backref='drug_food_interactions')

class Severity(db.Model):
    __tablename__ = 'severity'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(20), nullable=True, unique=True)
    description = db.Column(db.Text, nullable=True)

class DrugInteraction(db.Model):
    __tablename__ = 'drug_interaction'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    id = db.Column(db.Integer, primary_key=True)
    drug1_id = db.Column(db.Integer, db.ForeignKey('public.drug.id'), nullable=False)
    drug2_id = db.Column(db.Integer, db.ForeignKey('public.drug.id'), nullable=False)
    interaction_type = db.Column(db.String(50), nullable=False)
    interaction_description = db.Column(db.Text, nullable=False)
    severity_id = db.Column(db.Integer, db.ForeignKey('public.severity.id'), nullable=False)
    mechanism = db.Column(db.Text, nullable=True)
    pharmacokinetics = db.Column(db.Text, nullable=True)
    monitoring = db.Column(db.Text, nullable=True)
    alternatives = db.Column(db.Text, nullable=True)
    reference = db.Column(db.Text, nullable=True)
    predicted_severity_id = db.Column(db.Integer, db.ForeignKey('public.severity.id'), nullable=True)
    prediction_confidence = db.Column(db.Float, nullable=True)
    processed = db.Column(db.Boolean, default=False)
    time_to_peak = db.Column(db.Float, nullable=True)
    
    drug1 = db.relationship("Drug", foreign_keys=[drug1_id])
    drug2 = db.relationship("Drug", foreign_keys=[drug2_id])
    severity_level = db.relationship("Severity", foreign_keys=[severity_id], backref='manual_severity_interactions')
    predicted_severity_level = db.relationship("Severity", foreign_keys=[predicted_severity_id], backref='predicted_severity_interactions')
    routes = db.relationship(
        "RouteOfAdministration",
        secondary='public.interaction_route',
        backref=db.backref("interactions", lazy='dynamic')
    )


class RouteOfAdministration(db.Model):
    __tablename__ = 'route_of_administration'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    type = db.Column(db.String(50), nullable=False)
    description = db.Column(db.Text, nullable=True)
    parent_id = db.Column(db.Integer, db.ForeignKey('public.route_of_administration.id'), nullable=True)
    parent = db.relationship(
        "RouteOfAdministration",
        remote_side=[id],
        backref=db.backref("children", cascade="all, delete-orphan")
    )


class Receptor(db.Model):
    __tablename__ = 'receptor'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    type = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=True)
    molecular_weight = db.Column(db.String(50), nullable=True)
    length = db.Column(db.Integer, nullable=True)
    gene_name = db.Column(db.String(100), nullable=True)
    subcellular_location = db.Column(db.Text, nullable=True)
    function = db.Column(db.Text, nullable=True)
    iuphar_id = db.Column(db.String(50))
    pdb_ids = db.Column(db.Text, nullable=True)
    binding_site_x = db.Column(db.Float, nullable=True)
    binding_site_y = db.Column(db.Float, nullable=True)
    binding_site_z = db.Column(db.Float, nullable=True)

class DrugReceptorInteraction(db.Model):
    __tablename__ = 'drug_receptor_interaction'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    id = db.Column(db.Integer, primary_key=True)
    drug_id = db.Column(db.Integer, db.ForeignKey('public.drug.id'), nullable=False)
    receptor_id = db.Column(db.Integer, db.ForeignKey('public.receptor.id'), nullable=False)
    affinity = db.Column(db.Float, nullable=True)
    interaction_type = db.Column(db.String(50), nullable=True)
    mechanism = db.Column(db.Text, nullable=True)
    pdb_file = db.Column(db.String(200), nullable=True)
    units = db.Column(db.String, nullable=True)
    affinity_parameter = db.Column(db.String(50), nullable=True)
    drug = db.relationship('Drug', backref=db.backref('interactions', lazy=True))
    receptor = db.relationship('Receptor', backref=db.backref('interactions', lazy=True))


class DrugDiseaseInteraction(db.Model):
    __tablename__ = 'drug_disease_interaction'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    id = db.Column(db.Integer, primary_key=True)
    drug_id = db.Column(db.Integer, db.ForeignKey('public.drug.id'), nullable=False)
    indication_id = db.Column(db.Integer, db.ForeignKey('public.indication.id'), nullable=False)
    interaction_type = db.Column(db.String(50), nullable=False)
    description = db.Column(db.Text, nullable=True)
    severity = db.Column(db.String(20), default="Moderate")
    recommendation = db.Column(db.Text, nullable=True)
    drug = db.relationship("Drug", backref="disease_interactions")
    indication = db.relationship("Indication", backref="drug_interactions")   

# New Model for Drug-Lab Test Interactions

class Unit(db.Model):
    __tablename__ = 'unit'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False, unique=True)
    description = db.Column(db.Text, nullable=True)

class LabTest(db.Model):
    __tablename__ = 'lab_test'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    id = db.Column(db.Integer, primary_key=True)
    name_en = db.Column(db.String(255), nullable=False)
    name_tr = db.Column(db.String(255), nullable=True)
    description = db.Column(db.Text, nullable=True)
    reference_range = db.Column(db.String(100), nullable=True)
    unit_id = db.Column(db.Integer, db.ForeignKey('public.unit.id'), nullable=True)
    unit = db.relationship("Unit", backref="lab_tests")

class DrugLabTestInteraction(db.Model):
    __tablename__ = 'drug_lab_test_interaction'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    id = db.Column(db.Integer, primary_key=True)
    drug_id = db.Column(db.Integer, db.ForeignKey('public.drug.id'), nullable=False)
    lab_test_id = db.Column(db.Integer, db.ForeignKey('public.lab_test.id'), nullable=False)
    interaction_type = db.Column(db.String(50), nullable=False)
    description = db.Column(db.Text, nullable=True)
    severity_id = db.Column(db.Integer, db.ForeignKey('public.severity.id'), nullable=False)
    recommendation = db.Column(db.Text, nullable=True)
    reference = db.Column(db.Text, nullable=True)
    drug = db.relationship("Drug", backref="lab_test_interactions")
    lab_test = db.relationship("LabTest", backref="drug_interactions")
    severity = db.relationship("Severity", backref="lab_test_interactions")


class DrugRoute(db.Model):
    __tablename__ = 'drug_route'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    id = db.Column(db.Integer, primary_key=True)
    drug_detail_id = db.Column(db.Integer, db.ForeignKey('public.drug_detail.id', ondelete="CASCADE"), nullable=False)
    route_id = db.Column(db.Integer, db.ForeignKey('public.route_of_administration.id', ondelete="CASCADE"), nullable=False)
    pharmacodynamics = db.Column(db.Text, nullable=True)
    pharmacokinetics = db.Column(db.Text, nullable=True)
    absorption_rate_min = db.Column(db.Float, nullable=True)
    absorption_rate_max = db.Column(db.Float, nullable=True)
    absorption_rate_unit_id = db.Column(db.Integer, db.ForeignKey('public.unit.id'), nullable=True)
    vod_rate_min = db.Column(db.Float, nullable=True)
    vod_rate_max = db.Column(db.Float, nullable=True)
    vod_rate_unit_id = db.Column(db.Integer, db.ForeignKey('public.unit.id'), nullable=True)
    protein_binding_min = db.Column(db.Float, nullable=True)
    protein_binding_max = db.Column(db.Float, nullable=True)
    half_life_min = db.Column(db.Float, nullable=True)
    half_life_max = db.Column(db.Float, nullable=True)
    half_life_unit_id = db.Column(db.Integer, db.ForeignKey('public.unit.id'), nullable=True)
    clearance_rate_min = db.Column(db.Float, nullable=True)
    clearance_rate_max = db.Column(db.Float, nullable=True)
    clearance_rate_unit_id = db.Column(db.Integer, db.ForeignKey('public.unit.id'), nullable=True)
    bioavailability_min = db.Column(db.Float, nullable=True)
    bioavailability_max = db.Column(db.Float, nullable=True)
    tmax_min = db.Column(db.Float, nullable=True)
    tmax_max = db.Column(db.Float, nullable=True)
    tmax_unit_id = db.Column(db.Integer, db.ForeignKey('public.unit.id'), nullable=True)
    cmax_min = db.Column(db.Float, nullable=True)
    cmax_max = db.Column(db.Float, nullable=True)
    cmax_unit_id = db.Column(db.Integer, db.ForeignKey('public.unit.id'), nullable=True)
    therapeutic_min = db.Column(db.Float, nullable=True)
    therapeutic_max = db.Column(db.Float, nullable=True)
    therapeutic_unit_id = db.Column(db.Integer, db.ForeignKey('public.unit.id'), nullable=True)
    route = db.relationship("RouteOfAdministration")
    drug_detail = db.relationship("DrugDetail", back_populates="routes")
    absorption_rate_unit = db.relationship('Unit', foreign_keys=[absorption_rate_unit_id])
    vod_rate_unit = db.relationship('Unit', foreign_keys=[vod_rate_unit_id])
    half_life_unit = db.relationship('Unit', foreign_keys=[half_life_unit_id])
    clearance_rate_unit = db.relationship('Unit', foreign_keys=[clearance_rate_unit_id])
    tmax_unit = db.relationship('Unit', foreign_keys=[tmax_unit_id])
    cmax_unit = db.relationship('Unit', foreign_keys=[cmax_unit_id])
    therapeutic_unit = db.relationship('Unit', foreign_keys=[therapeutic_unit_id])
    metabolism_organs = db.relationship('MetabolismOrgan', secondary='public.drug_route_metabolism_organ', backref='drug_routes')
    metabolism_enzymes = db.relationship('MetabolismEnzyme', secondary='public.drug_route_metabolism_enzyme', backref='drug_routes')
    metabolites = db.relationship('Metabolite', secondary='public.drug_route_metabolite', backref='routes')
    route_indications = db.relationship("RouteIndication", backref="drug_route", cascade="all, delete-orphan")


class RouteIndication(db.Model):
    __tablename__ = 'route_indication'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    id = db.Column(db.Integer, primary_key=True)
    drug_detail_id = db.Column(db.Integer, db.ForeignKey('public.drug_detail.id'), nullable=False)
    route_id = db.Column(db.Integer, db.ForeignKey('public.route_of_administration.id'), nullable=False)
    indication_id = db.Column(db.Integer, db.ForeignKey('public.indication.id'), nullable=False)
    drug_route_id = db.Column(db.Integer, db.ForeignKey('public.drug_route.id', ondelete="CASCADE"), nullable=True)
    drug_detail = db.relationship('DrugDetail', backref='route_indications')
    route = db.relationship('RouteOfAdministration')
    indication = db.relationship('Indication')


class SideEffect(db.Model):
    __tablename__ = 'side_effect'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    id = db.Column(db.Integer, primary_key=True)
    name_en = db.Column(db.String(100), nullable=False)
    name_tr = db.Column(db.String(100), nullable=True)
    details = db.relationship('DrugDetail', secondary=detail_side_effect, backref='side_effects', lazy='dynamic')

class Pathway(db.Model):
    __tablename__ = 'pathway'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    id = db.Column(db.Integer, primary_key=True)
    pathway_id = db.Column(db.String(50), unique=True, nullable=False)
    name = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text, nullable=True)
    organism = db.Column(db.String(50), nullable=False)
    url = db.Column(db.String(255), nullable=True)
    drugs = db.relationship('Drug', secondary=pathway_drug, backref='pathways', lazy='dynamic')

class MetabolismOrgan(db.Model):
    __tablename__ = 'metabolism_organ'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique=True)

class MetabolismEnzyme(db.Model):
    __tablename__ = 'metabolism_enzyme'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique=True)

class Metabolite(db.Model):
    __tablename__ = 'metabolite'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    parent_id = db.Column(db.Integer, db.ForeignKey('public.metabolite.id'), nullable=True)
    drug_id = db.Column(db.Integer, db.ForeignKey('public.drug.id'), nullable=True)
    parent = db.relationship("Metabolite", remote_side=[id], backref="children")
    drug = db.relationship("Drug", backref="metabolites")
    drug_routes = db.relationship('DrugRoute', secondary=drug_route_metabolite, backref='metabolite_routes')

# PharmGKB için database modelleri:
class Publication(db.Model):
    __tablename__ = 'publication'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    pmid = db.Column(db.String(50), primary_key=True)
    title = db.Column(db.Text, nullable=True)
    year = db.Column(db.String(4), nullable=True)
    journal = db.Column(db.Text, nullable=True)
    
    clinical_evidence = db.relationship('ClinicalAnnEvidencePublication', back_populates='publication')
    automated_annotations = db.relationship('AutomatedAnnotation', back_populates='publication')

# Clinical Annotations
class Gene(db.Model):
    __tablename__ = 'gene'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    gene_id = db.Column(db.String(50), primary_key=True)
    gene_symbol = db.Column(db.String(100), unique=True, nullable=False, index=True)
    
    clinical_annotations = db.relationship('ClinicalAnnotationGene', back_populates='gene')
    variant_annotations = db.relationship('VariantAnnotationGene', back_populates='gene')
    drug_labels = db.relationship('DrugLabelGene', back_populates='gene')
    clinical_variants = db.relationship('ClinicalVariant', back_populates='gene')

class Phenotype(db.Model):
    __tablename__ = 'phenotype'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), unique=True, nullable=False, index=True)
    
    clinical_annotations = db.relationship('ClinicalAnnotationPhenotype', back_populates='phenotype')
    clinical_variants = db.relationship('ClinicalVariantPhenotype', back_populates='phenotype')

class Variant(db.Model):
    __tablename__ = 'variant'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    id = db.Column(db.Integer, primary_key=True)
    pharmgkb_id = db.Column(db.String(50), unique=True, nullable=True, index=True)
    name = db.Column(db.String(800), unique=True, nullable=False, index=True)
    
    clinical_annotations = db.relationship('ClinicalAnnotationVariant', back_populates='variant')
    variant_annotations = db.relationship('VariantAnnotationVariant', back_populates='variant')
    drug_labels = db.relationship('DrugLabelVariant', back_populates='variant')
    clinical_variants = db.relationship('ClinicalVariantVariant', back_populates='variant')

# Clinical Annotations
class ClinicalAnnotation(db.Model):
    __tablename__ = 'clinical_annotation'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    clinical_annotation_id = db.Column(db.String, primary_key=True)
    level_of_evidence = db.Column(db.String, index=True)
    phenotype_category = db.Column(db.String)
    url = db.Column(db.String)
    latest_history_date = db.Column(db.Date)
    specialty_population = db.Column(db.String, nullable=True)
    level_override = db.Column(db.String, nullable=True)
    level_modifiers = db.Column(db.String, nullable=True)
    score = db.Column(db.Float)
    pmid_count = db.Column(db.Integer)
    evidence_count = db.Column(db.Integer)
    
    alleles = db.relationship('ClinicalAnnAllele', back_populates='annotation', cascade="all, delete-orphan")
    history = db.relationship('ClinicalAnnHistory', back_populates='annotation', cascade="all, delete-orphan")
    evidence = db.relationship('ClinicalAnnEvidence', back_populates='annotation', cascade="all, delete-orphan")
    drugs = db.relationship('ClinicalAnnotationDrug', back_populates='annotation', cascade="all, delete-orphan")
    genes = db.relationship('ClinicalAnnotationGene', back_populates='annotation', cascade="all, delete-orphan")
    phenotypes = db.relationship('ClinicalAnnotationPhenotype', back_populates='annotation', cascade="all, delete-orphan")
    variants = db.relationship('ClinicalAnnotationVariant', back_populates='annotation', cascade="all, delete-orphan")

class ClinicalAnnAllele(db.Model):
    __tablename__ = 'clinical_ann_allele'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    id = db.Column(db.Integer, primary_key=True)
    clinical_annotation_id = db.Column(db.String, db.ForeignKey('public.clinical_annotation.clinical_annotation_id'), index=True)
    genotype_allele = db.Column(db.String, index=True)
    annotation_text = db.Column(db.Text)
    allele_function = db.Column(db.String, nullable=True)
    
    annotation = db.relationship('ClinicalAnnotation', back_populates='alleles')

class ClinicalAnnHistory(db.Model):
    __tablename__ = 'clinical_ann_history'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    id = db.Column(db.Integer, primary_key=True)
    clinical_annotation_id = db.Column(db.String, db.ForeignKey('public.clinical_annotation.clinical_annotation_id'), index=True)
    date = db.Column(db.Date, index=True)
    type = db.Column(db.String)
    comment = db.Column(db.Text)
    
    annotation = db.relationship('ClinicalAnnotation', back_populates='history')

class ClinicalAnnEvidence(db.Model):
    __tablename__ = 'clinical_ann_evidence'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    evidence_id = db.Column(db.String, primary_key=True)
    clinical_annotation_id = db.Column(db.String, db.ForeignKey('public.clinical_annotation.clinical_annotation_id'), index=True)
    evidence_type = db.Column(db.String, index=True)
    evidence_url = db.Column(db.String)
    summary = db.Column(db.Text)
    score = db.Column(db.Float)
    
    annotation = db.relationship('ClinicalAnnotation', back_populates='evidence')
    publications = db.relationship('ClinicalAnnEvidencePublication', back_populates='evidence', cascade="all, delete-orphan")

# Junction Tables for ClinicalAnnotation
class ClinicalAnnotationDrug(db.Model):
    __tablename__ = 'clinical_annotation_drug'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    clinical_annotation_id = db.Column(db.String, db.ForeignKey('public.clinical_annotation.clinical_annotation_id'), primary_key=True)
    drug_id = db.Column(db.Integer, db.ForeignKey('public.drug.id'), primary_key=True)
    
    annotation = db.relationship('ClinicalAnnotation', back_populates='drugs')
    drug = db.relationship('Drug', back_populates='clinical_annotations')

class ClinicalAnnotationGene(db.Model):
    __tablename__ = 'clinical_annotation_gene'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    clinical_annotation_id = db.Column(db.String, db.ForeignKey('public.clinical_annotation.clinical_annotation_id'), primary_key=True)
    gene_id = db.Column(db.String, db.ForeignKey('public.gene.gene_id'), primary_key=True)
    
    annotation = db.relationship('ClinicalAnnotation', back_populates='genes')
    gene = db.relationship('Gene', back_populates='clinical_annotations')

class ClinicalAnnotationPhenotype(db.Model):
    __tablename__ = 'clinical_annotation_phenotype'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    clinical_annotation_id = db.Column(db.String, db.ForeignKey('public.clinical_annotation.clinical_annotation_id'), primary_key=True)
    phenotype_id = db.Column(db.Integer, db.ForeignKey('public.phenotype.id'), primary_key=True)
    
    annotation = db.relationship('ClinicalAnnotation', back_populates='phenotypes')
    phenotype = db.relationship('Phenotype', back_populates='clinical_annotations')

class ClinicalAnnotationVariant(db.Model):
    __tablename__ = 'clinical_annotation_variant'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    clinical_annotation_id = db.Column(db.String, db.ForeignKey('public.clinical_annotation.clinical_annotation_id'), primary_key=True)
    variant_id = db.Column(db.Integer, db.ForeignKey('public.variant.id'), primary_key=True)
    
    annotation = db.relationship('ClinicalAnnotation', back_populates='variants')
    variant = db.relationship('Variant', back_populates='clinical_annotations')

class ClinicalAnnEvidencePublication(db.Model):
    __tablename__ = 'clinical_ann_evidence_publication'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    evidence_id = db.Column(db.String, db.ForeignKey('public.clinical_ann_evidence.evidence_id'), primary_key=True)
    pmid = db.Column(db.String(50), db.ForeignKey('public.publication.pmid'), primary_key=True)
    
    evidence = db.relationship('ClinicalAnnEvidence', back_populates='publications')
    publication = db.relationship('Publication', back_populates='clinical_evidence')

# Variant Annotations
class VariantAnnotation(db.Model):
    __tablename__ = 'variant_annotation'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    variant_annotation_id = db.Column(db.String, primary_key=True)
    
    study_parameters = db.relationship('StudyParameters', back_populates='variant_annotation', cascade="all, delete-orphan")
    fa_annotations = db.relationship('VariantFAAnn', back_populates='variant_annotation', cascade="all, delete-orphan")
    drug_annotations = db.relationship('VariantDrugAnn', back_populates='variant_annotation', cascade="all, delete-orphan")
    pheno_annotations = db.relationship('VariantPhenoAnn', back_populates='variant_annotation', cascade="all, delete-orphan")
    genes = db.relationship('VariantAnnotationGene', back_populates='variant_annotation', cascade="all, delete-orphan")
    variants = db.relationship('VariantAnnotationVariant', back_populates='variant_annotation', cascade="all, delete-orphan")
    drugs = db.relationship('VariantAnnotationDrug', back_populates='variant_annotation', cascade="all, delete-orphan")

class StudyParameters(db.Model):
    __tablename__ = 'study_parameters'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    study_parameters_id = db.Column(db.String, primary_key=True)
    variant_annotation_id = db.Column(db.String, db.ForeignKey('public.variant_annotation.variant_annotation_id'), index=True)
    study_type = db.Column(db.String, nullable=True)
    study_cases = db.Column(db.Integer, nullable=True)
    study_controls = db.Column(db.Integer, nullable=True)
    characteristics = db.Column(db.Text, nullable=True)
    characteristics_type = db.Column(db.String, nullable=True)
    frequency_in_cases = db.Column(db.Float, nullable=True)
    allele_of_frequency_in_cases = db.Column(db.String, nullable=True)
    frequency_in_controls = db.Column(db.Float, nullable=True)
    allele_of_frequency_in_controls = db.Column(db.String, nullable=True)
    p_value = db.Column(db.String, nullable=True)
    ratio_stat_type = db.Column(db.String, nullable=True)
    ratio_stat = db.Column(db.Float, nullable=True)
    confidence_interval_start = db.Column(db.Float, nullable=True)
    confidence_interval_stop = db.Column(db.Float, nullable=True)
    biogeographical_groups = db.Column(db.String, nullable=True)
    
    variant_annotation = db.relationship('VariantAnnotation', back_populates='study_parameters')

class VariantFAAnn(db.Model):
    __tablename__ = 'variant_fa_ann'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    variant_annotation_id = db.Column(db.String, db.ForeignKey('public.variant_annotation.variant_annotation_id'), primary_key=True)
    pmid = db.Column(db.String(50), nullable=True, index=True)
    phenotype_category = db.Column(db.String(100), nullable=True)
    significance = db.Column(db.String, nullable=True)
    notes = db.Column(db.Text, nullable=True)
    sentence = db.Column(db.Text)
    alleles = db.Column(db.String)
    specialty_population = db.Column(db.String, nullable=True)
    assay_type = db.Column(db.String, nullable=True)
    metabolizer_types = db.Column(db.String, nullable=True)
    is_plural = db.Column(db.String, nullable=True)
    is_associated = db.Column(db.String)
    direction_of_effect = db.Column(db.String, nullable=True)
    functional_terms = db.Column(db.String, nullable=True)
    gene_product = db.Column(db.String, nullable=True)
    when_treated_with = db.Column(db.String, nullable=True)
    multiple_drugs = db.Column(db.String, nullable=True)
    cell_type = db.Column(db.String, nullable=True)
    comparison_alleles = db.Column(db.String, nullable=True)
    comparison_metabolizer_types = db.Column(db.String, nullable=True)
    
    variant_annotation = db.relationship('VariantAnnotation', back_populates='fa_annotations')

class VariantDrugAnn(db.Model):
    __tablename__ = 'variant_drug_ann'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    variant_annotation_id = db.Column(db.String, db.ForeignKey('public.variant_annotation.variant_annotation_id'), primary_key=True)
    pmid = db.Column(db.String(50), nullable=True, index=True)
    phenotype_category = db.Column(db.String(100), nullable=True)
    significance = db.Column(db.String, nullable=True)
    notes = db.Column(db.Text, nullable=True)
    sentence = db.Column(db.Text)
    alleles = db.Column(db.String)
    specialty_population = db.Column(db.String, nullable=True)
    metabolizer_types = db.Column(db.String, nullable=True)
    is_plural = db.Column(db.String, nullable=True)
    is_associated = db.Column(db.String)
    direction_of_effect = db.Column(db.String, nullable=True)
    pd_pk_terms = db.Column(db.String, nullable=True)
    multiple_drugs = db.Column(db.String, nullable=True)
    population_types = db.Column(db.String, nullable=True)
    population_phenotypes_diseases = db.Column(db.String, nullable=True)
    multiple_phenotypes_diseases = db.Column(db.String, nullable=True)
    comparison_alleles = db.Column(db.String, nullable=True)
    comparison_metabolizer_types = db.Column(db.String, nullable=True)
    
    variant_annotation = db.relationship('VariantAnnotation', back_populates='drug_annotations')

class VariantPhenoAnn(db.Model):
    __tablename__ = 'variant_pheno_ann'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    variant_annotation_id = db.Column(db.String, db.ForeignKey('public.variant_annotation.variant_annotation_id'), primary_key=True)
    pmid = db.Column(db.String(50), nullable=True, index=True)
    phenotype_category = db.Column(db.String(100), nullable=True)
    significance = db.Column(db.String, nullable=True)
    notes = db.Column(db.Text, nullable=True)
    sentence = db.Column(db.Text)
    alleles = db.Column(db.String)
    specialty_population = db.Column(db.String, nullable=True)
    metabolizer_types = db.Column(db.String, nullable=True)
    is_plural = db.Column(db.String, nullable=True)
    is_associated = db.Column(db.String)
    direction_of_effect = db.Column(db.String, nullable=True)
    side_effect_efficacy_other = db.Column(db.String, nullable=True)
    phenotype = db.Column(db.String, nullable=True)
    multiple_phenotypes = db.Column(db.String, nullable=True)
    when_treated_with = db.Column(db.String, nullable=True)
    multiple_drugs = db.Column(db.String, nullable=True)
    population_types = db.Column(db.String, nullable=True)
    population_phenotypes_diseases = db.Column(db.String, nullable=True)
    multiple_phenotypes_diseases = db.Column(db.String, nullable=True)
    comparison_alleles = db.Column(db.String, nullable=True)
    comparison_metabolizer_types = db.Column(db.String, nullable=True)
    
    variant_annotation = db.relationship('VariantAnnotation', back_populates='pheno_annotations')

# Junction Tables for VariantAnnotation
class VariantAnnotationDrug(db.Model):
    __tablename__ = 'variant_annotation_drug'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    variant_annotation_id = db.Column(db.String, db.ForeignKey('public.variant_annotation.variant_annotation_id'), primary_key=True)
    drug_id = db.Column(db.Integer, db.ForeignKey('public.drug.id'), primary_key=True)
    
    variant_annotation = db.relationship('VariantAnnotation', back_populates='drugs')
    drug = db.relationship('Drug', back_populates='variant_annotations')

class VariantAnnotationGene(db.Model):
    __tablename__ = 'variant_annotation_gene'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    variant_annotation_id = db.Column(db.String, db.ForeignKey('public.variant_annotation.variant_annotation_id'), primary_key=True)
    gene_id = db.Column(db.String, db.ForeignKey('public.gene.gene_id'), primary_key=True)
    
    variant_annotation = db.relationship('VariantAnnotation', back_populates='genes')
    gene = db.relationship('Gene', back_populates='variant_annotations')

class VariantAnnotationVariant(db.Model):
    __tablename__ = 'variant_annotation_variant'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    variant_annotation_id = db.Column(db.String, db.ForeignKey('public.variant_annotation.variant_annotation_id'), primary_key=True)
    variant_id = db.Column(db.Integer, db.ForeignKey('public.variant.id'), primary_key=True)
    
    variant_annotation = db.relationship('VariantAnnotation', back_populates='variants')
    variant = db.relationship('Variant', back_populates='variant_annotations')

# Relationships
class Relationship(db.Model):
    __tablename__ = 'relationships'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    id = db.Column(db.Integer, primary_key=True)
    entity1_id = db.Column(db.String, index=True)
    entity1_name = db.Column(db.String)
    entity1_type = db.Column(db.String)
    entity2_id = db.Column(db.String, index=True)
    entity2_name = db.Column(db.String)
    entity2_type = db.Column(db.String)
    evidence = db.Column(db.String)
    association = db.Column(db.String)
    pk = db.Column(db.String, nullable=True)
    pd = db.Column(db.String, nullable=True)
    pmids = db.Column(db.String, nullable=True)

# Drug Labels
class DrugLabel(db.Model):
    __tablename__ = 'drug_label'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    pharmgkb_id = db.Column(db.String, primary_key=True)
    name = db.Column(db.String)
    source = db.Column(db.String)
    biomarker_flag = db.Column(db.String, nullable=True)
    testing_level = db.Column(db.String, nullable=True)
    has_prescribing_info = db.Column(db.String, nullable=True)
    has_dosing_info = db.Column(db.String, nullable=True)
    has_alternate_drug = db.Column(db.String, nullable=True)
    has_other_prescribing_guidance = db.Column(db.String, nullable=True)
    cancer_genome = db.Column(db.String, nullable=True)
    prescribing = db.Column(db.String, nullable=True)
    latest_history_date = db.Column(db.Date)
    
    drugs = db.relationship('DrugLabelDrug', back_populates='drug_label', cascade="all, delete-orphan")
    genes = db.relationship('DrugLabelGene', back_populates='drug_label', cascade="all, delete-orphan")
    variants = db.relationship('DrugLabelVariant', back_populates='drug_label', cascade="all, delete-orphan")

class DrugLabelDrug(db.Model):
    __tablename__ = 'drug_label_drug'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    pharmgkb_id = db.Column(db.String, db.ForeignKey('public.drug_label.pharmgkb_id'), primary_key=True)
    drug_id = db.Column(db.Integer, db.ForeignKey('public.drug.id'), primary_key=True)
    
    drug_label = db.relationship('DrugLabel', back_populates='drugs')
    drug = db.relationship('Drug', back_populates='drug_labels')

class DrugLabelGene(db.Model):
    __tablename__ = 'drug_label_gene'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    pharmgkb_id = db.Column(db.String, db.ForeignKey('public.drug_label.pharmgkb_id'), primary_key=True)
    gene_id = db.Column(db.String, db.ForeignKey('public.gene.gene_id'), primary_key=True)
    
    drug_label = db.relationship('DrugLabel', back_populates='genes')
    gene = db.relationship('Gene', back_populates='drug_labels')

class DrugLabelVariant(db.Model):
    __tablename__ = 'drug_label_variant'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    pharmgkb_id = db.Column(db.String, db.ForeignKey('public.drug_label.pharmgkb_id'), primary_key=True)
    variant_id = db.Column(db.Integer, db.ForeignKey('public.variant.id'), primary_key=True)
    
    drug_label = db.relationship('DrugLabel', back_populates='variants')
    variant = db.relationship('Variant', back_populates='drug_labels')

# Clinical Variants
class ClinicalVariant(db.Model):
    __tablename__ = 'clinical_variants'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    id = db.Column(db.Integer, primary_key=True)
    variant_type = db.Column(db.String)
    level_of_evidence = db.Column(db.String)
    gene_id = db.Column(db.String, db.ForeignKey('public.gene.gene_id'), index=True)
    
    gene = db.relationship('Gene', back_populates='clinical_variants')
    drugs = db.relationship('ClinicalVariantDrug', back_populates='clinical_variant', cascade="all, delete-orphan")
    phenotypes = db.relationship('ClinicalVariantPhenotype', back_populates='clinical_variant', cascade="all, delete-orphan")
    variants = db.relationship('ClinicalVariantVariant', back_populates='clinical_variant', cascade="all, delete-orphan")

class ClinicalVariantDrug(db.Model):
    __tablename__ = 'clinical_variant_drug'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    clinical_variant_id = db.Column(db.Integer, db.ForeignKey('public.clinical_variants.id'), primary_key=True)
    drug_id = db.Column(db.Integer, db.ForeignKey('public.drug.id'), primary_key=True)
    
    clinical_variant = db.relationship('ClinicalVariant', back_populates='drugs')
    drug = db.relationship('Drug', back_populates='clinical_variants')

class ClinicalVariantPhenotype(db.Model):
    __tablename__ = 'clinical_variant_phenotype'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    clinical_variant_id = db.Column(db.Integer, db.ForeignKey('public.clinical_variants.id'), primary_key=True)
    phenotype_id = db.Column(db.Integer, db.ForeignKey('public.phenotype.id'), primary_key=True)
    
    clinical_variant = db.relationship('ClinicalVariant', back_populates='phenotypes')
    phenotype = db.relationship('Phenotype', back_populates='clinical_variants')

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
    
    id = db.Column(db.Integer, primary_key=True)
    source_type = db.Column(db.String)
    source_id = db.Column(db.String, index=True)
    source_name = db.Column(db.Text)
    object_type = db.Column(db.String)
    object_id = db.Column(db.String, index=True)
    object_name = db.Column(db.String)

class AutomatedAnnotation(db.Model):
    __tablename__ = 'automated_annotations'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    id = db.Column(db.Integer, primary_key=True)
    chemical_id = db.Column(db.String, index=True, nullable=True)
    chemical_name = db.Column(db.String, nullable=True)
    chemical_in_text = db.Column(db.String, nullable=True)
    variation_id = db.Column(db.String, index=True, nullable=True)
    variation_name = db.Column(db.String, nullable=True)
    variation_type = db.Column(db.String, nullable=True)
    variation_in_text = db.Column(db.String, nullable=True)
    gene_ids = db.Column(db.Text, nullable=True)
    gene_symbols = db.Column(db.Text, index=True, nullable=True)
    gene_in_text = db.Column(db.Text, nullable=True)
    literature_id = db.Column(db.String, nullable=True)
    literature_title = db.Column(db.Text)
    publication_year = db.Column(db.String, nullable=True)
    journal = db.Column(db.Text, nullable=True)
    sentence = db.Column(db.Text)
    source = db.Column(db.String)
    pmid = db.Column(db.String(50), db.ForeignKey('public.publication.pmid'), index=True, nullable=True)
    
    publication = db.relationship('Publication', back_populates='automated_annotations')
# PharmGKB için database sonu.......




class News(db.Model):
    __tablename__ = 'news'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(50), nullable=False)
    drug_id = db.Column(db.Integer, db.ForeignKey('public.drug.id'), nullable=True)
    drug = db.relationship('Drug', backref='fda_news', lazy=True)
    publication_date = db.Column(db.Date, nullable=False, default=datetime.utcnow)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, onupdate=datetime.utcnow)

bcrypt = Bcrypt()

class User(db.Model):
    __tablename__ = 'user'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(128), nullable=False)
    name = db.Column(db.String(50), nullable=False)
    surname = db.Column(db.String(50), nullable=False)
    date_of_birth = db.Column(db.Date, nullable=False)
    occupation = db.Column(db.String(100), nullable=True)
    is_admin = db.Column(db.Boolean, default=False)
    is_verified = db.Column(db.Boolean, default=False)
    verification_code = db.Column(db.String(6), nullable=True)
    last_seen = db.Column(db.DateTime, default=datetime.utcnow)
    
    def set_password(self, password):
        self.password = bcrypt.generate_password_hash(password).decode('utf-8')
    
    def check_password(self, password):
        return bcrypt.check_password_hash(self.password, password)
    
    def is_online(self):
        if self.last_seen is None:
            return False
        return datetime.utcnow() - self.last_seen < timedelta(minutes=5)

class Occupation(db.Model):
    __tablename__ = 'occupation'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)

class Visitor(db.Model):
    __tablename__ = 'visitor'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    id = db.Column(db.Integer, primary_key=True)
    visitor_hash = db.Column(db.String(32), nullable=False, index=True)
    ip_address = db.Column(db.String(45))
    user_agent = db.Column(db.Text)
    country = db.Column(db.String(100))
    city = db.Column(db.String(100))
    region = db.Column(db.String(100))
    timezone = db.Column(db.String(50))
    isp = db.Column(db.String(255))
    page_url = db.Column(db.Text)
    referrer = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    language = db.Column(db.String(10))
    screen_width = db.Column(db.Integer)
    screen_height = db.Column(db.Integer)
    viewport_width = db.Column(db.Integer)
    viewport_height = db.Column(db.Integer)
    color_depth = db.Column(db.Integer)
    pixel_ratio = db.Column(db.Float)
    connection_type = db.Column(db.String(50))
    device_memory = db.Column(db.Integer)
    hardware_concurrency = db.Column(db.Integer)
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    accuracy = db.Column(db.Float)
    

class PageView(db.Model):
    __tablename__ = 'page_view'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    id = db.Column(db.Integer, primary_key=True)
    visitor_hash = db.Column(db.String(32), nullable=False, index=True)
    page_url = db.Column(db.Text)
    page_title = db.Column(db.String(255))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    time_on_page = db.Column(db.Integer)
    scroll_depth = db.Column(db.Integer)
    exit_page = db.Column(db.Boolean, default=False)

class VisitorSession(db.Model):
    __tablename__ = 'visitor_session'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
   
    id = db.Column(db.Integer, primary_key=True)
    visitor_hash = db.Column(db.String(32), nullable=False, index=True)
    session_start = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    session_end = db.Column(db.DateTime, nullable=True, index=True)
    duration_seconds = db.Column(db.Integer, nullable=True)
    pages_visited = db.Column(db.Integer, default=0)
    device_type = db.Column(db.String(50))
    browser = db.Column(db.String(100))
    os = db.Column(db.String(100))
    screen_resolution = db.Column(db.String(50))
    device_brand = db.Column(db.String(100))
    device_model = db.Column(db.String(100))
    is_bot = db.Column(db.Boolean, default=False)
    is_touch_capable = db.Column(db.Boolean, default=False)
    entry_page = db.Column(db.Text)
    exit_page = db.Column(db.Text)
    total_clicks = db.Column(db.Integer, default=0)
    total_scrolls = db.Column(db.Integer, default=0)
    max_scroll_depth = db.Column(db.Integer, default=0)


    
class SearchQuery(db.Model):
    """Track search queries made on the platform"""
    __tablename__ = 'search_query'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    id = db.Column(db.Integer, primary_key=True)
    query_text = db.Column(db.Text, nullable=False, index=True)
    visitor_hash = db.Column(db.String(32), index=True)
    user_id = db.Column(db.Integer, db.ForeignKey('public.user.id'), nullable=True, index=True)
    results_count = db.Column(db.Integer, default=0)
    category = db.Column(db.String(100))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    clicked_result_position = db.Column(db.Integer)
    clicked_result_id = db.Column(db.Integer)
    time_to_first_click = db.Column(db.Integer)
    refined_query = db.Column(db.Boolean, default=False)
    
    user = db.relationship('User', backref='searches')

class DrugView(db.Model):
    """Track individual drug page views"""
    __tablename__ = 'drug_view'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    id = db.Column(db.Integer, primary_key=True)
    drug_id = db.Column(db.Integer, db.ForeignKey('public.drug.id'), nullable=False, index=True)
    visitor_hash = db.Column(db.String(32), index=True)
    user_id = db.Column(db.Integer, db.ForeignKey('public.user.id'), nullable=True, index=True)
    view_duration = db.Column(db.Integer)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    sections_viewed = db.Column(db.JSON)
    scroll_depth = db.Column(db.Integer)
    interactions = db.Column(db.Integer, default=0)
    shared = db.Column(db.Boolean, default=False)
    bookmarked = db.Column(db.Boolean, default=False)
    
    drug = db.relationship('Drug', backref='views')
    user = db.relationship('User', backref='drug_views')

class InteractionCheck(db.Model):
    """Track drug interaction checks"""
    __tablename__ = 'interaction_check'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    id = db.Column(db.Integer, primary_key=True)
    drug_ids = db.Column(db.JSON, nullable=False)
    visitor_hash = db.Column(db.String(32), index=True)
    user_id = db.Column(db.Integer, db.ForeignKey('public.user.id'), nullable=True, index=True)
    interactions_found = db.Column(db.Integer, default=0)
    severity_levels = db.Column(db.JSON)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    checked_food_interactions = db.Column(db.Boolean, default=False)
    checked_disease_interactions = db.Column(db.Boolean, default=False)
    checked_lab_interactions = db.Column(db.Boolean, default=False)
    time_spent = db.Column(db.Integer)
    exported = db.Column(db.Boolean, default=False)
    
    user = db.relationship('User', backref='interaction_checks')

class AnalyticsEvent(db.Model):
    """Generic event tracking for custom analytics"""
    __tablename__ = 'analytics_event'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    id = db.Column(db.Integer, primary_key=True)
    event_type = db.Column(db.String(100), nullable=False, index=True)
    event_category = db.Column(db.String(100), index=True)
    event_data = db.Column(db.JSON)
    visitor_hash = db.Column(db.String(32), index=True)
    user_id = db.Column(db.Integer, db.ForeignKey('public.user.id'), nullable=True, index=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    page_url = db.Column(db.Text)
    event_value = db.Column(db.Float)
    
    user = db.relationship('User', backref='analytics_events')

class UserClick(db.Model):
    """Track click events"""
    __tablename__ = 'user_click'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    id = db.Column(db.Integer, primary_key=True)
    visitor_hash = db.Column(db.String(32), nullable=False, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey('public.user.id'), nullable=True, index=True)
    element_id = db.Column(db.String(255))
    element_class = db.Column(db.String(255))
    element_tag = db.Column(db.String(50))
    element_text = db.Column(db.Text)
    page_url = db.Column(db.Text)
    x_position = db.Column(db.Integer)
    y_position = db.Column(db.Integer)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    user = db.relationship('User', backref='clicks')

class UserScroll(db.Model):
    """Track scroll events"""
    __tablename__ = 'user_scroll'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    id = db.Column(db.Integer, primary_key=True)
    visitor_hash = db.Column(db.String(32), nullable=False, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey('public.user.id'), nullable=True, index=True)
    page_url = db.Column(db.Text)
    scroll_depth = db.Column(db.Integer)
    scroll_percentage = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    user = db.relationship('User', backref='scrolls')

class FormSubmission(db.Model):
    """Track form submissions"""
    __tablename__ = 'form_submission'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    id = db.Column(db.Integer, primary_key=True)
    visitor_hash = db.Column(db.String(32), nullable=False, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey('public.user.id'), nullable=True, index=True)
    form_name = db.Column(db.String(255))
    page_url = db.Column(db.Text)
    fields_filled = db.Column(db.JSON)
    submission_success = db.Column(db.Boolean, default=True)
    time_to_submit = db.Column(db.Integer)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    user = db.relationship('User', backref='form_submissions')

class ErrorLog(db.Model):
    """Track client-side errors"""
    __tablename__ = 'error_log'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    id = db.Column(db.Integer, primary_key=True)
    visitor_hash = db.Column(db.String(32), index=True)
    user_id = db.Column(db.Integer, db.ForeignKey('public.user.id'), nullable=True, index=True)
    error_message = db.Column(db.Text)
    error_stack = db.Column(db.Text)
    page_url = db.Column(db.Text)
    browser = db.Column(db.String(100))
    os = db.Column(db.String(100))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    user = db.relationship('User', backref='errors')

class DailyStats(db.Model):
    """Aggregated daily statistics for performance"""
    __tablename__ = 'daily_stats'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False, unique=True, index=True)
    unique_visitors = db.Column(db.Integer, default=0)
    total_pageviews = db.Column(db.Integer, default=0)
    total_searches = db.Column(db.Integer, default=0)
    total_drug_views = db.Column(db.Integer, default=0)
    total_interaction_checks = db.Column(db.Integer, default=0)
    new_users = db.Column(db.Integer, default=0)
    avg_session_duration = db.Column(db.Float)
    bounce_rate = db.Column(db.Float)
    top_drugs = db.Column(db.JSON)
    top_searches = db.Column(db.JSON)
    top_countries = db.Column(db.JSON)
    total_clicks = db.Column(db.Integer, default=0)
    total_scrolls = db.Column(db.Integer, default=0)
    total_errors = db.Column(db.Integer, default=0)
    mobile_percentage = db.Column(db.Float)
    tablet_percentage = db.Column(db.Float)
    desktop_percentage = db.Column(db.Float)
    avg_page_load_time = db.Column(db.Float)

class DoseResponseSimulation(db.Model):
    __tablename__ = 'dose_response_simulation'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    id = Column(String(36), primary_key=True)
    emax = Column(Float, nullable=False)
    ec50 = Column(Float, nullable=False)
    n = Column(Float, nullable=False)
    e0 = Column(Float, nullable=False, default=0.0)
    concentrations = Column(JSON, nullable=False)
    dosing_regimen = Column(String(50), nullable=False, default='single')
    doses = Column(JSON, nullable=True)
    intervals = Column(JSON, nullable=True)
    elimination_rate = Column(Float, nullable=True, default=0.1)
    created_at = Column(DateTime, default=datetime.utcnow)
    user_id = Column(Integer, nullable=True)
    concentration_unit = Column(String(20), default='µM')
    effect_unit = Column(String(20), default='%')


# DATABASE MODELS
# Database Models - Add these to your existing models section
class SUT_Version(db.Model):
    __tablename__ = 'sut_version'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    id = db.Column(db.Integer, primary_key=True)
    version_number = db.Column(db.Integer, nullable=False)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    filename = db.Column(db.String(255))
    file_hash = db.Column(db.String(64))
    is_active = db.Column(db.Boolean, default=True)
    items = db.relationship('SUT_Item', back_populates='version', cascade='all, delete-orphan')
    changes = db.relationship('SUT_Change', back_populates='version', cascade='all, delete-orphan')

class SUT_Item(db.Model):
    __tablename__ = 'sut_item'
    __table_args__ = (
        db.Index('idx_version_parent', 'version_id', 'parent_number'),
        db.Index('idx_version_item', 'version_id', 'item_number'),
        {'schema': 'public', 'extend_existing': True}
    )
    
    id = db.Column(db.Integer, primary_key=True)
    version_id = db.Column(db.Integer, db.ForeignKey('public.sut_version.id'), nullable=False)
    item_number = db.Column(db.String(50), nullable=False, index=True)
    title = db.Column(db.Text, nullable=False)
    content = db.Column(db.Text)
    parent_number = db.Column(db.String(50), index=True)
    level = db.Column(db.Integer)
    order_index = db.Column(db.Integer)
    full_text = db.Column(db.Text)
    version = db.relationship('SUT_Version', back_populates='items')

class SUT_Change(db.Model):
    __tablename__ = 'sut_change'
    __table_args__ = {'schema': 'public', 'extend_existing': True}
    
    id = db.Column(db.Integer, primary_key=True)
    version_id = db.Column(db.Integer, db.ForeignKey('public.sut_version.id'), nullable=False)
    change_date = db.Column(db.DateTime, default=datetime.utcnow)
    item_number = db.Column(db.String(50))
    change_type = db.Column(db.String(20))
    old_content = db.Column(db.Text)
    new_content = db.Column(db.Text)
    version = db.relationship('SUT_Version', back_populates='changes')
    
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
            flash("You are not an Admin!", "danger")
            return redirect(url_for('home'))
        return f(*args, **kwargs)
    return decorated_function

#Anasayfa
@app.route('/')
def home():
    try:
        announcements = News.query.filter(News.category.ilike('Announcement')).order_by(News.publication_date.desc()).all()
        updates = News.query.filter(News.category.ilike('Update')).order_by(News.publication_date.desc()).all()
        drug_updates = News.query.filter(News.category.ilike('Drug Update')).order_by(News.publication_date.desc()).all()
        fda_approvals = News.query.filter(News.category.ilike('FDA Approval')).order_by(News.publication_date.desc()).limit(5).all()  # Latest 5
    except Exception as e:
        print(f"Oops, toy box problem: {e}")
        announcements = updates = drug_updates = fda_approvals = []

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
        drug_updates=drug_updates,
        fda_approvals=fda_approvals,  # YENİ EKLEDİK!
        user_email=user_email,
        user=user
    )

#Online Users
# Add before_request handler
@app.before_request
def update_last_seen():
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        if user:
            user.last_seen = datetime.utcnow()
            db.session.commit()

# Add route for viewing online users
@app.route('/online-users')
def online_users():
    if 'user_id' not in session:
        flash('Please login first', 'warning')
        return redirect(url_for('login'))
    
    user = User.query.get(session['user_id'])
    if not user or not user.is_admin:
        flash('Access denied. Admin only.', 'danger')
        return redirect(url_for('home'))
    
    five_minutes_ago = datetime.utcnow() - timedelta(minutes=5)
    online_users_list = User.query.filter(User.last_seen >= five_minutes_ago, User.last_seen.isnot(None)).order_by(User.last_seen.desc()).all()
    all_users = User.query.order_by(User.last_seen.desc().nullslast()).all()
    
    return render_template('online_users.html', online_users=online_users_list, all_users=all_users, user=user)

# API endpoint for real-time updates
@app.route('/api/online-users')
def api_online_users():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    user = User.query.get(session['user_id'])
    if not user or not user.is_admin:
        return jsonify({'error': 'Forbidden'}), 403
    
    five_minutes_ago = datetime.utcnow() - timedelta(minutes=5)
    online_users_list = User.query.filter(User.last_seen >= five_minutes_ago, User.last_seen.isnot(None)).all()
    
    return jsonify({
        'online_count': len(online_users_list),
        'users': [{
            'id': u.id,
            'name': f"{u.name} {u.surname}",
            'email': u.email,
            'last_seen': u.last_seen.isoformat() if u.last_seen else None,
            'is_online': u.is_online()
        } for u in online_users_list]
    })

# Add this route to update all existing users with last_seen value
@app.route('/admin/init-last-seen')
def init_last_seen():
    if 'user_id' not in session:
        return "Unauthorized", 401
    
    user = User.query.get(session['user_id'])
    if not user or not user.is_admin:
        return "Forbidden", 403
    
    users = User.query.filter(User.last_seen.is_(None)).all()
    for u in users:
        u.last_seen = datetime.utcnow()
    db.session.commit()
    
    return f"Updated {len(users)} users with last_seen timestamp"

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

#KAyıt ve Login olma sayfaları
load_dotenv()

def send_verification_email(email, code):
    msg = MIMEText(f"Your verification code is: {code}")
    msg['Subject'] = 'Drugly.ai Email Verification'
    msg['From'] = os.getenv('EMAIL_ADDRESS')
    msg['To'] = email

    try:
        server = smtplib.SMTP('smtp.office365.com', 587)
        server.starttls()
        server.login(os.getenv('EMAIL_ADDRESS'), os.getenv('EMAIL_PASSWORD'))
        server.sendmail(msg['From'], msg['To'], msg.as_string())
        server.quit()
        return True
    except Exception as e:
        print(f"Email sending failed: {e}")
        return False

def send_reset_code(email, code):
    msg = MIMEText(f"Your password reset code is: {code}")
    msg['Subject'] = 'Drugly.ai Password Reset'
    msg['From'] = os.getenv('EMAIL_ADDRESS')
    msg['To'] = email

    try:
        server = smtplib.SMTP('smtp.office365.com', 587)
        server.starttls()
        server.login(os.getenv('EMAIL_ADDRESS'), os.getenv('EMAIL_PASSWORD'))
        server.sendmail(msg['From'], msg['To'], msg.as_string())
        server.quit()
        return True
    except Exception as e:
        print(f"Email sending failed: {e}")
        return False

def is_strong_password(password):
    if len(password) < 8:
        return False
    if not any(c.isupper() for c in password):
        return False
    if not any(c.isdigit() for c in password):
        return False
    if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
        return False
    return True

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash("Please log in to access this page.", "danger")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session:
        flash("You are already logged in.", "info")
        return redirect(url_for('home'))

    occupations = Occupation.query.order_by(Occupation.name).all()
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        name = request.form['name']
        surname = request.form['surname']
        date_of_birth = request.form['date_of_birth']
        occupation = request.form['occupation']

        if password != confirm_password:
            flash("Passwords do not match.", "danger")
            return redirect(url_for('register'))

        if not is_strong_password(password):
            flash("Password must be at least 8 characters, with an uppercase letter, number, and special character.", "danger")
            return redirect(url_for('register'))

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash("Email already registered.", "danger")
            return redirect(url_for('register'))

        verification_code = ''.join([str(random.randint(0, 9)) for _ in range(6)])
        new_user = User(
            email=email,
            name=name,
            surname=surname,
            date_of_birth=datetime.strptime(date_of_birth, '%Y-%m-%d'),
            occupation=occupation,
            verification_code=verification_code
        )
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()

        if send_verification_email(email, verification_code):
            flash("Registration successful! Please check your email for the verification code.", "success")
            session['pending_user_id'] = new_user.id
            return redirect(url_for('verify_email'))
        else:
            flash("Failed to send verification email. Please try again.", "danger")
            db.session.delete(new_user)
            db.session.commit()
            return redirect(url_for('register'))

    return render_template('register.html', occupations=occupations)

@app.route('/verify-email', methods=['GET', 'POST'])
def verify_email():
    if 'pending_user_id' not in session:
        flash("No pending verification found.", "danger")
        return redirect(url_for('register'))

    if request.method == 'POST':
        code = request.form['code']
        user = User.query.get(session['pending_user_id'])

        if user and user.verification_code == code:
            user.is_verified = True
            user.verification_code = None
            db.session.commit()
            session.pop('pending_user_id')
            flash("Email verified successfully! Please log in.", "success")
            return redirect(url_for('login'))
        else:
            flash("Invalid verification code.", "danger")

    return render_template('verify_email.html')

@app.route('/resend-verification', methods=['POST'])
def resend_verification():
    if 'pending_user_id' not in session:
        return jsonify({'success': False, 'message': 'No pending verification found.'})

    user = User.query.get(session['pending_user_id'])
    if not user:
        session.pop('pending_user_id', None)
        return jsonify({'success': False, 'message': 'User not found.'})

    verification_code = ''.join([str(random.randint(0, 9)) for _ in range(6)])
    user.verification_code = verification_code
    db.session.commit()

    if send_verification_email(user.email, verification_code):
        return jsonify({'success': True, 'message': 'New verification code sent to your email.'})
    else:
        return jsonify({'success': False, 'message': 'Failed to send verification code. Please try again.'})

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and user.is_verified and user.check_password(password):
            session['user_id'] = user.id
            flash("Welcome back!", "success")
            return redirect(url_for('home'))
        elif user and not user.is_verified:
            flash("Please verify your email before logging in.", "danger")
        else:
            flash("Invalid email or password.", "danger")
    return render_template('login.html')

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']
        user = User.query.filter_by(email=email).first()
        if user:
            reset_code = ''.join([str(random.randint(0, 9)) for _ in range(6)])
            user.verification_code = reset_code
            db.session.commit()

            if send_reset_code(email, reset_code):
                flash("A reset code has been sent to your email.", "success")
                session['reset_user_id'] = user.id
                return redirect(url_for('reset_password'))
            else:
                flash("Failed to send reset code. Please try again.", "danger")
        else:
            flash("Email not found.", "danger")
    return render_template('forgot_password.html')

@app.route('/reset-password', methods=['GET', 'POST'])
def reset_password():
    if 'reset_user_id' not in session:
        flash("No reset request found.", "danger")
        return redirect(url_for('forgot_password'))

    if request.method == 'POST':
        code = request.form['code']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        user = User.query.get(session['reset_user_id'])

        if password != confirm_password:
            flash("Passwords do not match.", "danger")
            return redirect(url_for('reset_password'))

        if not is_strong_password(password):
            flash("Password must be at least 8 characters, with an uppercase letter, number, and special character.", "danger")
            return redirect(url_for('reset_password'))

        if user and user.verification_code == code:
            user.set_password(password)
            user.verification_code = None
            db.session.commit()
            session.pop('reset_user_id')
            flash("Password reset successfully! Please log in.", "success")
            return redirect(url_for('login'))
        else:
            flash("Invalid reset code.", "danger")

    return render_template('reset_password.html')

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    user = User.query.get(session['user_id'])
    occupations = Occupation.query.order_by(Occupation.name).all()

    if request.method == 'POST':
        user.name = request.form['name']
        user.surname = request.form['surname']
        user.date_of_birth = datetime.strptime(request.form['date_of_birth'], '%Y-%m-%d')
        user.occupation = request.form['occupation']
        db.session.commit()

        flash("Profile updated successfully!", "success")
        return redirect(url_for('profile'))

    return render_template('profile.html', user=user, occupations=occupations, user_email=user.email)

@app.route('/change-password', methods=['POST'])
@login_required
def change_password():
    user = User.query.get(session['user_id'])
    current_password = request.form['current_password']
    new_password = request.form['new_password']
    confirm_new_password = request.form['confirm_new_password']

    if not user.check_password(current_password):
        flash("Current password is incorrect.", "danger")
        return redirect(url_for('profile'))

    if new_password != confirm_new_password:
        flash("New passwords do not match.", "danger")
        return redirect(url_for('profile'))

    if not is_strong_password(new_password):
        flash("New password must be at least 8 characters, with an uppercase letter, number, and special character.", "danger")
        return redirect(url_for('profile'))

    user.set_password(new_password)
    db.session.commit()
    flash("Password changed successfully!", "success")
    return redirect(url_for('profile'))

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('pending_user_id', None)
    session.pop('reset_user_id', None)
    flash("You have been logged out.", "info")
    return redirect(url_for('home'))

#CLEARING THE WHOLEDATABASE:
@app.route('/clear_database', methods=['GET', 'POST'])
def clear_database():
    # Session-based login kontrolü
    if 'user_id' not in session:
        flash('Bu sayfaya erişmek için giriş yapmalısınız.', 'danger')
        return redirect(url_for('login'))
    # Admin kontrolü
    user = db.session.get(User, session['user_id'])
    if not user or not user.is_admin:
        flash('Bu işlem için yetkiniz yok.', 'danger')
        return redirect(url_for('home'))
    if request.method == 'POST':
        try:
            # Önce bağımlı tabloları sil (foreign key bağımlılıkları dikkate alınarak)
            db.session.execute(drug_route_metabolism_organ.delete())
            db.session.execute(drug_route_metabolism_enzyme.delete())
            db.session.execute(drug_route_metabolite.delete())
            db.session.execute(drug_salt.delete())
            db.session.execute(detail_side_effect.delete())
            db.session.execute(pathway_drug.delete())
            db.session.execute(interaction_route.delete())
            db.session.execute(drug_category_association.delete())
            db.session.execute(ClinicalAnnotationDrug.__table__.delete())
            db.session.execute(ClinicalAnnotationGene.__table__.delete())
            db.session.execute(ClinicalAnnotationPhenotype.__table__.delete())
            db.session.execute(ClinicalAnnotationVariant.__table__.delete())
            db.session.execute(ClinicalAnnEvidencePublication.__table__.delete())
            db.session.execute(VariantAnnotationDrug.__table__.delete())
            db.session.execute(VariantAnnotationGene.__table__.delete())
            db.session.execute(VariantAnnotationVariant.__table__.delete())
            db.session.execute(DrugLabelDrug.__table__.delete())
            db.session.execute(DrugLabelGene.__table__.delete())
            db.session.execute(DrugLabelVariant.__table__.delete())
            db.session.execute(ClinicalVariantDrug.__table__.delete())
            db.session.execute(ClinicalVariantPhenotype.__table__.delete())
            db.session.execute(ClinicalVariantVariant.__table__.delete())
            # Bağımlı ana tablolar
            db.session.execute(RouteIndication.__table__.delete())
            db.session.execute(DrugRoute.__table__.delete())
            db.session.execute(DrugLabTestInteraction.__table__.delete())
            db.session.execute(DrugDiseaseInteraction.__table__.delete())
            db.session.execute(DrugReceptorInteraction.__table__.delete())
            db.session.execute(DrugDetail.__table__.delete())
            db.session.execute(DrugInteraction.__table__.delete())
            db.session.execute(Metabolite.__table__.delete())  # Drug'dan önce silindi
            db.session.execute(Drug.__table__.delete())
            db.session.execute(DrugCategory.__table__.delete())
            db.session.execute(Salt.__table__.delete())
            db.session.execute(SafetyCategory.__table__.delete())
            db.session.execute(Indication.__table__.delete())
            db.session.execute(Target.__table__.delete())
            db.session.execute(Severity.__table__.delete())
            #db.session.execute(RouteOfAdministration.__table__.delete())
            db.session.execute(Receptor.__table__.delete())
            #db.session.execute(LabTest.__table__.delete())
            #db.session.execute(Unit.__table__.delete())
            db.session.execute(MetabolismOrgan.__table__.delete())
            db.session.execute(MetabolismEnzyme.__table__.delete())
            db.session.execute(Gene.__table__.delete())
            db.session.execute(Phenotype.__table__.delete())
            db.session.execute(Variant.__table__.delete())
            db.session.execute(Publication.__table__.delete())
            
            # FIXED: Delete child tables before parent tables
            db.session.execute(ClinicalAnnAllele.__table__.delete())  # Moved before ClinicalAnnotation
            db.session.execute(ClinicalAnnHistory.__table__.delete())  # Moved before ClinicalAnnotation
            db.session.execute(ClinicalAnnEvidence.__table__.delete())  # Moved before ClinicalAnnotation
            db.session.execute(ClinicalAnnotation.__table__.delete())  # Now safe to delete
            
            db.session.execute(VariantAnnotation.__table__.delete())
            db.session.execute(StudyParameters.__table__.delete())
            db.session.execute(VariantFAAnn.__table__.delete())
            db.session.execute(VariantDrugAnn.__table__.delete())
            db.session.execute(VariantPhenoAnn.__table__.delete())
            db.session.execute(Relationship.__table__.delete())
            db.session.execute(DrugLabel.__table__.delete())
            db.session.execute(ClinicalVariant.__table__.delete())
            db.session.execute(AutomatedAnnotation.__table__.delete())
            db.session.execute(News.__table__.delete())
            #db.session.execute(User.__table__.delete())
            #db.session.execute(Occupation.__table__.delete())
            db.session.execute(DoseResponseSimulation.__table__.delete())
            
            db.session.commit()
            
            # Reset all auto-increment sequences to start from 1
            logger.info("Resetting all database sequences...")
            result = db.session.execute(db.text("""
                SELECT sequence_name 
                FROM information_schema.sequences 
                WHERE sequence_schema = 'public'
            """))
            
            sequences = [row[0] for row in result]
            
            for seq in sequences:
                try:
                    db.session.execute(db.text(f"ALTER SEQUENCE {seq} RESTART WITH 1"))
                    logger.info(f"Reset sequence: {seq}")
                except Exception as seq_error:
                    logger.warning(f"Could not reset sequence {seq}: {seq_error}")
            
            db.session.commit()
            logger.info("All sequences reset successfully")
            
            flash('Veritabanı başarıyla temizlendi ve ID\'ler sıfırlandı.', 'success')
        except SQLAlchemyError as e:
            db.session.rollback()
            flash(f'Veritabanı temizlenirken hata oluştu: {str(e)}', 'error')
            logging.error(f"Database clear error: {str(e)}")
        return redirect(url_for('clear_database'))
    return render_template('clear_database.html')
#THE END OF CLEARING THE WHOLEDATABASE

#ICD -11 tree view
@app.route('/icd11')
def icd11_view():
    if 'user_id' not in session:
        flash('Please log in to view the ICD-11 hierarchy.', 'warning')
        return redirect(url_for('login'))
    
    user = User.query.get(session['user_id'])
    if not user:
        flash('User not found. Please log in again.', 'error')
        return redirect(url_for('login'))
    
    return render_template('icd11_view.html', user_email=user.email)

# ATC CODES
ALLOWED_EXTENSIONS = {'csv'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def import_atc_codes(csv_file_path):
    level1_cache = {}
    level2_cache = {}
    level3_cache = {}
    level4_cache = {}
    level5_cache = {}
    
    with open(csv_file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            atc_code = row['atc_code'].strip()
            atc_name = row['atc_name'].strip()
            ddd = row['ddd'].strip() if row['ddd'].strip() != 'NA' else None
            uom = row['uom'].strip() if row['uom'].strip() != 'NA' else None
            adm_r = row['adm_r'].strip() if row['adm_r'].strip() != 'NA' else None
            note = row['note'].strip() if row['note'].strip() != 'NA' else None
            
            code_length = len(atc_code)
            
            if code_length == 1:
                if atc_code not in level1_cache:
                    level1 = ATCLevel1.query.filter_by(code=atc_code).first()
                    if not level1:
                        level1 = ATCLevel1(code=atc_code, name=atc_name)
                        db.session.add(level1)
                        db.session.flush()
                    else:
                        level1.name = atc_name
                    level1_cache[atc_code] = level1
            
            elif code_length == 3:
                level1_code = atc_code[0]
                if level1_code not in level1_cache:
                    level1_cache[level1_code] = ATCLevel1.query.filter_by(code=level1_code).first()
                
                if atc_code not in level2_cache:
                    level2 = ATCLevel2.query.filter_by(code=atc_code).first()
                    if not level2:
                        level2 = ATCLevel2(
                            code=atc_code,
                            name=atc_name,
                            level1_id=level1_cache[level1_code].id
                        )
                        db.session.add(level2)
                        db.session.flush()
                    else:
                        level2.name = atc_name
                        level2.level1_id = level1_cache[level1_code].id
                    level2_cache[atc_code] = level2
            
            elif code_length == 4:
                level2_code = atc_code[:3]
                if level2_code not in level2_cache:
                    level2_cache[level2_code] = ATCLevel2.query.filter_by(code=level2_code).first()
                
                if atc_code not in level3_cache:
                    level3 = ATCLevel3.query.filter_by(code=atc_code).first()
                    if not level3:
                        level3 = ATCLevel3(
                            code=atc_code,
                            name=atc_name,
                            level2_id=level2_cache[level2_code].id
                        )
                        db.session.add(level3)
                        db.session.flush()
                    else:
                        level3.name = atc_name
                        level3.level2_id = level2_cache[level2_code].id
                    level3_cache[atc_code] = level3
            
            elif code_length == 5:
                level3_code = atc_code[:4]
                if level3_code not in level3_cache:
                    level3_cache[level3_code] = ATCLevel3.query.filter_by(code=level3_code).first()
                
                if atc_code not in level4_cache:
                    level4 = ATCLevel4.query.filter_by(code=atc_code).first()
                    if not level4:
                        level4 = ATCLevel4(
                            code=atc_code,
                            name=atc_name,
                            level3_id=level3_cache[level3_code].id
                        )
                        db.session.add(level4)
                        db.session.flush()
                    else:
                        level4.name = atc_name
                        level4.level3_id = level3_cache[level3_code].id
                    level4_cache[atc_code] = level4
            
            elif code_length == 7:
                level4_code = atc_code[:5]
                if level4_code not in level4_cache:
                    level4_cache[level4_code] = ATCLevel4.query.filter_by(code=level4_code).first()
                
                cache_key = f"{atc_code}_{adm_r}"
                if cache_key not in level5_cache:
                    level5 = ATCLevel5.query.filter_by(code=atc_code, adm_r=adm_r).first()
                    if not level5:
                        level5 = ATCLevel5(
                            code=atc_code,
                            name=atc_name,
                            level4_id=level4_cache[level4_code].id,
                            ddd=ddd,
                            uom=uom,
                            adm_r=adm_r,
                            note=note
                        )
                        db.session.add(level5)
                        db.session.flush()
                    else:
                        level5.name = atc_name
                        level5.level4_id = level4_cache[level4_code].id
                        level5.ddd = ddd
                        level5.uom = uom
                        level5.note = note
                    level5_cache[cache_key] = level5
    
    db.session.commit()
    print("ATC codes imported successfully!")

def match_drugs_to_atc():
    drugs = Drug.query.all()
    matched_count = 0
    unmatched_count = 0
    
    for drug in drugs:
        drug_name = drug.name_en.lower().strip()
        
        atc_level5_entries = ATCLevel5.query.filter(
            db.func.lower(ATCLevel5.name) == drug_name
        ).all()
        
        if atc_level5_entries:
            for atc_entry in atc_level5_entries:
                existing_mapping = DrugATCMapping.query.filter_by(
                    drug_id=drug.id,
                    atc_level5_id=atc_entry.id
                ).first()
                
                if not existing_mapping:
                    mapping = DrugATCMapping(
                        drug_id=drug.id,
                        atc_level5_id=atc_entry.id
                    )
                    db.session.add(mapping)
            
            matched_count += 1
            print(f"Matched: {drug.name_en} -> {len(atc_level5_entries)} ATC code(s)")
        else:
            unmatched_count += 1
            print(f"No match: {drug.name_en}")
    
    db.session.commit()
    print(f"\nMatching complete!")
    print(f"Matched drugs: {matched_count}")
    print(f"Unmatched drugs: {unmatched_count}")


@app.route('/atc')
def atc_view():
    if 'user_id' not in session:
        flash('Please log in to view the ATC hierarchy.', 'warning')
        return redirect(url_for('login'))
    
    user = User.query.get(session['user_id'])
    if not user:
        flash('User not found. Please log in again.', 'error')
        return redirect(url_for('login'))
    
    from sqlalchemy.orm import joinedload
    
    try:
        level1_entries = ATCLevel1.query.options(
            joinedload(ATCLevel1.level2_children)
            .joinedload(ATCLevel2.level3_children)
            .joinedload(ATCLevel3.level4_children)
            .joinedload(ATCLevel4.level5_children)
        ).order_by(ATCLevel1.code).all()
        
        atc_hierarchy = []
        for level1 in level1_entries:
            level1_data = {
                'level1': level1,
                'level2_list': []
            }
            
            for level2 in sorted(level1.level2_children, key=lambda x: x.code):
                level2_data = {
                    'level2': level2,
                    'level3_list': []
                }
                
                for level3 in sorted(level2.level3_children, key=lambda x: x.code):
                    level3_data = {
                        'level3': level3,
                        'level4_list': []
                    }
                    
                    for level4 in sorted(level3.level4_children, key=lambda x: x.code):
                        level4_data = {
                            'level4': level4,
                            'level5_list': sorted(level4.level5_children, key=lambda x: x.code)
                        }
                        level3_data['level4_list'].append(level4_data)
                    
                    level2_data['level3_list'].append(level3_data)
                
                level1_data['level2_list'].append(level2_data)
            
            atc_hierarchy.append(level1_data)
        
        return render_template('atc_view.html', atc_hierarchy=atc_hierarchy, user_email=user.email, user=user)
        
    except Exception as e:
        print(f"Error loading ATC hierarchy: {str(e)}")
        import traceback
        traceback.print_exc()
        flash(f'Error loading ATC hierarchy: {str(e)}', 'danger')
        return render_template('atc_view.html', atc_hierarchy=[], user_email=user.email, user=user)

@app.route('/atc/<string:code>')
def atc_detail(code):
    code = code.upper().strip()
    code_length = len(code)
    
    if code_length == 1:
        level1 = ATCLevel1.query.filter_by(code=code).first_or_404()
        return render_template('atc_detail.html', level='1', data=level1)
    elif code_length == 3:
        level2 = ATCLevel2.query.filter_by(code=code).first_or_404()
        return render_template('atc_detail.html', level='2', data=level2)
    elif code_length == 4:
        level3 = ATCLevel3.query.filter_by(code=code).first_or_404()
        return render_template('atc_detail.html', level='3', data=level3)
    elif code_length == 5:
        level4 = ATCLevel4.query.filter_by(code=code).first_or_404()
        return render_template('atc_detail.html', level='4', data=level4)
    elif code_length == 7:
        level5_entries = ATCLevel5.query.filter_by(code=code).all()
        if not level5_entries:
            flash('ATC code not found!', 'danger')
            return redirect(url_for('atc_view'))
        return render_template('atc_detail.html', level='5', data=level5_entries)
    else:
        flash('Invalid ATC code format!', 'danger')
        return redirect(url_for('atc_view'))

@app.route('/atc_upload', methods=['GET', 'POST'])
@admin_required
def atc_upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected!', 'danger')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected!', 'danger')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"atc_{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(filepath)
            
            try:
                import_atc_codes(filepath)
                flash('ATC codes imported successfully!', 'success')
            except Exception as e:
                flash(f'Error importing ATC codes: {str(e)}', 'danger')
                db.session.rollback()
            
            return redirect(url_for('atc_upload'))
        else:
            flash('Invalid file type! Only CSV files are allowed.', 'danger')
            return redirect(request.url)
    
    total_level1 = ATCLevel1.query.count()
    total_level2 = ATCLevel2.query.count()
    total_level3 = ATCLevel3.query.count()
    total_level4 = ATCLevel4.query.count()
    total_level5 = ATCLevel5.query.count()
    total_mappings = DrugATCMapping.query.count()
    
    recent_uploads = []
    if os.path.exists(app.config['UPLOAD_FOLDER']):
        files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.startswith('atc_') and f.endswith('.csv')]
        files.sort(reverse=True)
        recent_uploads = files[:10]
    
    return render_template('atc_upload.html', 
                         total_level1=total_level1,
                         total_level2=total_level2,
                         total_level3=total_level3,
                         total_level4=total_level4,
                         total_level5=total_level5,
                         total_mappings=total_mappings,
                         recent_uploads=recent_uploads)

@app.route('/atc_upload/manual', methods=['GET', 'POST'])
@admin_required
def atc_manual_add():
    if request.method == 'POST':
        level = request.form.get('level')
        code = request.form.get('code').upper().strip()
        name = request.form.get('name').strip()
        
        try:
            if level == '1':
                if ATCLevel1.query.filter_by(code=code).first():
                    flash('ATC Level 1 code already exists!', 'danger')
                    return redirect(url_for('atc_manual_add'))
                
                new_entry = ATCLevel1(code=code, name=name)
                db.session.add(new_entry)
                db.session.commit()
                flash(f'ATC Level 1 code {code} added successfully!', 'success')
            
            elif level == '2':
                level1_id = request.form.get('level1_id', type=int)
                if not level1_id:
                    flash('Parent Level 1 is required!', 'danger')
                    return redirect(url_for('atc_manual_add'))
                
                if ATCLevel2.query.filter_by(code=code).first():
                    flash('ATC Level 2 code already exists!', 'danger')
                    return redirect(url_for('atc_manual_add'))
                
                new_entry = ATCLevel2(code=code, name=name, level1_id=level1_id)
                db.session.add(new_entry)
                db.session.commit()
                flash(f'ATC Level 2 code {code} added successfully!', 'success')
            
            elif level == '3':
                level2_id = request.form.get('level2_id', type=int)
                if not level2_id:
                    flash('Parent Level 2 is required!', 'danger')
                    return redirect(url_for('atc_manual_add'))
                
                if ATCLevel3.query.filter_by(code=code).first():
                    flash('ATC Level 3 code already exists!', 'danger')
                    return redirect(url_for('atc_manual_add'))
                
                new_entry = ATCLevel3(code=code, name=name, level2_id=level2_id)
                db.session.add(new_entry)
                db.session.commit()
                flash(f'ATC Level 3 code {code} added successfully!', 'success')
            
            elif level == '4':
                level3_id = request.form.get('level3_id', type=int)
                if not level3_id:
                    flash('Parent Level 3 is required!', 'danger')
                    return redirect(url_for('atc_manual_add'))
                
                if ATCLevel4.query.filter_by(code=code).first():
                    flash('ATC Level 4 code already exists!', 'danger')
                    return redirect(url_for('atc_manual_add'))
                
                new_entry = ATCLevel4(code=code, name=name, level3_id=level3_id)
                db.session.add(new_entry)
                db.session.commit()
                flash(f'ATC Level 4 code {code} added successfully!', 'success')
            
            elif level == '5':
                level4_id = request.form.get('level4_id', type=int)
                ddd = request.form.get('ddd')
                uom = request.form.get('uom')
                adm_r = request.form.get('adm_r')
                note = request.form.get('note')
                
                if not level4_id:
                    flash('Parent Level 4 is required!', 'danger')
                    return redirect(url_for('atc_manual_add'))
                
                ddd = ddd if ddd and ddd.upper() != 'NA' else None
                uom = uom if uom and uom.upper() != 'NA' else None
                adm_r = adm_r if adm_r and adm_r.upper() != 'NA' else None
                note = note if note and note.upper() != 'NA' else None
                
                existing = ATCLevel5.query.filter_by(code=code, adm_r=adm_r).first()
                if existing:
                    flash('ATC Level 5 code with this administration route already exists!', 'danger')
                    return redirect(url_for('atc_manual_add'))
                
                new_entry = ATCLevel5(
                    code=code, 
                    name=name, 
                    level4_id=level4_id,
                    ddd=ddd,
                    uom=uom,
                    adm_r=adm_r,
                    note=note
                )
                db.session.add(new_entry)
                db.session.commit()
                flash(f'ATC Level 5 code {code} added successfully!', 'success')
            
            return redirect(url_for('atc_upload'))
        
        except Exception as e:
            db.session.rollback()
            flash(f'Error adding ATC code: {str(e)}', 'danger')
            return redirect(url_for('atc_manual_add'))
    
    level1_list = ATCLevel1.query.order_by(ATCLevel1.code).all()
    level2_list = ATCLevel2.query.order_by(ATCLevel2.code).all()
    level3_list = ATCLevel3.query.order_by(ATCLevel3.code).all()
    level4_list = ATCLevel4.query.order_by(ATCLevel4.code).all()
    
    return render_template('atc_manual_add.html',
                         level1_list=level1_list,
                         level2_list=level2_list,
                         level3_list=level3_list,
                         level4_list=level4_list)

@app.route('/atc_upload/match_drugs', methods=['POST'])
@admin_required
def match_drugs_route():
    try:
        match_drugs_to_atc()
        flash('Drug matching completed successfully!', 'success')
    except Exception as e:
        flash(f'Error matching drugs: {str(e)}', 'danger')
    
    return redirect(url_for('atc_upload'))

@app.route('/atc_upload/delete/<string:level>/<int:id>', methods=['POST'])
@admin_required
def delete_atc(level, id):
    try:
        if level == '1':
            entry = ATCLevel1.query.get_or_404(id)
        elif level == '2':
            entry = ATCLevel2.query.get_or_404(id)
        elif level == '3':
            entry = ATCLevel3.query.get_or_404(id)
        elif level == '4':
            entry = ATCLevel4.query.get_or_404(id)
        elif level == '5':
            entry = ATCLevel5.query.get_or_404(id)
        else:
            flash('Invalid level!', 'danger')
            return redirect(url_for('atc_upload'))
        
        code = entry.code
        db.session.delete(entry)
        db.session.commit()
        flash(f'ATC code {code} deleted successfully!', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting ATC code: {str(e)}', 'danger')
    
    return redirect(url_for('atc_upload'))

@app.route('/api/atc/search')
def atc_search():
    query = request.args.get('q', '').lower()
    if len(query) < 2:
        return jsonify([])
    
    results = []
    
    level5_results = ATCLevel5.query.filter(
        db.or_(
            ATCLevel5.code.ilike(f'%{query}%'),
            ATCLevel5.name.ilike(f'%{query}%')
        )
    ).limit(20).all()
    
    for item in level5_results:
        results.append({
            'code': item.code,
            'name': item.name,
            'level': 5,
            'id': item.id
        })
    
    return jsonify(results)




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
    if request.method == 'POST':
        drug_id = request.form.get('drug_id', type=int)
        salt_id = request.form.get('salt_id', type=int)
        if drug_id and salt_id:
            existing = db.session.query(drug_salt).filter(and_(
                drug_salt.c.drug_id == drug_id,
                drug_salt.c.salt_id == salt_id
            )).first()
            if not existing:
                db.session.execute(drug_salt.insert().values(drug_id=drug_id, salt_id=salt_id))
                db.session.commit()
                flash('Drug-Salt association added successfully!', 'success')
            else:
                flash('This drug-salt association already exists.', 'warning')
        else:
            flash('Please select both a drug and a salt.', 'error')
        return redirect(url_for('manage_matches'))

    # Pagination for drugs
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 15, type=int)
    drugs_query = Drug.query.paginate(page=page, per_page=per_page, error_out=False)
    drugs = drugs_query.items
    total_pages = drugs_query.pages
    current_page = drugs_query.page

    # Fetch all salts
    salts = Salt.query.all()

    # Fetch existing matches with explicit joins
    matches = db.session.query(Drug, Salt).join(
        drug_salt, Drug.id == drug_salt.c.drug_id
    ).join(
        Salt, Salt.id == drug_salt.c.salt_id
    ).all()

    return render_template(
        'matches.html',
        drugs=drugs,
        salts=salts,
        matches=matches,
        total_pages=total_pages,
        current_page=current_page,
        per_page=per_page
    )

def generate_3d_structure(smiles, output_filename):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.error(f"Invalid SMILES string: {smiles}")
            return None, "Invalid SMILES string"
        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
            logger.error("Failed to generate 3D coordinates")
            return None, "Failed to generate 3D coordinates"
        AllChem.MMFFOptimizeMolecule(mol)
        output_path = os.path.join('static', 'uploads', '3d_structures', output_filename).replace('\\', '/')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        Chem.MolToPDBFile(mol, output_path)
        logger.info(f"3D structure generated at {output_path}")
        return os.path.join('uploads', '3d_structures', output_filename).replace('\\', '/'), None
    except Exception as e:
        logger.error(f"Error generating 3D structure: {str(e)}")
        return None, f"Error generating 3D structure: {str(e)}"


# Details_Add Route:

# Define allowed tags, attributes, and styles for sanitization
ALLOWED_TAGS = ['b', 'i', 'u', 'p', 'strong', 'em', 'span', 'ul', 'ol', 'li', 'a']
ALLOWED_STYLES = ['color', 'background-color']

# Replace the existing allow_style_attrs function and cleaner setup with:
def allow_style_attrs(tag, name, value):
    if name == 'style':
        # Parse and filter style attribute
        styles = [style.strip() for style in value.split(';') if style.strip()]
        filtered_styles = []
        for style in styles:
            if ':' not in style:
                continue
            prop = style.split(':', 1)[0].strip().lower()
            if prop in ALLOWED_STYLES:
                filtered_styles.append(style)
        return bool(filtered_styles)  # Return True/False, not the value
    elif name == 'href' and tag == 'a':
        return True
    return False

# Use bleach.clean consistently, not the cleaner object
def sanitize_field(field_value):
    if not field_value:
        return None
    return bleach.clean(
        field_value,
        tags=ALLOWED_TAGS,
        attributes={'a': ['href'], 'span': allow_style_attrs, 'p': allow_style_attrs},
        strip=True
    )

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
    units = Unit.query.all()
    safety_categories = SafetyCategory.query.all()
    categories = DrugCategory.query.order_by(DrugCategory.name).all()
    atc_level5_list = ATCLevel5.query.order_by(ATCLevel5.code).all()  # NEW: Get all ATC codes
    units_json = [{'id': unit.id, 'name': unit.name} for unit in units]
    
    if request.method == 'POST':
        logger.debug("Entering POST block")
        logger.debug(f"Form data: {request.form}")
        drug_id = request.form.get('drug_id')
        salt_id = request.form.get('salt_id')
        selected_category_ids = request.form.getlist('category_ids[]')
        selected_atc_codes = request.form.getlist('atc_codes[]')  # NEW: Get selected ATC codes
        
        # Validate drug_id
        if not drug_id or not drug_id.isdigit():
            logger.error("Invalid drug_id: must be an integer")
            return render_template(
                'add_detail.html', drugs=drugs, salts=salts, indications=indications,
                targets=targets, routes=routes, side_effects=side_effects, metabolites=metabolites,
                metabolism_organs=metabolism_organs, metabolism_enzymes=metabolism_enzymes,
                units=units, units_json=units_json, safety_categories=safety_categories, categories=categories,
                atc_level5_list=atc_level5_list,  # NEW
                error_message="Drug ID is required and must be an integer!"
            )
        drug_id = int(drug_id)
        drug = Drug.query.get(drug_id)
        if not drug:
            logger.error(f"Drug with ID {drug_id} not found")
            return render_template(
                'add_detail.html', drugs=drugs, salts=salts, indications=indications,
                targets=targets, routes=routes, side_effects=side_effects, metabolites=metabolites,
                metabolism_organs=metabolism_organs, metabolism_enzymes=metabolism_enzymes,
                units=units, units_json=units_json, safety_categories=safety_categories, categories=categories,
                atc_level5_list=atc_level5_list,  # NEW
                error_message="Drug not found!"
            )
        
        # Validate and convert salt_id
        if salt_id == '' or salt_id is None:
            salt_id = None
        else:
            try:
                salt_id = int(salt_id)
            except ValueError:
                logger.error(f"Invalid salt_id: {salt_id}")
                return render_template(
                    'add_detail.html', drugs=drugs, salts=salts, indications=indications,
                    targets=targets, routes=routes, side_effects=side_effects, metabolites=metabolites,
                    metabolism_organs=metabolism_organs, metabolism_enzymes=metabolism_enzymes,
                    units=units, units_json=units_json, safety_categories=safety_categories, categories=categories,
                    atc_level5_list=atc_level5_list,  # NEW
                    error_message="Invalid salt_id. Must be an integer or empty."
                )
        
        # Validate category IDs
        valid_category_ids = []
        if selected_category_ids:
            for cat_id in selected_category_ids:
                try:
                    cat_id = int(cat_id)
                    if DrugCategory.query.get(cat_id):
                        valid_category_ids.append(cat_id)
                    else:
                        logger.warning(f"Invalid category ID: {cat_id}")
                except ValueError:
                    logger.warning(f"Invalid category ID format: {cat_id}")
        logger.debug(f"Valid category IDs: {valid_category_ids}")
        
        # NEW: Validate ATC codes
        valid_atc_ids = []
        if selected_atc_codes:
            for atc_id in selected_atc_codes:
                try:
                    atc_id = int(atc_id)
                    if ATCLevel5.query.get(atc_id):
                        valid_atc_ids.append(atc_id)
                    else:
                        logger.warning(f"Invalid ATC ID: {atc_id}")
                except ValueError:
                    logger.warning(f"Invalid ATC ID format: {atc_id}")
        logger.debug(f"Valid ATC IDs: {valid_atc_ids}")
        
        # Check for existing detail
        existing_detail = DrugDetail.query.filter_by(drug_id=drug_id, salt_id=salt_id).first()
        if existing_detail:
            logger.error(f"DrugDetail already exists for drug_id={drug_id}, salt_id={salt_id}")
            return render_template(
                'add_detail.html', drugs=drugs, salts=salts, indications=indications,
                targets=targets, routes=routes, side_effects=side_effects, metabolites=metabolites,
                metabolism_organs=metabolism_organs, metabolism_enzymes=metabolism_enzymes,
                units=units, units_json=units_json, safety_categories=safety_categories, categories=categories,
                atc_level5_list=atc_level5_list,  # NEW
                error_message="Bu etken madde ve tuz kombinasyonu için zaten detaylı bilgi mevcut!"
            )
        
        try:
            # Sanitize rich text fields
            def clean_text(field):
                try:
                    return bleach.clean(
                        field,
                        tags=['b', 'i', 'u', 'p', 'strong', 'em', 'span', 'ul', 'ol', 'li', 'a'],
                        attributes={'span': ['style'], 'a': ['href']},
                        styles=['color', 'background-color']
                    ) if field else None
                except TypeError as e:
                    logger.warning(f"bleach.clean failed with styles: {str(e)}, retrying without styles")
                    return bleach.clean(
                        field,
                        tags=['b', 'i', 'u', 'p', 'strong', 'em', 'span', 'ul', 'ol', 'li', 'a'],
                        attributes={'span': ['style'], 'a': ['href']}
                    ) if field else None
            
            mechanism_of_action = clean_text(request.form.get('mechanism_of_action'))
            references = clean_text(request.form.get('references'))
            synthesis = clean_text(request.form.get('synthesis'))
            pharmacodynamics = clean_text(request.form.get('pharmacodynamics'))
            pharmacokinetics = clean_text(request.form.get('pharmacokinetics'))
            black_box_details = clean_text(request.form.get('black_box_details')) if 'black_box_warning' in request.form else None
            
            # Pregnancy safety for each trimester
            pregnancy_safety_trimester1_id = request.form.get('pregnancy_safety_trimester1_id', type=int)
            pregnancy_details_trimester1 = None
            if pregnancy_safety_trimester1_id:
                pregnancy_safety = SafetyCategory.query.get(pregnancy_safety_trimester1_id)
                if pregnancy_safety and pregnancy_safety.name in ['Caution', 'Contraindicated']:
                    pregnancy_details_trimester1 = clean_text(request.form.get('pregnancy_details_trimester1'))
            
            pregnancy_safety_trimester2_id = request.form.get('pregnancy_safety_trimester2_id', type=int)
            pregnancy_details_trimester2 = None
            if pregnancy_safety_trimester2_id:
                pregnancy_safety = SafetyCategory.query.get(pregnancy_safety_trimester2_id)
                if pregnancy_safety and pregnancy_safety.name in ['Caution', 'Contraindicated']:
                    pregnancy_details_trimester2 = clean_text(request.form.get('pregnancy_details_trimester2'))
            
            pregnancy_safety_trimester3_id = request.form.get('pregnancy_safety_trimester3_id', type=int)
            pregnancy_details_trimester3 = None
            if pregnancy_safety_trimester3_id:
                pregnancy_safety = SafetyCategory.query.get(pregnancy_safety_trimester3_id)
                if pregnancy_safety and pregnancy_safety.name in ['Caution', 'Contraindicated']:
                    pregnancy_details_trimester3 = clean_text(request.form.get('pregnancy_details_trimester3'))
            
            lactation_safety_id = request.form.get('lactation_safety_id', type=int)
            lactation_details = None
            if lactation_safety_id:
                lactation_safety = SafetyCategory.query.get(lactation_safety_id)
                if lactation_safety and lactation_safety.name in ['Caution', 'Contraindicated']:
                    lactation_details = clean_text(request.form.get('lactation_details'))
            
            smiles = request.form.get('smiles')
            logger.debug(f"SMILES received: {smiles}")
            logger.debug(f"drug_id={drug_id}, salt_id={salt_id} (type: {type(salt_id)})")
            
            selected_routes = request.form.getlist('route_id[]')
            selected_side_effects = request.form.getlist('side_effects[]')
            
            fda_approved = 'fda_approved' in request.form
            ema_approved = 'ema_approved' in request.form
            titck_approved = 'titck_approved' in request.form
            
            molecular_formula = request.form.get('molecular_formula')
            
            # Retrieve metabolism-related IDs from form
            metabolism_organs_ids = request.form.getlist('metabolism_organs[]')
            metabolism_enzymes_ids = request.form.getlist('metabolism_enzymes[]')
            metabolite_ids = request.form.getlist('metabolites[]')
            
            # Validate metabolism-related IDs
            valid_metabolism_organs_ids = []
            valid_metabolism_enzymes_ids = []
            valid_metabolite_ids = []
            
            for organ_id in metabolism_organs_ids:
                try:
                    organ_id = int(organ_id)
                    if MetabolismOrgan.query.get(organ_id):
                        valid_metabolism_organs_ids.append(organ_id)
                except ValueError:
                    logger.warning(f"Invalid metabolism organ ID: {organ_id}")
            
            for enzyme_id in metabolism_enzymes_ids:
                try:
                    enzyme_id = int(enzyme_id)
                    if MetabolismEnzyme.query.get(enzyme_id):
                        valid_metabolism_enzymes_ids.append(enzyme_id)
                except ValueError:
                    logger.warning(f"Invalid metabolism enzyme ID: {enzyme_id}")
            
            for metabolite_id in metabolite_ids:
                try:
                    metabolite_id = int(metabolite_id)
                    if Metabolite.query.get(metabolite_id):
                        valid_metabolite_ids.append(metabolite_id)
                except ValueError:
                    logger.warning(f"Invalid metabolite ID: {metabolite_id}")
            
            logger.debug(f"Valid metabolism organs IDs: {valid_metabolism_organs_ids}")
            logger.debug(f"Valid metabolism enzymes IDs: {valid_metabolism_enzymes_ids}")
            logger.debug(f"Valid metabolite IDs: {valid_metabolite_ids}")
            
            # Handle uploaded structure file (priority)
            structure_filename = None
            structure = request.files.get('structure')
            if structure:
                original_filename = secure_filename(structure.filename)
                file_ext = os.path.splitext(original_filename)[1] or '.png'
                structure_filename = f"structure_{drug_id}_salt_{salt_id or 'none'}{file_ext}"
                structure_path = os.path.join('static', 'Uploads', structure_filename).replace('\\', '/')
                os.makedirs(os.path.dirname(structure_path), exist_ok=True)
                structure.save(structure_path)
                structure_filename = os.path.join('Uploads', structure_filename).replace('\\', '/')
                logger.info(f"Uploaded structure file saved as {structure_path}")
            elif smiles:
                svg_filename = f"sketcher_{drug_id}_salt_{salt_id or 'none'}.svg"
                svg_path = os.path.join('static', 'Uploads', svg_filename).replace('\\', '/')
                logger.debug(f"Generating SVG for drug_id={drug_id}, salt_id={salt_id or 'none'}, path={svg_path}")
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        os.makedirs(os.path.dirname(svg_path), exist_ok=True)
                        Draw.MolToFile(mol, svg_path, size=(300, 300))
                        structure_filename = os.path.join('Uploads', svg_filename).replace('\\', '/')
                        logger.info(f"Structure file saved as {svg_path}")
                    else:
                        logger.error(f"Invalid SMILES: {smiles}")
                except Exception as e:
                    logger.error(f"Error generating SVG: {str(e)}")
            else:
                logger.warning("No structure file uploaded or SMILES provided")
            
            # Generate 3D PDB
            structure_3d_filename = None
            if smiles:
                pdb_filename = f"drug_{drug_id}_salt_{salt_id or 'none'}_3d.pdb"
                structure_3d_path, error = generate_3d_structure(smiles, pdb_filename)
                if error:
                    logger.error(f"3D structure generation failed: {error}")
                else:
                    structure_3d_filename = structure_3d_path
                    logger.info(f"3D structure saved as {structure_3d_filename}")
            else:
                logger.debug("No SMILES provided, skipping 3D structure generation")
            
            # DrugDetail fields with units
            molecular_weight = request.form.get('molecular_weight', type=float)
            molecular_weight_unit_id = request.form.get('molecular_weight_unit_id', type=int)
            boiling_point = request.form.get('boiling_point', type=float)
            boiling_point_unit_id = request.form.get('boiling_point_unit_id', type=int)
            melting_point = request.form.get('melting_point', type=float)
            melting_point_unit_id = request.form.get('melting_point_unit_id', type=int)
            density = request.form.get('density', type=float)
            density_unit_id = request.form.get('density_unit_id', type=int)
            solubility = request.form.get('solubility', type=float)
            solubility_unit_id = request.form.get('solubility_unit_id', type=int)
            flash_point = request.form.get('flash_point', type=float)
            flash_point_unit_id = request.form.get('flash_point_unit_id', type=int)
            half_life = request.form.get('half_life', type=float)
            half_life_unit_id = request.form.get('half_life_unit_id', type=int)
            clearance_rate = request.form.get('clearance_rate', type=float)
            clearance_rate_unit_id = request.form.get('clearance_rate_unit_id', type=int)
            bioavailability = request.form.get('bioavailability', type=float)
            
            # Remaining form fields
            iupac_name = request.form.get('iupac_name')
            inchikey = request.form.get('inchikey')
            pubchem_cid = request.form.get('pubchem_cid')
            pubchem_sid = request.form.get('pubchem_sid')
            cas_id = request.form.get('cas_id')
            ec_number = request.form.get('ec_number')
            nci_code = request.form.get('nci_code')
            rxcui = request.form.get('rxcui')
            snomed_id = request.form.get('snomed_id')
            black_box_warning = 'black_box_warning' in request.form
            indications = ','.join(request.form.getlist('indications[]')) if request.form.getlist('indications[]') else None
            
            selected_target_molecules = request.form.getlist('target_molecules[]')
            target_molecules = ','.join(selected_target_molecules) if selected_target_molecules else None
            
            # Create new DrugDetail
            new_detail = DrugDetail(
                drug_id=drug_id,
                salt_id=salt_id,
                mechanism_of_action=mechanism_of_action,
                molecular_formula=molecular_formula,
                synthesis=synthesis,
                structure=structure_filename,
                structure_3d=structure_3d_filename,
                iupac_name=iupac_name,
                smiles=smiles,
                inchikey=inchikey,
                pubchem_cid=pubchem_cid,
                pubchem_sid=pubchem_sid,
                cas_id=cas_id,
                ec_number=ec_number,
                nci_code=nci_code,
                rxcui=rxcui,
                snomed_id=snomed_id,
                molecular_weight=molecular_weight,
                molecular_weight_unit_id=molecular_weight_unit_id,
                solubility=solubility,
                solubility_unit_id=solubility_unit_id,
                pharmacodynamics=pharmacodynamics,
                black_box_warning=black_box_warning,
                black_box_details=black_box_details,
                indications=indications,
                target_molecules=target_molecules,
                pharmacokinetics=pharmacokinetics,
                boiling_point=boiling_point,
                boiling_point_unit_id=boiling_point_unit_id,
                melting_point=melting_point,
                melting_point_unit_id=melting_point_unit_id,
                density=density,
                density_unit_id=density_unit_id,
                flash_point=flash_point,
                flash_point_unit_id=flash_point_unit_id,
                fda_approved=fda_approved,
                ema_approved=ema_approved,
                titck_approved=titck_approved,
                half_life=half_life,
                half_life_unit_id=half_life_unit_id,
                clearance_rate=clearance_rate,
                clearance_rate_unit_id=clearance_rate_unit_id,
                bioavailability=bioavailability,
                references=references,
                pregnancy_safety_trimester1_id=pregnancy_safety_trimester1_id,
                pregnancy_details_trimester1=pregnancy_details_trimester1,
                pregnancy_safety_trimester2_id=pregnancy_safety_trimester2_id,
                pregnancy_details_trimester2=pregnancy_details_trimester2,
                pregnancy_safety_trimester3_id=pregnancy_safety_trimester3_id,
                pregnancy_details_trimester3=pregnancy_details_trimester3,
                lactation_safety_id=lactation_safety_id,
                lactation_details=lactation_details
            )
            db.session.add(new_detail)
            db.session.flush()  # Ensure new_detail.id is available
            
            # Update drug categories
            if valid_category_ids:
                drug.categories.clear()  # Clear existing categories
                selected_categories = DrugCategory.query.filter(DrugCategory.id.in_(valid_category_ids)).all()
                drug.categories.extend(selected_categories)
                logger.info(f"Updated categories for drug_id={drug_id}: {valid_category_ids}")
            else:
                logger.debug(f"No categories selected for drug_id={drug_id}")
            
            # NEW: Add ATC code mappings
            if valid_atc_ids:
                for atc_id in valid_atc_ids:
                    # Check if mapping already exists
                    existing_mapping = DrugATCMapping.query.filter_by(
                        drug_id=drug.id,
                        atc_level5_id=atc_id
                    ).first()
                    
                    if not existing_mapping:
                        new_mapping = DrugATCMapping(
                            drug_id=drug.id,
                            atc_level5_id=atc_id
                        )
                        db.session.add(new_mapping)
                logger.info(f"Added {len(valid_atc_ids)} ATC code mappings for drug_id={drug_id}")
            
            # Process DrugRoute entries
            if not selected_routes:
                logger.warning("No routes selected")
            
            for route_id in selected_routes:
                logger.debug(f"Processing route_id={route_id}")
                pd = request.form.get(f'route_pharmacodynamics_{route_id}', '')
                pk = request.form.get(f'route_pharmacokinetics_{route_id}', '')
                
                # PK parameters with units
                absorption_rate = request.form.get(f'absorption_rate_{route_id}', '')
                absorption_rate_unit_id = request.form.get(f'absorption_rate_unit_id_{route_id}', type=int)
                vod_rate = request.form.get(f'vod_rate_{route_id}', '')
                vod_rate_unit_id = request.form.get(f'vod_rate_unit_id_{route_id}', type=int)
                protein_binding = request.form.get(f'protein_binding_{route_id}', '')
                half_life = request.form.get(f'half_life_{route_id}', '')
                half_life_unit_id = request.form.get(f'half_life_unit_id_{route_id}', type=int)
                clearance_rate = request.form.get(f'clearance_rate_{route_id}', '')
                clearance_rate_unit_id = request.form.get(f'clearance_rate_unit_id_{route_id}', type=int)
                bioavailability = request.form.get(f'bioavailability_{route_id}', '')
                tmax = request.form.get(f'tmax_{route_id}', '')
                tmax_unit_id = request.form.get(f'tmax_unit_id_{route_id}', type=int)
                cmax = request.form.get(f'cmax_{route_id}', '')
                cmax_unit_id = request.form.get(f'cmax_unit_id_{route_id}', type=int)
                therapeutic_range = request.form.get(f'therapeutic_range_{route_id}', '')
                therapeutic_unit_id = request.form.get(f'therapeutic_unit_id_{route_id}', type=int)
                
                # Parse PK ranges
                def parse_range(value):
                    if not value:
                        return None, None
                    value = value.replace('–', '-')
                    if '-' in value:
                        try:
                            min_val, max_val = map(float, value.split('-'))
                            return min_val, max_val
                        except ValueError:
                            logger.error(f"Failed to parse range '{value}'")
                            return None, None
                    try:
                        val = float(value)
                        return val, val
                    except ValueError:
                        logger.error(f"Failed to parse single value '{value}'")
                        return None, None
                
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
                therapeutic_min, therapeutic_max = parse_range(therapeutic_range)
                
                logger.debug(f"Route ID {route_id} -> "
                             f"Absorption: {absorption_min}-{absorption_max}, Unit ID: {absorption_rate_unit_id}, "
                             f"VoD: {vod_min}-{vod_max}, Unit ID: {vod_rate_unit_id}, "
                             f"Protein Binding: {protein_min}-{protein_max}, "
                             f"Half-Life: {half_life_min}-{half_life_max}, Unit ID: {half_life_unit_id}, "
                             f"Clearance: {clearance_min}-{clearance_max}, Unit ID: {clearance_rate_unit_id}, "
                             f"Bioavailability: {bio_min}-{bio_max}, "
                             f"Tmax: {tmax_min}-{tmax_max}, Unit ID: {tmax_unit_id}, "
                             f"Cmax: {cmax_min}-{cmax_max}, Unit ID: {cmax_unit_id}, "
                             f"Therapeutic Range: {therapeutic_min}-{therapeutic_max}, Unit ID: {therapeutic_unit_id}, "
                             f"Organs IDs: {valid_metabolism_organs_ids}, "
                             f"Enzymes IDs: {valid_metabolism_enzymes_ids}, "
                             f"Metabolite IDs: {valid_metabolite_ids}, "
                             f"PD: {pd}, PK: {pk}")
                
                if not RouteOfAdministration.query.get(route_id):
                    logger.error(f"Invalid route_id {route_id}")
                    continue
                
                new_drug_route = DrugRoute(
                    drug_detail_id=new_detail.id,
                    route_id=route_id,
                    pharmacodynamics=pd,
                    pharmacokinetics=pk,
                    absorption_rate_min=absorption_min,
                    absorption_rate_max=absorption_max,
                    absorption_rate_unit_id=absorption_rate_unit_id,
                    vod_rate_min=vod_min,
                    vod_rate_max=vod_max,
                    vod_rate_unit_id=vod_rate_unit_id,
                    protein_binding_min=protein_min,
                    protein_binding_max=protein_max,
                    half_life_min=half_life_min,
                    half_life_max=half_life_max,
                    half_life_unit_id=half_life_unit_id,
                    clearance_rate_min=clearance_min,
                    clearance_rate_max=clearance_max,
                    clearance_rate_unit_id=clearance_rate_unit_id,
                    bioavailability_min=bio_min,
                    bioavailability_max=bio_max,
                    tmax_min=tmax_min,
                    tmax_max=tmax_max,
                    tmax_unit_id=tmax_unit_id,
                    cmax_min=cmax_min,
                    cmax_max=cmax_max,
                    cmax_unit_id=cmax_unit_id,
                    therapeutic_min=therapeutic_min,
                    therapeutic_max=therapeutic_max,
                    therapeutic_unit_id=therapeutic_unit_id
                )
                db.session.add(new_drug_route)
                db.session.flush()
                
                # Link metabolism data via relationships
                if valid_metabolism_organs_ids:
                    organs = MetabolismOrgan.query.filter(MetabolismOrgan.id.in_([int(id) for id in valid_metabolism_organs_ids])).all()
                    new_drug_route.metabolism_organs.extend(organs)
                
                if valid_metabolism_enzymes_ids:
                    enzymes = MetabolismEnzyme.query.filter(MetabolismEnzyme.id.in_([int(id) for id in valid_metabolism_enzymes_ids])).all()
                    new_drug_route.metabolism_enzymes.extend(enzymes)
                
                if valid_metabolite_ids:
                    metabolites = Metabolite.query.filter(Metabolite.id.in_([int(id) for id in valid_metabolite_ids])).all()
                    new_drug_route.metabolites.extend(metabolites)
                
                # Route-specific indications
                selected_route_indications = request.form.getlist(f'route_indications_{route_id}[]')
                for indication_id in selected_route_indications:
                    new_route_indication = RouteIndication(
                        drug_detail_id=new_detail.id,
                        route_id=route_id,
                        indication_id=indication_id
                    )
                    db.session.add(new_route_indication)
            
            # Add side effects
            for side_effect_id in selected_side_effects:
                side_effect = SideEffect.query.get(side_effect_id)
                if side_effect:
                    new_detail.side_effects.append(side_effect)
            
            db.session.commit()
            logger.info("All records saved successfully")
            
            # Create a News entry for the new DrugDetail
            drug_name = drug.name_en if hasattr(drug, 'name_en') and drug.name_en else f"Drug ID {drug_id}"
            news = News(
                category='Drug Update',
                title=f'New Drug Details Added: {drug_name}',
                description=f'Detailed information for <a href="/drug/{drug_id}">{drug_name}</a> has been added to Drugly.',
                publication_date=datetime.utcnow()
            )
            db.session.add(news)
            db.session.commit()
            logger.info(f"News entry created for drug_id={drug_id}: {drug_name}")
            
            flash('Drug details added and announced on the homepage!', 'success')
            return redirect(url_for('view_details'))
        
        except Exception as e:
            db.session.rollback()
            logger.error(f"Exception occurred: {str(e)}")
            return render_template(
                'add_detail.html', drugs=drugs, salts=salts, indications=indications,
                targets=targets, routes=routes, side_effects=side_effects, metabolites=metabolites,
                metabolism_organs=metabolism_organs, metabolism_enzymes=metabolism_enzymes,
                units=units, units_json=units_json, safety_categories=safety_categories, categories=categories,
                atc_level5_list=atc_level5_list,  # NEW
                error_message=f"Error saving details: {str(e)}"
            )
    
    return render_template(
        'add_detail.html', drugs=drugs, salts=salts, indications=indications,
        targets=targets, routes=routes, side_effects=side_effects, metabolites=metabolites,
        metabolism_organs=metabolism_organs, metabolism_enzymes=metabolism_enzymes,
        units=units, units_json=units_json, safety_categories=safety_categories, categories=categories,
        atc_level5_list=atc_level5_list  # NEW
    )

def generate_and_update_structures():
    with app.app_context():
        details = DrugDetail.query.all()
        for detail in details:
            # Define consistent paths
            svg_filename = f'sketcher_{detail.drug_id}_salt_{detail.salt_id or "none"}.svg'
            pdb_filename = f'drug_{detail.drug_id}_salt_{detail.salt_id or "none"}_3d.pdb'
            svg_path = os.path.join('static', 'uploads', svg_filename)
            pdb_path = os.path.join('static', 'uploads', '3d_structures', pdb_filename)
            db_svg_path = f'uploads/{svg_filename}'
            db_pdb_path = f'uploads/3d_structures/{pdb_filename}'

            # Generate 2D SVG if SMILES exists and structure is missing or file doesn't exist
            if detail.smiles and (not detail.structure or not os.path.exists(svg_path)):
                try:
                    mol = Chem.MolFromSmiles(detail.smiles)
                    if mol:
                        os.makedirs(os.path.dirname(svg_path), exist_ok=True)
                        Draw.MolToFile(mol, svg_path, size=(300, 300))
                        detail.structure = db_svg_path
                        db.session.commit()
                        print(f"Generated SVG for drug_id={detail.drug_id}, salt_id={detail.salt_id or 'none'}")
                    else:
                        print(f"Invalid SMILES for drug_id={detail.drug_id}: {detail.smiles}")
                except Exception as e:
                    print(f"Error generating SVG for drug_id={detail.drug_id}: {e}")

            # Generate 3D PDB if SMILES exists and structure_3d is missing or file doesn't exist
            if detail.smiles and (not detail.structure_3d or not os.path.exists(pdb_path)):
                try:
                    structure_3d_path, error = generate_3d_structure(detail.smiles, pdb_filename)
                    if not error:
                        detail.structure_3d = db_pdb_path
                        db.session.commit()
                        print(f"Generated PDB for drug_id={detail.drug_id}, salt_id={detail.salt_id or 'none'}")
                    else:
                        print(f"Error generating PDB for drug_id={detail.drug_id}: {error}")
                except Exception as e:
                    print(f"Error generating PDB for drug_id={detail.drug_id}: {e}")

            # Normalize existing paths (fix casing and prefixes)
            if detail.structure and detail.structure.lower().startswith('uploads/'):
                detail.structure = detail.structure.replace('uploads/', 'Uploads/')
                db.session.commit()
            if detail.structure_3d and detail.structure_3d.lower().startswith('uploads/'):
                detail.structure_3d = detail.structure_3d.replace('uploads/', 'Uploads/')
                db.session.commit()

if __name__ == '__main__':
    generate_and_update_structures()

@app.route('/details', methods=['GET'])
@login_required
def view_details():
    details = DrugDetail.query.all()
    enriched_details = []
    for detail in details:
        # Define paths
        svg_filename = f'sketcher_{detail.drug_id}_salt_{detail.salt_id or "none"}.svg'
        pdb_filename = f'drug_{detail.drug_id}_salt_{detail.salt_id or "none"}_3d.pdb'
        svg_path = os.path.join('static', 'Uploads', svg_filename)
        pdb_path = os.path.join('static', 'Uploads', '3d_structures', pdb_filename)
        db_svg_path = f'uploads/{svg_filename}'
        db_pdb_path = f'uploads/3d_structures/{pdb_filename}'
        
        # Generate 2D SVG if missing
        if detail.smiles and (not detail.structure or not os.path.exists(svg_path)):
            try:
                mol = Chem.MolFromSmiles(detail.smiles)
                if mol:
                    os.makedirs(os.path.dirname(svg_path), exist_ok=True)
                    Draw.MolToFile(mol, svg_path, size=(300, 300))
                    detail.structure = db_svg_path
                    db.session.commit()
            except Exception as e:
                print(f"Error generating SVG for drug_id={detail.drug_id}: {e}")
        
        # Generate 3D PDB if missing
        if detail.smiles and (not detail.structure_3d or not os.path.exists(pdb_path)):
            try:
                structure_3d_path, error = generate_3d_structure(detail.smiles, pdb_filename)
                if not error:
                    detail.structure_3d = db_pdb_path
                    db.session.commit()
            except Exception as e:
                print(f"Error generating PDB for drug_id={detail.drug_id}: {e}")
        
        # Prepare routes and other data
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
                'indications': route_indications,
                'absorption_rate': f"{route.absorption_rate_min or ''}-{route.absorption_rate_max or ''}" if route.absorption_rate_min or route.absorption_rate_max else 'N/A',
                'absorption_rate_unit': Unit.query.get(route.absorption_rate_unit_id).name if route.absorption_rate_unit_id else 'N/A',
                'vod_rate': f"{route.vod_rate_min or ''}-{route.vod_rate_max or ''}" if route.vod_rate_min or route.vod_rate_max else 'N/A',
                'vod_rate_unit': Unit.query.get(route.vod_rate_unit_id).name if route.vod_rate_unit_id else 'N/A',
                'protein_binding': f"{(route.protein_binding_min * 100 or '')}-{(route.protein_binding_max * 100 or '')}%" if route.protein_binding_min or route.protein_binding_max else 'N/A',
                'half_life': f"{route.half_life_min or ''}-{route.half_life_max or ''}" if route.half_life_min or route.half_life_max else 'N/A',
                'half_life_unit': Unit.query.get(route.half_life_unit_id).name if route.half_life_unit_id else 'N/A',
                'clearance_rate': f"{route.clearance_rate_min or ''}-{route.clearance_rate_max or ''}" if route.clearance_rate_min or route.clearance_rate_max else 'N/A',
                'clearance_rate_unit': Unit.query.get(route.clearance_rate_unit_id).name if route.clearance_rate_unit_id else 'N/A',
                'bioavailability': f"{(route.bioavailability_min * 100 or '')}-{(route.bioavailability_max * 100 or '')}%" if route.bioavailability_min or route.bioavailability_max else 'N/A',
                'tmax': f"{route.tmax_min or ''}-{route.tmax_max or ''}" if route.tmax_min or route.tmax_max else 'N/A',
                'tmax_unit': Unit.query.get(route.tmax_unit_id).name if route.tmax_unit_id else 'N/A',
                'cmax': f"{route.cmax_min or ''}-{route.cmax_max or ''}" if route.cmax_min or route.cmax_max else 'N/A',
                'cmax_unit': Unit.query.get(route.cmax_unit_id).name if route.cmax_unit_id else 'N/A',
                'therapeutic_range': f"{route.therapeutic_min or ''}-{route.therapeutic_max or ''}" if route.therapeutic_min or route.therapeutic_max else 'N/A',
                'therapeutic_unit': Unit.query.get(route.therapeutic_unit_id).name if route.therapeutic_unit_id else 'N/A'
            })
        
        indications_list = []
        if detail.indications:
            indication_ids = [int(ind_id) for ind_id in detail.indications.split(',') if ind_id.isdigit()]
            indications = Indication.query.filter(Indication.id.in_(indication_ids)).all()
            indications_list = [f"{ind.name_en} ({ind.name_tr})" if ind.name_tr else ind.name_en for ind in indications]
        
        target_molecules_list = []
        if detail.target_molecules:
            target_ids = detail.target_molecules.split(',')
            target_molecules_list = [
                db.session.get(Target, int(target_id)).name_en
                for target_id in target_ids
                if db.session.get(Target, int(target_id))
            ]
        
        side_effects_list = [
            {"id": se.id, "name_en": se.name_en, "name_tr": se.name_tr or 'N/A'}
            for se in detail.side_effects
        ]
        
        # Fetch categories
        categories_list = [
            {"id": cat.id, "name": cat.name}
            for cat in detail.drug.categories
        ]
        
        # NEW: Fetch ATC codes for this drug
        atc_codes_list = []
        drug_atc_mappings = DrugATCMapping.query.filter_by(drug_id=detail.drug_id).all()
        for mapping in drug_atc_mappings:
            atc5 = mapping.atc_level5
            atc4 = atc5.level4_parent
            atc3 = atc4.level3_parent
            atc2 = atc3.level2_parent
            atc1 = atc2.level1_parent
            
            atc_codes_list.append({
                'code': atc5.code,
                'name': atc5.name,
                'ddd': atc5.ddd or 'N/A',
                'uom': atc5.uom or 'N/A',
                'adm_r': atc5.adm_r or 'N/A',
                'hierarchy': f"{atc1.code} > {atc2.code} > {atc3.code} > {atc4.code} > {atc5.code}",
                'full_path': f"{atc1.name} / {atc2.name} / {atc3.name} / {atc4.name} / {atc5.name}"
            })
        
        # Fetch pregnancy (trimester-specific) and lactation safety info
        pregnancy_safety_t1 = detail.pregnancy_safety_trimester1.name if detail.pregnancy_safety_trimester1 else 'N/A'
        pregnancy_safety_t2 = detail.pregnancy_safety_trimester2.name if detail.pregnancy_safety_trimester2 else 'N/A'
        pregnancy_safety_t3 = detail.pregnancy_safety_trimester3.name if detail.pregnancy_safety_trimester3 else 'N/A'
        lactation_safety = detail.lactation_safety.name if detail.lactation_safety else 'N/A'
        
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
            'molecular_weight_unit': Unit.query.get(detail.molecular_weight_unit_id).name if detail.molecular_weight_unit_id else 'N/A',
            'solubility': detail.solubility,
            'solubility_unit': Unit.query.get(detail.solubility_unit_id).name if detail.solubility_unit_id else 'N/A',
            'side_effects': side_effects_list,
            'pharmacodynamics': detail.pharmacodynamics,
            'indications': indications_list,
            'target_molecules': target_molecules_list,
            'pharmacokinetics': detail.pharmacokinetics,
            'boiling_point': detail.boiling_point,
            'boiling_point_unit': Unit.query.get(detail.boiling_point_unit_id).name if detail.boiling_point_unit_id else 'N/A',
            'melting_point': detail.melting_point,
            'melting_point_unit': Unit.query.get(detail.melting_point_unit_id).name if detail.melting_point_unit_id else 'N/A',
            'density': detail.density,
            'density_unit': Unit.query.get(detail.density_unit_id).name if detail.density_unit_id else 'N/A',
            'flash_point': detail.flash_point,
            'flash_point_unit': Unit.query.get(detail.flash_point_unit_id).name if detail.flash_point_unit_id else 'N/A',
            'routes': routes_info,
            'black_box_warning': detail.black_box_warning,
            'black_box_details': detail.black_box_details,
            'mechanism_of_action': detail.mechanism_of_action,
            'references': detail.references,
            'pregnancy_safety_trimester1': pregnancy_safety_t1,
            'pregnancy_details_trimester1': detail.pregnancy_details_trimester1 or 'N/A',
            'pregnancy_safety_trimester2': pregnancy_safety_t2,
            'pregnancy_details_trimester2': detail.pregnancy_details_trimester2 or 'N/A',
            'pregnancy_safety_trimester3': pregnancy_safety_t3,
            'pregnancy_details_trimester3': detail.pregnancy_details_trimester3 or 'N/A',
            'lactation_safety': lactation_safety,
            'lactation_details': detail.lactation_details or 'N/A',
            'categories': categories_list,
            'atc_codes': atc_codes_list  # NEW: ATC codes
        })
    
    return render_template('details_list.html', details=enriched_details)



@app.route('/details', methods=['GET'])
def manage_details():
    details = DrugDetail.query.all()
    return render_template('details_list.html', details=details)


#ICD-11 Başlangıç
# Helper Functions
def clean_title(title):
    return title.strip().lstrip('- ').strip() if isinstance(title, str) else ''

def find_column(df, possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    return None

def import_icd11_mms_from_file(file_path):
    file_ext = os.path.splitext(file_path)[1].lower()
    chunk_size = 500  # Increased for faster processing
    batch_size = 250  # Increased to reduce commits
    if file_ext == '.xlsx':
        df = pd.read_excel(file_path, dtype=str)
    elif file_ext in ['.txt', '.tsv']:
        df = pd.read_csv(file_path, sep='\t', dtype=str, encoding='utf-8')
    else:
        raise ValueError(f"Unsupported file type: {file_ext}")
    
    process = psutil.Process()
    app.logger.info(f"Initial memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    
    df = df.fillna('')
    total_rows = len(df)

    # Column mapping
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

    # Dictionaries to track entities
    uri_to_id = {}
    code_to_id = {}
    block_to_id = {}
    chapter_to_id = {}
    added_count = {'chapter': 0, 'block': 0, 'category': 0, 'other': 0}
    skipped_rows = []
    batch_size = 500

    # Separate entries into chapters, blocks, and categories
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

    print(f"Processing {len(chapters)} chapters, {len(blocks)} blocks, {len(categories)} categories")

    # Process chapters first
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

    # Process blocks in hierarchical order
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

    print(f"block_to_id after processing blocks: {block_to_id}")

    # Process categories in hierarchical order
    category_depths = {}
    for index, row in categories:
        depth = int(row[found_columns['DepthInKind']])
        if depth not in category_depths:
            category_depths[depth] = []
        category_depths[depth].append((index, row))

    for depth in sorted(category_depths.keys()):
        # Sort categories within each depth by code to ensure base categories (e.g., KA45) are processed before subcategories (e.g., KA45.0)
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

            parent_id = get_parent_id(row, depth, chapter_no, code_to_id, block_to_id, chapter_to_id, found_columns)

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

    # Validate hierarchy
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
        return None

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

    if class_kind == 'category':
        if code and '.' in code:
            code_parts = code.split('.')
            for i in range(len(code_parts) - 1, 0, -1):
                parent_code = '.'.join(code_parts[:i])
                if parent_code in code_to_id:
                    parent = Indication.query.get(code_to_id[parent_code])
                    if parent.class_kind == 'category':
                        return code_to_id[parent_code]

        deepest_block_id = None
        for i in range(1, 6):
            grouping_key = f'Grouping{i}'
            if grouping_key in found_columns and row[found_columns[grouping_key]]:
                parent_block = row[found_columns[grouping_key]].strip()
                if parent_block in block_to_id:
                    deepest_block_id = block_to_id[parent_block]

        if deepest_block_id:
            return deepest_block_id

        if chapter_no in chapter_to_id:
            return chapter_to_id[chapter_no]

    return None

# RAM Monitoring Endpoint
@app.route('/monitor/ram', methods=['GET'])
def monitor_ram():
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        return jsonify({
            'rss': memory_info.rss,  # Resident Set Size in bytes
            'vms': memory_info.vms   # Virtual Memory Size in bytes
        })
    except Exception as e:
        app.logger.error(f"Error monitoring RAM: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Routes
@app.route('/indications', methods=['GET', 'POST'])
def manage_indications():
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                flash("Dosya seçilmedi!", "error")
                return redirect(url_for('manage_indications'))

            if '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in ['txt', 'tsv', 'xlsx']:
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
                file.save(file_path)
                try:
                    added_count, skipped_rows = import_icd11_mms_from_file(file_path)
                    flash(f"{added_count} endikasyon başarıyla içe aktarıldı!", "success")
                    if skipped_rows:
                        flash(f"{len(skipped_rows)} satır atlandı. Ayrıntılar için logları kontrol edin.", "warning")
                except Exception as e:
                    flash(f"ICD-11 verisi içe aktarılırken hata: {e}", "error")
                    app.logger.error(f"Import error: {e}")
                finally:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                return redirect(url_for('manage_indications'))
            else:
                flash("Geçersiz dosya türü! Lütfen .txt, .tsv veya .xlsx dosyası yükleyin.", "error")
                return redirect(url_for('manage_indications'))
        else:
            # Handle new indication form submission
            name_en = request.form.get('name_en')
            name_tr = request.form.get('name_tr')
            description = request.form.get('description')
            synonyms = request.form.get('synonyms')
            code = request.form.get('code')
            try:
                new_indication = Indication(
                    name_en=name_en,
                    name_tr=name_tr,
                    description=description,
                    synonyms=synonyms,
                    code=code,
                    class_kind='category',  # Default for manual additions
                    depth=1,
                    is_leaf=True
                )
                db.session.add(new_indication)
                db.session.commit()
                flash("Endikasyon başarıyla eklendi!", "success")
            except Exception as e:
                db.session.rollback()
                flash(f"Endikasyon eklenirken hata: {e}", "error")
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
            app.logger.debug(f"Filtering by class_kind: {expected_class_kind}")
        
        # Log all children before pagination
        all_children = children_query.all()
        app.logger.debug(f"Total children found: {len(all_children)}")
        for child in all_children:
            app.logger.debug(f"Child: ID={child.id}, Name={child.name_en}, Code={child.code}, ClassKind={child.class_kind}, HasChildren={child.has_children}, IsLeaf={child.is_leaf}")

        # Sort and paginate
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

        app.logger.debug(f"Paginated children count: {len(children)} on page {page}")
        response = [{
            'id': ind.id,
            'name_en': str(ind.name_en),
            'chapter_no': str(ind.chapter_no) if ind.chapter_no else '',
            'class_kind': str(ind.class_kind),
            'code': str(ind.code) if ind.code else '',
            'has_children': ind.has_children,
            'is_leaf': ind.is_leaf
        } for ind in children]
        app.logger.debug(f"Response: {response}")
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


#SEARCH ROUTE
@app.route('/search', methods=['GET', 'POST'])
def search():
    # Check if user is logged in
    user_id = session.get('user_id')
    if not user_id:
        flash("Please log in to access the search page.", "danger")
        return redirect(url_for('login'))
    
    # Fetch user from database
    user = User.query.get(user_id)
    user_email = user.email if user else None
    
    query = request.form.get('query', '').strip()
    drugs = set()
    diseases = []
    salts = []
    target_molecules = []
    side_effects = []
    categories = []
    atc_results = []
    
    if query:
        search_pattern = f'%{query}%'
        
        # Drugs - Search both English and Turkish names
        drugs.update(Drug.query.filter(
            db.or_(
                Drug.name_en.ilike(search_pattern),
                Drug.name_tr.ilike(search_pattern)
            )
        ).limit(50).all())
        
        # Salts
        salts = Salt.query.filter(
            db.or_(
                Salt.name_en.ilike(search_pattern),
                Salt.name_tr.ilike(search_pattern)
            )
        ).limit(50).all()
        
        # Indications/Diseases
        indications = Indication.query.filter(
            db.or_(
                Indication.name_en.ilike(search_pattern),
                Indication.name_tr.ilike(search_pattern)
            )
        ).limit(50).all()
        
        for indication in indications:
            related_details = DrugDetail.query.filter(
                DrugDetail.indications.contains(str(indication.id))
            ).limit(20).all()
            
            related_drugs_list = [detail.drug for detail in related_details if detail.drug]
            for drug in related_drugs_list:
                drugs.add(drug)
            
            diseases.append({
                'indication': indication,
                'related_drugs': related_drugs_list
            })
        
        # Target Molecules
        target_molecules = Target.query.filter(
            Target.name_en.ilike(search_pattern)
        ).limit(50).all()
        
        for target in target_molecules:
            related_details = DrugDetail.query.filter(
                DrugDetail.target_molecules.contains(str(target.id))
            ).limit(20).all()
            for detail in related_details:
                if detail.drug:
                    drugs.add(detail.drug)
        
        # Side Effects
        side_effects = SideEffect.query.filter(
            db.or_(
                SideEffect.name_en.ilike(search_pattern),
                SideEffect.name_tr.ilike(search_pattern)
            )
        ).limit(50).all()
        
        for side_effect in side_effects:
            related_details = side_effect.details.limit(20).all()
            for detail in related_details:
                if detail.drug:
                    drugs.add(detail.drug)
        
        # Categories
        matching_categories = DrugCategory.query.filter(
            DrugCategory.name.ilike(search_pattern)
        ).limit(50).all()
        
        for category in matching_categories:
            related_drugs = category.drugs.limit(20).all()
            
            categories.append({
                'category': category,
                'related_drugs': related_drugs
            })
            drugs.update(related_drugs)
        
        # ATC Search - All levels
        atc1_results = ATCLevel1.query.filter(
            db.or_(
                ATCLevel1.code.ilike(search_pattern),
                ATCLevel1.name.ilike(search_pattern)
            )
        ).limit(10).all()
        
        atc2_results = ATCLevel2.query.filter(
            db.or_(
                ATCLevel2.code.ilike(search_pattern),
                ATCLevel2.name.ilike(search_pattern)
            )
        ).limit(10).all()
        
        atc3_results = ATCLevel3.query.filter(
            db.or_(
                ATCLevel3.code.ilike(search_pattern),
                ATCLevel3.name.ilike(search_pattern)
            )
        ).limit(10).all()
        
        atc4_results = ATCLevel4.query.filter(
            db.or_(
                ATCLevel4.code.ilike(search_pattern),
                ATCLevel4.name.ilike(search_pattern)
            )
        ).limit(10).all()
        
        atc5_results = ATCLevel5.query.filter(
            db.or_(
                ATCLevel5.code.ilike(search_pattern),
                ATCLevel5.name.ilike(search_pattern)
            )
        ).limit(10).all()
        
        # Collect all ATC results
        for atc in atc1_results:
            atc_results.append({'level': 1, 'code': atc.code, 'name': atc.name, 'atc': atc})
        for atc in atc2_results:
            atc_results.append({'level': 2, 'code': atc.code, 'name': atc.name, 'atc': atc})
        for atc in atc3_results:
            atc_results.append({'level': 3, 'code': atc.code, 'name': atc.name, 'atc': atc})
        for atc in atc4_results:
            atc_results.append({'level': 4, 'code': atc.code, 'name': atc.name, 'atc': atc})
        for atc in atc5_results:
            related_drugs_via_atc = [mapping.drug for mapping in atc.drug_mappings if mapping.drug]
            drugs.update(related_drugs_via_atc)
            atc_results.append({'level': 5, 'code': atc.code, 'name': atc.name, 'atc': atc, 'related_drugs': related_drugs_via_atc})
    
    # Sort drugs alphabetically
    drugs_list = sorted(list(drugs), key=lambda x: x.name_en or x.name_tr or '')
    
    return render_template(
        'search.html',
        query=query,
        drugs=drugs_list,
        diseases=diseases,
        salts=salts,
        target_molecules=target_molecules,
        side_effects=side_effects,
        categories=categories,
        atc_results=atc_results,
        user=user,
        user_email=user_email
    )

@app.route('/search_suggestions', methods=['POST'])
def search_suggestions():
    # Check if user is logged in
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Unauthorized'}), 401
    query = request.json.get('query', '').strip()
    suggestions = {
        'drugs': [],
        'salts': [],
        'diseases': [],
        'target_molecules': [],
        'side_effects': [],
        'categories': [],
        'atc': []
    }
    if query:
        # Drugs
        drugs = Drug.query.filter(
            (Drug.name_en.ilike(f'%{query}%')) | (Drug.name_tr.ilike(f'%{query}%'))
        ).limit(5).all()
        suggestions['drugs'] = [{'id': d.id, 'name': d.name_en or d.name_tr} for d in drugs]
        # Salts
        salts = Salt.query.filter(
            (Salt.name_en.ilike(f'%{query}%')) | (Salt.name_tr.ilike(f'%{query}%'))
        ).limit(5).all()
        suggestions['salts'] = [{'id': s.id, 'name': s.name_en or s.name_tr} for s in salts]
        # Diseases (Indications)
        indications = Indication.query.filter(
            (Indication.name_en.ilike(f'%{query}%')) | (Indication.name_tr.ilike(f'%{query}%'))
        ).limit(5).all()
        suggestions['diseases'] = [{'indication': {'id': i.id, 'name_en': i.name_en or i.name_tr}} for i in indications]
        # Target Molecules
        targets = Target.query.filter(
            Target.name_en.ilike(f'%{query}%')
        ).limit(5).all()
        suggestions['target_molecules'] = [{'id': t.id, 'name': t.name_en} for t in targets]
        # Side Effects
        side_effects = SideEffect.query.filter(
            (SideEffect.name_en.ilike(f'%{query}%')) | (SideEffect.name_tr.ilike(f'%{query}%'))
        ).limit(5).all()
        suggestions['side_effects'] = [{'id': s.id, 'name': s.name_en or s.name_tr} for s in side_effects]
        # Categories
        categories = DrugCategory.query.filter(
            DrugCategory.name.ilike(f'%{query}%')
        ).limit(5).all()
        suggestions['categories'] = [{'category': {'id': c.id, 'name': c.name}} for c in categories]
        
        # ATC Codes - All levels
        atc_suggestions = []
        
        atc5 = ATCLevel5.query.filter(
            db.or_(
                ATCLevel5.code.ilike(f'%{query}%'),
                ATCLevel5.name.ilike(f'%{query}%')
            )
        ).limit(3).all()
        atc_suggestions.extend([{'id': a.id, 'code': a.code, 'name': a.name, 'level': 5} for a in atc5])
        
        atc4 = ATCLevel4.query.filter(
            db.or_(
                ATCLevel4.code.ilike(f'%{query}%'),
                ATCLevel4.name.ilike(f'%{query}%')
            )
        ).limit(2).all()
        atc_suggestions.extend([{'id': a.id, 'code': a.code, 'name': a.name, 'level': 4} for a in atc4])
        
        atc3 = ATCLevel3.query.filter(
            db.or_(
                ATCLevel3.code.ilike(f'%{query}%'),
                ATCLevel3.name.ilike(f'%{query}%')
            )
        ).limit(2).all()
        atc_suggestions.extend([{'id': a.id, 'code': a.code, 'name': a.name, 'level': 3} for a in atc3])
        
        atc2 = ATCLevel2.query.filter(
            db.or_(
                ATCLevel2.code.ilike(f'%{query}%'),
                ATCLevel2.name.ilike(f'%{query}%')
            )
        ).limit(1).all()
        atc_suggestions.extend([{'id': a.id, 'code': a.code, 'name': a.name, 'level': 2} for a in atc2])
        
        atc1 = ATCLevel1.query.filter(
            db.or_(
                ATCLevel1.code.ilike(f'%{query}%'),
                ATCLevel1.name.ilike(f'%{query}%')
            )
        ).limit(1).all()
        atc_suggestions.extend([{'id': a.id, 'code': a.code, 'name': a.name, 'level': 1} for a in atc1])
        
        suggestions['atc'] = atc_suggestions
    return jsonify(suggestions)






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

# New API endpoint to fetch routes for Select2
@app.route('/api/routes', methods=['GET'])
def get_routes():
    try:
        search_term = request.args.get('q', '')
        page = request.args.get('page', 1, type=int)
        limit = request.args.get('limit', 10, type=int)
        offset = (page - 1) * limit

        query = RouteOfAdministration.query
        if search_term:
            query = query.filter(RouteOfAdministration.name.ilike(f"%{search_term}%"))

        routes = query.offset(offset).limit(limit).all()
        total_routes = query.count()

        results = [
            {"id": route.id, "text": route.name}
            for route in routes
        ]

        return jsonify({
            "results": results,
            "pagination": {
                "more": offset + len(routes) < total_routes
            }
        })
    except Exception as e:
        logger.error(f"Error fetching routes: {e}")
        return jsonify({"results": [], "pagination": {"more": False}}), 500
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
@login_required
def drug_detail(drug_id):
    drug = Drug.query.get_or_404(drug_id)
    
    # 🔥 ADD TRACKING HERE 🔥
    try:
        visitor_hash = get_visitor_hash(
            request.headers.get('X-Forwarded-For', request.remote_addr).split(',')[0].strip(),
            request.headers.get('User-Agent', '')
        )
        
        drug_view = DrugView(
            drug_id=drug_id,
            visitor_hash=visitor_hash,
            user_id=session.get('user_id'),
            view_duration=0,  # Will be updated by JS if needed
            timestamp=datetime.utcnow()
        )
        db.session.add(drug_view)
        db.session.commit()
        logger.info(f"Tracked drug view: drug_id={drug_id}, user_id={session.get('user_id')}")
    except Exception as track_error:
        logger.error(f"Error tracking drug view: {str(track_error)}")
        # Don't fail the page if tracking fails
    # 🔥 END TRACKING 🔥

    # NEW: Fetch ATC codes for this drug
    atc_codes_list = []
    drug_atc_mappings = DrugATCMapping.query.filter_by(drug_id=drug_id).all()
    for mapping in drug_atc_mappings:
        atc5 = mapping.atc_level5
        atc4 = atc5.level4_parent
        atc3 = atc4.level3_parent
        atc2 = atc3.level2_parent
        atc1 = atc2.level1_parent
        
        atc_codes_list.append({
            'code': atc5.code,
            'name': atc5.name,
            'ddd': atc5.ddd or 'N/A',
            'uom': atc5.uom or 'N/A',
            'adm_r': atc5.adm_r or 'N/A',
            'note': atc5.note or 'N/A',
            'hierarchy': f"{atc1.code} > {atc2.code} > {atc3.code} > {atc4.code} > {atc5.code}",
            'hierarchy_names': {
                'level1': {'code': atc1.code, 'name': atc1.name},
                'level2': {'code': atc2.code, 'name': atc2.name},
                'level3': {'code': atc3.code, 'name': atc3.name},
                'level4': {'code': atc4.code, 'name': atc4.name},
                'level5': {'code': atc5.code, 'name': atc5.name}
            }
        })
        
    # Fetch salts
    salts = drug.salts.all()
    
    # Function to get full parent hierarchy
    def get_parent_hierarchy(category):
        hierarchy = []
        current = category
        while current.parent:
            hierarchy.append(current.parent.name)
            current = current.parent
        return ' > '.join(reversed(hierarchy)) if hierarchy else None

    # Prepare category display with parent hierarchy
    categories_display = []
    for category in drug.categories:
        parent_hierarchy = get_parent_hierarchy(category)
        display_name = f"{category.name}" + (f" (Parent: {parent_hierarchy})" if parent_hierarchy else "")
        categories_display.append(display_name)
    categories_str = ', '.join(categories_display) if categories_display else 'N/A'

    # Prepare drug-level information
    drug_info = {
        'id': drug.id,
        'name_en': drug.name_en,
        'name_tr': drug.name_tr,
        'categories': categories_str,
        'fda_approved': drug.fda_approved,
        'indications': drug.indications,
        'chembl_id': drug.chembl_id,
        'pharmgkb_id': drug.pharmgkb_id
    }
    
    # Fetch DrugDetail entries
    details = DrugDetail.query.filter_by(drug_id=drug_id).all()
    enriched_details = []
    
    for detail in details:
        # Define paths
        salt_suffix = f"_salt_{detail.salt_id or 'none'}"
        svg_filename = f'sketcher_{drug_id}{salt_suffix}.svg'
        svg_path = os.path.join('static', 'Uploads', svg_filename).replace('\\', '/')
        logger.debug(f"Checking SVG for drug_id={drug_id}, salt_id={detail.salt_id or 'none'}, path={svg_path}")
        if detail.smiles:
            if not os.path.exists(svg_path):
                try:
                    logger.info(f"Generating SVG for drug_id={drug_id}, salt_id={detail.salt_id or 'none'}")
                    mol = Chem.MolFromSmiles(detail.smiles)
                    if mol is None:
                        logger.error(f"Invalid SMILES for drug_id={drug_id}: {detail.smiles}")
                    else:
                        os.makedirs(os.path.dirname(svg_path), exist_ok=True)
                        Draw.MolToFile(mol, svg_path, size=(300, 300))
                        detail.structure = os.path.join('Uploads', svg_filename).replace('\\', '/')
                        db.session.commit()
                        logger.info(f"Saved SVG to {svg_path}")
                except Exception as e:
                    logger.error(f"Error generating SVG for drug_id={drug_id}: {str(e)}")
            else:
                logger.debug(f"SVG already exists at {svg_path}")
        else:
            logger.warning(f"No SMILES provided for drug_id={drug_id}, salt_id={detail.salt_id or 'none'}")
        
        # Generate 3D PDB if SMILES exists
        pdb_filename = f"drug_{drug_id}_salt_{detail.salt_id or 'none'}_3d.pdb"
        pdb_path = os.path.join('static', 'Uploads', '3d_structures', pdb_filename).replace('\\', '/')
        if detail.smiles and not os.path.exists(pdb_path):
            try:
                logger.info(f"Generating PDB for drug_id={drug_id}, salt_id={detail.salt_id or 'none'}")
                structure_3d_path, error = generate_3d_structure(detail.smiles, pdb_filename)
                if error:
                    logger.error(f"Error generating PDB for drug_id={drug_id}: {error}")
                else:
                    detail.structure_3d = structure_3d_path.replace('static/', '')
                    db.session.commit()
                    logger.info(f"Saved PDB to {structure_3d_path}")
            except Exception as e:
                logger.error(f"Error generating PDB for drug_id={drug_id}: {e}")
        
        routes_info = []
        for route in detail.routes:
            route_indications = [
                {
                    'name_en': ri.indication.name_en,
                    'name_tr': ri.indication.name_tr or 'N/A',
                    'icd11_code': ri.indication.code or 'N/A'
                }
                for ri in RouteIndication.query.filter_by(
                    drug_detail_id=detail.id,
                    route_id=route.route_id
                ).all()
            ]
            # Fetch metabolism data with validation
            metabolism_organs = [organ.name for organ in route.metabolism_organs] if route.metabolism_organs else []
            metabolism_enzymes = [enzyme.name for enzyme in route.metabolism_enzymes] if route.metabolism_enzymes else []
            metabolites = [
                {'id': met.id, 'name': met.name, 'parent_id': met.parent_id} for met in route.metabolites
            ] if route.metabolites else []
            
            # Debug logging for metabolism data
            if not metabolism_organs:
                logger.warning(f"No metabolism organs for drug_id={drug_id}, drug_detail_id={detail.id}, route_id={route.route_id}")
                organ_ids = [organ.id for organ in route.metabolism_organs]
                logger.debug(f"Metabolism organ IDs: {organ_ids}")
            if not metabolism_enzymes:
                logger.warning(f"No metabolism enzymes for drug_id={drug_id}, drug_detail_id={detail.id}, route_id={route.route_id}")
                enzyme_ids = [enzyme.id for enzyme in route.metabolism_enzymes]
                logger.debug(f"Metabolism enzyme IDs: {enzyme_ids}")
            if not metabolites:
                logger.warning(f"No metabolites for drug_id={drug_id}, drug_detail_id={detail.id}, route_id={route.route_id}")
                metabolite_ids = [met.id for met in route.metabolites]
                logger.debug(f"Metabolite IDs: {metabolite_ids}")
            
            routes_info.append({
                'name': route.route.name,
                'pharmacodynamics': route.pharmacodynamics,
                'pharmacokinetics': route.pharmacokinetics,
                'indications': route_indications,
                'absorption_rate': {
                    'min': route.absorption_rate_min,
                    'max': route.absorption_rate_max,
                    'unit': Unit.query.get(route.absorption_rate_unit_id).name if route.absorption_rate_unit_id else 'N/A'
                },
                'volume_of_distribution': {
                    'min': route.vod_rate_min,
                    'max': route.vod_rate_max,
                    'unit': Unit.query.get(route.vod_rate_unit_id).name if route.vod_rate_unit_id else 'N/A'
                },
                'protein_binding': {
                    'min': route.protein_binding_min,
                    'max': route.protein_binding_max,
                    'unit': '%'
                },
                'half_life': {
                    'min': route.half_life_min,
                    'max': route.half_life_max,
                    'unit': Unit.query.get(route.half_life_unit_id).name if route.half_life_unit_id else 'N/A'
                },
                'clearance_rate': {
                    'min': route.clearance_rate_min,
                    'max': route.clearance_rate_max,
                    'unit': Unit.query.get(route.clearance_rate_unit_id).name if route.clearance_rate_unit_id else 'N/A'
                },
                'bioavailability': {
                    'min': route.bioavailability_min,
                    'max': route.bioavailability_max,
                    'unit': '%'
                },
                'tmax': {
                    'min': route.tmax_min,
                    'max': route.tmax_max,
                    'unit': Unit.query.get(route.tmax_unit_id).name if route.tmax_unit_id else 'N/A'
                },
                'cmax': {
                    'min': route.cmax_min,
                    'max': route.cmax_max,
                    'unit': Unit.query.get(route.cmax_unit_id).name if route.cmax_unit_id else 'N/A'
                },
                'therapeutic_range': {
                    'min': route.therapeutic_min,
                    'max': route.therapeutic_max,
                    'unit': Unit.query.get(route.therapeutic_unit_id).name if route.therapeutic_unit_id else 'N/A'
                },
                'metabolism_organs': metabolism_organs,
                'metabolism_enzymes': metabolism_enzymes,
                'metabolites': metabolites
            })

        indications_list = []
        if detail.indications:
            indication_ids = [int(ind_id) for ind_id in detail.indications.split(',') if ind_id.isdigit()]
            indications = Indication.query.filter(Indication.id.in_(indication_ids)).all()
            indications_list = [
                {
                    'name_en': ind.name_en,
                    'name_tr': ind.name_tr or 'N/A',
                    'icd11_code': ind.code or 'N/A'
                } for ind in indications
            ]

        target_molecules_list = []
        if detail.target_molecules:
            target_ids = [tid for tid in detail.target_molecules.split(',') if tid.isdigit()]
            targets = Target.query.filter(Target.id.in_([int(tid) for tid in target_ids])).all()
            target_molecules_list = [t.name_en for t in targets]

        side_effects_list = [
            {"name_en": se.name_en, "name_tr": se.name_tr or 'N/A'}
            for se in detail.side_effects
        ]

        # Fetch pregnancy and lactation safety info
        pregnancy_safety_trimester1 = detail.pregnancy_safety_trimester1.name if detail.pregnancy_safety_trimester1 else 'N/A'
        pregnancy_safety_trimester2 = detail.pregnancy_safety_trimester2.name if detail.pregnancy_safety_trimester2 else 'N/A'
        pregnancy_safety_trimester3 = detail.pregnancy_safety_trimester3.name if detail.pregnancy_safety_trimester3 else 'N/A'
        lactation_safety = detail.lactation_safety.name if detail.lactation_safety else 'N/A'

        enriched_details.append({
            'id': detail.id,
            'salt_id': detail.salt_id,
            'salt_name': detail.salt.name_en if detail.salt else None,
            'is_drug_only': detail.salt_id is None,
            'mechanism_of_action': detail.mechanism_of_action,
            'fda_approved': detail.fda_approved,
            'ema_approved': detail.ema_approved,
            'titck_approved': detail.titck_approved,
            'molecular_formula': detail.molecular_formula,
            'synthesis': detail.synthesis,
            'structure': detail.structure or os.path.join('Uploads', svg_filename).replace('\\', '/'),
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
            'molecular_weight_unit': Unit.query.get(detail.molecular_weight_unit_id).name if detail.molecular_weight_unit_id else 'N/A',
            'solubility': detail.solubility,
            'solubility_unit': Unit.query.get(detail.solubility_unit_id).name if detail.solubility_unit_id else 'N/A',
            'pharmacodynamics': detail.pharmacodynamics,
            'pharmacokinetics': detail.pharmacokinetics,
            'boiling_point': detail.boiling_point,
            'boiling_point_unit': Unit.query.get(detail.boiling_point_unit_id).name if detail.boiling_point_unit_id else 'N/A',
            'melting_point': detail.melting_point,
            'melting_point_unit': Unit.query.get(detail.melting_point_unit_id).name if detail.melting_point_unit_id else 'N/A',
            'density': detail.density,
            'density_unit': Unit.query.get(detail.density_unit_id).name if detail.density_unit_id else 'N/A',
            'flash_point': detail.flash_point,
            'flash_point_unit': Unit.query.get(detail.flash_point_unit_id).name if detail.flash_point_unit_id else 'N/A',
            'routes': routes_info,
            'indications': indications_list,
            'target_molecules': target_molecules_list,
            'side_effects': side_effects_list,
            'black_box_warning': detail.black_box_warning,
            'black_box_details': detail.black_box_details,
            'references': detail.references,
            'pregnancy_safety_trimester1': pregnancy_safety_trimester1,
            'pregnancy_details_trimester1': detail.pregnancy_details_trimester1,
            'pregnancy_safety_trimester2': pregnancy_safety_trimester2,
            'pregnancy_details_trimester2': detail.pregnancy_details_trimester2,
            'pregnancy_safety_trimester3': pregnancy_safety_trimester3,
            'pregnancy_details_trimester3': detail.pregnancy_details_trimester3,
            'lactation_safety': lactation_safety,
            'lactation_details': detail.lactation_details
        })

    return render_template(
        'drug_detail.html',
        drug=drug_info,
        salts=salts,
        details=enriched_details,
        atc_codes=atc_codes_list
    )

import qrcode
import io

@app.route('/generate_qr/<int:drug_id>')
def generate_qr(drug_id):
    drug_detail_url = url_for('drug_detail', drug_id=drug_id, _external=True)
    qr = qrcode.make(drug_detail_url)
    qr_io = io.BytesIO()
    qr.save(qr_io, 'PNG')
    qr_io.seek(0)
    return send_file(qr_io, mimetype='image/png')





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

#EDITING THE DETAILS

# Define allowed tags, attributes, and styles for sanitization
ALLOWED_TAGS = ['b', 'i', 'u', 'p', 'strong', 'em', 'span', 'ul', 'ol', 'li', 'a']
ALLOWED_STYLES = ['color', 'background-color']

def allow_style_attrs(tag, name, value):
    if name != 'style':
        return name in ['href'] and tag == 'a'
    styles = [style.strip() for style in value.split(';') if style.strip()]
    filtered_styles = [style for style in styles if style.split(':', 1)[0].strip().lower() in ALLOWED_STYLES]
    return '; '.join(filtered_styles) if filtered_styles else None

ALLOWED_ATTRIBUTES = {
    'a': ['href'],
    'span': allow_style_attrs
}

cleaner = Cleaner(tags=ALLOWED_TAGS, attributes=ALLOWED_ATTRIBUTES, strip=True)

def sanitize_field(field_value):
    return cleaner.clean(field_value) if field_value else None

@app.route('/details/update/<int:detail_id>', methods=['GET', 'POST'])
@login_required
def update_detail(detail_id):
    # Eagerly load DrugDetail with related data
    detail = (
        DrugDetail.query
        .options(
            joinedload(DrugDetail.routes)
            .joinedload(DrugRoute.route_indications)
            .joinedload(RouteIndication.indication),
            joinedload(DrugDetail.routes)
            .joinedload(DrugRoute.metabolism_organs),
            joinedload(DrugDetail.routes)
            .joinedload(DrugRoute.metabolism_enzymes),
            joinedload(DrugDetail.routes)
            .joinedload(DrugRoute.metabolites),
            joinedload(DrugDetail.side_effects),
            joinedload(DrugDetail.drug).joinedload(Drug.categories)
        )
        .get_or_404(detail_id)
    )
    # Fetch related data
    drugs = Drug.query.all()
    salts = Salt.query.all()
    targets = Target.query.all()
    routes = RouteOfAdministration.query.all()
    side_effects = SideEffect.query.all()
    metabolites = Metabolite.query.all()
    metabolism_organs = MetabolismOrgan.query.all()
    metabolism_enzymes = MetabolismEnzyme.query.all()
    units = Unit.query.all()
    safety_categories = SafetyCategory.query.all()
    categories = DrugCategory.query.order_by(DrugCategory.name).all()
    units_json = [{'id': unit.id, 'name': unit.name} for unit in units]
    # Load all indications to allow full selection in the update form
    indications = Indication.query.all()
    drug = detail.drug
    if not drug:
        logger.error(f"Drug with ID {detail.drug_id} not found for DrugDetail ID {detail_id}")
        return render_template(
            'update_detail.html', detail=detail, drugs=drugs, salts=salts, indications=indications,
            targets=targets, routes=routes, side_effects=side_effects, metabolites=metabolites,
            metabolism_organs=metabolism_organs, metabolism_enzymes=metabolism_enzymes,
            units=units, units_json=units_json, safety_categories=safety_categories, categories=categories,
            error_message="Associated drug not found!"
        )
    if request.method == 'POST':
        logger.debug("Entering POST block")
        logger.debug(f"Form data: {request.form}")
        try:
            drug_id = request.form.get('drug_id')
            if not drug_id or not drug_id.isdigit():
                logger.error("Invalid drug_id: must be an integer")
                return render_template(
                    'update_detail.html', detail=detail, drugs=drugs, salts=salts, indications=indications,
                    targets=targets, routes=routes, side_effects=side_effects, metabolites=metabolites,
                    metabolism_organs=metabolism_organs, metabolism_enzymes=metabolism_enzymes,
                    units=units, units_json=units_json, safety_categories=safety_categories, categories=categories,
                    error_message="Drug ID is required and must be an integer!"
                )
            drug_id = int(drug_id)
            drug = Drug.query.get(drug_id)
            if not drug:
                logger.error(f"Drug with ID {drug_id} not found")
                return render_template(
                    'update_detail.html', detail=detail, drugs=drugs, salts=salts, indications=indications,
                    targets=targets, routes=routes, side_effects=side_effects, metabolites=metabolites,
                    metabolism_organs=metabolism_organs, metabolism_enzymes=metabolism_enzymes,
                    units=units, units_json=units_json, safety_categories=safety_categories, categories=categories,
                    error_message="Drug not found!"
                )
            salt_id = request.form.get('salt_id')
            if salt_id == '' or salt_id is None:
                salt_id = None
            else:
                try:
                    salt_id = int(salt_id)
                    if salt_id and not Salt.query.get(salt_id):
                        logger.error(f"Salt with ID {salt_id} not found")
                        return render_template(
                            'update_detail.html', detail=detail, drugs=drugs, salts=salts, indications=indications,
                            targets=targets, routes=routes, side_effects=side_effects, metabolites=metabolites,
                            metabolism_organs=metabolism_organs, metabolism_enzymes=metabolism_enzymes,
                            units=units, units_json=units_json, safety_categories=safety_categories, categories=categories,
                            error_message="Invalid salt ID."
                        )
                except ValueError:
                    logger.error(f"Invalid salt_id: {salt_id}")
                    return render_template(
                        'update_detail.html', detail=detail, drugs=drugs, salts=salts, indications=indications,
                        targets=targets, routes=routes, side_effects=side_effects, metabolites=metabolites,
                        metabolism_organs=metabolism_organs, metabolism_enzymes=metabolism_enzymes,
                        units=units, units_json=units_json, safety_categories=safety_categories, categories=categories,
                        error_message="Invalid salt_id. Must be an integer or empty."
                    )
            existing_detail = DrugDetail.query.filter_by(drug_id=drug_id, salt_id=salt_id).filter(DrugDetail.id != detail_id).first()
            if existing_detail:
                logger.error(f"DrugDetail already exists for drug_id={drug_id}, salt_id={salt_id}")
                return render_template(
                    'update_detail.html', detail=detail, drugs=drugs, salts=salts, indications=indications,
                    targets=targets, routes=routes, side_effects=side_effects, metabolites=metabolites,
                    metabolism_organs=metabolism_organs, metabolism_enzymes=metabolism_enzymes,
                    units=units, units_json=units_json, safety_categories=safety_categories, categories=categories,
                    error_message="This drug and salt combination already exists!"
                )
            selected_category_ids = request.form.getlist('category_ids[]')
            valid_category_ids = [int(cat_id) for cat_id in selected_category_ids if DrugCategory.query.get(int(cat_id))]
            mechanism_of_action = sanitize_field(request.form.get('mechanism_of_action'))
            synthesis = sanitize_field(request.form.get('synthesis'))
            pharmacodynamics = sanitize_field(request.form.get('pharmacodynamics'))
            pharmacokinetics = sanitize_field(request.form.get('pharmacokinetics'))
            references = sanitize_field(request.form.get('references'))
            black_box_details = sanitize_field(request.form.get('black_box_details')) if 'black_box_warning' in request.form else None
            pregnancy_safety_id = request.form.get('pregnancy_safety_id', type=int)
            pregnancy_details = None
            if pregnancy_safety_id:
                pregnancy_safety = SafetyCategory.query.get(pregnancy_safety_id)
                if pregnancy_safety and pregnancy_safety.name in ['Caution', 'Contraindicated']:
                    pregnancy_details = sanitize_field(request.form.get('pregnancy_details'))
            lactation_safety_id = request.form.get('lactation_safety_id', type=int)
            lactation_details = None
            if lactation_safety_id:
                lactation_safety = SafetyCategory.query.get(lactation_safety_id)
                if lactation_safety and lactation_safety.name in ['Caution', 'Contraindicated']:
                    lactation_details = sanitize_field(request.form.get('lactation_details'))
            smiles = request.form.get('smiles')
            structure_filename = detail.structure
            old_structure_path = os.path.join('static', detail.structure) if detail.structure else None
            structure = request.files.get('structure')
            if structure:
                original_filename = secure_filename(structure.filename)
                file_ext = os.path.splitext(original_filename)[1] or '.png'
                structure_filename = f"structure_{drug_id}_salt_{salt_id or 'none'}{file_ext}"
                structure_path = os.path.join('static', 'Uploads', structure_filename).replace('\\', '/')
                os.makedirs(os.path.dirname(structure_path), exist_ok=True)
                if old_structure_path and os.path.exists(old_structure_path):
                    try:
                        os.remove(old_structure_path)
                        logger.info(f"Deleted old structure file: {old_structure_path}")
                    except Exception as e:
                        logger.warning(f"Failed to delete old structure file {old_structure_path}: {str(e)}")
                structure.save(structure_path)
                structure_filename = os.path.join('Uploads', structure_filename).replace('\\', '/')
                logger.info(f"Uploaded structure file saved as {structure_path}")
            elif smiles and (not detail.structure or not os.path.exists(os.path.join('static', detail.structure))):
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        svg_filename = f"sketcher_{drug_id}_salt_{salt_id or 'none'}.svg"
                        svg_path = os.path.join('static', 'Uploads', svg_filename).replace('\\', '/')
                        os.makedirs(os.path.dirname(svg_path), exist_ok=True)
                        if old_structure_path and os.path.exists(old_structure_path):
                            try:
                                os.remove(old_structure_path)
                                logger.info(f"Deleted old structure file: {old_structure_path}")
                            except Exception as e:
                                logger.warning(f"Failed to delete old structure file {old_structure_path}: {str(e)}")
                        Draw.MolToFile(mol, svg_path, size=(300, 300))
                        structure_filename = os.path.join('Uploads', svg_filename).replace('\\', '/')
                        logger.info(f"Structure file saved as {svg_path}")
                    else:
                        logger.error(f"Invalid SMILES: {smiles}")
                        return render_template(
                            'update_detail.html', detail=detail, drugs=drugs, salts=salts, indications=indications,
                            targets=targets, routes=routes, side_effects=side_effects, metabolites=metabolites,
                            metabolism_organs=metabolism_organs, metabolism_enzymes=metabolism_enzymes,
                            units=units, units_json=units_json, safety_categories=safety_categories, categories=categories,
                            error_message="Invalid SMILES string provided."
                        )
                except Exception as e:
                    logger.error(f"Error generating SVG: {str(e)}")
                    return render_template(
                        'update_detail.html', detail=detail, drugs=drugs, salts=salts, indications=indications,
                        targets=targets, routes=routes, side_effects=side_effects, metabolites=metabolites,
                        metabolism_organs=metabolism_organs, metabolism_enzymes=metabolism_enzymes,
                        units=units, units_json=units_json, safety_categories=safety_categories, categories=categories,
                        error_message=f"Error generating structure from SMILES: {str(e)}"
                    )
            structure_3d_filename = detail.structure_3d
            old_structure_3d_path = os.path.join('static', detail.structure_3d) if detail.structure_3d else None
            if smiles:
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if not mol:
                        logger.error(f"Invalid SMILES for 3D structure: {smiles}")
                        return render_template(
                            'update_detail.html', detail=detail, drugs=drugs, salts=salts, indications=indications,
                            targets=targets, routes=routes, side_effects=side_effects, metabolites=metabolites,
                            metabolism_organs=metabolism_organs, metabolism_enzymes=metabolism_enzymes,
                            units=units, units_json=units_json, safety_categories=safety_categories, categories=categories,
                            error_message="Invalid SMILES string for 3D structure generation."
                        )
                    pdb_filename = f"drug_{drug_id}_salt_{salt_id or 'none'}_3d.pdb"
                    structure_3d_path, error = generate_3d_structure(smiles, pdb_filename)
                    if error:
                        logger.error(f"3D structure generation failed: {error}")
                    else:
                        if old_structure_3d_path and os.path.exists(old_structure_3d_path):
                            try:
                                os.remove(old_structure_3d_path)
                                logger.info(f"Deleted old 3D structure file: {old_structure_3d_path}")
                            except Exception as e:
                                logger.warning(f"Failed to delete old 3D structure file {old_structure_3d_path}: {str(e)}")
                        structure_3d_filename = structure_3d_path
                        logger.info(f"3D structure saved as {structure_3d_filename}")
                except Exception as e:
                    logger.error(f"Error generating 3D structure: {str(e)}")
                    return render_template(
                        'update_detail.html', detail=detail, drugs=drugs, salts=salts, indications=indications,
                        targets=targets, routes=routes, side_effects=side_effects, metabolites=metabolites,
                        metabolism_organs=metabolism_organs, metabolism_enzymes=metabolism_enzymes,
                        units=units, units_json=units_json, safety_categories=safety_categories, categories=categories,
                        error_message=f"Error generating 3D structure: {str(e)}"
                    )
            changed_fields = []
            if detail.drug_id != drug_id:
                changed_fields.append(f"drug_id: {detail.drug_id} -> {drug_id}")
            if detail.salt_id != salt_id:
                changed_fields.append(f"salt_id: {detail.salt_id} -> {salt_id}")
            detail.drug_id = drug_id
            detail.salt_id = salt_id
            detail.mechanism_of_action = mechanism_of_action
            detail.molecular_formula = request.form.get('molecular_formula')
            detail.synthesis = synthesis
            detail.structure = structure_filename
            detail.structure_3d = structure_3d_filename
            detail.iupac_name = request.form.get('iupac_name')
            detail.smiles = smiles
            detail.inchikey = request.form.get('inchikey')
            detail.pubchem_cid = request.form.get('pubchem_cid')
            detail.pubchem_sid = request.form.get('pubchem_sid')
            detail.cas_id = request.form.get('cas_id')
            detail.ec_number = request.form.get('ec_number')
            detail.nci_code = request.form.get('nci_code')
            detail.rxcui = request.form.get('rxcui')
            detail.snomed_id = request.form.get('snomed_id')
            detail.molecular_weight = request.form.get('molecular_weight', type=float)
            detail.molecular_weight_unit_id = request.form.get('molecular_weight_unit_id', type=int)
            detail.solubility = request.form.get('solubility', type=float)
            detail.solubility_unit_id = request.form.get('solubility_unit_id', type=int)
            detail.pharmacodynamics = pharmacodynamics
            detail.pharmacokinetics = pharmacokinetics
            detail.boiling_point = request.form.get('boiling_point', type=float)
            detail.boiling_point_unit_id = request.form.get('boiling_point_unit_id', type=int)
            detail.melting_point = request.form.get('melting_point', type=float)
            detail.melting_point_unit_id = request.form.get('melting_point_unit_id', type=int)
            detail.density = request.form.get('density', type=float)
            detail.density_unit_id = request.form.get('density_unit_id', type=int)
            detail.flash_point = request.form.get('flash_point', type=float)
            detail.flash_point_unit_id = request.form.get('flash_point_unit_id', type=int)
            detail.half_life = request.form.get('half_life', type=float)
            detail.half_life_unit_id = request.form.get('half_life_unit_id', type=int)
            detail.clearance_rate = request.form.get('clearance_rate', type=float)
            detail.clearance_rate_unit_id = request.form.get('clearance_rate_unit_id', type=int)
            detail.bioavailability = request.form.get('bioavailability', type=float)
            detail.fda_approved = 'fda_approved' in request.form
            detail.ema_approved = 'ema_approved' in request.form
            detail.titck_approved = 'titck_approved' in request.form
            detail.black_box_warning = 'black_box_warning' in request.form
            detail.black_box_details = black_box_details
            detail.references = references
            detail.pregnancy_safety_id = pregnancy_safety_id
            detail.pregnancy_details = pregnancy_details
            detail.lactation_safety_id = lactation_safety_id
            detail.lactation_details = lactation_details
            detail.indications = ','.join(request.form.getlist('indications[]')) or None
            detail.target_molecules = ','.join(request.form.getlist('target_molecules[]')) or None
            selected_side_effects = request.form.getlist('side_effects[]')
            valid_side_effects = [SideEffect.query.get(se_id) for se_id in selected_side_effects if SideEffect.query.get(se_id)]
            detail.side_effects = valid_side_effects
            changed_fields.append(f"side_effects: updated to {len(valid_side_effects)} items")
            if valid_category_ids:
                drug.categories.clear()
                selected_categories = DrugCategory.query.filter(DrugCategory.id.in_(valid_category_ids)).all()
                drug.categories.extend(selected_categories)
                changed_fields.append(f"categories: updated to {valid_category_ids}")
                logger.info(f"Updated categories for drug_id={drug_id}: {valid_category_ids}")
            else:
                logger.debug(f"No categories selected for drug_id={drug_id}")
            def parse_range(value):
                if not value:
                    return None, None
                value = value.replace('–', '-')
                if '-' in value:
                    try:
                        min_val, max_val = map(float, value.split('-'))
                        return min_val, max_val
                    except ValueError:
                        logger.warning(f"Failed to parse range '{value}'")
                        return None, None
                try:
                    val = float(value)
                    return val, val
                except ValueError:
                    logger.warning(f"Failed to parse single value '{value}'")
                    return None, None
            selected_routes = request.form.getlist('route_id[]')
            existing_routes = {route.route_id: route for route in detail.routes}
            removed_routes = []
            for route_id in selected_routes:
                try:
                    route_id = int(route_id)
                    if not RouteOfAdministration.query.get(route_id):
                        logger.error(f"Invalid route_id: {route_id}")
                        continue
                    pd = sanitize_field(request.form.get(f'route_pharmacodynamics_{route_id}', ''))
                    pk = sanitize_field(request.form.get(f'route_pharmacokinetics_{route_id}', ''))
                    absorption_rate = request.form.get(f'absorption_rate_{route_id}', '')
                    absorption_rate_unit_id = request.form.get(f'absorption_rate_unit_id_{route_id}', type=int)
                    vod_rate = request.form.get(f'vod_rate_{route_id}', '')
                    vod_rate_unit_id = request.form.get(f'vod_rate_unit_id_{route_id}', type=int)
                    protein_binding = request.form.get(f'protein_binding_{route_id}', '')
                    half_life = request.form.get(f'half_life_{route_id}', '')
                    half_life_unit_id = request.form.get(f'half_life_unit_id_{route_id}', type=int)
                    clearance_rate = request.form.get(f'clearance_rate_{route_id}', '')
                    clearance_rate_unit_id = request.form.get(f'clearance_rate_unit_id_{route_id}', type=int)
                    bioavailability = request.form.get(f'bioavailability_{route_id}', '')
                    tmax = request.form.get(f'tmax_{route_id}', '')
                    tmax_unit_id = request.form.get(f'tmax_unit_id_{route_id}', type=int)
                    cmax = request.form.get(f'cmax_{route_id}', '')
                    cmax_unit_id = request.form.get(f'cmax_unit_id_{route_id}', type=int)
                    therapeutic_range = request.form.get(f'therapeutic_range_{route_id}', '')
                    therapeutic_unit_id = request.form.get(f'therapeutic_unit_id_{route_id}', type=int)
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
                    therapeutic_min, therapeutic_max = parse_range(therapeutic_range)
                    metabolism_organs_ids = request.form.getlist(f'metabolism_organs_{route_id}[]')
                    metabolism_enzymes_ids = request.form.getlist(f'metabolism_enzymes_{route_id}[]')
                    metabolite_ids = request.form.getlist(f'metabolites_{route_id}[]')
                    valid_metabolism_organs_ids = [int(organ_id) for organ_id in metabolism_organs_ids if MetabolismOrgan.query.get(int(organ_id))]
                    valid_metabolism_enzymes_ids = [int(enzyme_id) for enzyme_id in metabolism_enzymes_ids if MetabolismEnzyme.query.get(int(enzyme_id))]
                    valid_metabolite_ids = [int(metabolite_id) for metabolite_id in metabolite_ids if Metabolite.query.get(int(metabolite_id))]
                    logger.debug(f"Route ID {route_id} -> "
                                 f"Absorption: {absorption_min}-{absorption_max}, Unit ID: {absorption_rate_unit_id}, "
                                 f"VoD: {vod_min}-{vod_max}, Unit ID: {vod_rate_unit_id}, "
                                 f"Protein Binding: {protein_min}-{protein_max}, "
                                 f"Half-Life: {half_life_min}-{half_life_max}, Unit ID: {half_life_unit_id}, "
                                 f"Clearance: {clearance_min}-{clearance_max}, Unit ID: {clearance_rate_unit_id}, "
                                 f"Bioavailability: {bio_min}-{bio_max}, "
                                 f"Tmax: {tmax_min}-{tmax_max}, Unit ID: {tmax_unit_id}, "
                                 f"Cmax: {cmax_min}-{cmax_max}, Unit ID: {cmax_unit_id}, "
                                 f"Therapeutic Range: {therapeutic_min}-{therapeutic_max}, Unit ID: {therapeutic_unit_id}, "
                                 f"Organs IDs: {valid_metabolism_organs_ids}, "
                                 f"Enzymes IDs: {valid_metabolism_enzymes_ids}, "
                                 f"Metabolite IDs: {valid_metabolite_ids}, "
                                 f"PD: {pd}, PK: {pk}")
                    if route_id in existing_routes:
                        route = existing_routes[route_id]
                        route.pharmacodynamics = pd or route.pharmacodynamics
                        route.pharmacokinetics = pk or route.pharmacokinetics
                        route.absorption_rate_min = absorption_min
                        route.absorption_rate_max = absorption_max
                        route.absorption_rate_unit_id = absorption_rate_unit_id
                        route.vod_rate_min = vod_min
                        route.vod_rate_max = vod_max
                        route.vod_rate_unit_id = vod_rate_unit_id
                        route.protein_binding_min = protein_min
                        route.protein_binding_max = protein_max
                        route.half_life_min = half_life_min
                        route.half_life_max = half_life_max
                        route.half_life_unit_id = half_life_unit_id
                        route.clearance_rate_min = clearance_min
                        route.clearance_rate_max = clearance_max
                        route.clearance_rate_unit_id = clearance_rate_unit_id
                        route.bioavailability_min = bio_min
                        route.bioavailability_max = bio_max
                        route.tmax_min = tmax_min
                        route.tmax_max = tmax_max
                        route.tmax_unit_id = tmax_unit_id
                        route.cmax_min = cmax_min
                        route.cmax_max = cmax_max
                        route.cmax_unit_id = cmax_unit_id
                        route.therapeutic_min = therapeutic_min
                        route.therapeutic_max = therapeutic_max
                        route.therapeutic_unit_id = therapeutic_unit_id
                        changed_fields.append(f"Updated route_id={route_id}")
                        route.metabolism_organs = MetabolismOrgan.query.filter(MetabolismOrgan.id.in_(valid_metabolism_organs_ids)).all()
                        route.metabolism_enzymes = MetabolismEnzyme.query.filter(MetabolismEnzyme.id.in_(valid_metabolism_enzymes_ids)).all()
                        route.metabolites = Metabolite.query.filter(Metabolite.id.in_(valid_metabolite_ids)).all()
                    else:
                        new_route = DrugRoute(
                            drug_detail_id=detail.id,
                            route_id=route_id,
                            pharmacodynamics=pd,
                            pharmacokinetics=pk,
                            absorption_rate_min=absorption_min,
                            absorption_rate_max=absorption_max,
                            absorption_rate_unit_id=absorption_rate_unit_id,
                            vod_rate_min=vod_min,
                            vod_rate_max=vod_max,
                            vod_rate_unit_id=vod_rate_unit_id,
                            protein_binding_min=protein_min,
                            protein_binding_max=protein_max,
                            half_life_min=half_life_min,
                            half_life_max=half_life_max,
                            half_life_unit_id=half_life_unit_id,
                            clearance_rate_min=clearance_min,
                            clearance_rate_max=clearance_max,
                            clearance_rate_unit_id=clearance_rate_unit_id,
                            bioavailability_min=bio_min,
                            bioavailability_max=bio_max,
                            tmax_min=tmax_min,
                            tmax_max=tmax_max,
                            tmax_unit_id=tmax_unit_id,
                            cmax_min=cmax_min,
                            cmax_max=cmax_max,
                            cmax_unit_id=cmax_unit_id,
                            therapeutic_min=therapeutic_min,
                            therapeutic_max=therapeutic_max,
                            therapeutic_unit_id=therapeutic_unit_id
                        )
                        db.session.add(new_route)
                        new_route.metabolism_organs.extend(MetabolismOrgan.query.filter(MetabolismOrgan.id.in_(valid_metabolism_organs_ids)).all())
                        new_route.metabolism_enzymes.extend(MetabolismEnzyme.query.filter(MetabolismEnzyme.id.in_(valid_metabolism_enzymes_ids)).all())
                        new_route.metabolites.extend(Metabolite.query.filter(Metabolite.id.in_(valid_metabolite_ids)).all())
                        changed_fields.append(f"Added new route_id={route_id}")
                    RouteIndication.query.filter_by(drug_detail_id=detail.id, route_id=route_id).delete()
                    selected_route_indications = request.form.getlist(f'route_indications_{route_id}[]')
                    for indication_id in selected_route_indications:
                        try:
                            indication_id = int(indication_id)
                            if Indication.query.get(indication_id):
                                new_route_indication = RouteIndication(
                                    drug_detail_id=detail.id,
                                    route_id=route_id,
                                    indication_id=indication_id
                                )
                                db.session.add(new_route_indication)
                        except ValueError:
                            logger.warning(f"Invalid indication ID format for route_id={route_id}: {indication_id}")
                except ValueError:
                    logger.error(f"Invalid route_id format: {route_id}")
                    continue
            for existing_route_id in list(existing_routes.keys()):
                if str(existing_route_id) not in selected_routes:
                    route_to_remove = existing_routes[existing_route_id]
                    db.session.delete(route_to_remove)
                    removed_routes.append(existing_route_id)
                    logger.warning(f"Removed route_id={existing_route_id} for drug_detail_id={detail.id}")
            if changed_fields or removed_routes:
                logger.info(f"Changes for DrugDetail ID {detail.id}: {', '.join(changed_fields)}")
                if removed_routes:
                    logger.info(f"Removed routes: {removed_routes}")
            db.session.commit()
            logger.info(f"DrugDetail updated with ID: {detail.id}")
            drug_name = drug.name_en if hasattr(drug, 'name_en') and drug.name_en else f"Drug ID {drug_id}"
            news = News(
                category='Drug Update',
                title=f'Drug Details Updated: {drug_name}',
                description=f'Updated details for <a href="/drug/{drug_id}">{drug_name}</a> are now available on Drugly.',
                publication_date=datetime.utcnow()
            )
            db.session.add(news)
            db.session.commit()
            logger.info(f"News entry created for drug_id={drug_id}: {drug_name}")
            flash('Drug details updated and announced on the homepage!', 'success')
            return redirect(url_for('view_details'))
        except Exception as e:
            db.session.rollback()
            logger.error(f"Exception occurred: {str(e)}")
            return render_template(
                'update_detail.html', detail=detail, drugs=drugs, salts=salts, indications=indications,
                targets=targets, routes=routes, side_effects=side_effects, metabolites=metabolites,
                metabolism_organs=metabolism_organs, metabolism_enzymes=metabolism_enzymes,
                units=units, units_json=units_json, safety_categories=safety_categories, categories=categories,
                error_message=f"Error updating details: {str(e)}"
            )
    logger.debug(f"Preparing data for GET request, detail_id={detail_id}")
    selected_indications = detail.indications.split(',') if detail.indications else []
    selected_targets = detail.target_molecules.split(',') if detail.target_molecules else []
    selected_side_effects = [se.id for se in detail.side_effects] if detail.side_effects else []
    selected_categories = [cat.id for cat in drug.categories] if drug and drug.categories else []
    selected_routes = [
        {
            'route_id': route.route_id,
            'pharmacodynamics': route.pharmacodynamics,
            'pharmacokinetics': route.pharmacokinetics,
            'indications': [ri.indication_id for ri in route.route_indications],
            'absorption_rate_min': route.absorption_rate_min,
            'absorption_rate_max': route.absorption_rate_max,
            'absorption_rate_unit_id': route.absorption_rate_unit_id,
            'vod_rate_min': route.vod_rate_min,
            'vod_rate_max': route.vod_rate_max,
            'vod_rate_unit_id': route.vod_rate_unit_id,
            'protein_binding_min': route.protein_binding_min * 100 if route.protein_binding_min is not None else None,
            'protein_binding_max': route.protein_binding_max * 100 if route.protein_binding_max is not None else None,
            'half_life_min': route.half_life_min,
            'half_life_max': route.half_life_max,
            'half_life_unit_id': route.half_life_unit_id,
            'clearance_rate_min': route.clearance_rate_min,
            'clearance_rate_max': route.clearance_rate_max,
            'clearance_rate_unit_id': route.clearance_rate_unit_id,
            'bioavailability_min': route.bioavailability_min * 100 if route.bioavailability_min is not None else None,
            'bioavailability_max': route.bioavailability_max * 100 if route.bioavailability_max is not None else None,
            'tmax_min': route.tmax_min,
            'tmax_max': route.tmax_max,
            'tmax_unit_id': route.tmax_unit_id,
            'cmax_min': route.cmax_min,
            'cmax_max': route.cmax_max,
            'cmax_unit_id': route.cmax_unit_id,
            'therapeutic_min': route.therapeutic_min,
            'therapeutic_max': route.therapeutic_max,
            'therapeutic_unit_id': route.therapeutic_unit_id,
            'metabolism_organs': [mo.id for mo in route.metabolism_organs],
            'metabolism_enzymes': [me.id for me in route.metabolism_enzymes],
            'metabolites': [m.id for m in route.metabolites]
        }
        for route in detail.routes
    ]
    logger.debug(f"Rendering template for detail_id={detail_id}")
    return render_template(
        'update_detail.html',
        detail=detail,
        drugs=drugs,
        salts=salts,
        indications=indications,
        targets=targets,
        routes=routes,
        side_effects=side_effects,
        metabolites=metabolites,
        metabolism_organs=metabolism_organs,
        metabolism_enzymes=metabolism_enzymes,
        units=units,
        units_json=units_json,
        safety_categories=safety_categories,
        categories=categories,
        selected_indications=selected_indications,
        selected_targets=selected_targets,
        selected_side_effects=selected_side_effects,
        selected_categories=selected_categories,
        selected_routes=selected_routes
    )






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


# Updated /interactions route
@app.route('/interactions', methods=['GET', 'POST'])
@login_required
def check_interactions():
    drugs = Drug.query.all()
    interaction_results = []
    
    # Get current user information - MATCHING YOUR WORKING PATTERN
    user = None
    user_email = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        if user:
            user_email = user.email
    
    if request.method == 'POST':
        drug1_id = request.form.get('drug1_id')
        drug2_id = request.form.get('drug2_id')
        route_ids = request.form.getlist('route_ids[]')
        logger.debug(f"Received: drug1_id={drug1_id}, drug2_id={drug2_id}, route_ids={route_ids}")
        
        try:
            drug1_id = int(drug1_id) if drug1_id else None
            drug2_id = int(drug2_id) if drug2_id else None
            route_ids = [int(route_id) for route_id in route_ids if route_id] if route_ids else []
            
            if not drug1_id or not drug2_id or drug1_id == drug2_id:
                logger.error("Invalid or identical drug IDs provided")
                return render_template('interactions.html', 
                                    drugs=drugs, 
                                    interaction_results=[], 
                                    error="Please select two different drugs.",
                                    user_email=user_email,
                                    user=user)
            
            query = DrugInteraction.query.filter(
                or_(
                    and_(DrugInteraction.drug1_id == drug1_id, DrugInteraction.drug2_id == drug2_id),
                    and_(DrugInteraction.drug1_id == drug2_id, DrugInteraction.drug2_id == drug1_id)
                )
            )
            
            if route_ids:
                query = query.outerjoin(
                    DrugInteraction.routes
                ).filter(
                    or_(RouteOfAdministration.id.in_(route_ids), RouteOfAdministration.id.is_(None))
                ).distinct()
            
            interactions = query.all()
            logger.debug(f"Found {len(interactions)} interactions")
            
            # 🔥 ADD THIS TRACKING CODE HERE 🔥
            try:
                visitor_hash = get_visitor_hash(
                    request.headers.get('X-Forwarded-For', request.remote_addr).split(',')[0].strip(),
                    request.headers.get('User-Agent', '')
                )
                
                severity_levels = [interaction.severity_level.name for interaction in interactions if interaction.severity_level]
                
                interaction_check = InteractionCheck(
                    drug_ids=[drug1_id, drug2_id],
                    visitor_hash=visitor_hash,
                    user_id=session.get('user_id'),
                    interactions_found=len(interactions),
                    severity_levels=severity_levels,
                    time_spent=0  # Can be tracked with JS if needed
                )
                db.session.add(interaction_check)
                db.session.commit()
                logger.info(f"Tracked interaction check: {len(interactions)} interactions found")
            except Exception as track_error:
                logger.error(f"Error tracking interaction: {str(track_error)}")
                # Don't fail the request if tracking fails
            # 🔥 END OF TRACKING CODE 🔥
            
            for interaction in interactions:
                predicted_severity = getattr(interaction, 'predicted_severity', 'Bilinmiyor')
                interaction_results.append({
                    'drug1': interaction.drug1.name_en if interaction.drug1 else "Bilinmeyen",
                    'drug2': interaction.drug2.name_en if interaction.drug2 else "Bilinmeyen",
                    'route': ', '.join([route.name for route in interaction.routes]) if interaction.routes else "Genel",
                    'interaction_type': interaction.interaction_type,
                    'interaction_description': interaction.interaction_description,
                    'severity': interaction.severity_level.name if interaction.severity_level else "Bilinmiyor",
                    'predicted_severity': interaction.predicted_severity_level.name if interaction.predicted_severity_level else "Bilinmiyor",
                    'mechanism': interaction.mechanism or "Belirtilmemiş",
                    'monitoring': interaction.monitoring or "Belirtilmemiş",
                    'alternatives': interaction.alternatives or "Belirtilmemiş",
                    'reference': interaction.reference or "Belirtilmemiş",
                })
        except ValueError as e:
            logger.error(f"Invalid ID format: {e}")
            return render_template('interactions.html', 
                                 drugs=drugs, 
                                 interaction_results=[], 
                                 error="Invalid input format.",
                                 user_email=user_email,
                                 user=user)
        except Exception as e:
            logger.error(f"Error querying interactions: {e}")
            return render_template('interactions.html', 
                                 drugs=drugs, 
                                 interaction_results=[], 
                                 error="An error occurred while querying interactions.",
                                 user_email=user_email,
                                 user=user)
    
    return render_template('interactions.html', 
                         drugs=drugs, 
                         interaction_results=interaction_results,
                         user_email=user_email,
                         user=user)

# Cache for PubMed abstracts
ABSTRACT_CACHE = {}

# Initialize zero-shot classifier
device = 0 if torch.cuda.is_available() else -1
classifier = None

def get_classifier():
    """Lazy load classifier to avoid startup delays."""
    global classifier
    if classifier is None:
        try:
            classifier = pipeline(
                "zero-shot-classification",
                model="MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
                device=device
            )
            logger.info("Successfully loaded DeBERTa-v3 classifier")
        except Exception as e:
            logger.error(f"Failed to load DeBERTa-v3: {e}. Falling back to BART.")
            try:
                classifier = pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli",
                    device=device
                )
                logger.info("Successfully loaded BART classifier as fallback")
            except Exception as fallback_error:
                logger.error(f"Failed to load any classifier: {fallback_error}")
                raise RuntimeError("Could not initialize severity classifier")
    return classifier


def fetch_pubmed_abstracts(drug1, drug2, max_retries=3):
    """Fetch relevant PubMed abstracts for drug interactions."""
    cache_key = f"{drug1}:{drug2}"
    if cache_key in ABSTRACT_CACHE:
        logger.info(f"Using cached PubMed abstracts for {drug1} and {drug2}")
        return ABSTRACT_CACHE[cache_key]
    
    Entrez.email = os.environ.get('PUBMED_EMAIL')
    Entrez.api_key = os.environ.get('PUBMED_API_KEY')
    
    term = f'("{drug1}" AND "{drug2}" AND (drug interaction OR contraindication OR adverse effect OR pharmacodynamic OR pharmacokinetic))'
    logger.info(f"Fetching PubMed abstracts for term: {term}")
    
    for attempt in range(max_retries):
        try:
            handle = Entrez.esearch(db="pubmed", term=term, retmax=20)
            record = Entrez.read(handle)
            ids = record["IdList"]
            handle.close()
            
            if not ids:
                logger.info(f"No PubMed results for: {drug1} AND {drug2}")
                ABSTRACT_CACHE[cache_key] = []
                return []
            
            time.sleep(0.5)
            
            fetch_handle = Entrez.efetch(db="pubmed", id=ids, rettype="abstract", retmode="text")
            raw_data = fetch_handle.read()
            abstracts = [
                a.strip() for a in raw_data.split("\n\n")
                if a.strip() and len(a) > 100 and any(term in a.lower() for term in ["interaction", "adverse", "contraindication", "severe", "critical", "risk", "effect"])
            ]
            fetch_handle.close()
            
            logger.info(f"Fetched {len(abstracts)} relevant abstracts for {drug1} and {drug2}")
            ABSTRACT_CACHE[cache_key] = abstracts[:5]
            return ABSTRACT_CACHE[cache_key]
            
        except Exception as e:
            logger.error(f"PubMed fetch attempt {attempt + 1}/{max_retries} failed: {e}")
            time.sleep(2 ** attempt)
    
    logger.error(f"Failed to fetch PubMed data for {drug1} and {drug2} after {max_retries} attempts")
    ABSTRACT_CACHE[cache_key] = []
    return []

def classify_severity(description, drug1_name, drug2_name, manual_severity=None):
    """Classify interaction severity using AI with enhanced rules."""
    labels = ["Mild", "Moderate", "Severe", "Critical"]
    label_map = {"Mild": "Hafif", "Moderate": "Orta", "Severe": "Şiddetli", "Critical": "Hayati Risk İçeren"}
    severity_id_map = {"Hafif": 1, "Orta": 2, "Şiddetli": 3, "Hayati Risk İçeren": 4}  # ADDED
    
    critical_terms = [
        "contraindicated", "life-threatening", "severe hypotension", "organ failure",
        "anaphylaxis", "arrhythmia", "respiratory failure", "cardiac arrest",
        "severe adverse", "critical interaction", "prohibited", "fatal", "toxic", 
        "overdose risk", "hospitalization", "death", "coma", "seizure", "stroke",
        "myocardial infarction", "renal failure", "hepatotoxicity", "agranulocytosis"
    ]
    
    severe_terms = [
        "significant risk", "serious adverse", "requires monitoring", "dose adjustment",
        "increased toxicity", "enhanced effect", "decreased efficacy", "clinical significance"
    ]
    
    description_lower = description.lower()
    is_critical_in_description = any(term in description_lower for term in critical_terms)
    is_severe_in_description = any(term in description_lower for term in severe_terms)
    
    try:
        abstracts = fetch_pubmed_abstracts(drug1_name, drug2_name)
    except Exception as e:
        logger.error(f"Error fetching PubMed abstracts: {e}")
        abstracts = []
    
    if abstracts:
        combined_abstracts = " ".join(abstracts)
        sentences = re.split(r'[.!?]+', combined_abstracts)
        relevant_sentences = [
            s.strip() for s in sentences
            if len(s.strip()) > 20 and any(term in s.lower() for term in ["interaction", "adverse", "contraindication", "risk", "effect", "severe", "critical"])
        ]
        evidence = " ".join(relevant_sentences[:5])[:1000]
        is_critical_in_evidence = any(term in evidence.lower() for term in critical_terms)
        is_severe_in_evidence = any(term in evidence.lower() for term in severe_terms)
    else:
        evidence = ""
        is_critical_in_evidence = False
        is_severe_in_evidence = False
    
    desc_sentences = re.split(r'[.!?]+', description)
    relevant_desc = [
        s.strip() for s in desc_sentences
        if len(s.strip()) > 10 and any(term in s.lower() for term in ["interaction", "adverse", "contraindication", "risk", "effect", "severe", "critical"])
    ]
    processed_description = " ".join(relevant_desc[:5]) or description[:500]
    
    prompt = (
        f"Analyze the drug interaction severity between {drug1_name} and {drug2_name}.\n\n"
        f"Interaction Description: {processed_description}\n\n"
        f"Scientific Evidence: {evidence[:500] if evidence else 'No additional evidence available.'}\n\n"
        f"Classification Guidelines:\n"
        f"- Mild: Minimal clinical significance, no intervention needed, minor or theoretical risk.\n"
        f"- Moderate: Clinically relevant, requires monitoring, possible dose adjustment, manageable risk.\n"
        f"- Severe: High risk of significant harm, requires careful management, alternative therapy often recommended.\n"
        f"- Critical: Life-threatening, contraindicated combination, must be avoided, immediate danger to patient.\n\n"
        f"Classify this interaction strictly into ONE category based on the worst-case clinical outcome."
    )
    
    logger.debug(f"Classification prompt for {drug1_name} + {drug2_name}: {prompt[:300]}...")
    
    try:
        clf = get_classifier()
        result = clf(prompt, labels, multi_label=False)
        predicted_label = label_map[result["labels"][0]]
        confidence = result["scores"][0]
        
        if is_critical_in_description or is_critical_in_evidence:
            logger.info(f"Critical terms detected for {drug1_name} + {drug2_name}")
            predicted_label = "Hayati Risk İçeren"
            confidence = max(confidence, 0.95)
        elif is_severe_in_description or is_severe_in_evidence:
            if predicted_label in ["Hafif", "Orta"]:
                logger.info(f"Severe terms detected, upgrading severity for {drug1_name} + {drug2_name}")
                predicted_label = "Şiddetli"
                confidence = max(confidence, 0.85)
        
        if confidence < 0.75:
            logger.warning(f"Low confidence ({confidence:.2f}) for {drug1_name} + {drug2_name}. Predicted: {predicted_label}")
            severity_order = ["Hafif", "Orta", "Şiddetli", "Hayati Risk İçeren"]
            current_index = severity_order.index(predicted_label)
            if current_index < len(severity_order) - 1:
                predicted_label = severity_order[current_index + 1]
            confidence = 0.75
        
        if manual_severity == "Hayati Risk İçeren":
            logger.info(f"Manual severity is Critical for {drug1_name} + {drug2_name}, maintaining classification")
            predicted_label = "Hayati Risk İçeren"
            confidence = 0.99
        
        if manual_severity and predicted_label != manual_severity:
            severity_order = ["Hafif", "Orta", "Şiddetli", "Hayati Risk İçeren"]
            manual_idx = severity_order.index(manual_severity) if manual_severity in severity_order else -1
            predicted_idx = severity_order.index(predicted_label) if predicted_label in severity_order else -1
            
            if abs(manual_idx - predicted_idx) > 1:
                logger.warning(
                    f"Significant severity mismatch for {drug1_name} + {drug2_name}: "
                    f"Manual={manual_severity}, Predicted={predicted_label} (Confidence={confidence:.2f})"
                )
        
    except Exception as e:
        logger.error(f"Error classifying severity for {drug1_name} + {drug2_name}: {e}")
        predicted_label = "Şiddetli"
        confidence = 0.0
    
    predicted_severity_id = severity_id_map.get(predicted_label, 2)  # ADDED
    return predicted_label, confidence, predicted_severity_id  # CHANGED


@app.route('/interactions/manage', methods=['GET', 'POST'])
@login_required
@admin_required
def manage_interactions():
    drugs = Drug.query.order_by(Drug.name_en).all()
    routes = RouteOfAdministration.query.order_by(RouteOfAdministration.name).all()
    severities = Severity.query.order_by(Severity.id).all()
    
    page = request.args.get('page', 1, type=int)
    per_page = 10
    order_by_column = request.args.get('order_by', 'id')
    order_direction = request.args.get('direction', 'desc')
    
    valid_columns = ['id', 'drug1_id', 'drug2_id', 'interaction_type', 'severity_id']
    if order_by_column not in valid_columns:
        order_by_column = 'id'
    
    interactions_query = DrugInteraction.query.options(
        db.joinedload(DrugInteraction.drug1),
        db.joinedload(DrugInteraction.drug2),
        db.joinedload(DrugInteraction.severity_level),
        db.joinedload(DrugInteraction.routes)
    ).order_by(
        getattr(getattr(DrugInteraction, order_by_column), order_direction)()
    )
    
    interactions = interactions_query.paginate(page=page, per_page=per_page, error_out=False)
    
    if request.method == 'POST':
        try:
            drug1_id = request.form.get('drug1_id')
            drug2_id = request.form.get('drug2_id')
            route_ids = request.form.getlist('route_ids')
            interaction_type = request.form.get('interaction_type')
            interaction_description = request.form.get('interaction_description')
            reference = request.form.get('reference', '')
            mechanism = request.form.get('mechanism', '')
            monitoring = request.form.get('monitoring', '')
            alternatives = request.form.getlist('alternatives')
            
            logger.debug(f"Received: drug1_id={drug1_id}, drug2_id={drug2_id}, route_ids={route_ids}")
            
            if not drug1_id or not drug2_id or not interaction_type or not interaction_description:
                raise ValueError("Required fields are missing")
            
            drug1_id = int(drug1_id)
            drug2_id = int(drug2_id)
            
            if drug1_id == drug2_id:
                raise ValueError("Cannot create interaction between the same drug")
            
            route_ids = [int(route_id) for route_id in route_ids if route_id]
            
            existing = DrugInteraction.query.filter(
                or_(
                    and_(DrugInteraction.drug1_id == drug1_id, DrugInteraction.drug2_id == drug2_id),
                    and_(DrugInteraction.drug1_id == drug2_id, DrugInteraction.drug2_id == drug1_id)
                )
            ).first()
            
            if existing:
                raise ValueError("An interaction between these drugs already exists")
            
            drug1 = db.session.get(Drug, drug1_id)
            drug2 = db.session.get(Drug, drug2_id)
            
            if not drug1 or not drug2:
                raise ValueError("Invalid drug IDs")
            
            alternatives_str = ''
            if alternatives:
                alternative_drugs = Drug.query.filter(Drug.id.in_(alternatives)).all()
                alternatives_str = ', '.join([drug.name_en for drug in alternative_drugs])
            
            try:
                predicted_severity, prediction_confidence, predicted_severity_id = classify_severity(
                    interaction_description, drug1.name_en, drug2.name_en
                )
            except Exception as e:
                logger.error(f"Error in severity classification: {e}")
                predicted_severity = "Şiddetli"
                prediction_confidence = 0.0
                predicted_severity_id = 3

            new_interaction = DrugInteraction(
                drug1_id=drug1_id,
                drug2_id=drug2_id,
                interaction_type=interaction_type,
                interaction_description=interaction_description,
                severity_id=predicted_severity_id,
                reference=reference,
                mechanism=mechanism,
                monitoring=monitoring,
                alternatives=alternatives_str,
                predicted_severity_id=predicted_severity_id,
                prediction_confidence=prediction_confidence,
                processed=True
            )
            
            if route_ids:
                valid_routes = RouteOfAdministration.query.filter(RouteOfAdministration.id.in_(route_ids)).all()
                if len(valid_routes) != len(route_ids):
                    logger.warning(f"Some route IDs were invalid: requested={route_ids}, found={[r.id for r in valid_routes]}")
                new_interaction.routes = valid_routes
            
            db.session.add(new_interaction)
            db.session.commit()
            
            logger.info(
                f"Interaction created (ID: {new_interaction.id}) between {drug1.name_en} and {drug2.name_en}, "
                f"AI-Predicted Severity: {predicted_severity} (Confidence: {prediction_confidence:.2f})"
            )
            
            return redirect(url_for('manage_interactions'))
            
        except ValueError as ve:
            db.session.rollback()
            logger.error(f"Validation error: {str(ve)}")
            return render_template(
                'manage_interactions.html',
                drugs=drugs,
                routes=routes,
                severities=severities,
                interactions=interactions,
                error=str(ve)
            )
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error creating interaction: {str(e)}", exc_info=True)
            return render_template(
                'manage_interactions.html',
                drugs=drugs,
                routes=routes,
                severities=severities,
                interactions=interactions,
                error="An unexpected error occurred. Please try again."
            )
    
    return render_template(
        'manage_interactions.html',
        drugs=drugs,
        routes=routes,
        severities=severities,
        interactions=interactions
    )

@app.route('/interactions/update/<int:interaction_id>', methods=['GET', 'POST'])
@login_required
@admin_required
def update_interaction(interaction_id):
    interaction = DrugInteraction.query.options(
        db.joinedload(DrugInteraction.drug1),
        db.joinedload(DrugInteraction.drug2),
        db.joinedload(DrugInteraction.severity_level),
        db.joinedload(DrugInteraction.routes)
    ).get_or_404(interaction_id)
    
    drugs = Drug.query.order_by(Drug.name_en).all()
    routes = RouteOfAdministration.query.order_by(RouteOfAdministration.name).all()
    severities = Severity.query.order_by(Severity.id).all()
    
    if request.method == 'POST':
        try:
            drug1_id = request.form.get('drug1_id')
            drug2_id = request.form.get('drug2_id')
            route_ids = request.form.getlist('route_ids')
            interaction_type = request.form.get('interaction_type')
            interaction_description = request.form.get('interaction_description')
            reference = request.form.get('reference', '')
            mechanism = request.form.get('mechanism', '')
            monitoring = request.form.get('monitoring', '')
            alternatives = request.form.getlist('alternatives')
            
            if not drug1_id or not drug2_id or not interaction_type or not interaction_description:
                raise ValueError("Required fields are missing")
            
            drug1_id = int(drug1_id)
            drug2_id = int(drug2_id)
            
            if drug1_id == drug2_id:
                raise ValueError("Cannot create interaction between the same drug")
            
            route_ids = [int(route_id) for route_id in route_ids if route_id]
            
            existing = DrugInteraction.query.filter(
                DrugInteraction.id != interaction_id,
                or_(
                    and_(DrugInteraction.drug1_id == drug1_id, DrugInteraction.drug2_id == drug2_id),
                    and_(DrugInteraction.drug1_id == drug2_id, DrugInteraction.drug2_id == drug1_id)
                )
            ).first()
            
            if existing:
                raise ValueError("An interaction between these drugs already exists")
            
            drug1 = db.session.get(Drug, drug1_id)
            drug2 = db.session.get(Drug, drug2_id)
            
            if not drug1 or not drug2:
                raise ValueError("Invalid drug IDs")
            
            interaction.drug1_id = drug1_id
            interaction.drug2_id = drug2_id
            interaction.interaction_type = interaction_type
            interaction.interaction_description = interaction_description
            interaction.reference = reference
            interaction.mechanism = mechanism
            interaction.monitoring = monitoring
            
            if alternatives:
                alternative_drugs = Drug.query.filter(Drug.id.in_(alternatives)).all()
                interaction.alternatives = ', '.join([drug.name_en for drug in alternative_drugs])
            else:
                interaction.alternatives = ''
            
            try:
                predicted_severity, interaction.prediction_confidence, predicted_severity_id = classify_severity(
                    interaction_description, drug1.name_en, drug2.name_en
                )
                
                interaction.predicted_severity_id = predicted_severity_id
                interaction.severity_id = predicted_severity_id
                
            except Exception as e:
                logger.error(f"Error in severity classification: {e}")
                interaction.prediction_confidence = 0.0
                interaction.predicted_severity_id = 3
                interaction.severity_id = 3
            
            if route_ids:
                valid_routes = RouteOfAdministration.query.filter(RouteOfAdministration.id.in_(route_ids)).all()
                if len(valid_routes) != len(route_ids):
                    logger.warning(f"Some route IDs were invalid for interaction {interaction_id}")
                interaction.routes = valid_routes
            else:
                interaction.routes = []
            
            interaction.processed = True
            
            db.session.commit()
            
            logger.info(
                f"Interaction {interaction_id} updated: {drug1.name_en} + {drug2.name_en}, "
                f"AI-Predicted: {interaction.predicted_severity} "
                f"(Confidence: {interaction.prediction_confidence:.2f})"
            )
            
            return redirect(url_for('manage_interactions'))
            
        except ValueError as ve:
            db.session.rollback()
            logger.error(f"Validation error updating interaction {interaction_id}: {str(ve)}")
            selected_alternatives = []
            selected_route_ids = []
            return render_template(
                'update_interaction.html',
                interaction=interaction,
                drugs=drugs,
                routes=routes,
                severities=severities,
                selected_alternatives=selected_alternatives,
                selected_route_ids=selected_route_ids,
                error=str(ve)
            )
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error updating interaction {interaction_id}: {str(e)}", exc_info=True)
            selected_alternatives = []
            selected_route_ids = []
            return render_template(
                'update_interaction.html',
                interaction=interaction,
                drugs=drugs,
                routes=routes,
                severities=severities,
                selected_alternatives=selected_alternatives,
                selected_route_ids=selected_route_ids,
                error="An unexpected error occurred. Please try again."
            )
    
    selected_alternatives = []
    if interaction.alternatives:
        try:
            alternative_names = [name.strip() for name in interaction.alternatives.split(',') if name.strip()]
            alternative_drugs = Drug.query.filter(Drug.name_en.in_(alternative_names)).all()
            selected_alternatives = [
                {'id': drug.id, 'text': f"{drug.name_en} ({drug.name_tr})"}
                for drug in alternative_drugs
                if drug.name_en and drug.name_tr
            ]
        except Exception as e:
            logger.error(f"Error preparing selected_alternatives: {str(e)}")
    
    selected_route_ids = [route.id for route in interaction.routes] if interaction.routes else []
    
    return render_template(
        'update_interaction.html',
        interaction=interaction,
        drugs=drugs,
        routes=routes,
        severities=severities,
        selected_alternatives=selected_alternatives,
        selected_route_ids=selected_route_ids
    )

@app.route('/interactions/batch_process', methods=['POST'])
@login_required
@admin_required
def batch_process_interactions():
    try:
        batch_size = 50
        total_processed = 0
        total_errors = 0
        
        interactions = DrugInteraction.query.filter_by(processed=False).options(
            db.joinedload(DrugInteraction.drug1),
            db.joinedload(DrugInteraction.drug2),
            db.joinedload(DrugInteraction.severity_level)
        ).all()
        
        logger.info(f"Starting batch processing of {len(interactions)} interactions")
        
        for i in range(0, len(interactions), batch_size):
            batch = interactions[i:i + batch_size]
            batch_errors = 0
            
            for interaction in batch:
                try:
                    drug1 = interaction.drug1
                    drug2 = interaction.drug2
                    
                    if not drug1 or not drug2:
                        logger.error(f"Invalid drug references for interaction {interaction.id}")
                        batch_errors += 1
                        continue
                    
                    manual_severity = interaction.severity_level.name if interaction.severity_level else "Orta"
                    
                    predicted_severity, prediction_confidence, predicted_severity_id = classify_severity(
                        interaction.interaction_description, 
                        drug1.name_en, 
                        drug2.name_en, 
                        manual_severity
                    )

                    if predicted_severity != manual_severity:
                        severity_diff = abs(
                            ["Hafif", "Orta", "Şiddetli", "Hayati Risk İçeren"].index(predicted_severity) -
                            ["Hafif", "Orta", "Şiddetli", "Hayati Risk İçeren"].index(manual_severity)
                        )
                        
                        if severity_diff > 1:
                            logger.warning(
                                f"Significant mismatch for interaction {interaction.id} ({drug1.name_en} + {drug2.name_en}): "
                                f"Manual={manual_severity}, Predicted={predicted_severity} "
                                f"(Confidence={prediction_confidence:.2f})"
                            )

                    interaction.predicted_severity_id = predicted_severity_id
                    interaction.prediction_confidence = prediction_confidence
                    interaction.processed = True
                    
                except Exception as e:
                    logger.error(f"Error processing interaction {interaction.id}: {str(e)}")
                    batch_errors += 1
                    continue
            
            try:
                db.session.commit()
                total_processed += len(batch) - batch_errors
                total_errors += batch_errors
                logger.info(f"Batch {i//batch_size + 1}: Processed {len(batch) - batch_errors}/{len(batch)} interactions")
            except Exception as e:
                db.session.rollback()
                logger.error(f"Error committing batch {i//batch_size + 1}: {str(e)}")
                total_errors += len(batch)
            
            time.sleep(1)
        
        message = f"Batch processing completed: {total_processed} successful, {total_errors} errors"
        logger.info(message)
        
        return jsonify({
            "message": message,
            "processed": total_processed,
            "errors": total_errors
        }), 200
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Critical error in batch processing: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/interactions/delete/<int:interaction_id>', methods=['POST'])
def delete_interaction(interaction_id):
    interaction = DrugInteraction.query.get_or_404(interaction_id)
    db.session.delete(interaction)
    db.session.commit()
    return redirect(url_for('manage_interactions'))

@app.route('/interactions/delete_all', methods=['POST'])
@login_required
@admin_required
def delete_all_interactions():
    try:
        # Delete related records first
        db.session.execute(text("DELETE FROM public.interaction_route"))
        
        # Delete all interactions
        num_deleted = db.session.query(DrugInteraction).delete()
        
        # Reset the sequence to start from 1
        db.session.execute(text("ALTER SEQUENCE public.drug_interaction_id_seq RESTART WITH 1"))
        
        db.session.commit()
        
        logger.info(f"Deleted {num_deleted} interactions, their related routes, and reset ID sequence")
        return jsonify({
            "message": "All interactions deleted successfully and ID sequence reset",
            "count": num_deleted
        }), 200
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error deleting all interactions: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/interactions', methods=['POST'])
def get_interactions():
    draw = request.form.get('draw', type=int)
    start = request.form.get('start', type=int)
    length = request.form.get('length', type=int)
    search_value = request.form.get('search[value]', '')
    order_column_index = request.form.get('order[0][column]', type=int)
    order_direction = request.form.get('order[0][dir]', 'asc')
    
    drug1_alias = aliased(Drug, name='drug1')
    drug2_alias = aliased(Drug, name='drug2')
    
    column_map = {
        0: DrugInteraction.id,
        1: drug1_alias.name_en,
        2: drug2_alias.name_en,
        3: DrugInteraction.id,
        4: DrugInteraction.interaction_type,
        5: DrugInteraction.interaction_description,
        6: Severity.name,
        7: DrugInteraction.predicted_severity_id,  # ← THIS LINE IS THE PROBLEM
        8: DrugInteraction.mechanism,
        9: DrugInteraction.monitoring,
        10: DrugInteraction.alternatives,
        11: DrugInteraction.reference,
        12: DrugInteraction.id,
    }
    
    order_column = column_map.get(order_column_index, DrugInteraction.id)
    total_records = DrugInteraction.query.count()
    
    query = DrugInteraction.query \
        .join(drug1_alias, DrugInteraction.drug1_id == drug1_alias.id) \
        .join(drug2_alias, DrugInteraction.drug2_id == drug2_alias.id) \
        .join(Severity, DrugInteraction.severity_id == Severity.id)
    
    if search_value:
        query = query.filter(
            db.or_(
                drug1_alias.name_en.ilike(f"%{search_value}%"),
                drug2_alias.name_en.ilike(f"%{search_value}%"),
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
        routes_str = ', '.join([route.name for route in interaction.routes]) if interaction.routes else 'Genel'
        
        data.append({
            "id": interaction.id,
            "drug1_name": interaction.drug1.name_en if interaction.drug1 else "N/A",
            "drug2_name": interaction.drug2.name_en if interaction.drug2 else "N/A",
            "route": routes_str,
            "interaction_type": interaction.interaction_type,
            "interaction_description": interaction.interaction_description,
            "severity": interaction.severity_level.name if interaction.severity_level else "N/A",
            "predicted_severity": interaction.predicted_severity_level.name if interaction.predicted_severity_level else "Not Available",
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


# Add this route for bulk upload
@app.route('/interactions/bulk_upload', methods=['GET', 'POST'])
@login_required
def bulk_upload_interactions():
    if request.method == 'POST':
        try:
            if 'file' not in request.files:
                return jsonify({"error": "No file uploaded"}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400
            
            if not file.filename.endswith(('.xlsx', '.xls')):
                return jsonify({"error": "Invalid file format. Please upload Excel file"}), 400
            
            import pandas as pd
            import io
            
            df = pd.read_excel(io.BytesIO(file.read()))
            
            required_columns = ['drug1_id', 'drug2_id', 'interaction_description']
            if not all(col in df.columns for col in required_columns):
                return jsonify({"error": "Excel file must contain: drug1_id, drug2_id, interaction_description"}), 400
            
            total_rows = len(df)
            success_count = 0
            error_count = 0
            duplicate_count = 0
            errors = []
            
            # Get default severity (Moderate)
            default_severity = Severity.query.filter_by(name='Moderate').first()
            if not default_severity:
                return jsonify({"error": "Severity 'Moderate' not found in database"}), 500
            
            for index, row in df.iterrows():
                try:
                    drug1_id = int(row['drug1_id'])
                    drug2_id = int(row['drug2_id'])
                    interaction_description = str(row['interaction_description']).strip()
                    
                    if drug1_id == drug2_id:
                        error_count += 1
                        errors.append(f"Row {index + 2}: Cannot create interaction between same drug (ID: {drug1_id})")
                        continue
                    
                    drug1 = db.session.get(Drug, drug1_id)
                    drug2 = db.session.get(Drug, drug2_id)
                    
                    if not drug1:
                        error_count += 1
                        errors.append(f"Row {index + 2}: Drug1 ID {drug1_id} not found")
                        continue
                    
                    if not drug2:
                        error_count += 1
                        errors.append(f"Row {index + 2}: Drug2 ID {drug2_id} not found")
                        continue
                    
                    existing = DrugInteraction.query.filter(
                        or_(
                            and_(DrugInteraction.drug1_id == drug1_id, DrugInteraction.drug2_id == drug2_id),
                            and_(DrugInteraction.drug1_id == drug2_id, DrugInteraction.drug2_id == drug1_id)
                        )
                    ).first()
                    
                    if existing:
                        duplicate_count += 1
                        continue
                    
                    try:
                        predicted_severity, prediction_confidence = classify_severity(
                            interaction_description, drug1.name_en, drug2.name_en, default_severity.name
                        )
                    except Exception as e:
                        logger.error(f"Row {index + 2}: Severity classification error: {str(e)}")
                        predicted_severity = "Moderate"
                        prediction_confidence = 0.0
                    
                    new_interaction = DrugInteraction(
                        drug1_id=drug1_id,
                        drug2_id=drug2_id,
                        interaction_type='Pharmacodynamic',
                        interaction_description=interaction_description,
                        severity_id=default_severity.id,
                        predicted_severity=predicted_severity,
                        prediction_confidence=prediction_confidence,
                        processed=True
                    )
                    
                    db.session.add(new_interaction)
                    success_count += 1
                    
                    if success_count % 50 == 0:
                        db.session.commit()
                        logger.info(f"Committed batch: {success_count} interactions")
                    
                except Exception as e:
                    error_count += 1
                    errors.append(f"Row {index + 2}: {str(e)}")
                    continue
            
            db.session.commit()
            
            result = {
                "message": "Upload completed",
                "total": total_rows,
                "success": success_count,
                "duplicates": duplicate_count,
                "errors": error_count,
                "error_details": errors[:20]
            }
            
            logger.info(f"Bulk upload completed: {success_count}/{total_rows} successful, {duplicate_count} duplicates, {error_count} errors")
            
            return jsonify(result), 200
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Bulk upload error: {str(e)}", exc_info=True)
            return jsonify({"error": str(e)}), 500
    
    return render_template('bulk_upload_interactions.html')
#The END for interactions    

#Drug Food Interaction Section
# Manage Foods
@app.route('/foods/manage', methods=['GET', 'POST'])
@login_required
def manage_foods():
    if request.method == 'POST':
        name_en = request.form.get('name_en')
        name_tr = request.form.get('name_tr')
        category = request.form.get('category')
        description = request.form.get('description')
        
        new_food = Food(
            name_en=name_en,
            name_tr=name_tr,
            category=category,
            description=description
        )
        db.session.add(new_food)
        db.session.commit()
        flash('Food added successfully!', 'success')
        return redirect(url_for('manage_foods'))
    
    page = request.args.get('page', 1, type=int)
    per_page = 20
    foods = Food.query.paginate(page=page, per_page=per_page)
    return render_template('manage_foods.html', foods=foods)

@app.route('/foods/delete/<int:food_id>', methods=['POST'])
@login_required
def delete_food(food_id):
    food = Food.query.get_or_404(food_id)
    db.session.delete(food)
    db.session.commit()
    flash('Food deleted successfully!', 'success')
    return redirect(url_for('manage_foods'))

# Manage Drug-Food Interactions
@app.route('/drug-food-interactions/manage', methods=['GET', 'POST'])
@login_required
def manage_drug_food_interactions():
    drugs = Drug.query.order_by(Drug.name_en).all()
    foods = Food.query.order_by(Food.name_en).all()
    severities = Severity.query.order_by(Severity.id).all()
    
    if request.method == 'POST':
        try:
            drug_id = int(request.form.get('drug_id'))
            food_id = int(request.form.get('food_id'))
            interaction_type = request.form.get('interaction_type')
            description = request.form.get('description')
            severity_id = int(request.form.get('severity_id'))
            timing_instruction = request.form.get('timing_instruction')
            recommendation = request.form.get('recommendation')
            reference = request.form.get('reference')
            
            drug = Drug.query.get(drug_id)
            food = Food.query.get(food_id)
            severity = Severity.query.get(severity_id)
            
            if not drug or not food or not severity:
                raise ValueError("Invalid drug, food, or severity")
            
            # Check for duplicate
            existing = DrugFoodInteraction.query.filter_by(
                drug_id=drug_id, food_id=food_id
            ).first()
            
            if existing:
                flash('This drug-food interaction already exists!', 'warning')
                return redirect(url_for('manage_drug_food_interactions'))
            
            # Predict severity
            predicted_severity, confidence = classify_food_interaction_severity(
                description, drug.name_en, food.name_en, severity.name
            )
            
            new_interaction = DrugFoodInteraction(
                drug_id=drug_id,
                food_id=food_id,
                interaction_type=interaction_type,
                description=description,
                severity_id=severity_id,
                timing_instruction=timing_instruction,
                recommendation=recommendation,
                reference=reference,
                predicted_severity=predicted_severity,
                prediction_confidence=confidence
            )
            
            db.session.add(new_interaction)
            db.session.commit()
            flash('Drug-food interaction added successfully!', 'success')
            return redirect(url_for('manage_drug_food_interactions'))
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error adding drug-food interaction: {str(e)}")
            flash(f'Error: {str(e)}', 'error')
    
    page = request.args.get('page', 1, type=int)
    per_page = 10
    interactions = DrugFoodInteraction.query.options(
        db.joinedload(DrugFoodInteraction.drug),
        db.joinedload(DrugFoodInteraction.food),
        db.joinedload(DrugFoodInteraction.severity)
    ).paginate(page=page, per_page=per_page)
    
    return render_template('manage_drug_food_interactions.html',
                         drugs=drugs, foods=foods, severities=severities,
                         interactions=interactions)

@app.route('/drug-food-interactions/update/<int:interaction_id>', methods=['GET', 'POST'])
@login_required
def update_drug_food_interaction(interaction_id):
    interaction = DrugFoodInteraction.query.get_or_404(interaction_id)
    drugs = Drug.query.order_by(Drug.name_en).all()
    foods = Food.query.order_by(Food.name_en).all()
    severities = Severity.query.order_by(Severity.id).all()
    
    if request.method == 'POST':
        try:
            interaction.drug_id = int(request.form.get('drug_id'))
            interaction.food_id = int(request.form.get('food_id'))
            interaction.interaction_type = request.form.get('interaction_type')
            interaction.description = request.form.get('description')
            interaction.severity_id = int(request.form.get('severity_id'))
            interaction.timing_instruction = request.form.get('timing_instruction')
            interaction.recommendation = request.form.get('recommendation')
            interaction.reference = request.form.get('reference')
            
            drug = Drug.query.get(interaction.drug_id)
            food = Food.query.get(interaction.food_id)
            severity = Severity.query.get(interaction.severity_id)
            
            # Predict severity
            predicted_severity, confidence = classify_food_interaction_severity(
                interaction.description, drug.name_en, food.name_en, severity.name
            )
            
            interaction.predicted_severity = predicted_severity
            interaction.prediction_confidence = confidence
            
            db.session.commit()
            flash('Drug-food interaction updated successfully!', 'success')
            return redirect(url_for('manage_drug_food_interactions'))
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error updating drug-food interaction: {str(e)}")
            flash(f'Error: {str(e)}', 'error')
    
    return render_template('update_drug_food_interaction.html',
                         interaction=interaction, drugs=drugs, foods=foods,
                         severities=severities)

@app.route('/drug-food-interactions/delete/<int:interaction_id>', methods=['POST'])
@login_required
def delete_drug_food_interaction(interaction_id):
    interaction = DrugFoodInteraction.query.get_or_404(interaction_id)
    db.session.delete(interaction)
    db.session.commit()
    flash('Drug-food interaction deleted successfully!', 'success')
    return redirect(url_for('manage_drug_food_interactions'))

# Check Drug-Food Interactions
@app.route('/drug-food-interactions/check', methods=['GET', 'POST'])
@login_required
def check_drug_food_interactions():
    drugs = Drug.query.all()
    foods = Food.query.all()
    results = []
    
    if request.method == 'POST':
        drug_ids = request.form.getlist('drug_ids')
        food_ids = request.form.getlist('food_ids')
        
        for drug_id in drug_ids:
            for food_id in food_ids:
                interaction = DrugFoodInteraction.query.filter_by(
                    drug_id=int(drug_id), food_id=int(food_id)
                ).first()
                
                if interaction:
                    results.append({
                        'drug': interaction.drug.name_en,
                        'food': interaction.food.name_en,
                        'interaction_type': interaction.interaction_type,
                        'description': interaction.description,
                        'severity': interaction.severity.name,
                        'predicted_severity': interaction.predicted_severity,
                        'timing_instruction': interaction.timing_instruction,
                        'recommendation': interaction.recommendation,
                        'reference': interaction.reference
                    })
    
    return render_template('check_drug_food_interactions.html',
                         drugs=drugs, foods=foods, results=results)

# API endpoint for foods
# 1. Get all foods (with pagination and search)
@app.route('/api/foods', methods=['GET'])
def api_get_foods():
    try:
        search = request.args.get('q', '').strip()
        page = request.args.get('page', 1, type=int)
        limit = request.args.get('limit', 20, type=int)
        category = request.args.get('category', '').strip()
        
        query = Food.query
        
        if search:
            query = query.filter(
                or_(
                    Food.name_en.ilike(f'%{search}%'),
                    Food.name_tr.ilike(f'%{search}%'),
                    Food.description.ilike(f'%{search}%')
                )
            )
        
        if category:
            query = query.filter(Food.category == category)
        
        paginated = query.paginate(page=page, per_page=limit, error_out=False)
        
        foods = [{
            'id': food.id,
            'name_en': food.name_en,
            'name_tr': food.name_tr,
            'category': food.category,
            'description': food.description
        } for food in paginated.items]
        
        return jsonify({
            'success': True,
            'data': foods,
            'pagination': {
                'page': page,
                'per_page': limit,
                'total': paginated.total,
                'pages': paginated.pages,
                'has_next': paginated.has_next,
                'has_prev': paginated.has_prev
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error in api_get_foods: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


# 2. Get single food by ID
@app.route('/api/foods/<int:food_id>', methods=['GET'])
def api_get_food(food_id):
    try:
        food = Food.query.get_or_404(food_id)
        
        return jsonify({
            'success': True,
            'data': {
                'id': food.id,
                'name_en': food.name_en,
                'name_tr': food.name_tr,
                'category': food.category,
                'description': food.description
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error in api_get_food: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 404
    
# 3. Create new food (requires authentication)
@app.route('/api/foods', methods=['POST'])
@login_required
def api_create_food():
    try:
        data = request.get_json()
        
        if not data or not data.get('name_en'):
            return jsonify({
                'success': False,
                'error': 'name_en is required'
            }), 400
        
        new_food = Food(
            name_en=data.get('name_en'),
            name_tr=data.get('name_tr'),
            category=data.get('category'),
            description=data.get('description')
        )
        
        db.session.add(new_food)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Food created successfully',
            'data': {
                'id': new_food.id,
                'name_en': new_food.name_en,
                'name_tr': new_food.name_tr,
                'category': new_food.category,
                'description': new_food.description
            }
        }), 201
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error in api_create_food: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

# 4. Update food
@app.route('/api/foods/<int:food_id>', methods=['PUT'])
@login_required
def api_update_food(food_id):
    try:
        food = Food.query.get_or_404(food_id)
        data = request.get_json()
        
        if 'name_en' in data:
            food.name_en = data['name_en']
        if 'name_tr' in data:
            food.name_tr = data['name_tr']
        if 'category' in data:
            food.category = data['category']
        if 'description' in data:
            food.description = data['description']
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Food updated successfully',
            'data': {
                'id': food.id,
                'name_en': food.name_en,
                'name_tr': food.name_tr,
                'category': food.category,
                'description': food.description
            }
        }), 200
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error in api_update_food: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500
    
#5. Delete food
@app.route('/api/foods/<int:food_id>', methods=['DELETE'])
@login_required
def api_delete_food(food_id):
    try:
        food = Food.query.get_or_404(food_id)
        db.session.delete(food)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Food deleted successfully'
        }), 200
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error in api_delete_food: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


# 6. Get all drug-food interactions (with filters)
@app.route('/api/drug-food-interactions', methods=['GET'])
def api_get_drug_food_interactions():
    try:
        page = request.args.get('page', 1, type=int)
        limit = request.args.get('limit', 20, type=int)
        drug_id = request.args.get('drug_id', type=int)
        food_id = request.args.get('food_id', type=int)
        severity = request.args.get('severity', '').strip()
        interaction_type = request.args.get('interaction_type', '').strip()
        
        query = DrugFoodInteraction.query.options(
            db.joinedload(DrugFoodInteraction.drug),
            db.joinedload(DrugFoodInteraction.food),
            db.joinedload(DrugFoodInteraction.severity)
        )
        
        if drug_id:
            query = query.filter(DrugFoodInteraction.drug_id == drug_id)
        
        if food_id:
            query = query.filter(DrugFoodInteraction.food_id == food_id)
        
        if severity:
            query = query.join(Severity).filter(Severity.name.ilike(f'%{severity}%'))
        
        if interaction_type:
            query = query.filter(DrugFoodInteraction.interaction_type.ilike(f'%{interaction_type}%'))
        
        paginated = query.paginate(page=page, per_page=limit, error_out=False)
        
        interactions = [{
            'id': interaction.id,
            'drug': {
                'id': interaction.drug.id,
                'name_en': interaction.drug.name_en,
                'name_tr': interaction.drug.name_tr
            },
            'food': {
                'id': interaction.food.id,
                'name_en': interaction.food.name_en,
                'name_tr': interaction.food.name_tr,
                'category': interaction.food.category
            },
            'interaction_type': interaction.interaction_type,
            'description': interaction.description,
            'severity': interaction.severity.name,
            'predicted_severity': interaction.predicted_severity,
            'prediction_confidence': interaction.prediction_confidence,
            'timing_instruction': interaction.timing_instruction,
            'recommendation': interaction.recommendation,
            'reference': interaction.reference
        } for interaction in paginated.items]
        
        return jsonify({
            'success': True,
            'data': interactions,
            'pagination': {
                'page': page,
                'per_page': limit,
                'total': paginated.total,
                'pages': paginated.pages,
                'has_next': paginated.has_next,
                'has_prev': paginated.has_prev
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error in api_get_drug_food_interactions: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


# 7. Get single drug-food interaction
@app.route('/api/drug-food-interactions/<int:interaction_id>', methods=['GET'])
def api_get_drug_food_interaction(interaction_id):
    try:
        interaction = DrugFoodInteraction.query.options(
            db.joinedload(DrugFoodInteraction.drug),
            db.joinedload(DrugFoodInteraction.food),
            db.joinedload(DrugFoodInteraction.severity)
        ).get_or_404(interaction_id)
        
        return jsonify({
            'success': True,
            'data': {
                'id': interaction.id,
                'drug': {
                    'id': interaction.drug.id,
                    'name_en': interaction.drug.name_en,
                    'name_tr': interaction.drug.name_tr
                },
                'food': {
                    'id': interaction.food.id,
                    'name_en': interaction.food.name_en,
                    'name_tr': interaction.food.name_tr,
                    'category': interaction.food.category
                },
                'interaction_type': interaction.interaction_type,
                'description': interaction.description,
                'severity': interaction.severity.name,
                'predicted_severity': interaction.predicted_severity,
                'prediction_confidence': interaction.prediction_confidence,
                'timing_instruction': interaction.timing_instruction,
                'recommendation': interaction.recommendation,
                'reference': interaction.reference
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error in api_get_drug_food_interaction: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 404

# 8. Create drug-food interaction
@app.route('/api/drug-food-interactions', methods=['POST'])
@login_required
def api_create_drug_food_interaction():
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['drug_id', 'food_id', 'interaction_type', 'description', 'severity_id']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'{field} is required'
                }), 400
        
        drug_id = int(data['drug_id'])
        food_id = int(data['food_id'])
        severity_id = int(data['severity_id'])
        
        # Check if entities exist
        drug = Drug.query.get(drug_id)
        food = Food.query.get(food_id)
        severity = Severity.query.get(severity_id)
        
        if not drug or not food or not severity:
            return jsonify({
                'success': False,
                'error': 'Invalid drug_id, food_id, or severity_id'
            }), 400
        
        # Check for duplicate
        existing = DrugFoodInteraction.query.filter_by(
            drug_id=drug_id, food_id=food_id
        ).first()
        
        if existing:
            return jsonify({
                'success': False,
                'error': 'This drug-food interaction already exists'
            }), 409
        
        # Predict severity
        predicted_severity, confidence = classify_food_interaction_severity(
            data['description'], drug.name_en, food.name_en, severity.name
        )
        
        new_interaction = DrugFoodInteraction(
            drug_id=drug_id,
            food_id=food_id,
            interaction_type=data['interaction_type'],
            description=data['description'],
            severity_id=severity_id,
            timing_instruction=data.get('timing_instruction'),
            recommendation=data.get('recommendation'),
            reference=data.get('reference'),
            predicted_severity=predicted_severity,
            prediction_confidence=confidence
        )
        
        db.session.add(new_interaction)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Drug-food interaction created successfully',
            'data': {
                'id': new_interaction.id,
                'drug_id': new_interaction.drug_id,
                'food_id': new_interaction.food_id,
                'interaction_type': new_interaction.interaction_type,
                'severity': severity.name,
                'predicted_severity': predicted_severity,
                'prediction_confidence': confidence
            }
        }), 201
        
    except ValueError as e:
        return jsonify({'success': False, 'error': 'Invalid data format'}), 400
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error in api_create_drug_food_interaction: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

# 9. Update drug-food interaction
@app.route('/api/drug-food-interactions/<int:interaction_id>', methods=['PUT'])
@login_required
def api_update_drug_food_interaction(interaction_id):
    try:
        interaction = DrugFoodInteraction.query.get_or_404(interaction_id)
        data = request.get_json()
        
        if 'drug_id' in data:
            interaction.drug_id = int(data['drug_id'])
        if 'food_id' in data:
            interaction.food_id = int(data['food_id'])
        if 'interaction_type' in data:
            interaction.interaction_type = data['interaction_type']
        if 'description' in data:
            interaction.description = data['description']
        if 'severity_id' in data:
            interaction.severity_id = int(data['severity_id'])
        if 'timing_instruction' in data:
            interaction.timing_instruction = data['timing_instruction']
        if 'recommendation' in data:
            interaction.recommendation = data['recommendation']
        if 'reference' in data:
            interaction.reference = data['reference']
        
        # Re-predict severity if description changed
        if 'description' in data:
            drug = Drug.query.get(interaction.drug_id)
            food = Food.query.get(interaction.food_id)
            severity = Severity.query.get(interaction.severity_id)
            
            predicted_severity, confidence = classify_food_interaction_severity(
                interaction.description, drug.name_en, food.name_en, severity.name
            )
            
            interaction.predicted_severity = predicted_severity
            interaction.prediction_confidence = confidence
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Drug-food interaction updated successfully',
            'data': {
                'id': interaction.id,
                'predicted_severity': interaction.predicted_severity,
                'prediction_confidence': interaction.prediction_confidence
            }
        }), 200
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error in api_update_drug_food_interaction: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


# 10. Delete drug-food interaction
@app.route('/api/drug-food-interactions/<int:interaction_id>', methods=['DELETE'])
@login_required
def api_delete_drug_food_interaction(interaction_id):
    try:
        interaction = DrugFoodInteraction.query.get_or_404(interaction_id)
        db.session.delete(interaction)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Drug-food interaction deleted successfully'
        }), 200
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error in api_delete_drug_food_interaction: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


# 11. Check interactions (POST with body)
@app.route('/api/drug-food-interactions/check', methods=['POST'])
def api_check_drug_food_interactions():
    try:
        data = request.get_json()
        
        drug_ids = data.get('drug_ids', [])
        food_ids = data.get('food_ids', [])
        
        if not drug_ids or not food_ids:
            return jsonify({
                'success': False,
                'error': 'drug_ids and food_ids are required'
            }), 400
        
        results = []
        
        for drug_id in drug_ids:
            for food_id in food_ids:
                interaction = DrugFoodInteraction.query.filter_by(
                    drug_id=int(drug_id), food_id=int(food_id)
                ).options(
                    db.joinedload(DrugFoodInteraction.drug),
                    db.joinedload(DrugFoodInteraction.food),
                    db.joinedload(DrugFoodInteraction.severity)
                ).first()
                
                if interaction:
                    results.append({
                        'drug': {
                            'id': interaction.drug.id,
                            'name_en': interaction.drug.name_en,
                            'name_tr': interaction.drug.name_tr
                        },
                        'food': {
                            'id': interaction.food.id,
                            'name_en': interaction.food.name_en,
                            'name_tr': interaction.food.name_tr,
                            'category': interaction.food.category
                        },
                        'interaction_type': interaction.interaction_type,
                        'description': interaction.description,
                        'severity': interaction.severity.name,
                        'predicted_severity': interaction.predicted_severity,
                        'timing_instruction': interaction.timing_instruction,
                        'recommendation': interaction.recommendation,
                        'reference': interaction.reference
                    })
        
        return jsonify({
            'success': True,
            'interactions_found': len(results),
            'data': results
        }), 200
        
    except Exception as e:
        logger.error(f"Error in api_check_drug_food_interactions: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


# 12. Get food categories
@app.route('/api/food-categories', methods=['GET'])
def api_get_food_categories():
    try:
        categories = db.session.query(Food.category).distinct().filter(Food.category.isnot(None)).all()
        category_list = [cat[0] for cat in categories]
        
        return jsonify({
            'success': True,
            'data': sorted(category_list)
        }), 200
        
    except Exception as e:
        logger.error(f"Error in api_get_food_categories: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


# 13. Bulk create foods
@app.route('/api/foods/bulk', methods=['POST'])
@login_required
def api_bulk_create_foods():
    try:
        data = request.get_json()
        foods_data = data.get('foods', [])
        
        if not foods_data:
            return jsonify({
                'success': False,
                'error': 'foods array is required'
            }), 400
        
        created = []
        errors = []
        
        for idx, food_data in enumerate(foods_data):
            try:
                if not food_data.get('name_en'):
                    errors.append(f"Row {idx}: name_en is required")
                    continue
                
                new_food = Food(
                    name_en=food_data.get('name_en'),
                    name_tr=food_data.get('name_tr'),
                    category=food_data.get('category'),
                    description=food_data.get('description')
                )
                
                db.session.add(new_food)
                db.session.flush()
                created.append(new_food.id)
                
            except Exception as e:
                errors.append(f"Row {idx}: {str(e)}")
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'Created {len(created)} foods',
            'created_ids': created,
            'errors': errors
        }), 201
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error in api_bulk_create_foods: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


# 14. Get interactions by drug
@app.route('/api/drugs/<int:drug_id>/food-interactions', methods=['GET'])
def api_get_drug_food_interactions_by_drug(drug_id):
    try:
        drug = Drug.query.get_or_404(drug_id)
        
        interactions = DrugFoodInteraction.query.filter_by(drug_id=drug_id).options(
            db.joinedload(DrugFoodInteraction.food),
            db.joinedload(DrugFoodInteraction.severity)
        ).all()
        
        results = [{
            'id': interaction.id,
            'food': {
                'id': interaction.food.id,
                'name_en': interaction.food.name_en,
                'name_tr': interaction.food.name_tr,
                'category': interaction.food.category
            },
            'interaction_type': interaction.interaction_type,
            'description': interaction.description,
            'severity': interaction.severity.name,
            'predicted_severity': interaction.predicted_severity,
            'timing_instruction': interaction.timing_instruction,
            'recommendation': interaction.recommendation
        } for interaction in interactions]
        
        return jsonify({
            'success': True,
            'drug': {
                'id': drug.id,
                'name_en': drug.name_en,
                'name_tr': drug.name_tr
            },
            'interactions_count': len(results),
            'data': results
        }), 200
        
    except Exception as e:
        logger.error(f"Error in api_get_drug_food_interactions_by_drug: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 404            

# Helper function for severity classification for FOOD - DRUG
def classify_food_interaction_severity(description, drug_name, food_name, manual_severity):
    """Classify drug-food interaction severity using AI model."""
    labels = ["Mild", "Moderate", "Severe", "Critical"]
    label_map = {"Mild": "Hafif", "Moderate": "Orta", "Severe": "Şiddetli", "Critical": "Hayati Risk İçeren"}
    
    critical_terms = [
        "contraindicated", "avoid completely", "life-threatening", "severe toxicity",
        "dangerous", "fatal", "toxic levels", "critical"
    ]
    
    severe_terms = [
        "significantly reduced", "major decrease", "increased toxicity",
        "substantially affects", "serious risk", "avoid", "warning"
    ]
    
    description_lower = description.lower()
    is_critical = any(term in description_lower for term in critical_terms)
    is_severe = any(term in description_lower for term in severe_terms)
    
    prompt = (
        f"Analyze the drug-food interaction between {drug_name} and {food_name}.\n\n"
        f"Description: {description}\n\n"
        f"Classification Guidelines:\n"
        f"- Mild: Minor effect, no specific timing needed\n"
        f"- Moderate: Clinically relevant, timing recommendations apply\n"
        f"- Severe: Significant impact on efficacy or safety, strict avoidance or timing required\n"
        f"- Critical: Dangerous combination, must be completely avoided\n\n"
        f"Classify this interaction into ONE category."
    )
    
    try:
        clf = get_classifier()
        result = clf(prompt, labels, multi_label=False)
        predicted_label = label_map[result["labels"][0]]
        confidence = result["scores"][0]
        
        if is_critical:
            predicted_label = "Hayati Risk İçeren"
            confidence = max(confidence, 0.95)
        elif is_severe and predicted_label in ["Hafif", "Orta"]:
            predicted_label = "Şiddetli"
            confidence = max(confidence, 0.85)
        
        if confidence < 0.75:
            severity_order = ["Hafif", "Orta", "Şiddetli", "Hayati Risk İçeren"]
            current_index = severity_order.index(predicted_label)
            if current_index < len(severity_order) - 1:
                predicted_label = severity_order[current_index + 1]
            confidence = 0.75
        
    except Exception as e:
        logger.error(f"Error classifying food interaction severity: {e}")
        predicted_label = "Orta"
        confidence = 0.0
    
    return predicted_label, confidence


# Updated /cdss/advanced Route
@app.route('/cdss/advanced', methods=['GET', 'POST'])
@login_required
def cdss_advanced():
    drugs = Drug.query.all()
    routes = RouteOfAdministration.query.all()
    indications = Indication.query.all()
    lab_tests = LabTest.query.all()
    foods = Food.query.all()
    interaction_results = None
    error = None
    if request.method == 'POST':
        try:
            age = request.form.get('age', type=int, default=30)
            weight = request.form.get('weight', type=float, default=70.0)
            crcl = request.form.get('crcl', type=float, default=None)
            serum_creatinine = request.form.get('serum_creatinine', type=float, default=1.0)  # New: Allow user to input serum creatinine
            gender = request.form.get('gender', default='M')
            pregnancy = request.form.get('pregnancy', default='no')  # New: Pregnancy status (yes/no)
            hepatic_impairment = request.form.get('hepatic_impairment', default='none')  # New: Hepatic impairment level (none/mild/moderate/severe)
            selected_drugs = request.form.getlist('drugs')
            selected_indications = request.form.getlist('indications')
            selected_lab_tests = request.form.getlist('lab_tests')
            selected_foods = request.form.getlist('foods')  # Add this
            selected_route = request.form.get('route_id')
            route_id = int(selected_route) if selected_route else None
            logger.debug(f"Form data: drugs={selected_drugs}, indications={selected_indications}, "
                        f"lab_tests={selected_lab_tests}, route={selected_route}, age={age}, "
                        f"weight={weight}, crcl={crcl}, serum_creatinine={serum_creatinine}, gender={gender}, "
                        f"pregnancy={pregnancy}, hepatic_impairment={hepatic_impairment}")
            # Enhanced input validation
            if age < 0 or age > 120 or weight <= 0 or weight > 300 or (crcl is not None and (crcl < 0 or crcl > 200)) or serum_creatinine <= 0:
                logger.warning("Invalid input detected")
                error = "Invalid input—check age, weight, CrCl, or serum creatinine!"
            elif not selected_drugs:
                error = "At least one drug must be selected!"
            elif pregnancy not in ['yes', 'no']:
                error = "Invalid pregnancy status!"
            elif hepatic_impairment not in ['none', 'mild', 'moderate', 'severe']:
                error = "Invalid hepatic impairment level!"
            if error:
                return render_template('cdss_advanced.html', drugs=drugs, routes=routes,
                                     indications=indications, lab_tests=lab_tests,
                                     interaction_results=None, error=error)
            # Calculate CrCl if not provided, using provided serum_creatinine
            if crcl is None and age and weight and gender and serum_creatinine:
                crcl = calculate_crcl(age, weight, gender, serum_creatinine)
            # New: Polypharmacy warning if more than 5 drugs selected
            if len(selected_drugs) > 5:
                logger.warning("Polypharmacy detected: >5 drugs selected")
                # This could be added to results later, but for now, log it
            # Analyze all interactions with new parameters
            interaction_results = analyze_interactions(
                selected_drugs=selected_drugs,
                route_id=route_id,
                age=age,
                weight=weight,
                crcl=crcl,
                conditions=[int(i) for i in selected_indications if i],
                lab_tests=[int(t) for t in selected_lab_tests if t],
                pregnancy=pregnancy == 'yes',
                hepatic_impairment=hepatic_impairment,
                selected_foods=selected_foods
            )
            logger.debug(f"Interaction results: {len(interaction_results)} found")
        except ValueError as e:
            db.session.rollback()
            logger.error(f"Invalid input: {str(e)}")
            error = "Invalid input format—check IDs and values."
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error processing CDSS request: {str(e)}")
            error = "An error occurred while processing your request."
    return render_template('cdss_advanced.html', drugs=drugs, routes=routes,
                         indications=indications, lab_tests=lab_tests, foods=foods,
                         interaction_results=interaction_results, error=error)
# Helper Functions
def calculate_crcl(age, weight, gender, serum_creatinine=1.0):
    """Calculate CrCl using Cockcroft-Gault formula."""
    factor = 0.85 if gender.upper() == 'F' else 1.0
    crcl_value = ((140 - age) * weight * factor) / (72 * serum_creatinine)
    # Enhanced: Add bounds to prevent unrealistic values
    return max(10, min(200, crcl_value))  # Clamp between 10-200 mL/min
def adjust_severity(base_severity, age, crcl, pregnancy, hepatic_impairment, boost=0):
    """Adjust severity based on patient factors. Enhanced with pregnancy and hepatic impairment."""
    severity_map = {'Hafif': 1, 'Moderate': 2, 'Severe': 3, 'Critical': 4}
    score = severity_map.get(base_severity, 2) + boost
    if age > 75 or crcl < 30:
        score += 1
    elif age < 18:
        score += 0.5
    if pregnancy:
        score += 1.5  # Boost for pregnancy
    if hepatic_impairment == 'mild':
        score += 0.5
    elif hepatic_impairment == 'moderate':
        score += 1
    elif hepatic_impairment == 'severe':
        score += 1.5
    # Enhanced: Map back to levels with new 'Critical' option
    if score >= 4:
        return "Critical"
    elif score >= 3:
        return "Severe"
    elif score >= 2:
        return "Moderate"
    return "Hafif"
def simulate_pk_interaction(detail1, detail2, interaction, age, weight, crcl, pregnancy, hepatic_impairment):
    """Simulate pharmacokinetic interaction risk profile using a one-compartment model with absorption."""
    half_life1 = detail1.half_life or 4.0
    half_life2 = detail2.half_life or 4.0
    clearance1 = detail1.clearance_rate or 100.0
    clearance2 = detail2.clearance_rate or 100.0
    bioavail1 = detail1.bioavailability or 1.0
    bioavail2 = detail2.bioavailability or 1.0
    # Assume absorption rate constants (ka) for oral administration simulation
    ka1 = 0.5  # 1/hr, arbitrary value for demonstration
    ka2 = 0.5
    # Elimination rate constants
    ke1 = np.log(2) / half_life1
    ke2 = np.log(2) / half_life2
    # Adjust for age
    if age > 65:
        clearance1 *= 0.75
        clearance2 *= 0.75
    elif age < 18:
        clearance1 *= 1.2
        clearance2 *= 1.2
    # Adjust for renal function
    if crcl:
        renal_factor = min(crcl / 100, 1.0)
        clearance1 *= renal_factor
        clearance2 *= renal_factor
    # New: Adjust for hepatic impairment
    hepatic_factor = 1.0
    if hepatic_impairment == 'mild':
        hepatic_factor = 0.9
    elif hepatic_impairment == 'moderate':
        hepatic_factor = 0.7
    elif hepatic_impairment == 'severe':
        hepatic_factor = 0.5
    clearance1 *= hepatic_factor
    clearance2 *= hepatic_factor
    # New: Adjust for pregnancy (increased volume of distribution)
    if pregnancy:
        clearance1 *= 1.2
        clearance2 *= 1.2
    # Adjust for weight (assuming volume scales with weight)
    if weight:
        volume_factor = weight / 70.0
    else:
        volume_factor = 1.0
    # Time points
    time_points = np.linspace(0, 24, 100)
    # Simulate concentrations using Bateman function for oral absorption: C(t) = (ka * dose * F / V * (ke - ka)) * (exp(-ka*t) - exp(-ke*t))
    # Simplified without dose/V (normalized)
    conc1 = bioavail1 * ka1 / (ke1 - ka1) * (np.exp(-ka1 * time_points) - np.exp(-ke1 * time_points)) * clearance1 / volume_factor
    conc2 = bioavail2 * ka2 / (ke2 - ka2) * (np.exp(-ka2 * time_points) - np.exp(-ke2 * time_points)) * clearance2 / volume_factor
    # Handle cases where ka == ke
    conc1[np.isnan(conc1)] = 0
    conc2[np.isnan(conc2)] = 0
    base_risk = {'Hafif': 1, 'Moderate': 2, 'Severe': 3, 'Critical': 4}.get(interaction.severity_level.name, 1)
    risk_profile = base_risk * conc1 * conc2 * (1 if crcl > 30 else 1.5) * (1.2 if pregnancy else 1.0)
    # Enhanced: Add noise for more realistic simulation
    risk_profile += np.random.normal(0, 0.1 * base_risk, len(time_points))
    return time_points, risk_profile

def suggest_alternatives(drug_id, top_k=3):
    """Suggest alternative drugs using cosine similarity on pharmacokinetic features (AI/ML-based)."""
    drug_detail = DrugDetail.query.filter_by(drug_id=drug_id).first()
    if not drug_detail:
        return []
    # Feature vector: half_life, clearance_rate, bioavailability
    features = np.array([
        drug_detail.half_life or 0,
        drug_detail.clearance_rate or 0,
        drug_detail.bioavailability or 0
    ])
    if np.all(features == 0):
        return []
    all_drugs = Drug.query.all()
    similarities = []
    for other_drug in all_drugs:
        if other_drug.id == drug_id:
            continue
        other_detail = DrugDetail.query.filter_by(drug_id=other_drug.id).first()
        if not other_detail:
            continue
        other_features = np.array([
            other_detail.half_life or 0,
            other_detail.clearance_rate or 0,
            other_detail.bioavailability or 0
        ])
        if np.all(other_features == 0):
            continue
        # Cosine similarity
        sim = np.dot(features, other_features) / (np.linalg.norm(features) * np.linalg.norm(other_features))
        similarities.append((other_drug.name_en, sim))
    # Sort by similarity descending
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_alternatives = [name for name, sim in similarities[:top_k]]
    return top_alternatives

def analyze_interactions(selected_drugs, route_id, age, weight, crcl, conditions, lab_tests, pregnancy, hepatic_impairment, selected_foods=None):
    """Analyze drug-drug, drug-disease, and drug-lab test interactions. Enhanced with new patient factors, polypharmacy check, and AI-suggested alternatives."""
    results = []
    drug_ids = [int(d) for d in selected_drugs if d]
    food_ids = [int(f) for f in selected_foods if selected_foods and f] if selected_foods else []

    severity_map_db_to_func = {'Hafif': 'Hafif', 'Orta': 'Moderate', 'Şiddetli': 'Severe', 'Kritik': 'Critical', 'Hayati Risk İçeren': 'Critical'}
    severity_order = {"Critical": 4, "Severe": 3, "Moderate": 2, "Hafif": 1}
    # New: Polypharmacy alert if >5 drugs
    if len(drug_ids) > 5:
        poly_boost = 1 if len(drug_ids) > 10 else 0.5
        results.append({
            'type': 'Polypharmacy Alert',
            'drug1': 'Multiple Drugs',
            'drug2': f"{len(drug_ids)} drugs selected",
            'route': "N/A",
            'interaction_type': "High Risk",
            'description': "Polypharmacy increases risk of interactions. Review regimen carefully.",
            'severity': "Moderate",
            'predicted_severity': adjust_severity("Moderate", age, crcl, pregnancy, hepatic_impairment, boost=poly_boost),
            'peak_risk': 0,
            'risk_profile': [(0, 3)],
            'mechanism': "Cumulative Interactions",
            'monitoring': "Monitor for adverse effects; deprescribe if possible",
            'alternatives': "Simplify regimen",
            'reference': "Clinical guideline on polypharmacy",
            'evidence_level': "High"
        })
    # 1. Drug-Drug Interactions
    for drug1_id, drug2_id in combinations(drug_ids, 2):
        query = DrugInteraction.query.filter(
            or_(
                and_(DrugInteraction.drug1_id == drug1_id, DrugInteraction.drug2_id == drug2_id),
                and_(DrugInteraction.drug1_id == drug2_id, DrugInteraction.drug2_id == drug1_id)
            )
        )
        if route_id is not None:
            query = query.outerjoin(
                interaction_route,
                DrugInteraction.id == interaction_route.c.interaction_id
            ).filter(
                or_(interaction_route.c.route_id == route_id, interaction_route.c.route_id.is_(None))
            ).distinct()
        interactions = query.all()
        for interaction in interactions:
            drug1 = Drug.query.get(drug1_id)
            drug2 = Drug.query.get(drug2_id)
            detail1 = DrugDetail.query.filter_by(drug_id=drug1_id).first() or DrugDetail()
            detail2 = DrugDetail.query.filter_by(drug_id=drug2_id).first() or DrugDetail()
            time_points, risk_profile = simulate_pk_interaction(detail1, detail2, interaction, age, weight, crcl, pregnancy, hepatic_impairment)
            base_severity = severity_map_db_to_func.get(interaction.severity_level.name, 'Moderate')
            alternatives = interaction.alternatives or "Not Provided"
            if alternatives == "Not Provided":
                suggested = suggest_alternatives(drug2_id)
                if suggested:
                    alternatives = ', '.join(suggested) + " (AI suggested)"
            result = {
                'type': 'Drug-Drug Interaction',
                'drug1': drug1.name_en,
                'drug2': drug2.name_en,
                'route': RouteOfAdministration.query.get(route_id).name if route_id else "General",
                'interaction_type': interaction.interaction_type,
                'description': interaction.interaction_description,
                'severity': base_severity,
                'predicted_severity': adjust_severity(base_severity, age, crcl, pregnancy, hepatic_impairment),
                'peak_risk': interaction.time_to_peak or time_points[np.argmax(risk_profile)],
                'risk_profile': list(zip(time_points, risk_profile)),
                'mechanism': interaction.mechanism or "Not Provided",
                'monitoring': interaction.monitoring or "Not Provided",
                'alternatives': alternatives,
                'reference': interaction.reference or "Not Provided",
                'evidence_level': "High" if interaction.prediction_confidence > 0.9 else "Medium"  # New: Add evidence level based on confidence
            }
            results.append(result)
    # 2. Drug-Disease Interactions
    for drug_id in drug_ids:
        for condition_id in conditions:
            interaction = DrugDiseaseInteraction.query.filter_by(
                drug_id=drug_id, indication_id=condition_id
            ).first()
            if interaction:
                drug = Drug.query.get(drug_id)
                condition = Indication.query.get(condition_id)
                base_severity = severity_map_db_to_func.get(interaction.severity, 'Moderate')
                # Enhanced: Additional boost if pregnancy or hepatic issues relevant to condition
                extra_boost = 1 if (pregnancy and "pregnancy" in condition.name_en.lower()) else 0
                extra_boost += 1 if (hepatic_impairment != 'none' and "liver" in condition.name_en.lower()) else 0
                alternatives = "Consider alternative therapy"
                suggested = suggest_alternatives(drug_id)
                if suggested:
                    alternatives = ', '.join(suggested) + " (AI suggested)"
                result = {
                    'type': 'Drug-Disease Interaction',
                    'drug1': drug.name_en,
                    'drug2': condition.name_en,
                    'route': "N/A",
                    'interaction_type': interaction.interaction_type,
                    'description': interaction.description or "No description provided",
                    'severity': base_severity,
                    'predicted_severity': adjust_severity(base_severity, age, crcl, pregnancy, hepatic_impairment, boost=extra_boost),
                    'peak_risk': 0,
                    'risk_profile': [(0, 2)],
                    'mechanism': "Drug-Disease Interaction",
                    'monitoring': interaction.recommendation or "Not Provided",
                    'alternatives': alternatives,
                    'reference': "Database entry",
                    'evidence_level': "Medium"
                }
                results.append(result)
    # 3. Drug-Lab Test Interactions
    for drug_id in drug_ids:
        for lab_test_id in lab_tests:
            interaction = DrugLabTestInteraction.query.filter_by(
                drug_id=drug_id, lab_test_id=lab_test_id
            ).first()
            if interaction:
                drug = Drug.query.get(drug_id)
                lab_test = LabTest.query.get(lab_test_id)
                base_severity = severity_map_db_to_func.get(interaction.severity, 'Moderate')
                alternatives = "Use alternative lab test if available"
                suggested = suggest_alternatives(drug_id)
                if suggested:
                    alternatives = ', '.join(suggested) + " (AI suggested for drug replacement)"
                result = {
                    'type': 'Drug-Lab Test Interaction',
                    'drug1': drug.name_en,
                    'drug2': lab_test.name_en,
                    'route': "N/A",
                    'interaction_type': interaction.interaction_type,
                    'description': interaction.description or "No description provided",
                    'severity': base_severity,
                    'predicted_severity': adjust_severity(base_severity, age, crcl, pregnancy, hepatic_impairment),
                    'peak_risk': 0,
                    'risk_profile': [(0, 2)],
                    'mechanism': "Drug-Lab Test Interference",
                    'monitoring': interaction.recommendation or "Confirm with alternative test",
                    'alternatives': alternatives,
                    'reference': "Database entry",
                    'evidence_level': "Medium"
                }
                results.append(result)
    # 4. Condition Risk Checks (Enhanced with more risks and patient factors)
    condition_risks = {
        "renal failure": {"severity_boost": 1.5 if crcl < 30 else 1, "desc": "Worsened by renal impairment", "monitoring": "Monitor renal function"},
        "liver disease": {"severity_boost": 1.5 if hepatic_impairment != 'none' else 1, "desc": "Risk of hepatotoxicity", "monitoring": "Check LFTs"},
        "heart failure": {"severity_boost": 0.5, "desc": "May exacerbate fluid retention", "monitoring": "Monitor BP and weight"},
        "pregnancy": {"severity_boost": 2 if pregnancy else 0, "desc": "Potential teratogenic effects", "monitoring": "Ultrasound monitoring if applicable"},
        "hypertension": {"severity_boost": 0.5, "desc": "May affect blood pressure control", "monitoring": "Regular BP checks"},
        "diabetes": {"severity_boost": 0.5, "desc": "Risk of glucose dysregulation", "monitoring": "Monitor blood glucose"}
        # Add more as needed based on common conditions
    }
    for drug_id in drug_ids:
        drug = Drug.query.get(drug_id)
        for condition_id in conditions:
            condition = Indication.query.get(condition_id)
            if condition and condition.name_en.lower() in condition_risks:
                risk_info = condition_risks[condition.name_en.lower()]
                base_severity = "Moderate"
                adjusted_severity = adjust_severity(base_severity, age, crcl, pregnancy, hepatic_impairment, boost=risk_info["severity_boost"])
                alternatives = "Consider alternative therapy"
                suggested = suggest_alternatives(drug_id)
                if suggested:
                    alternatives = ', '.join(suggested) + " (AI suggested)"
                result = {
                    'type': 'Drug-Condition Alert',
                    'drug1': drug.name_en,
                    'drug2': condition.name_en,
                    'route': "N/A",
                    'interaction_type': "Potential Risk",
                    'description': f"Caution: {drug.name_en} with {condition.name_en}. {risk_info['desc']}",
                    'severity': base_severity,
                    'predicted_severity': adjusted_severity,
                    'peak_risk': 0,
                    'risk_profile': [(0, 2 + risk_info["severity_boost"])],
                    'mechanism': "Condition-Drug Interaction",
                    'monitoring': risk_info["monitoring"],
                    'alternatives': alternatives,
                    'reference': "Clinical guideline",
                    'evidence_level': "High"
                }
                results.append(result)

    # Add Drug-Food Interactions
    if food_ids:
        for drug_id in drug_ids:
            for food_id in food_ids:
                interaction = DrugFoodInteraction.query.filter_by(
                    drug_id=drug_id, food_id=food_id
                ).first()
                
                if interaction:
                    drug = Drug.query.get(drug_id)
                    food = Food.query.get(food_id)
                    base_severity = severity_map_db_to_func.get(interaction.severity.name, 'Moderate')
                    
                    result = {
                        'type': 'Drug-Food Interaction',
                        'drug1': drug.name_en,
                        'drug2': food.name_en,
                        'route': "N/A",
                        'interaction_type': interaction.interaction_type,
                        'description': interaction.description,
                        'severity': base_severity,
                        'predicted_severity': adjust_severity(base_severity, age, crcl, pregnancy, hepatic_impairment),
                        'peak_risk': 0,
                        'risk_profile': [(0, 2)],
                        'mechanism': f"Food Interaction - {interaction.interaction_type}",
                        'monitoring': interaction.timing_instruction or "Follow timing instructions",
                        'alternatives': interaction.recommendation or "Not Provided",
                        'reference': interaction.reference or "Database entry",
                        'evidence_level': "High" if interaction.prediction_confidence and interaction.prediction_confidence > 0.9 else "Medium"
                    }
                    results.append(result)

    # New: Add summary at the beginning
    if results:
        total_interactions = len(results)
        max_severity = max([severity_order.get(r['predicted_severity'], 0) for r in results])
        max_severity_label = next(k for k, v in severity_order.items() if v == max_severity)
        overall_risk_score = sum([severity_order.get(r['predicted_severity'], 0) for r in results]) / total_interactions if total_interactions > 0 else 0
        summary = {
            'type': 'Summary',
            'drug1': 'Overall',
            'drug2': 'Regimen',
            'route': "N/A",
            'interaction_type': "Overview",
            'description': f"Total interactions/alerts: {total_interactions}. Highest severity: {max_severity_label}. Overall risk score: {overall_risk_score:.2f}/4.",
            'severity': max_severity_label,
            'predicted_severity': max_severity_label,
            'peak_risk': 0,
            'risk_profile': [(0, overall_risk_score)],
            'mechanism': "N/A",
            'monitoring': "Review all alerts below",
            'alternatives': "Optimize based on suggestions",
            'reference': "CDSS Analysis",
            'evidence_level': "High"
        }
        results.insert(0, summary)
    # New: Sort remaining results by predicted_severity descending for prioritization
    results[1:] = sorted(results[1:], key=lambda x: severity_order.get(x['predicted_severity'], 0), reverse=True)
    return results


@app.route('/unit', methods=['GET'])
@login_required
def list_units():
    units = Unit.query.all()
    return render_template('list_units.html', units=units)

@app.route('/unit/add', methods=['GET', 'POST'])
@login_required
def add_unit():
    if request.method == 'POST':
        name = request.form.get('name')
        description = request.form.get('description')

        if not name:
            flash("Unit name is required!", "error")
            return redirect(url_for('add_unit'))

        try:
            unit = Unit(name=name, description=description)
            db.session.add(unit)
            db.session.commit()
            flash("Unit added successfully!", "success")
            return redirect(url_for('list_units'))
        except IntegrityError:
            db.session.rollback()
            flash("A unit with this name already exists!", "error")
            return redirect(url_for('add_unit'))
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error adding unit: {str(e)}")
            flash("An error occurred while adding the unit.", "error")
            return redirect(url_for('add_unit'))

    return render_template('add_unit.html')

@app.route('/unit/edit/<int:id>', methods=['GET', 'POST'])
@login_required
def edit_unit(id):
    unit = Unit.query.get_or_404(id)
    if request.method == 'POST':
        name = request.form.get('name')
        description = request.form.get('description')

        if not name:
            flash("Unit name is required!", "error")
            return redirect(url_for('edit_unit', id=id))

        try:
            unit.name = name
            unit.description = description
            db.session.commit()
            flash("Unit updated successfully!", "success")
            return redirect(url_for('list_units'))
        except IntegrityError:
            db.session.rollback()
            flash("A unit with this name already exists!", "error")
            return redirect(url_for('edit_unit', id=id))
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error updating unit (ID: {id}): {str(e)}")
            flash("An error occurred while updating the unit.", "error")
            return redirect(url_for('edit_unit', id=id))

    return render_template('edit_unit.html', unit=unit)

@app.route('/unit/delete/<int:id>', methods=['POST'])
@login_required
def delete_unit(id):
    unit = Unit.query.get_or_404(id)
    try:
        # Check if unit is referenced by any lab tests
        if LabTest.query.filter_by(unit_id=id).count() > 0:
            flash("Cannot delete unit because it is referenced by lab tests!", "error")
            return redirect(url_for('list_units'))
        
        db.session.delete(unit)
        db.session.commit()
        flash("Unit deleted successfully!", "success")
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error deleting unit (ID: {id}): {str(e)}")
        flash("An error occurred while deleting the unit.", "error")
    return redirect(url_for('list_units'))

# New Route for Adding Lab Tests
@app.route('/lab_test/add', methods=['GET', 'POST'])
@login_required
def add_lab_test():
    units = Unit.query.all()
    if request.method == 'POST':
        name_en = request.form.get('name_en')
        name_tr = request.form.get('name_tr')
        description = request.form.get('description')
        reference_range = request.form.get('reference_range')
        unit_id = request.form.get('unit_id', type=int)

        if not name_en:
            flash("English name is required!", "error")
            return redirect(url_for('add_lab_test'))

        try:
            lab_test = LabTest(
                name_en=name_en,
                name_tr=name_tr,
                description=description,
                reference_range=reference_range,
                unit_id=unit_id or None  # Handle empty selection
            )
            db.session.add(lab_test)
            db.session.commit()
            flash("Lab test added successfully!", "success")
            return redirect(url_for('list_lab_tests'))
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error adding lab test: {str(e)}")
            flash("An error occurred while adding the lab test.", "error")
            return redirect(url_for('add_lab_test'))

    return render_template('add_lab_test.html', units=units)

@app.route('/lab_test/edit/<int:id>', methods=['GET', 'POST'])
@login_required
def edit_lab_test(id):
    lab_test = LabTest.query.get_or_404(id)
    units = Unit.query.all()
    if request.method == 'POST':
        lab_test.name_en = request.form.get('name_en')
        lab_test.name_tr = request.form.get('name_tr')
        lab_test.description = request.form.get('description')
        lab_test.reference_range = request.form.get('reference_range')
        lab_test.unit_id = request.form.get('unit_id', type=int) or None

        if not lab_test.name_en:
            flash("English name is required!", "error")
            return redirect(url_for('edit_lab_test', id=id))

        try:
            db.session.commit()
            flash("Lab test updated successfully!", "success")
            return redirect(url_for('list_lab_tests'))
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error updating lab test (ID: {id}): {str(e)}")
            flash("An error occurred while updating the lab test.", "error")
            return redirect(url_for('edit_lab_test', id=id))

    return render_template('edit_lab_test.html', lab_test=lab_test, units=units)

@app.route('/lab_test/delete/<int:id>', methods=['POST'])
@login_required
def delete_lab_test(id):
    lab_test = LabTest.query.get_or_404(id)
    try:
        db.session.delete(lab_test)
        db.session.commit()
        flash("Lab test deleted successfully!", "success")
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error deleting lab test (ID: {id}): {str(e)}")
        flash("An error occurred while deleting the lab test.", "error")
    return redirect(url_for('list_lab_tests'))

@app.route('/search_lab_tests', methods=['GET'])
@login_required
def search_lab_tests():
    query = request.args.get('q', '')
    page = request.args.get('page', 1, type=int)
    per_page = 10
    lab_tests = LabTest.query.filter(
        or_(
            LabTest.name_en.ilike(f'%{query}%'),
            LabTest.name_tr.ilike(f'%{query}%')
        )
    ).paginate(page=page, per_page=per_page)
    results = [
        {'id': lt.id, 'text': f"{lt.name_en} ({lt.unit or 'N/A'})"}
        for lt in lab_tests.items
    ]
    return jsonify({
        'results': results,
        'pagination': {'more': lab_tests.has_next}
    })    

# Updated Route for Adding Drug-Lab Test Interactions
@app.route('/drug_lab_test/add', methods=['GET', 'POST'])
@login_required
def add_drug_lab_test_interaction():
    drugs = Drug.query.all()
    lab_tests = LabTest.query.all()
    severities = Severity.query.all()

    if request.method == 'POST':
        drug_id = request.form.get('drug_id', type=int)
        lab_test_id = request.form.get('lab_test_id', type=int)
        interaction_type = request.form.get('interaction_type')
        description = request.form.get('description')
        severity_id = request.form.get('severity_id', type=int)
        recommendation = request.form.get('recommendation')
        reference = request.form.get('reference')

        if not (drug_id and lab_test_id and interaction_type and severity_id):
            flash("Drug, lab test, interaction type, and severity are required!", "error")
            return redirect(url_for('add_drug_lab_test_interaction'))

        # Validate severity_id
        severity = Severity.query.get(severity_id)
        if not severity:
            flash("Invalid severity selected!", "error")
            return redirect(url_for('add_drug_lab_test_interaction'))

        try:
            interaction = DrugLabTestInteraction(
                drug_id=drug_id,
                lab_test_id=lab_test_id,
                interaction_type=interaction_type,
                description=description,
                severity_id=severity_id,  # Store severity_id instead of string
                recommendation=recommendation,
                reference=reference
            )
            db.session.add(interaction)
            db.session.commit()
            flash("Drug-lab test interaction added!", "success")
            return redirect(url_for('list_drug_lab_test_interactions'))
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error adding drug-lab test interaction: {str(e)}")
            flash("An error occurred while adding the interaction.", "error")
            return redirect(url_for('add_drug_lab_test_interaction'))

    return render_template('add_drug_lab_test.html', drugs=drugs, lab_tests=lab_tests, severities=severities)

@app.route('/drug_lab_test/edit/<int:id>', methods=['GET', 'POST'])
@login_required
def edit_drug_lab_test_interaction(id):
    interaction = DrugLabTestInteraction.query.get_or_404(id)
    drugs = Drug.query.all()
    lab_tests = LabTest.query.all()
    severities = Severity.query.all()

    if request.method == 'POST':
        drug_id = request.form.get('drug_id', type=int)
        lab_test_id = request.form.get('lab_test_id', type=int)
        interaction_type = request.form.get('interaction_type')
        severity_id = request.form.get('severity_id', type=int)
        description = request.form.get('description')
        recommendation = request.form.get('recommendation')
        reference = request.form.get('reference')

        if not (drug_id and lab_test_id and interaction_type and severity_id):
            flash("Drug, lab test, interaction type, and severity are required!", "error")
            return redirect(url_for('edit_drug_lab_test_interaction', id=id))

        # Validate inputs
        drug = Drug.query.get(drug_id)
        lab_test = LabTest.query.get(lab_test_id)
        severity = Severity.query.get(severity_id)
        if not (drug and lab_test and severity):
            flash("Invalid drug, lab test, or severity selected!", "error")
            return redirect(url_for('edit_drug_lab_test_interaction', id=id))

        try:
            interaction.drug_id = drug_id
            interaction.lab_test_id = lab_test_id
            interaction.interaction_type = interaction_type
            interaction.severity_id = severity_id
            interaction.description = description
            interaction.recommendation = recommendation
            interaction.reference = reference
            db.session.commit()
            flash("Drug-lab test interaction updated successfully!", "success")
            return redirect(url_for('list_drug_lab_test_interactions'))
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error updating drug-lab test interaction (ID: {id}): {str(e)}")
            flash("An error occurred while updating the interaction.", "error")
            return redirect(url_for('edit_drug_lab_test_interaction', id=id))

    return render_template('edit_drug_lab_test_interaction.html', interaction=interaction, drugs=drugs, lab_tests=lab_tests, severities=severities)

@app.route('/drug_lab_test/delete/<int:id>', methods=['POST'])
@login_required
def delete_drug_lab_test_interaction(id):
    interaction = DrugLabTestInteraction.query.get_or_404(id)
    try:
        db.session.delete(interaction)
        db.session.commit()
        flash("Drug-lab test interaction deleted successfully!", "success")
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error deleting drug-lab test interaction (ID: {id}): {str(e)}")
        flash("An error occurred while deleting the interaction.", "error")
    return redirect(url_for('list_drug_lab_test_interactions'))

@app.route('/drug_lab_test', methods=['GET'])
@login_required
def list_drug_lab_test_interactions():
    interactions = DrugLabTestInteraction.query.all()
    return render_template('list_drug_lab_test.html', interactions=interactions)

@app.route('/lab_test', methods=['GET'])
@login_required
def list_lab_tests():
    lab_tests = LabTest.query.all()
    return render_template('list_lab_tests.html', lab_tests=lab_tests)


@app.route('/drug_disease/add', methods=['GET', 'POST'])
def add_drug_disease_interaction():
    drugs = Drug.query.all()
    indications = Indication.query.all()
    severities = Severity.query.all()

    if request.method == 'POST':
        drug_id = request.form.get('drug_id', type=int)
        indication_id = request.form.get('indication_id', type=int)
        interaction_type = request.form.get('interaction_type')
        description = request.form.get('description')
        severity = request.form.get('severity')
        recommendation = request.form.get('recommendation')

        if not (drug_id and indication_id and interaction_type and severity):
            flash("Drug, disease, interaction type, and severity are required!", "error")
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

    return render_template('add_drug_disease.html', drugs=drugs, indications=indications, severities=severities)

@app.route('/drug_disease', methods=['GET'])
def list_drug_disease_interactions():
    interactions = DrugDiseaseInteraction.query.all()
    return render_template('list_drug_disease.html', interactions=interactions)

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
    if not re.match("^[ARNDCEQGHILKMFPSTWYV]+$", sequence):
        logging.warning(f"Invalid sequence for molecular weight: {sequence}")
        return "Invalid Sequence"
    return molecular_weight(sequence, seq_type='protein')


#UNIPROT API'DEN RESEPTÖRLER İLE İLGİLİ BİLGİLERİ ALMA...
@app.route('/api/uniprot', methods=['GET', 'POST'])
def fetch_uniprot_data():
    if request.method == 'POST':
        receptor_name = request.form.get('receptor_name')
        if not receptor_name or receptor_name.strip() == "":
            return "Receptor name cannot be empty."
        
        url = f"https://rest.uniprot.org/uniprotkb/search?query={receptor_name}+AND+organism_id:9606&fields=accession,protein_name,organism_name,gene_names,length,cc_subcellular_location,cc_function,sequence,xref_pdb,ft_binding"
        logging.info(f"Request URL: {url}")
        
        response = requests.get(url)
        if response.status_code != 200:
            logging.error(f"API Error: {response.text}")
            return f"Failed to fetch data from UniProt. Status: {response.status_code}"
        
        data = response.json()
        logging.info(f"UniProt API Response: {data}")
        results = data.get('results', [])
        
        if not results:
            return f"No UniProt entries found for {receptor_name}"
        
        for result in results:
            organism_name = result.get('organism', {}).get('scientificName', 'Unknown')
            if organism_name.lower() != "homo sapiens":
                logging.info(f"Skipping non-human: {organism_name}")
                continue
            
            accession = result.get('primaryAccession', 'Unknown')
            protein_description = result.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', 'Unknown')
            
            genes = result.get('genes', [])
            gene_primary = genes[0].get('geneName', {}).get('value', 'Unknown') if genes else 'Unknown'
            
            sequence_info = result.get('sequence', {})
            length = sequence_info.get('length', 0)
            molecular_weight_val = sequence_info.get('molWeight', 'Unknown')
            
            subcellular_location = 'Unknown'
            for comment in result.get('comments', []):
                if comment.get('commentType') == 'SUBCELLULAR LOCATION':
                    subcell_locations = comment.get('subcellularLocations', [])
                    if subcell_locations:
                        subcellular_location = subcell_locations[0].get('location', {}).get('value', 'Unknown')
            
            function = 'Unknown'
            for comment in result.get('comments', []):
                if comment.get('commentType') == 'FUNCTION':
                    function_texts = comment.get('texts', [])
                    if function_texts:
                        function = function_texts[0].get('value', 'Unknown')
            
            pdb_ids = [xref.get('id') for xref in result.get('uniProtKBCrossReferences', []) if xref.get('database') == 'PDB']
            
            binding_sites = []
            for feature in result.get('features', []):
                if feature.get('type') == 'BINDING':
                    pos = feature['location']['start']['value']
                    ligand = feature.get('description', 'Unknown')
                    binding_sites.append({"residue": pos, "ligand": ligand})
            
            logging.info(f"Binding sites for {accession}: {binding_sites}")
            
            binding_site_coords = {"x": 0, "y": 0, "z": 0}
            if pdb_ids and binding_sites:
                pdb_id = pdb_ids[0]
                pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
                try:
                    pdb_content = requests.get(pdb_url, timeout=10).text
                    parser = PDBParser(QUIET=True)
                    structure = parser.get_structure(pdb_id, io.StringIO(pdb_content))
                    chain = list(structure[0].get_chains())[0]
                    residue_num = binding_sites[0]["residue"]
                    residue = chain[residue_num]
                    ca = residue["CA"]
                    binding_site_coords = {
                        "x": float(ca.get_coord()[0]),
                        "y": float(ca.get_coord()[1]),
                        "z": float(ca.get_coord()[2])
                    }
                    logging.info(f"Binding coords for {pdb_id}: {binding_site_coords}")
                except Exception as e:
                    logging.error(f"Error parsing PDB {pdb_id}: {e}")
            
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
        receptors = Receptor.query.filter(Receptor.iuphar_id.isnot(None)).all()
        return render_template('fetch_iuphar.html', receptors=receptors)
    
    try:
        receptor_id = request.form.get('receptor_id', '').strip()
        receptor = Receptor.query.get(receptor_id)
        
        if not receptor or not receptor.iuphar_id:
            return "Invalid receptor or missing IUPHAR ID.", 400
        
        target_id = receptor.iuphar_id
        interaction_url = f"https://www.guidetopharmacology.org/services/targets/{target_id}/interactions"
        
        response = requests.get(interaction_url, timeout=15)
        if response.status_code != 200:
            return f"Failed to fetch interactions. Status: {response.status_code}", 500
        
        interactions = response.json()
        if not isinstance(interactions, list) or not interactions:
            return f"No interactions found for Target ID: {target_id}.", 404
        
        saved_interactions = 0
        for interaction in interactions:
            species = interaction.get('targetSpecies', 'Unknown')
            if species.lower() != "human":
                continue
            
            ligand_name = interaction.get('ligandName', '')
            drug = Drug.query.filter(db.func.lower(Drug.name_en) == db.func.lower(ligand_name)).first()
            
            if not drug:
                logging.warning(f"Ligand not found: {ligand_name}")
                continue
            
            affinity_value = parse_affinity(interaction.get('affinity'))
            if affinity_value is None:
                continue
            
            affinity_parameter = interaction.get('affinityParameter', 'N/A')
            interaction_type = interaction.get('type', 'N/A')
            mechanism = interaction.get('action', 'N/A')
            
            existing = DrugReceptorInteraction.query.filter_by(
                drug_id=drug.id,
                receptor_id=receptor.id,
                interaction_type=interaction_type,
                affinity=affinity_value,
                affinity_parameter=affinity_parameter
            ).first()
            
            if not existing:
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
        
        db.session.commit()
        return f"Successfully saved {saved_interactions} interactions for Target ID {target_id}.", 200
    
    except Exception as e:
        logging.error(f"Error: {e}")
        return jsonify({"error": "Error processing interactions.", "details": str(e)}), 500

def parse_affinity(affinity_value):
    try:
        if isinstance(affinity_value, str):
            return float(affinity_value.split(" - ")[0]) if " - " in affinity_value else float(affinity_value)
        return float(affinity_value) if affinity_value else None
    except ValueError:
        logging.warning(f"Unable to parse affinity: {affinity_value}")
        return None




@app.route('/receptors/map', methods=['GET', 'POST'])
def map_receptor_iuphar():
    if request.method == 'POST':
        receptor_id = request.form.get('receptor_id')
        iuphar_id = request.form.get('iuphar_id')
        receptor = Receptor.query.get(receptor_id)
        
        if receptor:
            receptor.iuphar_id = iuphar_id
            db.session.commit()
            return redirect(url_for('map_receptor_iuphar'))
        return "Receptor not found.", 404
    
    receptors = Receptor.query.all()
    return render_template('map_receptors.html', receptors=receptors)



@app.route('/interactions/drug-receptor', methods=['GET', 'POST'])
def manage_drug_receptor_interactions():
    if request.method == 'POST':
        drug_id = request.form.get('drug_id')
        receptor_id = request.form.get('receptor_id')
        affinity = request.form.get('affinity')
        interaction_type = request.form.get('interaction_type')
        mechanism = request.form.get('mechanism')
        return redirect(url_for('manage_drug_receptor_interactions'))
    
    interactions = DrugReceptorInteraction.query.all()
    drugs = Drug.query.all()
    receptors = Receptor.query.all()
    
    enriched_interactions = []
    for interaction in interactions:
        drug = Drug.query.get(interaction.drug_id)
        receptor = Receptor.query.get(interaction.receptor_id)
        
        if not drug or not receptor:
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
            query = DrugReceptorInteraction.query.join(Receptor).join(Drug)
            
            if search_value:
                query = query.filter(
                    db.or_(
                        Drug.name_en.ilike(f"%{search_value}%"),
                        Receptor.name.ilike(f"%{search_value}%"),
                        DrugReceptorInteraction.interaction_type.ilike(f"%{search_value}%"),
                        DrugReceptorInteraction.mechanism.ilike(f"%{search_value}%")
                    )
                )
            
            columns = [Drug.name_en, Receptor.name, DrugReceptorInteraction.affinity, 
                      DrugReceptorInteraction.affinity_parameter, DrugReceptorInteraction.interaction_type, 
                      DrugReceptorInteraction.mechanism]
            
            if 0 <= order_column_index < len(columns):
                column = columns[order_column_index]
                query = query.order_by(db.desc(column)) if order_direction == 'desc' else query.order_by(column)
            
            total_count = DrugReceptorInteraction.query.count()
            filtered_count = query.count()
            paginated_items = query.offset(start).limit(length).all()
            
            data = [
                {
                    "Ligand": i.drug.name_en if i.drug else f"Unknown {i.drug_id}",
                    "Receptor": i.receptor.name if i.receptor else f"Unknown {i.receptor_id}",
                    "Affinity": i.affinity or "N/A",
                    "Affinity Parameter": i.affinity_parameter or "N/A",
                    "Interaction Type": i.interaction_type or "N/A",
                    "Mechanism": i.mechanism or "N/A"
                }
                for i in paginated_items
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
            
            columns = [Receptor.name, Receptor.type, Receptor.molecular_weight, 
                      Receptor.length, Receptor.gene_name, Receptor.subcellular_location, 
                      Receptor.function]
            
            if 0 <= order_column_index < len(columns):
                column = columns[order_column_index]
                query = query.order_by(db.desc(column)) if order_direction == 'desc' else query.order_by(column)
            
            total_count = Receptor.query.count()
            filtered_count = query.count()
            paginated_items = query.offset(start).limit(length).all()
            
            data = [
                {
                    "Name": r.name or "Unknown",
                    "Type": r.type or "Unknown",
                    "Molecular Weight": r.molecular_weight or "N/A",
                    "Length": r.length or "N/A",
                    "Gene Name": r.gene_name or "N/A",
                    "Localization": r.subcellular_location or "N/A",
                    "Function": r.function or "N/A"
                }
                for r in paginated_items
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
        response = requests.get(target_url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            receptor_name = data.get('name', f"Unknown Target {target_id}")
            logging.info(f"Fetched receptor name: {receptor_name}")
            return receptor_name
        
        logging.error(f"Failed to fetch receptor name: {response.status_code}")
        return f"Target {target_id}"
    
    except Exception as e:
        logging.error(f"Failed to fetch receptor name: {e}")
        return f"Target {target_id}"



@app.route('/api/affinity-data', methods=['GET'])
def affinity_data():
    interactions = DrugReceptorInteraction.query.all()
    chart_data = {"labels": [], "values": []}
    
    for interaction in interactions:
        receptor_name = interaction.receptor.name if interaction.receptor else "Unknown"
        ligand_name = interaction.drug.name_en if interaction.drug else "Unknown"
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
        "labels": [i[0] for i in interactions],
        "values": [i[1] for i in interactions]
    }
    return jsonify(chart_data)



# Reseptör - Ligand Etkileşim Kodları...
# Helper function to get executable with correct extension
def get_executable(name):
    if platform.system() == "Windows":
        return shutil.which(f"{name}.exe") or shutil.which(name)
    return shutil.which(name)

def safe_remove(path, retries=3, delay=0.5):
    path = Path(path)
    for _ in range(retries):
        try:
            if path.exists():
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    path.unlink()
            return
        except (OSError, PermissionError):
            time.sleep(delay)
    logging.warning(f"Failed to remove {path}")


@app.route('/api/receptors', methods=['GET'])
def get_receptors():
    search = request.args.get('search', '').strip()
    limit = request.args.get('limit', 10, type=int)
    page = request.args.get('page', 1, type=int)
    
    query = Receptor.query.filter(Receptor.name.ilike(f"%{search}%")) if search else Receptor.query
    paginated_query = query.paginate(page=page, per_page=limit)
    
    results = [{"id": r.id, "text": r.name} for r in paginated_query.items]
    return jsonify({"results": results, "has_next": paginated_query.has_next})



@app.route('/receptor-ligand-simulator', methods=['GET', 'POST'])
def receptor_ligand_simulator():
    if request.method == 'GET':
        return render_template('receptor_ligand_simulator.html')
    
    receptor = request.form.get('receptor')
    ligand = request.form.get('ligand')
    
    if ligand:
        session['ligand'] = ligand
    else:
        ligand = session.get('ligand')
    
    if not receptor or not ligand:
        return jsonify({"error": "Receptor and ligand required."}), 400
    
    try:
        receptor_file_path = os.path.join('static', f'receptor_{uuid.uuid4().hex}.pdb')
        with open(receptor_file_path, "w") as f:
            f.write(receptor)
        
        ligand_file_url = session.get('ligand_file_url')
        if not ligand_file_url:
            ligand_file_path = os.path.join('static', f'ligand_{uuid.uuid4().hex}.pdb')
            with open(ligand_file_path, "w") as f:
                f.write(ligand)
            session['ligand_file_url'] = f"/{ligand_file_path}"
            ligand_file_url = session['ligand_file_url']
        
        return jsonify({
            "message": "Success",
            "receptor_file_url": f"/{receptor_file_path}",
            "ligand_file_url": ligand_file_url
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    


@app.route('/api/convert_ligand', methods=['GET'])
def convert_ligand():
    drug_id = request.args.get('drug_id')
    if not drug_id:
        return jsonify({"error": "Drug ID required."}), 400
    
    try:
        drug_id = int(drug_id)
    except ValueError:
        return jsonify({"error": "Invalid Drug ID."}), 400
    
    drug = db.session.get(Drug, drug_id)
    if not drug:
        return jsonify({"error": f"No Drug found for ID {drug_id}."}), 404
    
    drug_detail = DrugDetail.query.filter_by(drug_id=drug_id).first()
    if not drug_detail:
        return jsonify({"error": f"No DrugDetail for ID {drug_id}."}), 404
    
    if not drug_detail.smiles:
        return jsonify({"error": f"No SMILES for ID {drug_id}."}), 404
    
    pdb_file = None
    try:
        if not drug_detail.smiles.strip() or len(drug_detail.smiles) > 1000:
            raise ValueError("Invalid SMILES")
        
        mol = Chem.MolFromSmiles(drug_detail.smiles)
        if mol is None:
            raise ValueError("Invalid SMILES")
        
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        
        pdb_file = os.path.abspath(f"static/ligand_{uuid.uuid4().hex}.pdb")
        Chem.MolToPDBFile(mol, pdb_file)
        
        if not os.path.exists(pdb_file):
            raise RuntimeError("Failed to generate PDB.")
        
        with open(pdb_file, 'r') as f:
            pdb_content = f.read()
        
        return jsonify({"pdb": pdb_content}), 200
    
    except ValueError as e:
        logging.error(f"SMILES error for drug_id={drug_id}: {e}")
        return jsonify({"error": "Invalid SMILES.", "details": str(e)}), 400
    except Exception as e:
        logging.error(f"Error in convert_ligand: {e}")
        return jsonify({"error": f"Unexpected error: {e}"}), 500
    finally:
        if pdb_file:
            safe_remove(pdb_file)


@app.route('/api/get_receptor_structure', methods=['GET'])
def get_receptor_structure():
    receptor_id = request.args.get('receptor_id')
    if not receptor_id:
        return jsonify({"error": "Receptor ID required."}), 400
    
    receptor = db.session.get(Receptor, receptor_id)
    if not receptor or not receptor.pdb_ids:
        return jsonify({"error": "Receptor not found or no PDB IDs."}), 404
    
    pdb_id = receptor.pdb_ids.split(",")[0]
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    
    response = requests.get(url, timeout=15)
    if response.status_code != 200:
        return jsonify({"error": f"Failed to fetch PDB {pdb_id}."}), 500
    
    pdb_content = response.text
    binding_site_coords = get_pocket_coords(pdb_content, pdb_id)
    
    receptor.binding_site_x = binding_site_coords["x"]
    receptor.binding_site_y = binding_site_coords["y"]
    receptor.binding_site_z = binding_site_coords["z"]
    db.session.commit()
    
    return jsonify({"pdb": pdb_content, "binding_site": binding_site_coords}), 200

def get_pocket_coords(pdb_content, pdb_id):
    temp_pdb_path = None
    try:
        temp_pdb = Path(f"temp_{pdb_id}.pdb")
        temp_pdb_path = temp_pdb.absolute()
        
        with temp_pdb_path.open("w") as f:
            f.write(pdb_content)
        
        if not temp_pdb_path.exists():
            raise FileNotFoundError(f"Failed to create {temp_pdb_path}")
        
        structure = parsePDB(str(temp_pdb_path))
        protein = structure.select('protein')
        
        if not protein:
            raise ValueError(f"No protein atoms in {pdb_id}")
        
        exposed_residues = protein.select('resname PHE TYR TRP HIS ASP GLU LYS ARG and within 8 of all')
        
        if not exposed_residues:
            exposed_residues = protein.select('resname ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR')
            logging.warning(f"Using fallback residues for {pdb_id}")
        
        if not exposed_residues:
            logging.warning(f"No pocket residues for {pdb_id}")
            return {"x": 0, "y": 0, "z": 0}
        
        coords = exposed_residues.getCoords()
        if len(coords) < 3:
            logging.warning(f"Insufficient pocket residues for {pdb_id}")
            return {"x": 0, "y": 0, "z": 0}
        
        centroid = calcCenter(coords)
        logging.info(f"ProDy binding site for {pdb_id}: {centroid}")
        
        return {"x": float(centroid[0]), "y": float(centroid[1]), "z": float(centroid[2])}
    
    except Exception as e:
        logging.error(f"ProDy failed for {pdb_id}: {e}")
        return {"x": 0, "y": 0, "z": 0}
    finally:
        if temp_pdb_path:
            safe_remove(temp_pdb_path)


@app.route('/api/get_interaction_data', methods=['GET'])
def get_interaction_data():
    drug_id = request.args.get('drug_id')
    receptor_id = request.args.get('receptor_id')
    
    if not drug_id or not receptor_id:
        return jsonify({"error": "Both drug_id and receptor_id required."}), 400
    
    try:
        interaction = DrugReceptorInteraction.query.filter_by(
            drug_id=drug_id, receptor_id=receptor_id
        ).first()
        
        if not interaction:
            return jsonify({"error": "No interaction found."}), 404
        
        receptor = db.session.get(Receptor, receptor_id)
        if not receptor:
            return jsonify({"error": "Receptor not found."}), 404
        
        drug = db.session.get(Drug, drug_id)
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
    

@app.route('/health', methods=['GET'])
def health_check():
    try:
        # Check ProDy
        import prody
        prody_version = prody.__version__
        # Check RDKit
        from rdkit import Chem
        rdkit_version = Chem.rdkitVersion
        return jsonify({
            "status": "healthy",
            "prody": f"Version {prody_version}",
            "rdkit": f"Version {rdkit_version}"
        }), 200
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500
#Reseptör - Ligand Finitooo

# Yan etki API
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


#Hedef Molekül API
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
@contextmanager
def db_transaction():
    """Context manager for database transactions with automatic rollback."""
    try:
        yield
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        logger.error(f"Transaction failed: {str(e)}")
        raise
    finally:
        db.session.close()

def retry_on_failure(max_retries=MAX_RETRIES):
    """Decorator for retry logic."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except SQLAlchemyError as e:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}, retrying...")
                    db.session.rollback()
            return None
        return wrapper
    return decorator

# ============================================================================
# HELPER FUNCTIONS - ENHANCED
# ============================================================================

def save_uploaded_file(file_storage):
    """Save uploaded TSV file to disk with validation."""
    if not file_storage or not file_storage.filename:
        raise ValueError("No file provided")
    
    # Validate file extension
    if not file_storage.filename.lower().endswith('.tsv'):
        raise ValueError("Only .tsv files are supported")
    
    # Check file size (optional - add max size check)
    file_storage.seek(0, os.SEEK_END)
    file_size = file_storage.tell()
    file_storage.seek(0)
    
    if file_size == 0:
        raise ValueError("File is empty")
    
    MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
    if file_size > MAX_FILE_SIZE:
        raise ValueError(f"File too large: {file_size} bytes (max: {MAX_FILE_SIZE})")
    
    filename = secure_filename(file_storage.filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{filename}"
    
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    file_storage.save(file_path)
    logger.info(f"Saved file: {filename} ({file_size} bytes) to {file_path}")
    
    return file_path

def parse_date(date_str, row_id=""):
    """Parse date string in YYYY-MM-DD format with enhanced error handling."""
    if not date_str or not date_str.strip():
        return None
    
    date_str = date_str.strip()
    
    # Try multiple date formats
    formats = ["%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%m/%d/%Y"]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    
    logger.warning(f"Invalid date format '{date_str}' for row {row_id}")
    return None

def safe_split(text, delimiter=';'):
    """Split text by delimiter and strip whitespace."""
    if not text or not isinstance(text, str):
        return []
    return [t.strip() for t in text.split(delimiter) if t.strip()]

def safe_float(value, default=None):
    """Safely convert to float."""
    if not value or value == '':
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int(value, default=None):
    """Safely convert to integer."""
    if not value or value == '':
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

# ============================================================================
# DATABASE QUERY HELPERS - OPTIMIZED
# ============================================================================

def check_drugs_exist(drug_names, context_id=""):
    """
    Check if drugs exist in database and return found drugs + missing ones.
    Uses case-insensitive matching on multiple fields.
    
    ⚠️ CRITICAL: Only matches drugs already in your database!
    """
    if not drug_names:
        return [], set()
    
    missing = set()
    found_drugs = []
    
    # Normalize drug names
    drug_names_lower = {d.strip().lower(): d for d in drug_names if d and d.strip()}
    
    if not drug_names_lower:
        return [], set()
    
    # Batch query for better performance
    try:
        results = db.session.query(Drug).filter(
            or_(
                func.lower(Drug.name_en).in_(drug_names_lower.keys()),
                func.lower(Drug.name_tr).in_(drug_names_lower.keys()),
                func.lower(Drug.pharmgkb_id).in_(drug_names_lower.keys())
            )
        ).all()
        
        # Map found drugs
        found_names_lower = set()
        for drug in results:
            # Check which name matched
            for key in drug_names_lower.keys():
                if (key == drug.name_en.lower() or 
                    key == drug.name_tr.lower() or 
                    (drug.pharmgkb_id and key == drug.pharmgkb_id.lower())):
                    found_drugs.append(drug)
                    found_names_lower.add(key)
                    logger.debug(f"✓ Found drug: {drug_names_lower[key]} -> {drug.name_en} (ID: {drug.id})")
                    break
        
        # Identify missing drugs
        missing = {drug_names_lower[k] for k in drug_names_lower.keys() - found_names_lower}
        
        if missing:
            logger.warning(f"✗ Missing drugs for {context_id}: {', '.join(list(missing)[:5])}")
        
    except SQLAlchemyError as e:
        logger.error(f"Database error checking drugs: {str(e)}")
        return [], set(drug_names)
    
    return found_drugs, missing

@retry_on_failure()
def get_or_create_gene(gene_symbol, gene_id=None):
    """Get existing gene or create new one with retry logic."""
    if not gene_symbol or not gene_symbol.strip():
        return None
    
    gene_symbol = gene_symbol.strip()
    
    # Try to find existing
    gene = db.session.query(Gene).filter_by(gene_symbol=gene_symbol).first()
    
    if not gene:
        # Generate ID if not provided
        if not gene_id:
            gene_id = f"PA{gene_symbol}"
        
        gene = Gene(gene_id=gene_id, gene_symbol=gene_symbol)
        db.session.add(gene)
        
        try:
            db.session.flush()
            logger.debug(f"Created new gene: {gene_symbol} (ID: {gene_id})")
        except IntegrityError:
            db.session.rollback()
            # Another process created it, try to fetch again
            gene = db.session.query(Gene).filter_by(gene_symbol=gene_symbol).first()
    
    return gene

@retry_on_failure()
def get_or_create_variant(variant_name, pharmgkb_id=None):
    """Get existing variant or create new one with retry logic."""
    if not variant_name or not variant_name.strip():
        return None
    
    variant_name = variant_name.strip()
    
    # Try by name first
    var = db.session.query(Variant).filter_by(name=variant_name).first()
    
    if not var and pharmgkb_id:
        # Try by PharmGKB ID
        var = db.session.query(Variant).filter_by(pharmgkb_id=pharmgkb_id).first()
    
    if not var:
        var = Variant(name=variant_name, pharmgkb_id=pharmgkb_id)
        db.session.add(var)
        
        try:
            db.session.flush()
            logger.debug(f"Created new variant: {variant_name}")
        except IntegrityError:
            db.session.rollback()
            # Fetch again if created by another process
            var = db.session.query(Variant).filter_by(name=variant_name).first()
    
    return var

@retry_on_failure()
def get_or_create_phenotype(phenotype_name):
    """Get existing phenotype or create new one with retry logic."""
    if not phenotype_name or not phenotype_name.strip():
        return None
    
    phenotype_name = phenotype_name.strip()
    
    pheno = db.session.query(Phenotype).filter_by(name=phenotype_name).first()
    
    if not pheno:
        pheno = Phenotype(name=phenotype_name)
        db.session.add(pheno)
        
        try:
            db.session.flush()
            logger.debug(f"Created new phenotype: {phenotype_name}")
        except IntegrityError:
            db.session.rollback()
            pheno = db.session.query(Phenotype).filter_by(name=phenotype_name).first()
    
    return pheno

@retry_on_failure()
def get_or_create_publication(pmid, title=None, year=None, journal=None):
    """Get existing publication or create new one with retry logic."""
    if not pmid or not pmid.strip():
        return None
    
    pmid = pmid.strip()
    
    pub = db.session.get(Publication, pmid)
    
    if not pub:
        pub = Publication(
            pmid=pmid,
            title=title or None,
            year=year,
            journal=journal
        )
        db.session.add(pub)
        
        try:
            db.session.flush()
            logger.debug(f"Created new publication: PMID {pmid}")
        except IntegrityError:
            db.session.rollback()
            pub = db.session.get(Publication, pmid)
    
    return pub

def bulk_insert_with_progress(items, model_name, batch_size=BATCH_SIZE):
    """
    Bulk insert items with progress tracking and automatic commits.
    Enhanced with better error handling.
    """
    if not items:
        logger.warning(f"[{model_name}] No items to insert")
        return 0
    
    total = len(items)
    inserted = 0
    failed = 0
    
    logger.info(f"[{model_name}] Starting bulk insert of {total} items")
    
    for i in range(0, total, batch_size):
        batch = items[i:i + batch_size]
        
        try:
            db.session.bulk_save_objects(batch)
            inserted += len(batch)
            
            # Commit periodically
            if (i // batch_size) % COMMIT_FREQUENCY == 0:
                db.session.commit()
                progress = (inserted * 100) // total
                logger.info(f"[{model_name}] Progress: {inserted}/{total} ({progress}%)")
        
        except SQLAlchemyError as e:
            logger.error(f"[{model_name}] Batch failed at {i}: {str(e)}")
            db.session.rollback()
            failed += len(batch)
            
            # Try individual inserts for this batch
            for item in batch:
                try:
                    db.session.add(item)
                    db.session.commit()
                    inserted += 1
                except SQLAlchemyError:
                    db.session.rollback()
                    failed += 1
    
    # Final commit
    try:
        db.session.commit()
        logger.info(f"[{model_name}] ✓ Complete: {inserted}/{total} inserted, {failed} failed")
    except SQLAlchemyError as e:
        db.session.rollback()
        logger.error(f"[{model_name}] Final commit failed: {str(e)}")
    
    return inserted

# ============================================================================
# ETL FUNCTIONS - CLINICAL ANNOTATIONS (ENHANCED)
# ============================================================================

def load_clinical_annotations_tsv(filepath):
    """
    Load clinical_annotations.tsv with optimized bulk processing.
    ✅ ENHANCED: Better error handling, progress tracking, drug matching
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    missing_drugs_global = set()
    inserted = 0
    skipped = 0
    errors = []
    
    logger.info(f"[clinical_annotations] Starting import from {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            
            # Validate headers
            required_cols = ["Clinical Annotation ID", "Gene", "Level of Evidence"]
            if not all(col in reader.fieldnames for col in required_cols):
                raise ValueError(f"Missing required columns. Expected: {required_cols}")
            
            total_rows = sum(1 for _ in open(filepath, 'r', encoding='utf-8')) - 1
            logger.info(f"[clinical_annotations] Processing {total_rows} rows")
            
            for idx, row in enumerate(reader, 1):
                ca_id = row.get("Clinical Annotation ID", "").strip()
                
                if not ca_id:
                    skipped += 1
                    continue
                
                try:
                    # Get or create ClinicalAnnotation
                    ca = db.session.get(ClinicalAnnotation, ca_id)
                    if not ca:
                        ca = ClinicalAnnotation(clinical_annotation_id=ca_id)
                        db.session.add(ca)
                    
                    # Update fields with safe conversions
                    ca.level_of_evidence = row.get("Level of Evidence", "").strip() or None
                    ca.level_override = row.get("Level Override", "").strip() or None
                    ca.level_modifiers = row.get("Level Modifiers", "").strip() or None
                    ca.score = safe_float(row.get("Score"))
                    ca.pmid_count = safe_int(row.get("PMID Count"))
                    ca.evidence_count = safe_int(row.get("Evidence Count"))
                    ca.phenotype_category = row.get("Phenotype Category", "").strip() or None
                    ca.url = row.get("URL", "").strip() or None
                    ca.latest_history_date = parse_date(row.get("Latest History Date (YYYY-MM-DD)"), ca_id)
                    ca.specialty_population = row.get("Specialty Population", "").strip() or None
                    
                    db.session.flush()  # Get the ID if new
                    
                    # Clear existing relationships (for re-imports)
                    db.session.query(ClinicalAnnotationDrug).filter_by(clinical_annotation_id=ca_id).delete()
                    db.session.query(ClinicalAnnotationGene).filter_by(clinical_annotation_id=ca_id).delete()
                    db.session.query(ClinicalAnnotationPhenotype).filter_by(clinical_annotation_id=ca_id).delete()
                    db.session.query(ClinicalAnnotationVariant).filter_by(clinical_annotation_id=ca_id).delete()
                    
                    # ✅ CRITICAL: Process drugs - Only link to existing drugs in DB
                    raw_drugs = row.get("Drug(s)", "").strip()
                    if raw_drugs:
                        drug_names = safe_split(raw_drugs, ';')
                        found_drugs, missing = check_drugs_exist(drug_names, ca_id)
                        missing_drugs_global.update(missing)
                        
                        for drug in found_drugs:
                            ca_drug = ClinicalAnnotationDrug(
                                clinical_annotation_id=ca_id,
                                drug_id=drug.id
                            )
                            db.session.add(ca_drug)
                    
                    # Process genes
                    raw_genes = row.get("Gene", "").strip()
                    if raw_genes:
                        for gene_symbol in safe_split(raw_genes, ';'):
                            gene = get_or_create_gene(gene_symbol)
                            if gene:
                                ca_gene = ClinicalAnnotationGene(
                                    clinical_annotation_id=ca_id,
                                    gene_id=gene.gene_id
                                )
                                db.session.add(ca_gene)
                    
                    # Process phenotypes
                    raw_phenotypes = row.get("Phenotype(s)", "").strip()
                    if raw_phenotypes:
                        for pheno_name in safe_split(raw_phenotypes, ';'):
                            pheno = get_or_create_phenotype(pheno_name)
                            if pheno:
                                ca_pheno = ClinicalAnnotationPhenotype(
                                    clinical_annotation_id=ca_id,
                                    phenotype_id=pheno.id
                                )
                                db.session.add(ca_pheno)
                    
                    # Process variants/haplotypes
                    raw_variants = row.get("Variant/Haplotypes", "").strip()
                    if raw_variants:
                        for var_name in safe_split(raw_variants, ';'):
                            var = get_or_create_variant(var_name)
                            if var:
                                ca_var = ClinicalAnnotationVariant(
                                    clinical_annotation_id=ca_id,
                                    variant_id=var.id
                                )
                                db.session.add(ca_var)
                    
                    inserted += 1
                    
                    # Periodic commits for memory management
                    if inserted % 100 == 0:
                        db.session.commit()
                        progress = (idx * 100) // total_rows
                        logger.info(f"[clinical_annotations] Progress: {idx}/{total_rows} ({progress}%) - {inserted} inserted")
                
                except Exception as e:
                    error_msg = f"Row {ca_id} (line {idx}) failed: {str(e)}"
                    logger.error(f"[clinical_annotations] {error_msg}")
                    errors.append(error_msg)
                    db.session.rollback()
                    skipped += 1
            
            # Final commit
            db.session.commit()
    
    except Exception as e:
        logger.error(f"[clinical_annotations] Fatal error: {str(e)}")
        db.session.rollback()
        raise
    
    # Summary
    logger.info(f"[clinical_annotations] ✓ COMPLETE")
    logger.info(f"  - Inserted: {inserted}")
    logger.info(f"  - Skipped: {skipped}")
    logger.info(f"  - Errors: {len(errors)}")
    
    if missing_drugs_global:
        logger.warning(f"[clinical_annotations] ⚠️  Missing drugs ({len(missing_drugs_global)}): {', '.join(sorted(missing_drugs_global)[:20])}{'...' if len(missing_drugs_global) > 20 else ''}")
    
    return inserted, skipped, missing_drugs_global

def load_clinical_ann_history_tsv(filepath):
    """Load clinical_ann_history.tsv with enhanced error handling."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    inserted = 0
    skipped = 0
    errors = []
    
    logger.info(f"[clinical_ann_history] Starting import from {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            
            total_rows = sum(1 for _ in open(filepath, 'r', encoding='utf-8')) - 1
            
            for idx, row in enumerate(reader, 1):
                ca_id = row.get("Clinical Annotation ID", "").strip()
                
                if not ca_id:
                    skipped += 1
                    continue
                
                # Verify parent exists
                if not db.session.get(ClinicalAnnotation, ca_id):
                    skipped += 1
                    continue
                
                try:
                    hist = ClinicalAnnHistory(
                        clinical_annotation_id=ca_id,
                        date=parse_date(row.get("Date (YYYY-MM-DD)"), ca_id),
                        type=row.get("Type", "").strip() or None,
                        comment=row.get("Comment", "").strip() or None
                    )
                    db.session.add(hist)
                    inserted += 1
                    
                    if inserted % 500 == 0:
                        db.session.commit()
                        progress = (idx * 100) // total_rows
                        logger.info(f"[clinical_ann_history] Progress: {idx}/{total_rows} ({progress}%)")
                
                except Exception as e:
                    error_msg = f"Row {ca_id} (line {idx}) failed: {str(e)}"
                    logger.error(f"[clinical_ann_history] {error_msg}")
                    errors.append(error_msg)
                    db.session.rollback()
                    skipped += 1
            
            db.session.commit()
    
    except Exception as e:
        logger.error(f"[clinical_ann_history] Fatal error: {str(e)}")
        db.session.rollback()
        raise
    
    logger.info(f"[clinical_ann_history] ✓ Complete: Inserted={inserted}, Skipped={skipped}, Errors={len(errors)}")
    return inserted, skipped, set()

def load_clinical_ann_alleles_tsv(filepath):
    """Load clinical_ann_alleles.tsv with enhanced error handling."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    inserted = 0
    skipped = 0
    errors = []
    
    logger.info(f"[clinical_ann_alleles] Starting import from {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            
            total_rows = sum(1 for _ in open(filepath, 'r', encoding='utf-8')) - 1
            
            for idx, row in enumerate(reader, 1):
                ca_id = row.get("Clinical Annotation ID", "").strip()
                genotype = row.get("Genotype/Allele", "").strip()
                
                if not ca_id or not genotype:
                    skipped += 1
                    continue
                
                # Verify parent exists
                if not db.session.get(ClinicalAnnotation, ca_id):
                    skipped += 1
                    continue
                
                try:
                    allele = ClinicalAnnAllele(
                        clinical_annotation_id=ca_id,
                        genotype_allele=genotype,
                        annotation_text=row.get("Annotation Text", "").strip() or None,
                        allele_function=row.get("Allele Function", "").strip() or None
                    )
                    db.session.add(allele)
                    inserted += 1
                    
                    if inserted % 500 == 0:
                        db.session.commit()
                        progress = (idx * 100) // total_rows
                        logger.info(f"[clinical_ann_alleles] Progress: {idx}/{total_rows} ({progress}%)")
                
                except Exception as e:
                    error_msg = f"Row {ca_id}/{genotype} (line {idx}) failed: {str(e)}"
                    logger.error(f"[clinical_ann_alleles] {error_msg}")
                    errors.append(error_msg)
                    db.session.rollback()
                    skipped += 1
            
            db.session.commit()
    
    except Exception as e:
        logger.error(f"[clinical_ann_alleles] Fatal error: {str(e)}")
        db.session.rollback()
        raise
    
    logger.info(f"[clinical_ann_alleles] ✓ Complete: Inserted={inserted}, Skipped={skipped}, Errors={len(errors)}")
    return inserted, skipped, set()

def load_clinical_ann_evidence_tsv(filepath):
    """Load clinical_ann_evidence.tsv with enhanced error handling."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    inserted = 0
    skipped = 0
    errors = []
    
    logger.info(f"[clinical_ann_evidence] Starting import from {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            
            total_rows = sum(1 for _ in open(filepath, 'r', encoding='utf-8')) - 1
            
            for idx, row in enumerate(reader, 1):
                ev_id = row.get("Evidence ID", "").strip()
                ca_id = row.get("Clinical Annotation ID", "").strip()
                pmid = row.get("PMID", "").strip()
                
                if not ev_id or not ca_id:
                    skipped += 1
                    continue
                
                # Verify parent exists
                if not db.session.get(ClinicalAnnotation, ca_id):
                    skipped += 1
                    continue
                
                try:
                    # Get or create evidence
                    ev = db.session.get(ClinicalAnnEvidence, ev_id)
                    if not ev:
                        ev = ClinicalAnnEvidence(evidence_id=ev_id)
                        db.session.add(ev)
                    
                    ev.clinical_annotation_id = ca_id
                    ev.evidence_type = row.get("Evidence Type", "").strip() or None
                    ev.evidence_url = row.get("Evidence URL", "").strip() or None
                    ev.summary = row.get("Summary", "").strip() or None
                    ev.score = safe_float(row.get("Score"))
                    
                    db.session.flush()
                    
                    # Handle publication link
                    if pmid:
                        pub = get_or_create_publication(pmid)
                        if pub:
                            # Check if link exists
                            ev_pub = db.session.query(ClinicalAnnEvidencePublication).filter_by(
                                evidence_id=ev_id,
                                pmid=pmid
                            ).first()
                            
                            if not ev_pub:
                                ev_pub = ClinicalAnnEvidencePublication(
                                    evidence_id=ev_id,
                                    pmid=pmid
                                )
                                db.session.add(ev_pub)
                    
                    inserted += 1
                    
                    if inserted % 500 == 0:
                        db.session.commit()
                        progress = (idx * 100) // total_rows
                        logger.info(f"[clinical_ann_evidence] Progress: {idx}/{total_rows} ({progress}%)")
                
                except Exception as e:
                    error_msg = f"Row {ev_id} (line {idx}) failed: {str(e)}"
                    logger.error(f"[clinical_ann_evidence] {error_msg}")
                    errors.append(error_msg)
                    db.session.rollback()
                    skipped += 1
            
            db.session.commit()
    
    except Exception as e:
        logger.error(f"[clinical_ann_evidence] Fatal error: {str(e)}")
        db.session.rollback()
        raise
    
    logger.info(f"[clinical_ann_evidence] ✓ Complete: Inserted={inserted}, Skipped={skipped}, Errors={len(errors)}")
    return inserted, skipped, set()

# ============================================================================
# VALIDATION & REPORTING
# ============================================================================

def validate_import_results(results):
    """Validate and summarize import results."""
    total_inserted = sum(r[0] for r in results.values())
    total_skipped = sum(r[1] for r in results.values())
    all_missing_drugs = set()
    
    for missing in [r[2] for r in results.values()]:
        all_missing_drugs.update(missing)
    
    logger.info("=" * 80)
    logger.info("PHARMGKB IMPORT SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total inserted: {total_inserted}")
    logger.info(f"Total skipped: {total_skipped}")
    
    if all_missing_drugs:
        logger.warning(f"⚠️  Missing drugs ({len(all_missing_drugs)}): {', '.join(sorted(all_missing_drugs)[:30])}")
    
    for name, (ins, skip, _) in results.items():
        logger.info(f"  {name}: {ins} inserted, {skip} skipped")
    
    logger.info("=" * 80)
    
    return {
        'total_inserted': total_inserted,
        'total_skipped': total_skipped,
        'missing_drugs': list(all_missing_drugs),
        'details': {k: {'inserted': v[0], 'skipped': v[1]} for k, v in results.items()}
    }

# ============================================================================
# ETL FUNCTIONS - VARIANT ANNOTATIONS (ENHANCED)
# ============================================================================

def load_study_parameters_tsv(filepath):
    """
    Load study_parameters.tsv with enhanced error handling and validation.
    ✅ ENHANCED: Better type conversions, progress tracking, parent validation
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    inserted = 0
    skipped = 0
    errors = []
    
    logger.info(f"[study_parameters] Starting import from {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            
            # Validate headers
            required_cols = ["Study Parameters ID", "Variant Annotation ID"]
            if not all(col in reader.fieldnames for col in required_cols):
                raise ValueError(f"Missing required columns: {required_cols}")
            
            total_rows = sum(1 for _ in open(filepath, 'r', encoding='utf-8')) - 1
            logger.info(f"[study_parameters] Processing {total_rows} rows")
            
            for idx, row in enumerate(reader, 1):
                sp_id = row.get("Study Parameters ID", "").strip()
                va_id = row.get("Variant Annotation ID", "").strip()
                
                if not sp_id or not va_id:
                    skipped += 1
                    continue
                
                try:
                    # Ensure parent VariantAnnotation exists
                    va = db.session.get(VariantAnnotation, va_id)
                    if not va:
                        va = VariantAnnotation(variant_annotation_id=va_id)
                        db.session.add(va)
                        db.session.flush()
                        logger.debug(f"Created VariantAnnotation: {va_id}")
                    
                    # Get or create StudyParameters
                    sp = db.session.get(StudyParameters, sp_id)
                    if not sp:
                        sp = StudyParameters(study_parameters_id=sp_id)
                        db.session.add(sp)
                    
                    # Update fields with safe conversions
                    sp.variant_annotation_id = va_id
                    sp.study_type = row.get("Study Type", "").strip() or None
                    sp.study_cases = safe_int(row.get("Study Cases"))
                    sp.study_controls = safe_int(row.get("Study Controls"))
                    sp.characteristics = row.get("Characteristics", "").strip() or None
                    sp.characteristics_type = row.get("Characteristics Type", "").strip() or None
                    sp.frequency_in_cases = safe_float(row.get("Frequency In Cases"))
                    sp.allele_of_frequency_in_cases = row.get("Allele Of Frequency In Cases", "").strip() or None
                    sp.frequency_in_controls = safe_float(row.get("Frequency In Controls"))
                    sp.allele_of_frequency_in_controls = row.get("Allele Of Frequency In Controls", "").strip() or None
                    sp.p_value = row.get("P Value", "").strip() or None
                    sp.ratio_stat_type = row.get("Ratio Stat Type", "").strip() or None
                    sp.ratio_stat = safe_float(row.get("Ratio Stat"))
                    sp.confidence_interval_start = safe_float(row.get("Confidence Interval Start"))
                    sp.confidence_interval_stop = safe_float(row.get("Confidence Interval Stop"))
                    sp.biogeographical_groups = row.get("Biogeographical Groups", "").strip() or None
                    
                    inserted += 1
                    
                    # Periodic commits
                    if inserted % 500 == 0:
                        db.session.commit()
                        progress = (idx * 100) // total_rows
                        logger.info(f"[study_parameters] Progress: {idx}/{total_rows} ({progress}%) - {inserted} inserted")
                
                except Exception as e:
                    error_msg = f"Row {sp_id} (line {idx}) failed: {str(e)}"
                    logger.error(f"[study_parameters] {error_msg}")
                    errors.append(error_msg)
                    db.session.rollback()
                    skipped += 1
            
            db.session.commit()
    
    except Exception as e:
        logger.error(f"[study_parameters] Fatal error: {str(e)}")
        db.session.rollback()
        raise
    
    logger.info(f"[study_parameters] ✓ Complete: Inserted={inserted}, Skipped={skipped}, Errors={len(errors)}")
    return inserted, skipped, set()

def load_var_fa_ann_tsv(filepath):
    """
    Load var_fa_ann.tsv (Functional Assay Annotations).
    ✅ ENHANCED: Better relationship handling, bulk operations, drug matching
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    missing_drugs_global = set()
    inserted = 0
    skipped = 0
    errors = []
    
    logger.info(f"[var_fa_ann] Starting import from {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            
            total_rows = sum(1 for _ in open(filepath, 'r', encoding='utf-8')) - 1
            logger.info(f"[var_fa_ann] Processing {total_rows} rows")
            
            for idx, row in enumerate(reader, 1):
                va_id = row.get("Variant Annotation ID", "").strip()
                
                if not va_id:
                    skipped += 1
                    continue
                
                try:
                    # Ensure parent VariantAnnotation exists
                    va = db.session.get(VariantAnnotation, va_id)
                    if not va:
                        va = VariantAnnotation(variant_annotation_id=va_id)
                        db.session.add(va)
                        db.session.flush()
                    
                    # Get or create VariantFAAnn (one-to-one relationship)
                    vfa = db.session.query(VariantFAAnn).filter_by(variant_annotation_id=va_id).first()
                    if not vfa:
                        vfa = VariantFAAnn(variant_annotation_id=va_id)
                        db.session.add(vfa)
                    
                    # ✅ IMPORTANT: Set all fields including PMID and Phenotype Category
                    vfa.pmid = row.get("PMID", "").strip() or None
                    vfa.phenotype_category = row.get("Phenotype Category", "").strip() or None
                    vfa.significance = row.get("Significance", "").strip() or None
                    vfa.notes = row.get("Notes", "").strip() or None
                    vfa.sentence = row.get("Sentence", "").strip() or None
                    vfa.alleles = row.get("Alleles", "").strip() or None
                    vfa.specialty_population = row.get("Specialty Population", "").strip() or None
                    vfa.assay_type = row.get("Assay type", "").strip() or None
                    vfa.metabolizer_types = row.get("Metabolizer types", "").strip() or None
                    vfa.is_plural = row.get("isPlural", "").strip() or None
                    vfa.is_associated = row.get("Is/Is Not associated", "").strip() or None
                    vfa.direction_of_effect = row.get("Direction of effect", "").strip() or None
                    vfa.functional_terms = row.get("Functional terms", "").strip() or None
                    vfa.gene_product = row.get("Gene/gene product", "").strip() or None
                    vfa.when_treated_with = row.get("When treated with/exposed to/when assayed with", "").strip() or None
                    vfa.multiple_drugs = row.get("Multiple drugs And/or", "").strip() or None
                    vfa.cell_type = row.get("Cell type", "").strip() or None
                    vfa.comparison_alleles = row.get("Comparison Allele(s) or Genotype(s)", "").strip() or None
                    vfa.comparison_metabolizer_types = row.get("Comparison Metabolizer types", "").strip() or None
                    
                    db.session.flush()
                    
                    # ⚠️ CRITICAL: Clear existing relationships before adding new ones
                    # This prevents duplicates on re-imports
                    db.session.query(VariantAnnotationDrug).filter_by(variant_annotation_id=va_id).delete()
                    db.session.query(VariantAnnotationGene).filter_by(variant_annotation_id=va_id).delete()
                    db.session.query(VariantAnnotationVariant).filter_by(variant_annotation_id=va_id).delete()
                    
                    # Process drugs - Only link to existing drugs in your DB
                    raw_drugs = row.get("Drug(s)", "").strip()
                    if raw_drugs:
                        drug_names = safe_split(raw_drugs, ';')
                        found_drugs, missing = check_drugs_exist(drug_names, va_id)
                        missing_drugs_global.update(missing)
                        
                        for drug in found_drugs:
                            va_drug = VariantAnnotationDrug(
                                variant_annotation_id=va_id,
                                drug_id=drug.id
                            )
                            db.session.add(va_drug)
                    
                    # Process genes
                    raw_genes = row.get("Gene", "").strip()
                    if raw_genes:
                        for gene_symbol in safe_split(raw_genes, ';'):
                            gene = get_or_create_gene(gene_symbol)
                            if gene:
                                va_gene = VariantAnnotationGene(
                                    variant_annotation_id=va_id,
                                    gene_id=gene.gene_id
                                )
                                db.session.add(va_gene)
                    
                    # Process variants/haplotypes
                    raw_variants = row.get("Variant/Haplotypes", "").strip()
                    if raw_variants:
                        for var_name in safe_split(raw_variants, ';'):
                            var = get_or_create_variant(var_name)
                            if var:
                                va_var = VariantAnnotationVariant(
                                    variant_annotation_id=va_id,
                                    variant_id=var.id
                                )
                                db.session.add(va_var)
                    
                    inserted += 1
                    
                    # Periodic commits
                    if inserted % 500 == 0:
                        db.session.commit()
                        progress = (idx * 100) // total_rows
                        logger.info(f"[var_fa_ann] Progress: {idx}/{total_rows} ({progress}%) - {inserted} inserted")
                
                except Exception as e:
                    error_msg = f"Row {va_id} (line {idx}) failed: {str(e)}"
                    logger.error(f"[var_fa_ann] {error_msg}")
                    errors.append(error_msg)
                    db.session.rollback()
                    skipped += 1
            
            db.session.commit()
    
    except Exception as e:
        logger.error(f"[var_fa_ann] Fatal error: {str(e)}")
        db.session.rollback()
        raise
    
    logger.info(f"[var_fa_ann] ✓ Complete: Inserted={inserted}, Skipped={skipped}, Errors={len(errors)}")
    if missing_drugs_global:
        logger.warning(f"[var_fa_ann] ⚠️  Missing drugs ({len(missing_drugs_global)}): {', '.join(sorted(missing_drugs_global)[:20])}")
    
    return inserted, skipped, missing_drugs_global

def load_var_drug_ann_tsv(filepath):
    """
    Load var_drug_ann.tsv (Drug Annotations).
    ✅ ENHANCED: Better handling of drug relationships and data validation
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    missing_drugs_global = set()
    inserted = 0
    skipped = 0
    errors = []
    
    logger.info(f"[var_drug_ann] Starting import from {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            
            total_rows = sum(1 for _ in open(filepath, 'r', encoding='utf-8')) - 1
            logger.info(f"[var_drug_ann] Processing {total_rows} rows")
            
            for idx, row in enumerate(reader, 1):
                va_id = row.get("Variant Annotation ID", "").strip()
                
                if not va_id:
                    skipped += 1
                    continue
                
                try:
                    # Ensure parent VariantAnnotation exists
                    va = db.session.get(VariantAnnotation, va_id)
                    if not va:
                        va = VariantAnnotation(variant_annotation_id=va_id)
                        db.session.add(va)
                        db.session.flush()
                    
                    # Get or create VariantDrugAnn (one-to-one relationship)
                    vda = db.session.query(VariantDrugAnn).filter_by(variant_annotation_id=va_id).first()
                    if not vda:
                        vda = VariantDrugAnn(variant_annotation_id=va_id)
                        db.session.add(vda)
                    
                    # ✅ Set all fields
                    vda.pmid = row.get("PMID", "").strip() or None
                    vda.phenotype_category = row.get("Phenotype Category", "").strip() or None
                    vda.significance = row.get("Significance", "").strip() or None
                    vda.notes = row.get("Notes", "").strip() or None
                    vda.sentence = row.get("Sentence", "").strip() or None
                    vda.alleles = row.get("Alleles", "").strip() or None
                    vda.specialty_population = row.get("Specialty Population", "").strip() or None
                    vda.metabolizer_types = row.get("Metabolizer types", "").strip() or None
                    vda.is_plural = row.get("isPlural", "").strip() or None
                    vda.is_associated = row.get("Is/Is Not associated", "").strip() or None
                    vda.direction_of_effect = row.get("Direction of effect", "").strip() or None
                    vda.pd_pk_terms = row.get("PD/PK terms", "").strip() or None
                    vda.multiple_drugs = row.get("Multiple drugs And/or", "").strip() or None
                    vda.population_types = row.get("Population types", "").strip() or None
                    vda.population_phenotypes_diseases = row.get("Population Phenotypes or diseases", "").strip() or None
                    vda.multiple_phenotypes_diseases = row.get("Multiple phenotypes or diseases And/or", "").strip() or None
                    vda.comparison_alleles = row.get("Comparison Allele(s) or Genotype(s)", "").strip() or None
                    vda.comparison_metabolizer_types = row.get("Comparison Metabolizer types", "").strip() or None
                    
                    db.session.flush()
                    
                    # Process drugs (clear existing first)
                    raw_drugs = row.get("Drug(s)", "").strip()
                    if raw_drugs:
                        # Clear existing drug relationships for this annotation
                        db.session.query(VariantAnnotationDrug).filter_by(variant_annotation_id=va_id).delete()
                        
                        drug_names = safe_split(raw_drugs, ';')
                        found_drugs, missing = check_drugs_exist(drug_names, va_id)
                        missing_drugs_global.update(missing)
                        
                        for drug in found_drugs:
                            va_drug = VariantAnnotationDrug(
                                variant_annotation_id=va_id,
                                drug_id=drug.id
                            )
                            db.session.add(va_drug)
                    
                    # Process genes (if present in this file)
                    raw_genes = row.get("Gene", "").strip()
                    if raw_genes:
                        db.session.query(VariantAnnotationGene).filter_by(variant_annotation_id=va_id).delete()
                        
                        for gene_symbol in safe_split(raw_genes, ';'):
                            gene = get_or_create_gene(gene_symbol)
                            if gene:
                                va_gene = VariantAnnotationGene(
                                    variant_annotation_id=va_id,
                                    gene_id=gene.gene_id
                                )
                                db.session.add(va_gene)
                    
                    # Process variants (if present)
                    raw_variants = row.get("Variant/Haplotypes", "").strip()
                    if raw_variants:
                        db.session.query(VariantAnnotationVariant).filter_by(variant_annotation_id=va_id).delete()
                        
                        for var_name in safe_split(raw_variants, ';'):
                            var = get_or_create_variant(var_name)
                            if var:
                                va_var = VariantAnnotationVariant(
                                    variant_annotation_id=va_id,
                                    variant_id=var.id
                                )
                                db.session.add(va_var)
                    
                    inserted += 1
                    
                    # Periodic commits
                    if inserted % 500 == 0:
                        db.session.commit()
                        progress = (idx * 100) // total_rows
                        logger.info(f"[var_drug_ann] Progress: {idx}/{total_rows} ({progress}%) - {inserted} inserted")
                
                except Exception as e:
                    error_msg = f"Row {va_id} (line {idx}) failed: {str(e)}"
                    logger.error(f"[var_drug_ann] {error_msg}")
                    errors.append(error_msg)
                    db.session.rollback()
                    skipped += 1
            
            db.session.commit()
    
    except Exception as e:
        logger.error(f"[var_drug_ann] Fatal error: {str(e)}")
        db.session.rollback()
        raise
    
    logger.info(f"[var_drug_ann] ✓ Complete: Inserted={inserted}, Skipped={skipped}, Errors={len(errors)}")
    if missing_drugs_global:
        logger.warning(f"[var_drug_ann] ⚠️  Missing drugs ({len(missing_drugs_global)}): {', '.join(sorted(missing_drugs_global)[:20])}")
    
    return inserted, skipped, missing_drugs_global

def load_var_pheno_ann_tsv(filepath):
    """
    Load var_pheno_ann.tsv (Phenotype Annotations).
    ✅ ENHANCED: Complete field mapping and relationship handling
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    missing_drugs_global = set()
    inserted = 0
    skipped = 0
    errors = []
    
    logger.info(f"[var_pheno_ann] Starting import from {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            
            total_rows = sum(1 for _ in open(filepath, 'r', encoding='utf-8')) - 1
            logger.info(f"[var_pheno_ann] Processing {total_rows} rows")
            
            for idx, row in enumerate(reader, 1):
                va_id = row.get("Variant Annotation ID", "").strip()
                
                if not va_id:
                    skipped += 1
                    continue
                
                try:
                    # Ensure parent VariantAnnotation exists
                    va = db.session.get(VariantAnnotation, va_id)
                    if not va:
                        va = VariantAnnotation(variant_annotation_id=va_id)
                        db.session.add(va)
                        db.session.flush()
                    
                    # Get or create VariantPhenoAnn (one-to-one relationship)
                    vpa = db.session.query(VariantPhenoAnn).filter_by(variant_annotation_id=va_id).first()
                    if not vpa:
                        vpa = VariantPhenoAnn(variant_annotation_id=va_id)
                        db.session.add(vpa)
                    
                    # ✅ Set all fields
                    vpa.pmid = row.get("PMID", "").strip() or None
                    vpa.phenotype_category = row.get("Phenotype Category", "").strip() or None
                    vpa.significance = row.get("Significance", "").strip() or None
                    vpa.notes = row.get("Notes", "").strip() or None
                    vpa.sentence = row.get("Sentence", "").strip() or None
                    vpa.alleles = row.get("Alleles", "").strip() or None
                    vpa.specialty_population = row.get("Specialty Population", "").strip() or None
                    vpa.metabolizer_types = row.get("Metabolizer types", "").strip() or None
                    vpa.is_plural = row.get("isPlural", "").strip() or None
                    vpa.is_associated = row.get("Is/Is Not associated", "").strip() or None
                    vpa.direction_of_effect = row.get("Direction of effect", "").strip() or None
                    vpa.side_effect_efficacy_other = row.get("Side effect/efficacy/other", "").strip() or None
                    vpa.phenotype = row.get("Phenotype", "").strip() or None
                    vpa.multiple_phenotypes = row.get("Multiple phenotypes And/or", "").strip() or None
                    vpa.when_treated_with = row.get("When treated with/exposed to/when assayed with", "").strip() or None
                    vpa.multiple_drugs = row.get("Multiple drugs And/or", "").strip() or None
                    vpa.population_types = row.get("Population types", "").strip() or None
                    vpa.population_phenotypes_diseases = row.get("Population Phenotypes or diseases", "").strip() or None
                    vpa.multiple_phenotypes_diseases = row.get("Multiple phenotypes or diseases And/or", "").strip() or None
                    vpa.comparison_alleles = row.get("Comparison Allele(s) or Genotype(s)", "").strip() or None
                    vpa.comparison_metabolizer_types = row.get("Comparison Metabolizer types", "").strip() or None
                    
                    db.session.flush()
                    
                    # Process drugs (clear existing first)
                    raw_drugs = row.get("Drug(s)", "").strip()
                    if raw_drugs:
                        db.session.query(VariantAnnotationDrug).filter_by(variant_annotation_id=va_id).delete()
                        
                        drug_names = safe_split(raw_drugs, ';')
                        found_drugs, missing = check_drugs_exist(drug_names, va_id)
                        missing_drugs_global.update(missing)
                        
                        for drug in found_drugs:
                            va_drug = VariantAnnotationDrug(
                                variant_annotation_id=va_id,
                                drug_id=drug.id
                            )
                            db.session.add(va_drug)
                    
                    # Process genes (if present)
                    raw_genes = row.get("Gene", "").strip()
                    if raw_genes:
                        db.session.query(VariantAnnotationGene).filter_by(variant_annotation_id=va_id).delete()
                        
                        for gene_symbol in safe_split(raw_genes, ';'):
                            gene = get_or_create_gene(gene_symbol)
                            if gene:
                                va_gene = VariantAnnotationGene(
                                    variant_annotation_id=va_id,
                                    gene_id=gene.gene_id
                                )
                                db.session.add(va_gene)
                    
                    # Process variants (if present)
                    raw_variants = row.get("Variant/Haplotypes", "").strip()
                    if raw_variants:
                        db.session.query(VariantAnnotationVariant).filter_by(variant_annotation_id=va_id).delete()
                        
                        for var_name in safe_split(raw_variants, ';'):
                            var = get_or_create_variant(var_name)
                            if var:
                                va_var = VariantAnnotationVariant(
                                    variant_annotation_id=va_id,
                                    variant_id=var.id
                                )
                                db.session.add(va_var)
                    
                    inserted += 1
                    
                    # Periodic commits
                    if inserted % 500 == 0:
                        db.session.commit()
                        progress = (idx * 100) // total_rows
                        logger.info(f"[var_pheno_ann] Progress: {idx}/{total_rows} ({progress}%) - {inserted} inserted")
                
                except Exception as e:
                    error_msg = f"Row {va_id} (line {idx}) failed: {str(e)}"
                    logger.error(f"[var_pheno_ann] {error_msg}")
                    errors.append(error_msg)
                    db.session.rollback()
                    skipped += 1
            
            db.session.commit()
    
    except Exception as e:
        logger.error(f"[var_pheno_ann] Fatal error: {str(e)}")
        db.session.rollback()
        raise
    
    logger.info(f"[var_pheno_ann] ✓ Complete: Inserted={inserted}, Skipped={skipped}, Errors={len(errors)}")
    if missing_drugs_global:
        logger.warning(f"[var_pheno_ann] ⚠️  Missing drugs ({len(missing_drugs_global)}): {', '.join(sorted(missing_drugs_global)[:20])}")
    
    return inserted, skipped, missing_drugs_global

# ============================================================================
# ETL FUNCTIONS - RELATIONSHIPS (ENHANCED)
# ============================================================================

def load_relationships_tsv(filepath):
    """
    Load relationships.tsv with enhanced validation and entity tracking.
    ✅ ENHANCED: Better entity validation, duplicate handling, progress tracking
    
    This table links entities (drugs, genes, variants, diseases) together with
    evidence of their relationships (e.g., "warfarin associated with CYP2C9").
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    missing_entities = {
        'chemicals': set(),
        'genes': set(),
        'variants': set(),
        'diseases': set()
    }
    inserted = 0
    skipped = 0
    errors = []
    
    logger.info(f"[relationships] Starting import from {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            
            # Validate headers
            required_cols = ["Entity1_id", "Entity1_name", "Entity1_type", 
                           "Entity2_id", "Entity2_name", "Entity2_type"]
            if not all(col in reader.fieldnames for col in required_cols):
                raise ValueError(f"Missing required columns: {required_cols}")
            
            total_rows = sum(1 for _ in open(filepath, 'r', encoding='utf-8')) - 1
            logger.info(f"[relationships] Processing {total_rows} rows")
            
            for idx, row in enumerate(reader, 1):
                entity1_id = row.get("Entity1_id", "").strip()
                entity2_id = row.get("Entity2_id", "").strip()
                
                if not entity1_id or not entity2_id:
                    skipped += 1
                    continue
                
                try:
                    entity1_name = row.get("Entity1_name", "").strip()
                    entity1_type = row.get("Entity1_type", "").strip().lower()
                    entity2_name = row.get("Entity2_name", "").strip()
                    entity2_type = row.get("Entity2_type", "").strip().lower()
                    
                    # ✅ ENHANCED: Validate entities exist in database
                    entity_valid = True
                    
                    # Check Entity 1
                    if entity1_type == "chemical":
                        if entity1_name:
                            found_drugs, missing = check_drugs_exist([entity1_name], f"E1:{entity1_id}")
                            if not found_drugs:
                                missing_entities['chemicals'].add(entity1_name)
                                entity_valid = False
                    
                    elif entity1_type == "gene":
                        if entity1_name:
                            gene = db.session.query(Gene).filter_by(gene_symbol=entity1_name).first()
                            if not gene:
                                missing_entities['genes'].add(entity1_name)
                                # Don't fail - gene might be created later or not in our scope
                    
                    elif entity1_type == "variant":
                        if entity1_name:
                            variant = db.session.query(Variant).filter(
                                or_(
                                    Variant.name == entity1_name,
                                    Variant.pharmgkb_id == entity1_id
                                )
                            ).first()
                            if not variant:
                                missing_entities['variants'].add(entity1_name)
                    
                    elif entity1_type == "disease":
                        if entity1_name:
                            pheno = db.session.query(Phenotype).filter_by(name=entity1_name).first()
                            if not pheno:
                                missing_entities['diseases'].add(entity1_name)
                    
                    # Check Entity 2
                    if entity2_type == "chemical":
                        if entity2_name:
                            found_drugs, missing = check_drugs_exist([entity2_name], f"E2:{entity2_id}")
                            if not found_drugs:
                                missing_entities['chemicals'].add(entity2_name)
                                entity_valid = False
                    
                    elif entity2_type == "gene":
                        if entity2_name:
                            gene = db.session.query(Gene).filter_by(gene_symbol=entity2_name).first()
                            if not gene:
                                missing_entities['genes'].add(entity2_name)
                    
                    elif entity2_type == "variant":
                        if entity2_name:
                            variant = db.session.query(Variant).filter(
                                or_(
                                    Variant.name == entity2_name,
                                    Variant.pharmgkb_id == entity2_id
                                )
                            ).first()
                            if not variant:
                                missing_entities['variants'].add(entity2_name)
                    
                    elif entity2_type == "disease":
                        if entity2_name:
                            pheno = db.session.query(Phenotype).filter_by(name=entity2_name).first()
                            if not pheno:
                                missing_entities['diseases'].add(entity2_name)
                    
                    # ✅ CRITICAL: Skip if chemical (drug) is missing
                    # We want to maintain referential integrity for drugs
                    if not entity_valid:
                        skipped += 1
                        logger.debug(f"Skipping relationship {entity1_id}-{entity2_id}: missing drug entity")
                        continue
                    
                    # ✅ Check for duplicates before inserting
                    existing = db.session.query(Relationship).filter_by(
                        entity1_id=entity1_id,
                        entity2_id=entity2_id
                    ).first()
                    
                    if existing:
                        # Update existing relationship
                        rel = existing
                        logger.debug(f"Updating existing relationship: {entity1_id}-{entity2_id}")
                    else:
                        # Create new relationship
                        rel = Relationship()
                        db.session.add(rel)
                    
                    # Set/update all fields
                    rel.entity1_id = entity1_id
                    rel.entity1_name = entity1_name or None
                    rel.entity1_type = entity1_type or None
                    rel.entity2_id = entity2_id
                    rel.entity2_name = entity2_name or None
                    rel.entity2_type = entity2_type or None
                    rel.evidence = row.get("Evidence", "").strip() or None
                    rel.association = row.get("Association", "").strip() or None
                    rel.pk = row.get("PK", "").strip() or None
                    rel.pd = row.get("PD", "").strip() or None
                    rel.pmids = row.get("PMIDs", "").strip() or None
                    
                    inserted += 1
                    
                    # Periodic commits
                    if inserted % 500 == 0:
                        db.session.commit()
                        progress = (idx * 100) // total_rows
                        logger.info(f"[relationships] Progress: {idx}/{total_rows} ({progress}%) - {inserted} inserted")
                
                except Exception as e:
                    error_msg = f"Row {entity1_id}-{entity2_id} (line {idx}) failed: {str(e)}"
                    logger.error(f"[relationships] {error_msg}")
                    errors.append(error_msg)
                    db.session.rollback()
                    skipped += 1
            
            db.session.commit()
    
    except Exception as e:
        logger.error(f"[relationships] Fatal error: {str(e)}")
        db.session.rollback()
        raise
    
    # Summary with detailed entity tracking
    logger.info(f"[relationships] ✓ Complete: Inserted={inserted}, Skipped={skipped}, Errors={len(errors)}")
    
    total_missing = sum(len(v) for v in missing_entities.values())
    if total_missing > 0:
        logger.warning(f"[relationships] ⚠️  Missing entities summary:")
        logger.warning(f"  - Missing chemicals/drugs: {len(missing_entities['chemicals'])}")
        logger.warning(f"  - Missing genes: {len(missing_entities['genes'])}")
        logger.warning(f"  - Missing variants: {len(missing_entities['variants'])}")
        logger.warning(f"  - Missing diseases: {len(missing_entities['diseases'])}")
        
        # Log sample of missing drugs (most important)
        if missing_entities['chemicals']:
            sample = list(missing_entities['chemicals'])[:10]
            logger.warning(f"  - Sample missing drugs: {', '.join(sample)}")
    
    # Flatten for return value
    all_missing = set()
    for entity_set in missing_entities.values():
        all_missing.update(entity_set)
    
    return inserted, skipped, all_missing

# ============================================================================
# ETL FUNCTIONS - DRUG LABELS (ENHANCED)
# ============================================================================

def load_drug_labels_tsv(filepath):
    """
    Load drugLabels.tsv with enhanced drug name parsing and validation.
    ✅ ENHANCED: Better chemical name parsing, duplicate prevention, progress tracking
    
    This file contains FDA/EMA/HCSC drug labels with pharmacogenomic information.
    Chemicals field may contain: "drug1; drug2" or "drug1 / drug2" combinations.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    missing_drugs_global = set()
    inserted = 0
    skipped = 0
    errors = []
    
    logger.info(f"[drug_labels] Starting import from {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            
            # Validate headers
            required_cols = ["PharmGKB ID", "Name", "Source"]
            if not all(col in reader.fieldnames for col in required_cols):
                raise ValueError(f"Missing required columns: {required_cols}")
            
            total_rows = sum(1 for _ in open(filepath, 'r', encoding='utf-8')) - 1
            logger.info(f"[drug_labels] Processing {total_rows} rows")
            
            for idx, row in enumerate(reader, 1):
                pharmgkb_id = row.get("PharmGKB ID", "").strip()
                
                if not pharmgkb_id:
                    skipped += 1
                    continue
                
                try:
                    # Get or create DrugLabel
                    dl = db.session.get(DrugLabel, pharmgkb_id)
                    if not dl:
                        dl = DrugLabel(pharmgkb_id=pharmgkb_id)
                        db.session.add(dl)
                        logger.debug(f"Created new DrugLabel: {pharmgkb_id}")
                    
                    # Update all fields
                    dl.name = row.get("Name", "").strip() or None
                    dl.source = row.get("Source", "").strip() or None
                    dl.biomarker_flag = row.get("Biomarker Flag", "").strip() or None
                    dl.testing_level = row.get("Testing Level", "").strip() or None
                    dl.has_prescribing_info = row.get("Has Prescribing Info", "").strip() or None
                    dl.has_dosing_info = row.get("Has Dosing Info", "").strip() or None
                    dl.has_alternate_drug = row.get("Has Alternate Drug", "").strip() or None
                    dl.has_other_prescribing_guidance = row.get("Has Other Prescribing Guidance", "").strip() or None
                    dl.cancer_genome = row.get("Cancer Genome", "").strip() or None
                    dl.prescribing = row.get("Prescribing", "").strip() or None
                    dl.latest_history_date = parse_date(row.get("Latest History Date (YYYY-MM-DD)"), pharmgkb_id)
                    
                    db.session.flush()
                    
                    # ⚠️ CRITICAL: Clear existing relationships (prevents duplicates on re-import)
                    db.session.query(DrugLabelDrug).filter_by(pharmgkb_id=pharmgkb_id).delete()
                    db.session.query(DrugLabelGene).filter_by(pharmgkb_id=pharmgkb_id).delete()
                    db.session.query(DrugLabelVariant).filter_by(pharmgkb_id=pharmgkb_id).delete()
                    
                    # ✅ ENHANCED: Process chemicals/drugs with better parsing
                    raw_chems = row.get("Chemicals", "").strip()
                    if raw_chems:
                        # Split by both ';' and '/' (common in combination drugs)
                        # e.g., "abacavir / lamivudine" or "drug1; drug2"
                        chem_list = []
                        
                        # First split by ';'
                        for part in raw_chems.split(';'):
                            # Then split each part by '/'
                            for chem in part.split('/'):
                                chem_clean = chem.strip().lower()
                                if chem_clean:
                                    chem_list.append(chem_clean)
                        
                        if chem_list:
                            found_drugs, missing = check_drugs_exist(chem_list, pharmgkb_id)
                            missing_drugs_global.update(missing)
                            
                            for drug in found_drugs:
                                dl_drug = DrugLabelDrug(
                                    pharmgkb_id=pharmgkb_id,
                                    drug_id=drug.id
                                )
                                db.session.add(dl_drug)
                                logger.debug(f"Linked drug {drug.name_en} to label {pharmgkb_id}")
                    
                    # Process genes
                    raw_genes = row.get("Genes", "").strip()
                    if raw_genes:
                        for gene_symbol in safe_split(raw_genes, ';'):
                            gene = get_or_create_gene(gene_symbol)
                            if gene:
                                dl_gene = DrugLabelGene(
                                    pharmgkb_id=pharmgkb_id,
                                    gene_id=gene.gene_id
                                )
                                db.session.add(dl_gene)
                    
                    # Process variants/haplotypes
                    raw_variants = row.get("Variants/Haplotypes", "").strip()
                    if raw_variants:
                        for var_name in safe_split(raw_variants, ';'):
                            var = get_or_create_variant(var_name)
                            if var:
                                dl_var = DrugLabelVariant(
                                    pharmgkb_id=pharmgkb_id,
                                    variant_id=var.id
                                )
                                db.session.add(dl_var)
                    
                    inserted += 1
                    
                    # Periodic commits
                    if inserted % 500 == 0:
                        db.session.commit()
                        progress = (idx * 100) // total_rows
                        logger.info(f"[drug_labels] Progress: {idx}/{total_rows} ({progress}%) - {inserted} inserted")
                
                except Exception as e:
                    error_msg = f"Row {pharmgkb_id} (line {idx}) failed: {str(e)}"
                    logger.error(f"[drug_labels] {error_msg}")
                    errors.append(error_msg)
                    db.session.rollback()
                    skipped += 1
            
            db.session.commit()
    
    except Exception as e:
        logger.error(f"[drug_labels] Fatal error: {str(e)}")
        db.session.rollback()
        raise
    
    logger.info(f"[drug_labels] ✓ Complete: Inserted={inserted}, Skipped={skipped}, Errors={len(errors)}")
    if missing_drugs_global:
        logger.warning(f"[drug_labels] ⚠️  Missing drugs ({len(missing_drugs_global)}): {', '.join(sorted(missing_drugs_global)[:20])}")
    
    return inserted, skipped, missing_drugs_global

def load_drug_labels_byGene_tsv(filepath):
    """
    Load drugLabels.byGene.tsv - supplementary gene-to-label mappings.
    ✅ ENHANCED: Better validation, duplicate prevention, orphan detection
    
    This file provides additional gene-to-drug-label relationships that may
    not be present in the main drugLabels.tsv file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    inserted = 0
    skipped = 0
    orphan_labels = set()
    orphan_genes = set()
    errors = []
    
    logger.info(f"[drug_labels_byGene] Starting import from {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            
            # Validate headers
            required_cols = ["Gene ID", "Gene Symbol", "Label IDs"]
            if not all(col in reader.fieldnames for col in required_cols):
                raise ValueError(f"Missing required columns: {required_cols}")
            
            total_rows = sum(1 for _ in open(filepath, 'r', encoding='utf-8')) - 1
            logger.info(f"[drug_labels_byGene] Processing {total_rows} rows")
            
            for idx, row in enumerate(reader, 1):
                gene_id = row.get("Gene ID", "").strip()
                gene_symbol = row.get("Gene Symbol", "").strip()
                label_ids = row.get("Label IDs", "").strip()
                
                if not gene_id or not gene_symbol or not label_ids:
                    skipped += 1
                    continue
                
                try:
                    # Ensure gene exists (create if missing)
                    gene = get_or_create_gene(gene_symbol, gene_id)
                    if not gene:
                        orphan_genes.add(f"{gene_id}:{gene_symbol}")
                        skipped += 1
                        continue
                    
                    # Process label IDs (semicolon-separated)
                    pharmgkb_ids = [lid.strip() for lid in label_ids.split(';') if lid.strip()]
                    
                    for pharmgkb_id in pharmgkb_ids:
                        # Check if DrugLabel exists
                        dl = db.session.get(DrugLabel, pharmgkb_id)
                        if not dl:
                            orphan_labels.add(pharmgkb_id)
                            logger.debug(f"DrugLabel {pharmgkb_id} not found for Gene {gene_symbol}")
                            continue
                        
                        # ✅ Check if relationship already exists (prevent duplicates)
                        existing = db.session.query(DrugLabelGene).filter_by(
                            pharmgkb_id=pharmgkb_id,
                            gene_id=gene.gene_id
                        ).first()
                        
                        if not existing:
                            dlg = DrugLabelGene(
                                pharmgkb_id=pharmgkb_id,
                                gene_id=gene.gene_id
                            )
                            db.session.add(dlg)
                            inserted += 1
                            logger.debug(f"Linked gene {gene_symbol} to label {pharmgkb_id}")
                        else:
                            logger.debug(f"Relationship already exists: {gene_symbol} <-> {pharmgkb_id}")
                    
                    # Periodic commits
                    if inserted % 500 == 0:
                        db.session.commit()
                        progress = (idx * 100) // total_rows
                        logger.info(f"[drug_labels_byGene] Progress: {idx}/{total_rows} ({progress}%) - {inserted} inserted")
                
                except Exception as e:
                    error_msg = f"Row {gene_id} (line {idx}) failed: {str(e)}"
                    logger.error(f"[drug_labels_byGene] {error_msg}")
                    errors.append(error_msg)
                    db.session.rollback()
                    skipped += 1
            
            db.session.commit()
    
    except Exception as e:
        logger.error(f"[drug_labels_byGene] Fatal error: {str(e)}")
        db.session.rollback()
        raise
    
    logger.info(f"[drug_labels_byGene] ✓ Complete: Inserted={inserted}, Skipped={skipped}, Errors={len(errors)}")
    
    # Report orphaned data
    if orphan_labels:
        logger.warning(f"[drug_labels_byGene] ⚠️  Orphan labels (not in drugLabels.tsv): {len(orphan_labels)}")
        logger.warning(f"  Sample: {', '.join(list(orphan_labels)[:10])}")
    
    if orphan_genes:
        logger.warning(f"[drug_labels_byGene] ⚠️  Orphan genes (failed to create): {len(orphan_genes)}")
        logger.warning(f"  Sample: {', '.join(list(orphan_genes)[:10])}")
    
    return inserted, skipped, set()


# ============================================================================
# ETL FUNCTIONS - CLINICAL VARIANTS (ENHANCED)
# ============================================================================

def load_clinical_variants_tsv(filepath):
    """
    Load clinicalVariants.tsv with enhanced duplicate handling and validation.
    ✅ ENHANCED: Duplicate prevention, better parsing, composite key handling
    
    This file contains high-level clinical variant-drug-phenotype associations.
    Each row represents a clinical finding about a variant's effect on drug response.
    
    Note: ClinicalVariant has auto-incrementing ID, so we must handle duplicates
    by checking if same variant+gene+drug+phenotype combination exists.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    missing_drugs_global = set()
    inserted = 0
    updated = 0
    skipped = 0
    errors = []
    
    logger.info(f"[clinical_variants] Starting import from {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            
            # Validate headers
            required_cols = ["variant", "gene", "level of evidence"]
            if not all(col in reader.fieldnames for col in required_cols):
                raise ValueError(f"Missing required columns: {required_cols}")
            
            total_rows = sum(1 for _ in open(filepath, 'r', encoding='utf-8')) - 1
            logger.info(f"[clinical_variants] Processing {total_rows} rows")
            
            for idx, row in enumerate(reader, 1):
                variant_name = row.get("variant", "").strip()
                gene_symbol = row.get("gene", "").strip()
                
                if not variant_name or not gene_symbol:
                    skipped += 1
                    continue
                
                try:
                    # Validate/create gene
                    gene = get_or_create_gene(gene_symbol)
                    if not gene:
                        logger.warning(f"Failed to create gene: {gene_symbol}")
                        skipped += 1
                        continue
                    
                    # Get or create variant
                    var = get_or_create_variant(variant_name)
                    if not var:
                        logger.warning(f"Failed to create variant: {variant_name}")
                        skipped += 1
                        continue
                    
                    # ✅ CRITICAL: Check for existing ClinicalVariant with same composite key
                    # Since there's no unique constraint, we check manually
                    variant_type = row.get("type", "").strip() or None
                    level_of_evidence = row.get("level of evidence", "").strip() or None
                    
                    # Try to find existing by variant ID + gene ID
                    existing_cvs = db.session.query(ClinicalVariant).filter_by(
                        gene_id=gene.gene_id
                    ).join(ClinicalVariantVariant).filter(
                        ClinicalVariantVariant.variant_id == var.id
                    ).all()
                    
                    # Find exact match if multiple exist
                    cv = None
                    for existing_cv in existing_cvs:
                        if (existing_cv.variant_type == variant_type and 
                            existing_cv.level_of_evidence == level_of_evidence):
                            cv = existing_cv
                            logger.debug(f"Found existing ClinicalVariant ID {cv.id}")
                            break
                    
                    if not cv:
                        # Create new ClinicalVariant
                        cv = ClinicalVariant(
                            variant_type=variant_type,
                            level_of_evidence=level_of_evidence,
                            gene_id=gene.gene_id
                        )
                        db.session.add(cv)
                        db.session.flush()  # Get the auto-generated ID
                        logger.debug(f"Created new ClinicalVariant ID {cv.id}")
                        
                        # Add variant relationship (for new CV only)
                        cv_var = ClinicalVariantVariant(
                            clinical_variant_id=cv.id,
                            variant_id=var.id
                        )
                        db.session.add(cv_var)
                        inserted += 1
                    else:
                        # Update existing
                        cv.variant_type = variant_type
                        cv.level_of_evidence = level_of_evidence
                        updated += 1
                    
                    db.session.flush()
                    
                    # ⚠️ Clear existing relationships (for both new and existing)
                    db.session.query(ClinicalVariantDrug).filter_by(
                        clinical_variant_id=cv.id
                    ).delete()
                    db.session.query(ClinicalVariantPhenotype).filter_by(
                        clinical_variant_id=cv.id
                    ).delete()
                    
                    # ✅ Process chemicals/drugs with better parsing
                    raw_chems = row.get("chemicals", "").strip()
                    if raw_chems:
                        # Split by both '/' and ',' (some entries use different delimiters)
                        drug_list = []
                        for part in raw_chems.split(','):
                            for chem in part.split('/'):
                                chem_clean = chem.strip().lower()
                                if chem_clean:
                                    drug_list.append(chem_clean)
                        
                        if drug_list:
                            found_drugs, missing = check_drugs_exist(
                                drug_list, 
                                f"{variant_name}-{gene_symbol}"
                            )
                            missing_drugs_global.update(missing)
                            
                            for drug in found_drugs:
                                cv_drug = ClinicalVariantDrug(
                                    clinical_variant_id=cv.id,
                                    drug_id=drug.id
                                )
                                db.session.add(cv_drug)
                    
                    # Process phenotypes (comma-separated)
                    raw_phenos = row.get("phenotypes", "").strip()
                    if raw_phenos:
                        for pheno_name in safe_split(raw_phenos, delimiter=','):
                            pheno = get_or_create_phenotype(pheno_name)
                            if pheno:
                                cv_pheno = ClinicalVariantPhenotype(
                                    clinical_variant_id=cv.id,
                                    phenotype_id=pheno.id
                                )
                                db.session.add(cv_pheno)
                    
                    # Periodic commits
                    if (inserted + updated) % 500 == 0:
                        db.session.commit()
                        progress = (idx * 100) // total_rows
                        logger.info(f"[clinical_variants] Progress: {idx}/{total_rows} ({progress}%) - {inserted} new, {updated} updated")
                
                except Exception as e:
                    error_msg = f"Row {variant_name}/{gene_symbol} (line {idx}) failed: {str(e)}"
                    logger.error(f"[clinical_variants] {error_msg}")
                    errors.append(error_msg)
                    db.session.rollback()
                    skipped += 1
            
            db.session.commit()
    
    except Exception as e:
        logger.error(f"[clinical_variants] Fatal error: {str(e)}")
        db.session.rollback()
        raise
    
    logger.info(f"[clinical_variants] ✓ Complete: Inserted={inserted}, Updated={updated}, Skipped={skipped}, Errors={len(errors)}")
    if missing_drugs_global:
        logger.warning(f"[clinical_variants] ⚠️  Missing drugs ({len(missing_drugs_global)}): {', '.join(sorted(missing_drugs_global)[:20])}")
    
    return inserted, skipped, missing_drugs_global


# ============================================================================
# ETL FUNCTIONS - OCCURRENCES (ENHANCED)
# ============================================================================

def load_occurrences_tsv(filepath):
    """
    Load occurrences.tsv with enhanced duplicate handling and entity validation.
    ✅ ENHANCED: Duplicate prevention, entity tracking, bulk operations
    
    This file tracks where entities (drugs, genes, variants, diseases) are mentioned
    in literature sources (PMIDs, PMC IDs). It's a many-to-many relationship between
    sources and objects.
    
    Example: "PMID:12345678 mentions Chemical:warfarin"
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    missing_entities = {
        'chemicals': set(),
        'genes': set(),
        'variants': set(),
        'haplotypes': set(),
        'diseases': set()
    }
    inserted = 0
    duplicates = 0
    skipped = 0
    errors = []
    
    logger.info(f"[occurrences] Starting import from {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            
            # Validate headers
            required_cols = ["Source Type", "Source ID", "Object Type", "Object ID"]
            if not all(col in reader.fieldnames for col in required_cols):
                raise ValueError(f"Missing required columns: {required_cols}")
            
            total_rows = sum(1 for _ in open(filepath, 'r', encoding='utf-8')) - 1
            logger.info(f"[occurrences] Processing {total_rows} rows")
            
            # ✅ Batch collection for bulk insert
            batch = []
            seen_pairs = set()  # Track (source_id, object_id) to prevent duplicates in this run
            
            for idx, row in enumerate(reader, 1):
                source_id = row.get("Source ID", "").strip()
                object_id = row.get("Object ID", "").strip()
                
                if not source_id or not object_id:
                    skipped += 1
                    continue
                
                try:
                    source_type = row.get("Source Type", "").strip()
                    source_name = row.get("Source Name", "").strip()
                    object_type = row.get("Object Type", "").strip()
                    object_name = row.get("Object Name", "").strip()
                    
                    # ✅ CRITICAL: Check for duplicate in this batch
                    pair_key = (source_id, object_id)
                    if pair_key in seen_pairs:
                        duplicates += 1
                        continue
                    
                    # ✅ Check if already exists in database
                    existing = db.session.query(Occurrence).filter_by(
                        source_id=source_id,
                        object_id=object_id
                    ).first()
                    
                    if existing:
                        duplicates += 1
                        seen_pairs.add(pair_key)
                        continue
                    
                    # ✅ ENHANCED: Validate and track entities
                    source_type_lower = source_type.lower()
                    object_type_lower = object_type.lower()
                    
                    # Track source entities
                    if source_type_lower == "chemical" and source_name:
                        found_drugs, missing = check_drugs_exist([source_name], f"S:{source_id}")
                        if not found_drugs:
                            missing_entities['chemicals'].add(source_name)
                    
                    elif source_type_lower == "gene" and source_name:
                        gene = db.session.query(Gene).filter_by(gene_symbol=source_name).first()
                        if not gene:
                            missing_entities['genes'].add(source_name)
                    
                    elif source_type_lower in ("variant", "haplotype") and source_name:
                        variant = db.session.query(Variant).filter(
                            or_(
                                Variant.name == source_name,
                                Variant.pharmgkb_id == source_id
                            )
                        ).first()
                        if not variant:
                            missing_entities[source_type_lower + 's'].add(source_name)
                    
                    elif source_type_lower == "disease" and source_name:
                        pheno = db.session.query(Phenotype).filter_by(name=source_name).first()
                        if not pheno:
                            missing_entities['diseases'].add(source_name)
                    
                    # Track object entities
                    if object_type_lower == "chemical" and object_name:
                        found_drugs, missing = check_drugs_exist([object_name], f"O:{object_id}")
                        if not found_drugs:
                            missing_entities['chemicals'].add(object_name)
                    
                    elif object_type_lower == "gene" and object_name:
                        gene = db.session.query(Gene).filter_by(gene_symbol=object_name).first()
                        if not gene:
                            missing_entities['genes'].add(object_name)
                    
                    elif object_type_lower in ("variant", "haplotype") and object_name:
                        variant = db.session.query(Variant).filter(
                            or_(
                                Variant.name == object_name,
                                Variant.pharmgkb_id == object_id
                            )
                        ).first()
                        if not variant:
                            missing_entities[object_type_lower + 's'].add(object_name)
                    
                    elif object_type_lower == "disease" and object_name:
                        pheno = db.session.query(Phenotype).filter_by(name=object_name).first()
                        if not pheno:
                            missing_entities['diseases'].add(object_name)
                    
                    # ✅ Create Occurrence object for batch
                    occ = Occurrence(
                        source_type=source_type or None,
                        source_id=source_id,
                        source_name=source_name or None,
                        object_type=object_type or None,
                        object_id=object_id,
                        object_name=object_name or None
                    )
                    
                    batch.append(occ)
                    seen_pairs.add(pair_key)
                    inserted += 1
                    
                    # ✅ Bulk insert when batch is full
                    if len(batch) >= 1000:
                        db.session.bulk_save_objects(batch)
                        db.session.commit()
                        batch = []
                        progress = (idx * 100) // total_rows
                        logger.info(f"[occurrences] Progress: {idx}/{total_rows} ({progress}%) - {inserted} inserted, {duplicates} duplicates")
                
                except Exception as e:
                    error_msg = f"Row {source_id}-{object_id} (line {idx}) failed: {str(e)}"
                    logger.error(f"[occurrences] {error_msg}")
                    errors.append(error_msg)
                    skipped += 1
            
            # ✅ Insert remaining batch
            if batch:
                db.session.bulk_save_objects(batch)
                db.session.commit()
    
    except Exception as e:
        logger.error(f"[occurrences] Fatal error: {str(e)}")
        db.session.rollback()
        raise
    
    # Summary with detailed entity tracking
    logger.info(f"[occurrences] ✓ Complete: Inserted={inserted}, Duplicates={duplicates}, Skipped={skipped}, Errors={len(errors)}")
    
    total_missing = sum(len(v) for v in missing_entities.values())
    if total_missing > 0:
        logger.warning(f"[occurrences] ⚠️  Missing entities summary:")
        logger.warning(f"  - Missing chemicals/drugs: {len(missing_entities['chemicals'])}")
        logger.warning(f"  - Missing genes: {len(missing_entities['genes'])}")
        logger.warning(f"  - Missing variants: {len(missing_entities['variants'])}")
        logger.warning(f"  - Missing haplotypes: {len(missing_entities['haplotypes'])}")
        logger.warning(f"  - Missing diseases: {len(missing_entities['diseases'])}")
        
        # Log sample of missing entities
        if missing_entities['chemicals']:
            sample = list(missing_entities['chemicals'])[:10]
            logger.warning(f"  - Sample missing drugs: {', '.join(sample)}")
    
    # Flatten for return value
    all_missing = set()
    for entity_set in missing_entities.values():
        all_missing.update(entity_set)
    
    return inserted, skipped, all_missing

# ============================================================================
# ETL FUNCTIONS - AUTOMATED ANNOTATIONS (ENHANCED)
# ============================================================================

def load_automated_annotations_tsv(filepath):
    """
    Load automated_annotations.tsv with enhanced bulk processing and validation.
    ✅ ENHANCED: Bulk inserts, duplicate prevention, better entity tracking
    
    This file contains automated text-mining results that link chemicals, genes,
    and variants mentioned in scientific literature. Generated by NLP tools.
    
    Note: This is typically a VERY LARGE file (100K+ rows), so bulk operations
    and memory management are critical.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    missing_entities = {
        'chemicals': set(),
        'genes': set(),
        'variants': set()
    }
    inserted = 0
    duplicates = 0
    skipped = 0
    errors = []
    
    logger.info(f"[automated_annotations] Starting import from {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            
            # Validate headers
            required_cols = ["PMID", "Sentence", "Source"]
            if not all(col in reader.fieldnames for col in required_cols):
                raise ValueError(f"Missing required columns: {required_cols}")
            
            total_rows = sum(1 for _ in open(filepath, 'r', encoding='utf-8')) - 1
            logger.info(f"[automated_annotations] Processing {total_rows} rows")
            
            # ✅ Batch collection for bulk insert
            batch = []
            seen_keys = set()  # Track duplicates in this batch
            
            # ✅ Cache for publications (avoid repeated queries)
            pub_cache = {}
            
            for idx, row in enumerate(reader, 1):
                pmid = row.get("PMID", "").strip() or None
                lit_title = row.get("Literature Title", "").strip() or None
                sentence = row.get("Sentence", "").strip() or None
                
                # ✅ Relaxed validation - at least one field must exist
                if not (pmid or lit_title or sentence):
                    skipped += 1
                    continue
                
                try:
                    chem_id = row.get("Chemical ID", "").strip() or None
                    chem_name = row.get("Chemical Name", "").strip() or None
                    var_id = row.get("Variation ID", "").strip() or None
                    gene_ids = row.get("Gene IDs", "").strip() or None
                    
                    # ✅ Create composite key for duplicate detection
                    # Use combination of PMID + Chemical + Variation + Sentence hash
                    key_parts = [
                        pmid or "NO_PMID",
                        chem_id or chem_name or "NO_CHEM",
                        var_id or "NO_VAR",
                        sentence[:50] if sentence else "NO_SENT"  # First 50 chars
                    ]
                    composite_key = "|".join(key_parts)
                    
                    if composite_key in seen_keys:
                        duplicates += 1
                        continue
                    
                    # ✅ ENHANCED: Validate chemical/drug
                    if chem_name:
                        found_drugs, missing = check_drugs_exist(
                            [chem_name.lower()], 
                            f"PMID:{pmid or 'unknown'}"
                        )
                        if not found_drugs:
                            missing_entities['chemicals'].add(chem_name)
                    
                    # ✅ Track genes (optional validation)
                    gene_symbols = row.get("Gene Symbols", "").strip() or None
                    if gene_symbols:
                        for gene_symbol in safe_split(gene_symbols, ','):
                            gene = db.session.query(Gene).filter_by(
                                gene_symbol=gene_symbol
                            ).first()
                            if not gene:
                                missing_entities['genes'].add(gene_symbol)
                    
                    # ✅ Track variants (optional validation)
                    var_name = row.get("Variation Name", "").strip() or None
                    if var_name:
                        variant = db.session.query(Variant).filter(
                            or_(
                                Variant.name == var_name,
                                Variant.pharmgkb_id == var_id
                            )
                        ).first()
                        if not variant:
                            missing_entities['variants'].add(var_name)
                    
                    # ✅ Create/cache Publication if PMID exists
                    if pmid:
                        if pmid not in pub_cache:
                            pub = get_or_create_publication(
                                pmid,
                                lit_title,
                                row.get("Publication Year", "").strip() or None,
                                row.get("Journal", "").strip() or None
                            )
                            pub_cache[pmid] = pub
                    
                    # ✅ Create AutomatedAnnotation object
                    ann = AutomatedAnnotation(
                        chemical_id=chem_id,
                        chemical_name=chem_name,
                        chemical_in_text=row.get("Chemical in Text", "").strip() or None,
                        variation_id=var_id,
                        variation_name=var_name,
                        variation_type=row.get("Variation Type", "").strip() or None,
                        variation_in_text=row.get("Variation in Text", "").strip() or None,
                        gene_ids=gene_ids,
                        gene_symbols=gene_symbols,
                        gene_in_text=row.get("Gene in Text", "").strip() or None,
                        literature_id=row.get("Literature ID", "").strip() or None,
                        pmid=pmid,
                        literature_title=lit_title,
                        publication_year=row.get("Publication Year", "").strip() or None,
                        journal=row.get("Journal", "").strip() or None,
                        sentence=sentence,
                        source=row.get("Source", "").strip() or None
                    )
                    
                    batch.append(ann)
                    seen_keys.add(composite_key)
                    inserted += 1
                    
                    # ✅ Bulk insert when batch is full
                    if len(batch) >= 1000:
                        db.session.bulk_save_objects(batch)
                        db.session.commit()
                        batch = []
                        progress = (idx * 100) // total_rows
                        logger.info(f"[automated_annotations] Progress: {idx}/{total_rows} ({progress}%) - {inserted} inserted, {duplicates} duplicates")
                
                except Exception as e:
                    error_msg = f"Row PMID:{pmid} (line {idx}) failed: {str(e)}"
                    logger.error(f"[automated_annotations] {error_msg}")
                    errors.append(error_msg)
                    skipped += 1
            
            # ✅ Insert remaining batch
            if batch:
                db.session.bulk_save_objects(batch)
                db.session.commit()
    
    except Exception as e:
        logger.error(f"[automated_annotations] Fatal error: {str(e)}")
        db.session.rollback()
        raise
    
    # Summary with detailed entity tracking
    logger.info(f"[automated_annotations] ✓ Complete: Inserted={inserted}, Duplicates={duplicates}, Skipped={skipped}, Errors={len(errors)}")
    
    total_missing = sum(len(v) for v in missing_entities.values())
    if total_missing > 0:
        logger.warning(f"[automated_annotations] ⚠️  Missing entities summary:")
        logger.warning(f"  - Missing chemicals/drugs: {len(missing_entities['chemicals'])}")
        logger.warning(f"  - Missing genes: {len(missing_entities['genes'])}")
        logger.warning(f"  - Missing variants: {len(missing_entities['variants'])}")
        
        # Log sample of missing entities
        if missing_entities['chemicals']:
            sample = list(missing_entities['chemicals'])[:15]
            logger.warning(f"  - Sample missing drugs: {', '.join(sample)}")
    
    # Flatten for return value
    all_missing = set()
    for entity_set in missing_entities.values():
        all_missing.update(entity_set)
    
    return inserted, skipped, all_missing


# ============================================================================
# FLASK ROUTES - ENHANCED WITH ASYNC PROCESSING & PROGRESS TRACKING
# ============================================================================

upload_progress_data = {}  # Changed from upload_progress

def background_upload(upload_id, files_config, user_session_id):
    """
    Background task for processing uploads with progress tracking.
    ✅ FIXED: Added app context for database operations in thread
    """
    # ✅ CRITICAL: Add Flask app context for database operations
    with app.app_context():
        total_files = len(files_config)
        processed = 0
        
        upload_progress_data[upload_id] = {
            'status': 'processing',
            'current_file': None,
            'progress': 0,
            'total_files': total_files,
            'processed_files': 0,
            'results': {},
            'errors': [],
            'started_at': datetime.utcnow().isoformat()
        }
        
        try:
            for key, (file_path, func) in files_config.items():
                upload_progress_data[upload_id]['current_file'] = key
                upload_progress_data[upload_id]['progress'] = int((processed / total_files) * 100)
                
                try:
                    logger.info(f"[{upload_id}] Processing {key}...")
                    inserted, skipped, missing = func(file_path)
                    
                    upload_progress_data[upload_id]['results'][key] = {
                        'inserted': inserted,
                        'skipped': skipped,
                        'missing': len(missing) if isinstance(missing, (set, list)) else 0,
                        'status': 'success'
                    }
                    
                    logger.info(f"[{upload_id}] ✓ {key}: {inserted} inserted, {skipped} skipped")
                    
                except Exception as e:
                    error_msg = f"{key}: {str(e)}"
                    logger.error(f"[{upload_id}] ✗ {error_msg}")
                    upload_progress_data[upload_id]['errors'].append(error_msg)
                    upload_progress_data[upload_id]['results'][key] = {
                        'status': 'error',
                        'error': str(e)
                    }
                    db.session.rollback()
                
                finally:
                    # Clean up file
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                    except Exception as e:
                        logger.warning(f"Failed to delete {file_path}: {str(e)}")
                
                processed += 1
                upload_progress_data[upload_id]['processed_files'] = processed
            
            # Mark as completed
            upload_progress_data[upload_id]['status'] = 'completed'
            upload_progress_data[upload_id]['progress'] = 100
            upload_progress_data[upload_id]['completed_at'] = datetime.utcnow().isoformat()
            
        except Exception as e:
            logger.error(f"[{upload_id}] Fatal error: {str(e)}")
            upload_progress_data[upload_id]['status'] = 'failed'
            upload_progress_data[upload_id]['error'] = str(e)
            db.session.rollback()

# ============================================================================
# CLINICAL ANNOTATIONS ROUTES
# ============================================================================

@app.route("/upload_clinical_annotations", methods=["GET", "POST"])
def upload_clinical_annotations():
    """
    Upload Clinical Annotations files with async processing.
    ✅ ENHANCED: Background processing, progress tracking, better error handling
    """
    if request.method == "GET":
        return render_template("upload_clinical_annotations.html")
    
    # Generate unique upload ID
    upload_id = f"clinical_ann_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    
    files_map = {
        "clinical_annotations": ("clinical_annotations", load_clinical_annotations_tsv),
        "clinical_ann_history": ("clinical_ann_history", load_clinical_ann_history_tsv),
        "clinical_ann_alleles": ("clinical_ann_alleles", load_clinical_ann_alleles_tsv),
        "clinical_ann_evidence": ("clinical_ann_evidence", load_clinical_ann_evidence_tsv)
    }
    
    files_config = {}
    uploaded_count = 0
    
    # Save all files first
    for key, (display_name, func) in files_map.items():
        file = request.files.get(key)
        if file and file.filename:
            try:
                path = save_uploaded_file(file)
                files_config[display_name] = (path, func)
                uploaded_count += 1
            except Exception as e:
                logger.error(f"Error saving {key}: {str(e)}")
                flash(f"✗ Error saving {display_name}.tsv: {str(e)}", "danger")
    
    if not files_config:
        flash("⚠️  No valid files uploaded", "warning")
        return redirect(url_for("upload_clinical_annotations"))
    
    # Start background processing
    thread = threading.Thread(
        target=background_upload,
        args=(upload_id, files_config, session.get('user_id', 'anonymous'))
    )
    thread.daemon = True
    thread.start()
    
    flash(f"✓ {uploaded_count} file(s) queued for processing. Track progress below.", "info")
    
    return jsonify({
        'status': 'queued',
        'upload_id': upload_id,
        'files_count': uploaded_count,
        'progress_url': url_for('upload_progress_api', upload_id=upload_id)
    })

# ============================================================================
# VARIANT ANNOTATIONS ROUTES
# ============================================================================

@app.route("/upload_variant_annotations", methods=["GET", "POST"])
def upload_variant_annotations():
    """Upload Variant Annotations files with async processing."""
    if request.method == "GET":
        return render_template("upload_variant_annotations.html")
    
    upload_id = f"var_ann_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    
    files_map = {
        "study_parameters": ("study_parameters", load_study_parameters_tsv),
        "var_fa_ann": ("var_fa_ann", load_var_fa_ann_tsv),
        "var_drug_ann": ("var_drug_ann", load_var_drug_ann_tsv),
        "var_pheno_ann": ("var_pheno_ann", load_var_pheno_ann_tsv)
    }
    
    files_config = {}
    uploaded_count = 0
    
    for key, (display_name, func) in files_map.items():
        file = request.files.get(key)
        if file and file.filename:
            try:
                path = save_uploaded_file(file)
                files_config[display_name] = (path, func)
                uploaded_count += 1
            except Exception as e:
                logger.error(f"Error saving {key}: {str(e)}")
                flash(f"✗ Error saving {display_name}.tsv: {str(e)}", "danger")
    
    if not files_config:
        flash("⚠️  No valid files uploaded", "warning")
        return redirect(url_for("upload_variant_annotations"))
    
    thread = threading.Thread(
        target=background_upload,
        args=(upload_id, files_config, session.get('user_id', 'anonymous'))
    )
    thread.daemon = True
    thread.start()
    
    flash(f"✓ {uploaded_count} file(s) queued for processing.", "info")
    
    return jsonify({
        'status': 'queued',
        'upload_id': upload_id,
        'files_count': uploaded_count,
        'progress_url': url_for('upload_progress_api', upload_id=upload_id)
    })

# ============================================================================
# RELATIONSHIPS ROUTES
# ============================================================================

@app.route("/upload_relationships", methods=["GET", "POST"])
def upload_relationships():
    """Upload Relationships file with async processing."""
    if request.method == "GET":
        return render_template("upload_relationships.html")
    
    upload_id = f"relationships_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    
    rel_file = request.files.get("relationships_file")
    if not rel_file or not rel_file.filename:
        flash("⚠️  No file uploaded", "warning")
        return redirect(url_for("upload_relationships"))
    
    try:
        path = save_uploaded_file(rel_file)
        files_config = {"relationships": (path, load_relationships_tsv)}
        
        thread = threading.Thread(
            target=background_upload,
            args=(upload_id, files_config, session.get('user_id', 'anonymous'))
        )
        thread.daemon = True
        thread.start()
        
        flash("✓ File queued for processing.", "info")
        
        return jsonify({
            'status': 'queued',
            'upload_id': upload_id,
            'progress_url': url_for('upload_progress_api', upload_id=upload_id)
        })
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        flash(f"✗ Error: {str(e)}", "danger")
        return redirect(url_for("upload_relationships"))

# ============================================================================
# DRUG LABELS ROUTES
# ============================================================================

@app.route("/upload_drug_labels", methods=["GET", "POST"])
def upload_drug_labels():
    """Upload Drug Labels files with async processing."""
    if request.method == "GET":
        return render_template("upload_drug_labels.html")
    
    upload_id = f"drug_labels_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    
    files_map = {
        "drug_labels": ("drug_labels", load_drug_labels_tsv),
        "drug_labels_byGene": ("drug_labels_byGene", load_drug_labels_byGene_tsv)
    }
    
    files_config = {}
    uploaded_count = 0
    
    for key, (display_name, func) in files_map.items():
        file = request.files.get(key)
        if file and file.filename:
            try:
                path = save_uploaded_file(file)
                files_config[display_name] = (path, func)
                uploaded_count += 1
            except Exception as e:
                logger.error(f"Error saving {key}: {str(e)}")
                flash(f"✗ Error saving {display_name}.tsv: {str(e)}", "danger")
    
    if not files_config:
        flash("⚠️  No valid files uploaded", "warning")
        return redirect(url_for("upload_drug_labels"))
    
    thread = threading.Thread(
        target=background_upload,
        args=(upload_id, files_config, session.get('user_id', 'anonymous'))
    )
    thread.daemon = True
    thread.start()
    
    flash(f"✓ {uploaded_count} file(s) queued for processing.", "info")
    
    return jsonify({
        'status': 'queued',
        'upload_id': upload_id,
        'files_count': uploaded_count,
        'progress_url': url_for('upload_progress_api', upload_id=upload_id)
    })

# ============================================================================
# CLINICAL VARIANTS ROUTES
# ============================================================================

@app.route("/upload_clinical_variants", methods=["GET", "POST"])
def upload_clinical_variants():
    """Upload Clinical Variants file with async processing."""
    if request.method == "GET":
        return render_template("upload_clinical_variants.html")
    
    upload_id = f"clinical_variants_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    
    cv_file = request.files.get("clinical_variants_file")
    if not cv_file or not cv_file.filename:
        flash("⚠️  No file uploaded", "warning")
        return redirect(url_for("upload_clinical_variants"))
    
    try:
        path = save_uploaded_file(cv_file)
        files_config = {"clinical_variants": (path, load_clinical_variants_tsv)}
        
        thread = threading.Thread(
            target=background_upload,
            args=(upload_id, files_config, session.get('user_id', 'anonymous'))
        )
        thread.daemon = True
        thread.start()
        
        flash("✓ File queued for processing.", "info")
        
        return jsonify({
            'status': 'queued',
            'upload_id': upload_id,
            'progress_url': url_for('upload_progress_api', upload_id=upload_id)
        })
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        flash(f"✗ Error: {str(e)}", "danger")
        return redirect(url_for("upload_clinical_variants"))

# ============================================================================
# OCCURRENCES ROUTES
# ============================================================================

@app.route("/upload_occurrences", methods=["GET", "POST"])
def upload_occurrences():
    """Upload Occurrences file with async processing."""
    if request.method == "GET":
        return render_template("upload_occurrences.html")
    
    upload_id = f"occurrences_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    
    occ_file = request.files.get("occurrences_file")
    if not occ_file or not occ_file.filename:
        flash("⚠️  No file uploaded", "warning")
        return redirect(url_for("upload_occurrences"))
    
    try:
        path = save_uploaded_file(occ_file)
        files_config = {"occurrences": (path, load_occurrences_tsv)}
        
        thread = threading.Thread(
            target=background_upload,
            args=(upload_id, files_config, session.get('user_id', 'anonymous'))
        )
        thread.daemon = True
        thread.start()
        
        flash("✓ File queued for processing.", "info")
        
        return jsonify({
            'status': 'queued',
            'upload_id': upload_id,
            'progress_url': url_for('upload_progress_api', upload_id=upload_id)
        })
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        flash(f"✗ Error: {str(e)}", "danger")
        return redirect(url_for("upload_occurrences"))

# ============================================================================
# AUTOMATED ANNOTATIONS ROUTES
# ============================================================================

@app.route("/upload_automated_annotations", methods=["GET", "POST"])
def upload_automated_annotations():
    """Upload Automated Annotations file with async processing."""
    if request.method == "GET":
        return render_template("upload_automated_annotations.html")
    
    upload_id = f"auto_ann_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    
    auto_file = request.files.get("automated_file")
    if not auto_file or not auto_file.filename:
        flash("⚠️  No file uploaded", "warning")
        return redirect(url_for("upload_automated_annotations"))
    
    try:
        path = save_uploaded_file(auto_file)
        files_config = {"automated_annotations": (path, load_automated_annotations_tsv)}
        
        thread = threading.Thread(
            target=background_upload,
            args=(upload_id, files_config, session.get('user_id', 'anonymous'))
        )
        thread.daemon = True
        thread.start()
        
        flash("✓ File queued for processing (this may take 10-15 minutes for large files).", "info")
        
        return jsonify({
            'status': 'queued',
            'upload_id': upload_id,
            'progress_url': url_for('upload_progress_api', upload_id=upload_id)
        })
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        flash(f"✗ Error: {str(e)}", "danger")
        return redirect(url_for("upload_automated_annotations"))

# ============================================================================
# PROGRESS TRACKING API
# ============================================================================

@app.route("/api/upload_progress/<upload_id>")
def upload_progress_api(upload_id):
    """
    API endpoint to check upload progress.
    Returns JSON with current status, progress, and results.
    """
    if upload_id not in upload_progress_data:
        return jsonify({'error': 'Upload ID not found'}), 404
    
    return jsonify(upload_progress_data[upload_id])

@app.route("/upload_progress/<upload_id>")
def upload_progress_page(upload_id):
    """
    HTML page to view upload progress with auto-refresh.
    """
    return render_template("upload_progress.html", upload_id=upload_id)

# ============================================================================
# CLEANUP OLD PROGRESS DATA (OPTIONAL CRON JOB)
# ============================================================================

@app.route("/api/cleanup_progress", methods=["POST"])
def cleanup_progress():
    """
    Clean up progress data older than 24 hours.
    Should be called by cron job or scheduled task.
    """
    cutoff = datetime.utcnow().timestamp() - (24 * 3600)
    cleaned = 0
    
    for upload_id in list(upload_progress_data.keys()):
        data = upload_progress_data[upload_id]
        if 'started_at' in data:
            started = datetime.fromisoformat(data['started_at']).timestamp()
            if started < cutoff:
                del upload_progress_data[upload_id]
                cleaned += 1
    
    return jsonify({'cleaned': cleaned, 'remaining': len(upload_progress_data)})


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
@app.route("/search_variants", methods=["GET"])
def search_variants():
    """
    Search for variants across multiple tables.
    ✅ MINOR OPTIMIZATION: Added caching hint and limit validation
    """
    query = request.args.get("q", "").strip().lower()
    page = request.args.get("page", 1, type=int)
    limit = min(request.args.get("limit", 10, type=int), 50)  # ✅ Cap at 50
    offset = (page - 1) * limit
    
    if not query or len(query) < 2:  # ✅ Minimum query length
        return jsonify({
            "results": [],
            "pagination": {"more": False}
        })
    
    with app.app_context():
        variants = set()
        
        try:
            # Query ClinicalAnnotation via ClinicalAnnotationVariant
            cas = db.session.query(ClinicalAnnotation).join(
                ClinicalAnnotationVariant
            ).join(Variant).filter(
                or_(
                    Variant.name.ilike(f"%{query}%"),
                    Variant.pharmgkb_id.ilike(f"%{query}%")
                )
            ).limit(100).all()  # ✅ Limit subqueries
            variants.update(v.variant.name for ca in cas for v in ca.variants if v.variant and v.variant.name)
            
            # Query ClinicalAnnAllele
            ca_alleles = db.session.query(ClinicalAnnAllele).filter(
                ClinicalAnnAllele.genotype_allele.ilike(f"%{query}%")
            ).limit(100).all()
            variants.update(ca.genotype_allele for ca in ca_alleles if ca.genotype_allele)
            
            # Query VariantFAAnn, VariantDrugAnn, VariantPhenoAnn
            for va_type in [VariantFAAnn, VariantDrugAnn, VariantPhenoAnn]:
                vas = db.session.query(va_type).filter(
                    or_(
                        va_type.variant_annotation_id.ilike(f"%{query}%"),
                        va_type.alleles.ilike(f"%{query}%")
                    )
                ).limit(100).all()
                variants.update(va.variant_annotation_id for va in vas if va.variant_annotation_id)
            
            # Query ClinicalVariant
            cvs = db.session.query(ClinicalVariant).join(
                ClinicalVariantVariant
            ).join(Variant).filter(
                or_(
                    Variant.name.ilike(f"%{query}%"),
                    Variant.pharmgkb_id.ilike(f"%{query}%")
                )
            ).limit(100).all()
            variants.update(v.variant.name for cv in cvs for v in cv.variants if v.variant and v.variant.name)
            
            # Query AutomatedAnnotation
            autos = db.session.query(AutomatedAnnotation).filter(
                or_(
                    AutomatedAnnotation.variation_id.ilike(f"%{query}%"),
                    AutomatedAnnotation.variation_name.ilike(f"%{query}%")
                )
            ).limit(100).all()
            variants.update(auto.variation_name for auto in autos if auto.variation_name)
            
            # ✅ Sort for consistent pagination
            variant_list = sorted(list(variants))
            paginated = variant_list[offset:offset + limit]
            has_more = len(variant_list) > offset + limit
            
            logger.debug(f"search_variants: query={query}, found={len(variant_list)}, returned={len(paginated)}")
            
            return jsonify({
                "results": [{"id": v, "text": v} for v in paginated],
                "pagination": {"more": has_more}
            })
        
        except Exception as e:
            logger.error(f"Error in search_variants: {str(e)}")
            return jsonify({"error": "Search failed", "details": str(e)}), 500

# Search Phenotypes
@app.route("/search_phenotypes", methods=["GET"])
def search_phenotypes():
    """
    Search for phenotypes across multiple tables.
    ✅ MINOR OPTIMIZATION: Added caching hint and limit validation
    """
    query = request.args.get("q", "").strip().lower()
    page = request.args.get("page", 1, type=int)
    limit = min(request.args.get("limit", 10, type=int), 50)  # ✅ Cap at 50
    offset = (page - 1) * limit
    
    if not query or len(query) < 2:  # ✅ Minimum query length
        return jsonify({
            "results": [],
            "pagination": {"more": False}
        })
    
    with app.app_context():
        phenotypes = set()
        
        try:
            # Query ClinicalAnnotation via ClinicalAnnotationPhenotype
            cas = db.session.query(ClinicalAnnotation).join(
                ClinicalAnnotationPhenotype
            ).join(Phenotype).filter(
                Phenotype.name.ilike(f"%{query}%")
            ).limit(100).all()
            phenotypes.update(
                p.phenotype.name for ca in cas for p in ca.phenotypes 
                if p.phenotype and p.phenotype.name and query in p.phenotype.name.lower()
            )
            
            # Query VariantPhenoAnn
            vpas = db.session.query(VariantPhenoAnn).filter(
                VariantPhenoAnn.phenotype.ilike(f"%{query}%")
            ).limit(100).all()
            phenotypes.update(
                vpa.phenotype for vpa in vpas 
                if vpa.phenotype and query in vpa.phenotype.lower()
            )
            
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
            ).limit(100).all()
            phenotypes.update(
                rel.entity1_name if rel.entity1_type.lower() == "disease" else rel.entity2_name
                for rel in rels
                if (rel.entity1_name or rel.entity2_name) and 
                   query in (rel.entity1_name or rel.entity2_name or "").lower()
            )
            
            # Query ClinicalVariant
            cvs = db.session.query(ClinicalVariant).join(
                ClinicalVariantPhenotype
            ).join(Phenotype).filter(
                Phenotype.name.ilike(f"%{query}%")
            ).limit(100).all()
            phenotypes.update(
                p.phenotype.name for cv in cvs for p in cv.phenotypes 
                if p.phenotype and p.phenotype.name and query in p.phenotype.name.lower()
            )
            
            # ✅ Sort for consistent pagination
            phenotype_list = sorted(list(phenotypes))
            paginated = phenotype_list[offset:offset + limit]
            has_more = len(phenotype_list) > offset + limit
            
            logger.debug(f"search_phenotypes: query={query}, found={len(phenotype_list)}, returned={len(paginated)}")
            
            return jsonify({
                "results": [{"id": p, "text": p} for p in paginated],
                "pagination": {"more": has_more}
            })
        
        except Exception as e:
            logger.error(f"Error in search_phenotypes: {str(e)}")
            return jsonify({"error": "Search failed", "details": str(e)}), 500

# Search Genes
@app.route("/search_genes", methods=["GET"])
def search_genes():
    """
    Search for genes across multiple tables.
    ✅ MINOR OPTIMIZATION: Added caching hint and limit validation
    """
    with app.app_context():
        try:
            query = request.args.get("q", "").strip().lower()
            limit = min(request.args.get("limit", 10, type=int), 50)  # ✅ Cap at 50
            page = request.args.get("page", 1, type=int)
            offset = (page - 1) * limit
            
            if not query or len(query) < 2:  # ✅ Minimum query length
                return jsonify({
                    "results": [],
                    "pagination": {"more": False}
                })
            
            genes = set()
            
            # Query ClinicalAnnotation via ClinicalAnnotationGene
            cas = db.session.query(ClinicalAnnotation).join(
                ClinicalAnnotationGene
            ).join(Gene).filter(
                or_(
                    Gene.gene_symbol.ilike(f"%{query}%"),
                    Gene.gene_id.ilike(f"%{query}%")
                )
            ).limit(100).all()
            genes.update(
                g.gene.gene_symbol for ca in cas for g in ca.genes 
                if g.gene and g.gene.gene_symbol and query in g.gene.gene_symbol.lower()
            )
            
            # Query AutomatedAnnotation
            autos = db.session.query(AutomatedAnnotation).filter(
                AutomatedAnnotation.gene_ids.ilike(f"%{query}%")
            ).limit(100).all()
            genes.update(
                g.strip() for auto in autos 
                for g in safe_split(auto.gene_ids, ',') 
                if g and query in g.lower()
            )
            
            # Query DrugLabelGene
            dlgs = db.session.query(DrugLabelGene).join(Gene).filter(
                or_(
                    Gene.gene_symbol.ilike(f"%{query}%"),
                    Gene.gene_id.ilike(f"%{query}%")
                )
            ).limit(100).all()
            genes.update(
                dlg.gene.gene_symbol for dlg in dlgs 
                if dlg.gene and dlg.gene.gene_symbol and query in dlg.gene.gene_symbol.lower()
            )
            
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
            ).limit(100).all()
            genes.update(
                rel.entity1_name if rel.entity1_type.lower() == "gene" else rel.entity2_name
                for rel in rels
                if (rel.entity1_name or rel.entity2_name) and 
                   query in (rel.entity1_name or rel.entity2_name or "").lower()
            )
            
            # ✅ Sort for consistent pagination
            gene_list = sorted(list(genes))
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
                    .outerjoin(Severity, DrugInteraction.severity_id == Severity.id)  # Join with Severity
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
                        # Access severity through the relationship, with safe fallback
                        severity_name = di.severity_level.name if di.severity_level else di.predicted_severity or "Unknown"
                        
                        # Calculate score based on severity
                        if severity_name and isinstance(severity_name, str):
                            severity_lower = severity_name.lower()
                            if "severe" in severity_lower or "major" in severity_lower:
                                score = 2.0
                            elif "moderate" in severity_lower:
                                score = 1.5
                            elif "minor" in severity_lower or "mild" in severity_lower:
                                score = 1.0
                            else:
                                score = 1.0  # Default for unknown severity
                        else:
                            score = 1.0
                        
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
                                "severity": severity_name,
                                "interaction_type": di.interaction_type or "N/A",
                                "mechanism": di.mechanism or "N/A",
                                "monitoring": di.monitoring or "N/A",
                                "alternatives": di.alternatives or "N/A"
                            }
                        }
                        
                        if has_meaningful_data(prediction):
                            results["predictions"].append(prediction)
                            evidence_weight += score
                            evidence_count += 1
                            
                            # Enhanced evidence text with severity and monitoring info
                            evidence_text = f"Drug Interaction: {drug1} and {drug2} ({di.interaction_description or 'No description'})"
                            if di.monitoring:
                                evidence_text += f" | Monitoring: {di.monitoring[:100]}"
                            if severity_name:
                                evidence_text += f" | Severity: {severity_name}"
                            
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
                        .outerjoin(Drug, DrugReceptorInteraction.drug_id == Drug.id)
                        .outerjoin(Receptor, DrugReceptorInteraction.receptor_id == Receptor.id)
                        .filter(or_(False, *conditions))
                        .limit(3)
                        .all()
                    )
                    
                    for dri in query:
                        drug_name = dri.drug.name_en if dri.drug and dri.drug.name_en else "Unknown"
                        receptor_name = dri.receptor.name if dri.receptor and dri.receptor.name else "Unknown"
                        
                        if (not drug or matches_any(drug_name, drug, False)) and \
                        (not gene or matches_any(receptor_name, gene, True)):
                            
                            # Safe affinity check
                            has_affinity = dri.affinity and dri.affinity not in [None, 0, ""]
                            score = 1.5 if has_affinity else 1.0
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
                                    "affinity": str(dri.affinity) if dri.affinity else "N/A",
                                    "mechanism": dri.mechanism or "N/A"
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
                    "drug_categories": [],
                },
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "query": {"variant": variant, "drug": drug, "phenotype": phenotype, "gene": gene, "level_of_evidence": level_of_evidence}
                }
            }
            
            def apply_filters(query, model, has_joins=False):
                conditions = []
                
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
                
                elif model == StudyParameters:
                    if variant:
                        conditions.append(StudyParameters.variant_annotation_id.ilike(f"%{variant}%"))
                    if phenotype:
                        conditions.append(StudyParameters.characteristics.ilike(f"%{phenotype}%"))
                
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
            
            # Top Entities - FIXED JOIN CONDITIONS
            # Top Genes
            base_query = (db.session.query(Gene.gene_symbol, func.count(ClinicalAnnotation.clinical_annotation_id).label('count'))
                         .join(ClinicalAnnotationGene, ClinicalAnnotationGene.gene_id == Gene.gene_id)
                         .join(ClinicalAnnotation, ClinicalAnnotationGene.clinical_annotation_id == ClinicalAnnotation.clinical_annotation_id))
            top_genes_query = apply_filters(base_query, ClinicalAnnotation, has_joins=True)
            top_genes = top_genes_query.group_by(Gene.gene_symbol).order_by(func.count(ClinicalAnnotation.clinical_annotation_id).desc()).limit(5).all()
            results["stats"]["top_entities"]["genes"] = [{"name": g[0], "count": g[1]} for g in top_genes if g[0]]
            
            # Top Drugs
            base_query = (db.session.query(Drug.name_en, func.count(DrugLabel.pharmgkb_id).label('count'))
                         .join(DrugLabelDrug, DrugLabelDrug.drug_id == Drug.id)
                         .join(DrugLabel, DrugLabelDrug.pharmgkb_id == DrugLabel.pharmgkb_id))
            top_drugs_query = apply_filters(base_query, DrugLabel, has_joins=True)
            top_drugs = top_drugs_query.group_by(Drug.name_en).order_by(func.count(DrugLabel.pharmgkb_id).desc()).limit(5).all()
            results["stats"]["top_entities"]["drugs"] = [{"name": d[0], "count": d[1]} for d in top_drugs if d[0]]
            
            # Top Variants - FIX: Join on Variant.id, not Variant.pharmgkb_id
            base_query = (db.session.query(Variant.name, func.count(ClinicalAnnotation.clinical_annotation_id).label('count'))
                         .join(ClinicalAnnotationVariant, ClinicalAnnotationVariant.variant_id == Variant.id)  # FIXED HERE
                         .join(ClinicalAnnotation, ClinicalAnnotationVariant.clinical_annotation_id == ClinicalAnnotation.clinical_annotation_id))
            top_variants_query = apply_filters(base_query, ClinicalAnnotation, has_joins=True)
            top_variants = top_variants_query.group_by(Variant.name).order_by(func.count(ClinicalAnnotation.clinical_annotation_id).desc()).limit(5).all()
            results["stats"]["top_entities"]["variants"] = [{"name": v[0], "count": v[1]} for v in top_variants if v[0]]
            
            # Top Phenotypes
            base_query = (db.session.query(Phenotype.name, func.count(ClinicalAnnotation.clinical_annotation_id).label('count'))
                         .join(ClinicalAnnotationPhenotype, ClinicalAnnotationPhenotype.phenotype_id == Phenotype.id)
                         .join(ClinicalAnnotation, ClinicalAnnotationPhenotype.clinical_annotation_id == ClinicalAnnotation.clinical_annotation_id))
            phenotypes_query = apply_filters(base_query, ClinicalAnnotation, has_joins=True)
            top_phenotypes = phenotypes_query.group_by(Phenotype.name).order_by(func.count(ClinicalAnnotation.clinical_annotation_id).desc()).limit(5).all()
            results["stats"]["top_entities"]["phenotypes"] = [{"name": p[0], "count": p[1]} for p in top_phenotypes if p[0]]
            
            # Top Sources
            top_sources = apply_filters(db.session.query(Occurrence.source_type, func.count(Occurrence.source_type).label('count')), Occurrence)\
                .group_by(Occurrence.source_type).order_by(func.count(Occurrence.source_type).desc()).limit(5).all()
            results["stats"]["top_entities"]["sources"] = [{"name": s[0] or "Unknown", "count": s[1]} for s in top_sources]
            
            # Top Chemicals
            top_chemicals = apply_filters(db.session.query(AutomatedAnnotation.chemical_name, func.count(AutomatedAnnotation.chemical_name).label('count')), AutomatedAnnotation)\
                .group_by(AutomatedAnnotation.chemical_name).order_by(func.count(AutomatedAnnotation.chemical_name).desc()).limit(5).all()
            results["stats"]["top_entities"]["chemicals"] = [{"name": c[0] or "Unknown", "count": c[1]} for c in top_chemicals]
            
            # Top Alleles
            base_query = (db.session.query(ClinicalAnnAllele.genotype_allele, func.count(ClinicalAnnAllele.genotype_allele).label('count'))
                         .join(ClinicalAnnotation, ClinicalAnnAllele.clinical_annotation_id == ClinicalAnnotation.clinical_annotation_id))
            top_alleles_query = apply_filters(base_query, ClinicalAnnotation, has_joins=True)
            top_alleles = top_alleles_query.group_by(ClinicalAnnAllele.genotype_allele).order_by(func.count(ClinicalAnnAllele.genotype_allele).desc()).limit(5).all()
            results["stats"]["top_entities"]["alleles"] = [{"name": a[0], "count": a[1]} for a in top_alleles if a[0]]
            
            # Drug Category Distribution
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
            
            # Trends (keep existing code)
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
            
            # Child table statistics (keep existing code)
            top_variants_child = apply_filters(db.session.query(StudyParameters.variant_annotation_id, func.count(StudyParameters.variant_annotation_id).label('count')), StudyParameters)\
                .group_by(StudyParameters.variant_annotation_id).order_by(func.count(StudyParameters.variant_annotation_id).desc()).limit(5).all()
            results["stats"]["child_top_entities"]["variants"] = [{"name": v[0], "count": v[1]} for v in top_variants_child if v[0]]
            
            top_study_types = apply_filters(db.session.query(StudyParameters.study_type, func.count(StudyParameters.study_type).label('count')), StudyParameters)\
                .group_by(StudyParameters.study_type).order_by(func.count(StudyParameters.study_type).desc()).limit(5).all()
            results["stats"]["child_top_entities"]["study_types"] = [{"name": s[0] or "Unknown", "count": s[1]} for s in top_study_types]
            
            base_query = db.session.query(VariantFAAnn.sentence, func.count(VariantFAAnn.sentence).label('count')).join(VariantAnnotation, VariantFAAnn.variant_annotation_id == VariantAnnotation.variant_annotation_id)
            top_fa_sentences_query = apply_filters(base_query, VariantFAAnn, has_joins=True)
            top_fa_sentences = top_fa_sentences_query.group_by(VariantFAAnn.sentence).order_by(func.count(VariantFAAnn.sentence).desc()).limit(5).all()
            results["stats"]["child_top_entities"]["fa_sentences"] = [{"name": s[0] or "Unknown", "count": s[1]} for s in top_fa_sentences]
            
            base_query = db.session.query(VariantDrugAnn.sentence, func.count(VariantDrugAnn.sentence).label('count')).join(VariantAnnotation, VariantDrugAnn.variant_annotation_id == VariantAnnotation.variant_annotation_id)
            top_drug_sentences_query = apply_filters(base_query, VariantDrugAnn, has_joins=True)
            top_drug_sentences = top_drug_sentences_query.group_by(VariantDrugAnn.sentence).order_by(func.count(VariantDrugAnn.sentence).desc()).limit(5).all()
            results["stats"]["child_top_entities"]["drug_sentences"] = [{"name": s[0] or "Unknown", "count": s[1]} for s in top_drug_sentences]
            
            base_query = db.session.query(VariantPhenoAnn.phenotype, func.count(VariantPhenoAnn.phenotype).label('count')).join(VariantAnnotation, VariantPhenoAnn.variant_annotation_id == VariantAnnotation.variant_annotation_id)
            top_phenotypes_query = apply_filters(base_query, VariantPhenoAnn, has_joins=True)
            top_phenotypes_child = top_phenotypes_query.group_by(VariantPhenoAnn.phenotype).order_by(func.count(VariantPhenoAnn.phenotype).desc()).limit(5).all()
            results["stats"]["child_top_entities"]["phenotypes"] = [{"name": p[0] or "Unknown", "count": p[1]} for p in top_phenotypes_child]
            
            # Child trends
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


#ANNOUNCEMENT SECTION!!!
# Sanitize rich text fields for news routes
def clean_text(field):
    try:
        # Wrap plain text in <p> if needed
        if field.strip() and not field.startswith('<') and not field.endswith('>'):
            field = f'<p>{field}</p>'
        return bleach.clean(
            field,
            tags=['b', 'i', 'u', 'p', 'strong', 'em', 'span', 'a', 'ul', 'ol', 'li', 'img', 'table', 'tr', 'td', 'th'],
            attributes={
                'span': ['style'],
                'a': ['href'],
                'img': ['src', 'alt'],
                'font': ['face', 'size']
            },
            styles=['color', 'background-color', 'font-family', 'font-size']
        ) if field else None
    except TypeError as e:
        logger.warning(f"bleach.clean failed with styles: {str(e)}, retrying without styles")
        return bleach.clean(
            field,
            tags=['b', 'i', 'u', 'p', 'strong', 'em', 'span', 'a', 'ul', 'ol', 'li', 'img', 'table', 'tr', 'td', 'th'],
            attributes={
                'span': ['style'],
                'a': ['href'],
                'img': ['src', 'alt'],
                'font': ['face', 'size']
            }
        ) if field else None

@app.route('/news/manage', methods=['GET', 'POST'])
@admin_required
def manage_news():
    valid_categories = ['Announcement', 'Update', 'Drug Update', 'FDA Approval']  # Added!

    if request.method == 'POST':
        title = request.form.get('title')
        description = request.form.get('description')
        category = request.form.get('category')
        publication_date = request.form.get('publication_date')
        drug_id = request.form.get('drug_id')  # Can be empty

        # Validation
        if not all([title, description, category, publication_date]):
            flash("All fields are required!", "danger")
            return redirect(url_for('manage_news'))

        if category not in valid_categories:
            flash("Invalid category!", "danger")
            return redirect(url_for('manage_news'))

        if category == 'FDA Approval' and not drug_id:
            flash("Please select a drug for FDA Approval news!", "danger")
            return redirect(url_for('manage_news'))

        try:
            publication_date = datetime.strptime(publication_date, '%Y-%m-%d').date()
        except:
            flash("Invalid date format!", "danger")
            return redirect(url_for('manage_news'))

        description = clean_text(description)

        news_item = News(
            title=title.strip(),
            description=description,
            category=category,
            publication_date=publication_date,
            drug_id=int(drug_id) if drug_id and drug_id.isdigit() else None
        )

        db.session.add(news_item)
        db.session.commit()
        flash("News added successfully!", "success")
        return redirect(url_for('manage_news'))

    # Fetch all
    announcements = News.query.filter_by(category='Announcement').order_by(News.publication_date.desc()).all()
    updates = News.query.filter_by(category='Update').order_by(News.publication_date.desc()).all()
    drug_updates = News.query.filter_by(category='Drug Update').order_by(News.publication_date.desc()).all()
    fda_approvals = News.query.filter_by(category='FDA Approval').order_by(News.publication_date.desc()).all()

    return render_template(
        'manage_news.html',
        announcements=announcements,
        updates=updates,
        drug_updates=drug_updates,
        fda_approvals=fda_approvals,
        valid_categories=valid_categories,
        today=date.today()
    )

@app.route('/news/edit/<int:news_id>', methods=['GET', 'POST'])
@admin_required
def edit_news(news_id):
    news_item = News.query.get_or_404(news_id)
    valid_categories = ['Announcement', 'Update', 'Drug Update', 'FDA Approval']

    if request.method == 'POST':
        title = request.form.get('title')
        description = request.form.get('description')
        category = request.form.get('category')
        publication_date = request.form.get('publication_date')
        drug_id = request.form.get('drug_id', None)

        if not all([title, description, category, publication_date]):
            flash("All fields required!", "danger")
            return render_template('edit_news.html', news_item=news_item, valid_categories=valid_categories)

        if category == 'FDA Approval' and not drug_id:
            flash("Please select a drug for FDA Approval!", "danger")
            return render_template('edit_news.html', news_item=news_item, valid_categories=valid_categories)

        try:
            publication_date = datetime.strptime(publication_date, '%Y-%m-%d').date()
        except:
            flash("Invalid date!", "danger")
            return render_template('edit_news.html', news_item=news_item, valid_categories=valid_categories)

        news_item.title = title.strip()
        news_item.description = clean_text(description)
        news_item.category = category
        news_item.publication_date = publication_date
        news_item.drug_id = int(drug_id) if drug_id and drug_id.isdigit() else None

        db.session.commit()
        flash("News updated!", "success")
        return redirect(url_for('manage_news'))

    return render_template('edit_news.html', news_item=news_item, valid_categories=valid_categories)

@app.route('/news/delete/<int:news_id>', methods=['POST'])
@admin_required
def delete_news(news_id):
    news_item = News.query.get_or_404(news_id)
    try:
        title = news_item.title
        db.session.delete(news_item)
        db.session.commit()
        logger.info(f"News item deleted: {title} (ID: {news_id})")
        flash("News item deleted successfully.", "success")
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error deleting news item: {str(e)}")
        flash(f"Error deleting news item: {str(e)}", "danger")
    return redirect(url_for('manage_news'))

@app.route('/news')
def public_news():
    # === ADD THIS BLOCK (same as in your /about route) ===
    user = None
    user_email = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        if user:
            user_email = user.email
    # =====================================================

    # Fetch all news in reverse chronological order
    all_news = News.query.order_by(News.publication_date.desc()).all()
   
    # Separate FDA approvals for special highlighting
    fda_approvals = [n for n in all_news if n.category == 'FDA Approval']
    other_news = [n for n in all_news if n.category != 'FDA Approval']
   
    return render_template(
        'public_news.html',
        fda_approvals=fda_approvals,
        other_news=other_news,
        all_news=all_news,
        user=user,          # ← Add these two lines
        user_email=user_email  # ← Essential for the navbar logic
    )

@app.route('/news/<int:news_id>')
def news_detail(news_id):
    news = News.query.get_or_404(news_id)
    return render_template('news_detail.html', news=news)


# Doz - Cevap Simülasyonu....
# Request model (Pydantic V2)
# Updated Pydantic model
class DoseResponseRequest(BaseModel):
    emax: float
    ec50: float
    n: float
    e0: float = 0.0
    concentrations: list[float] | None = None
    log_range: dict | None = None
    dosing_regimen: str = 'single'
    doses: list[float] | None = None
    intervals: list[float] | None = None
    elimination_rate: float = 0.1
    concentration_unit: str = 'µM'
    effect_unit: str = '%'

    @field_validator('emax')
    @classmethod
    def check_emax(cls, v):
        if not 0 < v <= 1000:
            raise ValueError("Emax must be between 0 and 1000")
        return v

    @field_validator('ec50')
    @classmethod
    def check_ec50(cls, v):
        if not 0 < v <= 10000:
            raise ValueError("EC50 must be between 0 and 10000")
        return v

    @field_validator('n')
    @classmethod
    def check_hill(cls, v):
        if not 0 < v <= 10:
            raise ValueError("Hill coefficient must be between 0 and 10")
        return v

    @field_validator('e0')
    @classmethod
    def check_e0(cls, v):
        if not 0 <= v <= 1000:
            raise ValueError("Baseline effect (E0) must be between 0 and 1000")
        return v

    @field_validator('elimination_rate')
    @classmethod
    def check_elimination_rate(cls, v):
        if v is not None and not 0 < v <= 1:
            raise ValueError("Elimination rate must be between 0 and 1")
        return v

    @field_validator('concentrations')
    @classmethod
    def check_concentrations(cls, v):
        if v is None:
            return v
        if not v:
            raise ValueError("Concentrations list cannot be empty")
        if any(c <= 0 for c in v):
            raise ValueError("Concentrations must be positive")
        if len(v) != len(set(v)):
            raise ValueError("Concentrations must be unique")
        return sorted(v)

    @field_validator('log_range')
    @classmethod
    def check_log_range(cls, v):
        if v is None:
            return v
        required_keys = ['start', 'stop', 'num']
        if not all(k in v for k in required_keys):
            raise ValueError("log_range must include start, stop, and num")
        if not (0 < v['start'] < v['stop'] and v['num'] >= 10):
            raise ValueError("Invalid log_range: start must be less than stop, both positive, num >= 10")
        return v

    @field_validator('dosing_regimen')
    @classmethod
    def check_dosing_regimen(cls, v):
        if v not in ['single', 'multiple']:
            raise ValueError("Dosing regimen must be 'single' or 'multiple'")
        return v

    @field_validator('doses')
    @classmethod
    def check_doses(cls, v):
        if v and any(d <= 0 for d in v):
            raise ValueError("Doses must be positive")
        return v

    @field_validator('intervals')
    @classmethod
    def check_intervals(cls, v):
        if v and any(i <= 0 for i in v):
            raise ValueError("Intervals must be positive")
        return v

    @model_validator(mode='after')
    def validate_requirements(self):
        if self.concentrations is not None and self.log_range is not None:
            raise ValueError("Provide either concentrations or log_range, not both")
        if self.concentrations is None and self.log_range is None:
            raise ValueError("Must provide either concentrations or log_range")
        if self.dosing_regimen == 'multiple':
            if not self.doses:
                raise ValueError("Doses are required for multiple dosing")
            if not self.intervals:
                raise ValueError("Intervals are required for multiple dosing")
            if len(self.intervals) != len(self.doses) - 1:
                raise ValueError("Intervals length must be one less than doses length")
        return self


class DoseResponsePoint(BaseModel):
    concentration: float
    effect: float
    time: float | None = None


class DoseResponseResponse(BaseModel):
    data: list[DoseResponsePoint]
    metadata: dict


def calculate_hill_effect(concentration: float, emax: float, ec50: float, n: float, e0: float) -> float:
    """Calculate effect using Hill equation."""
    if concentration <= 0:
        return e0
    effect = e0 + (emax * (concentration ** n)) / ((ec50 ** n) + (concentration ** n))
    return max(min(effect, emax + e0), e0)


@lru_cache(maxsize=100)
def simulate_single_dose(emax: float, ec50: float, n: float, e0: float, concentrations: tuple) -> list:
    """Simulate single-dose response using Hill equation."""
    results = []
    for c in concentrations:
        effect = calculate_hill_effect(c, emax, ec50, n, e0)
        results.append((c, effect))
    return results


def simulate_multiple_dose(emax: float, ec50: float, n: float, e0: float, 
                          doses: list, intervals: list, elimination_rate: float) -> list:
    """Simulate multiple-dose response with accumulation and elimination."""
    dose_times = np.cumsum([0] + intervals)
    total_time = dose_times[-1] + max(intervals, default=24)
    time_points = np.linspace(0, total_time, 500)
    
    results = []
    current_concentration = 0.0
    dt = time_points[1] - time_points[0] if len(time_points) > 1 else 0.1
    
    dose_index = 0
    for t in time_points:
        # Apply dose if at dose time
        if dose_index < len(doses) and dose_index < len(dose_times):
            if abs(t - dose_times[dose_index]) < dt / 2:
                current_concentration += doses[dose_index]
                dose_index += 1
        
        # Calculate effect
        effect = calculate_hill_effect(current_concentration, emax, ec50, n, e0)
        results.append((current_concentration, effect, float(t)))
        
        # Apply elimination
        current_concentration *= np.exp(-elimination_rate * dt)
        current_concentration = max(current_concentration, 0)
    
    return results


def calculate_pseudo_r2(effects: list[float], concentrations: list[float], 
                       emax: float, ec50: float, n: float, e0: float) -> float:
    """Calculate pseudo R-squared for model fit."""
    if not effects or len(effects) < 2:
        return 1.0
    
    mean_effect = np.mean(effects)
    ss_tot = np.sum((np.array(effects) - mean_effect) ** 2)
    
    if ss_tot == 0:
        return 1.0
    
    predicted_effects = [calculate_hill_effect(c, emax, ec50, n, e0) for c in concentrations if c > 0]
    actual_effects = effects[:len(predicted_effects)]
    
    if len(actual_effects) != len(predicted_effects):
        return 1.0
    
    ss_res = np.sum((np.array(actual_effects) - np.array(predicted_effects)) ** 2)
    return max(0.0, 1 - (ss_res / ss_tot))


def generate_concentrations(log_range: dict) -> list[float]:
    """Generate concentration array from log range specification."""
    return np.logspace(
        np.log10(log_range['start']),
        np.log10(log_range['stop']),
        int(log_range['num'])
    ).tolist()


def create_simulation_metadata(simulation_id: str, input_data: DoseResponseRequest, 
                              results: list, concentrations_used: list[float], 
                              pseudo_r2: float) -> dict:
    """Create metadata dictionary for simulation response."""
    return {
        "simulation_id": simulation_id,
        "model": "Hill Equation",
        "parameters": {
            "Emax": input_data.emax,
            "EC50": input_data.ec50,
            "Hill_Coefficient": input_data.n,
            "E0": input_data.e0,
            "Dosing_Regimen": input_data.dosing_regimen,
            "Doses": input_data.doses if input_data.dosing_regimen == 'multiple' else None,
            "Intervals": input_data.intervals if input_data.dosing_regimen == 'multiple' else None,
            "Elimination_Rate": input_data.elimination_rate if input_data.dosing_regimen == 'multiple' else None
        },
        "units": {
            "concentration": input_data.concentration_unit,
            "effect": input_data.effect_unit
        },
        "concentration_range": {
            "min": min(concentrations_used) if concentrations_used else 0,
            "max": max(concentrations_used) if concentrations_used else 0
        },
        "pseudo_r2": round(pseudo_r2, 4),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "point_count": len(results)
    }


def save_simulation_to_db(simulation_id: str, input_data: DoseResponseRequest, 
                         concentrations: list[float]) -> None:
    """Save simulation to database."""
    simulation = DoseResponseSimulation(
        id=simulation_id,
        emax=input_data.emax,
        ec50=input_data.ec50,
        n=input_data.n,
        e0=input_data.e0,
        concentrations=concentrations,
        dosing_regimen=input_data.dosing_regimen,
        doses=input_data.doses,
        intervals=input_data.intervals,
        elimination_rate=input_data.elimination_rate,
        user_id=session.get('user_id'),
        concentration_unit=input_data.concentration_unit,
        effect_unit=input_data.effect_unit
    )
    db.session.add(simulation)
    db.session.commit()


@app.route('/simulate-dose-response', methods=['POST'])
def simulate_dose_response():
    try:
        request_data = request.get_json()
        if not request_data:
            logger.error("No JSON data provided in request")
            return jsonify({"error": "No data provided"}), HTTPStatus.BAD_REQUEST

        # Validate input
        input_data = DoseResponseRequest.model_validate(request_data)
        logger.info(f"Received valid request with dosing regimen: {input_data.dosing_regimen}")

        # Generate or use provided concentrations
        concentrations = (generate_concentrations(input_data.log_range) 
                         if input_data.log_range else input_data.concentrations)

        simulation_id = str(uuid.uuid4())

        # Run simulation
        if input_data.dosing_regimen == 'single':
            result_pairs = simulate_single_dose(
                input_data.emax, input_data.ec50, input_data.n, input_data.e0, 
                tuple(concentrations)
            )
            results = [
                DoseResponsePoint(concentration=c, effect=e)
                for c, e in result_pairs
            ]
        else:
            result_triples = simulate_multiple_dose(
                input_data.emax, input_data.ec50, input_data.n, input_data.e0,
                input_data.doses, input_data.intervals, input_data.elimination_rate
            )
            results = [
                DoseResponsePoint(concentration=c, effect=e, time=t)
                for c, e, t in result_triples
            ]

        # Calculate pseudo R-squared
        effects = [r.effect for r in results]
        concentrations_used = [r.concentration for r in results]
        pseudo_r2 = calculate_pseudo_r2(effects, concentrations_used, 
                                       input_data.emax, input_data.ec50, 
                                       input_data.n, input_data.e0)

        # Create metadata
        metadata = create_simulation_metadata(simulation_id, input_data, results, 
                                             concentrations_used, pseudo_r2)

        # Save to database
        save_simulation_to_db(simulation_id, input_data, concentrations)

        response = DoseResponseResponse(data=results, metadata=metadata)
        logger.info(f"Simulation completed successfully: ID {simulation_id}")
        
        return jsonify(response.model_dump()), HTTPStatus.OK

    except ValidationError as ve:
        logger.error(f"Validation error: {ve.errors()}")
        return jsonify({"error": "Invalid input", "details": ve.errors()}), HTTPStatus.BAD_REQUEST
    except Exception as e:
        logger.exception(f"Unexpected error: {str(e)}")
        db.session.rollback()
        return jsonify({"error": "Internal server error", "details": str(e)}), HTTPStatus.INTERNAL_SERVER_ERROR


@app.route('/simulate-dose-response/export/<simulation_id>', methods=['GET'])
def export_simulation(simulation_id):
    try:
        simulation = DoseResponseSimulation.query.get(simulation_id)
        if not simulation:
            return jsonify({"error": "Simulation not found"}), HTTPStatus.NOT_FOUND

        # Re-run simulation
        if simulation.dosing_regimen == 'single':
            result_pairs = simulate_single_dose(
                simulation.emax, simulation.ec50, simulation.n, simulation.e0, 
                tuple(simulation.concentrations)
            )
            results = [
                {"Concentration": c, "Effect": e}
                for c, e in result_pairs
            ]
            fieldnames = ["Concentration", "Effect"]
        else:
            result_triples = simulate_multiple_dose(
                simulation.emax, simulation.ec50, simulation.n, simulation.e0,
                simulation.doses, simulation.intervals, simulation.elimination_rate
            )
            results = [
                {"Time": t, "Concentration": c, "Effect": e}
                for c, e, t in result_triples
            ]
            fieldnames = ["Time", "Concentration", "Effect"]

        # Generate CSV
        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

        logger.info(f"Exported simulation: ID {simulation_id}")

        return Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={"Content-Disposition": f"attachment;filename=simulation_{simulation_id}.csv"}
        )

    except Exception as e:
        logger.exception(f"Export error: {str(e)}")
        return jsonify({"error": "Failed to export simulation", "details": str(e)}), HTTPStatus.INTERNAL_SERVER_ERROR


@app.route('/simulate-dose-response', methods=['GET'])
def serve_form():
    try:
        return send_from_directory(app.template_folder, 'simulate_dose_response.html')
    except Exception as e:
        logger.error(f"Error serving HTML: {str(e)}")
        return jsonify({"error": "Could not load form"}), HTTPStatus.NOT_FOUND
#Doz-Cevap Finito....





# İlaç Kategorileri
# API endpoint for categories (replacing the previous suggestion)
@app.route('/api/categories', methods=['GET'])
def get_categories():
    query = request.args.get('q', '').strip()
    limit = int(request.args.get('limit', 10))
    page = int(request.args.get('page', 1))
    offset = (page - 1) * limit

    # Fetch all categories
    categories_query = DrugCategory.query.order_by(DrugCategory.name)
    if query:
        categories_query = categories_query.filter(DrugCategory.name.ilike(f'%{query}%'))

    categories = categories_query.all()
    total = len(categories)

    # Function to get full parent hierarchy
    def get_parent_hierarchy(category):
        hierarchy = []
        current = category
        while current.parent:
            hierarchy.append(current.parent.name)
            current = current.parent
        return ' > '.join(reversed(hierarchy)) if hierarchy else None

    # Reuse get_category_name for indented display
    def get_category_name(category, depth=0):
        prefix = "  " * depth
        return f"{prefix}{category.name}"

    def build_flat_list(cats, depth=0):
        result = []
        for cat in cats:
            if not cat.parent_id or depth == 0:
                parent_hierarchy = get_parent_hierarchy(cat)
                display_name = f"{get_category_name(cat, depth)}" + (f" (Parent: {parent_hierarchy})" if parent_hierarchy else "")
                result.append((cat.id, display_name))
                result.extend(build_flat_list(cat.children, depth + 1))
        return result

    # Build flattened list with hierarchy
    flat_categories = build_flat_list(categories)

    # Apply pagination
    paginated_categories = flat_categories[offset:offset + limit]

    # Map to Select2 format
    results = [
        {
            'id': cat_id,
            'text': cat_name
        }
        for cat_id, cat_name in paginated_categories
    ]

    return jsonify({
        'results': results,
        'pagination': {'more': offset + limit < total}
    })

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
            'drug_count': cat.drugs.count(),  # Use the 'drugs' relationship to count associated drugs
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
    drug_count = category.drugs.count()  # Use the 'drugs' relationship instead of querying Drug directly
    
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

#Pharmacokinetics Module
@app.route('/metabolism', methods=['GET', 'POST'])
@login_required
def manage_metabolism():
    """Manage metabolism organs, enzymes, metabolites, and drugs."""
    organs = MetabolismOrgan.query.all()
    enzymes = MetabolismEnzyme.query.all()
    metabolites = Metabolite.query.all()
    drugs = Drug.query.all()

    if request.method == 'POST':
        try:
            if 'organ_name' in request.form:
                name = request.form.get('organ_name').strip()
                if not name:
                    flash("Organ name is required.", "error")
                elif MetabolismOrgan.query.filter_by(name=name).first():
                    flash("Organ already exists.", "error")
                else:
                    new_organ = MetabolismOrgan(name=name)
                    db.session.add(new_organ)
                    db.session.commit()
                    flash("Organ added successfully.", "success")
            elif 'enzyme_name' in request.form:
                name = request.form.get('enzyme_name').strip()
                if not name:
                    flash("Enzyme name is required.", "error")
                elif MetabolismEnzyme.query.filter_by(name=name).first():
                    flash("Enzyme already exists.", "error")
                else:
                    new_enzyme = MetabolismEnzyme(name=name)
                    db.session.add(new_enzyme)
                    db.session.commit()
                    flash("Enzyme added successfully.", "success")
            elif 'metabolite_name' in request.form:
                class MetaboliteInput(BaseModel):
                    name: str
                    parent_id: int | None = None
                    drug_id: int
                data = MetaboliteInput(**{
                    'name': request.form.get('metabolite_name').strip(),
                    'parent_id': request.form.get('parent_id', type=int),
                    'drug_id': request.form.get('drug_id', type=int)
                })
                if not data.drug_id or not data.name:
                    flash("Drug ID and metabolite name are required.", "error")
                    return redirect(url_for('manage_metabolism'))

                drug = Drug.query.get(data.drug_id)
                if not drug:
                    flash(f"Drug with ID {data.drug_id} not found.", "error")
                    return redirect(url_for('manage_metabolism'))

                parent_metabolite = Metabolite.query.filter_by(name=drug.name_en, parent_id=None, drug_id=data.drug_id).first()
                if not parent_metabolite:
                    parent_metabolite = Metabolite(name=drug.name_en, parent_id=None, drug_id=data.drug_id)
                    db.session.add(parent_metabolite)
                    db.session.flush()

                if data.parent_id:
                    parent_check = Metabolite.query.get(data.parent_id)
                    if not parent_check or parent_check.drug_id != data.drug_id:
                        flash(f"Invalid parent metabolite for drug ID {data.drug_id}.", "error")
                        return redirect(url_for('manage_metabolism'))

                if Metabolite.query.filter_by(name=data.name, parent_id=data.parent_id, drug_id=data.drug_id).first():
                    flash(f"Metabolite '{data.name}' already exists.", "error")
                else:
                    new_metabolite = Metabolite(name=data.name, parent_id=data.parent_id, drug_id=data.drug_id)
                    db.session.add(new_metabolite)
                    db.session.commit()
                    flash(f"Metabolite '{data.name}' added successfully for {drug.name_en}.", "success")
            return redirect(url_for('manage_metabolism'))
        except ValidationError as e:
            flash(f"Validation error: {str(e)}", "error")
            return redirect(url_for('manage_metabolism'))
        except Exception as e:
            db.session.rollback()
            flash("An error occurred.", "error")
            logging.error(f"Error in manage_metabolism: {str(e)}")
            return redirect(url_for('manage_metabolism'))

    return render_template('metabolism.html', organs=organs, enzymes=enzymes, metabolites=metabolites, drugs=drugs)

@app.route('/api/metabolism/organs', methods=['GET'])
def get_metabolism_organs():
    """Fetch paginated metabolism organs with search."""
    search = request.args.get('q', '').strip()
    limit = request.args.get('limit', 10, type=int)
    page = request.args.get('page', 1, type=int)

    query = MetabolismOrgan.query
    if search:
        query = query.filter(MetabolismOrgan.name.ilike(f'%{search}%'))
    paginated = query.paginate(page=page, per_page=limit)

    results = [{'id': organ.id, 'text': organ.name} for organ in paginated.items]
    return jsonify({'results': results, 'pagination': {'more': paginated.has_next, 'total': paginated.total}})

@app.route('/api/metabolism/enzymes', methods=['GET'])
def get_metabolism_enzymes():
    """Fetch paginated metabolism enzymes with search."""
    search = request.args.get('q', '').strip()
    limit = request.args.get('limit', 10, type=int)
    page = request.args.get('page', 1, type=int)

    query = MetabolismEnzyme.query
    if search:
        query = query.filter(MetabolismEnzyme.name.ilike(f'%{search}%'))
    paginated = query.paginate(page=page, per_page=limit)

    results = [{'id': enzyme.id, 'text': enzyme.name} for enzyme in paginated.items]
    return jsonify({'results': results, 'pagination': {'more': paginated.has_next, 'total': paginated.total}})

@app.route('/api/metabolites', methods=['GET'])
def get_metabolites():
    """Fetch paginated metabolites with search and drug filter."""
    search = request.args.get('q', '').strip()
    drug_id = request.args.get('drug_id', type=int)
    page = request.args.get('page', 1, type=int)
    limit = request.args.get('limit', 10, type=int)

    query = Metabolite.query
    if search:
        query = query.filter(Metabolite.name.ilike(f'%{search}%'))
    if drug_id:
        query = query.filter(Metabolite.drug_id == drug_id)
    
    paginated = query.paginate(page=page, per_page=limit)
    results = [{'id': m.id, 'text': m.name} for m in paginated.items]
    return jsonify({
        'results': results,
        'pagination': {'more': paginated.has_next, 'total': paginated.total}
    })

@app.route('/api/metabolites/full', methods=['GET'])
def get_metabolites_full():
    """Fetch full metabolite details by IDs and drug ID."""
    ids = request.args.get('ids', '').split(',')
    drug_id = request.args.get('drug_id', type=int)

    if ids == ['']:
        return jsonify([])

    try:
        ids = [int(id) for id in ids if id]
        if not ids:
            return jsonify([]), 400
        
        query = Metabolite.query.filter(Metabolite.id.in_(ids))
        if drug_id:
            if not Drug.query.get(drug_id):
                return jsonify([]), 404
            query = query.filter(Metabolite.drug_id == drug_id)

        metabolites = query.all()
        return jsonify([{'id': m.id, 'name': m.name, 'parent_id': m.parent_id} for m in metabolites])
    except Exception as e:
        logging.error(f"Error fetching metabolites: {str(e)}")
        return jsonify([]), 500

@app.route('/api/metabolites/add', methods=['POST'])
@login_required
def add_metabolite():
    """Add a new metabolite with validation."""
    class MetaboliteInput(BaseModel):
        name: str
        parent_id: int | None = None
        drug_id: int
    try:
        data = MetaboliteInput(**request.get_json())
        if Metabolite.query.filter_by(name=data.name, parent_id=data.parent_id, drug_id=data.drug_id).first():
            return jsonify({'error': 'Metabolite already exists'}), 400
        new_metabolite = Metabolite(name=data.name, parent_id=data.parent_id, drug_id=data.drug_id)
        db.session.add(new_metabolite)
        db.session.commit()
        return jsonify({'id': new_metabolite.id, 'name': new_metabolite.name, 'parent_id': new_metabolite.parent_id}), 201
    except ValidationError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error adding metabolite: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/drug_routes', methods=['GET'])
def get_drug_routes():
    """Fetch drug routes for a given drug ID."""
    drug_id = request.args.get('drug_id', type=int)
    if not drug_id:
        return jsonify([])
    routes = DrugRoute.query.join(DrugDetail).filter(DrugDetail.drug_id == drug_id).all()
    return jsonify([{
        'route_id': route.route_id,
        'metabolites': route.metabolites or '{}'
    } for route in routes])

@app.route('/pharmacokinetics', methods=['GET'])
def pharmacokinetics():
    """
    Enhanced pharmacokinetics module with proper PK modeling:
    - One-compartment model with first-order absorption for oral/extravascular
    - IV bolus model for intravenous routes
    - Proper bioavailability handling
    - Therapeutic window visualization
    - Multiple dosing regimen simulation
    - Metabolite tracking
    """
    selected_drug_id = request.args.get('drug_id', type=int)
    dose = request.args.get('dose', 100, type=float)
    dosing_interval = request.args.get('interval', 24, type=float)  # hours
    num_doses = request.args.get('num_doses', 1, type=int)
    
    pk_data = []
    selected_drug = None
    chart_data = []
    
    if selected_drug_id:
        selected_drug = Drug.query.get(selected_drug_id)
        if not selected_drug:
            return render_template(
                'pharmacokinetics.html', 
                error="Drug not found", 
                pk_data=None, 
                selected_drug_id=selected_drug_id, 
                chart_data=None
            ), 404
        
        details = DrugDetail.query.filter_by(drug_id=selected_drug_id).options(
            db.joinedload(DrugDetail.routes)
        ).all()
        
        # Color palette for multiple routes
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        route_idx = 0
        
        for detail in details:
            for route in detail.routes:
                # Extract and validate PK parameters
                half_life_min = route.half_life_min or 0
                half_life_max = route.half_life_max or 0
                half_life_avg = (half_life_min + half_life_max) / 2 if half_life_max > 0 else half_life_min
                
                vod_min = route.vod_rate_min or 0
                vod_max = route.vod_rate_max or 0
                vod_avg = (vod_min + vod_max) / 2 if vod_max > 0 else vod_min
                
                bio_min = route.bioavailability_min or 0
                bio_max = route.bioavailability_max or 0
                bio_avg = (bio_min + bio_max) / 2 if bio_max > 0 else bio_min
                
                tmax_min = route.tmax_min or 0
                tmax_max = route.tmax_max or 0
                tmax_avg = (tmax_min + tmax_max) / 2 if tmax_max > 0 else tmax_min
                
                cmax_min = route.cmax_min or 0
                cmax_max = route.cmax_max or 0
                cmax_avg = (cmax_min + cmax_max) / 2 if cmax_max > 0 else cmax_min
                
                # Skip if essential parameters are missing
                if half_life_avg <= 0 or vod_avg <= 0:
                    logging.warning(
                        f"Skipping route {route.route.name} for drug_id {selected_drug_id}: "
                        f"Invalid PK parameters (t½={half_life_avg}h, Vd={vod_avg}L)"
                    )
                    continue
                
                # Calculate elimination rate constant
                ke = math.log(2) / half_life_avg
                
                # Determine route type and calculate accordingly
                route_name = route.route.name.lower()
                is_iv = 'intravenous' in route_name or 'iv' in route_name
                
                # Time array for simulation (0-72h for better visualization)
                time_max = max(72, dosing_interval * num_doses * 1.5)
                time = np.arange(0, time_max, 0.1)
                
                if is_iv:
                    # IV Bolus: C(t) = C0 * e^(-ke*t)
                    # For multiple doses: superposition principle
                    c0 = dose / vod_avg
                    concentrations = np.zeros_like(time)
                    
                    for dose_num in range(num_doses):
                        dose_time = dose_num * dosing_interval
                        time_after_dose = time - dose_time
                        concentrations += np.where(
                            time_after_dose >= 0,
                            c0 * np.exp(-ke * time_after_dose),
                            0
                        )
                    
                    ka = None  # No absorption for IV
                    
                else:
                    # Oral/Extravascular: One-compartment with first-order absorption
                    # C(t) = (F*D*Ka)/(Vd*(Ka-Ke)) * (e^(-Ke*t) - e^(-Ka*t))
                    
                    # Estimate Ka from Tmax if available
                    if tmax_avg > 0 and tmax_avg < half_life_avg:
                        # At Tmax: dC/dt = 0, which gives Ka = Ke * ln(Ka/Ke) / (1 - Ke/Ka)
                        # Simplified approximation: Ka ≈ Ke / (1 - Ke*Tmax/ln(Ka/Ke))
                        # Better approach: Ka*Tmax - ln(Ka/Ke) = ln(Ka/Ke)
                        # Solving: Ka = ln(Ka/Ke) / (Tmax - 1/Ke * ln(Ka/Ke))
                        
                        # Iterative solution for Ka from Tmax
                        def tmax_equation(ka_val):
                            if ka_val <= ke:
                                return float('inf')
                            return math.log(ka_val / ke) / (ka_val - ke) - tmax_avg
                        
                        try:
                            # Initial guess: Ka = 3 * Ke (typical for oral absorption)
                            ka_initial = max(3 * ke, 0.5)
                            ka = optimize.fsolve(tmax_equation, ka_initial)[0]
                            
                            # Validate Ka
                            if ka <= ke or ka <= 0 or ka > 10:
                                ka = 3 * ke  # Fallback to typical value
                        except:
                            ka = 3 * ke  # Fallback
                    else:
                        # Default absorption rate (faster than elimination)
                        ka = max(3 * ke, 0.5)
                    
                    # Calculate concentrations for multiple doses
                    concentrations = np.zeros_like(time)
                    effective_dose = dose * bio_avg
                    
                    for dose_num in range(num_doses):
                        dose_time = dose_num * dosing_interval
                        time_after_dose = time - dose_time
                        
                        # One-compartment oral model
                        term = (effective_dose * ka) / (vod_avg * (ka - ke))
                        concentrations += np.where(
                            time_after_dose >= 0,
                            term * (np.exp(-ke * time_after_dose) - np.exp(-ka * time_after_dose)),
                            0
                        )
                
                # Calculate AUC (0-∞) analytically for single dose
                if num_doses == 1:
                    if is_iv:
                        auc_analytical = c0 / ke
                    else:
                        auc_analytical = (dose * bio_avg) / (vod_avg * ke)
                else:
                    # For multiple doses, use trapezoidal rule
                    auc_analytical = np.trapz(concentrations, time)
                
                # Calculate AUC numerically for comparison
                auc_numerical = np.trapz(concentrations, time)
                
                # Calculate steady-state parameters (for multiple dosing)
                if num_doses > 1:
                    if is_iv:
                        css_max = (c0 / (1 - np.exp(-ke * dosing_interval)))
                        css_min = css_max * np.exp(-ke * dosing_interval)
                    else:
                        css_max = ((dose * bio_avg * ka) / (vod_avg * (ka - ke))) * \
                                  (1 / (1 - np.exp(-ke * dosing_interval)) - 1 / (1 - np.exp(-ka * dosing_interval)))
                        css_min = css_max * np.exp(-ke * dosing_interval)
                    
                    css_avg = css_max * (1 - np.exp(-ke * dosing_interval)) / (ke * dosing_interval)
                else:
                    css_max = css_min = css_avg = None
                
                # Calculate actual Cmax and Tmax from simulation
                sim_cmax = np.max(concentrations)
                sim_tmax = time[np.argmax(concentrations)]
                
                # Prepare chart data
                chart_dataset = {
                    'label': f"{route.route.name} (AUC: {round(auc_numerical, 2)} mg·h/L)",
                    'data': concentrations.tolist(),
                    'borderColor': colors[route_idx % len(colors)],
                    'backgroundColor': colors[route_idx % len(colors)] + '20',  # 20% opacity
                    'fill': False,
                    'tension': 0.4,
                    'pointRadius': 0
                }
                chart_data.append(chart_dataset)
                
                # Add therapeutic window if available
                therapeutic_min = route.therapeutic_min or 0
                therapeutic_max = route.therapeutic_max or 0
                
                # Prepare PK data entry
                pk_entry = {
                    'route_name': route.route.name,
                    'absorption_rate_min': route.absorption_rate_min or 0,
                    'absorption_rate_max': route.absorption_rate_max or 0,
                    'ka': round(ka, 4) if ka else None,
                    'vod_rate_min': vod_min,
                    'vod_rate_max': vod_max,
                    'vod_avg': round(vod_avg, 2),
                    'protein_binding_min': (route.protein_binding_min or 0) * 100,
                    'protein_binding_max': (route.protein_binding_max or 0) * 100,
                    'half_life_min': half_life_min,
                    'half_life_max': half_life_max,
                    'half_life_avg': round(half_life_avg, 2),
                    'ke': round(ke, 4),
                    'clearance_rate_min': route.clearance_rate_min or 0,
                    'clearance_rate_max': route.clearance_rate_max or 0,
                    'bioavailability_min': bio_min * 100,
                    'bioavailability_max': bio_max * 100,
                    'bioavailability_avg': round(bio_avg * 100, 1),
                    'tmax_min': tmax_min,
                    'tmax_max': tmax_max,
                    'tmax_avg': round(tmax_avg, 2) if tmax_avg > 0 else None,
                    'tmax_simulated': round(sim_tmax, 2),
                    'cmax_min': cmax_min,
                    'cmax_max': cmax_max,
                    'cmax_avg': round(cmax_avg, 2) if cmax_avg > 0 else None,
                    'cmax_simulated': round(sim_cmax, 2),
                    'pharmacodynamics': route.pharmacodynamics or "N/A",
                    'pharmacokinetics': route.pharmacokinetics or "N/A",
                    'therapeutic_min': therapeutic_min,
                    'therapeutic_max': therapeutic_max,
                    'therapeutic_unit': route.therapeutic_unit.name if route.therapeutic_unit else "mg/L",
                    'metabolites': [
                        {'id': met.id, 'name': met.name, 'parent_id': met.parent_id}
                        for met in route.metabolites
                    ] if route.metabolites else [],
                    'metabolism_organs': [organ.name for organ in route.metabolism_organs],
                    'metabolism_enzymes': [enzyme.name for enzyme in route.metabolism_enzymes],
                    'auc_analytical': round(auc_analytical, 2),
                    'auc_numerical': round(auc_numerical, 2),
                    'css_max': round(css_max, 2) if css_max else None,
                    'css_min': round(css_min, 2) if css_min else None,
                    'css_avg': round(css_avg, 2) if css_avg else None,
                    'concentrations': list(zip(time.tolist(), concentrations.tolist())),
                    'is_iv': is_iv
                }
                pk_data.append(pk_entry)
                route_idx += 1
        
        # Add therapeutic window bands to chart if any route has them
        therapeutic_datasets = []
        for pk_entry in pk_data:
            if pk_entry['therapeutic_min'] > 0 and pk_entry['therapeutic_max'] > 0:
                # Add therapeutic max line
                therapeutic_datasets.append({
                    'label': f"Therapeutic Maximum ({pk_entry['route_name']})",
                    'data': [pk_entry['therapeutic_max']] * len(time),
                    'borderColor': 'rgba(255, 0, 0, 0.5)',
                    'borderDash': [5, 5],
                    'fill': False,
                    'pointRadius': 0,
                    'borderWidth': 2
                })
                # Add therapeutic min line
                therapeutic_datasets.append({
                    'label': f"Therapeutic Minimum ({pk_entry['route_name']})",
                    'data': [pk_entry['therapeutic_min']] * len(time),
                    'borderColor': 'rgba(0, 255, 0, 0.5)',
                    'borderDash': [5, 5],
                    'fill': False,
                    'pointRadius': 0,
                    'borderWidth': 2
                })
                break  # Only add once
        
        chart_data.extend(therapeutic_datasets)
    
    # Generate Chart.js configuration
    chart = {
        'type': 'line',
        'data': {
            'labels': time.tolist() if 'time' in locals() else [],
            'datasets': chart_data
        },
        'options': {
            'responsive': True,
            'maintainAspectRatio': False,
            'interaction': {
                'mode': 'index',
                'intersect': False
            },
            'plugins': {
                'title': {
                    'display': True,
                    'text': f'Plasma Concentration vs Time - {selected_drug.name_en if selected_drug else "Selected Drug"} ' +
                            f'(Dose: {dose} mg' + (f', q{dosing_interval}h × {num_doses}' if num_doses > 1 else '') + ')',
                    'font': {'size': 16}
                },
                'legend': {
                    'display': True,
                    'position': 'top'
                },
                'tooltip': {
                    'callbacks': {
                        'label': 'function(context) { return context.dataset.label + ": " + context.parsed.y.toFixed(2) + " mg/L"; }'
                    }
                }
            },
            'scales': {
                'x': {
                    'title': {
                        'display': True,
                        'text': 'Time (hours)',
                        'font': {'size': 14}
                    },
                    'ticks': {
                        'maxTicksLimit': 20
                    }
                },
                'y': {
                    'title': {
                        'display': True,
                        'text': 'Plasma Concentration (mg/L)',
                        'font': {'size': 14}
                    },
                    'beginAtZero': True
                }
            }
        }
    } if chart_data else None
    
    return render_template(
        'pharmacokinetics.html', 
        pk_data=pk_data, 
        selected_drug_id=selected_drug_id, 
        selected_drug=selected_drug,
        dose=dose,
        dosing_interval=dosing_interval,
        num_doses=num_doses,
        chart_data=chart
    )
#Pharmacokinetics END...

#SAĞLIK UYGULAMA TEBLİĞİ (SUT) 
# Helper Functions
def extract_item_number(text):
    pattern = r'^(\d+(?:\.\d+)*(?:\.[A-Z]+)?)\s*[-–—]?\s*'
    match = re.match(pattern, text.strip())
    return match.group(1) if match else None

def get_parent_number(item_number):
    if not item_number:
        return None
    parts = item_number.split('.')
    if len(parts) <= 1:
        return None
    return '.'.join(parts[:-1])

def calculate_level(item_number):
    if not item_number:
        return 0
    return item_number.count('.')

def parse_docx_hierarchical(file_path):
    doc = Document(file_path)
    items = []
    current_order = 0
    
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
            
        item_number = extract_item_number(text)
        if item_number:
            title = re.sub(r'^(\d+(?:\.\d+)*(?:\.[A-Z]+)?)\s*[-–—]?\s*', '', text).strip()
            parent = get_parent_number(item_number)
            level = calculate_level(item_number)
            
            items.append({
                'item_number': item_number,
                'title': title,
                'content': text,
                'parent_number': parent,
                'level': level,
                'order_index': current_order,
                'full_text': text
            })
            current_order += 1
    
    return items

def detect_changes(old_items, new_items):
    changes = []
    old_dict = {item['item_number']: item for item in old_items}
    new_dict = {item['item_number']: item for item in new_items}
    
    for item_num in new_dict:
        if item_num not in old_dict:
            changes.append({
                'item_number': item_num,
                'change_type': 'added',
                'old_content': None,
                'new_content': new_dict[item_num]['full_text']
            })
        elif old_dict[item_num]['full_text'] != new_dict[item_num]['full_text']:
            changes.append({
                'item_number': item_num,
                'change_type': 'modified',
                'old_content': old_dict[item_num]['full_text'],
                'new_content': new_dict[item_num]['full_text']
            })
    
    for item_num in old_dict:
        if item_num not in new_dict:
            changes.append({
                'item_number': item_num,
                'change_type': 'deleted',
                'old_content': old_dict[item_num]['full_text'],
                'new_content': None
            })
    
    return changes

# Routes
@app.route('/sut_upload')
def sut_upload_page():
    return render_template('sut_upload.html')

@app.route('/sut_full')
def sut_full_page():
    return render_template('sut_full.html')

@app.route('/api/sut/upload', methods=['POST'])
def upload_sut():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.endswith('.docx'):
        return jsonify({'error': 'Only .docx files allowed'}), 400
    
    filepath = None
    try:
        filename = secure_filename(file.filename)
        
        # Fix for Windows path - use forward slashes
        import tempfile
        temp_dir = tempfile.gettempdir()
        filepath = os.path.join(temp_dir, filename).replace('\\', '/')
        
        file.save(filepath)
        
        items = parse_docx_hierarchical(filepath)
        
        with open(filepath, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        
        existing = SUT_Version.query.filter_by(file_hash=file_hash).first()
        if existing:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': 'This file version already exists', 'version_id': existing.id}), 409
        
        latest_version = SUT_Version.query.order_by(SUT_Version.version_number.desc()).first()
        new_version_number = (latest_version.version_number + 1) if latest_version else 1
        
        changes = []
        if latest_version:
            old_items_db = SUT_Item.query.filter_by(version_id=latest_version.id).all()
            old_items = [{
                'item_number': item.item_number,
                'title': item.title,
                'content': item.content,
                'full_text': item.full_text
            } for item in old_items_db]
            changes = detect_changes(old_items, items)
            SUT_Version.query.update({SUT_Version.is_active: False})
        
        new_version = SUT_Version(
            version_number=new_version_number,
            filename=filename,
            file_hash=file_hash,
            is_active=True
        )
        db.session.add(new_version)
        db.session.flush()
        
        for item_data in items:
            sut_item = SUT_Item(
                version_id=new_version.id,
                item_number=item_data['item_number'],
                title=item_data['title'],
                content=item_data['content'],
                parent_number=item_data['parent_number'],
                level=item_data['level'],
                order_index=item_data['order_index'],
                full_text=item_data['full_text']
            )
            db.session.add(sut_item)
        
        for change_data in changes:
            change = SUT_Change(
                version_id=new_version.id,
                item_number=change_data['item_number'],
                change_type=change_data['change_type'],
                old_content=change_data['old_content'],
                new_content=change_data['new_content']
            )
            db.session.add(change)
        
        db.session.commit()
        
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
        
        return jsonify({
            'success': True,
            'version_id': new_version.id,
            'version_number': new_version_number,
            'total_items': len(items),
            'changes_detected': len(changes)
        }), 200
        
    except Exception as e:
        db.session.rollback()
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass
        return jsonify({'error': str(e)}), 500


@app.route('/api/sut/items', methods=['GET'])
def get_sut_items():
    version_id = request.args.get('version_id', type=int)
    parent_number = request.args.get('parent_number', '')
    
    try:
        if not version_id:
            active_version = SUT_Version.query.filter_by(is_active=True).first()
            if not active_version:
                return jsonify({'error': 'No active version found'}), 404
            version_id = active_version.id
        
        # Handle root level items
        if parent_number == '' or parent_number == 'root':
            items = SUT_Item.query.filter(
                SUT_Item.version_id == version_id,
                db.or_(SUT_Item.parent_number == None, SUT_Item.parent_number == '')
            ).order_by(SUT_Item.order_index).all()
        else:
            items = SUT_Item.query.filter_by(
                version_id=version_id,
                parent_number=parent_number
            ).order_by(SUT_Item.order_index).all()
        
        # Check for children efficiently
        result = []
        for item in items:
            has_children = SUT_Item.query.filter_by(
                version_id=version_id,
                parent_number=item.item_number
            ).limit(1).count() > 0
            
            result.append({
                'id': item.id,
                'item_number': item.item_number,
                'title': item.title,
                'content': item.content,
                'level': item.level,
                'has_children': has_children
            })
        
        return jsonify(result), 200
        
    except Exception as e:
        print(f"Error in get_sut_items: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/sut/search', methods=['GET'])
def search_sut():
    query = request.args.get('q', '').strip()
    version_id = request.args.get('version_id', type=int)
    
    if not query:
        return jsonify([]), 200
    
    if not version_id:
        active_version = SUT_Version.query.filter_by(is_active=True).first()
        if not active_version:
            return jsonify({'error': 'No active version found'}), 404
        version_id = active_version.id
    
    items = SUT_Item.query.filter(
        SUT_Item.version_id == version_id,
        db.or_(
            SUT_Item.item_number.contains(query),
            SUT_Item.title.contains(query),
            SUT_Item.full_text.contains(query)
        )
    ).order_by(SUT_Item.order_index).limit(100).all()
    
    result = [{
        'id': item.id,
        'item_number': item.item_number,
        'title': item.title,
        'content': item.content,
        'full_text': item.full_text,
        'level': item.level
    } for item in items]
    
    return jsonify(result), 200

@app.route('/api/sut/versions', methods=['GET'])
def get_sut_versions():
    versions = SUT_Version.query.order_by(SUT_Version.version_number.desc()).all()
    result = [{
        'id': v.id,
        'version_number': v.version_number,
        'upload_date': v.upload_date.isoformat(),
        'filename': v.filename,
        'is_active': v.is_active,
        'item_count': SUT_Item.query.filter_by(version_id=v.id).count()
    } for v in versions]
    return jsonify(result), 200

@app.route('/api/sut/changes/<int:version_id>', methods=['GET'])
def get_sut_changes(version_id):
    changes = SUT_Change.query.filter_by(version_id=version_id).all()
    result = [{
        'id': c.id,
        'item_number': c.item_number,
        'change_type': c.change_type,
        'old_content': c.old_content,
        'new_content': c.new_content,
        'change_date': c.change_date.isoformat()
    } for c in changes]
    return jsonify(result), 200
#SUT, THE END!!!!


# AI CHATBOT
@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        conversation_history = data.get('history', [])
        
        if not user_message:
            return jsonify({'error': 'Message is required'}), 400
        
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return jsonify({'error': 'API key missing'}), 500
        
        client = anthropic.Anthropic(api_key=api_key)
        
        # Step 1: Search database
        db_result = search_database_for_context(user_message)
        
        # Step 2: Determine if database has good information
        db_has_info = db_result["has_info"]
        db_context = db_result["context"]
        drugs_found = db_result["drugs_found"]
        
        # Step 3: Choose strategy based on database results
        if db_has_info:
            # Database has good information - use it
            system_prompt = """You are an expert pharmaceutical assistant for drugly.ai.

You have been provided with VERIFIED INTERNAL DATABASE INFORMATION in the user's message.

CRITICAL RULES:
1. The [DATABASE INFORMATION] section contains the MOST ACCURATE and AUTHORITATIVE information
2. ALWAYS prioritize information from the database over any other knowledge
3. If the database provides specific details, USE THEM DIRECTLY
4. Only supplement with general knowledge for minor context - NEVER contradict the database
5. If asked about something not in the database, clearly state "This information is not available in our database"
6. Be precise, professional, and reference the database information
7. Always end with: "Please consult a healthcare professional before making any medical decisions."

Structure your response clearly with appropriate sections."""
            
            user_message_with_context = f"""{user_message}

[DATABASE INFORMATION]:
{db_context}

Remember: The above database information is your PRIMARY and MOST TRUSTED source. Use it first and foremost."""
            
            source = "database"
            
        else:
            # No good database info - use Claude's knowledge with web search capability
            system_prompt = """You are a world-class clinical pharmacologist and pharmaceutical expert.

The user's question was NOT found in the internal database, so you should use your comprehensive medical and pharmaceutical knowledge.

IMPORTANT INSTRUCTIONS:
1. Provide accurate, detailed, and up-to-date information
2. If the query is about recent drug approvals, new research, or current medical guidelines, you SHOULD use web search
3. For stable, established medical knowledge, use your training knowledge
4. Cite sources when possible (FDA, EMA, PubMed, clinical guidelines, etc.)
5. Be clear about the level of evidence for your statements
6. If you're uncertain about recent developments, explicitly state that and recommend checking with healthcare professionals
7. Always end with: "Please consult a healthcare professional before making any medical decisions."

Structure your response clearly with appropriate sections."""
            
            user_message_with_context = user_message
            source = "claude_knowledge"
        
        # Step 4: Build conversation messages
        messages = []
        for msg in conversation_history[-10:]:
            if msg.get('role') and msg.get('content'):
                messages.append({
                    "role": msg['role'],
                    "content": msg['content']
                })
        
        messages.append({
            "role": "user",
            "content": user_message_with_context
        })
        
        # Step 5: Call Claude API - CORRECT MODEL NAME
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            temperature=0.3,
            system=system_prompt,
            messages=messages
        )
        
        answer = response.content[0].text
        
        return jsonify({
            "response": answer,
            "source": source,
            "drugs_found": drugs_found,
            "db_context_length": len(db_context),
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Internal server error", "details": str(e)}), 500


def extract_drug_names_from_query(query):
    """Extract potential drug names from natural language query"""
    import re
    import string
    
    query_lower = query.lower()
    
    # Remove common question patterns
    patterns_to_remove = [
        r'\bwhat\s+(?:are|is)\s+(?:the\s+)?',
        r'\bhow\s+(?:does|do)\s+',
        r'\bcan\s+(?:you|i)\s+',
        r'\bshould\s+i\s+',
        r'\btell\s+me\s+(?:about\s+)?',
        r'\bshow\s+me\s+',
        r'\bgive\s+me\s+',
        r'\bplease\s+',
        r'\bside\s+effects?\b',
        r'\binteractions?\b',
        r'\binformation\b',
        r'\bdetails?\b',
        r'\bof\b',
        r'\bfor\b',
        r'\babout\b',
        r'\bwith\b',
    ]
    
    cleaned = query_lower
    for pattern in patterns_to_remove:
        cleaned = re.sub(pattern, ' ', cleaned)
    
    cleaned = cleaned.translate(str.maketrans('', '', string.punctuation))
    cleaned = ' '.join(cleaned.split()).strip()
    
    # Extract capitalized words (likely drug names)
    words = query.split()
    capitalized_words = []
    for w in words:
        clean_word = w.strip(string.punctuation)
        if (clean_word and 
            len(clean_word) > 2 and 
            clean_word[0].isupper() and 
            clean_word.lower() not in ['what', 'when', 'where', 'which', 'who', 'how', 'can', 'should', 'tell', 'show', 'give', 'please']):
            capitalized_words.append(clean_word)
    
    # Combine both approaches
    search_terms = []
    if capitalized_words:
        search_terms.extend(capitalized_words)
    if cleaned and len(cleaned) > 2:
        search_terms.append(cleaned)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_terms = []
    for term in search_terms:
        term_lower = term.lower()
        if term_lower not in seen and len(term_lower) > 2:
            seen.add(term_lower)
            unique_terms.append(term)
    
    return unique_terms


def search_database_for_context(query):
    """
    Search database and return structured result
    Returns: {
        "has_info": bool,
        "context": str,
        "drugs_found": list
    }
    """
    try:
        # Extract potential drug names
        search_terms = extract_drug_names_from_query(query)
        
        if not search_terms:
            return {
                "has_info": False,
                "context": "",
                "drugs_found": []
            }
        
        drugs_found = []
        
        # Search for drugs using extracted terms
        for term in search_terms[:3]:  # Limit to 3 terms
            term_lower = term.lower()
            
            if len(term_lower) < 3:
                continue
            
            # Exact match
            exact_matches = Drug.query.filter(
                db.or_(
                    db.func.lower(Drug.name_en) == term_lower,
                    db.func.lower(Drug.name_tr) == term_lower
                )
            ).limit(2).all()
            
            for drug in exact_matches:
                if drug.id not in [d.id for d in drugs_found]:
                    drugs_found.append(drug)
            
            # Starts with (high confidence)
            if len(drugs_found) < 3:
                starts_matches = Drug.query.filter(
                    db.or_(
                        db.func.lower(Drug.name_en).like(f'{term_lower}%'),
                        db.func.lower(Drug.name_tr).like(f'{term_lower}%')
                    )
                ).limit(3 - len(drugs_found)).all()
                
                for drug in starts_matches:
                    if drug.id not in [d.id for d in drugs_found]:
                        drugs_found.append(drug)
            
            # Contains (lower confidence)
            if len(drugs_found) < 2:
                contains_matches = Drug.query.filter(
                    db.or_(
                        db.func.lower(Drug.name_en).like(f'%{term_lower}%'),
                        db.func.lower(Drug.name_tr).like(f'%{term_lower}%')
                    )
                ).limit(2 - len(drugs_found)).all()
                
                for drug in contains_matches:
                    if drug.id not in [d.id for d in drugs_found]:
                        drugs_found.append(drug)
            
            if len(drugs_found) >= 2:
                break
        
        # If no drugs found, return no info
        if not drugs_found:
            return {
                "has_info": False,
                "context": "",
                "drugs_found": []
            }
        
        # Build comprehensive context for found drugs
        context_parts = []
        
        for drug in drugs_found[:2]:  # Limit to 2 drugs to avoid token explosion
            drug_info = build_drug_context(drug)
            context_parts.append(drug_info)
        
        full_context = "\n\n".join(context_parts)
        
        # Check if context has substantial information
        has_substantial_info = (
            len(full_context) > 500 and
            any([
                "Mechanism of Action:" in full_context,
                "Pharmacodynamics:" in full_context,
                "Pharmacokinetics:" in full_context,
                "Side Effects" in full_context,
                "Drug-Drug Interactions" in full_context
            ])
        )
        
        return {
            "has_info": has_substantial_info,
            "context": full_context[:15000],  # Token limit safety
            "drugs_found": [{"id": d.id, "name": d.name_en} for d in drugs_found]
        }
        
    except Exception as e:
        print(f"❌ Database search error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "has_info": False,
            "context": "",
            "drugs_found": []
        }


def build_drug_context(drug):
    """Build comprehensive context for a single drug"""
    
    sections = []
    
    # Header
    header = f"{'='*70}\n"
    header += f"DRUG: {drug.name_en}"
    if drug.name_tr and drug.name_tr != drug.name_en:
        header += f" (Turkish: {drug.name_tr})"
    header += f"\nDrug ID: {drug.id}\n"
    header += f"FDA Approved: {'Yes' if drug.fda_approved else 'No'}\n"
    header += f"{'='*70}"
    sections.append(header)
    
    # Alternative names
    if drug.alternative_names:
        sections.append(f"Alternative Names: {drug.alternative_names}")
    
    # Categories
    if drug.categories:
        cats = ", ".join([c.name for c in drug.categories])
        sections.append(f"Drug Categories: {cats}")
    
    # Indications
    if drug.indications:
        indications = drug.indications.strip()[:1500]
        sections.append(f"\nIndications:\n{indications}")
    
    # Drug Details
    if drug.drug_details:
        detail = drug.drug_details[0]
        
        # Mechanism of Action
        if detail.mechanism_of_action:
            moa = detail.mechanism_of_action.strip()[:1500]
            sections.append(f"\nMechanism of Action:\n{moa}")
        
        # Pharmacodynamics
        if detail.pharmacodynamics:
            pd = detail.pharmacodynamics.strip()[:1500]
            sections.append(f"\nPharmacodynamics:\n{pd}")
        
        # Pharmacokinetics
        if detail.pharmacokinetics:
            pk = detail.pharmacokinetics.strip()[:1500]
            sections.append(f"\nPharmacokinetics:\n{pk}")
        
        # Black Box Warning
        if detail.black_box_warning:
            warning = "\n⚠️ BLACK BOX WARNING: YES"
            if detail.black_box_details:
                warning += f"\n{detail.black_box_details.strip()[:1000]}"
            sections.append(warning)
        
        # Chemical Properties
        chem_props = []
        if detail.molecular_formula:
            chem_props.append(f"Molecular Formula: {detail.molecular_formula}")
        if detail.molecular_weight:
            unit = detail.molecular_weight_unit.name if detail.molecular_weight_unit else ""
            chem_props.append(f"Molecular Weight: {detail.molecular_weight} {unit}")
        if detail.smiles:
            chem_props.append(f"SMILES: {detail.smiles[:200]}")
        if chem_props:
            sections.append("\nChemical Properties:\n" + "\n".join(chem_props))
        
        # Safety Information
        safety_info = []
        if detail.pregnancy_safety_trimester1:
            safety_info.append(f"Pregnancy (Trimester 1): {detail.pregnancy_safety_trimester1.name}")
            if detail.pregnancy_details_trimester1:
                safety_info.append(f"  Details: {detail.pregnancy_details_trimester1[:300]}")
        
        if detail.pregnancy_safety_trimester2:
            safety_info.append(f"Pregnancy (Trimester 2): {detail.pregnancy_safety_trimester2.name}")
        
        if detail.pregnancy_safety_trimester3:
            safety_info.append(f"Pregnancy (Trimester 3): {detail.pregnancy_safety_trimester3.name}")
        
        if detail.lactation_safety:
            safety_info.append(f"Lactation Safety: {detail.lactation_safety.name}")
            if detail.lactation_details:
                safety_info.append(f"  Details: {detail.lactation_details[:300]}")
        
        if safety_info:
            sections.append("\nPregnancy & Lactation Safety:\n" + "\n".join(safety_info))
        
        # Side Effects - FIXED
        try:
            side_effects_list = list(detail.side_effects)[:30]  # Convert to list and slice
            if side_effects_list:
                se_list = []
                for se in side_effects_list:
                    name = se.name_en
                    if se.name_tr and se.name_tr != se.name_en:
                        name += f" ({se.name_tr})"
                    se_list.append(f"  • {name}")
                sections.append(f"\nCommon Side Effects ({len(side_effects_list)} listed):\n" + "\n".join(se_list))
        except Exception as e:
            print(f"Side effects error: {e}")
    
    # Drug Interactions
    interactions = DrugInteraction.query.filter(
        db.or_(
            DrugInteraction.drug1_id == drug.id,
            DrugInteraction.drug2_id == drug.id
        )
    ).limit(15).all()
    
    if interactions:
        int_list = []
        for inter in interactions:
            other_drug = inter.drug2 if inter.drug1_id == drug.id else inter.drug1
            severity = inter.severity_level.name if inter.severity_level else "Unknown"
            desc = inter.interaction_description[:250]
            int_list.append(f"  • {other_drug.name_en}:")
            int_list.append(f"    Type: {inter.interaction_type}")
            int_list.append(f"    Severity: {severity}")
            int_list.append(f"    Description: {desc}")
            if inter.mechanism:
                int_list.append(f"    Mechanism: {inter.mechanism[:200]}")
        sections.append(f"\nDrug-Drug Interactions ({len(interactions)} shown):\n" + "\n".join(int_list))
    
    # Food Interactions
    food_interactions = DrugFoodInteraction.query.filter_by(drug_id=drug.id).limit(10).all()
    if food_interactions:
        food_list = []
        for fi in food_interactions:
            food_list.append(f"  • {fi.food.name_en}:")
            food_list.append(f"    Type: {fi.interaction_type}")
            food_list.append(f"    Severity: {fi.severity.name if fi.severity else 'Unknown'}")
            food_list.append(f"    Description: {fi.description[:200]}")
            if fi.recommendation:
                food_list.append(f"    Recommendation: {fi.recommendation[:200]}")
        sections.append(f"\nFood Interactions ({len(food_interactions)} shown):\n" + "\n".join(food_list))
    
    # Disease Interactions
    disease_interactions = DrugDiseaseInteraction.query.filter_by(drug_id=drug.id).limit(10).all()
    if disease_interactions:
        disease_list = []
        for di in disease_interactions:
            disease_list.append(f"  • {di.indication.name_en}:")
            disease_list.append(f"    Type: {di.interaction_type}")
            disease_list.append(f"    Severity: {di.severity}")
            disease_list.append(f"    Description: {di.description[:200]}")
        sections.append(f"\nDisease Interactions ({len(disease_interactions)} shown):\n" + "\n".join(disease_list))
    
    return "\n\n".join(sections)


@app.route('/api/chatbot/drug-info', methods=['POST'])
def chatbot_drug_info():
    try:
        data = request.json
        drug_name = data.get('drug_name', '')
        query_type = data.get('query_type', 'general')
        
        if not drug_name:
            return jsonify({'error': 'Drug name is required'}), 400
        
        drug = Drug.query.filter(
            (Drug.name_en.ilike(f'%{drug_name}%')) | 
            (Drug.name_tr.ilike(f'%{drug_name}%'))
        ).first()
        
        if not drug:
            return jsonify({'error': 'Drug not found'}), 404
        
        drug_info = {
            'name_en': drug.name_en,
            'name_tr': drug.name_tr,
            'fda_approved': drug.fda_approved,
            'indications': drug.indications,
            'categories': [cat.name for cat in drug.categories],
        }
        
        if drug.drug_details:
            detail = drug.drug_details[0]
            drug_info.update({
                'mechanism_of_action': detail.mechanism_of_action,
                'pharmacodynamics': detail.pharmacodynamics,
                'pharmacokinetics': detail.pharmacokinetics,
                'molecular_formula': detail.molecular_formula,
                'black_box_warning': detail.black_box_warning,
                'black_box_details': detail.black_box_details,
            })
        
        if query_type in ['interactions', 'all']:
            interactions = DrugInteraction.query.filter(
                (DrugInteraction.drug1_id == drug.id) | 
                (DrugInteraction.drug2_id == drug.id)
            ).limit(10).all()
            
            drug_info['interactions'] = [{
                'drug1': i.drug1.name_en,
                'drug2': i.drug2.name_en,
                'type': i.interaction_type,
                'description': i.interaction_description,
                'severity': i.severity_level.name if i.severity_level else None
            } for i in interactions]
        
        if query_type in ['side_effects', 'all'] and drug.drug_details:
            side_effects = list(drug.drug_details[0].side_effects)[:10]
            drug_info['side_effects'] = [se.name_en for se in side_effects]
        
        return jsonify({
            'drug_info': drug_info,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/chatbot/drug-search', methods=['GET'])
def chatbot_drug_search():
    try:
        query = request.args.get('q', '')
        limit = int(request.args.get('limit', 10))
        
        if not query or len(query) < 2:
            return jsonify({'drugs': []})
        
        drugs = Drug.query.filter(
            (Drug.name_en.ilike(f'%{query}%')) | 
            (Drug.name_tr.ilike(f'%{query}%'))
        ).limit(limit).all()
        
        results = [{
            'id': drug.id,
            'name_en': drug.name_en,
            'name_tr': drug.name_tr,
            'fda_approved': drug.fda_approved
        } for drug in drugs]
        
        return jsonify({'drugs': results})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/chatbot/interaction-check', methods=['POST'])
def chatbot_interaction_check():
    try:
        data = request.json
        drug_ids = data.get('drug_ids', [])
        
        if len(drug_ids) < 2:
            return jsonify({'error': 'At least 2 drugs required'}), 400
        
        interactions = []
        
        for i in range(len(drug_ids)):
            for j in range(i + 1, len(drug_ids)):
                drug1_id = drug_ids[i]
                drug2_id = drug_ids[j]
                
                interaction = DrugInteraction.query.filter(
                    ((DrugInteraction.drug1_id == drug1_id) & (DrugInteraction.drug2_id == drug2_id)) |
                    ((DrugInteraction.drug1_id == drug2_id) & (DrugInteraction.drug2_id == drug1_id))
                ).first()
                
                if interaction:
                    interactions.append({
                        'drug1': interaction.drug1.name_en,
                        'drug2': interaction.drug2.name_en,
                        'type': interaction.interaction_type,
                        'description': interaction.interaction_description,
                        'severity': interaction.severity_level.name if interaction.severity_level else None,
                        'mechanism': interaction.mechanism,
                        'monitoring': interaction.monitoring
                    })
        
        return jsonify({
            'interactions': interactions,
            'count': len(interactions),
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/chatbot/food-interactions', methods=['POST'])
def chatbot_food_interactions():
    try:
        data = request.json
        drug_id = data.get('drug_id')
        
        if not drug_id:
            return jsonify({'error': 'Drug ID is required'}), 400
        
        interactions = DrugFoodInteraction.query.filter_by(drug_id=drug_id).all()
        
        results = [{
            'food': i.food.name_en,
            'type': i.interaction_type,
            'description': i.description,
            'severity': i.severity.name if i.severity else None,
            'recommendation': i.recommendation,
            'timing': i.timing_instruction
        } for i in interactions]
        
        return jsonify({
            'interactions': results,
            'count': len(results),
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/chatbot/disease-interactions', methods=['POST'])
def chatbot_disease_interactions():
    try:
        data = request.json
        drug_id = data.get('drug_id')
        
        if not drug_id:
            return jsonify({'error': 'Drug ID is required'}), 400
        
        interactions = DrugDiseaseInteraction.query.filter_by(drug_id=drug_id).all()
        
        results = [{
            'disease': i.indication.name_en,
            'type': i.interaction_type,
            'description': i.description,
            'severity': i.severity,
            'recommendation': i.recommendation
        } for i in interactions]
        
        return jsonify({
            'interactions': results,
            'count': len(results),
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/chatbot/safety-info', methods=['POST'])
def chatbot_safety_info():
    try:
        data = request.json
        drug_id = data.get('drug_id')
        
        if not drug_id:
            return jsonify({'error': 'Drug ID is required'}), 400
        
        drug = Drug.query.get(drug_id)
        if not drug or not drug.drug_details:
            return jsonify({'error': 'Drug not found'}), 404
        
        detail = drug.drug_details[0]
        
        safety_info = {
            'black_box_warning': detail.black_box_warning,
            'black_box_details': detail.black_box_details,
            'pregnancy': {
                'trimester1': {
                    'category': detail.pregnancy_safety_trimester1.name if detail.pregnancy_safety_trimester1 else None,
                    'details': detail.pregnancy_details_trimester1
                },
                'trimester2': {
                    'category': detail.pregnancy_safety_trimester2.name if detail.pregnancy_safety_trimester2 else None,
                    'details': detail.pregnancy_details_trimester2
                },
                'trimester3': {
                    'category': detail.pregnancy_safety_trimester3.name if detail.pregnancy_safety_trimester3 else None,
                    'details': detail.pregnancy_details_trimester3
                }
            },
            'lactation': {
                'category': detail.lactation_safety.name if detail.lactation_safety else None,
                'details': detail.lactation_details
            }
        }
        
        return jsonify({
            'safety_info': safety_info,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/chatbot/pharmacogenomics', methods=['POST'])
def chatbot_pharmacogenomics():
    try:
        data = request.json
        drug_id = data.get('drug_id')
        
        if not drug_id:
            return jsonify({'error': 'Drug ID is required'}), 400
        
        drug = Drug.query.get(drug_id)
        if not drug:
            return jsonify({'error': 'Drug not found'}), 404
        
        clinical_annotations = []
        for ca in drug.clinical_annotations:
            annotation = ca.annotation
            clinical_annotations.append({
                'id': annotation.clinical_annotation_id,
                'level_of_evidence': annotation.level_of_evidence,
                'phenotype_category': annotation.phenotype_category,
                'genes': [g.gene.gene_symbol for g in annotation.genes],
                'variants': [v.variant.name for v in annotation.variants],
                'url': annotation.url
            })
        
        variant_annotations = []
        for va in drug.variant_annotations:
            annotation = va.variant_annotation
            variant_annotations.append({
                'id': annotation.variant_annotation_id,
                'genes': [g.gene.gene_symbol for g in annotation.genes],
                'variants': [v.variant.name for v in annotation.variants]
            })
        
        drug_labels = []
        for dl in drug.drug_labels:
            label = dl.drug_label
            drug_labels.append({
                'id': label.pharmgkb_id,
                'name': label.name,
                'source': label.source,
                'biomarker_flag': label.biomarker_flag,
                'testing_level': label.testing_level,
                'has_prescribing_info': label.has_prescribing_info
            })
        
        return jsonify({
            'clinical_annotations': clinical_annotations,
            'variant_annotations': variant_annotations,
            'drug_labels': drug_labels,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/chatbot-page')
def chatbot_page():
    if 'user_id' not in session:
        flash('Please login to access the AI Chatbot', 'warning')
        return redirect(url_for('login'))
    
    user = User.query.get(session['user_id'])
    return render_template('chatbot.html', user=user)


#Track users!
def get_visitor_hash(ip, user_agent):
    return hashlib.md5(f"{ip}{user_agent}".encode()).hexdigest()
    
@app.route('/api/track', methods=['POST'])
def track_visitor():
    try:
        data = request.json
        ip = request.headers.get('X-Forwarded-For', request.remote_addr)
        if ip:
            ip = ip.split(',')[0].strip()
        user_agent = request.headers.get('User-Agent', '')
        
        visitor_hash = get_visitor_hash(ip, user_agent)
        
        ua_data = parse_user_agent(user_agent)
        
        visitor = Visitor(
            visitor_hash=visitor_hash,
            ip_address=ip,
            user_agent=user_agent,
            country=data.get('country'),
            city=data.get('city'),
            region=data.get('region'),
            timezone=data.get('timezone'),
            isp=data.get('isp'),
            page_url=data.get('page_url'),
            referrer=data.get('referrer'),
            language=data.get('language'),
            screen_width=data.get('screen_width'),
            screen_height=data.get('screen_height'),
            viewport_width=data.get('viewport_width'),
            viewport_height=data.get('viewport_height'),
            color_depth=data.get('color_depth'),
            pixel_ratio=data.get('pixel_ratio'),
            connection_type=data.get('connection_type'),
            device_memory=data.get('device_memory'),
            hardware_concurrency=data.get('hardware_concurrency'),
            latitude=data.get('latitude'),
            longitude=data.get('longitude'),
            accuracy=data.get('accuracy')
        )
        db.session.add(visitor)
        
        page_view = PageView(
            visitor_hash=visitor_hash,
            page_url=data.get('page_url'),
            page_title=data.get('page_title'),
            time_on_page=data.get('time_on_page'),
            scroll_depth=data.get('scroll_depth')
        )
        db.session.add(page_view)
        
        active_session = VisitorSession.query.filter_by(
            visitor_hash=visitor_hash,
            session_end=None
        ).order_by(VisitorSession.session_start.desc()).first()
        
        if not active_session or (datetime.utcnow() - active_session.session_start) > timedelta(minutes=30):
            session_data = VisitorSession(
                visitor_hash=visitor_hash,
                device_type=ua_data['device_type'],
                browser=ua_data['browser'],
                os=ua_data['os'],
                screen_resolution=data.get('screen_resolution'),
                device_brand=ua_data['device_brand'],
                device_model=ua_data['device_model'],
                is_bot=ua_data['is_bot'],
                is_touch_capable=ua_data['is_touch_capable'],
                entry_page=data.get('page_url'),
                pages_visited=1
            )
            db.session.add(session_data)
        else:
            active_session.pages_visited += 1
            active_session.session_end = datetime.utcnow()
            active_session.duration_seconds = int((active_session.session_end - active_session.session_start).total_seconds())
            active_session.exit_page = data.get('page_url')
            # FIX: Handle None values for scroll_depth
            current_scroll = data.get('scroll_depth', 0) or 0
            max_scroll = active_session.max_scroll_depth or 0
            if current_scroll > max_scroll:
                active_session.max_scroll_depth = current_scroll
        
        db.session.commit()
        
        return jsonify({"status": "success"}), 200
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error tracking visitor: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"status": "error", "message": str(e)}), 500
    
@app.route('/api/analytics/devices', methods=['GET'])
@admin_required
def get_device_analytics():
    """Get device, browser, and OS statistics"""
    try:
        period = request.args.get('period', 'week')
        
        now = datetime.utcnow()
        if period == 'today':
            start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == 'week':
            start_date = now - timedelta(days=7)
        elif period == 'month':
            start_date = now - timedelta(days=30)
        else:
            start_date = now - timedelta(days=365)
        
        # Device type breakdown
        device_types = db.session.query(
            VisitorSession.device_type,
            db.func.count(VisitorSession.id).label('count')
        ).filter(VisitorSession.session_start >= start_date)\
        .group_by(VisitorSession.device_type)\
        .all()
        
        # Browser breakdown
        browsers = db.session.query(
            VisitorSession.browser,
            db.func.count(VisitorSession.id).label('count')
        ).filter(VisitorSession.session_start >= start_date)\
        .group_by(VisitorSession.browser)\
        .order_by(db.text('count DESC'))\
        .limit(20).all()
        
        # OS breakdown
        operating_systems = db.session.query(
            VisitorSession.os,
            db.func.count(VisitorSession.id).label('count')
        ).filter(VisitorSession.session_start >= start_date)\
        .group_by(VisitorSession.os)\
        .order_by(db.text('count DESC'))\
        .limit(20).all()
        
        # Screen resolution breakdown
        resolutions = db.session.query(
            VisitorSession.screen_resolution,
            db.func.count(VisitorSession.id).label('count')
        ).filter(
            VisitorSession.session_start >= start_date,
            VisitorSession.screen_resolution.isnot(None)
        ).group_by(VisitorSession.screen_resolution)\
        .order_by(db.text('count DESC'))\
        .limit(20).all()
        
        return jsonify({
            'device_types': [{'type': d[0] or 'Unknown', 'count': d[1]} for d in device_types],
            'browsers': [{'browser': b[0] or 'Unknown', 'count': b[1]} for b in browsers],
            'operating_systems': [{'os': o[0] or 'Unknown', 'count': o[1]} for o in operating_systems],
            'screen_resolutions': [{'resolution': r[0] or 'Unknown', 'count': r[1]} for r in resolutions]
        }), 200
        
    except Exception as e:
        logger.error(f"Error in device analytics: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/analytics/sessions', methods=['GET'])
@admin_required
def get_session_analytics():
    """Get detailed session analytics"""
    try:
        period = request.args.get('period', 'week')
        
        now = datetime.utcnow()
        if period == 'today':
            start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == 'week':
            start_date = now - timedelta(days=7)
        elif period == 'month':
            start_date = now - timedelta(days=30)
        else:
            start_date = now - timedelta(days=365)
        
        # Total sessions
        total_sessions = db.session.query(db.func.count(VisitorSession.id))\
            .filter(VisitorSession.session_start >= start_date).scalar() or 0
        
        # Average session duration
        avg_duration = db.session.query(db.func.avg(VisitorSession.duration_seconds))\
            .filter(
                VisitorSession.session_start >= start_date,
                VisitorSession.duration_seconds.isnot(None)
            ).scalar()
        
        # Average pages per session
        avg_pages = db.session.query(db.func.avg(VisitorSession.pages_visited))\
            .filter(VisitorSession.session_start >= start_date).scalar()
        
        # Bounce rate (sessions with only 1 page)
        bounced_sessions = db.session.query(db.func.count(VisitorSession.id))\
            .filter(
                VisitorSession.session_start >= start_date,
                VisitorSession.pages_visited == 1
            ).scalar() or 0
        
        bounce_rate = (bounced_sessions / total_sessions * 100) if total_sessions > 0 else 0
        
        # Session duration distribution
        duration_distribution = db.session.query(
            db.case(
                (VisitorSession.duration_seconds < 30, '0-30s'),
                (VisitorSession.duration_seconds < 60, '30-60s'),
                (VisitorSession.duration_seconds < 180, '1-3min'),
                (VisitorSession.duration_seconds < 300, '3-5min'),
                (VisitorSession.duration_seconds < 600, '5-10min'),
                else_='10min+'
            ).label('duration_range'),
            db.func.count(VisitorSession.id).label('count')
        ).filter(
            VisitorSession.session_start >= start_date,
            VisitorSession.duration_seconds.isnot(None)
        ).group_by(db.text('duration_range')).all()
        
        # Pages per session distribution
        pages_distribution = db.session.query(
            db.case(
                (VisitorSession.pages_visited == 1, '1 page'),
                (VisitorSession.pages_visited <= 3, '2-3 pages'),
                (VisitorSession.pages_visited <= 5, '4-5 pages'),
                (VisitorSession.pages_visited <= 10, '6-10 pages'),
                else_='10+ pages'
            ).label('page_range'),
            db.func.count(VisitorSession.id).label('count')
        ).filter(VisitorSession.session_start >= start_date)\
        .group_by(db.text('page_range')).all()
        
        return jsonify({
            'total_sessions': total_sessions,
            'avg_duration_seconds': round(float(avg_duration), 1) if avg_duration else 0,
            'avg_pages_per_session': round(float(avg_pages), 1) if avg_pages else 0,
            'bounce_rate': round(bounce_rate, 2),
            'duration_distribution': [{'range': d[0], 'count': d[1]} for d in duration_distribution],
            'pages_distribution': [{'range': p[0], 'count': p[1]} for p in pages_distribution]
        }), 200
        
    except Exception as e:
        logger.error(f"Error in session analytics: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/analytics/user-journey', methods=['GET'])
@admin_required
def get_user_journey():
    """Get user journey and page flow analytics"""
    try:
        limit = request.args.get('limit', 100, type=int)
        period = request.args.get('period', 'week')
        
        now = datetime.utcnow()
        if period == 'today':
            start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == 'week':
            start_date = now - timedelta(days=7)
        elif period == 'month':
            start_date = now - timedelta(days=30)
        else:
            start_date = now - timedelta(days=365)
        
        # Entry pages (first page in session)
        entry_pages = db.session.query(
            Visitor.page_url,
            db.func.count(db.func.distinct(Visitor.visitor_hash)).label('count')
        ).filter(Visitor.timestamp >= start_date)\
        .group_by(Visitor.page_url)\
        .order_by(db.text('count DESC'))\
        .limit(20).all()
        
        # Exit pages (last page in session)
        exit_pages = db.session.query(
            PageView.page_url,
            db.func.count(PageView.id).label('count')
        ).filter(PageView.timestamp >= start_date)\
        .group_by(PageView.page_url)\
        .order_by(db.text('count DESC'))\
        .limit(20).all()
        
        # Most common page sequences
        sequences = db.session.query(
            PageView.visitor_hash,
            db.func.array_agg(PageView.page_url).label('pages')
        ).filter(PageView.timestamp >= start_date)\
        .group_by(PageView.visitor_hash)\
        .limit(limit).all()
        
        return jsonify({
            'entry_pages': [{'page': e[0], 'count': e[1]} for e in entry_pages],
            'exit_pages': [{'page': e[0], 'count': e[1]} for e in exit_pages],
            'sample_journeys': [{'visitor': s[0], 'pages': s[1]} for s in sequences[:20]]
        }), 200
        
    except Exception as e:
        logger.error(f"Error in user journey analytics: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/analytics/performance', methods=['GET'])
@admin_required
def get_performance_metrics():
    """Get performance and engagement metrics"""
    try:
        period = request.args.get('period', 'week')
        
        now = datetime.utcnow()
        if period == 'today':
            start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == 'week':
            start_date = now - timedelta(days=7)
        elif period == 'month':
            start_date = now - timedelta(days=30)
        else:
            start_date = now - timedelta(days=365)
        
        # Page load times (if you add this tracking)
        # Average time on page
        avg_drug_view_time = db.session.query(db.func.avg(DrugView.view_duration))\
            .filter(
                DrugView.timestamp >= start_date,
                DrugView.view_duration.isnot(None)
            ).scalar()
        
        # Engagement rate
        total_visitors = db.session.query(db.func.count(db.func.distinct(Visitor.visitor_hash)))\
            .filter(Visitor.timestamp >= start_date).scalar() or 1
        
        engaged_visitors = db.session.query(db.func.count(db.func.distinct(VisitorSession.visitor_hash)))\
            .filter(
                VisitorSession.session_start >= start_date,
                VisitorSession.duration_seconds > 30
            ).scalar() or 0
        
        engagement_rate = (engaged_visitors / total_visitors * 100) if total_visitors > 0 else 0
        
        # Return visitor rate
        returning_visitors = db.session.query(
            Visitor.visitor_hash,
            db.func.count(Visitor.id).label('visits')
        ).filter(Visitor.timestamp >= start_date)\
        .group_by(Visitor.visitor_hash)\
        .having(db.func.count(Visitor.id) > 1)\
        .all()
        
        return_rate = (len(returning_visitors) / total_visitors * 100) if total_visitors > 0 else 0
        
        return jsonify({
            'avg_drug_view_duration': round(float(avg_drug_view_time), 1) if avg_drug_view_time else 0,
            'engagement_rate': round(engagement_rate, 2),
            'return_visitor_rate': round(return_rate, 2),
            'returning_visitors': len(returning_visitors)
        }), 200
        
    except Exception as e:
        logger.error(f"Error in performance metrics: {str(e)}")
        return jsonify({"error": str(e)}), 500    


@app.route('/api/analytics/visitors', methods=['GET'])
@admin_required
def get_visitors():
    """Get detailed visitor information"""
    try:
        limit = request.args.get('limit', 100, type=int)
        
        # Get recent visitors with ALL detailed data
        visitors = db.session.query(Visitor)\
            .order_by(Visitor.timestamp.desc())\
            .limit(limit)\
            .all()
        
        return jsonify([{
            'ip_address': v.ip_address,
            'country': v.country,
            'city': v.city,
            'region': v.region,
            'page_url': v.page_url,
            'referrer': v.referrer,
            'timestamp': v.timestamp.isoformat() if v.timestamp else None,
            'language': v.language,
            'screen_width': v.screen_width,
            'screen_height': v.screen_height,
            'viewport_width': v.viewport_width,
            'viewport_height': v.viewport_height,
            'device_memory': v.device_memory,
            'hardware_concurrency': v.hardware_concurrency,
            'connection_type': v.connection_type,
            'latitude': v.latitude,
            'longitude': v.longitude,
            'timezone': v.timezone,
            'isp': v.isp,
            'user_agent': v.user_agent
        } for v in visitors]), 200
        
    except Exception as e:
        logger.error(f"Error getting visitors: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/analytics/stats', methods=['GET'])
@admin_required
def get_stats():
    try:
        unique_visitors = db.session.query(db.func.count(db.func.distinct(Visitor.visitor_hash))).scalar()
        total_pageviews = db.session.query(db.func.count(PageView.id)).scalar()
        
        top_countries = db.session.query(
            Visitor.country,
            db.func.count(Visitor.id).label('count')
        ).filter(Visitor.country.isnot(None)).group_by(Visitor.country).order_by(db.text('count DESC')).limit(10).all()
        
        top_pages = db.session.query(
            PageView.page_url,
            db.func.count(PageView.id).label('count')
        ).filter(PageView.page_url.isnot(None)).group_by(PageView.page_url).order_by(db.text('count DESC')).limit(10).all()
        
        return jsonify({
            "unique_visitors": unique_visitors or 0,
            "total_pageviews": total_pageviews or 0,
            "top_countries": [{"country": c[0], "count": c[1]} for c in top_countries],
            "top_pages": [{"page": p[0], "count": p[1]} for p in top_pages]
        }), 200
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/analytics/locations', methods=['GET'])
@admin_required
def get_locations():
    try:
        locations = db.session.query(
            Visitor.country,
            Visitor.city,
            db.func.count(Visitor.id).label('count')
        ).filter(Visitor.country.isnot(None)).group_by(Visitor.country, Visitor.city).order_by(db.text('count DESC')).all()
        
        result = [{"country": l[0], "city": l[1], "count": l[2]} for l in locations]
        
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Error getting locations: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500
    
#Statistics!!!
@app.route('/api/analytics/dashboard', methods=['GET'])
@admin_required
def get_analytics_dashboard():
    """
    Comprehensive dashboard stats with time period filtering
    Query params: period (today, week, month, year, all)
    """
    try:
        period = request.args.get('period', 'week')
        
        # Calculate date range
        now = datetime.utcnow()
        if period == 'today':
            start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == 'week':
            start_date = now - timedelta(days=7)
        elif period == 'month':
            start_date = now - timedelta(days=30)
        elif period == 'year':
            start_date = now - timedelta(days=365)
        else:  # all
            start_date = datetime(2020, 1, 1)
        
        # Basic metrics
        unique_visitors = db.session.query(db.func.count(db.func.distinct(Visitor.visitor_hash)))\
            .filter(Visitor.timestamp >= start_date).scalar() or 0
        
        total_pageviews = db.session.query(db.func.count(PageView.id))\
            .filter(PageView.timestamp >= start_date).scalar() or 0
        
        total_users = db.session.query(db.func.count(User.id)).scalar() or 0
        
        new_users = db.session.query(db.func.count(User.id))\
            .filter(User.last_seen >= start_date).scalar() or 0
        
        online_users = db.session.query(db.func.count(User.id))\
            .filter(User.last_seen >= now - timedelta(minutes=5)).scalar() or 0
        
        # Advanced metrics
        total_searches = db.session.query(db.func.count(SearchQuery.id))\
            .filter(SearchQuery.timestamp >= start_date).scalar() or 0
        
        total_drug_views = db.session.query(db.func.count(DrugView.id))\
            .filter(DrugView.timestamp >= start_date).scalar() or 0
        
        total_interaction_checks = db.session.query(db.func.count(InteractionCheck.id))\
            .filter(InteractionCheck.timestamp >= start_date).scalar() or 0
        
        # Calculate averages
        avg_pageviews_per_visitor = round(total_pageviews / unique_visitors, 2) if unique_visitors > 0 else 0
        
        # Top pages
        top_pages = db.session.query(
            PageView.page_url,
            db.func.count(PageView.id).label('views')
        ).filter(PageView.timestamp >= start_date)\
        .group_by(PageView.page_url)\
        .order_by(db.text('views DESC'))\
        .limit(10).all()
        
        # Top countries
        top_countries = db.session.query(
            Visitor.country,
            db.func.count(Visitor.id).label('count')
        ).filter(Visitor.timestamp >= start_date, Visitor.country.isnot(None))\
        .group_by(Visitor.country)\
        .order_by(db.text('count DESC'))\
        .limit(10).all()
        
        # Top cities
        top_cities = db.session.query(
            Visitor.city,
            Visitor.country,
            db.func.count(Visitor.id).label('count')
        ).filter(Visitor.timestamp >= start_date, Visitor.city.isnot(None))\
        .group_by(Visitor.city, Visitor.country)\
        .order_by(db.text('count DESC'))\
        .limit(10).all()
        
        # Referrer stats
        top_referrers = db.session.query(
            Visitor.referrer,
            db.func.count(Visitor.id).label('count')
        ).filter(Visitor.timestamp >= start_date, Visitor.referrer.isnot(None), Visitor.referrer != '')\
        .group_by(Visitor.referrer)\
        .order_by(db.text('count DESC'))\
        .limit(10).all()
        
        # Time series data (daily breakdown)
        daily_visits = db.session.query(
            db.func.date(Visitor.timestamp).label('date'),
            db.func.count(db.func.distinct(Visitor.visitor_hash)).label('visitors'),
            db.func.count(Visitor.id).label('visits')
        ).filter(Visitor.timestamp >= start_date)\
        .group_by(db.func.date(Visitor.timestamp))\
        .order_by(db.text('date')).all()
        
        return jsonify({
            'period': period,
            'date_range': {
                'start': start_date.isoformat(),
                'end': now.isoformat()
            },
            'overview': {
                'unique_visitors': unique_visitors,
                'total_pageviews': total_pageviews,
                'total_users': total_users,
                'new_users': new_users,
                'online_users': online_users,
                'avg_pageviews_per_visitor': avg_pageviews_per_visitor,
                'total_searches': total_searches,
                'total_drug_views': total_drug_views,
                'total_interaction_checks': total_interaction_checks
            },
            'top_pages': [{'url': p[0], 'views': p[1]} for p in top_pages],
            'top_countries': [{'country': c[0], 'count': c[1]} for c in top_countries],
            'top_cities': [{'city': c[0], 'country': c[1], 'count': c[2]} for c in top_cities],
            'top_referrers': [{'referrer': r[0], 'count': r[1]} for r in top_referrers],
            'daily_breakdown': [{
                'date': d[0].isoformat() if d[0] else None,
                'unique_visitors': d[1],
                'total_visits': d[2]
            } for d in daily_visits]
        }), 200
        
    except Exception as e:
        logger.error(f"Error in analytics dashboard: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/analytics/searches', methods=['GET'])
@admin_required
def get_search_analytics():
    """Get search query analytics"""
    try:
        limit = request.args.get('limit', 50, type=int)
        period = request.args.get('period', 'week')
        
        # Calculate date range
        now = datetime.utcnow()
        if period == 'today':
            start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == 'week':
            start_date = now - timedelta(days=7)
        elif period == 'month':
            start_date = now - timedelta(days=30)
        else:
            start_date = now - timedelta(days=365)
        
        # Top searches
        top_searches = db.session.query(
            SearchQuery.query_text,
            db.func.count(SearchQuery.id).label('count'),
            db.func.avg(SearchQuery.results_count).label('avg_results')
        ).filter(SearchQuery.timestamp >= start_date)\
        .group_by(SearchQuery.query_text)\
        .order_by(db.text('count DESC'))\
        .limit(limit).all()
        
        # Search categories breakdown
        category_breakdown = db.session.query(
            SearchQuery.category,
            db.func.count(SearchQuery.id).label('count')
        ).filter(SearchQuery.timestamp >= start_date, SearchQuery.category.isnot(None))\
        .group_by(SearchQuery.category)\
        .order_by(db.text('count DESC')).all()
        
        # Recent searches
        recent_searches = db.session.query(SearchQuery)\
            .filter(SearchQuery.timestamp >= start_date)\
            .order_by(SearchQuery.timestamp.desc())\
            .limit(20).all()
        
        # Zero result searches
        zero_results = db.session.query(
            SearchQuery.query_text,
            db.func.count(SearchQuery.id).label('count')
        ).filter(SearchQuery.timestamp >= start_date, SearchQuery.results_count == 0)\
        .group_by(SearchQuery.query_text)\
        .order_by(db.text('count DESC'))\
        .limit(20).all()
        
        return jsonify({
            'top_searches': [{
                'query': s[0],
                'count': s[1],
                'avg_results': round(float(s[2]), 1) if s[2] else 0
            } for s in top_searches],
            'category_breakdown': [{
                'category': c[0],
                'count': c[1]
            } for c in category_breakdown],
            'recent_searches': [{
                'query': s.query_text,
                'category': s.category,
                'results': s.results_count,
                'timestamp': s.timestamp.isoformat()
            } for s in recent_searches],
            'zero_results': [{
                'query': z[0],
                'count': z[1]
            } for z in zero_results]
        }), 200
        
    except Exception as e:
        logger.error(f"Error in search analytics: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/analytics/drugs', methods=['GET'])
@admin_required
def get_drug_analytics():
    """Get drug viewing analytics"""
    try:
        limit = request.args.get('limit', 20, type=int)
        period = request.args.get('period', 'week')
        
        # Calculate date range
        now = datetime.utcnow()
        if period == 'today':
            start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == 'week':
            start_date = now - timedelta(days=7)
        elif period == 'month':
            start_date = now - timedelta(days=30)
        else:
            start_date = now - timedelta(days=365)
        
        # Most viewed drugs
        top_drugs = db.session.query(
            Drug.id,
            Drug.name_en,
            Drug.name_tr,
            db.func.count(DrugView.id).label('views')
        ).join(DrugView, Drug.id == DrugView.drug_id)\
        .filter(DrugView.timestamp >= start_date)\
        .group_by(Drug.id, Drug.name_en, Drug.name_tr)\
        .order_by(db.text('views DESC'))\
        .limit(limit).all()
        
        # Total drug views
        total_views = db.session.query(db.func.count(DrugView.id))\
            .filter(DrugView.timestamp >= start_date).scalar() or 0
        
        # Average view duration
        avg_duration = db.session.query(db.func.avg(DrugView.view_duration))\
            .filter(DrugView.timestamp >= start_date, DrugView.view_duration.isnot(None)).scalar()
        
        return jsonify({
            'total_views': total_views,
            'avg_duration_seconds': round(float(avg_duration), 1) if avg_duration else 0,
            'top_drugs': [{
                'id': d[0],
                'name_en': d[1],
                'name_tr': d[2],
                'views': d[3]
            } for d in top_drugs]
        }), 200
        
    except Exception as e:
        logger.error(f"Error in drug analytics: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/analytics/interactions', methods=['GET'])
@admin_required
def get_interaction_analytics():
    """Get drug interaction check analytics"""
    try:
        period = request.args.get('period', 'week')
        
        # Calculate date range
        now = datetime.utcnow()
        if period == 'today':
            start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == 'week':
            start_date = now - timedelta(days=7)
        elif period == 'month':
            start_date = now - timedelta(days=30)
        else:
            start_date = now - timedelta(days=365)
        
        # Total checks
        total_checks = db.session.query(db.func.count(InteractionCheck.id))\
            .filter(InteractionCheck.timestamp >= start_date).scalar() or 0
        
        # Average interactions found per check
        avg_interactions = db.session.query(db.func.avg(InteractionCheck.interactions_found))\
            .filter(InteractionCheck.timestamp >= start_date).scalar()
        
        # Recent checks
        recent_checks = db.session.query(InteractionCheck)\
            .filter(InteractionCheck.timestamp >= start_date)\
            .order_by(InteractionCheck.timestamp.desc())\
            .limit(20).all()
        
        return jsonify({
            'total_checks': total_checks,
            'avg_interactions_found': round(float(avg_interactions), 2) if avg_interactions else 0,
            'recent_checks': [{
                'drug_count': len(c.drug_ids),
                'interactions_found': c.interactions_found,
                'severity_levels': c.severity_levels,
                'timestamp': c.timestamp.isoformat()
            } for c in recent_checks]
        }), 200
        
    except Exception as e:
        logger.error(f"Error in interaction analytics: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/analytics/users', methods=['GET'])
@admin_required
def get_user_analytics():
    """Get registered user analytics"""
    try:
        period = request.args.get('period', 'week')
        
        # Calculate date range
        now = datetime.utcnow()
        if period == 'today':
            start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == 'week':
            start_date = now - timedelta(days=7)
        elif period == 'month':
            start_date = now - timedelta(days=30)
        else:
            start_date = now - timedelta(days=365)
        
        # Total users
        total_users = db.session.query(db.func.count(User.id)).scalar() or 0
        
        # New users in period
        new_users = db.session.query(db.func.count(User.id))\
            .filter(User.last_seen >= start_date).scalar() or 0
        
        # Online users (last 5 minutes)
        online_users = db.session.query(db.func.count(User.id))\
            .filter(User.last_seen >= now - timedelta(minutes=5)).scalar() or 0
        
        # Verified users
        verified_users = db.session.query(db.func.count(User.id))\
            .filter(User.is_verified == True).scalar() or 0
        
        # Admin users
        admin_users = db.session.query(db.func.count(User.id))\
            .filter(User.is_admin == True).scalar() or 0
        
        # User activity over time
        daily_active_users = db.session.query(
            db.func.date(User.last_seen).label('date'),
            db.func.count(User.id).label('active_users')
        ).filter(User.last_seen >= start_date)\
        .group_by(db.func.date(User.last_seen))\
        .order_by(db.text('date')).all()
        
        # Most active users
        most_active = db.session.query(
            User.id,
            User.name,
            User.surname,
            User.email,
            db.func.count(SearchQuery.id).label('searches')
        ).join(SearchQuery, User.id == SearchQuery.user_id, isouter=True)\
        .group_by(User.id, User.name, User.surname, User.email)\
        .order_by(db.text('searches DESC'))\
        .limit(10).all()
        
        return jsonify({
            'total_users': total_users,
            'new_users': new_users,
            'online_users': online_users,
            'verified_users': verified_users,
            'admin_users': admin_users,
            'verification_rate': round((verified_users / total_users * 100), 2) if total_users > 0 else 0,
            'daily_active_users': [{
                'date': d[0].isoformat() if d[0] else None,
                'count': d[1]
            } for d in daily_active_users],
            'most_active_users': [{
                'id': u[0],
                'name': f"{u[1]} {u[2]}",
                'email': u[3],
                'searches': u[4]
            } for u in most_active]
        }), 200
        
    except Exception as e:
        logger.error(f"Error in user analytics: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/analytics/geo', methods=['GET'])
@admin_required
def get_geo_analytics():
    """Get geographical analytics with detailed breakdown"""
    try:
        period = request.args.get('period', 'week')
        
        # Calculate date range
        now = datetime.utcnow()
        if period == 'today':
            start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == 'week':
            start_date = now - timedelta(days=7)
        elif period == 'month':
            start_date = now - timedelta(days=30)
        else:
            start_date = now - timedelta(days=365)
        
        # Country statistics
        countries = db.session.query(
            Visitor.country,
            db.func.count(db.func.distinct(Visitor.visitor_hash)).label('unique_visitors'),
            db.func.count(Visitor.id).label('total_visits')
        ).filter(Visitor.timestamp >= start_date, Visitor.country.isnot(None))\
        .group_by(Visitor.country)\
        .order_by(db.text('total_visits DESC')).all()
        
        # City statistics for top countries
        cities = db.session.query(
            Visitor.country,
            Visitor.city,
            Visitor.region,
            db.func.count(Visitor.id).label('visits')
        ).filter(Visitor.timestamp >= start_date, Visitor.city.isnot(None))\
        .group_by(Visitor.country, Visitor.city, Visitor.region)\
        .order_by(db.text('visits DESC'))\
        .limit(50).all()
        
        # ISP statistics
        isps = db.session.query(
            Visitor.isp,
            db.func.count(Visitor.id).label('count')
        ).filter(Visitor.timestamp >= start_date, Visitor.isp.isnot(None))\
        .group_by(Visitor.isp)\
        .order_by(db.text('count DESC'))\
        .limit(20).all()
        
        # Timezone distribution
        timezones = db.session.query(
            Visitor.timezone,
            db.func.count(Visitor.id).label('count')
        ).filter(Visitor.timestamp >= start_date, Visitor.timezone.isnot(None))\
        .group_by(Visitor.timezone)\
        .order_by(db.text('count DESC')).all()
        
        return jsonify({
            'countries': [{
                'country': c[0],
                'unique_visitors': c[1],
                'total_visits': c[2]
            } for c in countries],
            'cities': [{
                'country': c[0],
                'city': c[1],
                'region': c[2],
                'visits': c[3]
            } for c in cities],
            'isps': [{
                'isp': i[0],
                'count': i[1]
            } for i in isps],
            'timezones': [{
                'timezone': t[0],
                'count': t[1]
            } for t in timezones]
        }), 200
        
    except Exception as e:
        logger.error(f"Error in geo analytics: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/analytics/realtime', methods=['GET'])
@admin_required
def get_realtime_analytics():
    """Get real-time analytics (last 5 minutes)"""
    try:
        five_min_ago = datetime.utcnow() - timedelta(minutes=5)
        
        # Active visitors right now
        active_visitors = db.session.query(db.func.count(db.func.distinct(Visitor.visitor_hash)))\
            .filter(Visitor.timestamp >= five_min_ago).scalar() or 0
        
        # Active pages
        active_pages = db.session.query(
            PageView.page_url,
            db.func.count(PageView.id).label('views')
        ).filter(PageView.timestamp >= five_min_ago)\
        .group_by(PageView.page_url)\
        .order_by(db.text('views DESC'))\
        .limit(10).all()
        
        # Recent visitors
        recent_visitors = db.session.query(Visitor)\
            .filter(Visitor.timestamp >= five_min_ago)\
            .order_by(Visitor.timestamp.desc())\
            .limit(20).all()
        
        # Active searches
        recent_searches = db.session.query(SearchQuery)\
            .filter(SearchQuery.timestamp >= five_min_ago)\
            .order_by(SearchQuery.timestamp.desc())\
            .limit(10).all()
        
        return jsonify({
            'active_visitors': active_visitors,
            'active_pages': [{
                'url': p[0],
                'views': p[1]
            } for p in active_pages],
            'recent_visitors': [{
                'country': v.country,
                'city': v.city,
                'page': v.page_url,
                'timestamp': v.timestamp.isoformat()
            } for v in recent_visitors],
            'recent_searches': [{
                'query': s.query_text,
                'category': s.category,
                'timestamp': s.timestamp.isoformat()
            } for s in recent_searches],
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error in realtime analytics: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/analytics/export', methods=['GET'])
@admin_required
def export_analytics():
    """Export analytics data as CSV"""
    try:
        data_type = request.args.get('type', 'visitors')
        period = request.args.get('period', 'week')
        
        # Calculate date range
        now = datetime.utcnow()
        if period == 'today':
            start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == 'week':
            start_date = now - timedelta(days=7)
        elif period == 'month':
            start_date = now - timedelta(days=30)
        else:
            start_date = now - timedelta(days=365)
        
        output = StringIO()
        writer = csv.writer(output)
        
        if data_type == 'visitors':
            writer.writerow(['Timestamp', 'IP Address', 'Country', 'City', 'Page URL', 'Referrer'])
            visitors = Visitor.query.filter(Visitor.timestamp >= start_date)\
                .order_by(Visitor.timestamp.desc()).all()
            for v in visitors:
                writer.writerow([
                    v.timestamp.isoformat() if v.timestamp else '',
                    v.ip_address or '',
                    v.country or '',
                    v.city or '',
                    v.page_url or '',
                    v.referrer or ''
                ])
        
        elif data_type == 'searches':
            writer.writerow(['Timestamp', 'Query', 'Category', 'Results Count', 'User ID'])
            searches = SearchQuery.query.filter(SearchQuery.timestamp >= start_date)\
                .order_by(SearchQuery.timestamp.desc()).all()
            for s in searches:
                writer.writerow([
                    s.timestamp.isoformat() if s.timestamp else '',
                    s.query_text or '',
                    s.category or '',
                    s.results_count or 0,
                    s.user_id or ''
                ])
        
        elif data_type == 'drug_views':
            writer.writerow(['Timestamp', 'Drug ID', 'Drug Name', 'View Duration', 'User ID'])
            drug_views = db.session.query(DrugView, Drug)\
                .join(Drug, DrugView.drug_id == Drug.id)\
                .filter(DrugView.timestamp >= start_date)\
                .order_by(DrugView.timestamp.desc()).all()
            for dv, drug in drug_views:
                writer.writerow([
                    dv.timestamp.isoformat() if dv.timestamp else '',
                    drug.id,
                    drug.name_en or '',
                    dv.view_duration or 0,
                    dv.user_id or ''
                ])
        
        # Create response
        output.seek(0)
        return send_file(
            BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'analytics_{data_type}_{period}_{now.strftime("%Y%m%d")}.csv'
        )
        
    except Exception as e:
        logger.error(f"Error exporting analytics: {str(e)}")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# TRACKING HELPER FUNCTIONS
# Add these after your existing tracking functions
# ============================================================================

@app.route('/api/track/search', methods=['POST'])
def track_search():
    """Track search queries"""
    try:
        data = request.json
        visitor_hash = get_visitor_hash(
            request.headers.get('X-Forwarded-For', request.remote_addr).split(',')[0].strip(),
            request.headers.get('User-Agent', '')
        )
        
        search = SearchQuery(
            query_text=data.get('query'),
            visitor_hash=visitor_hash,
            user_id=session.get('user_id'),
            results_count=data.get('results_count', 0),
            category=data.get('category'),
            clicked_result_position=data.get('clicked_result_position'),
            clicked_result_id=data.get('clicked_result_id'),
            time_to_first_click=data.get('time_to_first_click'),
            refined_query=data.get('refined_query', False)
        )
        db.session.add(search)
        db.session.commit()
        
        return jsonify({"status": "success"}), 200
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error tracking search: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500
    
@app.route('/api/track/drug-view', methods=['POST'])
def track_drug_view():
    """Track drug page views"""
    try:
        data = request.json
        visitor_hash = get_visitor_hash(
            request.headers.get('X-Forwarded-For', request.remote_addr).split(',')[0].strip(),
            request.headers.get('User-Agent', '')
        )
        
        drug_view = DrugView(
            drug_id=data.get('drug_id'),
            visitor_hash=visitor_hash,
            user_id=session.get('user_id'),
            view_duration=data.get('duration'),
            sections_viewed=data.get('sections_viewed'),
            scroll_depth=data.get('scroll_depth'),
            interactions=data.get('interactions', 0),
            shared=data.get('shared', False),
            bookmarked=data.get('bookmarked', False)
        )
        db.session.add(drug_view)
        db.session.commit()
        
        return jsonify({"status": "success"}), 200
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error tracking drug view: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500
    
@app.route('/api/track/interaction', methods=['POST'])
def track_interaction():
    """Track interaction checks"""
    try:
        data = request.json
        visitor_hash = get_visitor_hash(
            request.headers.get('X-Forwarded-For', request.remote_addr).split(',')[0].strip(),
            request.headers.get('User-Agent', '')
        )
        
        interaction_check = InteractionCheck(
            drug_ids=data.get('drug_ids'),
            visitor_hash=visitor_hash,
            user_id=session.get('user_id'),
            interactions_found=data.get('interactions_found', 0),
            severity_levels=data.get('severity_levels'),
            checked_food_interactions=data.get('checked_food_interactions', False),
            checked_disease_interactions=data.get('checked_disease_interactions', False),
            checked_lab_interactions=data.get('checked_lab_interactions', False),
            time_spent=data.get('time_spent'),
            exported=data.get('exported', False)
        )
        db.session.add(interaction_check)
        db.session.commit()
        
        return jsonify({"status": "success"}), 200
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error tracking interaction: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/track/event', methods=['POST'])
def track_event():
    """Track custom events"""
    try:
        data = request.json
        visitor_hash = get_visitor_hash(
            request.headers.get('X-Forwarded-For', request.remote_addr).split(',')[0].strip(),
            request.headers.get('User-Agent', '')
        )
        
        event = AnalyticsEvent(
            event_type=data.get('event_type'),
            event_category=data.get('event_category'),
            event_data=data.get('event_data'),
            visitor_hash=visitor_hash,
            user_id=session.get('user_id'),
            page_url=data.get('page_url'),
            event_value=data.get('event_value')
        )
        db.session.add(event)
        db.session.commit()
        
        return jsonify({"status": "success"}), 200
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error tracking event: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500
    
@app.route('/api/track/click', methods=['POST'])
def track_click():
    """Track click events"""
    try:
        data = request.json
        visitor_hash = get_visitor_hash(
            request.headers.get('X-Forwarded-For', request.remote_addr).split(',')[0].strip(),
            request.headers.get('User-Agent', '')
        )
        
        click = UserClick(
            visitor_hash=visitor_hash,
            user_id=session.get('user_id'),
            element_id=data.get('element_id'),
            element_class=data.get('element_class'),
            element_tag=data.get('element_tag'),
            element_text=data.get('element_text'),
            page_url=data.get('page_url'),
            x_position=data.get('x_position'),
            y_position=data.get('y_position')
        )
        db.session.add(click)
        db.session.commit()
        
        return jsonify({"status": "success"}), 200
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error tracking click: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/track/scroll', methods=['POST'])
def track_scroll():
    """Track scroll events"""
    try:
        data = request.json
        visitor_hash = get_visitor_hash(
            request.headers.get('X-Forwarded-For', request.remote_addr).split(',')[0].strip(),
            request.headers.get('User-Agent', '')
        )
        
        scroll = UserScroll(
            visitor_hash=visitor_hash,
            user_id=session.get('user_id'),
            page_url=data.get('page_url'),
            scroll_depth=data.get('scroll_depth'),
            scroll_percentage=data.get('scroll_percentage')
        )
        db.session.add(scroll)
        db.session.commit()
        
        return jsonify({"status": "success"}), 200
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error tracking scroll: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/track/form', methods=['POST'])
def track_form():
    """Track form submissions"""
    try:
        data = request.json
        visitor_hash = get_visitor_hash(
            request.headers.get('X-Forwarded-For', request.remote_addr).split(',')[0].strip(),
            request.headers.get('User-Agent', '')
        )
        
        form = FormSubmission(
            visitor_hash=visitor_hash,
            user_id=session.get('user_id'),
            form_name=data.get('form_name'),
            page_url=data.get('page_url'),
            fields_filled=data.get('fields_filled'),
            submission_success=data.get('submission_success', True),
            time_to_submit=data.get('time_to_submit')
        )
        db.session.add(form)
        db.session.commit()
        
        return jsonify({"status": "success"}), 200
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error tracking form: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/track/error', methods=['POST'])
def track_error():
    """Track client-side errors"""
    try:
        data = request.json
        visitor_hash = get_visitor_hash(
            request.headers.get('X-Forwarded-For', request.remote_addr).split(',')[0].strip(),
            request.headers.get('User-Agent', '')
        )
        
        error = ErrorLog(
            visitor_hash=visitor_hash,
            user_id=session.get('user_id'),
            error_message=data.get('error_message'),
            error_stack=data.get('error_stack'),
            page_url=data.get('page_url'),
            browser=data.get('browser'),
            os=data.get('os')
        )
        db.session.add(error)
        db.session.commit()
        
        return jsonify({"status": "success"}), 200
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error tracking error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/analytics/clicks', methods=['GET'])
@admin_required
def get_click_analytics():
    """Get click analytics"""
    try:
        period = request.args.get('period', 'week')
        
        now = datetime.utcnow()
        if period == 'today':
            start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == 'week':
            start_date = now - timedelta(days=7)
        elif period == 'month':
            start_date = now - timedelta(days=30)
        else:
            start_date = now - timedelta(days=365)
        
        # Total clicks
        total_clicks = db.session.query(db.func.count(UserClick.id))\
            .filter(UserClick.timestamp >= start_date).scalar() or 0
        
        # Total sessions
        total_sessions = db.session.query(db.func.count(VisitorSession.id))\
            .filter(VisitorSession.session_start >= start_date).scalar() or 1
        
        # Avg clicks per session
        avg_clicks_per_session = total_clicks / total_sessions if total_sessions > 0 else 0
        
        # Click-through rate (sessions with clicks / total sessions)
        sessions_with_clicks = db.session.query(db.func.count(db.func.distinct(UserClick.visitor_hash)))\
            .filter(UserClick.timestamp >= start_date).scalar() or 0
        click_through_rate = (sessions_with_clicks / total_sessions * 100) if total_sessions > 0 else 0
        
        # Top clicked elements
        top_clicked_elements = db.session.query(
            UserClick.element_id,
            UserClick.element_class,
            UserClick.element_tag,
            UserClick.element_text,
            db.func.count(UserClick.id).label('clicks')
        ).filter(UserClick.timestamp >= start_date)\
        .group_by(UserClick.element_id, UserClick.element_class, UserClick.element_tag, UserClick.element_text)\
        .order_by(db.text('clicks DESC'))\
        .limit(20).all()
        
        # Click heatmap
        click_heatmap = db.session.query(
            UserClick.page_url,
            UserClick.x_position,
            UserClick.y_position,
            db.func.count(UserClick.id).label('clicks')
        ).filter(UserClick.timestamp >= start_date)\
        .group_by(UserClick.page_url, UserClick.x_position, UserClick.y_position)\
        .order_by(db.text('clicks DESC'))\
        .limit(100).all()
        
        return jsonify({
            'total_clicks': total_clicks,
            'avg_clicks_per_session': round(avg_clicks_per_session, 2),
            'click_through_rate': round(click_through_rate, 2),
            'top_clicked_elements': [{
                'element_id': c[0],
                'element_class': c[1],
                'element_tag': c[2],
                'element_text': c[3],
                'clicks': c[4]
            } for c in top_clicked_elements],
            'click_heatmap': [{
                'page': c[0],
                'x': c[1],
                'y': c[2],
                'clicks': c[3]
            } for c in click_heatmap]
        }), 200
        
    except Exception as e:
        logger.error(f"Error in click analytics: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/analytics/scrolls', methods=['GET'])
@admin_required
def get_scroll_analytics():
    """Get scroll analytics"""
    try:
        period = request.args.get('period', 'week')
        
        now = datetime.utcnow()
        if period == 'today':
            start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == 'week':
            start_date = now - timedelta(days=7)
        elif period == 'month':
            start_date = now - timedelta(days=30)
        else:
            start_date = now - timedelta(days=365)
        
        avg_scroll_depth = db.session.query(
            UserScroll.page_url,
            db.func.avg(UserScroll.scroll_percentage).label('avg_scroll')
        ).filter(UserScroll.timestamp >= start_date)\
        .group_by(UserScroll.page_url)\
        .order_by(db.text('avg_scroll DESC'))\
        .limit(20).all()
        
        scroll_distribution = db.session.query(
            db.case(
                (UserScroll.scroll_percentage < 25, '0-25%'),
                (UserScroll.scroll_percentage < 50, '25-50%'),
                (UserScroll.scroll_percentage < 75, '50-75%'),
                (UserScroll.scroll_percentage < 100, '75-100%'),
                else_='100%'
            ).label('range'),
            db.func.count(UserScroll.id).label('count')
        ).filter(UserScroll.timestamp >= start_date)\
        .group_by(db.text('range')).all()
        
        return jsonify({
            'avg_scroll_by_page': [{
                'page': s[0],
                'avg_scroll': round(float(s[1]), 2)
            } for s in avg_scroll_depth],
            'scroll_distribution': [{
                'range': s[0],
                'count': s[1]
            } for s in scroll_distribution]
        }), 200
        
    except Exception as e:
        logger.error(f"Error in scroll analytics: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/analytics/forms', methods=['GET'])
@admin_required
def get_form_analytics():
    """Get form submission analytics"""
    try:
        period = request.args.get('period', 'week')
        
        now = datetime.utcnow()
        if period == 'today':
            start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == 'week':
            start_date = now - timedelta(days=7)
        elif period == 'month':
            start_date = now - timedelta(days=30)
        else:
            start_date = now - timedelta(days=365)
        
        form_stats = db.session.query(
            FormSubmission.form_name,
            db.func.count(FormSubmission.id).label('submissions'),
            db.func.avg(FormSubmission.time_to_submit).label('avg_time'),
            db.func.sum(db.case((FormSubmission.submission_success == True, 1), else_=0)).label('successful')
        ).filter(FormSubmission.timestamp >= start_date)\
        .group_by(FormSubmission.form_name).all()
        
        return jsonify({
            'forms': [{
                'name': f[0],
                'submissions': f[1],
                'avg_time_seconds': round(float(f[2]), 1) if f[2] else 0,
                'success_rate': round((f[3] / f[1] * 100), 2) if f[1] > 0 else 0
            } for f in form_stats]
        }), 200
        
    except Exception as e:
        logger.error(f"Error in form analytics: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/analytics/errors', methods=['GET'])
@admin_required
def get_error_analytics():
    """Get error analytics"""
    try:
        period = request.args.get('period', 'week')
        
        now = datetime.utcnow()
        if period == 'today':
            start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == 'week':
            start_date = now - timedelta(days=7)
        elif period == 'month':
            start_date = now - timedelta(days=30)
        else:
            start_date = now - timedelta(days=365)
        
        total_errors = db.session.query(db.func.count(ErrorLog.id))\
            .filter(ErrorLog.timestamp >= start_date).scalar() or 0
        
        top_errors = db.session.query(
            ErrorLog.error_message,
            ErrorLog.page_url,
            db.func.count(ErrorLog.id).label('count')
        ).filter(ErrorLog.timestamp >= start_date)\
        .group_by(ErrorLog.error_message, ErrorLog.page_url)\
        .order_by(db.text('count DESC'))\
        .limit(20).all()
        
        errors_by_browser = db.session.query(
            ErrorLog.browser,
            db.func.count(ErrorLog.id).label('count')
        ).filter(ErrorLog.timestamp >= start_date)\
        .group_by(ErrorLog.browser)\
        .order_by(db.text('count DESC')).all()
        
        recent_errors = db.session.query(ErrorLog)\
            .filter(ErrorLog.timestamp >= start_date)\
            .order_by(ErrorLog.timestamp.desc())\
            .limit(50).all()
        
        return jsonify({
            'total_errors': total_errors,
            'top_errors': [{
                'message': e[0],
                'page': e[1],
                'count': e[2]
            } for e in top_errors],
            'errors_by_browser': [{
                'browser': e[0],
                'count': e[1]
            } for e in errors_by_browser],
            'recent_errors': [{
                'message': e.error_message,
                'page': e.page_url,
                'browser': e.browser,
                'timestamp': e.timestamp.isoformat()
            } for e in recent_errors]
        }), 200
        
    except Exception as e:
        logger.error(f"Error in error analytics: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
# Route to manually trigger daily stats aggregation (admin only)
@app.route('/api/analytics/aggregate-daily', methods=['POST'])
@admin_required
def trigger_daily_aggregation():
    """Manually trigger daily stats aggregation"""
    try:
        aggregate_daily_stats()
        return jsonify({"status": "success", "message": "Daily stats aggregated"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    with app.app_context():  # Enter the app context
        db.create_all()  # Create tables in the database
        print("Database tables created successfully.")
    app.run(debug=True)
