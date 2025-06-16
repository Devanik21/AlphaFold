import streamlit as st
import google.generativeai as genai
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
import string
import re
import time
from datetime import datetime
import hashlib
import json
import io
import base64

# Imports for practical CONF tab tools
from scipy.stats import entropy
from statsmodels.tsa.stattools import acf
# Configure page
st.set_page_config(
    layout="wide", 
    page_title="AlphaFold Pro - Protein Structure Prediction Suite",
    page_icon="🧬",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .metric-card { /* Changed background to a darker shade */
        background: #2c3e50; /* Dark blue-gray */
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2a5298;
        margin: 0.5rem 0;
    }
    .prediction-status {
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
    }
    .status-running { /* Darker yellow/orange */
        background: #4A3B00; /* Dark yellow */
        color: #FFD700; /* Brighter yellow text */
    }
    .status-complete { /* Darker green */
        background: #1A3A1F; /* Dark green */
        color: #A8D5BA; /* Lighter green text */
    }
    .status-error { /* Darker red */
        background: #4C1C24; /* Dark red */
        color: #F5C6CB; /* Lighter red text */
    }
    .metric-card h4, .metric-card h2 { /* Ensure text in metric cards is light */
        color: #ecf0f1;
    }
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Constants
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
SECONDARY_STRUCTURES = ["Helix", "Sheet", "Coil", "Turn"]
CONFIDENCE_LEVELS = ["Very High (>90)", "High (70-90)", "Medium (50-70)", "Low (<50)"]

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'current_prediction' not in st.session_state:
    st.session_state.current_prediction = None
if 'analysis_cache' not in st.session_state:
    st.session_state.analysis_cache = {}

def generate_protein_sequence(length=None, complexity="medium"):
    """Generate realistic protein sequences based on complexity."""
    if length is None:
        length = random.randint(50, 300) if complexity == "medium" else random.randint(300, 800)
    
    # Realistic amino acid frequency distribution
    aa_weights = {
        'A': 8.25, 'R': 5.53, 'N': 4.06, 'D': 5.45, 'C': 1.37,
        'Q': 3.93, 'E': 6.75, 'G': 7.07, 'H': 2.27, 'I': 5.96,
        'L': 9.66, 'K': 5.84, 'M': 2.42, 'F': 3.86, 'P': 4.70,
        'S': 6.56, 'T': 5.34, 'W': 1.08, 'Y': 2.92, 'V': 6.87
    }
    
    amino_acids = list(aa_weights.keys())
    weights = list(aa_weights.values())
    
    sequence = ''.join(np.random.choice(amino_acids, size=length, p=np.array(weights)/sum(weights)))
    protein_id = f"PROT_{random.randint(10000, 99999)}"
    
    return f">{protein_id}\n{sequence}"

def validate_protein_sequence(sequence):
    """Validate protein sequence format and content."""
    lines = sequence.strip().split('\n')
    
    if not sequence.strip():
        return False, "Empty sequence"
    
    # Check FASTA format
    if sequence.startswith('>'):
        if len(lines) < 2:
            return False, "Invalid FASTA format: missing sequence"
        seq_lines = lines[1:]
    else:
        seq_lines = lines
    
    # Combine sequence lines
    seq = ''.join(seq_lines).upper().replace(' ', '').replace('\t', '')
    
    # Validate amino acids
    invalid_chars = set(seq) - set(AMINO_ACIDS)
    if invalid_chars:
        return False, f"Invalid amino acids found: {', '.join(invalid_chars)}"
    
    if len(seq) < 10:
        return False, "Sequence too short (minimum 10 residues)"
    
    if len(seq) > 2000:
        return False, "Sequence too long (maximum 2000 residues)"
    
    return True, seq

def generate_mock_predictions(sequence, model_name):
    """Generate comprehensive mock predictions."""
    seq_len = len(sequence)
    
    # Secondary structure prediction
    ss_pred = np.random.choice(SECONDARY_STRUCTURES, size=seq_len, 
                              p=[0.35, 0.25, 0.30, 0.10])
    
    # Confidence scores
    confidence = np.random.beta(3, 1, seq_len) * 100
    
    # Disorder prediction
    disorder_pred = np.random.random(seq_len) < 0.15
    
    # pLDDT scores (AlphaFold confidence)
    plddt = np.random.normal(75, 15, seq_len)
    plddt = np.clip(plddt, 0, 100)
    
    # Domain predictions
    domains = []
    if seq_len > 50:
        n_domains = max(1, seq_len // 150)
        for i in range(n_domains):
            start = random.randint(i * (seq_len // n_domains), 
                                 min((i + 1) * (seq_len // n_domains), seq_len - 20))
            end = min(start + random.randint(50, 100), seq_len)
            domains.append({
                'name': f'Domain_{i+1}',
                'start': start,
                'end': end,
                'type': random.choice(['Enzymatic', 'Binding', 'Structural', 'Regulatory'])
            })
    
    return {
        'sequence': sequence,
        'length': seq_len,
        'secondary_structure': ss_pred,
        'confidence': confidence,
        'disorder': disorder_pred,
        'plddt': plddt,
        'domains': domains,
        'model_used': model_name,
        'timestamp': datetime.now(),
        'overall_confidence': np.mean(plddt)
    }

def generate_mock_interaction_data(protein_id, num_interactions=5):
    interactions = []
    for i in range(num_interactions):
        partner_type = random.choice(["Protein", "Ligand"])
        if partner_type == "Protein":
            partner_id = f"PROT_{random.randint(1000,9999)}"
            interaction_detail = {"type": "PPI", "confidence": round(random.uniform(0.5, 0.99), 2)}
        else:
            partner_id = f"LIG_{''.join(random.choices(string.ascii_uppercase + string.digits, k=3))}"
            interaction_detail = {"type": "Protein-Ligand", "affinity_nM": round(random.uniform(10, 5000), 1)}
        interactions.append({
            "partner_id": partner_id,
            "details": interaction_detail
        })
    return interactions

def generate_mock_mutational_data(sequence_length, num_mutations=10):
    mutations = []
    for _ in range(num_mutations):
        pos = random.randint(1, sequence_length)
        original_aa = random.choice(AMINO_ACIDS)
        mutated_aa = random.choice([aa for aa in AMINO_ACIDS if aa != original_aa])
        ddg = round(random.uniform(-3.0, 3.0), 2)
        effect = "Neutral"
        if ddg > 1.0: effect = "Destabilizing"
        elif ddg < -1.0: effect = "Stabilizing"
        mutations.append({
            "Mutation": f"{original_aa}{pos}{mutated_aa}",
            "Predicted_ddG_kcal_mol": ddg,
            "Predicted_Effect": effect,
            "Tool": random.choice(["MockFoldX", "MockRosetta"])
        })
    return pd.DataFrame(mutations)

def generate_mock_protein_symmetry_data():
    symmetry_type = random.choice(["None", "C2", "C3", "C4", "D2", "D3", "Icosahedral (mock)"])
    if symmetry_type == "None":
        return {"type": "None", "axis": "N/A", "confidence": 0.0}
    return {
        "type": symmetry_type,
        "axis": random.choice(["X-axis", "Y-axis", "Z-axis", "Diagonal"]),
        "confidence": round(random.uniform(0.6, 0.98), 2)
    }

def generate_mock_coevolution_contacts(sequence_length, num_contacts_factor=0.02):
    num_contacts = int(sequence_length * num_contacts_factor * random.uniform(0.5, 1.5))
    contacts = []
    if sequence_length < 5: return pd.DataFrame()
    for _ in range(num_contacts):
        res1, res2 = sorted(random.sample(range(1, sequence_length + 1), 2))
        contacts.append({
            "Residue_1": res1,
            "Residue_2": res2,
            "Coevolution_Score": round(random.uniform(0.3, 0.95), 3),
            "Distance_Prediction_Mock_Angstrom": round(random.uniform(4.0, 15.0), 1)
        })
    return pd.DataFrame(contacts).sort_values(by="Coevolution_Score", ascending=False)

def generate_mock_structural_waters(num_waters_factor=0.1):
    num_waters = int(random.uniform(5, 20) * num_waters_factor) # Simplified
    waters = []
    for i in range(num_waters):
        waters.append({
            "Water_ID": f"HOH_{i+1}",
            "X_Coord_Mock": round(random.uniform(-20, 20), 2),
            "Y_Coord_Mock": round(random.uniform(-20, 20), 2),
            "Z_Coord_Mock": round(random.uniform(-20, 20), 2),
            "B_Factor_Mock": round(random.uniform(10, 60), 1),
            "Occupancy_Mock": round(random.uniform(0.8, 1.0), 2),
            "Bridging_Residues_Mock": f"R{random.randint(1,50)}-D{random.randint(51,100)}" if random.random() > 0.5 else "None"
        })
    return pd.DataFrame(waters)

def generate_mock_ligand_pockets(sequence_length, num_pockets=3):
    pockets = []
    for i in range(num_pockets):
        start_res = random.randint(1, sequence_length - 20)
        pocket_residues = sorted(random.sample(range(start_res, min(start_res + 30, sequence_length)), random.randint(5,15)))
        pockets.append({
            "pocket_id": f"Pocket_{i+1}",
            "residues": ", ".join(map(str, pocket_residues)),
            "volume_A3": round(random.uniform(100, 1000), 1),
            "druggability_score": round(random.uniform(0.1, 0.95), 2),
            "target_ligand_type": random.choice(["Inhibitor", "Activator", "Cofactor", "Substrate"])
        })
    return pockets

def generate_mock_surface_properties(sequence_length):
    properties = []
    for i in range(sequence_length):
        properties.append({
            "residue_index": i + 1,
            "hydrophobicity_kyte_doolittle": round(random.uniform(-4.5, 4.5), 2), # Kyte-Doolittle scale
            "electrostatic_potential_mock": round(random.uniform(-5, 5), 2), # Mock potential
            "solvent_accessibility_mock_percent": round(random.uniform(0, 100), 1)
        })
    return pd.DataFrame(properties)

def generate_mock_structural_comparison(num_hits=5):
    hits = []
    for i in range(num_hits):
        hits.append({
            "PDB_ID": f"{random.choice(string.digits)}{random.choice(string.ascii_uppercase)}{random.choice(string.ascii_uppercase)}{random.choice(string.ascii_uppercase)}",
            "Chain": random.choice(["A", "B", ""]),
            "RMSD_Angstrom": round(random.uniform(0.5, 5.0), 2),
            "Sequence_Identity_Percent": round(random.uniform(20, 99), 1),
            "Alignment_Score": random.randint(50, 500),
            "Description": random.choice(["Kinase domain", "Receptor binding domain", "Hypothetical protein", "Enzyme active site"])
        })
    return pd.DataFrame(hits).sort_values(by="RMSD_Angstrom")

def generate_mock_quality_assessment(sequence_length):
    ramachandran_favored = round(random.uniform(85, 98), 1)
    ramachandran_allowed = round(random.uniform(1, 15 - (ramachandran_favored-85)), 1)
    ramachandran_outlier = round(100 - ramachandran_favored - ramachandran_allowed, 1)

    return {
        "ramachandran_favored_percent": ramachandran_favored,
        "ramachandran_allowed_percent": ramachandran_allowed,
        "ramachandran_outliers_percent": ramachandran_outlier,
        "clashscore": round(random.uniform(0, 20), 2), # Lower is better
        "avg_bond_length_deviation_percent": round(random.uniform(0.1, 2.0), 2),
        "avg_bond_angle_deviation_degrees": round(random.uniform(0.5, 5.0), 1),
        "overall_gdt_ts_mock": round(random.uniform(50, 95), 1) # Global Distance Test
    }

def generate_mock_allosteric_sites(sequence_length, num_sites=2):
    sites = []
    possible_site_types = ["Activator Binding", "Inhibitor Binding", "Modulatory Interface", "Cryptic Pocket"]
    for i in range(num_sites):
        start_res = random.randint(1, sequence_length - 15)
        site_residues_indices = sorted(random.sample(range(start_res, min(start_res + 25, sequence_length)), random.randint(4,10)))
        sites.append({
            "site_id": f"AlloSite_{i+1}",
            "residues": ", ".join(map(str, site_residues_indices)),
            "prediction_score": round(random.uniform(0.3, 0.9), 2), # Higher is more likely
            "pocket_volume_A3_mock": round(random.uniform(50, 500), 1),
            "site_type_mock": random.choice(possible_site_types),
            "avg_conservation_mock": round(random.uniform(0.2, 0.95), 2) # Mock conservation score for the site
        })
    return sites

def generate_mock_pore_profile(channel_length_residues=50):
    # Simulate a pore along Z-axis, length in Angstroms
    z_coords = np.linspace(0, channel_length_residues * 1.5, 50) # Approx 1.5A per residue length in helix
    # Simulate a narrowing and widening pore
    radius = 5 * np.sin(z_coords / (channel_length_residues*1.5/np.pi) * 2) + \
             2 * np.cos(z_coords / (channel_length_residues*1.5/np.pi) * 5) + \
             random.uniform(1.5, 4) # Base radius
    radius = np.clip(radius, 0.5, 10) # Min/max radius
    return pd.DataFrame({"Position_Angstrom": np.round(z_coords,1), "Radius_Angstrom": np.round(radius,1)})

def generate_mock_protein_symmetry_data():
    symmetry_type = random.choice(["None", "C2", "C3", "C4", "D2", "D3", "Icosahedral (mock)"])
    if symmetry_type == "None":
        return {"type": "None", "axis": "N/A", "confidence": 0.0}
    return {
        "type": symmetry_type,
        "axis": random.choice(["X-axis", "Y-axis", "Z-axis", "Diagonal"]),
        "confidence": round(random.uniform(0.6, 0.98), 2)
    }

def generate_mock_coevolution_contacts(sequence_length, num_contacts_factor=0.02):
    num_contacts = int(sequence_length * num_contacts_factor * random.uniform(0.5, 1.5))
    contacts = []
    if sequence_length < 5: return pd.DataFrame()
    for _ in range(num_contacts):
        res1, res2 = sorted(random.sample(range(1, sequence_length + 1), 2))
        contacts.append({
            "Residue_1": res1,
            "Residue_2": res2,
            "Coevolution_Score": round(random.uniform(0.3, 0.95), 3),
            "Distance_Prediction_Mock_Angstrom": round(random.uniform(4.0, 15.0), 1)
        })
    return pd.DataFrame(contacts).sort_values(by="Coevolution_Score", ascending=False)

def generate_mock_structural_waters(num_waters_factor=0.1):
    num_waters = int(random.uniform(5, 20) * num_waters_factor) # Simplified
    waters = []
    for i in range(num_waters):
        waters.append({
            "Water_ID": f"HOH_{i+1}",
            "X_Coord_Mock": round(random.uniform(-20, 20), 2),
            "Y_Coord_Mock": round(random.uniform(-20, 20), 2),
            "Z_Coord_Mock": round(random.uniform(-20, 20), 2),
            "B_Factor_Mock": round(random.uniform(10, 60), 1),
            "Occupancy_Mock": round(random.uniform(0.8, 1.0), 2),
            "Bridging_Residues_Mock": f"R{random.randint(1,50)}-D{random.randint(51,100)}" if random.random() > 0.5 else "None"
        })
    return pd.DataFrame(waters)

def generate_mock_pore_profile(channel_length_residues=50):
    # Simulate a pore along Z-axis, length in Angstroms
    z_coords = np.linspace(0, channel_length_residues * 1.5, 50) # Approx 1.5A per residue length in helix
    # Simulate a narrowing and widening pore
    radius = 5 * np.sin(z_coords / (channel_length_residues*1.5/np.pi) * 2) + \
             2 * np.cos(z_coords / (channel_length_residues*1.5/np.pi) * 5) + \
             random.uniform(1.5, 4) # Base radius
    radius = np.clip(radius, 0.5, 10) # Min/max radius
    return pd.DataFrame({"Position_Angstrom": np.round(z_coords,1), "Radius_Angstrom": np.round(radius,1)})

def generate_mock_surface_curvature(sequence_length):
    # Simplified: assign curvature type per residue
    curvature_types = ["Convex", "Concave", "Saddle", "Flat"]
    curvatures = random.choices(curvature_types, weights=[0.4, 0.3, 0.15, 0.15], k=sequence_length)
    return pd.DataFrame({"Residue_Index": range(1, sequence_length + 1), "Curvature_Type_Pred": curvatures})

def generate_mock_packing_geometry(num_elements=5): # e.g., 5 helices/sheets
    packing = []
    elements = [f"{random.choice(['Helix', 'Sheet'])}_{i+1}" for i in range(num_elements)]
    if num_elements < 2: return pd.DataFrame()
    for i in range(num_elements):
        for j in range(i + 1, num_elements):
            packing.append({
                "Element_1": elements[i],
                "Element_2": elements[j],
                "Packing_Angle_Degrees_Mock": round(random.uniform(-90, 90), 1),
                "Closest_Distance_Angstrom_Mock": round(random.uniform(5, 15), 1)
            })
    return pd.DataFrame(packing)

def generate_mock_fold_recognition(num_hits=3):
    folds = ["Rossmann fold", "TIM barrel", "Beta-propeller", "Jelly roll", "Globin fold", "Alpha-alpha superhelix"]
    hits = []
    for i in range(num_hits):
        hits.append({
            "Fold_Database_ID_Mock": f"{random.choice(['CATH', 'SCOP'])}_{random.randint(1000,9999)}",
            "Fold_Name": random.choice(folds),
            "Z_Score_Mock": round(random.uniform(3.0, 15.0), 2),
            "Sequence_Identity_to_Exemplar_Percent_Mock": round(random.uniform(10, 40),1)
        })
    return pd.DataFrame(hits).sort_values(by="Z_Score_Mock", ascending=False)

def generate_mock_cryoem_fit():
    return {
        "Resolution_Angstrom_Mock": round(random.uniform(2.5, 6.0), 1),
        "Cross_Correlation_Score_Mock": round(random.uniform(0.5, 0.85), 3),
        "Map_Segmentation_Quality_Mock": random.choice(["Good", "Moderate", "Poor"])
    }

def generate_mock_saxs_profile():
    q_values = np.logspace(-2, 0, 100) # q range for SAXS
    rg_mock = random.uniform(15, 50) # Mock Radius of Gyration
    i_q = np.exp(-(q_values**2 * rg_mock**2) / 3) * random.uniform(1e3, 1e5) + np.random.normal(0, 0.05 * 1e4, 100) # Guinier approximation + noise
    i_q = np.maximum(i_q, 1) # Ensure positive intensity
    return pd.DataFrame({"q_Angstrom_inv": q_values, "Intensity_I_q_arbitrary_units": i_q}), rg_mock

def generate_mock_crystallization_propensity():
    # Based on Surface Entropy Reduction concepts, etc.
    return {
        "Overall_Propensity_Score_Mock": round(random.uniform(0.1, 0.9), 2), # Higher is better
        "Number_of_Low_Entropy_Patches_Mock": random.randint(0, 5),
        "Largest_Hydrophobic_Patch_Area_A2_Mock": round(random.uniform(100, 800),1)
    }

def generate_mock_rotamer_analysis(sequence_length):
    favored = random.uniform(0.85, 0.98)
    allowed = random.uniform(0.01, 0.15 - (favored - 0.85))
    outlier = 1.0 - favored - allowed
    return {
        "Favored_Rotamers_Percent": round(favored * 100, 1),
        "Allowed_Rotamers_Percent": round(allowed * 100, 1),
        "Outlier_Rotamers_Percent": round(outlier * 100, 1)
    }

def generate_mock_membrane_topology(sequence_length):
    is_membrane_protein = random.random() < 0.3 # 30% chance of being a membrane protein
    if not is_membrane_protein or sequence_length < 60:
        return {"is_membrane_protein": False, "helices": [], "topology_summary": "Predicted as globular protein."}

    num_helices = random.randint(1, min(7, sequence_length // 25))
    helices = []
    current_pos = 1
    for i in range(num_helices):
        if current_pos + 40 > sequence_length: break # Not enough space for more helices
        start = random.randint(current_pos, current_pos + 15)
        length = random.randint(18, 25)
        end = min(start + length -1, sequence_length)
        if end > sequence_length: break
        helices.append({"id": f"TMH{i+1}", "start": start, "end": end, "length": end - start + 1})
        current_pos = end + random.randint(5, 20)

    if not helices:
        return {"is_membrane_protein": False, "helices": [], "num_helices": 0, "n_terminus_location": "Unknown", "c_terminus_location": "Unknown", "topology_summary": "Predicted as globular protein (no clear TMHs found)."}

    n_term_location = random.choice(["Inside", "Outside"])
    c_term_location = n_term_location if num_helices % 2 == 0 else ("Outside" if n_term_location == "Inside" else "Inside")
    
    return {
        "is_membrane_protein": True,
        "helices": helices,
        "num_helices": len(helices),
        "n_terminus_location": n_term_location,
        "c_terminus_location": c_term_location,
        "topology_summary": f"Predicted membrane protein with {len(helices)} TMHs. N-terminus: {n_term_location}, C-terminus: {c_term_location}."
    }

def generate_mock_folding_pathway_insights(sequence_length):
    insights = [
        f"An early folding nucleus is predicted around residues {random.randint(10, sequence_length//3)}-{random.randint(sequence_length//3 + 1, sequence_length//2)}.",
        "Long-range interactions between N-terminal and C-terminal domains appear crucial for final fold acquisition.",
        f"A potential misfolding trap involving residues in the loop region {random.randint(sequence_length//2, sequence_length - 30)}-{random.randint(sequence_length//2+10, sequence_length-10)} might slow down folding.",
        "The formation of secondary structures (helices and sheets) is likely rapid, followed by slower tertiary packing.",
        "Chaperone assistance might be beneficial for efficient folding of larger domains.",
        "Overall folding is predicted to be cooperative with few stable intermediates."
    ]
    return random.sample(insights, k=random.randint(2,4))

def generate_mock_ppi_interface_data(sequence_length, partner_protein_id="PartnerX"):
    num_interface_residues = random.randint(5, 20)
    interface_residues = sorted(random.sample(range(1, sequence_length + 1), num_interface_residues))
    return {
        "partner_protein_id": partner_protein_id,
        "interface_residues": ", ".join(map(str, interface_residues)),
        "buried_surface_area_A2_mock": round(random.uniform(600, 2000), 1),
        "interface_hydrophobicity_score_mock": round(random.uniform(-1.5, 1.5), 2),
        "predicted_binding_energy_kcal_mol_mock": round(random.uniform(-5, -15), 1)
    }

def generate_mock_ramachandran_data(sequence_length):
    # Simulate phi and psi angles
    # Favoring allowed regions: alpha-helix, beta-sheet
    phi_psi_pairs = []
    for _ in range(sequence_length):
        region = random.choices(["alpha_L", "beta", "alpha_R", "disallowed"], weights=[0.4, 0.3, 0.1, 0.2])[0]
        if region == "alpha_L": # Left-handed alpha helix (less common but for variety)
            phi = random.uniform(40, 90)
            psi = random.uniform(0, 90)
        elif region == "beta": # Beta sheet
            phi = random.uniform(-180, -40)
            psi = random.uniform(90, 180) if random.random() > 0.5 else random.uniform(-180, -150)
        elif region == "alpha_R": # Right-handed alpha helix
            phi = random.uniform(-150, -40)
            psi = random.uniform(-70, 0)
        else: # Disallowed / generously allowed
            phi = random.uniform(-180, 180)
            psi = random.uniform(-180, 180)
        phi_psi_pairs.append({"phi": round(phi,1), "psi": round(psi,1)})
    return pd.DataFrame(phi_psi_pairs)

def generate_mock_nmr_spectra_data(sequence_length):
    # Simplified mock 1D Proton NMR-like spectrum
    ppm_range = np.linspace(0, 10, 500)
    intensity = np.zeros_like(ppm_range)
    num_peaks = sequence_length // 10 + random.randint(-5, 5)
    num_peaks = max(5, num_peaks) # Ensure at least a few peaks
    for _ in range(num_peaks):
        peak_pos = random.uniform(0.5, 9.5)
        intensity += np.exp(-((ppm_range - peak_pos)**2) / (2 * (random.uniform(0.01, 0.05))**2)) * random.uniform(0.1, 1)
    return pd.DataFrame({"Chemical_Shift_ppm": ppm_range, "Intensity_Arbitrary": intensity * 100})

def generate_mock_functional_prediction_data(sequence_length, domains_data):
    go_terms = {
        "Molecular Function": [f"GO:000{random.randint(1000,9999)} - {random.choice(['ATP binding', 'DNA binding', 'protein kinase activity', 'receptor activity', 'catalytic activity', 'transporter activity'])} (Score: {random.uniform(0.6,0.95):.2f})"],
        "Biological Process": [f"GO:000{random.randint(1000,9999)} - {random.choice(['metabolic process', 'signal transduction', 'cell cycle', 'transcription', 'translation', 'immune response'])} (Score: {random.uniform(0.5,0.9):.2f})"],
        "Cellular Component": [f"GO:000{random.randint(1000,9999)} - {random.choice(['nucleus', 'cytoplasm', 'mitochondrion', 'plasma membrane', 'ribosome', 'endoplasmic reticulum'])} (Score: {random.uniform(0.4,0.85):.2f})"]
    }
    if random.random() > 0.3: # Add a second term sometimes
        go_terms["Molecular Function"].append(f"GO:000{random.randint(1000,9999)} - {random.choice(['ion binding', 'structural molecule activity', 'enzyme regulator activity'])} (Score: {random.uniform(0.5,0.8):.2f})")

    ec_number = "N/A"
    if any(d['type'] == 'Enzymatic' for d in domains_data) or random.random() < 0.2: # If enzymatic domain or 20% chance
        ec_number = f"{random.randint(1,6)}.{random.randint(1,20)}.{random.randint(1,20)}.{random.randint(1,100)}"

    return {"go_terms": go_terms, "ec_number": ec_number, "predicted_pathways_mock": [random.choice(["Glycolysis", "Citric Acid Cycle", "MAPK signaling", "Apoptosis", "DNA Repair"]) for _ in range(random.randint(0,2))]}

def generate_mock_contact_map_data(sequence_length):
    contacts = np.random.rand(sequence_length, sequence_length) < 0.05 # 5% chance of contact
    # Make it symmetric and remove self-contacts
    contacts = np.triu(contacts, k=1) 
    contacts = contacts + contacts.T
    # Add some local contacts (common in helices/sheets)
    for i in range(sequence_length - 4):
        if random.random() < 0.3: contacts[i, i+3] = contacts[i+3, i] = 1 # i, i+3
        if random.random() < 0.2: contacts[i, i+4] = contacts[i+4, i] = 1 # i, i+4
    return contacts

def generate_mock_sasa_data(sequence_length):
    # Simulate SASA values, often higher for loops/turns, lower for core residues
    sasa = np.random.normal(loc=60, scale=40, size=sequence_length)
    # Add some periodic variation (e.g. exposed every few residues in a helix)
    sasa += 20 * np.sin(np.arange(sequence_length) * np.pi / 3.5)
    sasa = np.clip(sasa, 5, 200) # Realistic SASA range in Å^2
    return sasa

AA_HYDROPHOBICITY_KD = { # Kyte-Doolittle
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
    'X': 0.0 # Placeholder for unknown/gap
}

def generate_mock_b_factors_data(sequence_length):
    # Simulate B-factors, often higher for loops/flexible regions
    b_factors = np.abs(np.random.normal(loc=30, scale=15, size=sequence_length))
    # Add some higher values for potential loop regions
    for _ in range(sequence_length // 20): # Add a few flexible regions
        start = random.randint(0, sequence_length - 5)
        b_factors[start:start+random.randint(3,10)] *= random.uniform(1.5, 2.5)
    return np.clip(b_factors, 5, 150)

def generate_mock_aggregation_propensity_data(sequence_length):
    # Simulate aggregation propensity (e.g., on a 0-1 scale)
    propensity = np.random.beta(a=2, b=8, size=sequence_length) # Mostly low propensity
    # Add a few high propensity patches
    for _ in range(random.randint(0, sequence_length // 50)):
        start = random.randint(0, sequence_length - 10)
        propensity[start:start+random.randint(5,10)] = np.random.beta(a=8, b=2, size=len(propensity[start:start+random.randint(5,10)]))
    return np.clip(propensity, 0, 1)

def create_structure_plot(prediction_data):
    """Create interactive structure visualization."""
    seq_len = prediction_data['length']
    positions = list(range(1, seq_len + 1))
    
    # Create subplots
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=['Secondary Structure', 'Confidence (pLDDT)', 'Disorder Prediction', 'Domain Architecture'],
        vertical_spacing=0.08,
        row_heights=[0.3, 0.25, 0.2, 0.25]
    )
    
    # Secondary structure plot
    ss_colors = {'Helix': '#FF6B6B', 'Sheet': '#4ECDC4', 'Coil': '#45B7D1', 'Turn': '#FFA07A'}
    for ss_type in SECONDARY_STRUCTURES:
        mask = prediction_data['secondary_structure'] == ss_type
        if np.any(mask):
            fig.add_trace(
                go.Scatter(
                    x=np.array(positions)[mask],
                    y=[ss_type] * np.sum(mask),
                    mode='markers',
                    marker=dict(color=ss_colors[ss_type], size=8),
                    name=ss_type,
                    showlegend=True
                ),
                row=1, col=1
            )
    
    # Confidence plot
    fig.add_trace(
        go.Scatter(
            x=positions,
            y=prediction_data['plddt'],
            mode='lines',
            line=dict(color='#2E86AB', width=2),
            name='pLDDT Score',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Add confidence threshold lines
    fig.add_hline(y=90, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="orange", row=2, col=1)
    fig.add_hline(y=50, line_dash="dash", line_color="red", row=2, col=1)
    
    # Disorder prediction
    disorder_y = prediction_data['disorder'].astype(int)
    fig.add_trace(
        go.Scatter(
            x=positions,
            y=disorder_y,
            mode='lines',
            line=dict(color='#E74C3C', width=2),
            fill='tonexty',
            name='Disorder',
            showlegend=False
        ),
        row=3, col=1
    )
    
    # Domain architecture
    domain_colors = ['#3498DB', '#E67E22', '#2ECC71', '#9B59B6', '#F39C12']
    for i, domain in enumerate(prediction_data['domains']):
        fig.add_trace(
            go.Scatter(
                x=[domain['start'], domain['end']],
                y=[1, 1],
                mode='lines',
                line=dict(color=domain_colors[i % len(domain_colors)], width=15),
                name=domain['name'],
                showlegend=False
            ),
            row=4, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=800,
        title="Protein Structure Analysis",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Update axes
    fig.update_xaxes(title_text="Residue Position", row=4, col=1)
    fig.update_yaxes(title_text="Structure", row=1, col=1)
    fig.update_yaxes(title_text="Confidence", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="Disorder", range=[0, 1.2], row=3, col=1)
    fig.update_yaxes(title_text="Domains", range=[0, 2], row=4, col=1)
    
    return fig

def create_confidence_distribution(prediction_data):
    """Create confidence distribution chart."""
    plddt_scores = prediction_data['plddt']
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=plddt_scores,
        nbinsx=20,
        marker_color='#3498DB',
        opacity=0.7,
        name='pLDDT Distribution'
    ))
    
    fig.update_layout(
        title="Confidence Score Distribution",
        xaxis_title="pLDDT Score",
        yaxis_title="Number of Residues",
        bargap=0.1
    )
    
    return fig

def export_results(prediction_data, format_type):
    """Export prediction results in various formats."""
    if format_type == "JSON":
        # Convert numpy arrays to lists for JSON serialization
        export_data = prediction_data.copy()
        export_data['secondary_structure'] = export_data['secondary_structure'].tolist()
        export_data['confidence'] = export_data['confidence'].tolist()
        export_data['disorder'] = export_data['disorder'].tolist()
        export_data['plddt'] = export_data['plddt'].tolist()
        export_data['timestamp'] = export_data['timestamp'].isoformat()
        
        return json.dumps(export_data, indent=2)
    
    elif format_type == "CSV":
        df = pd.DataFrame({
            'Position': range(1, len(prediction_data['sequence']) + 1),
            'Residue': list(prediction_data['sequence']),
            'Secondary_Structure': prediction_data['secondary_structure'],
            'Confidence': prediction_data['confidence'],
            'pLDDT': prediction_data['plddt'],
            'Disorder': prediction_data['disorder']
        })
        return df.to_csv(index=False)
    
    elif format_type == "PDB":
        # Mock PDB format (simplified)
        pdb_lines = [
            "HEADER    PROTEIN STRUCTURE PREDICTION",
            f"TITLE     ALPHAFOLD PREDICTION",
            "REMARK   THIS IS A MOCK PDB FILE FOR DEMONSTRATION"
        ]
        
        for i, aa in enumerate(prediction_data['sequence']):
            line = f"ATOM  {i+1:5d}  CA  {aa} A{i+1:4d}    {i*3.8:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00{prediction_data['plddt'][i]:6.2f}           C"
            pdb_lines.append(line)
        
        pdb_lines.append("END")
        return '\n'.join(pdb_lines)

# Main UI
st.markdown("""
<div class="main-header">
    <h1>🧬 AlphaFold Pro</h1>
    <p>Advanced Protein Structure Prediction Suite</p>
    <p style="font-size: 0.9em; opacity: 0.8;">Powered by Google Gemini AI | Professional-Grade Analysis</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🔬 Analysis Configuration")
    
    # Model selection
    st.subheader("AI Model Selection")
    AVAILABLE_MODELS = {
        
        "Gemini 2.0 Flash": "gemini-2.0-flash",
        
    }
    
    selected_model = st.selectbox(
        "Choose AI Model:",
        options=list(AVAILABLE_MODELS.keys()),
        help="Select the AI model for structure analysis"
    )
    
    # API Configuration
    st.subheader("API Configuration")
    api_key = st.text_input(
        "Gemini API Key:",
        type="password",
        help="Enter your Google AI Studio API key"
    )
    
    # Sequence Input
    st.subheader("Sequence Input")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎲 Random Protein", help="Generate random sequence"):
            st.session_state.sequence_input = generate_protein_sequence()
    
    with col2:
        if st.button("🧪 Complex Protein", help="Generate complex sequence"):
            st.session_state.sequence_input = generate_protein_sequence(complexity="high")
    
    sequence_input = st.text_area(
        "Protein Sequence (FASTA or raw):",
        height=200,
        placeholder=">MyProtein\nMQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
        key="sequence_input"
    )
    
    # Analysis Options
    st.subheader("Analysis Options")
    include_disorder = st.checkbox("Disorder Prediction", value=True)
    include_domains = st.checkbox("Domain Analysis", value=True)
    confidence_threshold = st.slider("Confidence Threshold", 0, 100, 70)
    
    # Prediction Button
    predict_button = st.button(
        "🚀 Predict Structure",
        type="primary",
        help="Start structure prediction analysis"
    )
    
    st.divider()
    
    # Export Options
    if st.session_state.current_prediction:
        st.subheader("📊 Export Results")
        export_format = st.selectbox(
            "Export Format:",
            ["JSON", "CSV", "PDB"],
            help="Choose export format"
        )
        
        if st.button("💾 Download Results"):
            result = export_results(st.session_state.current_prediction, export_format)
            st.download_button(
                f"Download {export_format}",
                result,
                file_name=f"prediction_results.{export_format.lower()}",
                mime="text/plain"
            )

# Main content area
if predict_button:
    if not sequence_input.strip():
        st.error("❌ Please enter a protein sequence")
    elif not api_key:
        st.error("❌ Please enter your API key")
    else:
        # Validate sequence
        is_valid, result = validate_protein_sequence(sequence_input)
        
        if not is_valid:
            st.error(f"❌ Sequence validation failed: {result}")
        else:
            sequence = result
            
            # Create prediction hash for caching
            seq_hash = hashlib.md5(sequence.encode()).hexdigest()
            
            # Show progress
            progress_container = st.container()
            with progress_container:
                st.markdown('<div class="prediction-status status-running">🔄 Running Structure Prediction...</div>', 
                           unsafe_allow_html=True)
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Simulate realistic prediction process
                steps = [
                    "Preprocessing sequence...",
                    "Running MSA search...",
                    "Generating structural features...",
                    "Predicting secondary structure...",
                    "Calculating confidence scores...",
                    "Analyzing domains...",
                    "Finalizing predictions..."
                ]
                
                for i, step in enumerate(steps):
                    status_text.text(step)
                    progress_bar.progress((i + 1) / len(steps))
                    time.sleep(0.5)
                
                # Generate predictions
                mock_data = generate_mock_predictions(sequence, selected_model)
                
                # AI Analysis
                try:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel(AVAILABLE_MODELS[selected_model])
                    
                    prompt = f"""
                    As an expert structural biologist, analyze this protein sequence and provide a comprehensive analysis:
                    
                    Sequence Length: {len(sequence)} residues
                    Sequence: {sequence[:100]}{'...' if len(sequence) > 100 else ''}
                    
                    Please provide:
                    1. **Structural Classification**: Predict the overall fold class and architecture
                    2. **Secondary Structure Analysis**: Detailed prediction of helices, sheets, and loops
                    3. **Functional Domains**: Identify potential functional regions and motifs
                    4. **Stability Assessment**: Comment on predicted structural stability
                    5. **Functional Predictions**: Potential biological function based on structure
                    6. **Key Structural Features**: Notable characteristics and critical residues
                    7. **Confidence Assessment**: Areas of high/low prediction confidence
                    
                    Format as a detailed scientific report with specific residue ranges where applicable.
                    """
                    
                    response = model.generate_content(prompt)
                    ai_analysis = response.text
                    mock_data['ai_analysis'] = ai_analysis
                    
                except Exception as e:
                    mock_data['ai_analysis'] = f"AI Analysis Error: {str(e)}"
                
                st.session_state.current_prediction = mock_data
                st.session_state.prediction_history.append(mock_data)
                
                progress_container.empty()

# Display Results
if st.session_state.current_prediction:
    data = st.session_state.current_prediction
    
    st.markdown('<div class="prediction-status status-complete">✅ Structure Prediction Complete!</div>', 
               unsafe_allow_html=True)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Sequence Length</h4>
            <h2>{data['length']} AA</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        confidence_class = "high" if data['overall_confidence'] > 70 else "medium" if data['overall_confidence'] > 50 else "low"
        st.markdown(f"""
        <div class="metric-card">
            <h4>Overall Confidence</h4>
            <h2 class="confidence-{confidence_class}">{data['overall_confidence']:.1f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Domains Found</h4>
            <h2>{len(data['domains'])}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        disorder_pct = np.mean(data['disorder']) * 100
        st.markdown(f"""
        <div class="metric-card">
            <h4>Disorder Regions</h4>
            <h2>{disorder_pct:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Tabs for different analyses
    TAB_CODES = {
        "STRUCT": "📊 Structure Overview",
        "AI": "🎯 AI Analysis",
        "CONF": "📈 Confidence Analysis",
        "DOMAIN": "🧬 Domain Architecture",
        "INT": "🔗 Interaction Network",      # Shrunk from INTERACT
        "MUT": "🔬 Mutational Analysis",       # Shrunk from MUTATE
        "DYN": "⚙️ Molecular Dynamics",       # Shrunk from DYNAMICS
        "LIG": "💊 Ligand Binding",           # Shrunk from LIGAND
        "EVO": "🌳 Evolutionary Trace",       # Shrunk from EVOLVE
        "SURF": "🌐 Surface Properties",      # Shrunk from SURFACE
        "COMP": "🔄 Structural Comparison",   # Shrunk from COMPARE
        "QUAL": "🏅 Quality Assessment",      # Shrunk from QUALITY
        "ALLO": "🔮 Allosteric Sites",
        "MEMB": "🧱 Membrane Topology",
        "FOLD": "⏳ Folding Pathway",
        "PPI_IF": "🤝 PPI Interface", # Renamed from PPI to PPI_IF for clarity if needed, or keep as PPI
        "DRUG": "🎯 Druggability Analysis", # Shrunk from DRUGGABLE
        "CONS": "🛡️ Conservation Score",   # Shrunk from CONSERVE
        # New tabs inserted here
        "NMR": "📡 NMR Spectra",
        "SAX": "✨ SAXS Profile",
        "CRF": "🧊 Cryo-EM Fit",
        "XTP": "❄️ Crystallization",
        "FUNC": "🔬 Functional Prediction",
        "DATA": "📋 Detailed Data"
    }
    
    tab_keys = list(TAB_CODES.keys())
    # Unpack all tabs based on the order in TAB_CODES
    tabs_list = st.tabs(tab_keys)
    
    # Assign tabs to meaningful variables based on their codes
    tab_map = {code: tab for code, tab in zip(tab_keys, tabs_list)}

    # Display tab legend in sidebar
    with st.sidebar:
        st.subheader("Tab Legend")
        for code, full_name in TAB_CODES.items():
            st.markdown(f"- **{code}**: {full_name}")
        st.divider() # Add a divider after the legend if other sidebar items follow

    
    with tab_map["STRUCT"]:
        st.subheader("Structural Prediction Visualization")
        fig = create_structure_plot(data)
        st.plotly_chart(fig, use_container_width=True)
        
        # Secondary structure summary
        st.subheader("Secondary Structure Summary")
        ss_counts = pd.Series(data['secondary_structure']).value_counts()
        col1, col2 = st.columns(2)
        
        with col1:
            fig_pie = px.pie(
                values=ss_counts.values, 
                names=ss_counts.index,
                title="Secondary Structure Distribution"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.write("**Structure Statistics:**")
            for ss, count in ss_counts.items():
                percentage = (count / len(data['secondary_structure'])) * 100
                st.write(f"• {ss}: {count} residues ({percentage:.1f}%)")
        
        st.subheader("Advanced Structural Analysis Tools")
        with st.expander("📊 Ramachandran Plot Analysis"):
            df_phi_psi = generate_mock_ramachandran_data(data['length'])
            fig_rama = px.scatter(df_phi_psi, x="phi", y="psi", 
                                  title="Mock Ramachandran Plot (Phi/Psi Angles)",
                                  labels={"phi": "Phi (degrees)", "psi": "Psi (degrees)"},
                                  marginal_x="histogram", marginal_y="histogram",
                                  color_discrete_sequence=['#636EFA'])
            fig_rama.update_xaxes(range=[-180, 180], zeroline=False)
            fig_rama.update_yaxes(range=[-180, 180], zeroline=False)
            fig_rama.add_shape(type="rect", x0=-150, y0=-70, x1=-40, y1=0, line=dict(color="rgba(0,255,0,0.2)"), fillcolor="rgba(0,255,0,0.05)", name="Alpha (R)") # Alpha R (approx)
            fig_rama.add_shape(type="rect", x0=-180, y0=90, x1=-40, y1=180, line=dict(color="rgba(255,255,0,0.2)"), fillcolor="rgba(255,255,0,0.05)", name="Beta (approx)")    # Beta (approx)
            st.plotly_chart(fig_rama, use_container_width=True)
            st.caption("Distribution of backbone dihedral angles. Allowed regions (e.g., alpha-helical, beta-sheet) are highlighted illustratively. Outliers may indicate strained conformations.")

        with st.expander("🗺️ Residue Contact Map"):
            contact_map_data = generate_mock_contact_map_data(data['length'])
            fig_contact = px.imshow(contact_map_data, 
                                    title="Mock Residue Contact Map (Proximity < 8Å)",
                                    labels=dict(x="Residue Index", y="Residue Index", color="Contact"),
                                    color_continuous_scale="Greys")
            st.plotly_chart(fig_contact, use_container_width=True)
            st.caption("Predicted contacts between residue pairs (<8Å). Darker points indicate proximity. Patterns can reveal structural elements.")

        with st.expander("💧 Solvent Accessible Surface Area (SASA)"):
            sasa_data = generate_mock_sasa_data(data['length'])
            df_sasa = pd.DataFrame({'Residue Index': range(1, data['length'] + 1), 'SASA (Å²)': sasa_data})
            fig_sasa_plot = px.line(df_sasa, x='Residue Index', y='SASA (Å²)', 
                                    title="Mock Per-Residue Solvent Accessible Surface Area (SASA)",
                                    labels={'SASA (Å²)': 'SASA (Å²)'})
            st.plotly_chart(fig_sasa_plot, use_container_width=True)
            st.metric(label="Total SASA (Mock)", value=f"{np.sum(sasa_data):.1f} Å²")
            st.markdown("SASA indicates the surface area of each residue exposed to solvent. Higher values mean more exposure. This is relevant for identifying surface loops, binding sites, and core residues.")

        with st.expander("📏 Radius of Gyration (Rg) Analysis"):
            st.info("Placeholder for calculating and plotting the Radius of Gyration. Provides a measure of the protein's compactness.")
            # Mock Rg value
            mock_rg = round(0.8 * (data['length']**0.38), 2) # Approximation
            st.metric("Predicted Radius of Gyration (Mock)", f"{mock_rg} Å")
            st.markdown("Radius of Gyration (Rg) is a measure of the protein's overall compactness. Larger Rg values suggest a more extended conformation.")

        with st.expander("🔗 Hydrogen Bond Network"):
            st.info("Placeholder for identifying and visualizing the hydrogen bond network within the protein structure. Critical for stability.")
            mock_hbonds = random.randint(data['length']//2, data['length'] * 2)
            st.metric("Predicted Hydrogen Bonds (Mock)", mock_hbonds)
            st.markdown("Hydrogen bonds are crucial for stabilizing protein structure, particularly secondary structures like helices and sheets.")

        with st.expander("🌉 Salt Bridge Analysis"):
            st.info("Placeholder for detecting and listing potential salt bridges. Important for protein stability and interactions.")
            mock_salt_bridges = random.randint(max(0,data['length']//50 -1), data['length']//20 + 1)
            st.metric("Predicted Salt Bridges (Mock)", mock_salt_bridges)
            st.markdown("Salt bridges are electrostatic interactions between oppositely charged residues, contributing to protein stability and specific interactions.")

        with st.expander("🕳️ Surface Cavity and Pocket Detection"):
            st.info("Placeholder for identifying and characterizing cavities and pockets on the protein surface. Relevant for ligand binding and enzyme active sites.")
            num_pockets_surf = random.randint(1,5)
            st.metric("Predicted Surface Pockets/Cavities (Mock)", num_pockets_surf)
            st.markdown("Surface cavities and pockets are often sites for ligand binding, catalysis, or protein-protein interactions.")

        with st.expander("📐 Local Geometry Check (Bond Lengths/Angles)"):
            st.info("Placeholder for analyzing local structural geometry, such as bond lengths and angles, to identify strained or unusual conformations.")
            st.markdown("Average Bond Length Deviation (Mock): " + f"`{random.uniform(0.01, 0.05):.3f} Å`")
            st.markdown("Average Bond Angle Deviation (Mock): " + f"`{random.uniform(1.0, 3.0):.1f}°`")
            st.markdown("These metrics assess how well the local geometry conforms to idealized values. Large deviations can indicate strained regions.")

        with st.expander("🔄 Torsion Angle (Phi/Psi) Distribution"):
            st.info("Plots the distribution of Phi (Φ) and Psi (Ψ) backbone torsion angles. This complements the 2D Ramachandran plot by showing the individual 1D distributions of these critical angles, which define the protein backbone conformation.")
            df_phi_psi_dist = generate_mock_ramachandran_data(data['length']) # Re-use existing mock data generator

            col_phi, col_psi = st.columns(2)
            with col_phi:
                fig_phi_dist = px.histogram(df_phi_psi_dist, x="phi", nbins=30,
                                            title="Phi (Φ) Angle Distribution",
                                            labels={"phi": "Phi Angle (degrees)"},
                                            marginal="rug", color_discrete_sequence=['#0077b6'])
                fig_phi_dist.update_layout(bargap=0.1)
                st.plotly_chart(fig_phi_dist, use_container_width=True)
            with col_psi:
                fig_psi_dist = px.histogram(df_phi_psi_dist, x="psi", nbins=30,
                                            title="Psi (Ψ) Angle Distribution",
                                            labels={"psi": "Psi Angle (degrees)"},
                                            marginal="rug", color_discrete_sequence=['#fb8500'])
                fig_psi_dist.update_layout(bargap=0.1)
                st.plotly_chart(fig_psi_dist, use_container_width=True)
            st.markdown("Peaks in these distributions often correspond to common secondary structure elements (e.g., specific Phi/Psi ranges for alpha-helices or beta-sheets).")

        with st.expander("⚡ Intra-Protein Interaction Energy"):
            st.info("Placeholder for estimating non-bonded interaction energies (e.g., van der Waals, electrostatic) between different parts of the protein.")
            mock_energy = round(random.uniform(-500, -50) * (data['length']/100), 1)
            st.metric("Predicted Internal Energy (Mock)", f"{mock_energy} kcal/mol")
            st.markdown("A conceptual measure of the overall stability from internal non-bonded interactions. More negative values suggest greater stability.")

        with st.expander("🌡️ Protein B-Factor Analysis"):
            st.info("Visualizes and analyzes B-factors (temperature factors) to assess atomic displacement and flexibility. Higher B-factors indicate more mobile regions.")
            # Mock B-factor data
            seq_len_bfactor = data.get('length', 100)
            mock_b_factors_avg = round(random.uniform(15, 50), 1)
            st.metric(label="Average B-Factor (Mock)", value=f"{mock_b_factors_avg} Å²")
            st.markdown(f"Regions with B-Factors > {round(mock_b_factors_avg + 15,0)} Å² might represent flexible loops or termini.")

        with st.expander("🧬 Tertiary Structure Superposition"):
            st.info("Superimposes the predicted structure onto a reference structure (e.g., from PDB) and calculates Root Mean Square Deviation (RMSD) to quantify similarity.")
            mock_superposed_pdb = f"{random.choice(string.digits)}{random.choice(string.ascii_uppercase)}{random.choice(string.ascii_uppercase)}{random.choice(string.ascii_uppercase)}"
            mock_rmsd = round(random.uniform(0.5, 3.5), 2)
            st.metric(label=f"RMSD to {mock_superposed_pdb} (Mock)", value=f"{mock_rmsd} Å")
            st.markdown(f"A low RMSD (e.g., < 2.0 Å) indicates high structural similarity to the reference PDB ID {mock_superposed_pdb}.")

        with st.expander("🧩 Quaternary Structure Assembly Prediction"):
            st.info("Predicts how multiple protein subunits (chains) might assemble to form a functional complex, including symmetry and interface details.")
            mock_subunits = random.randint(1, 4)
            mock_symmetry = "None" if mock_subunits == 1 else random.choice(["C2", "C3", "D2"])
            st.metric(label="Predicted Subunits (Mock)", value=mock_subunits)
            st.markdown(f"**Predicted Symmetry (Mock):** {mock_symmetry}")
            st.markdown("This tool helps understand oligomeric states and protein complex formation.")

        with st.expander("💡 Electrostatic Potential Surface"):
            st.info("Calculates and visualizes the electrostatic potential on the protein's solvent-accessible surface, highlighting charged patches important for interactions.")
            st.markdown("**Key Electrostatic Features (Mock):**")
            st.markdown(f"- Predominantly {'Negative' if random.random() > 0.5 else 'Positive'} patch near residues {random.randint(10,20)}-{random.randint(25,35)}.")
            st.markdown(f"- Potential {random.choice(['DNA', 'RNA', 'ligand'])} binding site due to charge distribution.")

        with st.expander("🌊 Hydrophobicity Surface Map"):
            st.info("Maps hydrophobic and hydrophilic regions on the protein surface, which can indicate protein-protein interaction sites or membrane association regions.")
            st.markdown("**Key Hydrophobic Features (Mock):**")
            st.markdown(f"- Large hydrophobic patch identified around residues {random.randint(40,50)}-{random.randint(55,65)}, potentially involved in protein core or interface.")
            st.markdown(f"- Surface hydrophobicity suggests it's likely a {'soluble' if random.random() > 0.3 else 'membrane-associated'} protein.")

        with st.expander("⚖️ Predicted Stability (ΔG)"):
            st.info("Estimates the overall folding free energy (ΔG) of the protein structure. More negative values generally indicate higher stability.")
            mock_delta_g = round(random.uniform(-50, -5) * (data.get('length', 100)/100.0), 1)
            st.metric(label="Predicted Folding ΔG (Mock)", value=f"{mock_delta_g} kcal/mol")
            st.markdown("This value provides a theoretical measure of the protein's conformational stability.")

        with st.expander("🌀 Conformational Ensemble Generation"):
            st.info("Generates a representative ensemble of likely protein conformations, reflecting its dynamic nature, rather than a single static structure.")
            mock_conformers = random.randint(5, 20)
            mock_ensemble_rmsd = round(random.uniform(0.5, 2.5), 1)
            st.metric(label="Number of Conformers in Ensemble (Mock)", value=mock_conformers)
            st.markdown(f"**Ensemble RMSD Spread (Mock):** {mock_ensemble_rmsd} Å, indicating the diversity of conformations.")

        with st.expander("🎶 Normal Mode Analysis (NMA)"):
            st.info("Performs Normal Mode Analysis (NMA) to predict large-scale collective motions and functionally relevant flexibility of the protein.")
            st.markdown("**Dominant Motion Modes (Mock):**")
            st.markdown(f"- Mode 1: Hinge-bending motion between domains (if applicable, or N/C termini).")
            st.markdown(f"- Mode 2: Twisting motion along the main axis.")
            st.markdown("NMA helps understand how protein structure relates to its dynamic function.")

        with st.expander("⛓️ Disulfide Bond Prediction"):
            st.info("Identifies potential disulfide bonds based on cysteine residue proximity and geometry, which are important for protein stability, especially in extracellular proteins.")
            num_cysteines = data.get('sequence', "").count('C')
            mock_ss_bonds = random.randint(0, num_cysteines // 2)
            st.metric(label="Predicted Disulfide Bonds (Mock)", value=mock_ss_bonds)
            if mock_ss_bonds > 0:
                st.markdown(f"Potential S-S bond between Cys{random.randint(1,50)} and Cys{random.randint(51,100)} (example).")

        with st.expander("🏷️ Post-Translational Modification (PTM) Site Analysis"):
            st.info("Analyzes the structural context (e.g., accessibility, surrounding residues) of predicted Post-Translational Modification (PTM) sites like phosphorylation or ubiquitination.")
            mock_ptm_sites = random.randint(0, 5)
            st.metric(label="Predicted PTM Sites with Structural Context (Mock)", value=mock_ptm_sites)
            if mock_ptm_sites > 0:
                st.markdown(f"- Phosphorylation site at Ser{random.randint(1, data.get('length',100))} predicted to be surface exposed.")

        with st.expander("🧱 Aggregation Prone Region Prediction"):
            st.info("Identifies regions within the protein sequence and structure that are prone to aggregation, based on hydrophobicity, charge, and secondary structure propensity.")
            mock_agg_regions = random.randint(0, 3)
            st.metric(label="Predicted Aggregation Prone Regions (Mock)", value=mock_agg_regions)
            if mock_agg_regions > 0:
                st.markdown(f"- High aggregation propensity for residues {random.randint(20,30)}-{random.randint(31,40)}.")

        with st.expander("🖇️ Structural Alignment (Multiple Structures)"):
            st.info("Aligns multiple protein structures (if provided or found) to identify structurally conserved regions (SCRs) and variable regions (SVRs).")
            st.markdown("**Alignment Summary (Mock vs. 2 Hypothetical Homologs):**")
            st.markdown(f"- Core RMSD: {round(random.uniform(1.0, 2.5),1)} Å over {random.randint(data.get('length',100)//2, data.get('length',100)-10)} residues.")
            st.markdown(f"- Identified {random.randint(1,3)} major structurally conserved regions.")

        with st.expander("➰ Loop Region Modeling & Refinement"):
            st.info("Provides tools or insights for modeling and refining flexible loop regions, which are often critical for function but hard to predict accurately.")
            mock_loops_refined = random.randint(1, data.get('length',100)//20 +1)
            st.metric(label="Loops Modeled/Refined (Mock)", value=mock_loops_refined)
            st.markdown(f"Loop at residues {random.randint(10,20)}-{random.randint(21,30)} refined, improving local geometry score.")

        with st.expander("🎯 Active Site Characterization"):
            st.info("Performs detailed analysis of predicted active site(s), including geometry, key residues, volume, and electrostatic properties.")
            st.markdown("**Active Site Properties (Mock):**")
            st.markdown(f"- Key Catalytic Residues: His{random.randint(1,50)}, Asp{random.randint(51,100)}, Ser{random.randint(101,150)}")
            st.markdown(f"- Pocket Volume: {round(random.uniform(100,800),1)} Å³")

        with st.expander("💎 Metal Ion Coordination Site Prediction"):
            st.info("Identifies potential metal ion binding sites (e.g., for Zn, Mg, Fe) and their coordinating residues based on geometry and residue types.")
            mock_metal_sites = random.randint(0,2)
            st.metric(label="Predicted Metal Binding Sites (Mock)", value=mock_metal_sites)
            if mock_metal_sites > 0:
                st.markdown(f"- Potential Zn<sup>2+</sup> site coordinated by Cys{random.randint(1,40)}, Cys{random.randint(41,80)}, His{random.randint(81,120)}.")

        with st.expander("🍬 Glycosylation Site Structural Context"):
            st.info("Analyzes the structural environment (accessibility, secondary structure) of potential N-linked or O-linked glycosylation sites.")
            mock_glyco_sites = data.get('sequence',"").count('N') // 3 + data.get('sequence',"").count('S') // 5 + data.get('sequence',"").count('T') // 5
            st.metric(label="Potential Glycosylation Sites Analyzed (Mock)", value=random.randint(0, max(1,mock_glyco_sites)))
            st.markdown(f"N-glycosylation motif N-X-S/T at Asn{random.randint(1,data.get('length',100))} found in a surface loop.")

        with st.expander("✂️ Protein Cleavage Site Accessibility"):
            st.info("Assesses the solvent accessibility and structural context of predicted protease cleavage sites within the protein.")
            mock_cleavage_sites_exposed = random.randint(0,4)
            st.metric(label="Exposed Cleavage Sites (Mock)", value=mock_cleavage_sites_exposed)
            st.markdown(f"Trypsin cleavage site after Arg{random.randint(1,data.get('length',100))} predicted to be highly accessible.")

        with st.expander("🔍 Structural Motif Search (e.g., Helix-Turn-Helix)"):
            st.info("Searches for known structural motifs (e.g., Helix-Turn-Helix, Zinc Finger, Beta-Barrel) within the predicted 3D structure.")
            found_motifs = random.choice([0,1])
            st.metric(label="Known Structural Motifs Found (Mock)", value=found_motifs)
            if found_motifs > 0:
                st.markdown(f"- Helix-Loop-Helix motif detected at residues {random.randint(10,30)}-{random.randint(50,70)}.")

        with st.expander("📉 Inter-Residue Distance Matrix Plot"):
            st.info("Visualizes the matrix of distances between C-alpha atoms of all pairs of residues. Patterns can highlight domains and long-range interactions.")
            # This would typically be a heatmap, similar to contact map but with continuous distance values.
            st.markdown("A heatmap would be displayed here showing pairwise Cα-Cα distances.")
            st.markdown(f"Average Cα-Cα distance for non-adjacent residues (Mock): {round(random.uniform(5,25),1)} Å.")

        with st.expander("📦 Packing Density & Void Analysis"):
            st.info("Calculates local and global packing density and identifies internal voids or cavities within the protein structure, which can affect stability and dynamics.")
            mock_packing_density = round(random.uniform(0.65, 0.78), 2)
            mock_voids = random.randint(0,5)
            st.metric(label="Overall Packing Density (Mock)", value=mock_packing_density)
            st.metric(label="Number of Internal Voids > 10Å³ (Mock)", value=mock_voids)

        with st.expander("💠 Protein Symmetry Detection"):
            symmetry_data = generate_mock_protein_symmetry_data()
            st.metric(label="Predicted Symmetry Type (Mock)", value=symmetry_data['type'])
            if symmetry_data['type'] != "None":
                st.markdown(f"**Symmetry Axis (Mock):** {symmetry_data['axis']}")
                st.markdown(f"**Confidence (Mock):** {symmetry_data['confidence']:.2f}")
            st.markdown("Detects internal repeats or symmetry in oligomeric assemblies (if applicable).")

        with st.expander("💞 Co-evolutionary Contact Prediction Mapping"):
            df_coevo = generate_mock_coevolution_contacts(data.get('length', 100))
            if not df_coevo.empty:
                st.dataframe(df_coevo.head(), use_container_width=True)
                st.markdown("Predicted residue pairs that co-evolve, suggesting spatial proximity or functional interaction. These can guide 3D folding or identify interaction sites.")
            else:
                st.info("No significant co-evolutionary contacts predicted (mock data).")

        with st.expander("💦 Structural Water Molecule Prediction"):
            df_waters = generate_mock_structural_waters()
            st.metric(label="Predicted Structural Waters (Mock)", value=len(df_waters))
            if not df_waters.empty:
                st.dataframe(df_waters.head(), use_container_width=True)
                st.markdown("Predicts positions of water molecules that are integral to the protein's structure or function, often found in active sites or mediating interactions.")
            else:
                st.info("No significant structural waters predicted (mock data).")

        with st.expander("🚇 Ion Channel Pore Radius Profiling"):
            if data.get('length', 100) > 50 and random.random() < 0.2: # Simulate it being a channel protein
                df_pore = generate_mock_pore_profile(data.get('length',100))
                fig_pore = px.line(df_pore, x="Position_Angstrom", y="Radius_Angstrom",
                                   title="Mock Ion Channel Pore Radius Profile",
                                   labels={"Position_Angstrom": "Position along Pore Axis (Å)", "Radius_Angstrom": "Pore Radius (Å)"})
                st.plotly_chart(fig_pore, use_container_width=True)
                st.metric(label="Minimum Pore Radius (Mock)", value=f"{df_pore['Radius_Angstrom'].min():.1f} Å")
                st.markdown("For transmembrane channel proteins, this visualizes the dimensions of the pore, identifying constrictions and selectivity filters.")
            else:
                st.info("Protein not predicted as a channel or insufficient data for pore profiling (mock).")

        with st.expander("🟠 Protein Surface Curvature Analysis"):
            df_curvature = generate_mock_surface_curvature(data.get('length', 100))
            curvature_counts = df_curvature['Curvature_Type_Pred'].value_counts().reset_index()
            curvature_counts.columns = ['Predicted Curvature Type', 'Residue Count'] # Renaming for clarity
            fig_curv_pie = px.pie(curvature_counts, values='Residue Count', names='Predicted Curvature Type',
                                  title="Distribution of Surface Curvature Types (Predicted)",
                                  color_discrete_map={"Convex": "skyblue", "Concave": "salmon", "Saddle": "lightgreen", "Flat": "lightgrey"})
            st.plotly_chart(fig_curv_pie, use_container_width=True)
            st.markdown("Analyzes the shape of the protein surface. Concave regions often form binding pockets, while convex regions can be involved in interactions.")

        with st.expander("📚 Helix/Sheet Packing Geometry"):
            num_secondary_elements = len(data.get('secondary_structure', [])) // 20 # Rough estimate
            df_packing = generate_mock_packing_geometry(max(2, num_secondary_elements))
            if not df_packing.empty:
                st.dataframe(df_packing.head(), use_container_width=True)
                st.markdown("Characterizes how alpha-helices and beta-sheets pack against each other, defining the protein's core architecture.")
            else:
                st.info("Not enough secondary structure elements for detailed packing analysis (mock).")

        with st.expander("📁 Fold Recognition & Classification"):
            df_folds = generate_mock_fold_recognition()
            st.dataframe(df_folds, use_container_width=True)
            st.markdown("Compares the predicted 3D structure to databases of known protein folds (e.g., CATH, SCOP) to classify its architecture and infer potential functions.")

        with st.expander("🧊 Cryo-EM Map Fitting Score (Simulated)"):
            cryo_fit_data = generate_mock_cryoem_fit()
            st.metric(label="Simulated Cryo-EM Map Resolution (Mock)", value=f"{cryo_fit_data['Resolution_Angstrom_Mock']} Å")
            st.metric(label="Cross-Correlation Score with Map (Mock)", value=f"{cryo_fit_data['Cross_Correlation_Score_Mock']:.3f}")
            st.markdown(f"**Map Segmentation Quality (Mock):** {cryo_fit_data['Map_Segmentation_Quality_Mock']}")
            st.markdown("Assesses how well the predicted atomic model fits into an experimental Cryo-Electron Microscopy (Cryo-EM) density map, if available.")

        with st.expander("📡 NMR Chemical Shift Prediction"):
            # Simplified: show average predicted shift for a few nuclei types
            st.markdown(f"**Avg. Predicted Cα Shift (Mock):** {round(random.uniform(40, 70),1)} ppm")
            st.markdown(f"**Avg. Predicted HN Shift (Mock):** {round(random.uniform(7, 10),1)} ppm")
            st.markdown("Predicts Nuclear Magnetic Resonance (NMR) chemical shifts for backbone and sidechain atoms based on the 3D structure. Useful for validating structures against experimental NMR data.")

        with st.expander("✨ SAXS Profile Prediction"):
            df_saxs, rg_saxs_mock = generate_mock_saxs_profile()
            fig_saxs = px.line(df_saxs, x="q_Angstrom_inv", y="Intensity_I_q_arbitrary_units", log_y=True,
                               title="Predicted SAXS Profile (Mock)",
                               labels={"q_Angstrom_inv": "q (Å⁻¹)", "Intensity_I_q_arbitrary_units": "Intensity (log scale)"})
            st.plotly_chart(fig_saxs, use_container_width=True)
            st.metric(label="Radius of Gyration (Rg) from SAXS (Mock)", value=f"{rg_saxs_mock:.1f} Å")
            st.markdown("Predicts the Small-Angle X-ray Scattering (SAXS) profile, which provides information about the protein's overall shape and size in solution.")

        with st.expander("❄️ Crystallization Propensity Score"):
            crystallization_data = generate_mock_crystallization_propensity()
            st.metric(label="Overall Crystallization Propensity (Mock)", value=f"{crystallization_data['Overall_Propensity_Score_Mock']:.2f}")
            st.markdown(f"**Number of Low Surface Entropy Patches (Mock):** {crystallization_data['Number_of_Low_Entropy_Patches_Mock']}")
            st.markdown("Estimates the likelihood of a protein to form well-ordered crystals, based on surface properties like hydrophobicity, charge, and conformational homogeneity.")

        with st.expander("🤝 Interface Residue Propensity (for PPI)"):
            # Simplified: show average propensity for a mock interface
            st.markdown(f"**Avg. Interface Propensity Score for Mock Interface:** {round(random.uniform(0.5,0.85),2)}")
            st.markdown("Analyzes the physicochemical properties of residues at predicted protein-protein interaction interfaces to assess their likelihood of being involved in binding.")

        with st.expander("🕸️ Elastic Network Model Analysis"):
            # Similar to NMA, but can be focused on different aspects
            st.markdown(f"**Lowest Frequency Mode (Mock):** Describes {random.choice(['hinge-bending', 'breathing', 'twisting'])} motion.")
            st.markdown(f"**Predicted B-factors from ENM (Avg. Mock):** {round(random.uniform(20,60),1)} Å²")
            st.markdown("Uses a simplified spring network (Elastic Network Model) to predict collective motions and flexibility, often correlating with Normal Mode Analysis.")

        with st.expander("💊 Fragment-Based Docking Suitability"):
            st.markdown(f"**Most Druggable Pocket (Mock ID {random.randint(1,3)}) Suitability for Fragments:** {random.choice(['High', 'Medium', 'Low'])}")
            st.markdown("Evaluates identified binding pockets for their suitability for fragment-based drug discovery, considering size, shape, and chemical environment.")

        with st.expander("🔥 Hot Spot Residue Prediction (Interaction)"):
            num_hotspots = random.randint(1,5)
            hotspot_residues = sorted(random.sample(range(1, data.get('length',100)+1), num_hotspots))
            st.markdown(f"**Predicted Interaction Hot Spot Residues (Mock):** {', '.join(map(str, hotspot_residues))}")
            st.markdown("Identifies key residues that contribute disproportionately to the binding energy of protein-protein or protein-ligand interactions.")

        with st.expander("🔦 Protein Tunnelling Analysis"):
            num_tunnels = random.randint(0,3)
            st.metric(label="Predicted Tunnels/Channels (Mock)", value=num_tunnels)
            if num_tunnels > 0:
                st.markdown(f"**Main Tunnel (Mock):** Length {round(random.uniform(10,30),1)} Å, Bottleneck Radius {round(random.uniform(1.0,3.0),1)} Å.")
            st.markdown("Identifies and characterizes internal tunnels or channels that may be important for substrate access, product egress, or ion transport.")

        with st.expander("💨 Hydrodynamic Properties Estimation (e.g., Stokes Radius)"):
            stokes_radius_mock = round(0.7 * (data.get('length', 100)**0.33) * random.uniform(0.8, 1.2), 1) # Approximation
            diffusion_coeff_mock = round( (1.38e-23 * 300) / (6 * np.pi * 8.9e-4 * (stokes_radius_mock * 1e-10)) * 1e10, 1) # Stokes-Einstein, m^2/s to um^2/s
            st.metric(label="Estimated Stokes Radius (Mock)", value=f"{stokes_radius_mock} Å")
            st.metric(label="Estimated Diffusion Coefficient (Mock)", value=f"{diffusion_coeff_mock} µm²/s")
            st.markdown("Estimates properties like Stokes radius and diffusion coefficient from the 3D structure, relevant for understanding behavior in solution (e.g., in SEC, DLS).")

        with st.expander("🛡️ Structure-Based Antibody Epitope Prediction"):
            num_epitopes = random.randint(1,4)
            st.metric(label="Predicted Conformational Epitopes (Mock)", value=num_epitopes)
            if num_epitopes > 0:
                epitope_residues = sorted(random.sample(range(1, data.get('length',100)+1), random.randint(8,15)))
                st.markdown(f"**Example Epitope Patch (Mock):** Residues {', '.join(map(str, epitope_residues))}")
            st.markdown("Identifies continuous or discontinuous (conformational) regions on the protein surface likely to be recognized by antibodies.")

        with st.expander("🧭 Protein Dipole Moment Calculation"):
            dipole_magnitude_mock = round(random.uniform(50, 500) * (data.get('length',100)/100.0), 0)
            st.metric(label="Calculated Dipole Moment Magnitude (Mock)", value=f"{dipole_magnitude_mock} Debye")
            st.markdown("Calculates the overall electric dipole moment of the protein, which can influence interactions with other molecules and behavior in electric fields.")

        with st.expander("💠 Protein Symmetry Detection"):
            symmetry_data = generate_mock_protein_symmetry_data()
            st.metric(label="Predicted Symmetry Type (Mock)", value=symmetry_data['type'])
            if symmetry_data['type'] != "None":
                st.markdown(f"**Symmetry Axis (Mock):** {symmetry_data['axis']}")
                st.markdown(f"**Confidence (Mock):** {symmetry_data['confidence']:.2f}")
            st.markdown("Detects internal repeats or symmetry in oligomeric assemblies (if applicable).")

        with st.expander("💞 Co-evolutionary Contact Prediction Mapping"):
            df_coevo = generate_mock_coevolution_contacts(data.get('length', 100))
            if not df_coevo.empty:
                st.dataframe(df_coevo.head(), use_container_width=True)
                st.markdown("Predicted residue pairs that co-evolve, suggesting spatial proximity or functional interaction. These can guide 3D folding or identify interaction sites.")
            else:
                st.info("No significant co-evolutionary contacts predicted (mock data).")

        with st.expander("💦 Structural Water Molecule Prediction"):
            df_waters = generate_mock_structural_waters()
            st.metric(label="Predicted Structural Waters (Mock)", value=len(df_waters))
            if not df_waters.empty:
                st.dataframe(df_waters.head(), use_container_width=True)
                st.markdown("Predicts positions of water molecules that are integral to the protein's structure or function, often found in active sites or mediating interactions.")
            else:
                st.info("No significant structural waters predicted (mock data).")

        with st.expander("🚇 Ion Channel Pore Radius Profiling"):
            if data.get('length', 100) > 50 and random.random() < 0.2: # Simulate it being a channel protein
                df_pore = generate_mock_pore_profile(data.get('length',100))
                fig_pore = px.line(df_pore, x="Position_Angstrom", y="Radius_Angstrom",
                                   title="Mock Ion Channel Pore Radius Profile",
                                   labels={"Position_Angstrom": "Position along Pore Axis (Å)", "Radius_Angstrom": "Pore Radius (Å)"})
                st.plotly_chart(fig_pore, use_container_width=True)
                st.metric(label="Minimum Pore Radius (Mock)", value=f"{df_pore['Radius_Angstrom'].min():.1f} Å")
                st.markdown("For transmembrane channel proteins, this visualizes the dimensions of the pore, identifying constrictions and selectivity filters.")
            else:
                st.info("Protein not predicted as a channel or insufficient data for pore profiling (mock).")

        # --- End of newly added tools ---




        with st.expander("📖 Rotamer Library Analysis"):
            rotamer_data = generate_mock_rotamer_analysis(data.get('length',100))
            df_rotamer = pd.DataFrame(rotamer_data.items(), columns=["Rotamer Category", "Percentage"])
            fig_rotamer_pie = px.pie(df_rotamer, values="Percentage", names="Rotamer Category",
                                     title="Side-Chain Rotamer Distribution (Mock)",
                                     color_discrete_map={"Favored_Rotamers_Percent": "green", 
                                                         "Allowed_Rotamers_Percent": "orange", 
                                                         "Outlier_Rotamers_Percent": "red"})
            st.plotly_chart(fig_rotamer_pie, use_container_width=True)
            st.markdown("Assesses the conformational preferences of amino acid side chains against statistical libraries of known rotamers. High percentage of outliers might indicate strained regions.")
    
    with tab_map["AI"]:
        st.subheader("AI-Generated Structural Analysis")
        if 'ai_analysis' in data:
            st.markdown(data['ai_analysis'])
        else:
            st.info("AI analysis not available")
    
    with tab_map["CONF"]:
        st.subheader("Confidence Score Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            fig_conf = create_confidence_distribution(data)
            st.plotly_chart(fig_conf, use_container_width=True)
        
        with col2:
            # Confidence statistics
            plddt = data['plddt']
            st.write("**Confidence Statistics:**")
            st.write(f"• Mean pLDDT: {np.mean(plddt):.1f}")
            st.write(f"• Median pLDDT: {np.median(plddt):.1f}")
            st.write(f"• High confidence (>70): {np.sum(plddt > 70)} residues")
            st.write(f"• Low confidence (<50): {np.sum(plddt < 50)} residues")
            
            # Confidence regions
            high_conf_regions = []
            low_conf_regions = []
            
            for i, score in enumerate(plddt):
                if score > 70:
                    high_conf_regions.append(i + 1)
                elif score < 50:
                    low_conf_regions.append(i + 1)
            
            if high_conf_regions:
                st.success(f"High confidence regions: {len(high_conf_regions)} residues")
            if low_conf_regions:
                st.warning(f"Low confidence regions: {len(low_conf_regions)} residues")

            st.subheader("Advanced Confidence Analysis Tools")
            with st.expander("📊 Per-Residue Confidence Plot (pLDDT)"):
                # This is already shown in the main structure plot, but can be reiterated or styled differently
                fig_plddt_conf = go.Figure(data=[go.Scatter(x=list(range(1, data['length'] + 1)), y=data['plddt'], mode='lines+markers',
                                                          marker=dict(size=4), line=dict(color='teal'))])
                fig_plddt_conf.update_layout(title="Per-Residue pLDDT Score", xaxis_title="Residue Index", yaxis_title="pLDDT", height=300)
                fig_plddt_conf.add_hline(y=90, line_dash="dot", line_color="green", annotation_text="Very High")
                fig_plddt_conf.add_hline(y=70, line_dash="dot", line_color="orange", annotation_text="Confident")
                fig_plddt_conf.add_hline(y=50, line_dash="dot", line_color="red", annotation_text="Low")
                st.plotly_chart(fig_plddt_conf, use_container_width=True)
                st.markdown("This plot shows the pLDDT score for each residue, indicating local model confidence.")

            with st.expander("🗺️ Confidence Heatmap"):
                # For a 1D sequence, a heatmap isn't standard unless comparing to something else.
                # We can show a "heatmap-like" bar chart where color intensity represents confidence.
                df_plddt_bar = pd.DataFrame({'Residue Index': list(range(1, data['length'] + 1)), 'pLDDT': data['plddt']})
                fig_conf_heatmap_bar = px.bar(df_plddt_bar, x='Residue Index', y='pLDDT', color='pLDDT',
                                            title="Confidence Score Profile (Heatmap-style Bar)",
                                            color_continuous_scale=px.colors.sequential.Viridis,
                                            labels={'pLDDT': 'pLDDT Score'})
                fig_conf_heatmap_bar.update_layout(yaxis_range=[0,100], height=300)
                st.plotly_chart(fig_conf_heatmap_bar, use_container_width=True)
                st.markdown("This bar chart uses color intensity to represent pLDDT scores across the sequence, offering another view of confidence distribution.")

            with st.expander("📉 Confidence Score Moving Average"):
                window_size = st.slider("Moving Average Window:", min_value=3, max_value=max(5, data['length']//10), value=min(5, data['length']//10), key="conf_ma_window")
                plddt_series = pd.Series(data['plddt'])
                moving_avg = plddt_series.rolling(window=window_size, center=True, min_periods=1).mean()
                fig_ma = go.Figure()
                fig_ma.add_trace(go.Scatter(x=list(range(1, data['length'] + 1)), y=data['plddt'], mode='lines', name='pLDDT', line=dict(color='lightblue', width=1)))
                fig_ma.add_trace(go.Scatter(x=list(range(1, data['length'] + 1)), y=moving_avg, mode='lines', name=f'MA ({window_size})', line=dict(color='red', width=2)))
                fig_ma.update_layout(title="pLDDT with Moving Average", xaxis_title="Residue Index", yaxis_title="pLDDT", height=350)
                st.plotly_chart(fig_ma, use_container_width=True)
                st.markdown("The moving average helps to smooth out local fluctuations in confidence scores, highlighting broader trends and regions of sustained high or low confidence.")

            with st.expander("🧩 Segmented Confidence Analysis (Domains)"):
                if data['domains']:
                    st.write("Average pLDDT per predicted domain:")
                    domain_plddt_data = []
                    for domain in data['domains']:
                        # Ensure start and end are within bounds of plddt array
                        start_idx = max(0, domain['start'] - 1) # 0-indexed
                        end_idx = min(len(data['plddt']), domain['end'])
                        if start_idx < end_idx:
                            domain_plddt_scores = data['plddt'][start_idx:end_idx]
                            avg_domain_plddt = np.mean(domain_plddt_scores) if len(domain_plddt_scores) > 0 else np.nan
                            st.markdown(f"- **{domain['name']} ({domain['start']}-{domain['end']})**: {avg_domain_plddt:.2f}")
                            domain_plddt_data.append({'Domain': domain['name'], 'Avg_pLDDT': avg_domain_plddt})
                        else:
                            st.markdown(f"- **{domain['name']} ({domain['start']}-{domain['end']})**: Invalid range for pLDDT.")
                    if domain_plddt_data:
                        df_domain_plddt = pd.DataFrame(domain_plddt_data)
                        fig_domain_plddt = px.bar(df_domain_plddt, x='Domain', y='Avg_pLDDT', title="Average pLDDT per Domain", color='Avg_pLDDT', color_continuous_scale='Blues')
                        st.plotly_chart(fig_domain_plddt, use_container_width=True)
                else:
                    st.info("No domains defined for segmented analysis.")

            with st.expander("🔬 Confidence Correlation with Secondary Structure"):
                df_ss_conf = pd.DataFrame({'SS': data['secondary_structure'], 'pLDDT': data['plddt']})
                avg_conf_by_ss = df_ss_conf.groupby('SS')['pLDDT'].agg(['mean', 'median', 'count']).sort_values(by='mean', ascending=False)
                st.write("Average pLDDT by Secondary Structure Type:")
                st.dataframe(avg_conf_by_ss.reset_index().rename(columns={'mean': 'Mean pLDDT', 'median': 'Median pLDDT', 'count': 'Residue Count'}), use_container_width=True)
                fig_ss_conf_box = px.box(df_ss_conf, x='SS', y='pLDDT', color='SS', title="pLDDT Distribution by Secondary Structure")
                st.plotly_chart(fig_ss_conf_box, use_container_width=True)

            with st.expander("🔗 Confidence of Inter-Domain Linkers"):
                if data['domains'] and len(data['domains']) > 1:
                    st.write("Average pLDDT for Inter-Domain Linkers:")
                    linkers_plddt_all = []
                    linker_info = []
                    for i in range(len(data['domains']) - 1):
                        linker_start_res = data['domains'][i]['end'] + 1 # 1-indexed start of linker
                        linker_end_res = data['domains'][i+1]['start'] - 1 # 1-indexed end of linker
                        
                        linker_start_idx = data['domains'][i]['end'] # 0-indexed end of domain i
                        linker_end_idx = data['domains'][i+1]['start'] - 1 # 0-indexed start of domain i+1 (exclusive end for plddt slice)

                        if linker_start_idx < linker_end_idx: # Ensure linker exists
                            linker_plddt = data['plddt'][linker_start_idx : linker_end_idx]
                            if len(linker_plddt) > 0:
                                avg_linker_plddt = np.mean(linker_plddt)
                                linker_name = f"Linker: {data['domains'][i]['name']} - {data['domains'][i+1]['name']} ({linker_start_res}-{linker_end_res})"
                                st.markdown(f"- {linker_name}: {avg_linker_plddt:.2f}")
                                linkers_plddt_all.extend(linker_plddt)
                                linker_info.append({'Linker': linker_name, 'Avg_pLDDT': avg_linker_plddt, 'Length': len(linker_plddt)})
                    if linkers_plddt_all:
                        st.metric("Overall Average Linker pLDDT", f"{np.mean(linkers_plddt_all):.2f}")
                        if linker_info:
                            df_linker_info = pd.DataFrame(linker_info)
                            fig_linker_plddt = px.bar(df_linker_info, x='Linker', y='Avg_pLDDT', title="Average pLDDT per Linker", color='Length', color_continuous_scale='Purples')
                            st.plotly_chart(fig_linker_plddt, use_container_width=True)
                    else:
                        st.info("No significant inter-domain linker regions found or domains are overlapping/adjacent.")
                else:
                    st.info("Not enough domains (need at least 2) to analyze inter-domain linkers.")

            with st.expander("🧬 Confidence vs. Disorder Prediction"):
                df_conf_disorder = pd.DataFrame({
                    'Residue Index': list(range(1, data['length'] + 1)),
                    'pLDDT': data['plddt'],
                    'Is_Disordered': data['disorder'].astype(bool) # Ensure boolean for proper coloring
                })
                fig_conf_disorder = px.box(df_conf_disorder, x='Is_Disordered', y='pLDDT',
                                           color='Is_Disordered',
                                           title="pLDDT Scores for Ordered vs. Disordered Regions",
                                           labels={"Is_Disordered": "Predicted Disorder Status", "pLDDT": "pLDDT Score"},
                                           color_discrete_map={True: "salmon", False: "skyblue"})
                st.plotly_chart(fig_conf_disorder, use_container_width=True)
                avg_plddt_disordered = df_conf_disorder[df_conf_disorder['Is_Disordered'] == True]['pLDDT'].mean()
                avg_plddt_ordered = df_conf_disorder[df_conf_disorder['Is_Disordered'] == False]['pLDDT'].mean()
                st.markdown(f"**Average pLDDT in Disordered Regions:** {avg_plddt_disordered:.2f}")
                st.markdown(f"**Average pLDDT in Ordered Regions:** {avg_plddt_ordered:.2f}")
                st.markdown("Disordered regions often exhibit lower pLDDT scores, reflecting higher structural uncertainty.")

            with st.expander("🌍 Confidence Anomaly Detection (IQR based)"):
                plddt_series = pd.Series(data['plddt'])
                Q1 = plddt_series.quantile(0.25)
                Q3 = plddt_series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound_iqr = Q1 - 1.5 * IQR
                upper_bound_iqr = Q3 + 1.5 * IQR

                anomalies_low_iqr = plddt_series[plddt_series < lower_bound_iqr]
                anomalies_high_iqr = plddt_series[plddt_series > upper_bound_iqr]

                st.markdown(f"**Lower Bound (Q1 - 1.5*IQR):** {lower_bound_iqr:.2f}")
                st.markdown(f"**Upper Bound (Q3 + 1.5*IQR):** {upper_bound_iqr:.2f}")
                st.markdown(f"**Low Confidence Anomalies (pLDDT < {lower_bound_iqr:.2f}):** {len(anomalies_low_iqr)} residues. Indices: {', '.join(map(str, anomalies_low_iqr.index.map(lambda x: x+1).tolist()[:5]))}{'...' if len(anomalies_low_iqr) > 5 else ''}")
                st.markdown(f"**High Confidence Anomalies (pLDDT > {upper_bound_iqr:.2f}):** {len(anomalies_high_iqr)} residues. Indices: {', '.join(map(str, anomalies_high_iqr.index.map(lambda x: x+1).tolist()[:5]))}{'...' if len(anomalies_high_iqr) > 5 else ''}")
                st.markdown("Identifies residues with pLDDT scores falling outside 1.5 times the Interquartile Range, potentially indicating unusually stable or unstable micro-domains.")

            with st.expander("📈 Cumulative Confidence Distribution Function (CDF)"):
                plddt_sorted = np.sort(data['plddt'])
                cdf = np.arange(1, len(plddt_sorted) + 1) / len(plddt_sorted)
                fig_cdf = go.Figure(data=[go.Scatter(x=plddt_sorted, y=cdf, mode='lines', line=dict(color='darkviolet'))])
                fig_cdf.update_layout(title="Cumulative Distribution of pLDDT Scores",
                                      xaxis_title="pLDDT Score", yaxis_title="Cumulative Probability", height=350)
                st.plotly_chart(fig_cdf, use_container_width=True)
                percentile_70 = np.percentile(data['plddt'], 70)
                st.markdown(f"**70% of residues have a pLDDT score of {percentile_70:.2f} or lower (based on this prediction).**")
                st.markdown("The CDF shows the proportion of residues that have a pLDDT score less than or equal to a given value. Useful for understanding overall confidence distribution.")

            with st.expander("🎯 Confidence of User-Defined Functional Site Residues"):
                st.markdown("Enter comma-separated residue indices (1-based) of a known or hypothesized functional site to check their pLDDT scores.")
                site_residues_input = st.text_input("Functional Site Residues (e.g., 10,12,45,102):", key="conf_func_site_input")
                if site_residues_input:
                    try:
                        site_indices_user_func = [int(x.strip()) - 1 for x in site_residues_input.split(',') if x.strip().isdigit()] # 0-indexed
                        site_indices_user_func = [idx for idx in site_indices_user_func if 0 <= idx < data['length']] # Validate indices
                        if site_indices_user_func:
                            site_plddt_user_func = data['plddt'][site_indices_user_func]
                            avg_site_plddt_user_func = np.mean(site_plddt_user_func) if len(site_plddt_user_func) > 0 else np.nan
                            st.metric(label=f"Avg. pLDDT of Defined Site ({len(site_indices_user_func)} residues)", value=f"{avg_site_plddt_user_func:.2f}" if not np.isnan(avg_site_plddt_user_func) else "N/A")
                            st.write(f"pLDDT scores for site residues: {', '.join([f'{idx+1}:{data['plddt'][idx]:.2f}' for idx in site_indices_user_func])}")
                            if not np.isnan(avg_site_plddt_user_func) and avg_site_plddt_user_func < 70:
                                st.warning("The defined functional site has relatively low average confidence, which may impact functional interpretation.")
                        else:
                            st.info("No valid residue indices provided or indices out of range.")
                    except Exception as e:
                        st.error(f"Error parsing residue indices: {e}")
                st.markdown("Allows users to input specific residue indices (e.g., from experimental data) and assess their prediction confidence.")

            with st.expander("🗺️ 3D Confidence Mapping (Conceptual)"):
                st.info("This would typically involve coloring a 3D representation of the protein by pLDDT scores. Below is a conceptual 1D representation.")
                # Re-use the heatmap-style bar chart as a 1D proxy
                df_plddt_bar_3d = pd.DataFrame({'Residue Index': list(range(1, data['length'] + 1)), 'pLDDT': data['plddt']})
                fig_conf_heatmap_bar_3d = px.bar(df_plddt_bar_3d, x='Residue Index', y='pLDDT', color='pLDDT',
                                            title="1D Proxy for 3D Confidence Mapping (pLDDT per Residue)",
                                            color_continuous_scale=px.colors.sequential.Inferno,
                                            labels={'pLDDT': 'pLDDT Score'})
                fig_conf_heatmap_bar_3d.update_layout(yaxis_range=[0,100], height=300)
                st.plotly_chart(fig_conf_heatmap_bar_3d, use_container_width=True)
                st.markdown("In a 3D viewer, residues would be colored by their pLDDT score, allowing quick visual identification of well-folded vs. uncertain regions.")

            # --- Start of implementations for the user's provided list ---

            with st.expander("🌐 Confidence of Surface vs. Core Residues"):
                sasa_data_surf_core = generate_mock_sasa_data(data['length']) # Using existing mock SASA
                sasa_threshold_surf_core = np.median(sasa_data_surf_core) if len(sasa_data_surf_core) > 0 else 50 # Default threshold if no SASA
                
                df_surf_core = pd.DataFrame({
                    'Residue_Index': range(1, data['length'] + 1),
                    'pLDDT': data['plddt'],
                    'SASA': sasa_data_surf_core,
                    'Location': ['Surface' if s > sasa_threshold_surf_core else 'Core' for s in sasa_data_surf_core]
                })
                avg_plddt_surface = df_surf_core[df_surf_core['Location'] == 'Surface']['pLDDT'].mean()
                avg_plddt_core = df_surf_core[df_surf_core['Location'] == 'Core']['pLDDT'].mean()
                
                st.metric(label="Avg. pLDDT Surface Residues (Mock SASA)", value=f"{avg_plddt_surface:.2f}" if not np.isnan(avg_plddt_surface) else "N/A")
                st.metric(label="Avg. pLDDT Core Residues (Mock SASA)", value=f"{avg_plddt_core:.2f}" if not np.isnan(avg_plddt_core) else "N/A")
                
                if not df_surf_core.empty:
                    fig_surf_core = px.box(df_surf_core, x='Location', y='pLDDT', color='Location',
                                           title="pLDDT Distribution: Surface vs. Core (Mock SASA)",
                                           labels={"Location": "Predicted Location", "pLDDT": "pLDDT Score"})
                    st.plotly_chart(fig_surf_core, use_container_width=True)
                st.markdown(f"Residues classified based on mock SASA relative to median SASA ({sasa_threshold_surf_core:.1f} Å²). Core residues are often more confidently predicted than surface loops.")

            with st.expander("⚠️ Confidence-Based Warning Flags"):
                low_conf_thresh_warn = 50
                very_low_conf_thresh_warn = 30
                low_conf_residues_count = np.sum(data['plddt'] < low_conf_thresh_warn)
                very_low_conf_residues_count = np.sum(data['plddt'] < very_low_conf_thresh_warn)
                percentage_low_conf = (low_conf_residues_count / data['length']) * 100 if data['length'] > 0 else 0
                
                warnings_found = False
                if percentage_low_conf > 30:
                    st.error(f"**Warning:** {percentage_low_conf:.1f}% of residues have pLDDT < {low_conf_thresh_warn}. Overall model confidence is low.")
                    warnings_found = True
                elif percentage_low_conf > 10:
                    st.warning(f"**Caution:** {percentage_low_conf:.1f}% of residues have pLDDT < {low_conf_thresh_warn}. Some regions may be unreliable.")
                    warnings_found = True
                if very_low_conf_residues_count > 0:
                    st.error(f"**Critical Warning:** {very_low_conf_residues_count} residues have pLDDT < {very_low_conf_thresh_warn}, indicating very unreliable predictions for these regions.")
                    warnings_found = True
                if data.get('overall_confidence', 100) < 60:
                    st.warning(f"**Overall Average pLDDT is {data.get('overall_confidence', 0):.1f}, which is moderate. Interpret with caution.**")
                    warnings_found = True
                
                if not warnings_found:
                    st.success("No major confidence-based warning flags triggered based on pLDDT thresholds.")
                st.markdown("Flags potential issues based on pLDDT scores, such as large proportions of low-confidence residues or critically unreliable segments.")

            with st.expander("🔄 Confidence Stability Across Multiple Runs (Conceptual)"):
                st.markdown("To assess stability, this protein sequence would need to be predicted multiple times (e.g., with different random seeds if the model is stochastic, or by different models). The pLDDT profiles could then be compared.")
                if len(st.session_state.prediction_history) > 1:
                    st.info(f"There are {len(st.session_state.prediction_history)} predictions in history. If any are for the same sequence, their pLDDTs could be compared (e.g., by RMSD of pLDDT vectors or overlaying plots). This demo uses mock data, so direct comparison might not be meaningful unless sequence inputs were identical.")
                else:
                    st.info("Run multiple predictions (ideally of the same sequence with varied seeds/models) to enable stability analysis.")
                st.markdown("Assesses the consistency of pLDDT scores if the same protein is predicted multiple times, indicating model robustness.")

            # Localized Confidence Minima/Maxima is already implemented further down, this is the placeholder for it.
            # The existing implementation will be used.

            with st.expander("📚 Confidence Score Comparison to Database Averages (for similar folds)"):
                mock_fold_type = random.choice(["TIM Barrel", "Rossmann Fold", "Globin Fold", "Beta-Propeller", "Jelly Roll"])
                st.markdown(f"If this protein were identified as a **{mock_fold_type}**, its pLDDT profile could be compared to average pLDDT profiles of known {mock_fold_type}s from structural databases (e.g., CATH, SCOP).")
                st.markdown("This requires fold recognition (see 'Fold Recognition' tab in STRUCT) and access to fold-specific confidence databases.")
                
                # Conceptual plot
                mock_db_avg_plddt = np.sort(np.random.normal(loc=random.uniform(75,85), scale=random.uniform(5,12), size=data['length']))[::-1] # Smoother, generally high profile
                mock_db_avg_plddt = np.clip(mock_db_avg_plddt, 40, 100)

                fig_db_comp = go.Figure()
                fig_db_comp.add_trace(go.Scatter(x=list(range(1, data['length']+1)), y=data['plddt'], name="Current Protein pLDDT", line=dict(color='blue')))
                fig_db_comp.add_trace(go.Scatter(x=list(range(1, data['length']+1)), y=mock_db_avg_plddt, name=f"Mock Avg. pLDDT for {mock_fold_type}", line=dict(color='red', dash='dash')))
                fig_db_comp.update_layout(title=f"pLDDT vs. Mock Database Average for {mock_fold_type}", xaxis_title="Normalized Residue Index", yaxis_title="pLDDT", height=350)
                st.plotly_chart(fig_db_comp, use_container_width=True)
                st.markdown("Compares the protein's pLDDT profile to average confidence profiles of proteins with similar known folds from databases.")

            with st.expander("🧩 Confidence of Domain Boundaries"):
                if data['domains']:
                    boundary_width = st.slider("Boundary region width (+/- residues):", 1, 5, 3, key="conf_domain_boundary_width")
                    boundary_plddts_info = []
                    all_boundary_scores = []

                    for i, domain in enumerate(data['domains']):
                        # Start boundary of current domain
                        start_boundary_indices = list(range(max(0, domain['start'] - 1 - boundary_width), min(data['length'], domain['start'] -1 + boundary_width)))
                        if start_boundary_indices:
                            scores = data['plddt'][start_boundary_indices]
                            avg_plddt = np.mean(scores) if len(scores) > 0 else np.nan
                            boundary_plddts_info.append({'Boundary': f"{domain['name']} Start", 'Avg_pLDDT': avg_plddt, 'Residues': f"{max(1, domain['start'] - boundary_width)}-{min(data['length'], domain['start'] + boundary_width -1)}"})
                            if not np.isnan(avg_plddt): all_boundary_scores.append(avg_plddt)
                        
                        # End boundary of current domain
                        end_boundary_indices = list(range(max(0, domain['end'] - boundary_width), min(data['length'], domain['end'] + boundary_width)))
                        if end_boundary_indices:
                            scores = data['plddt'][end_boundary_indices]
                            avg_plddt = np.mean(scores) if len(scores) > 0 else np.nan
                            boundary_plddts_info.append({'Boundary': f"{domain['name']} End", 'Avg_pLDDT': avg_plddt, 'Residues': f"{max(1, domain['end'] - boundary_width +1)}-{min(data['length'], domain['end'] + boundary_width)}"})
                            if not np.isnan(avg_plddt): all_boundary_scores.append(avg_plddt)

                    if boundary_plddts_info:
                        df_boundary_plddts = pd.DataFrame(boundary_plddts_info)
                        st.dataframe(df_boundary_plddts, use_container_width=True)
                        if all_boundary_scores:
                            st.metric("Overall Avg. pLDDT at Domain Boundaries", f"{np.mean(all_boundary_scores):.2f}")
                    else:
                        st.info("Could not define boundary regions for pLDDT analysis (e.g., domains at termini or very short protein).")
                else:
                    st.info("No domains defined for boundary confidence analysis.")
                st.markdown("Analyzes pLDDT scores specifically at the predicted boundaries between structural domains. Lower confidence here can indicate flexible linkers or uncertainty in domain definition.")

            with st.expander("⚡ Confidence Correlation with Electrostatic Potential (Mock)"):
                charge_map = {'K': 1, 'R': 1, 'H': 0.5, 'D': -1, 'E': -1} # Simplified charges
                mock_electrostatics = [charge_map.get(res, 0) for res in data['sequence']]
                df_elec_conf = pd.DataFrame({'pLDDT': data['plddt'], 'MockCharge': mock_electrostatics})
                correlation_elec = df_elec_conf['pLDDT'].corr(df_elec_conf['MockCharge']) if len(df_elec_conf) > 1 else np.nan
                
                fig_elec_corr = px.scatter(df_elec_conf, x='MockCharge', y='pLDDT', 
                                           title=f"pLDDT vs. Mock Residue Charge (Correlation: {correlation_elec:.2f})", 
                                           trendline="ols" if len(df_elec_conf['MockCharge'].unique()) > 1 else None, 
                                           trendline_color_override="purple",
                                           marginal_y="violin", marginal_x="box")
                st.plotly_chart(fig_elec_corr, use_container_width=True)
                st.markdown(f"**Correlation Coefficient:** {correlation_elec:.2f}" if not np.isnan(correlation_elec) else "N/A (not enough data for correlation)")
                st.markdown("Examines if residue confidence correlates with a simplified mock charge. Real electrostatic potential requires 3D structure and specialized software (e.g., APBS).")

            with st.expander("📐 Confidence Score vs. Local Geometry (Mock Outliers)"):
                # Mock Ramachandran outliers (as true geometry isn't available)
                outlier_fraction = 0.05 # 5% mock outliers
                num_outliers = int(data['length'] * outlier_fraction)
                outlier_indices = random.sample(range(data['length']), k=max(1, num_outliers)) if data['length'] > 0 else []
                is_outlier = [True if i in outlier_indices else False for i in range(data['length'])]
                
                df_geom_conf = pd.DataFrame({'pLDDT': data['plddt'], 'IsMockOutlier': is_outlier})
                avg_plddt_outliers = df_geom_conf[df_geom_conf['IsMockOutlier']]['pLDDT'].mean()
                avg_plddt_non_outliers = df_geom_conf[~df_geom_conf['IsMockOutlier']]['pLDDT'].mean()
                
                st.metric(label="Avg. pLDDT for Mock Geometric Outliers", value=f"{avg_plddt_outliers:.2f}" if not np.isnan(avg_plddt_outliers) else "N/A")
                st.metric(label="Avg. pLDDT for Non-Outliers", value=f"{avg_plddt_non_outliers:.2f}" if not np.isnan(avg_plddt_non_outliers) else "N/A")
                
                if not df_geom_conf.empty:
                    fig_geom_box = px.box(df_geom_conf, x='IsMockOutlier', y='pLDDT', color='IsMockOutlier', title="pLDDT for Mock Geometric Outliers vs. Non-Outliers")
                    st.plotly_chart(fig_geom_box, use_container_width=True)
                st.markdown("Compares pLDDT of residues flagged as mock geometric outliers (e.g., Ramachandran). Lower confidence might correlate with strained geometry. True local geometry requires 3D coordinates.")

            with st.expander("🏷️ Confidence of PTM Site Residues (Potential)"):
                ptm_residues_map = {'S': 'Phosphorylation, Glycosylation', 'T': 'Phosphorylation, Glycosylation', 
                                    'Y': 'Phosphorylation', 'K': 'Ubiquitination, Acetylation, Methylation', 
                                    'N': 'Glycosylation', 'R': 'Methylation', 'Q': 'Deamidation'}
                potential_ptm_indices = [i for i, res in enumerate(data['sequence']) if res in ptm_residues_map]
                
                if potential_ptm_indices:
                    ptm_plddt_scores = data['plddt'][potential_ptm_indices]
                    avg_ptm_plddt = np.mean(ptm_plddt_scores) if len(ptm_plddt_scores) > 0 else np.nan
                    st.metric(label=f"Avg. pLDDT of Potential PTM Sites ({len(potential_ptm_indices)} residues)", value=f"{avg_ptm_plddt:.2f}" if not np.isnan(avg_ptm_plddt) else "N/A")
                    
                    df_ptm_sites = pd.DataFrame({
                        'Residue_Index': [i+1 for i in potential_ptm_indices],
                        'Residue_Type': [data['sequence'][i] for i in potential_ptm_indices],
                        'pLDDT': ptm_plddt_scores,
                        'Potential_PTMs': [ptm_residues_map[data['sequence'][i]] for i in potential_ptm_indices]
                    })
                    st.dataframe(df_ptm_sites.sort_values(by='pLDDT').head(10), use_container_width=True)
                    st.markdown(f"Potential PTM-prone residues show an average pLDDT of {avg_ptm_plddt:.2f}. Specific PTM prediction tools and structural context are needed for confirmation.")
                else:
                    st.info("No common PTM-prone residues (S,T,Y,K,N,R,Q) found in the sequence.")
                st.markdown("Analyzes pLDDT scores of residues known or commonly predicted to undergo Post-Translational Modifications. PTM sites in flexible/exposed regions might have lower confidence.")

            with st.expander("🛡️ Confidence of Epitope Regions (Mock)"):
                num_mock_epitopes = random.randint(1, max(1, data['length'] // 50))
                epitope_data_list = []
                for i in range(num_mock_epitopes):
                    epitope_len = random.randint(8, 15)
                    if data['length'] - epitope_len -1 < 0 : continue # Skip if protein too short
                    start = random.randint(0, data['length'] - epitope_len -1)
                    end = start + epitope_len
                    epitope_indices = list(range(start, end))
                    epitope_plddt = data['plddt'][epitope_indices]
                    avg_epitope_plddt = np.mean(epitope_plddt) if len(epitope_plddt) > 0 else np.nan
                    epitope_data_list.append({
                        'Epitope_ID': f"MockEpitope_{i+1}",
                        'Residues': f"{start+1}-{end}",
                        'Avg_pLDDT': avg_epitope_plddt
                    })
                
                if epitope_data_list:
                    df_epitopes = pd.DataFrame(epitope_data_list)
                    st.dataframe(df_epitopes, use_container_width=True)
                    fig_epitope_conf = px.bar(df_epitopes, x='Epitope_ID', y='Avg_pLDDT', title="Avg. pLDDT of Mock Epitope Regions", color='Avg_pLDDT', color_continuous_scale='Temps')
                    st.plotly_chart(fig_epitope_conf, use_container_width=True)
                else:
                    st.info("No mock epitopes generated (protein might be too short).")
                st.markdown("Analyzes pLDDT scores for mock-defined surface regions that could act as epitopes. Epitopes in flexible loops might have lower confidence.")

            with st.expander("🌍 Confidence Score Spatial Clustering (1D Proxy)"):
                bins = [0, 50, 70, 90, 101] # pLDDT bins
                labels = ["Very Low (0-50)", "Low (50-70)", "Confident (70-90)", "Very High (90-100)"]
                plddt_binned = pd.cut(data['plddt'], bins=bins, labels=labels, right=False, include_lowest=True)
                
                st.write("Distribution of residues by confidence bins:")
                bin_counts = pd.Series(plddt_binned).value_counts().sort_index().reset_index()
                bin_counts.columns = ['Confidence Bin', 'Residue Count']
                st.dataframe(bin_counts)
                
                fig_bin_pie = px.pie(bin_counts, values='Residue Count', names='Confidence Bin', title="Residue Distribution by Confidence Bins",
                                     color_discrete_map={
                                         "Very Low (0-50)": "red", "Low (50-70)": "orange",
                                         "Confident (70-90)": "lightgreen", "Very High (90-100)": "green"
                                     })
                st.plotly_chart(fig_bin_pie, use_container_width=True)
                st.markdown("This 1D binning approximates clustering. True spatial clustering requires 3D coordinates to find regions of similar confidence in 3D space.")

            with st.expander("📊 Confidence Score Thresholding Analysis"):
                thresholds = [50, 70, 90]
                results = []
                for t in thresholds:
                    count = np.sum(data['plddt'] >= t)
                    percentage = (count / data['length']) * 100
                    results.append({"Threshold": f"> {t}", "Residue Count": count, "Percentage": f"{percentage:.1f}%"})
                df_thresholds = pd.DataFrame(results)
                st.dataframe(df_thresholds, use_container_width=True)
                st.markdown("Analyzes the number and percentage of residues above certain pLDDT thresholds, giving an overview of overall model quality.")
            
            with st.expander("📊 Confidence Score Percentiles"):
                percentiles_to_calc = [1, 5, 10, 25, 50, 75, 90, 95, 99]
                plddt_percentiles = np.percentile(data['plddt'], percentiles_to_calc)
                df_percentiles = pd.DataFrame({'Percentile': percentiles_to_calc, 'pLDDT Score': plddt_percentiles})
                st.write("pLDDT Score at Various Percentiles:")
                st.dataframe(df_percentiles.set_index('Percentile'), use_container_width=True)
                st.markdown("The 50th percentile is the median pLDDT score. This helps understand the distribution and skewness of confidence scores.")

            with st.expander("📉 Confidence Drop-off Analysis (Termini)"):
                n_termini_len = st.slider("N/C-Termini Length to Analyze:", 5, max(10, data['length']//10), min(15, data['length']//3), key="conf_termini_len")
                
                n_term_plddt = data['plddt'][:n_termini_len]
                c_term_plddt = data['plddt'][-n_termini_len:]
                
                avg_n_term_plddt = np.mean(n_term_plddt) if len(n_term_plddt) > 0 else np.nan
                avg_c_term_plddt = np.mean(c_term_plddt) if len(c_term_plddt) > 0 else np.nan
                
                col_term1, col_term2 = st.columns(2)
                with col_term1:
                    st.metric(label=f"Avg. pLDDT N-terminus (first {n_termini_len} res)", value=f"{avg_n_term_plddt:.2f}" if not np.isnan(avg_n_term_plddt) else "N/A")
                with col_term2:
                    st.metric(label=f"Avg. pLDDT C-terminus (last {n_termini_len} res)", value=f"{avg_c_term_plddt:.2f}" if not np.isnan(avg_c_term_plddt) else "N/A")
                
                overall_avg_plddt = data.get('overall_confidence', np.mean(data['plddt']))
                if not np.isnan(avg_n_term_plddt) and avg_n_term_plddt < overall_avg_plddt - 10: # Arbitrary threshold for "notably lower"
                    st.warning(f"N-terminus shows notably lower confidence ({avg_n_term_plddt:.2f}) than overall average ({overall_avg_plddt:.2f}).")
                if not np.isnan(avg_c_term_plddt) and avg_c_term_plddt < overall_avg_plddt - 10:
                    st.warning(f"C-terminus shows notably lower confidence ({avg_c_term_plddt:.2f}) than overall average ({overall_avg_plddt:.2f}).")
                st.markdown("Analyzes how pLDDT scores behave towards the N and C termini, which are often more flexible and less confidently predicted.")

            with st.expander("📈 Confidence Score Plateau Analysis"):
                min_plateau_len = st.slider("Min Plateau Length:", 3, max(5, data['length']//10), min(10, data['length']//5), key="conf_plateau_len")
                plateau_std_thresh = st.slider("Max Std Dev for Plateau:", 0.5, 10.0, 2.0, 0.5, key="conf_plateau_std")

                plddt_series_plat = pd.Series(data['plddt'])
                rolling_std_plat = plddt_series_plat.rolling(window=min_plateau_len, center=True, min_periods=1).std()
                
                # Identify regions where rolling_std is below threshold
                plateau_candidate_indices = np.where(rolling_std_plat < plateau_std_thresh)[0]
                
                plateaus_identified = []
                if len(plateau_candidate_indices) > 0:
                    current_plateau = [plateau_candidate_indices[0]]
                    for i in range(1, len(plateau_candidate_indices)):
                        if plateau_candidate_indices[i] == plateau_candidate_indices[i-1] + 1:
                            current_plateau.append(plateau_candidate_indices[i])
                        else:
                            if len(current_plateau) >= min_plateau_len:
                                plateaus_identified.append(current_plateau)
                            current_plateau = [plateau_candidate_indices[i]]
                    if len(current_plateau) >= min_plateau_len: # Add last identified plateau
                        plateaus_identified.append(current_plateau)
                
                st.markdown(f"**Identified {len(plateaus_identified)} potential plateaus (stable pLDDT, min length {min_plateau_len}, std < {plateau_std_thresh}):**")
                plateau_df_list = []
                for i, plat_indices in enumerate(plateaus_identified):
                    avg_plddt_plat = np.mean(data['plddt'][plat_indices])
                    plateau_df_list.append({
                        'Plateau_ID': i+1, 
                        'Start': plat_indices[0]+1, 
                        'End': plat_indices[-1]+1, 
                        'Length': len(plat_indices), 
                        'Avg_pLDDT': round(avg_plddt_plat, 2)
                    })
                
                if plateau_df_list:
                    df_plateaus = pd.DataFrame(plateau_df_list)
                    st.dataframe(df_plateaus.head(10), use_container_width=True)
                else:
                    st.info("No significant plateaus identified with current settings.")
                st.markdown("Plateaus are regions where pLDDT scores are relatively constant, indicating consistently modeled segments (either high or low confidence).")

            with st.expander("⚙️ Confidence-Weighted Structural Alignment"):
                st.markdown("In real structural alignments (e.g., with TM-align, DALI, PyMOL's `super` command), pLDDT scores could be used to assign higher weights to residues with high confidence during the superposition process. This would prioritize aligning well-predicted regions over uncertain ones.")
                st.markdown("This application does not perform actual structural alignment, but this is a common use case for pLDDT scores in comparative modeling.")
            
            # "Confidence Score Variance Analysis" (general) placeholder is removed as "Local Window" is implemented and more practical.

            with st.expander("↔️ Confidence Score Symmetry (for symmetric proteins)"):
                st.markdown("If the protein is known or predicted to be symmetric (e.g., a homodimer), the pLDDT profiles of the symmetrically related units could be compared. Significant deviations might indicate modeling inconsistencies or actual asymmetry in a dynamic complex.")
                st.markdown("This requires prior knowledge or prediction of symmetry (see 'Protein Symmetry Detection' in STRUCT tab). For example, if C2 symmetry is predicted along the Z-axis for a dimer, you would compare pLDDT of chain A vs. chain B.")

            with st.expander("🤝 Confidence of Interface Residues (PPI)"):
                # Using the mock PPI data from the INT tab
                protein_id_ppi_conf = data['sequence'][:15] # Mock ID from sequence
                mock_interactions_ppi_conf = generate_mock_interaction_data(protein_id_ppi_conf, num_interactions=random.randint(1,2)) 
                
                ppi_interface_residues_info = []
                for interaction in mock_interactions_ppi_conf:
                    if interaction['details']['type'] == "PPI":
                        # Mock some interface residues for this partner
                        num_interface_res = random.randint(5, min(20, data['length']//5 if data['length'] > 25 else 5))
                        if num_interface_res > 0 and data['length'] >= num_interface_res:
                            interface_indices_ppi = sorted(random.sample(range(data['length']), k=num_interface_res))
                            interface_plddt_ppi = data['plddt'][interface_indices_ppi]
                            avg_interface_plddt_ppi = np.mean(interface_plddt_ppi) if len(interface_plddt_ppi) > 0 else np.nan
                            ppi_interface_residues_info.append({
                                'Partner_ID': interaction['partner_id'],
                                'Avg_Interface_pLDDT': avg_interface_plddt_ppi,
                                'Num_Interface_Residues': len(interface_indices_ppi),
                                'Interface_Residues_Indices': ", ".join(map(str, [x+1 for x in interface_indices_ppi][:10])) + ('...' if len(interface_indices_ppi)>10 else '')
                            })
                if ppi_interface_residues_info:
                    df_ppi_interfaces = pd.DataFrame(ppi_interface_residues_info)
                    st.dataframe(df_ppi_interfaces, use_container_width=True)
                else:
                    st.info("No mock PPI interface data generated for this analysis here. Check INT tab for general PPI predictions.")
                st.markdown("Analyzes pLDDT scores of residues predicted to be at protein-protein interfaces. High confidence at interfaces is desirable for reliable interaction models.")

            with st.expander("💊 Confidence of Ligand-Binding Pocket Residues"):
                # Uses mock ligand pockets from LIG tab's generator
                mock_pockets_conf = generate_mock_ligand_pockets(data['length'], num_pockets=random.randint(1,3))
                pocket_conf_data = []
                if mock_pockets_conf:
                    for pocket in mock_pockets_conf:
                        try:
                            pocket_res_indices = [int(x.strip()) - 1 for x in pocket['residues'].split(',') if x.strip().isdigit()]
                            pocket_res_indices = [idx for idx in pocket_res_indices if 0 <= idx < data['length']]
                            if pocket_res_indices:
                                pocket_plddts = data['plddt'][pocket_res_indices]
                                avg_pocket_plddt = np.mean(pocket_plddts) if len(pocket_plddts) > 0 else np.nan
                                pocket_conf_data.append({
                                    'Pocket_ID': pocket['pocket_id'],
                                    'Avg_pLDDT': avg_pocket_plddt,
                                    'Num_Residues': len(pocket_res_indices),
                                    'Druggability': pocket['druggability_score']
                                })
                        except ValueError: # Handle cases where residues might not be purely numeric
                            pass 
                    if pocket_conf_data:
                        df_pocket_conf = pd.DataFrame(pocket_conf_data)
                        st.dataframe(df_pocket_conf, use_container_width=True)
                    else:
                        st.info("No valid mock ligand binding pocket residues found for confidence analysis.")
                else:
                    st.info("No mock ligand binding pockets generated to analyze confidence. Check LIG tab.")
                st.markdown("Analyzes pLDDT scores of residues forming predicted ligand-binding pockets. High confidence in pocket regions is important for drug design.")

            # "Confidence of Flexible Hinge Regions" (general) placeholder is covered by the "Inferred" version.

            with st.expander("⚡ Confidence Score vs. Predicted ddG of Mutation"):
                df_mutations_ddg_conf = generate_mock_mutational_data(data['length'], num_mutations=max(10, data['length'] // 10))
                
                mut_positions_ddg = []
                for mut_str in df_mutations_ddg_conf['Mutation']:
                    match = re.search(r'\d+', mut_str)
                    if match:
                        mut_positions_ddg.append(int(match.group()) - 1) # 0-indexed
                
                valid_mut_indices = [p for p in mut_positions_ddg if 0 <= p < data['length']]
                
                if valid_mut_indices:
                    plddt_at_mutation_site = data['plddt'][valid_mut_indices]
                    # Align ddG with pLDDT (some mutations might be out of bounds if not careful with generation)
                    # We need to ensure that df_mutations_ddg_conf is filtered or aligned with valid_mut_indices
                    
                    # Create a temporary df for merging based on original mutation string to keep ddG values aligned
                    temp_mut_df = df_mutations_ddg_conf.copy()
                    temp_mut_df['Residue_Index_0_Based'] = mut_positions_ddg # Add the parsed 0-based index

                    # Filter this temp_mut_df for valid indices
                    valid_temp_mut_df = temp_mut_df[temp_mut_df['Residue_Index_0_Based'].isin(valid_mut_indices)]

                    df_ddg_conf_plot = pd.DataFrame({
                        'pLDDT_at_Site': data['plddt'][valid_temp_mut_df['Residue_Index_0_Based'].tolist()], # Get pLDDT for these valid sites
                        'Predicted_ddG': valid_temp_mut_df['Predicted_ddG_kcal_mol']
                    })

                    if not df_ddg_conf_plot.empty:
                        correlation_ddg = df_ddg_conf_plot['pLDDT_at_Site'].corr(df_ddg_conf_plot['Predicted_ddG'].abs()) if len(df_ddg_conf_plot) > 1 else np.nan
                        fig_ddg_corr = px.scatter(df_ddg_conf_plot, x='pLDDT_at_Site', y='Predicted_ddG', 
                                                  title=f"pLDDT vs. Predicted ΔΔG (Correlation with |ΔΔG|: {correlation_ddg:.2f})", 
                                                  trendline="ols" if len(df_ddg_conf_plot['pLDDT_at_Site'].unique()) > 1 and len(df_ddg_conf_plot['Predicted_ddG'].unique()) > 1 else None,
                                                  marginal_x="histogram", marginal_y="histogram")
                        st.plotly_chart(fig_ddg_corr, use_container_width=True)
                        st.markdown(f"**Correlation Coefficient (pLDDT vs. |ΔΔG|):** {correlation_ddg:.2f}" if not np.isnan(correlation_ddg) else "N/A")
                        st.markdown("Examines if the confidence of a wild-type residue's prediction correlates with the predicted stability change upon its mutation. Highly confident, stable residues might show larger (more disruptive) ΔΔG upon mutation, or vice-versa.")
                    else:
                        st.info("Not enough valid mutation data for ddG correlation after filtering.")
                else:
                    st.info("No valid mutation data for ddG correlation (e.g. protein too short or mock data issue).")

            # "Confidence Score Temperature Scale Mapping" (general) placeholder is covered by the "Visualization" version.
            # "Confidence Landscape Visualization" (general) placeholder is covered by the "Smoothed Profile" version.

            with st.expander("🔗 Confidence of Disulfide-Bonded Cysteines"):
                cys_indices_ss = [i for i, res in enumerate(data['sequence']) if res == 'C']
                bonded_cys_plddt_scores = []
                mock_ss_pairs_info = []

                if len(cys_indices_ss) >= 2:
                    num_mock_ss_bonds = random.randint(0, len(cys_indices_ss) // 2)
                    temp_cys_indices_for_pairing = list(cys_indices_ss) # Mutable copy
                    
                    for _ in range(num_mock_ss_bonds):
                        if len(temp_cys_indices_for_pairing) >= 2:
                            pair = random.sample(temp_cys_indices_for_pairing, 2)
                            p1_idx, p2_idx = sorted(pair)
                            
                            plddt1 = data['plddt'][p1_idx]
                            plddt2 = data['plddt'][p2_idx]
                            avg_pair_plddt = (plddt1 + plddt2) / 2
                            bonded_cys_plddt_scores.extend([plddt1, plddt2])
                            mock_ss_pairs_info.append({
                                'Pair': f"Cys{p1_idx+1}-Cys{p2_idx+1}",
                                f'pLDDT_Cys{p1_idx+1}': round(plddt1,2),
                                f'pLDDT_Cys{p2_idx+1}': round(plddt2,2),
                                'Avg_Pair_pLDDT': round(avg_pair_plddt,2)
                            })
                            # Remove paired cysteines from further consideration
                            temp_cys_indices_for_pairing.remove(p1_idx)
                            temp_cys_indices_for_pairing.remove(p2_idx)
                    
                    if mock_ss_pairs_info:
                        df_ss_bonds = pd.DataFrame(mock_ss_pairs_info)
                        st.write(f"**Mock Disulfide Bonds ({len(mock_ss_pairs_info)} pairs):**")
                        st.dataframe(df_ss_bonds, use_container_width=True)
                        if bonded_cys_plddt_scores:
                             st.metric(label="Avg. pLDDT of Mock SS-Bonded Cysteines", value=f"{np.mean(bonded_cys_plddt_scores):.2f}")
                    else:
                        st.info("No mock disulfide bonds formed (or not enough Cysteines for pairing).")
                else:
                    st.info("Not enough Cysteine residues in the sequence to form disulfide bonds.")
                st.markdown("Checks pLDDT scores of Cysteine residues predicted/mocked to form disulfide bonds. These bonds are often crucial for stability, so high confidence is expected for these residues.")

            # Practical Implementations for previously placeholder/mocked tools
            with st.expander("📉 Confidence Score Entropy"):
                plddt_counts = np.histogram(data['plddt'], bins=10, range=(0,100))[0]
                plddt_probs = plddt_counts / data['length']
                score_entropy = entropy(plddt_probs, base=2)
                st.metric(label="pLDDT Score Entropy (bits)", value=f"{score_entropy:.3f}")
                st.markdown("Measures the heterogeneity of pLDDT scores. Higher entropy indicates more diverse confidence levels across the protein.")

            with st.expander("🔎 Localized Confidence Minima/Maxima"):
                from scipy.signal import find_peaks
                plddt_scores = data['plddt']
                minima_indices, _ = find_peaks(-plddt_scores, distance=5, prominence=5) # Find peaks in negative for minima
                maxima_indices, _ = find_peaks(plddt_scores, distance=5, prominence=5) # Prominence helps filter small peaks

                st.markdown(f"**Localized Minima (Indices):** {', '.join(map(str, minima_indices[:5]+1))}{'...' if len(minima_indices) > 5 else ''}")
                st.markdown(f"**Localized Maxima (Indices):** {', '.join(map(str, maxima_indices[:5]+1))}{'...' if len(maxima_indices) > 5 else ''}")
                st.markdown("Identifies specific residues or short segments that are local minima or maxima in pLDDT scores, indicating focal points of low or high confidence.")

            with st.expander("💧 Confidence Correlation with Hydrophobicity (Kyte-Doolittle)"):
                hydrophobicity_scores = [AA_HYDROPHOBICITY_KD.get(res, 0) for res in data['sequence']]
                df_hydro_conf = pd.DataFrame({'pLDDT': data['plddt'], 'Hydrophobicity': hydrophobicity_scores})
                correlation_hydro = df_hydro_conf['pLDDT'].corr(df_hydro_conf['Hydrophobicity'])
                fig_hydro_corr = px.scatter(df_hydro_conf, x='Hydrophobicity', y='pLDDT', title=f"pLDDT vs. Kyte-Doolittle Hydrophobicity (Correlation: {correlation_hydro:.2f})", trendline="ols", trendline_color_override="green")
                st.plotly_chart(fig_hydro_corr, use_container_width=True)
                st.markdown(f"**Correlation Coefficient:** {correlation_hydro:.2f}")
                st.markdown("Examines if residue confidence correlates with its hydrophobicity. For example, buried hydrophobic cores might show higher confidence.")

            with st.expander("🔬 Confidence Score Variance Analysis (Local Window)"):
                window_var = st.slider("Variance Window Size:", min_value=3, max_value=max(5, data['length']//10), value=min(7, data['length']//10), key="conf_var_window")
                plddt_series_var = pd.Series(data['plddt'])
                local_variance = plddt_series_var.rolling(window=window_var, center=True, min_periods=1).var()
                fig_var = px.line(x=list(range(1, data['length'] + 1)), y=local_variance, title=f"Local pLDDT Score Variance (Window: {window_var})", labels={'x':'Residue Index', 'y':'Variance'})
                st.plotly_chart(fig_var, use_container_width=True)
                st.markdown("Calculates the variance of pLDDT scores in local windows. High variance indicates rapidly changing confidence.")

            with st.expander("📉 Confidence Score Autocorrelation"):
                max_lags = min(50, data['length'] // 2 -1)
                if max_lags > 0:
                    autocorr_values = acf(data['plddt'], nlags=max_lags, fft=False) # fft=False for robustness with smaller series
                    fig_autocorr = px.bar(x=list(range(len(autocorr_values))), y=autocorr_values, title=f"pLDDT Autocorrelation (up to {max_lags} lags)", labels={'x':'Lag', 'y':'Autocorrelation'})
                    st.plotly_chart(fig_autocorr, use_container_width=True)
                    st.markdown("Analyzes the autocorrelation of pLDDT scores to understand characteristic lengths of confident/unconfident regions. High autocorrelation at a given lag means scores are similar at that distance.")
                else:
                    st.info("Protein too short for meaningful autocorrelation analysis.")

            with st.expander("📊 Confidence Score Distribution by Residue Type"):
                df_res_conf = pd.DataFrame({'Residue': list(data['sequence']), 'pLDDT': data['plddt']})
                fig_res_conf_box = px.box(df_res_conf, x='Residue', y='pLDDT', color='Residue', title="pLDDT Distribution by Amino Acid Type", category_orders={"Residue": sorted(list(set(data['sequence'])))})
                st.plotly_chart(fig_res_conf_box, use_container_width=True)
                avg_plddt_by_res = df_res_conf.groupby('Residue')['pLDDT'].mean().sort_values(ascending=False)
                st.write("Average pLDDT per Residue Type:")
                st.dataframe(avg_plddt_by_res.reset_index(), use_container_width=True)
                st.markdown("Analyzes if certain amino acid types tend to have higher or lower pLDDT scores in this specific protein prediction.")

            with st.expander("💊 Confidence of User-Defined Ligand-Binding Pocket Residues"):
                st.markdown("Enter comma-separated residue indices (1-based) of a known or hypothesized ligand-binding pocket to check their pLDDT scores.")
                pocket_residues_input = st.text_input("Pocket Residues (e.g., 22,25,50,98):", key="conf_pocket_site_input")
                if pocket_residues_input:
                    try:
                        pocket_indices = [int(x.strip()) - 1 for x in pocket_residues_input.split(',') if x.strip().isdigit()] # 0-indexed
                        pocket_indices = [idx for idx in pocket_indices if 0 <= idx < data['length']] # Validate indices
                        if pocket_indices:
                            pocket_plddt = data['plddt'][pocket_indices]
                            avg_pocket_plddt = np.mean(pocket_plddt)
                            st.metric(label=f"Average pLDDT of Defined Pocket ({len(pocket_indices)} residues)", value=f"{avg_pocket_plddt:.2f}")
                            st.write(f"pLDDT scores for pocket residues: {', '.join([f'{idx+1}:{data['plddt'][idx]:.2f}' for idx in pocket_indices])}")
                            if avg_pocket_plddt < 70:
                                st.warning("The defined ligand-binding pocket has relatively low average confidence.")
                        else:
                            st.info("No valid residue indices provided for the pocket.")
                    except Exception as e:
                        st.error(f"Error parsing residue indices: {e}")

            with st.expander("🌀 Confidence of Flexible Hinge Regions (Inferred)"):
                st.markdown("Hinge regions are often characterized by lower confidence scores flanked by higher confidence domains. This tool looks for such patterns.")
                # Simple inference: look for significant dips in smoothed pLDDT
                plddt_series_hinge = pd.Series(data['plddt'])
                smoothed_plddt = plddt_series_hinge.rolling(window=max(5, data['length']//20), center=True, min_periods=1).mean()
                potential_hinges = smoothed_plddt[ (smoothed_plddt < smoothed_plddt.quantile(0.25)) & (plddt_series_hinge < 60) ] # Low smoothed and raw pLDDT
                if not potential_hinges.empty:
                    st.write(f"Found {len(potential_hinges)} potential flexible/hinge residues (smoothed pLDDT < {smoothed_plddt.quantile(0.25):.2f} and raw pLDDT < 60).")
                    st.write(f"Example indices: {', '.join(map(str, potential_hinges.index.map(lambda x: x+1).tolist()[:10]))}{'...' if len(potential_hinges)>10 else ''}")
                else:
                    st.info("No clear flexible hinge regions inferred based on simple pLDDT dips.")

            with st.expander("🌡️ Confidence Score Temperature Scale Mapping (Visualization)"):
                # Map pLDDT to a color scale (e.g., blue=high, red=low)
                df_temp_map = pd.DataFrame({'Residue Index': list(range(1, data['length'] + 1)), 'pLDDT': data['plddt']})
                fig_temp_map = px.bar(df_temp_map, x='Residue Index', y='pLDDT', color='pLDDT',
                                      title="pLDDT Mapped to Temperature-Like Color Scale",
                                      color_continuous_scale='RdYlBu', labels={'pLDDT': 'Confidence (Blue=High, Red=Low)'})
                fig_temp_map.update_layout(yaxis_range=[0,100], height=300)
                st.plotly_chart(fig_temp_map, use_container_width=True)
                st.markdown("Visualizes pLDDT scores using a color scale often associated with temperature factors (Blue=cold/ordered/confident, Red=hot/disordered/unconfident).")

            with st.expander("🗺️ Confidence Landscape Visualization (Smoothed Profile)"):
                # Use moving average as a representation of a smoothed landscape
                window_landscape = st.slider("Landscape Smoothing Window:", min_value=3, max_value=max(5, data['length']//5), value=min(10, data['length']//10), key="conf_landscape_window")
                plddt_series_land = pd.Series(data['plddt'])
                smoothed_landscape = plddt_series_land.rolling(window=window_landscape, center=True, min_periods=1).mean()
                fig_land = go.Figure()
                fig_land.add_trace(go.Scatter(x=list(range(1, data['length'] + 1)), y=data['plddt'], mode='lines', name='Raw pLDDT', line=dict(color='rgba(173,216,230,0.6)', width=1.5))) # Light blue for raw
                fig_land.add_trace(go.Scatter(x=list(range(1, data['length'] + 1)), y=smoothed_landscape, mode='lines', name=f'Smoothed Landscape (Window: {window_landscape})', line=dict(color='navy', width=2.5)))
                fig_land.update_layout(title="Confidence Landscape (Smoothed pLDDT Profile)", xaxis_title="Residue Index", yaxis_title="pLDDT Score", height=350)
                st.plotly_chart(fig_land, use_container_width=True)
                st.markdown("Provides a smoothed view of the confidence scores, highlighting broader regions of high or low confidence, akin to a 'landscape'.")

    
    with tab_map["DOMAIN"]:
        st.subheader("Domain Architecture Analysis")
        
        if data['domains']:
            for i, domain in enumerate(data['domains']):
                with st.expander(f"Domain {i+1}: {domain['name']} ({domain['type']})"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**Start:** {domain['start']}")
                    with col2:
                        st.write(f"**End:** {domain['end']}")
                    with col3:
                        st.write(f"**Length:** {domain['end'] - domain['start']} AA")
                    
                    st.write(f"**Type:** {domain['type']}")
                    
                    # Domain sequence
                    domain_seq = data['sequence'][domain['start']:domain['end']]
                    st.text_area(f"Domain {i+1} Sequence:", domain_seq, height=100)

                    # Mock domain-specific pLDDT
                    domain_plddt_scores = data['plddt'][domain['start']:domain['end']]
                    avg_domain_plddt = np.mean(domain_plddt_scores) if len(domain_plddt_scores) > 0 else np.nan
                    st.metric(label="Average pLDDT for this Domain", value=f"{avg_domain_plddt:.2f}" if not np.isnan(avg_domain_plddt) else "N/A")

            # Inter-domain linker analysis (if multiple domains)
            if len(data['domains']) > 1:
                st.subheader("Inter-Domain Linker Analysis")
                for i in range(len(data['domains']) - 1):
                    linker_start = data['domains'][i]['end'] + 1
                    linker_end = data['domains'][i+1]['start'] - 1
                    if linker_start <= linker_end:
                        st.markdown(f"- **Linker {i+1} (Residues {linker_start}-{linker_end}):** Length {linker_end - linker_start + 1} AA. Avg. pLDDT (Mock): {np.mean(data['plddt'][linker_start-1:linker_end]):.2f}")
        else:
            st.info("No distinct domains identified in this protein")
    
    with tab_map["INT"]:
        st.subheader("Predicted Interaction Network")
        protein_id = data['sequence'][:15] # Mock ID from sequence
        mock_interactions = generate_mock_interaction_data(protein_id, num_interactions=random.randint(3,7))

        if not mock_interactions:
            st.info("No interactions predicted for this protein.")
        else:
            st.write(f"Predicted interactions for protein (first 15AA): `{protein_id}...`")
            
            # Separate data for PPI and Ligand for clarity
            ppi_data = []
            ligand_data = []

            for inter in mock_interactions:
                if inter['details']['type'] == "PPI":
                    ppi_data.append({
                        "Partner Protein ID": inter['partner_id'],
                        "Confidence": inter['details']['confidence'],
                        "Predicted Interface Residues (Mock)": ", ".join(map(str, sorted(random.sample(range(1, data['length']+1), random.randint(5,15))))),
                        "Interaction Type (Mock)": random.choice(["Direct", "Indirect", "Transient"])
                    })
                else: # Protein-Ligand
                    ligand_data.append({
                        "Ligand ID": inter['partner_id'],
                        "Predicted Affinity (nM)": inter['details']['affinity_nM'],
                        "Binding Site Residues (Mock)": ", ".join(map(str, sorted(random.sample(range(1, data['length']+1), random.randint(3,10))))),
                        "Predicted Binding Mode (Mock)": random.choice(["Competitive", "Non-competitive", "Allosteric"])
                    })
            
            if ppi_data:
                st.markdown("##### Protein-Protein Interactions (PPIs)")
                df_ppi = pd.DataFrame(ppi_data)
                st.dataframe(df_ppi, use_container_width=True)
                
                if not df_ppi.empty:
                    fig_ppi_scores = px.bar(df_ppi, x="Partner Protein ID", y="Confidence", title="PPI Confidence Scores",
                                        color="Confidence", color_continuous_scale=px.colors.sequential.Viridis)
                st.plotly_chart(fig_ppi_scores, use_container_width=True)

            if ligand_data:
                st.markdown("##### Protein-Ligand Interactions")
                df_ligand = pd.DataFrame(ligand_data)
                st.dataframe(df_ligand, use_container_width=True)

                if not df_ligand.empty:
                    fig_ligand_affinity = px.bar(df_ligand, x="Ligand ID", y="Predicted Affinity (nM)", title="Predicted Ligand Affinities",
                                             color="Predicted Affinity (nM)", color_continuous_scale=px.colors.sequential.Plasma_r)
                fig_ligand_affinity.update_yaxes(type="log")
                st.plotly_chart(fig_ligand_affinity, use_container_width=True)
            
            # Conceptual Network Graph for PPIs
            if ppi_data:
                st.markdown("##### Conceptual PPI Network")
                dot_string = "digraph G {\n rankdir=LR;\n node [shape=protein, style=filled, color=skyblue];\n"
                dot_string += f' "{protein_id}..." [label="{protein_id}...", fillcolor=lightcoral];\n' # Query protein
                for ppi_entry in ppi_data:
                    partner = ppi_entry["Partner Protein ID"]
                    confidence = ppi_entry["Confidence"]
                    dot_string += f' "{protein_id}..." -> "{partner}" [label="{confidence:.2f}", penwidth={1 + confidence*2}];\n'
                dot_string += "}"
                st.graphviz_chart(dot_string)

            st.markdown("---")
            st.markdown("_Note: Interaction data is mock-generated. For real analysis, use tools like STRING-DB, BioGRID for PPIs, and docking software for ligand interactions._")

    with tab_map["MUT"]:
        st.subheader("In Silico Mutational Analysis")
        sequence_length = data.get('length', 100)
        df_mutations = generate_mock_mutational_data(sequence_length, num_mutations=random.randint(5,15))

        st.write("Predicted effects of single amino acid substitutions on protein stability (ΔΔG).")
        st.dataframe(df_mutations, use_container_width=True)
        
        if not df_mutations.empty:
            fig_ddg = px.bar(df_mutations, x="Mutation", y="Predicted_ddG_kcal_mol", 
                         color="Predicted_Effect", title="Predicted Stability Changes (ΔΔG)",
                         labels={"Predicted_ddG_kcal_mol": "ΔΔG (kcal/mol)"},
                         color_discrete_map={
                             "Destabilizing": "orangered",
                             "Stabilizing": "mediumseagreen",
                             "Neutral": "lightslategrey"
                         }, height=400)
            fig_ddg.add_hline(y=0, line_dash="dash", line_color="black")
            st.plotly_chart(fig_ddg, use_container_width=True)

            # Distribution of Predicted Effects
            effect_counts = df_mutations['Predicted_Effect'].value_counts().reset_index()
            effect_counts.columns = ['Effect', 'Count']
            fig_effect_dist = px.pie(effect_counts, values='Count', names='Effect', 
                                     title="Distribution of Predicted Mutational Effects",
                                     color_discrete_map={"Destabilizing": "orangered", "Stabilizing": "mediumseagreen", "Neutral": "lightslategrey"})
            st.plotly_chart(fig_effect_dist, use_container_width=True)

            # Hotspot prediction (mock)
            hotspots = df_mutations[np.abs(df_mutations['Predicted_ddG_kcal_mol']) > 1.5] # Arbitrary threshold for hotspot
            if not hotspots.empty:
                st.markdown("##### Potential Stability Hotspots (Mock |ΔΔG| > 1.5)")
                st.dataframe(hotspots[['Mutation', 'Predicted_ddG_kcal_mol', 'Predicted_Effect']].sort_values(by='Predicted_ddG_kcal_mol', key=abs, ascending=False), use_container_width=True)
            
        
        st.markdown("---")
        st.markdown("_Note: Mutational effects are mock-generated. Real analysis requires tools like FoldX, Rosetta, or specialized predictors._")

    with tab_map["DYN"]:
        st.subheader("Molecular Dynamics Simulation Insights")
        seq_len_dyn = data.get('length', 100)
        residue_indices = np.arange(1, seq_len_dyn + 1)

        # Mock RMSF data
        rmsf_mock_data = np.abs(np.random.normal(loc=0.8, scale=0.4, size=seq_len_dyn)) + \
                         0.5 * np.sin(residue_indices / (seq_len_dyn/10))**2 + \
                         np.random.uniform(0, 0.3, seq_len_dyn) # Add some noise
        rmsf_mock_data = np.clip(rmsf_mock_data, 0.2, 3.0)
        
        fig_rmsf = go.Figure(data=[go.Scatter(x=list(residue_indices), y=list(rmsf_mock_data), mode='lines', name='RMSF (Å)',
                                             line=dict(color='royalblue'))])
        fig_rmsf.update_layout(
            title="Residue Fluctuation (RMSF) Profile",
            xaxis_title="Residue Index",
            yaxis_title="RMSF (Å)",
            height=350
        )
        st.plotly_chart(fig_rmsf, use_container_width=True)
        st.markdown("RMSF plot indicates regions of higher flexibility (larger RMSF values) within the protein structure during a simulated timeframe.")

        # Mock RMSD data
        sim_time_ns = np.linspace(0, 100, 200) # 100 ns simulation, 200 frames
        rmsd_mock_data = 1.0 + 0.5 * np.sin(sim_time_ns / 20) + np.random.normal(0, 0.15, len(sim_time_ns))
        rmsd_mock_data[0] = 0 # Start at 0
        rmsd_mock_data = np.abs(rmsd_mock_data) # Ensure positive

        fig_rmsd = go.Figure(data=[go.Scatter(x=list(sim_time_ns), y=list(rmsd_mock_data), mode='lines', name='RMSD (Å)',
                                             line=dict(color='firebrick'))])
        fig_rmsd.update_layout(
            title="Protein Backbone RMSD Over Simulation Time",
            xaxis_title="Time (ns)",
            yaxis_title="RMSD (Å)",
            height=350
        )
        st.plotly_chart(fig_rmsd, use_container_width=True)
        st.markdown("RMSD plot shows the deviation of the protein structure from its initial conformation over time. A stable RMSD suggests the protein has reached equilibrium.")

        # Mock Radius of Gyration (Rg) over time
        rg_mock_data = rmsd_mock_data * random.uniform(1.5, 2.5) + random.uniform(10,15) # Scale RMSD to mock Rg
        rg_mock_data = np.clip(rg_mock_data, 5, 50)
        fig_rg_time = go.Figure(data=[go.Scatter(x=list(sim_time_ns), y=list(rg_mock_data), mode='lines', name='Rg (Å)',
                                             line=dict(color='green'))])
        fig_rg_time.update_layout(
            title="Radius of Gyration (Rg) Over Simulation Time",
            xaxis_title="Time (ns)",
            yaxis_title="Rg (Å)",
            height=350
        )
        st.plotly_chart(fig_rg_time, use_container_width=True)
        st.markdown("Rg plot shows changes in protein compactness over the simulation.")

        # Mock Number of Hydrogen Bonds over time
        hbond_mock_data = np.random.randint(seq_len_dyn // 2, seq_len_dyn * 1.5, size=len(sim_time_ns)) + (np.sin(sim_time_ns / 15) * seq_len_dyn * 0.1).astype(int)
        fig_hbond_time = px.line(x=sim_time_ns, y=hbond_mock_data, title="Number of Intra-Protein H-Bonds Over Time", labels={'x':'Time (ns)', 'y':'H-Bond Count'})
        st.plotly_chart(fig_hbond_time, use_container_width=True)
        st.markdown("Fluctuations in H-bonds can indicate conformational changes or (un)folding events.")
        
        st.markdown("---")
        st.markdown("_Note: MD simulation data is mock-generated. Real simulations require specialized software (GROMACS, Amber, etc.) and significant computational resources._")

    with tab_map["LIG"]:
        st.subheader("Ligand Binding Site Prediction")
        sequence_length = data.get('length', 100)
        mock_pockets = generate_mock_ligand_pockets(sequence_length, num_pockets=random.randint(1,4))

        if not mock_pockets:
            st.info("No distinct ligand binding pockets predicted.")
        else:
            st.write("Predicted ligand binding pockets and their characteristics:")
            for pocket in mock_pockets:
                with st.expander(f"{pocket['pocket_id']} (Druggability: {pocket['druggability_score']}) - Targets {pocket['target_ligand_type']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(label="Pocket Volume (Å³)", value=f"{pocket['volume_A3']:.1f}")
                    with col2:
                        st.metric(label="Druggability Score", value=f"{pocket['druggability_score']:.2f}")
                    st.markdown(f"**Key Residues:** `{pocket['residues']}`")
            
            df_pockets = pd.DataFrame(mock_pockets)
            if not df_pockets.empty:
                fig_druggability = px.bar(df_pockets, x="pocket_id", y="druggability_score",
                                      title="Druggability Scores of Predicted Pockets",
                                      color="druggability_score",
                                      color_continuous_scale=px.colors.sequential.Aggrnyl,
                                      labels={"pocket_id": "Pocket ID", "druggability_score": "Druggability Score"})
                st.plotly_chart(fig_druggability, use_container_width=True)

                # Pharmacophore features (mock)
                st.markdown("##### Mock Pharmacophore Features for Top Pocket")
                top_pocket = df_pockets.sort_values(by="druggability_score", ascending=False).iloc[0]
                st.markdown(f"**{top_pocket['pocket_id']}:**")
                st.markdown(f"- H-Bond Donors: {random.randint(1,5)}")
                st.markdown(f"- H-Bond Acceptors: {random.randint(1,5)}")
                st.markdown(f"- Hydrophobic Contacts: {random.randint(3,10)}")
                st.markdown(f"- Positive Ionizable: {random.randint(0,2)}")
                st.markdown(f"- Negative Ionizable: {random.randint(0,2)}")

        st.markdown("---")
        st.markdown("_Note: Ligand binding site data is mock-generated. Real analysis uses tools like CASTp, fpocket, AutoDock Vina, or commercial software._")

    with tab_map["EVO"]:
        st.subheader("Evolutionary Trace Analysis")
        seq_len_evo = data.get('length', 100)
        residue_indices_evo = np.arange(1, seq_len_evo + 1)

        # Mock conservation scores (1-9, 9 is highly conserved)
        # Simulate some conserved regions and some variable regions
        conservation_scores = np.random.randint(1, 5, size=seq_len_evo) # Mostly variable
        num_conserved_patches = random.randint(1, max(1, seq_len_evo // 50))
        for _ in range(num_conserved_patches):
            patch_start = random.randint(0, seq_len_evo - 10)
            patch_len = random.randint(5, 15)
            conservation_scores[patch_start : patch_start + patch_len] = np.random.randint(7, 10, size=patch_len)
        
        df_conservation = pd.DataFrame({
            "Residue Index": residue_indices_evo,
            "Conservation Score": conservation_scores
        })

        fig_cons = px.bar(df_conservation, x="Residue Index", y="Conservation Score",
                          title="Per-Residue Evolutionary Conservation Score",
                          color="Conservation Score",
                          color_continuous_scale=px.colors.diverging.RdYlBu_r, # Red (variable) to Blue (conserved)
                          labels={"Conservation Score": "Score (1=Variable, 9=Conserved)"})
        fig_cons.update_layout(height=350)
        st.plotly_chart(fig_cons, use_container_width=True)

        highly_conserved_residues = df_conservation[df_conservation["Conservation Score"] >= 7]["Residue Index"].tolist() # Adjusted threshold for example
        highly_variable_residues = df_conservation[df_conservation["Conservation Score"] <= 3]["Residue Index"].tolist() # Adjusted threshold

        # Conservation Score Distribution
        fig_cons_dist = px.histogram(df_conservation, x="Conservation Score", nbins=9, title="Distribution of Conservation Scores", marginal="rug")
        st.plotly_chart(fig_cons_dist, use_container_width=True)

        st.markdown("##### Conservation Summary")
        col_cons1, col_cons2 = st.columns(2)
        with col_cons1:
            st.metric(label="Average Conservation Score", value=f"{df_conservation['Conservation Score'].mean():.2f}")
        with col_cons2:
            st.metric(label="Std Dev of Conservation", value=f"{df_conservation['Conservation Score'].std():.2f}")
        if highly_conserved_residues: # Check if list is not empty
            st.success(f"**Highly Conserved Residues (Score >= 8):** {len(highly_conserved_residues)} residues. Example indices: {', '.join(map(str, random.sample(highly_conserved_residues, k=min(5, len(highly_conserved_residues)))))}")
        if highly_variable_residues:
            st.warning(f"**Highly Variable Residues (Score <= 2):** {len(highly_variable_residues)} residues. Example indices: {', '.join(map(str, random.sample(highly_variable_residues, k=min(5, len(highly_variable_residues)))))}")
        
        st.markdown("Conservation scores highlight residues important for structure or function. Highly conserved residues are often critical, while variable regions might tolerate more mutations or be involved in species-specific interactions.")
        st.markdown("---")
        st.markdown("_Note: Evolutionary conservation data is mock-generated. Real analysis uses tools like ConSurf, Rate4Site, based on multiple sequence alignments (MSAs)._")

    with tab_map["SURF"]:
        st.subheader("Surface Properties Analysis")
        sequence_length = data.get('length', 100)
        df_surface = generate_mock_surface_properties(sequence_length)

        st.write("Predicted per-residue surface properties:")
        st.dataframe(df_surface.head(), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig_hydro = px.line(df_surface, x="residue_index", y="hydrophobicity_kyte_doolittle",
                                title="Hydrophobicity Profile (Kyte-Doolittle)",
                                labels={"residue_index": "Residue Index", "hydrophobicity_kyte_doolittle": "Hydrophobicity"})
            fig_hydro.add_hline(y=0, line_dash="dash", line_color="grey")
            st.plotly_chart(fig_hydro, use_container_width=True)
        
        with col2:
            fig_sasa = px.line(df_surface, x="residue_index", y="solvent_accessibility_mock_percent",
                               title="Solvent Accessibility Profile (Mock %)",
                               labels={"residue_index": "Residue Index", "solvent_accessibility_mock_percent": "Accessibility (%)"},
                               color_discrete_sequence=['coral'])
            st.plotly_chart(fig_sasa, use_container_width=True)

        st.markdown("##### Mock Electrostatic Potential")
        st.info("Below is a conceptual representation. Real electrostatic potential requires 3D structure and specialized software (e.g., APBS, Delphi).")
        # Simplified bar chart for "electrostatic potential"
        fig_electro = px.bar(df_surface.iloc[::max(1, sequence_length//50)], x="residue_index", y="electrostatic_potential_mock", # Sample points for bar chart
                             title="Mock Electrostatic Potential along Sequence",
                             labels={"residue_index": "Residue Index", "electrostatic_potential_mock": "Mock Potential"},
                             color="electrostatic_potential_mock",
                             color_continuous_scale=px.colors.diverging.RdBu)
        st.plotly_chart(fig_electro, use_container_width=True)

        st.markdown("##### Surface Patch Analysis (Mock)")
        st.markdown(f"- Largest Hydrophobic Patch (Mock): Centered around residue {random.randint(1, sequence_length)}, Area ~{random.randint(200,800)} Å²")
        st.markdown(f"- Largest Positive Patch (Mock): Near residues {random.sample(range(1,sequence_length+1), 3)}, Potential for DNA/RNA binding.")
        st.markdown(f"- Largest Negative Patch (Mock): Near residues {random.sample(range(1,sequence_length+1), 3)}, Potential for cation binding.")

        st.markdown("##### Epitope Prediction (Mock)")
        num_epitopes = random.randint(1,5)
        for i in range(num_epitopes):
            st.markdown(f"- Potential Epitope {i+1}: Residues {random.randint(1,sequence_length-10)}-{random.randint(1,sequence_length-10)+random.randint(5,10)} (Accessibility Score: {random.uniform(0.6,0.95):.2f})")
        st.markdown("---")
        st.markdown("_Note: Surface property data is mock-generated. Real analysis requires 3D structure and tools like PyMOL, VMD, APBS, NACCESS._")

    with tab_map["COMP"]:
        st.subheader("Structural Comparison (vs. PDB)")
        df_comparison = generate_mock_structural_comparison(num_hits=random.randint(3,8))

        st.write("Top structural homologs found in a mock PDB search (lower RMSD is better):")
        st.dataframe(df_comparison, use_container_width=True,
                     column_config={
                         "RMSD_Angstrom": st.column_config.NumberColumn(format="%.2f Å"),
                         "Sequence_Identity_Percent": st.column_config.NumberColumn(format="%.1f %%")
                     })

        if not df_comparison.empty:
            fig_rmsd_comp = px.bar(df_comparison, x="PDB_ID", y="RMSD_Angstrom",
                               title="RMSD to Structural Homologs",
                               color="Sequence_Identity_Percent",
                               labels={"PDB_ID": "Homolog PDB ID", "RMSD_Angstrom": "RMSD (Å)"},
                               hover_data=["Description", "Sequence_Identity_Percent"])
            st.plotly_chart(fig_rmsd_comp, use_container_width=True)

            # Scatter plot of RMSD vs Sequence Identity
            fig_rmsd_vs_id = px.scatter(df_comparison, x="Sequence_Identity_Percent", y="RMSD_Angstrom",
                                        text="PDB_ID", title="RMSD vs. Sequence Identity",
                                        labels={"Sequence_Identity_Percent": "Sequence Identity (%)", "RMSD_Angstrom": "RMSD (Å)"},
                                        hover_data=["Description"])
            fig_rmsd_vs_id.update_traces(textposition='top center')
            st.plotly_chart(fig_rmsd_vs_id, use_container_width=True)

        st.markdown("---")
        st.markdown("_Note: Structural comparison data is mock-generated. Real analysis uses tools like DALI, TM-align, CE, or FoldSeek against the PDB database._")

    with tab_map["QUAL"]:
        st.subheader("Advanced Quality Assessment")
        sequence_length = data.get('length', 100)
        quality_metrics = generate_mock_quality_assessment(sequence_length)

        st.markdown("##### Ramachandran Plot Analysis (Mock Summary)")
        rama_data = {
            "Region": ["Favored", "Allowed", "Outliers"],
            "Percentage": [
                quality_metrics["ramachandran_favored_percent"],
                quality_metrics["ramachandran_allowed_percent"],
                quality_metrics["ramachandran_outliers_percent"]
            ]
        }
        df_rama = pd.DataFrame(rama_data)
        fig_rama = px.pie(df_rama, values="Percentage", names="Region", title="Ramachandran Plot Regions",
                          color_discrete_map={"Favored": "mediumseagreen", "Allowed": "gold", "Outliers": "orangered"})
        st.plotly_chart(fig_rama, use_container_width=True)

        st.markdown("##### Other Quality Metrics (Mock)")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Clashscore", f"{quality_metrics['clashscore']:.2f}", help="Lower is better. Measures steric clashes.")
        with col2:
            st.metric("Avg. Bond Length Deviation", f"{quality_metrics['avg_bond_length_deviation_percent']:.2f}%", help="Deviation from ideal bond lengths.")
        with col3:
            st.metric("Avg. Bond Angle Deviation", f"{quality_metrics['avg_bond_angle_deviation_degrees']:.1f}°", help="Deviation from ideal bond angles.")
        st.metric("Overall GDT_TS (Mock)", f"{quality_metrics['overall_gdt_ts_mock']:.1f}", help="Global Distance Test Total Score. Higher is better (0-100).")
        
        st.markdown("##### Rotamer Analysis (Mock Summary)")
        rotamer_data_qual = generate_mock_rotamer_analysis(sequence_length) # Use existing function
        st.markdown(f"- Favored Rotamers: {rotamer_data_qual['Favored_Rotamers_Percent']:.1f}%")
        st.markdown(f"- Allowed Rotamers: {rotamer_data_qual['Allowed_Rotamers_Percent']:.1f}%")
        st.markdown(f"- Outlier Rotamers: {rotamer_data_qual['Outlier_Rotamers_Percent']:.1f}% (Lower is better)")

        st.markdown("##### Packing Quality (Mock)")
        st.markdown(f"- Packing Defects Score (Mock): {random.uniform(0.1, 0.5):.2f} (Lower is better)")
        st.markdown(f"- Number of Internal Voids > 10Å³ (Mock): {random.randint(0,3)}")
        st.markdown("---")
        st.markdown("_Note: Quality assessment data is mock-generated. Real analysis uses tools like PROCHECK, MolProbity, QMEAN, or validation servers._")

    with tab_map["DRUG"]:
        st.subheader("Druggability Analysis")
        sequence_length = data.get('length', 100)
        # We can reuse the ligand pockets generated for the LIG tab
        if 'ligand_pockets_cache' not in st.session_state:
            st.session_state.ligand_pockets_cache = generate_mock_ligand_pockets(sequence_length, num_pockets=random.randint(1,4))
        
        mock_pockets_for_drug = st.session_state.ligand_pockets_cache

        if not mock_pockets_for_drug:
            st.info("No distinct pockets identified for druggability assessment.")
        else:
            st.write("Druggability assessment of predicted binding pockets:")
            total_druggable_score = 0
            num_pockets = len(mock_pockets_for_drug)
            
            for pocket in mock_pockets_for_drug:
                total_druggable_score += pocket['druggability_score']
                with st.expander(f"{pocket['pocket_id']} - Druggability: {pocket['druggability_score']:.2f}"):
                    st.metric(label="Druggability Score", value=f"{pocket['druggability_score']:.2f}",
                              help="A score from 0 (undruggable) to 1 (highly druggable).")
                    st.markdown(f"**Target Ligand Type:** {pocket['target_ligand_type']}")
                    st.markdown(f"**Pocket Volume:** {pocket['volume_A3']:.1f} Å³")
                    st.markdown(f"**Key Residues:** `{pocket['residues']}`")

            avg_druggability = total_druggable_score / num_pockets if num_pockets > 0 else 0
            
            df_pockets = pd.DataFrame(mock_pockets_for_drug)
            if df_pockets.empty:
                st.info("No pocket data available to plot druggability scores.")
            else:
                fig_druggability = px.bar(df_pockets, x="pocket_id", y="druggability_score",
                                          title="Druggability Scores of Predicted Pockets",
                                          color="druggability_score",
                                          color_continuous_scale=px.colors.sequential.Aggrnyl,
                                          labels={"pocket_id": "Pocket ID", "druggability_score": "Druggability Score"})
                st.plotly_chart(fig_druggability, use_container_width=True)
                st.markdown("---") # Moved after the plot if plot is shown
                st.markdown(f"##### Overall Druggability Assessment")
                if avg_druggability > 0.7:
                    st.success(f"High overall druggability potential (Average Score: {avg_druggability:.2f}). Several promising pockets identified.")
                elif avg_druggability > 0.4:
                    st.warning(f"Moderate overall druggability potential (Average Score: {avg_druggability:.2f}). Some pockets may be tractable.")
                else:
                    st.error(f"Low overall druggability potential (Average Score: {avg_druggability:.2f}). Targeting this protein may be challenging.")

            st.markdown("##### ADMET Properties for Top Pocket (Mock)")
            if mock_pockets_for_drug:
                top_pocket_admet = sorted(mock_pockets_for_drug, key=lambda x: x['druggability_score'], reverse=True)[0]
                st.markdown(f"**For {top_pocket_admet['pocket_id']}:**")
                st.markdown(f"- Lipinski Rule of 5 Violations (Conceptual): {random.randint(0,2)}")
                st.markdown(f"- Predicted Toxicity Class (Conceptual): {random.choice(['Low', 'Medium', 'High'])}")
                st.markdown(f"- Predicted Solubility (Conceptual): {random.choice(['Good', 'Moderate', 'Poor'])}")

        st.markdown("---")
        st.markdown("_Note: Druggability data is based on mock pocket predictions. Real analysis uses specialized software and considers factors like pocket geometry, hydrophobicity, and known drug targets._")

    with tab_map["CONS"]:
        st.subheader("Residue Conservation Scores")
        # This will be similar to EVO tab, but let's make it slightly different for variety
        # or if user intends a different focus for "CONS" vs "EVO"
        seq_len_cons = data.get('length', 100)
        residue_indices_cons = np.arange(1, seq_len_cons + 1)

        # Mock conservation scores (e.g., from a different algorithm or perspective)
        # Let's use a 0-1 scale for this one, where 1 is highly conserved.
        conservation_values = np.random.beta(a=2, b=5, size=seq_len_cons) # Skewed towards less conserved
        # Add some conserved patches
        num_patches = random.randint(1, max(1, seq_len_cons // 40))
        for _ in range(num_patches):
            patch_start = random.randint(0, seq_len_cons - 15)
            patch_len = random.randint(8, 20)
            conservation_values[patch_start : patch_start + patch_len] = np.random.beta(a=5, b=2, size=patch_len) # Skewed towards conserved
        conservation_values = np.clip(conservation_values, 0, 1)

        df_cons_scores = pd.DataFrame({
            "Residue Index": residue_indices_cons,
            "Conservation (0-1)": conservation_values
        })

        fig_cons_line = px.line(df_cons_scores, x="Residue Index", y="Conservation (0-1)",
                                title="Residue Conservation Profile (0=Variable, 1=Conserved)",
                                labels={"Conservation (0-1)": "Conservation Score"})
        fig_cons_line.update_traces(line_color='darkcyan')
        fig_cons_line.update_layout(height=350)
        st.plotly_chart(fig_cons_line, use_container_width=True)

        conserved_threshold = 0.8
        num_highly_conserved = df_cons_scores[df_cons_scores["Conservation (0-1)"] >= conserved_threshold].shape[0]
        st.metric(label=f"Residues with Conservation >= {conserved_threshold}", value=f"{num_highly_conserved} ({num_highly_conserved/seq_len_cons*100:.1f}%)")
        
        # Conservation Entropy
        cons_counts = np.histogram(conservation_values, bins=10, range=(0,1))[0]
        cons_probs = cons_counts / seq_len_cons
        cons_entropy = entropy(cons_probs, base=2)
        st.metric(label="Conservation Score Entropy (bits)", value=f"{cons_entropy:.3f}")

        st.markdown(f"**Structurally Important Conserved Residues (Mock):** Residues {random.randint(1, seq_len_cons//2)}, {random.randint(seq_len_cons//2 + 1, seq_len_cons)} are predicted to be in the core and highly conserved.")

        st.markdown("This plot shows the evolutionary conservation of each residue. Highly conserved residues (score near 1) are often critical for protein structure or function.")
        st.markdown("---")
        st.markdown("_Note: Conservation scores are mock-generated. Real analysis relies on Multiple Sequence Alignments (MSAs) and tools like ConSurf or Rate4Site._")

    with tab_map["ALLO"]:
        st.subheader("Allosteric Site Prediction")
        sequence_length = data.get('length', 100)
        mock_allo_sites = generate_mock_allosteric_sites(sequence_length, num_sites=random.randint(1,3)) # Ensure at least 1 site is usually predicted

        if not mock_allo_sites:
            st.info("No distinct allosteric sites predicted for this protein.")
        else:
            st.write("Predicted potential allosteric sites and their characteristics:")
            for site in mock_allo_sites:
                with st.expander(f"{site['site_id']} - {site['site_type_mock']} (Score: {site['prediction_score']:.2f})"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(label="Prediction Score", value=f"{site['prediction_score']:.2f}")
                    with col2:
                        st.metric(label="Mock Pocket Volume (Å³)", value=f"{site['pocket_volume_A3_mock']:.1f}")
                    with col3:
                        st.metric(label="Avg. Conservation (Mock)", value=f"{site['avg_conservation_mock']:.2f}")
                    
                    st.markdown(f"**Predicted Site Type:** {site['site_type_mock']}")
                    st.markdown(f"**Key Residues:** `{site['residues']}`")
            
            if mock_allo_sites: # Ensure there's data for plots
                df_allo = pd.DataFrame(mock_allo_sites)

                if len(mock_allo_sites) > 1:
                    type_counts = df_allo['site_type_mock'].value_counts().reset_index()
                    type_counts.columns = ['Site Type', 'Count']
                    fig_allo_type_dist = px.pie(type_counts, values='Count', names='Site Type', 
                                                title="Distribution of Predicted Allosteric Site Types",
                                                color_discrete_sequence=px.colors.qualitative.Pastel)
                    st.plotly_chart(fig_allo_type_dist, use_container_width=True)

                fig_allo_score = px.bar(df_allo, x="site_id", y="prediction_score",
                                          title="Prediction Scores of Potential Allosteric Sites",
                                          color="prediction_score",
                                          color_continuous_scale=px.colors.sequential.Tealgrn,
                                          labels={"site_id": "Site ID", "prediction_score": "Prediction Score"})
                st.plotly_chart(fig_allo_score, use_container_width=True)

                st.markdown("##### Predicted Allosteric Mechanism (Conceptual)")
                st.markdown(f"The predicted allosteric sites might modulate protein function via **{random.choice(['conformational change propagation', 'pocket opening/closing dynamics', 'interface stabilization/destabilization'])}**.")
        
        st.markdown("---")
        st.markdown("_Note: Allosteric site data is mock-generated. Real analysis uses tools like AlloPred, PARS, or specialized MD simulations._")

    with tab_map["MEMB"]:
        st.subheader("Membrane Protein Analysis")
        sequence_length = data.get('length', 100)
        membrane_data = generate_mock_membrane_topology(sequence_length)

        if not membrane_data["is_membrane_protein"]:
            st.info(membrane_data["topology_summary"])
        else:
            st.success(membrane_data["topology_summary"])
            st.metric(label="Predicted Transmembrane Helices (TMHs)", value=membrane_data["num_helices"])

            if membrane_data["helices"]:
                df_tmh = pd.DataFrame(membrane_data["helices"])
                st.markdown("##### Predicted TMH Segments:")
                st.dataframe(df_tmh, use_container_width=True)

                # Visualization of TMHs
                fig_tmh = go.Figure()
                for i, helix in enumerate(membrane_data["helices"]):
                    fig_tmh.add_trace(go.Scatter(
                        x=[helix["start"], helix["end"]],
                        y=[i+1, i+1],
                        mode="lines+markers",
                        line=dict(width=10, color=px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]),
                        name=helix["id"]
                    ))
                fig_tmh.update_layout(
                    title="Transmembrane Helix Topology",
                    xaxis_title="Residue Index",
                    yaxis_title="TMH Number",
                    yaxis=dict(tickvals=list(range(1, len(membrane_data["helices"]) + 1)), 
                               ticktext=[f"TMH{j+1}" for j in range(len(membrane_data["helices"]))]),
                    height=max(300, 50 * len(membrane_data["helices"])),
                    showlegend=False
                )
                fig_tmh.update_xaxes(range=[0, sequence_length + 1])
                st.plotly_chart(fig_tmh, use_container_width=True)

                st.markdown("##### Additional Membrane Properties (Mock)")
                for i, helix_data in enumerate(membrane_data["helices"]):
                    st.markdown(f"- **{helix_data['id']}:** Avg. Hydrophobicity (Mock): {random.uniform(1.5, 2.5):.2f}, Predicted Lipid Exposure (Mock): {random.uniform(0.4, 0.9):.2f}")
                
                st.markdown(f"**Re-entrant Loops Predicted (Mock):** {random.randint(0,1)}")
                st.markdown(f"**Overall Orientation in Membrane (Mock):** N-terminus {membrane_data['n_terminus_location']}, C-terminus {membrane_data['c_terminus_location']} based on positive-inside rule.")


        st.markdown("---")
        st.markdown("_Note: Membrane topology data is mock-generated. Real predictions use algorithms like TMHMM, Phobius, or DeepTMHMM._")

    with tab_map["FOLD"]:
        st.subheader("Folding Pathway Insights")
        sequence_length = data.get('length', 100)
        folding_insights = generate_mock_folding_pathway_insights(sequence_length)

        st.markdown("Conceptual insights into the predicted protein folding pathway:")
        for insight in folding_insights:
            st.markdown(f"- {insight}")
        
        st.markdown("##### Visual Concept: Folding Energy Landscape (Mock)")
        # Mock data for a simple energy landscape
        x_landscape = np.linspace(0, 10, 100)
        y_landscape = (np.sin(x_landscape) * 5 + 
                       np.cos(x_landscape*0.3)*3 - 
                       x_landscape*0.8 +  # General trend towards folded state
                       10 + np.random.normal(0,0.5,100)) 
        y_landscape -= np.min(y_landscape) # Normalize to start near 0

        fig_landscape = go.Figure(data=[go.Scatter(x=x_landscape, y=y_landscape, mode='lines', line=dict(color='purple'))])
        fig_landscape.update_layout(title="Conceptual Folding Energy Landscape",
                                    xaxis_title="Reaction Coordinate (Unfolded -> Folded)",
                                    yaxis_title="Relative Free Energy (Mock Units)",
                                    height=350)
        st.plotly_chart(fig_landscape, use_container_width=True)
        st.markdown("The landscape illustrates a simplified path from unfolded to folded state, possibly via intermediates (local minima).")

        st.markdown("##### Key Folding Parameters (Mock)")
        st.markdown(f"- **Predicted Contact Order (Mock):** {random.uniform(0.1, 0.3) * sequence_length:.1f} (Higher means more non-local contacts)")
        st.markdown(f"- **Predicted Folding Rate Class (Mock):** {random.choice(['Fast (<1s)', 'Medium (1s-1min)', 'Slow (>1min)'])}")
        st.markdown(f"- **Chaperone Dependency Score (Mock):** {random.uniform(0.1, 0.8):.2f} (Higher means more likely to need chaperones)")
        st.markdown("---")
        st.markdown("_Note: Folding pathway insights are highly conceptual and mock-generated. Real folding studies involve complex experiments and simulations (e.g., MD, kinetic studies)._")

    with tab_map["PPI_IF"]:
        st.subheader("Protein-Protein Interface Analysis")
        sequence_length = data.get('length', 100) # Ensure sequence_length is defined
        # Assuming one primary mock interface for simplicity here
        mock_interface = generate_mock_ppi_interface_data(sequence_length, partner_protein_id=f"PROT_{random.randint(1000,9999)}")
        
        st.markdown(f"##### Predicted Interface with {mock_interface['partner_protein_id']}")
        st.metric(label="Interface Residues", value=f"{len(mock_interface['interface_residues'].split(','))} residues")
        st.metric(label="Buried Surface Area (Mock)", value=f"{mock_interface['buried_surface_area_A2_mock']} Å²")
        st.metric(label="Interface Hydrophobicity (Mock)", value=f"{mock_interface['interface_hydrophobicity_score_mock']:.2f}")
        st.metric(label="Predicted Binding Energy (Mock)", value=f"{mock_interface['predicted_binding_energy_kcal_mol_mock']:.1f} kcal/mol")
        with st.expander("View Interface Residues"):
            st.code(mock_interface['interface_residues'], language=None)
        
        st.markdown("##### Interface Properties (Mock)")
        st.markdown(f"- **Interface Conservation Score (Mock):** {random.uniform(0.5, 0.9):.2f}")
        st.markdown(f"- **Shape Complementarity Score (Mock):** {random.uniform(0.6, 0.8):.2f}")
        hotspot_residues_ppi = sorted(random.sample(mock_interface['interface_residues'].split(','), k=min(len(mock_interface['interface_residues'].split(',')), random.randint(2,5))))
        st.markdown(f"- **Predicted Hotspot Residues at Interface (Mock):** {', '.join(hotspot_residues_ppi)}")
        st.markdown("---")
        st.markdown("_Note: PPI interface data is mock-generated. Real analysis uses tools like PISA, InterProSurf, or docking simulations followed by interface characterization._")

    with tab_map["NMR"]:
        st.subheader("NMR Spectra Simulation (Mock)")
        sequence_length = data.get('length', 100)
        df_nmr = generate_mock_nmr_spectra_data(sequence_length)
        
        fig_nmr = px.line(df_nmr, x="Chemical_Shift_ppm", y="Intensity_Arbitrary",
                          title="Mock 1D Proton NMR Spectrum",
                          labels={"Chemical_Shift_ppm": "Chemical Shift (ppm)", "Intensity_Arbitrary": "Intensity"})
        fig_nmr.update_layout(height=400)
        st.plotly_chart(fig_nmr, use_container_width=True)
        st.markdown(f"**Key Peaks (Mock):** Broad peak around {random.uniform(7,8.5):.1f} ppm (aromatic/amide), sharp peaks around {random.uniform(1,3):.1f} ppm (aliphatic).")
        st.markdown("This is a highly simplified mock 1D spectrum. Real NMR analysis involves complex multi-dimensional experiments and assignments to correlate signals with specific atoms in the 3D structure.")
        st.markdown("---")
        st.markdown("_Note: NMR spectra are mock-generated. Real NMR requires experimental data and specialized processing software._")

    with tab_map["SAX"]:
        st.subheader("SAXS Profile Analysis (Mock)")
        df_saxs, rg_saxs_mock = generate_mock_saxs_profile() # Re-use existing function
        
        fig_saxs_tab = px.line(df_saxs, x="q_Angstrom_inv", y="Intensity_I_q_arbitrary_units", log_y=True,
                           title="Predicted SAXS Profile (Mock)",
                           labels={"q_Angstrom_inv": "q (Å⁻¹)", "Intensity_I_q_arbitrary_units": "Intensity (log scale)"})
        fig_saxs_tab.update_layout(height=400)
        st.plotly_chart(fig_saxs_tab, use_container_width=True)
        
        st.metric(label="Radius of Gyration (Rg) from SAXS (Mock)", value=f"{rg_saxs_mock:.1f} Å")
        st.markdown(f"**Maximum Particle Dimension (Dmax) (Mock):** {rg_saxs_mock * random.uniform(2.5, 3.5):.1f} Å")
        st.markdown("The SAXS profile provides information about the protein's overall shape, size, and degree of compactness in solution. The Radius of Gyration (Rg) is a key parameter derived from SAXS data.")
        st.markdown("---")
        st.markdown("_Note: SAXS profiles are mock-generated. Real SAXS analysis involves experimental scattering data and modeling._")

    with tab_map["CRF"]:
        st.subheader("Cryo-EM Map Fitting Analysis (Mock)")
        cryo_fit_data = generate_mock_cryoem_fit() # Re-use existing function
        
        st.metric(label="Simulated Cryo-EM Map Resolution (Mock)", value=f"{cryo_fit_data['Resolution_Angstrom_Mock']} Å")
        st.metric(label="Cross-Correlation Score with Map (Mock)", value=f"{cryo_fit_data['Cross_Correlation_Score_Mock']:.3f}")
        st.markdown(f"**Map Segmentation Quality (Mock):** {cryo_fit_data['Map_Segmentation_Quality_Mock']}")
        st.markdown(f"**Model-to-Map FSC at 0.143 (Mock):** {random.uniform(0.4, 0.8):.2f}")
        st.markdown("This section assesses how well the predicted atomic model fits into an experimental Cryo-Electron Microscopy (Cryo-EM) density map, if available. Higher cross-correlation and FSC scores indicate a better fit.")
        st.markdown("---")
        st.markdown("_Note: Cryo-EM fit data is mock-generated. Real analysis requires an experimental density map and fitting software._")

    with tab_map["XTP"]:
        st.subheader("Crystallization Propensity Analysis (Mock)")
        crystallization_data = generate_mock_crystallization_propensity() # Re-use existing function
        st.metric(label="Overall Crystallization Propensity Score (Mock)", value=f"{crystallization_data['Overall_Propensity_Score_Mock']:.2f}")
        st.markdown(f"**Number of Low Surface Entropy Patches (Mock):** {crystallization_data['Number_of_Low_Entropy_Patches_Mock']}")
        st.markdown(f"**Largest Hydrophobic Patch Area (Mock):** {crystallization_data['Largest_Hydrophobic_Patch_Area_A2_Mock']:.1f} Å²")
        st.markdown("Estimates the likelihood of a protein to form well-ordered crystals, based on surface properties like hydrophobicity, charge distribution, and conformational homogeneity. This is crucial for X-ray crystallography.")
        st.markdown("---")
        st.markdown("_Note: Crystallization propensity is mock-generated. Real prediction uses tools like XtalPred, SERp, or Surface Entropy Reduction analysis._")

    with tab_map["FUNC"]:
        st.subheader("Functional Prediction & Annotation")
        sequence_length = data.get('length', 100)
        domains_data_for_func = data.get('domains', [])
        functional_data = generate_mock_functional_prediction_data(sequence_length, domains_data_for_func)

        st.markdown("##### Gene Ontology (GO) Term Predictions (Mock)")
        for go_category, terms in functional_data["go_terms"].items():
            st.markdown(f"**{go_category}:**")
            for term in terms:
                st.markdown(f"- {term}")
        
        st.markdown("---")
        st.markdown("##### Enzyme Commission (EC) Number Prediction (Mock)")
        if functional_data["ec_number"] != "N/A":
            st.success(f"**Predicted EC Number:** {functional_data['ec_number']}")
            st.markdown("Indicates potential enzymatic activity. The EC number classifies enzymes based on the chemical reactions they catalyze.")
        else:
            st.info("No specific EC number predicted (protein may not be an enzyme or prediction confidence is low).")

        st.markdown("---")
        st.markdown("##### Predicted Biological Pathways (Mock)")
        if functional_data["predicted_pathways_mock"]:
            st.markdown(f"Potentially involved in: {', '.join(functional_data['predicted_pathways_mock'])}")
        else:
            st.info("No specific pathways strongly predicted.")
        st.markdown("---")
        st.markdown("_Note: Functional predictions are mock-generated. Real annotation uses tools like InterProScan, eggNOG-mapper, BLAST against curated databases (UniProt, GO, KEGG)._")

    # This was originally tab5, now it's the last tab
    with tab_map["DATA"]:
        st.subheader("Detailed Prediction Data")
        
        # Create detailed dataframe
        detailed_df = pd.DataFrame({
            'Position': range(1, len(data['sequence']) + 1),
            'Residue': list(data['sequence']),
            'Secondary_Structure': data['secondary_structure'],
            'pLDDT_Score': np.round(data['plddt'], 2),
            'Confidence_Level': ['High' if x > 70 else 'Medium' if x > 50 else 'Low' for x in data['plddt']],
            'Disorder_Prediction': ['Disordered' if x else 'Ordered' for x in data['disorder']]
        })
        
        st.dataframe(detailed_df, use_container_width=True, height=400)
        
        # Sequence information
        st.subheader("Sequence Information")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Total Length:** {len(data['sequence'])} residues")
            st.write(f"**Molecular Weight:** ~{len(data['sequence']) * 110:.0f} Da")
            st.write(f"**Prediction Model:** {data['model_used']}")
            # Mock Isoelectric Point & Extinction Coefficient
            mock_pi = round(random.uniform(4.0, 10.0), 2)
            mock_ext_coeff = int(data['sequence'].count('W') * 5500 + data['sequence'].count('Y') * 1490 + data['sequence'].count('C') * 125 * (random.randint(0,data['sequence'].count('C')//2))) # Approximation
            st.write(f"**Isoelectric Point (pI) (Mock):** {mock_pi}")
            st.write(f"**Extinction Coefficient (Mock):** {mock_ext_coeff} M⁻¹cm⁻¹")

        # Amino Acid Composition
        st.subheader("Amino Acid Composition")
        aa_counts = pd.Series(list(data['sequence'])).value_counts().reset_index()
        aa_counts.columns = ['Amino Acid', 'Count']
        fig_aa_comp = px.bar(aa_counts, x='Amino Acid', y='Count', title="Amino Acid Composition", color='Amino Acid')
        st.plotly_chart(fig_aa_comp, use_container_width=True)
        
        with col2:
            st.write(f"**Analysis Date:** {data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            st.write(f"**Overall Quality:** {'High' if data['overall_confidence'] > 70 else 'Medium' if data['overall_confidence'] > 50 else 'Low'}")

else:
    # Welcome screen
    st.markdown("""
    ## 🔬 Welcome to AlphaFold Pro
    
    **Advanced Protein Structure Prediction Suite**
    
    This professional-grade application provides comprehensive protein structure analysis using state-of-the-art AI models.
    
    ### Features:
    - 🧬 **Secondary Structure Prediction** - Helices, sheets, and loops
    - 📊 **Confidence Scoring** - pLDDT-based reliability assessment  
    - 🎯 **Domain Analysis** - Functional region identification
    - 🔍 **Disorder Prediction** - Intrinsically disordered regions
    - 🤖 **AI-Powered Analysis** - Comprehensive structural insights
    - 📈 **Interactive Visualizations** - Professional charts and plots
    - 💾 **Multiple Export Formats** - JSON, CSV, PDB output
    
    ### Getting Started:
    1. Enter your Gemini API key in the sidebar
    2. Input your protein sequence (FASTA format recommended)
    3. Configure analysis parameters
    4. Click "Predict Structure" to begin analysis
    
    ### Sample Sequences Available:
    Use the "Random Protein" or "Complex Protein" buttons to generate test sequences.
    """)
    
    # Prediction history
    if st.session_state.prediction_history:
        st.subheader("📚 Recent Predictions")
        
        for i, pred in enumerate(reversed(st.session_state.prediction_history[-5:])):
            with st.expander(f"Prediction {len(st.session_state.prediction_history) - i} - {pred['timestamp'].strftime('%H:%M:%S')}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"Length: {pred['length']} AA")
                with col2:
                    st.write(f"Domains: {len(pred['domains'])}")
                with col3:
                    st.write(f"Confidence: {pred['overall_confidence']:.1f}")
                
                if st.button(f"Load Prediction {len(st.session_state.prediction_history) - i}", key=f"load_{i}"):
                    st.session_state.current_prediction = pred
                    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>AlphaFold Pro - Professional Protein Structure Prediction Suite</p>
    <p>⚠️ This is a demonstration application. Results are generated for educational purposes.</p>
    <p>For production use, integrate with actual AlphaFold or ColabFold backends.</p>
</div>
""", unsafe_allow_html=True)
