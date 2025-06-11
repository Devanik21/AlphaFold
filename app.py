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
    .metric-card {
        background: #f8f9fa;
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
    .status-running { background: #fff3cd; color: #856404; }
    .status-complete { background: #d4edda; color: #155724; }
    .status-error { background: #f8d7da; color: #721c24; }
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
        "Gemini 2.5 Flash Experimental": "gemini-2.5-flash",
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
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Structure Overview", 
        "🎯 AI Analysis", 
        "📈 Confidence Analysis",
        "🧬 Domain Architecture",
        "📋 Detailed Data"
    ])
    
    with tab1:
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
    
    with tab2:
        st.subheader("AI-Generated Structural Analysis")
        if 'ai_analysis' in data:
            st.markdown(data['ai_analysis'])
        else:
            st.info("AI analysis not available")
    
    with tab3:
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
    
    with tab4:
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
        else:
            st.info("No distinct domains identified in this protein")
    
    with tab5:
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
