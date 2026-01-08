"""
Interface Streamlit Professionnelle - Système de Triage Médical
================================================================
Interface moderne et interactive pour tester le système NLP
"""

import streamlit as st
import sys
import os
from datetime import datetime

# Setup path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from agents.data_loader.medical_data_loader import MedicalDataLoader
from agents.analyzer.nlp_analyzer_v3 import CompleteNLPAnalyzer
from agents.reasoner.medical_reasoner import MedicalReasoner
from agents.decider.decision_generator import DecisionGenerator

# Configuration de la page
st.set_page_config(
    page_title="🏥 Triage Médical Intelligent",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        text-align: center;
        margin-bottom: 3rem;
    }
    .stAlert {
        border-radius: 10px;
    }
    .symptom-card {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #1E88E5;
    }
    .disease-card {
        background-color: #FFF3E0;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #FF9800;
    }
    .urgency-high {
        background-color: #FFEBEE;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #F44336;
    }
    .urgency-medium {
        background-color: #FFF3E0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #FF9800;
    }
    .urgency-low {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
    }
    .nlp-step {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialisation session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.analyzer = None
    st.session_state.reasoner = None
    st.session_state.decider = None
    st.session_state.history = []

# Fonction d'initialisation
@st.cache_resource
def init_system():
    """Initialise le système médical"""
    try:
        data_path = "data/processed/dataset_processed.json"
        
        analyzer = CompleteNLPAnalyzer(data_path)
        reasoner = MedicalReasoner()
        decider = DecisionGenerator()
        
        return analyzer, reasoner, decider, True
    except Exception as e:
        st.error(f"❌ Erreur d'initialisation: {e}")
        return None, None, None, False

# Header
st.markdown('<div class="main-header">🏥 Système de Triage Médical Intelligent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Analyse NLP Avancée • Multilingue • Data-Driven AI</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Pays
    country = st.selectbox(
        "🌍 Pays",
        ["Tunisie", "France", "Maroc", "Algérie"],
        index=0
    )
    
    # Langue
    language = st.selectbox(
        "🗣️ Langue préférée",
        ["Français", "English", "العربية"],
        index=0
    )
    
    st.divider()
    
    # Stats
    st.header("📊 Statistiques")
    if st.session_state.history:
        st.metric("Consultations", len(st.session_state.history))
        total_symptoms = sum(len(h['symptoms']) for h in st.session_state.history)
        st.metric("Symptômes détectés", total_symptoms)
    else:
        st.info("Aucune consultation pour le moment")
    
    st.divider()
    
    # Actions
    if st.button("🔄 Réinitialiser", use_container_width=True):
        st.session_state.history = []
        st.rerun()
    
    if st.button("📥 Télécharger historique", use_container_width=True):
        st.info("Fonctionnalité bientôt disponible")

# Initialisation du système
if not st.session_state.initialized:
    with st.spinner("🔧 Initialisation du système médical..."):
        analyzer, reasoner, decider, success = init_system()
        
        if success:
            st.session_state.analyzer = analyzer
            st.session_state.reasoner = reasoner
            st.session_state.decider = decider
            st.session_state.initialized = True
            st.success("✅ Système initialisé avec succès!")
        else:
            st.error("❌ Impossible d'initialiser le système")
            st.stop()

# Interface principale
tab1, tab2, tab3 = st.tabs(["🩺 Consultation", "📊 Analyse Détaillée", "📚 Historique"])

with tab1:
    st.header("🩺 Décrivez vos symptômes")
    
    # Zone de saisie
    patient_input = st.text_area(
        "💬 Entrez vos symptômes en langage naturel",
        placeholder="Ex: J'ai mal à la tête et je me sens fatigué...\nEx: I have chest pain and difficulty breathing...",
        height=120,
        help="Vous pouvez écrire en français, anglais ou arabe"
    )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        analyze_button = st.button("🔍 Analyser", type="primary", use_container_width=True)
    
    with col2:
        if st.button("🗑️ Effacer", use_container_width=True):
            st.rerun()
    
    # Analyse
    if analyze_button and patient_input:
        with st.spinner("⏳ Analyse en cours..."):
            try:
                # Analyse NLP
                analysis = st.session_state.analyzer.analyze(patient_input)
                
                # Raisonnement médical
                reasoning = st.session_state.reasoner.reason(analysis)
                
                # Génération décision
                decision = st.session_state.decider.generate_decision(reasoning)
                
                # Sauvegarder dans historique
                st.session_state.history.append({
                    'timestamp': datetime.now(),
                    'input': patient_input,
                    'symptoms': analysis['symptoms'],
                    'diseases': analysis['possible_diseases'],
                    'urgency': reasoning.get('urgency', 'UNKNOWN')
                })
                
                # Affichage résultats
                st.success("✅ Analyse terminée!")
                
                # Urgence
                urgency = reasoning.get('urgency', 'URGENCE MODÉRÉE')
                
                if 'ÉLEVÉE' in urgency or 'VITALE' in urgency:
                    st.markdown('<div class="urgency-high">', unsafe_allow_html=True)
                    st.error(f"🚨 **URGENCE: {urgency}**")
                    st.markdown('</div>', unsafe_allow_html=True)
                elif 'MODÉRÉE' in urgency:
                    st.markdown('<div class="urgency-medium">', unsafe_allow_html=True)
                    st.warning(f"⚠️ **Urgence: {urgency}**")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="urgency-low">', unsafe_allow_html=True)
                    st.info(f"ℹ️ **Urgence: {urgency}**")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Résultats en colonnes
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("💊 Symptômes détectés")
                    if analysis['symptoms']:
                        for symptom in analysis['symptoms'][:5]:
                            st.markdown(f"""
                            <div class="symptom-card">
                                <strong>{symptom['symptom']}</strong><br>
                                <small>Confiance: {symptom['confidence']:.0%} • Méthode: {symptom['method']}</small>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("Aucun symptôme détecté")
                
                with col2:
                    st.subheader("🏥 Maladies possibles")
                    if analysis['possible_diseases']:
                        for disease, info in list(analysis['possible_diseases'].items())[:3]:
                            st.markdown(f"""
                            <div class="disease-card">
                                <strong>{disease}</strong><br>
                                <small>Score: {info['score']} • Urgence: {info['urgency']}</small>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("Aucune maladie identifiée")
                
                # Recommandations
                st.subheader("💡 Recommandations")
                
                recommendations = reasoning.get('recommendations', [])
                if recommendations:
                    for i, rec in enumerate(recommendations[:4], 1):
                        st.markdown(f"**{i}.** {rec}")
                
                # Spécialiste
                specialist = reasoning.get('specialist', 'Médecin généraliste')
                timing = reasoning.get('timing', '24-48 heures')
                
                st.info(f"👨‍⚕️ **Spécialiste recommandé:** {specialist}\n\n⏰ **Délai:** {timing}")
                
                # Numéros d'urgence
                st.subheader("🚨 Numéros d'urgence")
                emergency = analysis.get('emergency_numbers', {})
                
                cols = st.columns(4)
                if emergency:
                    with cols[0]:
                        st.metric("SAMU", emergency.get('samu', '190'))
                    with cols[1]:
                        st.metric("Urgences", emergency.get('urgences', '197'))
                    with cols[2]:
                        st.metric("Police", emergency.get('police', '197'))
                    with cols[3]:
                        st.metric("Pompiers", emergency.get('pompiers', '198'))
                
                # Avertissement
                st.warning("⚠️ **Important:** Ce système ne remplace pas un médecin. En cas de doute, consultez un professionnel de santé.")
                
            except Exception as e:
                st.error(f"❌ Erreur lors de l'analyse: {e}")

with tab2:
    st.header("📊 Analyse NLP Détaillée")
    
    if analyze_button and patient_input:
        st.subheader("🔬 Processus NLP Complet")
        
        # Les étapes NLP
        steps = [
            ("1️⃣ Détection Langue", f"Langue: {analysis.get('detected_language', 'N/A')}"),
            ("2️⃣ Correction Orthographique", f"{len(analysis.get('corrections', []))} correction(s)"),
            ("3️⃣ Normalisation", "Termes médicaux normalisés"),
            ("4️⃣ Tokenization", f"{len(analysis.get('processed_text', '').split())} tokens"),
            ("5️⃣ TF-IDF", "Pondération des termes importants"),
            ("6️⃣ POS Tagging", "Extraction NOUN/ADJ"),
            ("7️⃣ Word2Vec", "Similarités sémantiques"),
            ("8️⃣ Matching", f"{len(analysis['symptoms'])} symptômes trouvés"),
        ]
        
        for title, desc in steps:
            with st.expander(f"{title} - {desc}"):
                st.write(desc)
        
        # Statistiques
        st.subheader("📈 Statistiques")
        
        cols = st.columns(4)
        
        with cols[0]:
            st.metric("Symptômes", len(analysis['symptoms']))
        with cols[1]:
            st.metric("Maladies", len(analysis['possible_diseases']))
        with cols[2]:
            st.metric("Corrections", len(analysis.get('corrections', [])))
        with cols[3]:
            confidence = analysis['symptoms'][0]['confidence'] * 100 if analysis['symptoms'] else 0
            st.metric("Confiance", f"{confidence:.0f}%")
    else:
        st.info("👆 Effectuez une analyse dans l'onglet Consultation pour voir les détails")

with tab3:
    st.header("📚 Historique des Consultations")
    
    if st.session_state.history:
        for i, entry in enumerate(reversed(st.session_state.history), 1):
            with st.expander(f"Consultation {len(st.session_state.history) - i + 1} - {entry['timestamp'].strftime('%d/%m/%Y %H:%M')}"):
                st.markdown(f"**💬 Symptômes décrits:**\n\n{entry['input']}")
                
                st.markdown(f"**💊 Symptômes détectés:** {len(entry['symptoms'])}")
                for symptom in entry['symptoms'][:3]:
                    st.markdown(f"- {symptom['symptom']} ({symptom['confidence']:.0%})")
                
                st.markdown(f"**🏥 Maladies possibles:** {len(entry['diseases'])}")
                for disease in list(entry['diseases'].keys())[:2]:
                    st.markdown(f"- {disease}")
                
                st.markdown(f"**🚨 Urgence:** {entry['urgency']}")
    else:
        st.info("Aucune consultation enregistrée")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #757575; padding: 2rem 0;'>
    <p><strong>Système de Triage Médical Intelligent v3.0</strong></p>
    <p>Propulsé par NLP avancé • 4920 cas médicaux • Multilingue (FR/EN/AR)</p>
    <p><small>⚠️ Ce système est un outil d'aide à la décision. Il ne remplace pas l'avis d'un professionnel de santé.</small></p>
</div>
""", unsafe_allow_html=True)