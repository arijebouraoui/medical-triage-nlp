"""
Interface Streamlit avec Random Forest ML
"""

import streamlit as st
import sys
import os
from datetime import datetime

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from agents.data_loader.medical_data_loader import MedicalDataLoader
from agents.analyzer.nlp_analyzer_v3 import MedicalNLPAnalyzer
from agents.reasoner.medical_reasoner import MedicalReasoner
from agents.reasoner.ml_medical_reasoner import MLMedicalReasoner  # ← NOUVEAU
from agents.decider.decision_generator import DecisionGenerator

st.set_page_config(
    page_title="🏥 Triage Médical Intelligent",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
</style>
""", unsafe_allow_html=True)

if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.history = []
    st.session_state.selected_country = "Tunisie"  # Défaut
    st.session_state.use_ml_model = True  # ML par défaut

@st.cache_resource
def init_system(use_ml=True):
    """
    Initialise le système avec choix du modèle de raisonnement
    
    Args:
        use_ml: Si True, utilise Random Forest. Sinon, règles classiques
    """
    try:
        data_path = "data/processed/dataset_processed.json"
        data_loader = MedicalDataLoader(data_path)
        analyzer = MedicalNLPAnalyzer(data_path)
        
        # Charger le reasoner selon le choix
        if use_ml:
            model_path = 'models/random_forest_reasoner.pkl'
            if os.path.exists(model_path):
                reasoner = MLMedicalReasoner(model_path=model_path)
                st.success("✅ Random Forest chargé (97.27% accuracy)")
            else:
                st.error(f"❌ Modèle ML introuvable: {model_path}")
                st.info("Lancer: python train_ml_reasoner.py")
                reasoner = MedicalReasoner(data_loader)
                use_ml = False  # Fallback
        else:
            reasoner = MedicalReasoner(data_loader)
        
        return analyzer, reasoner, data_loader, True, use_ml
    except Exception as e:
        st.error(f"❌ Erreur: {e}")
        return None, None, None, False, False

st.markdown('<div class="main-header">🏥 Système de Triage Médical Intelligent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Analyse NLP Avancée • Multilingue • Machine Learning</div>', unsafe_allow_html=True)

# WARNING: Vérification des dépendances pour l'utilisateur
try:
    import deep_translator
    HAS_TRANSLATOR = True
except ImportError:
    HAS_TRANSLATOR = False

try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False

try:
    from spellchecker import SpellChecker
    HAS_PYSPELLCHECKER = True
except ImportError:
    HAS_PYSPELLCHECKER = False

with st.sidebar:
    st.header("⚙️ Configuration")
    
    # IMPORTANT: Sauvegarder le pays sélectionné
    country = st.selectbox("🌍 Pays", ["Tunisie", "France", "UK", "USA", "Canada"], index=0)
    st.session_state.selected_country = country
    
    language = st.selectbox("🗣️ Langue", ["Français", "English", "العربية"], index=0)
    
    st.divider()
    
    # ==========================================
    # NOUVEAU: CHOIX DU MODÈLE DE RAISONNEMENT
    # ==========================================
    st.header("🤖 Modèle de Raisonnement")
    
    use_ml_model = st.checkbox(
        "Random Forest ML",
        value=st.session_state.use_ml_model,
        help="Si coché: Random Forest (97% accuracy). Sinon: Règles + Statistiques"
    )
    
    # Si changement de modèle, réinitialiser
    if use_ml_model != st.session_state.use_ml_model:
        st.session_state.use_ml_model = use_ml_model
        st.session_state.initialized = False
        st.rerun()
    
    # Afficher info modèle
    if use_ml_model:
        st.success("✅ Random Forest")
        st.caption("📊 97.27% accuracy sur 989 cas")
        st.caption("🌳 100 arbres de décision")
        st.caption("📈 Entraîné sur 4,944 cas")
    else:
        st.info("ℹ️ Règles + Statistiques")
        st.caption("📋 17 priorités médicales")
        st.caption("🗳️ Votes majoritaires")
    
    st.divider()
    
    st.header("📊 Statistiques")
    if st.session_state.history:
        st.metric("Consultations", len(st.session_state.history))
        total_symptoms = sum(len(h['symptoms']) for h in st.session_state.history)
        st.metric("Symptômes", total_symptoms)
    else:
        st.info("Aucune consultation")
    
    st.divider()
    
    st.header("🔌 État du Système")
    if HAS_TRANSLATOR:
        st.success("✅ Traducteur Auto (Online)")
    else:
        st.error("❌ Traducteur Manquant")
        st.caption("`pip install deep-translator`")
        
    if HAS_SPACY:
        st.success("✅ NLP Avancé (SpaCy)")
    else:
        st.warning("⚠️ NLP Basique")
        st.caption("`python -m spacy download en_core_web_md`")

    if HAS_PYSPELLCHECKER:
        st.success("✅ Correcteur (Standard)")
    else:
        st.error("❌ Correcteur Manquant")
        st.caption("`pip install pyspellchecker`")
    
    st.divider()
    
    if st.button("🔄 Réinitialiser", use_container_width=True):
        st.session_state.history = []
        st.rerun()

if not st.session_state.initialized:
    with st.spinner("🔧 Initialisation..."):
        analyzer, reasoner, data_loader, success, ml_used = init_system(use_ml=st.session_state.use_ml_model)
        
        if success:
            st.session_state.analyzer = analyzer
            st.session_state.reasoner = reasoner
            st.session_state.data_loader = data_loader
            st.session_state.ml_reasoner_active = ml_used
            st.session_state.initialized = True
            st.success("✅ Système prêt!")
        else:
            st.error("❌ Erreur d'initialisation")
            st.stop()

tab1, tab2, tab3 = st.tabs(["🩺 Consultation", "📊 Analyse Détaillée", "📚 Historique"])

with tab1:
    st.header("🩺 Décrivez vos symptômes")
    
    patient_input = st.text_area(
        "💬 Entrez vos symptômes",
        placeholder="Ex: J'ai mal aux dents...",
        height=120
    )
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        analyze_button = st.button("🔍 Analyser", type="primary", use_container_width=True)
    
    if analyze_button and patient_input:
        with st.spinner("⏳ Analyse..."):
            try:
                analysis = st.session_state.analyzer.analyze(patient_input)
                reasoning = st.session_state.reasoner.reason(analysis)
                
                # FIX: Créer DecisionGenerator avec le pays sélectionné
                decider = DecisionGenerator(patient_country=st.session_state.selected_country)
                decision = decider.generate_decision(reasoning)
                
                st.session_state.current_analysis = analysis
                st.session_state.current_reasoning = reasoning
                
                # ML DATA
                ml_used = analysis.get('ml_used', False)
                ml_spec = analysis.get('ml_specialist', 'N/A')
                ml_spec_conf = analysis.get('ml_specialist_confidence', 0)
                ml_urgency = analysis.get('ml_urgency', 'N/A')
                ml_urgency_conf = analysis.get('ml_urgency_confidence', 0)
                
                st.session_state.history.append({
                    'timestamp': datetime.now(),
                    'input': patient_input,
                    'symptoms': analysis['symptoms'],
                    'diseases': analysis['possible_diseases'],
                    'urgency': reasoning.get('urgency', 'UNKNOWN')
                })
                
                st.success("✅ Analyse terminée!")
                
                urgency = reasoning.get('urgency', 'URGENCE MODÉRÉE')
                
                if 'ÉLEVÉE' in urgency or 'VITALE' in urgency:
                    st.error(f"🚨 **{urgency}**")
                elif 'MODÉRÉE' in urgency:
                    st.warning(f"⚠️ **{urgency}**")
                else:
                    st.info(f"ℹ️ **{urgency}**")
                
                # ==========================================
                # NOUVEAU: AFFICHER PROBABILITÉS RANDOM FOREST
                # ==========================================
                if st.session_state.ml_reasoner_active and 'model_probabilities' in reasoning:
                    with st.expander("🤖 Prédictions Random Forest ML", expanded=True):
                        probabilities = reasoning['model_probabilities']
                        
                        # Confiances globales
                        col1, col2 = st.columns(2)
                        with col1:
                            spec_conf = probabilities.get('specialist_confidence', 0)
                            st.metric("Confiance Spécialiste", f"{spec_conf:.1%}")
                        with col2:
                            urg_conf = probabilities.get('urgency_confidence', 0)
                            st.metric("Confiance Urgence", f"{urg_conf:.1%}")
                        
                        # Top 3 spécialistes
                        if 'top_3_specialists' in probabilities:
                            st.markdown("#### 📊 Top 3 Spécialistes (probabilités)")
                            
                            top_3 = probabilities['top_3_specialists']
                            cols = st.columns(3)
                            
                            for i, (specialist, proba) in enumerate(top_3.items()):
                                with cols[i]:
                                    # Ajouter emoji si c'est le choix final
                                    final_spec = reasoning.get('specialist', '')
                                    emoji = "✅ " if specialist == final_spec else ""
                                    st.metric(
                                        label=f"{emoji}{specialist}",
                                        value=f"{proba:.1%}"
                                    )
                        
                        st.caption("🌳 Random Forest: 100 arbres, 97.27% accuracy")
                
                # VISUALISATION CERVEAU IA (ancien code conservé)
                if ml_used:
                    with st.expander("🧠 Analyse du Cerveau Artificiel (True NLP)", expanded=False):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.metric("Confiance Spécialiste", f"{ml_spec_conf:.1%}", delta="AI Model")
                        # Comparaison Final vs IA (Spécialiste)
                        final_specialist = reasoning.get('specialist')
                        st.write(f"Suggestion IA: **{ml_spec}**")
                        
                        if ml_spec != final_specialist:
                             st.info(f"🛡️ **Protocole de Sécurité**\nLe système a priorisé **{final_specialist}** au lieu de l'IA.")

                        with c2:
                            st.metric("Confiance Urgence", f"{ml_urgency_conf:.1%}", delta="AI Model")
                            
                            # Comparaison Final vs IA (Urgence)
                            final_urgency = reasoning.get('urgency')
                            st.write(f"Suggestion IA: **{ml_urgency}**")

                            if ml_urgency != final_urgency:
                                st.error(f"🚨 **Niveau d'Urgence Ajusté**\nL'IA proposait *{ml_urgency}*, mais les symptômes requièrent **{final_urgency}**.")

                        if ml_spec_conf > 0.4 and ml_spec == final_specialist:
                            st.caption("✅ L'IA confirme le diagnostic.")
                        elif ml_spec != final_specialist:
                            pass # Déjà géré au dessus
                        else:
                            st.caption("⚠️ L'IA est incertaine, le système utilise les règles de sécurité.")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("💊 Symptômes")
                    if analysis['symptoms']:
                        for s in analysis['symptoms'][:5]:
                            st.write(f"• **{s['symptom']}** ({s['confidence']:.0%})")
                    else:
                        st.info("Aucun symptôme")
                
                with col2:
                    st.subheader("🏥 Maladies possibles")
                    if analysis['possible_diseases']:
                        for disease, info in list(analysis['possible_diseases'].items())[:3]:
                            st.write(f"• **{disease}** (Score: {info['score']})")
                    else:
                        st.info("Aucune maladie")
                
                st.subheader("💡 Recommandations")
                
                recommendations = reasoning.get('recommendations', [])
                for i, rec in enumerate(recommendations[:4], 1):
                    st.write(f"{i}. {rec}")
                
                specialist = reasoning.get('specialist', 'Médecin généraliste')
                timing = reasoning.get('timing', '24-48 heures')
                
                st.info(f"👨‍⚕️ **Spécialiste:** {specialist}\n\n⏰ **Délai:** {timing}")
                
                # FIX: Utiliser les numéros du DecisionGenerator
                st.subheader(f"🚨 Numéros d'urgence ({st.session_state.selected_country})")
                emergency = decider.emergency_numbers.get(st.session_state.selected_country, {})
                
                cols = st.columns(4)
                if emergency:
                    idx = 0
                    for key, value in emergency.items():
                        if idx < 4:
                            with cols[idx]:
                                st.metric(key.capitalize(), value)
                            idx += 1
                
                st.warning("⚠️ Ce système ne remplace pas un médecin.")
                
                st.divider()
                st.write("Ceci était-il correct ?")
                b1, b2 = st.columns(2)
                if b1.button("👍 Oui"):
                    st.toast("Merci pour votre feedback ! L'IA apprendra de ce cas.")
                    # TODO: Sauvegarder pour retraining
                if b2.button("👎 Non"):
                    st.toast("Noté. Nous allons vérifier ce cas.")
                
            except Exception as e:
                st.error(f"❌ Erreur: {e}")
                import traceback
                st.code(traceback.format_exc())

with tab2:
    st.header("📊 Analyse NLP Détaillée")
    
    if hasattr(st.session_state, 'current_analysis'):
        analysis = st.session_state.current_analysis
        
        st.subheader("🔬 Processus NLP Complet")
        
        with st.expander("1️⃣ Détection Langue", expanded=True):
            st.write(f"**Langue détectée:** {analysis.get('detected_language', 'N/A').upper()}")
        
        with st.expander("2️⃣ Correction Orthographique"):
            corrections = analysis.get('corrections', [])
            if corrections:
                st.write(f"**{len(corrections)} correction(s):**")
                for c in corrections[:5]:
                    st.write(f"• '{c.get('original', '')}' → '{c.get('corrected', '')}'")
            else:
                st.write("0 correction")
        
        with st.expander("3️⃣ Normalisation"):
            st.write("Termes médicaux normalisés")
            st.code(analysis.get('processed_text', ''))
        
        with st.expander("4️⃣ Tokenization"):
            tokens = analysis.get('processed_text', '').split()
            st.write(f"**{len(tokens)} tokens**")
            st.write(tokens[:20])
        
        with st.expander("8️⃣ Matching"):
            st.write(f"**{len(analysis['symptoms'])} symptôme(s) trouvé(s)**")
            for s in analysis['symptoms']:
                st.write(f"• **{s['symptom']}** - Méthode: {s['method']} - Confiance: {s['confidence']:.0%}")
        
        st.divider()
        st.subheader("📈 Statistiques")
        
        cols = st.columns(4)
        
        with cols[0]:
            st.metric("Symptômes", len(analysis['symptoms']))
        with cols[1]:
            st.metric("Maladies", len(analysis.get('possible_diseases', {})))
        with cols[2]:
            st.metric("Corrections", len(analysis.get('corrections', [])))
        with cols[3]:
            confidence = analysis['symptoms'][0]['confidence'] * 100 if analysis['symptoms'] else 0
            st.metric("Confiance", f"{confidence:.0f}%")
    else:
        st.info("👆 Effectuez une analyse dans l'onglet Consultation pour voir les détails")

with tab3:
    st.header("📚 Historique")
    
    if st.session_state.history:
        for i, entry in enumerate(reversed(st.session_state.history), 1):
            with st.expander(f"Consultation {len(st.session_state.history) - i + 1} - {entry['timestamp'].strftime('%d/%m/%Y %H:%M')}"):
                st.write(f"**Input:** {entry['input']}")
                st.write(f"**Symptômes:** {len(entry['symptoms'])}")
                st.write(f"**Urgence:** {entry['urgency']}")
    else:
        st.info("Aucune consultation")

st.divider()
st.markdown("""
<div style='text-align: center; color: #757575;'>
    <p><strong>Système de Triage Médical v4.0 ML</strong></p>
    <p>NLP Avancé • Random Forest • 97.27% Accuracy</p>
</div>
""", unsafe_allow_html=True)