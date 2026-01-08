"""
NLP Analyzer 
==========================================
Analyseur NLP complet qui:
- Charge les données depuis dataset_processed.json
- Utilise spell correction générique
- Applique stemming/lemmatization automatique
- Supporte multilingue (FR/EN/AR/ES)
- Pas de données hardcodées!

VERSION FINALE - IMPORTS FIXES
"""

import os
import sys
import json
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import re

# Ajouter le chemin du projet au PYTHONPATH
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Importer nos modules personnalisés
try:
    from agents.data_loader.medical_data_loader import MedicalDataLoader
    from agents.nlp.spell_corrector import SpellCorrector
    from agents.nlp.multilingual_processor import MultilingualProcessor, Language
    from agents.nlp.nlp_stemmer import NLPStemmer
    MODULES_LOADED = True
except ImportError as e:
    print(f"❌ ERREUR D'IMPORT: {e}")
    print("\n⚠️  Assurez-vous que la structure suivante existe:")
    print("   agents/")
    print("   ├── data_loader/")
    print("   │   ├── __init__.py")
    print("   │   └── medical_data_loader.py")
    print("   └── nlp/")
    print("       ├── __init__.py")
    print("       ├── spell_corrector.py")
    print("       ├── multilingual_processor.py")
    print("       └── nlp_stemmer.py")
    MODULES_LOADED = False
    sys.exit(1)

# Imports optionnels (spaCy, etc.)
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("⚠️  spaCy non disponible - fonctionnalités limitées")


class DataDrivenNLPAnalyzer:
    """Analyseur NLP entièrement data-driven"""
    
    def __init__(self, 
                 data_path: str = "data/processed/dataset_processed.json",
                 use_spacy: bool = True):
        """
        Args:
            data_path: Chemin vers dataset_processed.json
            use_spacy: Utiliser spaCy si disponible
        """
        print("\n" + "="*70)
        print("🚀 INITIALISATION NLP ANALYZER (DATA-DRIVEN)")
        print("="*70)
        
        # Charger les modules
        print("\n📊 Chargement des données...")
        self.data_loader = MedicalDataLoader(data_path)
        
        print("\n🔧 Initialisation spell corrector...")
        # Créer le correcteur avec le vocabulaire du dataset
        medical_vocab = self.data_loader.get_all_symptoms()
        self.spell_corrector = SpellCorrector(medical_vocab)
        
        print("\n🌍 Initialisation processeur multilingue...")
        self.multilingual = MultilingualProcessor()
        
        print("\n🌱 Initialisation stemmer...")
        self.stemmer = NLPStemmer()
        
        # SpaCy (optionnel)
        self.nlp = None
        if use_spacy and SPACY_AVAILABLE:
            try:
                print("\n🧠 Chargement spaCy...")
                self.nlp = spacy.load("en_core_web_sm")
                print("   ✅ spaCy chargé avec succès")
            except:
                print("   ⚠️  spaCy modèle non trouvé")
                print("   Installez avec: python -m spacy download en_core_web_sm")
        
        # Historique de session
        self.session_history = defaultdict(list)
        
        print("\n✅ INITIALISATION COMPLÈTE!")
        print("="*70)
    
    def analyze(self, 
                patient_text: str, 
                session_id: str = None,
                country: str = 'fr') -> Dict:
        """
        Analyse COMPLÈTE d'un texte patient
        
        Args:
            patient_text: Texte du patient
            session_id: ID de session
            country: Pays (pour numéros d'urgence)
        
        Returns:
            Dict complet avec tous les résultats
        """
        print(f"\n{'='*70}")
        print(f"🧠 ANALYSE: '{patient_text[:50]}...'")
        print(f"{'='*70}")
        
        # Session
        if session_id is None:
            session_id = f"session_{id(patient_text)}"
        
        # ÉTAPE 1: Détection de langue
        print("\n📝 ÉTAPE 1: Détection de langue")
        detected_lang = self.multilingual.detect_language(patient_text)
        print(f"   Langue détectée: {detected_lang.value}")
        
        # ÉTAPE 2: Spell correction
        print("\n🔧 ÉTAPE 2: Correction orthographique")
        corrected_text, corrections = self.spell_corrector.correct_text(patient_text)
        
        if corrections:
            print(f"   ✅ {len(corrections)} corrections effectuées:")
            for corr in corrections[:3]:  # Afficher 3 premières
                print(f"      '{corr['original']}' → '{corr['corrected']}'")
        else:
            print("   ✓ Aucune correction nécessaire")
        
        # ÉTAPE 3: Traduction vers anglais (si nécessaire)
        print("\n🌍 ÉTAPE 3: Traduction vers anglais")
        if detected_lang != Language.ENGLISH:
            translated_text = self.multilingual.translate_to_english(
                corrected_text, detected_lang
            )
            print(f"   Traduit: '{translated_text[:50]}...'")
        else:
            translated_text = corrected_text
            print("   Déjà en anglais")
        
        # ÉTAPE 4: Stemming et extraction
        print("\n🌱 ÉTAPE 4: Stemming et tokenization")
        stemmed_words = self.stemmer.stem_text(translated_text, 'en')
        print(f"   {len(stemmed_words)} tokens extraits")
        
        # ÉTAPE 5: Extraction de symptômes avec spaCy (si disponible)
        spacy_entities = []
        if self.nlp:
            print("\n🧠 ÉTAPE 5: Analyse spaCy")
            doc = self.nlp(translated_text)
            spacy_entities = [
                (ent.text, ent.label_) 
                for ent in doc.ents
            ]
            if spacy_entities:
                print(f"   Entités trouvées: {len(spacy_entities)}")
        
        # ÉTAPE 6: Matching avec la base de données
        print("\n💊 ÉTAPE 6: Matching symptômes")
        matched_symptoms = self._match_symptoms_intelligent(
            translated_text, stemmed_words
        )
        print(f"   ✅ {len(matched_symptoms)} symptômes identifiés")
        
        for symptom in matched_symptoms[:5]:  # Afficher les 5 premiers
            print(f"      - {symptom['symptom']} (méthode: {symptom['method']}, confiance: {symptom['confidence']:.2f})")
        
        # ÉTAPE 7: Recherche de maladies
        print("\n🏥 ÉTAPE 7: Recherche maladies possibles")
        possible_diseases = self.data_loader.find_diseases_by_symptoms(
            [s['symptom'] for s in matched_symptoms]
        )
        
        top_diseases = list(possible_diseases.items())[:3]
        if top_diseases:
            print(f"   Top 3 maladies:")
            for disease, info in top_diseases:
                print(f"      - {disease} (score: {info['score']}, {info['urgency']})")
        else:
            print("   ⚠️  Aucune maladie trouvée avec ces symptômes")
        
        # Historique de session
        self.session_history[session_id].append({
            'text': patient_text,
            'symptoms': matched_symptoms,
            'timestamp': 'now'  # TODO: ajouter vraie timestamp
        })
        
        # Numéros d'urgence
        emergency_info = self.multilingual.get_emergency_info(detected_lang)
        
        # RÉSULTAT FINAL
        result = {
            'symptoms': matched_symptoms,
            'all_session_symptoms': self._get_all_session_symptoms(session_id),
            'possible_diseases': dict(list(possible_diseases.items())[:5]),
            'corrections': corrections,
            'language': detected_lang.value,
            'session_id': session_id,
            'turn': len(self.session_history[session_id]),
            'emergency_numbers': emergency_info,
            'processed_text': translated_text,
            'stemmed_words': stemmed_words,
            'spacy_entities': spacy_entities,
            'statistics': {
                'total_symptoms_found': len(matched_symptoms),
                'total_corrections': len(corrections),
                'total_diseases_matched': len(possible_diseases)
            }
        }
        
        print(f"\n{'='*70}")
        print(f"✅ ANALYSE TERMINÉE")
        print(f"   Symptômes: {result['statistics']['total_symptoms_found']}")
        print(f"   Maladies possibles: {result['statistics']['total_diseases_matched']}")
        print(f"   Corrections: {result['statistics']['total_corrections']}")
        print(f"{'='*70}\n")
        
        return result
    
    def _match_symptoms_intelligent(self, 
                                   text: str, 
                                   stemmed_words: List[str]) -> List[Dict]:
        """
        Match intelligent avec la base de données
        Combine: exact match, stemming, semantic similarity
        """
        matched = []
        text_lower = text.lower()
        
        # Récupérer tous les symptômes connus
        all_symptoms = self.data_loader.get_all_symptoms()
        
        # Méthode 1: Exact match
        for symptom in all_symptoms:
            if symptom in text_lower:
                matched.append({
                    'symptom': symptom,
                    'method': 'exact',
                    'confidence': 1.0,
                    'original_text': symptom
                })
        
        # Méthode 2: Stemmed match
        for symptom in all_symptoms:
            symptom_stem = self.stemmer.stem_word(symptom, 'en')
            
            if symptom_stem in stemmed_words:
                # Vérifier si pas déjà trouvé
                if not any(m['symptom'] == symptom for m in matched):
                    matched.append({
                        'symptom': symptom,
                        'method': 'stemmed',
                        'confidence': 0.9,
                        'original_text': symptom
                    })
        
        # Méthode 3: Variation match (ex: "stomach pain" si "pain" détecté)
        words_in_text = set(text_lower.split())
        
        for word in words_in_text:
            variations = self.data_loader.get_symptom_variations(word)
            
            for variation in variations:
                if variation in text_lower:
                    if not any(m['symptom'] == variation for m in matched):
                        matched.append({
                            'symptom': variation,
                            'method': 'variation',
                            'confidence': 0.8,
                            'original_text': variation
                        })
        
        # Dédupliquer et trier par confiance
        unique_symptoms = {}
        for item in matched:
            symptom = item['symptom']
            if symptom not in unique_symptoms or \
               item['confidence'] > unique_symptoms[symptom]['confidence']:
                unique_symptoms[symptom] = item
        
        result = sorted(
            unique_symptoms.values(), 
            key=lambda x: x['confidence'], 
            reverse=True
        )
        
        return result
    
    def _get_all_session_symptoms(self, session_id: str) -> List[Dict]:
        """Récupère tous les symptômes de la session"""
        all_symptoms = []
        seen = set()
        
        for turn in self.session_history[session_id]:
            for symptom in turn['symptoms']:
                symptom_name = symptom['symptom']
                if symptom_name not in seen:
                    all_symptoms.append(symptom)
                    seen.add(symptom_name)
        
        return all_symptoms
    
    def get_session_summary(self, session_id: str) -> Dict:
        """Résumé complet d'une session"""
        all_symptoms = self._get_all_session_symptoms(session_id)
        
        # Rechercher maladies possibles
        diseases = self.data_loader.find_diseases_by_symptoms(
            [s['symptom'] for s in all_symptoms]
        )
        
        return {
            'session_id': session_id,
            'total_turns': len(self.session_history[session_id]),
            'total_symptoms': len(all_symptoms),
            'symptoms': all_symptoms,
            'possible_diseases': dict(list(diseases.items())[:10]),
            'history': self.session_history[session_id]
        }
    
    def clear_session(self, session_id: str):
        """Efface l'historique d'une session"""
        if session_id in self.session_history:
            del self.session_history[session_id]


# ==============================================================================
# FONCTION POUR COMPATIBILITÉ AVEC L'ANCIEN SYSTÈME
# ==============================================================================

def understand(patient_text: str, session_id: str = None, country: str = 'fr') -> Dict:
    """
    Fonction de compatibilité pour l'ancien système
    Permet d'utiliser analyzer.understand() au lieu de analyzer.analyze()
    """
    # Créer une instance globale si elle n'existe pas
    global _global_analyzer
    if '_global_analyzer' not in globals():
        _global_analyzer = DataDrivenNLPAnalyzer()
    
    return _global_analyzer.analyze(patient_text, session_id, country)


# ==============================================================================
# EXEMPLE D'UTILISATION
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "🎯"*35)
    print("TESTS DU SYSTÈME NLP DATA-DRIVEN")
    print("🎯"*35 + "\n")
    
    # Initialiser l'analyseur
    analyzer = DataDrivenNLPAnalyzer(
        data_path="data/processed/dataset_processed.json"
    )
    
    # Tests multilingues
    test_cases = [
        # Anglais
        ("I have a severe headache with nausea and vomiting", "en"),
        
        # Français
        ("J'ai mal à la tête et de la fièvre avec des vomissements", "fr"),
        
        # Espagnol
        ("Tengo dolor de cabeza y náusea con fiebre", "es"),
        
        # Avec fautes
        ("I have stomache payn and hedache", "en-typos"),
    ]
    
    print("\n" + "="*70)
    print("🧪 TESTS D'ANALYSE MULTILINGUE")
    print("="*70)
    
    for i, (test_text, lang_label) in enumerate(test_cases, 1):
        print(f"\n{'#'*70}")
        print(f"TEST {i}/{len(test_cases)} - {lang_label.upper()}")
        print(f"{'#'*70}")
        
        result = analyzer.analyze(test_text, session_id=f"test_{lang_label}")
        
        # Afficher résumé
        print(f"\n📊 RÉSUMÉ:")
        print(f"   Langue détectée: {result['language']}")
        print(f"   Symptômes trouvés: {len(result['symptoms'])}")
        print(f"   Corrections: {len(result['corrections'])}")
        print(f"   Maladies possibles: {len(result['possible_diseases'])}")
        
        if result['possible_diseases']:
            top_disease = list(result['possible_diseases'].keys())[0]
            print(f"   Top maladie: {top_disease}")
    
    # Test session multi-tours
    print(f"\n{'='*70}")
    print("🔄 TEST SESSION MULTI-TOUR")
    print(f"{'='*70}")
    
    session_id = "multi_turn_test"
    
    turns = [
        "I have a headache",
        "and also some nausea",
        "now I'm vomiting and feel dizzy"
    ]
    
    for i, turn in enumerate(turns, 1):
        print(f"\n▶️  Tour {i}/{len(turns)}: '{turn}'")
        result = analyzer.analyze(turn, session_id=session_id)
        print(f"   Symptômes ce tour: {len(result['symptoms'])}")
        print(f"   Symptômes totaux session: {len(result['all_session_symptoms'])}")
    
    # Résumé final
    summary = analyzer.get_session_summary(session_id)
    print(f"\n📋 RÉSUMÉ SESSION COMPLÈTE:")
    print(f"   ID: {summary['session_id']}")
    print(f"   Tours: {summary['total_turns']}")
    print(f"   Symptômes uniques: {summary['total_symptoms']}")
    
    if summary['possible_diseases']:
        print(f"\n   🏥 Top 3 maladies possibles:")
        for i, (disease, info) in enumerate(list(summary['possible_diseases'].items())[:3], 1):
            print(f"      {i}. {disease}")
            print(f"         Score: {info['score']}/{summary['total_symptoms']} symptômes")
            print(f"         Urgence: {info['urgency']}")
    
    print("\n" + "="*70)
    print("✅ TOUS LES TESTS TERMINÉS!")
    print("="*70 + "\n")