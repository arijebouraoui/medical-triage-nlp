"""
═══════════════════════════════════════════════════════════════════════
    SYSTÈME MULTI-AGENTS DE TRIAGE MÉDICAL
    Version 3.0 - DATA-DRIVEN NLP (100% basé sur données Kaggle)
═══════════════════════════════════════════════════════════════════════

Pipeline complet:
  Patient Input (langage naturel multilingue)
        ↓
  [Agent 1] Comprend le texte (NLP Data-Driven)
            - Détection langue (FR/EN/AR/ES)
            - Correction orthographique (Levenshtein)
            - Stemming/Lemmatization automatique
            - 4920 cas Kaggle chargés
        ↓
  [Agent 2] Raisonne médicalement
            - Base de données complète
            - Matching intelligent
        ↓
  [Agent 3] Communique la décision
            - Numéros d'urgence adaptés au pays
        ↓
  Rapport en langage naturel (PAS DE JSON!)

NOUVEAUTÉS VERSION 3.0:
- ✅ 100% Data-Driven (pas de hardcoding)
- ✅ 4920 cas médicaux de Kaggle
- ✅ Spell correction générique (tous les mots)
- ✅ Stemming automatique (tous les mots)
- ✅ Support multilingue complet (FR/EN/AR/ES)
- ✅ Détection automatique de langue
- ✅ Session multi-tours
"""

import sys
from pathlib import Path

# Ajouter le dossier du projet au path
project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import du nouveau système NLP data-driven
from agents.analyzer.nlp_analyzer_v3 import DataDrivenNLPAnalyzer
from agents.reasoner.medical_reasoner import MedicalReasoner
from agents.decider.decision_generator import DecisionGenerator


class MedicalTriageAI:
    """
    Système complet de triage médical avec agents NLP data-driven
    
    Principe fondamental VERSION 3.0:
    - 100% basé sur données (dataset_processed.json)
    - Pas de données hardcodées
    - Spell correction générique (Levenshtein)
    - Stemming/lemmatization automatique
    - Support multilingue complet (FR/EN/AR/ES)
    - Vraie compréhension du langage naturel
    - Communication humaine, pas JSON technique
    - Numéros d'urgence adaptés au pays du patient
    """
    
    def __init__(self, 
                 patient_country: str = "France", 
                 patient_city: str = None,
                 data_path: str = "data/processed/dataset_processed.json",
                 use_spacy: bool = True):
        """
        Initialise les 3 agents NLP (version data-driven)
        
        Args:
            patient_country: Pays du patient (pour numéros d'urgence)
            patient_city: Ville du patient (optionnel)
            data_path: Chemin vers dataset_processed.json
            use_spacy: Utiliser spaCy si disponible
        """
        print("\n" + "="*70)
        print("🏥 SYSTÈME MULTI-AGENTS DE TRIAGE MÉDICAL")
        print("   Version 3.0 - DATA-DRIVEN NLP AI")
        print("   📊 Propulsé par 4920 cas Kaggle")
        print("="*70 + "\n")
        
        print(f"🌍 Configuration pour: {patient_country}")
        if patient_city:
            print(f"📍 Ville: {patient_city}")
        
        print("\n🔧 Initialisation des agents intelligents...\n")
        
        # Agent 1: Nouveau système NLP data-driven
        print("📊 Agent 1: Chargement du système NLP data-driven...")
        self.analyzer = DataDrivenNLPAnalyzer(
            data_path=data_path,
            use_spacy=use_spacy
        )
        
        # Agent 2: Raisonne médicalement
        print("\n🧠 Agent 2: Initialisation du raisonneur médical...")
        self.reasoner = MedicalReasoner()
        
        # Agent 3: Communique en langage naturel (avec numéros d'urgence du pays)
        print("\n📝 Agent 3: Initialisation du générateur de décisions...")
        self.generator = DecisionGenerator(
            patient_country=patient_country,
            patient_city=patient_city
        )
        
        # Statistiques du système
        stats = self.analyzer.data_loader.get_statistics()
        
        print("\n" + "="*70)
        print("✅ SYSTÈME PRÊT")
        print("="*70)
        print(f"\n📊 STATISTIQUES:")
        print(f"   • Cas médicaux chargés: {stats['total_cases']}")
        print(f"   • Maladies dans la base: {stats['total_diseases']}")
        print(f"   • Symptômes uniques: {stats['total_symptoms']}")
        print(f"   • Langues supportées: Français, Anglais, Arabe, Espagnol")
        print(f"   • Spell correction: Générique (Levenshtein)")
        print(f"   • Stemming: Automatique (Porter/Snowball)")
        print("="*70 + "\n")
    
    
    def analyze_and_respond(self, 
                           patient_input: str, 
                           session_id: str = None,
                           verbose: bool = True) -> str:
        """
        Pipeline complet: comprend → raisonne → communique
        
        Args:
            patient_input: Ce que le patient dit (n'importe comment, n'importe quelle langue)
            session_id: ID de session pour tracking multi-tours
            verbose: Afficher les détails du processus
        
        Returns:
            Rapport en langage naturel (str, pas JSON!)
        """
        
        if verbose:
            print("\n" + "╔" + "═"*68 + "╗")
            print("║" + "  DÉBUT DE L'ANALYSE".center(68) + "║")
            print("╚" + "═"*68 + "╝")
        
        # Étape 1: Agent 1 - COMPREND (Nouveau système data-driven)
        if verbose:
            print("\n" + "─"*70)
            print("🤖 AGENT 1 - Compréhension NLP Data-Driven")
            print("─"*70)
        
        # Utiliser le nouveau système
        analysis = self.analyzer.analyze(
            patient_input, 
            session_id=session_id
        )
        
        if verbose:
            print(f"\n   ✅ Analyse terminée:")
            print(f"      • Langue détectée: {analysis['language']}")
            print(f"      • Symptômes trouvés: {len(analysis['symptoms'])}")
            if analysis['corrections']:
                print(f"      • Corrections orthographiques: {len(analysis['corrections'])}")
            print(f"      • Maladies possibles: {len(analysis['possible_diseases'])}")
        
        # Étape 2: Agent 2 - RAISONNE
        if verbose:
            print("\n" + "─"*70)
            print("🧠 AGENT 2 - Raisonnement médical")
            print("─"*70)
        
        reasoning = self.reasoner.reason(analysis)
        
        if verbose:
            print(f"\n   ✅ Raisonnement terminé:")
            print(f"      • Niveau d'urgence: {reasoning.get('urgency_level', 'N/A')}")
            print(f"      • Confidence: {reasoning.get('confidence', 0):.1%}")
        
        # Étape 3: Agent 3 - COMMUNIQUE
        if verbose:
            print("\n" + "─"*70)
            print("📝 AGENT 3 - Génération réponse")
            print("─"*70)
        
        final_report = self.generator.generate_decision(reasoning)
        
        if verbose:
            print("\n" + "╔" + "═"*68 + "╗")
            print("║" + "  ANALYSE TERMINÉE".center(68) + "║")
            print("╚" + "═"*68 + "╝\n")
        
        return final_report
    
    
    def analyze_session(self, session_id: str) -> dict:
        """
        Récupère le résumé complet d'une session
        
        Args:
            session_id: ID de la session
        
        Returns:
            Dict avec historique et statistiques
        """
        return self.analyzer.get_session_summary(session_id)
    
    
    def clear_session(self, session_id: str):
        """
        Efface l'historique d'une session
        
        Args:
            session_id: ID de la session à effacer
        """
        self.analyzer.clear_session(session_id)
    
    
    def analyze_batch(self, patient_inputs: list, session_prefix: str = "batch") -> list:
        """
        Analyse plusieurs cas
        
        Args:
            patient_inputs: Liste de textes patients
            session_prefix: Préfixe pour les IDs de session
        
        Returns:
            Liste de rapports
        """
        results = []
        
        for i, patient_input in enumerate(patient_inputs, 1):
            print(f"\n{'═'*70}")
            print(f"CAS {i}/{len(patient_inputs)}")
            print(f"{'═'*70}")
            
            session_id = f"{session_prefix}_{i}"
            report = self.analyze_and_respond(
                patient_input, 
                session_id=session_id,
                verbose=True
            )
            
            results.append({
                'input': patient_input,
                'session_id': session_id,
                'report': report
            })
        
        return results
    
    
    def get_system_statistics(self) -> dict:
        """Retourne les statistiques du système"""
        return self.analyzer.data_loader.get_statistics()


# ═══════════════════════════════════════════════════════════════════
# DÉMONSTRATION RAPIDE
# ═══════════════════════════════════════════════════════════════════

def quick_demo():
    """Démonstration rapide du système"""
    
    print("\n" + "="*70)
    print("🎬 DÉMONSTRATION SYSTÈME V3.0 - DATA-DRIVEN")
    print("="*70 + "\n")
    
    # Initialiser
    system = MedicalTriageAI()
    
    # Exemples multilingues
    test_cases = [
        # Français
        ("j'ai mal au ventre depuis 3 jours et je vomis", "fr"),
        
        # Anglais avec fautes
        ("I have severe hedache and stomache payn", "en"),
        
        # Session multi-tours
        ("I have a headache", "session_1"),
        ("and nausea", "session_1"),
        ("now I'm vomiting", "session_1"),
    ]
    
    for i, (patient_input, session) in enumerate(test_cases, 1):
        print(f"\n{'═'*70}")
        print(f"TEST {i}/{len(test_cases)}")
        print(f"{'═'*70}")
        print(f"\n📝 Patient dit: \"{patient_input}\"")
        print(f"🔖 Session: {session}")
        print("\n⏳ Analyse en cours...\n")
        
        # Analyser
        report = system.analyze_and_respond(patient_input, session_id=session)
        
        # Afficher le résultat
        print("\n" + "="*70)
        print("📋 RAPPORT POUR LE PATIENT")
        print("="*70 + "\n")
        print(report)
        print("\n")
    
    # Afficher résumé session
    print("\n" + "="*70)
    print("📊 RÉSUMÉ SESSION 'session_1'")
    print("="*70 + "\n")
    
    summary = system.analyze_session("session_1")
    print(f"Tours de conversation: {summary['total_turns']}")
    print(f"Symptômes uniques trouvés: {summary['total_symptoms']}")
    print(f"Symptômes: {[s['symptom'] for s in summary['symptoms']]}")
    
    if summary['possible_diseases']:
        print(f"\nTop 3 maladies possibles:")
        for i, (disease, info) in enumerate(list(summary['possible_diseases'].items())[:3], 1):
            print(f"  {i}. {disease}")
            print(f"     Score: {info['score']}/{summary['total_symptoms']}")
            print(f"     Urgence: {info['urgency']}")


def multilingual_demo():
    """Démonstration des capacités multilingues"""
    
    print("\n" + "="*70)
    print("🌍 DÉMONSTRATION MULTILINGUE")
    print("="*70 + "\n")
    
    system = MedicalTriageAI()
    
    multilingual_cases = [
        ("J'ai de la fièvre et mal à la tête", "Français"),
        ("I have a fever and headache", "English"),
        ("Tengo fiebre y dolor de cabeza", "Español"),
    ]
    
    for patient_input, language in multilingual_cases:
        print(f"\n{'─'*70}")
        print(f"Langue: {language}")
        print(f"Input: \"{patient_input}\"")
        print('─'*70)
        
        report = system.analyze_and_respond(patient_input, verbose=False)
        print(f"\n{report}\n")


if __name__ == "__main__":
    # Choisir la démo
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--multilingual":
        multilingual_demo()
    else:
        quick_demo()