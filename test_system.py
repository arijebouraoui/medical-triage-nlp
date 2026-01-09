"""
Test System - Valide le système NLP
====================================
Lance ce script pour valider que tout fonctionne
"""

import sys
import os

# Setup path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from agents.data_loader.medical_data_loader import MedicalDataLoader
from agents.analyzer.nlp_analyzer_v3 import MedicalNLPAnalyzer
from agents.reasoner.medical_reasoner import MedicalReasoner


def test_system():
    """Tests complets du système"""
    
    print("\n" + "="*70)
    print("🧪 TESTS DU SYSTÈME NLP MÉDICAL")
    print("="*70)
    
    try:
        # Init
        print("\n1️⃣  Initialisation...")
        data_loader = MedicalDataLoader("data/processed/dataset_processed.json")
        analyzer = MedicalNLPAnalyzer("data/processed/dataset_processed.json")
        reasoner = MedicalReasoner(data_loader)
        print("   ✅ Système initialisé")
        
    except Exception as e:
        print(f"   ❌ Erreur initialisation: {e}")
        return False
    
    # Tests
    tests = [
        {
            'name': 'Français - Cœur',
            'input': "j'ai mal au coeur",
            'expected_symptom': 'chest pain',
            'expected_specialist': 'Cardiologue',
        },
        {
            'name': 'Français - Dents',
            'input': "j'ai mal aux dents",
            'expected_symptom': 'toothache',
            'expected_specialist': 'Dentiste',
        },
        {
            'name': 'Anglais - Tête',
            'input': "i have a headache",
            'expected_symptom': 'headache',
            'expected_specialist': 'Neurologue',
        },
        {
            'name': 'Anglais avec faute',
            'input': "i have chst pain",
            'expected_symptom': 'chest pain',
            'expected_specialist': 'Cardiologue',
        },
    ]
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(tests, 1):
        print(f"\n{i}️⃣  Test: {test['name']}")
        print(f"   Input: \"{test['input']}\"")
        
        try:
            # Analyse
            analysis = analyzer.analyze(test['input'])
            reasoning = reasoner.reason(analysis)
            
            symptoms = [s['symptom'] for s in analysis['symptoms']]
            specialist = reasoning.get('specialist', 'ERROR')
            
            # Vérifications
            symptom_ok = any(test['expected_symptom'] in s for s in symptoms)
            specialist_ok = specialist == test['expected_specialist']
            
            if symptom_ok and specialist_ok:
                print(f"   ✅ Symptôme: {symptoms[0] if symptoms else 'AUCUN'}")
                print(f"   ✅ Spécialiste: {specialist}")
                passed += 1
            else:
                print(f"   ❌ Symptôme: {symptoms[0] if symptoms else 'AUCUN'} (attendu: {test['expected_symptom']})")
                print(f"   ❌ Spécialiste: {specialist} (attendu: {test['expected_specialist']})")
                failed += 1
                
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
            failed += 1
    
    # Résultats
    print("\n" + "="*70)
    print(f"📊 RÉSULTATS: {passed}/{len(tests)} tests réussis")
    
    if failed == 0:
        print("🎉 TOUS LES TESTS PASSÉS!")
        print("="*70)
        return True
    else:
        print(f"⚠️  {failed} test(s) échoué(s)")
        print("="*70)
        return False


if __name__ == "__main__":
    success = test_system()
    
    if not success:
        print("\n❌ Certains tests ont échoué.")
        print("💡 Vérifie que tu as bien lancé: python setup_dataset.py")
        exit(1)
    else:
        print("\n✅ Système prêt à l'emploi!")
        print("🚀 Lance: streamlit run streamlit_app.py")