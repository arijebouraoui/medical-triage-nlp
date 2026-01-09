"""
Script pour entraîner Random Forest Reasoner
Auteur: Arije Bouraoui
Date: Janvier 2026
"""

from agents.data_loader.medical_data_loader import MedicalDataLoader
from agents.reasoner.ml_medical_reasoner import MLMedicalReasoner
import os

def main():
    print("=" * 60)
    print("🤖 ENTRAÎNEMENT RANDOM FOREST REASONER")
    print("=" * 60)
    
    # 1. Charger données
    print("\n📂 Chargement dataset...")
    data_path = 'data/processed/dataset_processed.json'
    
    if not os.path.exists(data_path):
        print(f"❌ ERREUR: Dataset non trouvé: {data_path}")
        print("   Assurez-vous que le fichier existe!")
        return
    
    data_loader = MedicalDataLoader(data_path)
    # Utiliser l'attribut dataset directement (pas de méthode get_dataset)
    dataset = data_loader.dataset
    print(f"✅ Dataset chargé: {len(dataset)} cas")
    
    # 2. Créer dossier models si nécessaire
    if not os.path.exists('models'):
        os.makedirs('models')
        print("✅ Dossier 'models/' créé")
    
    # 3. Créer et entraîner reasoner ML
    print("\n🔧 Initialisation et entraînement...")
    ml_reasoner = MLMedicalReasoner(data_loader=data_loader)
    
    # Entraînement se fait automatiquement dans __init__
    # Résultats affichés automatiquement
    
    # 4. Sauvegarder modèles
    print("\n💾 Sauvegarde modèles...")
    model_path = 'models/random_forest_reasoner.pkl'
    ml_reasoner.save_model(model_path)
    
    # 5. Test rapide
    print("\n🧪 Test rapide...")
    
    test_analysis = {
        'symptoms': [
            {'symptom': 'chest pain', 'confidence': 0.95},
            {'symptom': 'breathlessness', 'confidence': 0.90}
        ]
    }
    
    result = ml_reasoner.reason(test_analysis)
    
    print(f"\nTest: chest pain + breathlessness")
    print(f"  ✅ Spécialiste prédit: {result['specialist']}")
    print(f"  ✅ Urgence prédite: {result['urgency']}")
    print(f"  ✅ Confiance: {result['confidence']:.1f}%")
    print(f"\n  📊 Top 3 spécialistes:")
    for spec, proba in result['model_probabilities']['top_3_specialists'].items():
        print(f"     • {spec}: {proba:.2%}")
    
    print("\n" + "=" * 60)
    print("✅ ENTRAÎNEMENT TERMINÉ!")
    print(f"✅ Modèle sauvegardé: {model_path}")
    print("=" * 60)
    print("\n💡 Prochaine étape: Modifier streamlit_app.py pour utiliser le modèle ML")

if __name__ == "__main__":
    main()