"""
Setup Dataset - Ajoute automatiquement les spécialistes
========================================================
Exécute ce script UNE FOIS après installation
"""

import json
import shutil
from datetime import datetime
from collections import Counter

# MAPPING INTELLIGENT: Symptômes prioritaires
PRIORITY_SPECIALISTS = {
    'chest pain': 'Cardiologue',
    'heart attack': 'Cardiologue',
    'cardiac': 'Cardiologue',
    'toothache': 'Dentiste',
    'tooth pain': 'Dentiste',
    'gum bleeding': 'Dentiste',
}

# MAPPING PAR MOTS-CLÉS
KEYWORD_SPECIALISTS = {
    # Dentaire
    'tooth': 'Dentiste',
    'teeth': 'Dentiste',
    'gum': 'Dentiste',
    'dental': 'Dentiste',
    'jaw': 'Dentiste',
    
    # Cardiologue  
    'chest': 'Cardiologue',
    'heart': 'Cardiologue',
    'palpitation': 'Cardiologue',
    
    # Gastro
    'stomach': 'Gastro-entérologue',
    'abdomen': 'Gastro-entérologue',
    'belly': 'Gastro-entérologue',
    'nausea': 'Gastro-entérologue',
    'vomit': 'Gastro-entérologue',
    'diarrh': 'Gastro-entérologue',
    
    # Neurologue
    'headache': 'Neurologue',
    'head': 'Neurologue',
    'migraine': 'Neurologue',
    'dizz': 'Neurologue',
    
    # Pneumologue
    'breath': 'Pneumologue',
    'lung': 'Pneumologue',
    'cough': 'Pneumologue',
    'phlegm': 'Pneumologue',
    
    # Dermatologue
    'skin': 'Dermatologue',
    'rash': 'Dermatologue',
    'itch': 'Dermatologue',
    'pimple': 'Dermatologue',
    
    # Ophtalmologue
    'eye': 'Ophtalmologue',
    'vision': 'Ophtalmologue',
    'blurred': 'Ophtalmologue',
    
    # ORL
    'throat': 'ORL',
    'ear': 'ORL',
    'nose': 'ORL',
    'sinus': 'ORL',
    
    # Urologue
    'urin': 'Urologue',
    'bladder': 'Urologue',
    'micturit': 'Urologue',
    
    # Rhumatologue
    'joint': 'Rhumatologue',
    'muscle': 'Rhumatologue',
    'knee': 'Rhumatologue',
    'back': 'Rhumatologue',
    'neck': 'Rhumatologue',
}


def determine_specialist(symptoms):
    """Détermine le spécialiste avec priorités"""
    
    # 1. Vérifier priorités absolues
    for symptom in symptoms:
        symptom_lower = symptom.lower()
        if symptom_lower in PRIORITY_SPECIALISTS:
            return PRIORITY_SPECIALISTS[symptom_lower]
    
    # 2. Vote par mots-clés
    votes = []
    for symptom in symptoms:
        symptom_lower = symptom.lower()
        for keyword, specialist in KEYWORD_SPECIALISTS.items():
            if keyword in symptom_lower:
                votes.append(specialist)
                break
    
    # 3. Vote majoritaire
    if votes:
        most_common = Counter(votes).most_common(1)
        return most_common[0][0]
    
    return 'Médecin généraliste'


def setup_dataset():
    """Configure le dataset automatiquement"""
    
    print("="*70)
    print("🔧 SETUP AUTOMATIQUE DU DATASET")
    print("="*70)
    
    dataset_path = "data/processed/dataset_processed.json"
    
    # Backup
    backup_path = dataset_path.replace('.json', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    try:
        shutil.copy(dataset_path, backup_path)
        print(f"✅ Backup créé: {backup_path}")
    except FileNotFoundError:
        print(f"⚠️  Dataset non trouvé à: {dataset_path}")
        print("   Assure-toi que le dataset existe!")
        return False
    
    # Charger
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\n📊 Dataset: {len(data)} cas")
    
    # Ajouter spécialistes
    specialists_added = 0
    specialist_counts = {}
    
    for case in data:
        symptoms = case.get('symptoms', [])
        
        if symptoms:
            specialist = determine_specialist(symptoms)
            case['specialist'] = specialist
            specialists_added += 1
            specialist_counts[specialist] = specialist_counts.get(specialist, 0) + 1
    
    # Sauvegarder
    with open(dataset_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ {specialists_added} spécialistes ajoutés!")
    
    # Distribution
    print(f"\n📊 Distribution des spécialistes:")
    for specialist, count in sorted(specialist_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   • {specialist}: {count} cas")
    
    print("\n" + "="*70)
    print("✅ SETUP TERMINÉ!")
    print("="*70)
    print("\n🚀 Lance maintenant: streamlit run streamlit_app.py")
    
    return True


if __name__ == "__main__":
    success = setup_dataset()
    
    if not success:
        print("\n❌ Setup échoué! Vérifie que le dataset existe.")
        exit(1)