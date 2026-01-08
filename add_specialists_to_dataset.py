"""
Script pour ajouter AUTOMATIQUEMENT le champ "specialist" 
à TOUS les cas du dataset en fonction des symptômes
==========================================================
"""

import json
import shutil
from datetime import datetime


# MAPPING AUTOMATIQUE symptôme → spécialiste
SYMPTOM_TO_SPECIALIST = {
    # Dentaire
    'tooth': 'Dentiste',
    'teeth': 'Dentiste',
    'gum': 'Dentiste',
    'dental': 'Dentiste',
    'jaw': 'Dentiste',
    
    # Cardiologue
    'chest': 'Cardiologue',
    'heart': 'Cardiologue',
    'cardiac': 'Cardiologue',
    'palpitation': 'Cardiologue',
    
    # Gastro
    'stomach': 'Gastro-entérologue',
    'abdomen': 'Gastro-entérologue',
    'belly': 'Gastro-entérologue',
    'nausea': 'Gastro-entérologue',
    'vomit': 'Gastro-entérologue',
    'diarrh': 'Gastro-entérologue',
    'constipat': 'Gastro-entérologue',
    
    # Neurologue
    'headache': 'Neurologue',
    'migraine': 'Neurologue',
    'dizziness': 'Neurologue',
    'head': 'Neurologue',
    'brain': 'Neurologue',
    
    # Pneumologue
    'breath': 'Pneumologue',
    'lung': 'Pneumologue',
    'respiratory': 'Pneumologue',
    'cough': 'Pneumologue',
    'phlegm': 'Pneumologue',
    
    # Dermatologue
    'skin': 'Dermatologue',
    'rash': 'Dermatologue',
    'itching': 'Dermatologue',
    'itchy': 'Dermatologue',
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
    'bone': 'Rhumatologue',
    'knee': 'Rhumatologue',
    'hip': 'Rhumatologue',
    'back': 'Rhumatologue',
    'neck': 'Rhumatologue',
}


def determine_specialist(symptoms):
    """Détermine automatiquement le spécialiste"""
    
    votes = []
    
    for symptom in symptoms:
        symptom_lower = symptom.lower()
        
        # Chercher correspondance
        for keyword, specialist in SYMPTOM_TO_SPECIALIST.items():
            if keyword in symptom_lower:
                votes.append(specialist)
                break
    
    # Vote majoritaire
    if votes:
        from collections import Counter
        most_common = Counter(votes).most_common(1)
        return most_common[0][0]
    
    return 'Médecin généraliste'


def add_specialists_to_dataset(dataset_path):
    """Ajoute automatiquement specialist à tous les cas"""
    
    print("="*70)
    print("🔧 AJOUT AUTOMATIQUE DES SPÉCIALISTES")
    print("="*70)
    
    # Backup
    backup_path = dataset_path.replace('.json', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    shutil.copy(dataset_path, backup_path)
    print(f"✅ Backup: {backup_path}")
    
    # Charger
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    print(f"\n📊 Total cas: {len(data)}")
    
    # Ajouter specialist
    specialists_added = 0
    specialist_counts = {}
    
    for case in data:
        symptoms = case.get('symptoms', [])
        
        if symptoms:
            # Déterminer spécialiste automatiquement
            specialist = determine_specialist(symptoms)
            
            # Ajouter au cas
            case['specialist'] = specialist
            specialists_added += 1
            
            # Compter
            specialist_counts[specialist] = specialist_counts.get(specialist, 0) + 1
    
    # Sauvegarder
    with open(dataset_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n✅ {specialists_added} spécialistes ajoutés!")
    
    # Afficher distribution
    print(f"\n📊 Distribution des spécialistes:")
    for specialist, count in sorted(specialist_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   • {specialist}: {count} cas")
    
    print("\n" + "="*70)
    print("✅ TERMINÉ!")
    print("="*70)


if __name__ == "__main__":
    dataset_path = "data/processed/dataset_processed.json"
    add_specialists_to_dataset(dataset_path)
    
    print("\n🚀 Relance Streamlit pour charger les spécialistes!")
