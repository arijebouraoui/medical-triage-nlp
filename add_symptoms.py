"""
Script pour ajouter des symptômes au dataset

"""

import json
import shutil
from datetime import datetime

# Nouveaux cas à ajouter
new_cases = [
    # Dentaire
    {
        "symptoms": ["toothache", "jaw pain"],
        "disease": "Dental Cavity",
        "urgency_level": "Moderate",
        "specialist": "Dentiste"
    },
    {
        "symptoms": ["gum bleeding", "swollen gums", "gum pain"],
        "disease": "Gingivitis",
        "urgency_level": "Moderate",
        "specialist": "Dentiste"
    },
    {
        "symptoms": ["tooth pain", "sensitive teeth"],
        "disease": "Tooth Sensitivity",
        "urgency_level": "Low",
        "specialist": "Dentiste"
    },
    {
        "symptoms": ["bleeding gums"],
        "disease": "Gum Disease",
        "urgency_level": "Moderate",
        "specialist": "Dentiste"
    },
    
    # ORL
    {
        "symptoms": ["sore throat", "difficulty swallowing"],
        "disease": "Pharyngitis",
        "urgency_level": "Moderate",
        "specialist": "ORL"
    },
    {
        "symptoms": ["ear pain", "hearing loss"],
        "disease": "Ear Infection",
        "urgency_level": "Moderate",
        "specialist": "ORL"
    },
]


def add_symptoms_to_dataset(dataset_path: str, backup: bool = True):
    """Ajoute nouveaux symptômes au dataset"""
    
    print("="*70)
    print("📊 AJOUT DE SYMPTÔMES AU DATASET")
    print("="*70)
    
    # Backup
    if backup:
        backup_path = dataset_path.replace('.json', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        shutil.copy(dataset_path, backup_path)
        print(f"✅ Backup créé: {backup_path}")
    
    # Charger dataset
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    print(f"\n📖 Dataset actuel: {len(data)} cas")
    
    # Compter symptômes uniques actuels
    current_symptoms = set()
    for case in data:
        if 'symptoms' in case:
            for s in case['symptoms']:
                current_symptoms.add(s.lower())
    
    print(f"📋 Symptômes uniques: {len(current_symptoms)}")
    
    # Ajouter nouveaux cas
    print(f"\n➕ Ajout de {len(new_cases)} nouveaux cas...")
    
    for i, new_case in enumerate(new_cases, 1):
        data.append(new_case)
        print(f"   {i}. {new_case['disease']}: {', '.join(new_case['symptoms'][:3])}")
    
    # Compter nouveaux symptômes
    new_symptoms = set()
    for case in data:
        if 'symptoms' in case:
            for s in case['symptoms']:
                new_symptoms.add(s.lower())
    
    added_symptoms = new_symptoms - current_symptoms
    
    # Sauvegarder
    with open(dataset_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n✅ Dataset mis à jour!")
    print(f"   • Total cas: {len(data)} (+{len(new_cases)})")
    print(f"   • Total symptômes uniques: {len(new_symptoms)} (+{len(added_symptoms)})")
    
    if added_symptoms:
        print(f"\n📝 Nouveaux symptômes ajoutés:")
        for s in sorted(added_symptoms):
            print(f"   • {s}")
    
    print("="*70)


if __name__ == "__main__":
    # Chemin du dataset
    dataset_path = "data/processed/dataset_processed.json"
    
    # Ajouter symptômes
    add_symptoms_to_dataset(dataset_path, backup=True)
    
    print("\n✨ Terminé! Relance le système pour charger les nouveaux symptômes.")