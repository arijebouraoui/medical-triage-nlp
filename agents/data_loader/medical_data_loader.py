"""
Medical Data Loader
===================
Charge et indexe le dataset Kaggle (dataset_processed.json)
Fournit une recherche rapide de symptômes et maladies
"""

import json
import os
from typing import List, Dict, Set, Optional
from collections import defaultdict
import re


class MedicalDataLoader:
    """Charge et indexe les données médicales depuis dataset_processed.json"""
    
    def __init__(self, data_path: str = "data/processed/dataset_processed.json"):
        """
        Args:
            data_path: Chemin vers dataset_processed.json
        """
        self.data_path = data_path
        self.dataset = []
        self.diseases = set()
        self.symptoms = set()
        self.urgency_levels = set()
        
        # Index pour recherche rapide
        self.symptom_to_diseases = defaultdict(list)
        self.disease_to_symptoms = defaultdict(list)
        self.disease_to_urgency = {}
        
        # Charger les données
        self._load_data()
        self._build_indexes()
    
    def _load_data(self):
        """Charge le fichier JSON"""
        if not os.path.exists(self.data_path):
            print(f"⚠️  ATTENTION: {self.data_path} n'existe pas!")
            print(f"   Le système utilisera les données par défaut.")
            return
        
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                self.dataset = json.load(f)
            
            print(f"✅ Dataset chargé: {len(self.dataset)} cas médicaux")
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement: {e}")
            self.dataset = []
    
    def _build_indexes(self):
        """Construit les index pour recherche rapide"""
        for case in self.dataset:
            disease = case.get('disease', '').strip()
            symptoms = case.get('symptoms', [])
            urgency = case.get('urgency_level', 'URGENCE MODÉRÉE')
            
            # Ajouter aux ensembles
            self.diseases.add(disease)
            self.urgency_levels.add(urgency)
            
            # Index symptôme → maladies
            for symptom in symptoms:
                symptom_clean = symptom.strip().lower()
                self.symptoms.add(symptom_clean)
                self.symptom_to_diseases[symptom_clean].append(disease)
            
            # Index maladie → symptômes
            self.disease_to_symptoms[disease] = symptoms
            
            # Index maladie → urgence
            self.disease_to_urgency[disease] = urgency
        
        print(f"📊 Index créé:")
        print(f"   🏥 Maladies: {len(self.diseases)}")
        print(f"   💊 Symptômes uniques: {len(self.symptoms)}")
        print(f"   🚨 Niveaux d'urgence: {len(self.urgency_levels)}")
    
    def get_all_symptoms(self) -> List[str]:
        """Retourne tous les symptômes connus"""
        return sorted(list(self.symptoms))
    
    def get_all_diseases(self) -> List[str]:
        """Retourne toutes les maladies connues"""
        return sorted(list(self.diseases))
    
    def find_diseases_by_symptom(self, symptom: str) -> List[str]:
        """
        Trouve les maladies associées à un symptôme
        
        Args:
            symptom: Nom du symptôme (ex: "headache")
        
        Returns:
            Liste de maladies possibles
        """
        symptom_clean = symptom.strip().lower()
        return self.symptom_to_diseases.get(symptom_clean, [])
    
    def find_diseases_by_symptoms(self, symptoms: List[str]) -> Dict[str, Dict]:
        """
        Trouve les maladies qui correspondent à une liste de symptômes
        
        Args:
            symptoms: Liste de symptômes
        
        Returns:
            Dict {disease: {score, urgency, matched_symptoms}}
        """
        disease_matches = defaultdict(lambda: {
            'score': 0,
            'urgency': 'URGENCE MODÉRÉE',
            'matched_symptoms': []
        })
        
        for symptom in symptoms:
            symptom_clean = symptom.strip().lower()
            diseases = self.symptom_to_diseases.get(symptom_clean, [])
            
            for disease in diseases:
                disease_matches[disease]['score'] += 1
                disease_matches[disease]['urgency'] = self.disease_to_urgency.get(
                    disease, 'URGENCE MODÉRÉE'
                )
                disease_matches[disease]['matched_symptoms'].append(symptom)
        
        # Trier par score (nombre de symptômes matchés)
        sorted_diseases = dict(sorted(
            disease_matches.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        ))
        
        return sorted_diseases
    
    def get_symptom_variations(self, symptom: str) -> List[str]:
        """
        Trouve les variations d'un symptôme dans le dataset
        
        Args:
            symptom: Symptôme de base (ex: "pain")
        
        Returns:
            Liste de variations (ex: ["joint pain", "stomach pain", ...])
        """
        variations = []
        symptom_lower = symptom.strip().lower()
        
        for known_symptom in self.symptoms:
            if symptom_lower in known_symptom:
                variations.append(known_symptom)
        
        return sorted(variations)
    
    def get_urgency_for_disease(self, disease: str) -> str:
        """
        Retourne le niveau d'urgence pour une maladie
        
        Args:
            disease: Nom de la maladie
        
        Returns:
            Niveau d'urgence (URGENCE ÉLEVÉE, MODÉRÉE, etc.)
        """
        return self.disease_to_urgency.get(disease, 'URGENCE MODÉRÉE')
    
    def search_symptoms(self, query: str) -> List[str]:
        """
        Recherche de symptômes par mot-clé
        
        Args:
            query: Mot-clé à rechercher
        
        Returns:
            Liste de symptômes contenant le mot-clé
        """
        query_lower = query.strip().lower()
        matches = []
        
        for symptom in self.symptoms:
            if query_lower in symptom:
                matches.append(symptom)
        
        return sorted(matches)
    
    def get_case_by_id(self, case_id: int) -> Optional[Dict]:
        """
        Récupère un cas par son ID
        
        Args:
            case_id: ID du cas
        
        Returns:
            Dict contenant le cas ou None
        """
        for case in self.dataset:
            if case.get('id') == case_id:
                return case
        return None
    
    def get_statistics(self) -> Dict:
        """Retourne des statistiques sur le dataset"""
        urgency_counts = defaultdict(int)
        for case in self.dataset:
            urgency = case.get('urgency_level', 'UNKNOWN')
            urgency_counts[urgency] += 1
        
        return {
            'total_cases': len(self.dataset),
            'total_diseases': len(self.diseases),
            'total_symptoms': len(self.symptoms),
            'urgency_distribution': dict(urgency_counts),
            'avg_symptoms_per_case': sum(
                len(case.get('symptoms', [])) for case in self.dataset
            ) / len(self.dataset) if self.dataset else 0
        }


# ==============================================================================
# EXEMPLE D'UTILISATION
# ==============================================================================

if __name__ == "__main__":
    # Charger les données
    loader = MedicalDataLoader("data/processed/dataset_processed.json")
    
    print("\n" + "="*70)
    print("📊 STATISTIQUES DU DATASET")
    print("="*70)
    
    stats = loader.get_statistics()
    print(f"\n📈 Cas totaux: {stats['total_cases']}")
    print(f"🏥 Maladies: {stats['total_diseases']}")
    print(f"💊 Symptômes uniques: {stats['total_symptoms']}")
    print(f"📊 Moyenne symptômes/cas: {stats['avg_symptoms_per_case']:.1f}")
    
    print(f"\n🚨 Distribution des urgences:")
    for urgency, count in sorted(stats['urgency_distribution'].items(), 
                                   key=lambda x: x[1], reverse=True):
        print(f"   {urgency}: {count} cas")
    
    # Test de recherche
    print("\n" + "="*70)
    print("🔍 TEST DE RECHERCHE")
    print("="*70)
    
    test_symptoms = ["headache", "nausea", "vomiting"]
    print(f"\nSymptômes du patient: {test_symptoms}")
    
    diseases = loader.find_diseases_by_symptoms(test_symptoms)
    print(f"\n🏥 Top 5 maladies possibles:")
    for i, (disease, info) in enumerate(list(diseases.items())[:5], 1):
        print(f"\n{i}. {disease}")
        print(f"   Score: {info['score']}/{len(test_symptoms)} symptômes")
        print(f"   Urgence: {info['urgency']}")
        print(f"   Symptômes matchés: {', '.join(info['matched_symptoms'])}")
    
    # Recherche de variations
    print("\n" + "="*70)
    print("🔍 VARIATIONS DE SYMPTÔMES")
    print("="*70)
    
    pain_variations = loader.get_symptom_variations("pain")
    print(f"\nVariations de 'pain': {len(pain_variations)} trouvées")
    print(f"Exemples: {pain_variations[:10]}")