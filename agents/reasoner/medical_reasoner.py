"""
Medical Reasoner 
"""

from typing import Dict, List
from collections import Counter


class MedicalReasoner:
    """Raisonnement médical avec PRIORITÉS ABSOLUES"""
    
    def __init__(self, data_loader=None):
        """Initialise et apprend du dataset"""
        
        self.data_loader = data_loader
        
        # PRIORITÉS ABSOLUES (overrident le dataset!)
        self.PRIORITY_SPECIALISTS = {
            'chest pain': 'Cardiologue',
            'heart attack': 'Cardiologue',
            'cardiac arrest': 'Cardiologue',
            'headache': 'Neurologue',  # ← FIX!
            'migraine': 'Neurologue',
            'toothache': 'Dentiste',
            'tooth pain': 'Dentiste',
            'gum pain': 'Dentiste',
            'gum bleeding': 'Dentiste',
        }
        
        # APPRENTISSAGE AUTOMATIQUE
        if data_loader:
            self._learn_from_dataset()
        else:
            self.symptom_to_specialist = {}
            self.symptom_to_urgency = {}
    
    def _learn_from_dataset(self):
        """Apprend automatiquement du dataset"""
        
        print("\n📚 Apprentissage du reasoner...")
        
        all_data = self.data_loader.dataset
        
        self.symptom_to_specialist = {}
        symptom_specialist_votes = {}
        
        self.symptom_to_urgency = {}
        symptom_urgency_votes = {}
        
        for case in all_data:
            symptoms = case.get('symptoms', [])
            specialist = case.get('specialist', None)
            urgency = case.get('urgency_level', 'Moderate')
            
            for symptom in symptoms:
                symptom_lower = symptom.lower()
                
                if specialist:
                    if symptom_lower not in symptom_specialist_votes:
                        symptom_specialist_votes[symptom_lower] = []
                    symptom_specialist_votes[symptom_lower].append(specialist)
                
                if urgency:
                    if symptom_lower not in symptom_urgency_votes:
                        symptom_urgency_votes[symptom_lower] = []
                    symptom_urgency_votes[symptom_lower].append(urgency)
        
        # Spécialiste le plus fréquent
        for symptom, votes in symptom_specialist_votes.items():
            most_common = Counter(votes).most_common(1)
            if most_common:
                self.symptom_to_specialist[symptom] = most_common[0][0]
        
        # Urgence la plus fréquente
        for symptom, votes in symptom_urgency_votes.items():
            most_common = Counter(votes).most_common(1)
            if most_common:
                self.symptom_to_urgency[symptom] = most_common[0][0]
        
        print(f"   ✅ {len(self.symptom_to_specialist)} symptômes → spécialistes")
        print(f"   ✅ {len(self.symptom_to_urgency)} symptômes → urgences")
        
        # Exemples
        print(f"\n   Exemples appris:")
        for symptom, specialist in list(self.symptom_to_specialist.items())[:5]:
            print(f"      • {symptom} → {specialist}")
    
    def reason(self, analysis: Dict) -> Dict:
        """Raisonnement médical"""
        
        symptoms = analysis.get('symptoms', [])
        diseases = analysis.get('possible_diseases', {})
        
        if not symptoms:
            return self._default_reasoning()
        
        # Déterminer spécialiste (AVEC PRIORITÉS + ML)
        specialist = self._determine_specialist_with_priority(symptoms, analysis)
        
        # Déterminer urgence (AVEC ML)
        urgency = self._determine_urgency(symptoms, diseases, analysis)
        
        # Recommandations
        recommendations = self._generate_recommendations(symptoms, specialist)
        
        # Timing
        timing = self._determine_timing(urgency)
        
        result = {
            'specialist': specialist,
            'urgency': urgency,
            'timing': timing,
            'recommendations': recommendations,
            'confidence': self._calculate_confidence(symptoms, diseases),
        }
        
        return result
    
    def _determine_specialist_with_priority(self, symptoms: List[Dict], analysis: Dict = None) -> str:
        """
        FIX: Vérifie PRIORITÉS en PREMIER, puis ML, puis Voting
        =======================================================
        """
        # 1. VÉRIFIER PRIORITÉS ABSOLUES (Règles métiers strictes)
        for symptom in symptoms:
            symptom_name = symptom['symptom'].lower()
            
            # Priorité exacte
            if symptom_name in self.PRIORITY_SPECIALISTS:
                priority_specialist = self.PRIORITY_SPECIALISTS[symptom_name]
                print(f"   🎯 PRIORITÉ: {symptom_name} → {priority_specialist}")
                return priority_specialist
        
        # 2. IA / ML (Si disponible et confiant)
        if analysis and analysis.get('ml_used'):
            ml_spec = analysis.get('ml_specialist')
            ml_conf = analysis.get('ml_specialist_confidence', 0)
            
            # Si l'IA est sûre d'elle (> 40% car il y a beaucoup de classes)
            if ml_spec and ml_conf > 0.4:
                print(f"   🤖 AI CHOICE: {ml_spec} (conf={ml_conf:.0%})")
                return ml_spec
        
        # 3. Si pas de priorité ni ML, utiliser apprentissage (Voting)
        return self._determine_specialist(symptoms)
    
    def _determine_specialist(self, symptoms: List[Dict]) -> str:
        """Détermine spécialiste EN APPRENANT du dataset"""
        
        specialist_votes = []
        
        for symptom in symptoms:
            symptom_name = symptom['symptom'].lower()
            
            if symptom_name in self.symptom_to_specialist:
                specialist_votes.append(self.symptom_to_specialist[symptom_name])
            else:
                for known_symptom, specialist in self.symptom_to_specialist.items():
                    if any(word in symptom_name for word in known_symptom.split()):
                        specialist_votes.append(specialist)
                        break
        
        if specialist_votes:
            most_common = Counter(specialist_votes).most_common(1)
            return most_common[0][0]
        
        return 'Médecin généraliste'
    
    def _determine_urgency(self, symptoms: List[Dict], diseases: Dict, analysis: Dict = None) -> str:
        """Détermine urgence EN APPRENANT du dataset + ML"""
        
        # 1. IA / ML (Priorité à l'IA pour l'urgence globale du texte)
        if analysis and analysis.get('ml_used'):
            ml_urgency = analysis.get('ml_urgency')
            ml_conf = analysis.get('ml_urgency_confidence', 0)
            
            if ml_urgency and ml_conf > 0.5:
                # Vérifier si "URGENCE VITALE" n'est pas manqué par l'IA
                # (Safety check avec keywords)
                if ml_urgency != 'URGENCE VITALE':
                    for symptom in symptoms:
                        if symptom['symptom'].lower() in ['chest pain', 'heart attack', 'poison']:
                             return 'URGENCE VITALE'
                
                print(f"   🤖 AI URGENCY: {ml_urgency} (conf={ml_conf:.0%})")
                return ml_urgency
        
        urgency_votes = []
        
        for symptom in symptoms:
            symptom_name = symptom['symptom'].lower()
            
            if symptom_name in self.symptom_to_urgency:
                urgency_votes.append(self.symptom_to_urgency[symptom_name])
        
        if urgency_votes:
            urgency_counter = Counter(urgency_votes)
            
            if 'High' in urgency_counter or 'Vital' in urgency_counter:
                return 'URGENCE ÉLEVÉE'
            
            most_common = urgency_counter.most_common(1)
            urgency = most_common[0][0]
            
            if urgency == 'High':
                return 'URGENCE ÉLEVÉE'
            elif urgency == 'Low':
                return 'URGENCE FAIBLE'
            else:
                return 'URGENCE MODÉRÉE'
        
        return 'URGENCE MODÉRÉE'
    
    def _determine_timing(self, urgency: str) -> str:
        """Détermine délai"""
        
        if 'VITALE' in urgency or 'VITAL' in urgency:
            return 'IMMÉDIAT (Appeler les urgences)'
        elif 'ÉLEVÉE' in urgency or 'HIGH' in urgency:
            return 'Aujourd\'hui même'
        elif 'MODÉRÉE' in urgency or 'MODERATE' in urgency:
            return '24-48 heures'
        else:
            return 'Cette semaine'
    
    def _generate_recommendations(self, symptoms: List[Dict], specialist: str) -> List[str]:
        """Recommandations par spécialiste"""
        
        recommendations_map = {
            'Dentiste': [
                'Éviter les aliments trop chauds ou froids',
                'Brossage doux des dents',
                'Bain de bouche antiseptique',
                'Ne pas mâcher du côté douloureux',
            ],
            'Cardiologue': [
                'Repos complet',
                'Éviter tout effort physique',
                'Ne pas fumer',
                'Surveiller la pression artérielle',
            ],
            'Gastro-entérologue': [
                'Éviter les aliments épicés et gras',
                'Boire beaucoup d\'eau',
                'Repos digestif pendant 24h',
                'Manger léger (riz, bananes, toast)',
            ],
            'Neurologue': [
                'Repos dans un endroit calme et sombre',
                'Hydratation régulière',
                'Éviter les écrans',
                'Noter les symptômes et leur fréquence',
            ],
            'Pneumologue': [
                'Rester au chaud',
                'Boire des liquides chauds',
                'Repos',
                'Éviter les irritants (fumée, pollution)',
            ],
            'ORL': [
                'Repos vocal',
                'Humidifier l\'air',
                'Boire chaud (thé, tisane)',
                'Éviter les irritants',
            ],
        }
        
        return recommendations_map.get(specialist, [
            'Repos',
            'Hydratation régulière',
            'Surveiller l\'évolution des symptômes',
            'Alimentation équilibrée',
        ])
    
    def _calculate_confidence(self, symptoms: List[Dict], diseases: Dict) -> float:
        """Calcule confiance"""
        
        if not symptoms:
            return 50.0
        
        if diseases:
            return 100.0
        
        try:
            avg_confidence = sum(s.get('confidence', 0) for s in symptoms) / len(symptoms)
            return avg_confidence * 100
        except:
            return 75.0
    
    def _default_reasoning(self) -> Dict:
        """Raisonnement par défaut"""
        
        return {
            'specialist': 'Médecin généraliste',
            'urgency': 'URGENCE MODÉRÉE',
            'timing': '24-48 heures',
            'recommendations': [
                'Repos',
                'Surveiller l\'évolution des symptômes',
                'Hydratation régulière',
                'Alimentation équilibrée',
            ],
            'confidence': 50.0,
        }