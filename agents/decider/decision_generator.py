"""
Decision Generator 
====================================

"""

from typing import Dict, List


class DecisionGenerator:
    """Génère des rapports médicaux multilingues"""
    
    def __init__(self, patient_country: str = "Tunisie", patient_city: str = None):
        self.patient_country = patient_country
        self.patient_city = patient_city
        
        # Numéros d'urgence CORRECTS
        self.emergency_numbers = {
            'Tunisie': {
                'samu': '190',
                'urgences': '197',
                'police': '197',
                'pompiers': '198'
            },
            'France': {
                'samu': '15',
                'urgences': '112',
                'police': '17',
                'pompiers': '18'
            },
            'UK': {
                'emergency': '999',
                'urgences': '112',
                'police': '999',
                'ambulance': '999',
                'fire': '999'
            },
            'USA': {
                'emergency': '911',
                'police': '911',
                'ambulance': '911',
                'fire': '911'
            },
            'Canada': {
                'emergency': '911',
                'police': '911',
                'ambulance': '911',
                'fire': '911'
            },
        }
        
        # Traductions
        self.translations = {
            'en': {
                'medical_report': 'MEDICAL REPORT',
                'patient_info': 'PATIENT INFORMATION',
                'symptoms_reported': 'Reported symptoms',
                'specialist_recommended': 'RECOMMENDED SPECIALIST',
                'consultation': 'Consultation',
                'recommended_delay': 'Recommended delay',
                'emergency_numbers': 'EMERGENCY NUMBERS',
                'recommendations': 'RECOMMENDATIONS',
            },
            'fr': {
                'medical_report': 'RAPPORT MÉDICAL',
                'patient_info': 'INFORMATIONS PATIENT',
                'symptoms_reported': 'Symptômes rapportés',
                'specialist_recommended': 'SPÉCIALISTE RECOMMANDÉ',
                'consultation': 'Consultation',
                'recommended_delay': 'Délai recommandé',
                'emergency_numbers': 'NUMÉROS D\'URGENCE',
                'recommendations': 'RECOMMANDATIONS',
            }
        }
    
    def generate_decision(self, reasoning: Dict) -> str:
        """Génère un rapport médical"""
        
        language = reasoning.get('language', 'fr')
        if language not in ['en', 'fr']:
            language = 'en'
        
        symptoms = reasoning.get('symptoms', [])
        urgency = reasoning.get('urgency', 'URGENCE MODÉRÉE')
        specialist = reasoning.get('specialist', 'Médecin généraliste')
        timing = reasoning.get('timing', '24-48 heures')
        recommendations = reasoning.get('recommendations', [])
        
        t = self.translations[language]
        
        lines = []
        
        lines.append("")
        lines.append("╔" + "═"*68 + "╗")
        lines.append("║" + t['medical_report'].center(68) + "║")
        lines.append("╚" + "═"*68 + "╝")
        lines.append("")
        
        lines.append(f"👤 {t['patient_info']}")
        lines.append("─" * 70)
        
        symptom_names = [s.get('symptom', '') for s in symptoms if isinstance(s, dict)]
        symptom_text = ', '.join(symptom_names[:5]) if symptom_names else "Non spécifié"
        
        lines.append(f"   {t['symptoms_reported']}: {symptom_text}")
        lines.append("")
        
        lines.append(f"⚕️  {t['specialist_recommended']}")
        lines.append("─" * 70)
        lines.append(f"   {t['consultation']}: {specialist}")
        lines.append(f"   {t['recommended_delay']}: {timing}")
        lines.append("")
        
        lines.append(f"🚨 {t['emergency_numbers']} ({self.patient_country.upper()})")
        lines.append("─" * 70)
        
        emergency_nums = self.emergency_numbers.get(self.patient_country, self.emergency_numbers.get('Tunisie'))
        
        for key, value in emergency_nums.items():
            label = key.capitalize()
            lines.append(f"   {label}: {value}")
        lines.append("")
        
        if recommendations:
            lines.append(f"💊 {t['recommendations']}")
            lines.append("─" * 70)
            for i, rec in enumerate(recommendations, 1):
                lines.append(f"   {i}. {rec}")
            lines.append("")
        
        lines.append("─" * 70)
        lines.append("ℹ️  Ce rapport est généré automatiquement par IA")
        lines.append("   Il ne remplace pas l'avis d'un professionnel de santé")
        lines.append("─" * 70)
        lines.append("")
        
        return "\n".join(lines)