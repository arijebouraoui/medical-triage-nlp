"""
Interface Interactive de Triage Médical
========================================
Version 3.0 - Data-Driven avec rapports complets

Interface en ligne de commande pour:
- Consultation médicale interactive
- Support multilingue (FR/EN/AR/ES)
- Rapports complets avec:
  * Diagnostic
  * Niveau d'urgence
  * Spécialiste recommandé
  * Numéros d'urgence
  * Recommandations
  * Délai de consultation
"""

import sys
import os
from pathlib import Path

# Ajouter le chemin du projet
project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from medical_triage_system import MedicalTriageAI


class InteractiveTriageInterface:
    """Interface interactive pour le système de triage"""
    
    def __init__(self, patient_country: str = "Tunisie", patient_city: str = None):
        """
        Initialise l'interface
        
        Args:
            patient_country: Pays du patient
            patient_city: Ville du patient
        """
        self.system = None
        self.patient_country = patient_country
        self.patient_city = patient_city
        self.current_session = None
    
    def show_welcome(self):
        """Affiche le message de bienvenue"""
        print("\n" + "="*70)
        print("🏥 SYSTÈME DE TRIAGE MÉDICAL INTELLIGENT")
        print("   Version 3.0 - Data-Driven AI")
        print("="*70)
        print("\nBienvenue! Je suis votre assistant médical virtuel.")
        print("Je peux vous aider à comprendre vos symptômes et vous orienter.")
        print("\n📋 Fonctionnalités:")
        print("   • Support multilingue (Français, Anglais, Arabe, Espagnol)")
        print("   • Correction automatique des fautes d'orthographe")
        print("   • 4920 cas médicaux dans la base de données")
        print("   • Recommandations personnalisées")
        print("   • Numéros d'urgence de votre pays")
        print("\n⚠️  IMPORTANT: Ce système ne remplace pas un médecin!")
        print("   En cas de doute, consultez un professionnel de santé.")
        print("="*70)
    
    def show_instructions(self):
        """Affiche les instructions d'utilisation"""
        print("\n" + "─"*70)
        print("📝 COMMENT UTILISER CE SYSTÈME:")
        print("─"*70)
        print("\n1️⃣  Décrivez vos symptômes en langage naturel")
        print("   Exemples:")
        print("   • \"j'ai mal au ventre et je vomis\"")
        print("   • \"I have a headache and fever\"")
        print("   • \"Tengo dolor de cabeza\"")
        print("\n2️⃣  Vous pouvez ajouter des détails progressivement")
        print("   Le système se souviendra de vos symptômes précédents")
        print("\n3️⃣  Commandes disponibles:")
        print("   • 'quit' ou 'exit' → Quitter")
        print("   • 'help' → Afficher cette aide")
        print("   • 'new' → Nouvelle consultation (réinitialiser)")
        print("   • 'summary' → Voir le résumé de la session")
        print("   • 'stats' → Voir les statistiques du système")
        print("\n" + "─"*70)
    
    def show_examples(self):
        """Affiche des exemples d'utilisation"""
        print("\n" + "─"*70)
        print("💡 EXEMPLES D'UTILISATION:")
        print("─"*70)
        print("\n🇫🇷 Français:")
        print("   • \"j'ai mal à la tête depuis 2 jours\"")
        print("   • \"j'ai de la fièvre et je tousse\"")
        print("   • \"j'ai mal au ventre et des nausées\"")
        print("\n🇬🇧 English:")
        print("   • \"I have a severe headache\"")
        print("   • \"I'm having chest pain and difficulty breathing\"")
        print("   • \"I have stomach pain and vomiting\"")
        print("\n🇪🇸 Español:")
        print("   • \"Tengo dolor de cabeza y fiebre\"")
        print("   • \"Me duele el estómago\"")
        print("\n💡 Astuce: Pas besoin d'orthographe parfaite!")
        print("   Le système corrige automatiquement les fautes.")
        print("─"*70)
    
    def initialize_system(self):
        """Initialise le système médical"""
        print("\n⏳ Initialisation du système médical...")
        print("   (Cela peut prendre quelques secondes...)")
        
        try:
            self.system = MedicalTriageAI(
                patient_country=self.patient_country,
                patient_city=self.patient_city
            )
            return True
        except Exception as e:
            print(f"\n❌ ERREUR lors de l'initialisation: {e}")
            print("\n⚠️  Vérifiez que:")
            print("   1. Le fichier data/processed/dataset_processed.json existe")
            print("   2. Tous les modules sont correctement installés")
            print("   3. Vous êtes dans le bon dossier")
            return False
    
    def get_user_input(self) -> str:
        """Récupère l'input utilisateur"""
        print("\n" + "─"*70)
        print("💬 Décrivez vos symptômes (ou tapez 'help' pour aide):")
        print("─"*70)
        
        try:
            user_input = input("Vous: ").strip()
            return user_input
        except KeyboardInterrupt:
            print("\n\n👋 Au revoir!")
            return "quit"
        except Exception as e:
            print(f"\n❌ Erreur de saisie: {e}")
            return ""
    
    def process_command(self, user_input: str) -> bool:
        """
        Traite les commandes spéciales
        
        Args:
            user_input: Input utilisateur
        
        Returns:
            True si c'est une commande, False sinon
        """
        command = user_input.lower().strip()
        
        # Commande: quit/exit
        if command in ['quit', 'exit', 'q']:
            print("\n" + "="*70)
            print("👋 Merci d'avoir utilisé le système de triage médical")
            print("   Prenez soin de vous!")
            print("="*70 + "\n")
            return True
        
        # Commande: help
        elif command in ['help', 'aide', 'h', '?']:
            self.show_instructions()
            self.show_examples()
            return True
        
        # Commande: new (nouvelle consultation)
        elif command in ['new', 'nouveau', 'reset']:
            print("\n🔄 Nouvelle consultation...")
            self.current_session = None
            print("✅ Session réinitialisée!")
            return True
        
        # Commande: summary
        elif command in ['summary', 'résumé', 'resume']:
            if self.current_session:
                self.show_session_summary()
            else:
                print("\n⚠️  Aucune consultation en cours.")
                print("   Décrivez vos symptômes pour commencer.")
            return True
        
        # Commande: stats
        elif command in ['stats', 'statistics', 'statistiques']:
            self.show_system_stats()
            return True
        
        # Commande: examples
        elif command in ['examples', 'exemples', 'ex']:
            self.show_examples()
            return True
        
        return False
    
    def show_session_summary(self):
        """Affiche le résumé de la session en cours"""
        if not self.current_session:
            print("\n⚠️  Aucune session en cours")
            return
        
        try:
            summary = self.system.analyze_session(self.current_session)
            
            print("\n" + "="*70)
            print("📊 RÉSUMÉ DE LA CONSULTATION")
            print("="*70)
            print(f"\n🔖 Session ID: {summary['session_id']}")
            print(f"📝 Nombre d'échanges: {summary['total_turns']}")
            print(f"💊 Symptômes uniques identifiés: {summary['total_symptoms']}")
            
            # Liste des symptômes
            if summary['symptoms']:
                print(f"\n📋 Symptômes rapportés:")
                for i, symptom in enumerate(summary['symptoms'], 1):
                    symptom_name = symptom.get('symptom', 'inconnu')
                    confidence = symptom.get('confidence', 0)
                    print(f"   {i}. {symptom_name} (confiance: {confidence:.0%})")
            
            # Maladies possibles
            if summary['possible_diseases']:
                print(f"\n🏥 Top 3 maladies possibles:")
                for i, (disease, info) in enumerate(list(summary['possible_diseases'].items())[:3], 1):
                    print(f"   {i}. {disease}")
                    print(f"      • Score: {info['score']}/{summary['total_symptoms']} symptômes")
                    print(f"      • Urgence: {info['urgency']}")
            
            print("="*70)
            
        except Exception as e:
            print(f"\n❌ Erreur lors de la récupération du résumé: {e}")
    
    def show_system_stats(self):
        """Affiche les statistiques du système"""
        try:
            stats = self.system.get_system_statistics()
            
            print("\n" + "="*70)
            print("📊 STATISTIQUES DU SYSTÈME")
            print("="*70)
            print(f"\n📚 Base de données:")
            print(f"   • Cas médicaux: {stats['total_cases']}")
            print(f"   • Maladies référencées: {stats['total_diseases']}")
            print(f"   • Symptômes uniques: {stats['total_symptoms']}")
            print(f"   • Moyenne symptômes/cas: {stats['avg_symptoms_per_case']:.1f}")
            
            print(f"\n🚨 Distribution des urgences:")
            for urgency, count in sorted(stats['urgency_distribution'].items(), 
                                        key=lambda x: x[1], reverse=True):
                percentage = (count / stats['total_cases']) * 100
                print(f"   • {urgency}: {count} cas ({percentage:.1f}%)")
            
            print(f"\n🌍 Langues supportées:")
            print(f"   • Français 🇫🇷")
            print(f"   • Anglais 🇬🇧")
            print(f"   • Arabe 🇸🇦")
            print(f"   • Espagnol 🇪🇸")
            
            print(f"\n🔧 Fonctionnalités:")
            print(f"   • Correction orthographique: Levenshtein générique")
            print(f"   • Stemming: Porter/Snowball automatique")
            print(f"   • Détection de langue: Automatique")
            
            print("="*70)
            
        except Exception as e:
            print(f"\n❌ Erreur lors de la récupération des stats: {e}")
    
    def analyze_symptoms(self, user_input: str):
        """Analyse les symptômes du patient"""
        if not self.current_session:
            import random
            self.current_session = f"session_{random.randint(1000, 9999)}"
        
        try:
            print("\n⏳ Analyse en cours...")
            
            # Analyser avec le système complet
            report = self.system.analyze_and_respond(
                user_input,
                session_id=self.current_session
            )
            
            # Afficher le rapport
            print(report)
            
        except Exception as e:
            print(f"\n❌ ERREUR lors de l'analyse: {e}")
            print(f"\n🔍 Détails de l'erreur:")
            import traceback
            traceback.print_exc()
            print(f"\n💡 Suggestions:")
            print(f"   • Vérifiez que tous les fichiers sont à jour")
            print(f"   • Essayez de redémarrer le système")
            print(f"   • Tapez 'help' pour voir les exemples")
    
    def run(self):
        """Lance l'interface interactive"""
        # Bienvenue
        self.show_welcome()
        
        # Initialiser le système
        if not self.initialize_system():
            return
        
        # Instructions
        self.show_instructions()
        
        print("\n✅ Système prêt! Vous pouvez commencer.")
        
        # Boucle principale
        while True:
            # Récupérer input
            user_input = self.get_user_input()
            
            # Input vide
            if not user_input:
                continue
            
            # Traiter commandes
            if self.process_command(user_input):
                if user_input.lower().strip() in ['quit', 'exit', 'q']:
                    break
                continue
            
            # Analyser les symptômes
            self.analyze_symptoms(user_input)


# ==============================================================================
# POINT D'ENTRÉE
# ==============================================================================

def main():
    """Point d'entrée principal"""
    
    # Configuration par défaut
    default_country = "Tunisie"
    default_city = "Tunis"
    
    # Permettre de changer le pays via argument
    if len(sys.argv) > 1:
        default_country = sys.argv[1]
    
    if len(sys.argv) > 2:
        default_city = sys.argv[2]
    
    # Créer et lancer l'interface
    interface = InteractiveTriageInterface(
        patient_country=default_country,
        patient_city=default_city
    )
    
    interface.run()


if __name__ == "__main__":
    main()