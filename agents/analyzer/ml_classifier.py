import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from typing import List, Dict, Tuple
import time

class MedicalMLClassifier:
    """
    Classifieur Médical basé sur Scikit-Learn (True NLP)
    Prédit le spécialiste et le niveau d'urgence à partir du texte patient.
    """
    
    def __init__(self):
        self.is_trained = False
        self.specialist_pipeline = None
        self.urgency_pipeline = None
        self.model_stats = {}
    
    def train(self, dataset: List[Dict]):
        """
        Entraîne les modèles sur le dataset complet
        """
        print("\n📈 [ML] Démarrage de l'entraînement des modèles IA...")
        start_time = time.time()
        
        # Préparation des données
        texts = []
        specialists = []
        urgencies = []
        
        for case in dataset:
            text = case.get('patient_text', '')
            specialist = case.get('specialist')
            urgency = case.get('urgency_level')
            
            if text and specialist and urgency:
                texts.append(text)
                specialists.append(specialist)
                urgencies.append(urgency)
        
        if not texts:
            print("⚠️ [ML] Erreur: Pas de données d'entraînement valides trouvées!")
            return
            
        print(f"   📊 Données d'entraînement: {len(texts)} exemples")
        
        # Pipeline Spécialiste
        self.specialist_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))),
            ('clf', RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42))
        ])
        
        # Pipeline Urgence
        self.urgency_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))),
            ('clf', RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42))
        ])
        
        # Entraînement
        print("   🧠 Entraînement du modèle Spécialiste...")
        self.specialist_pipeline.fit(texts, specialists)
        
        print("   🧠 Entraînement du modèle Urgence...")
        self.urgency_pipeline.fit(texts, urgencies)
        
        self.is_trained = True
        duration = time.time() - start_time
        
        print(f"✅ [ML] Entraînement terminé en {duration:.2f}s")
        
        # Sauvegarder les classes pour info
        self.model_stats = {
            'specialist_classes': self.specialist_pipeline.classes_.tolist(),
            'urgency_classes': self.urgency_pipeline.classes_.tolist(),
            'training_samples': len(texts)
        }
        
    def predict(self, text: str) -> Dict:
        """
        Prédit spécialiste et urgence pour un texte donné
        """
        if not self.is_trained:
            return {}
            
        X = [text]
        
        # Prédiction Spécialiste
        specialist = self.specialist_pipeline.predict(X)[0]
        specialist_proba = max(self.specialist_pipeline.predict_proba(X)[0])
        
        # Prédiction Urgence
        urgency = self.urgency_pipeline.predict(X)[0]
        urgency_proba = max(self.urgency_pipeline.predict_proba(X)[0])
        
        # Correction Encodage
        urgency = urgency.replace('Ã‰', 'É').replace('Ã¨', 'è').replace('Ã', 'à')
        if "MOD" in urgency and "R" in urgency and "E" in urgency:
             if "ELEV" not in urgency:
                 urgency = urgency.replace("MODÃ‰RÃ‰E", "MODÉRÉE").replace("MODÃ‰RÃ‰", "MODÉRÉE")
        
        return {
            'ml_specialist': specialist,
            'ml_specialist_confidence': float(specialist_proba),
            'ml_urgency': urgency,
            'ml_urgency_confidence': float(urgency_proba),
            'ml_used': True
        }
