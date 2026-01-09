"""
Raisonnement médical avec Random Forest Classifier
Auteur: Arije Bouraoui
Date: Janvier 2026
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json
from typing import Dict, List
from collections import Counter

class MLMedicalReasoner:
    """
    Raisonnement basé sur Random Forest
    
    Apprentissage supervisé:
    - Features: Symptômes (one-hot encoding)
    - Target: Spécialiste
    """
    
    def __init__(self, data_loader=None, model_path=None):
        """
        Initialise le reasoner ML
        
        Args:
            data_loader: Instance de MedicalDataLoader
            model_path: Chemin vers modèle sauvegardé (optionnel)
        """
        self.data_loader = data_loader
        self.model_specialist = None
        self.model_urgency = None
        self.mlb_symptoms = MultiLabelBinarizer()
        self.specialist_classes = []
        self.urgency_classes = []
        
        # Charger modèle existant ou entraîner nouveau
        if model_path:
            self.load_model(model_path)
        elif data_loader:
            self.train_models()
    
    def prepare_features(self, dataset):
        """
        Prépare features (X) et targets (y) depuis dataset
        
        Returns:
            X_symptoms: Array (n_samples, n_symptoms) - one-hot encoding
            y_specialist: Array (n_samples,) - labels spécialistes
            y_urgency: Array (n_samples,) - labels urgence
        """
        symptoms_list = []
        specialists = []
        urgencies = []
        
        for case in dataset:
            symptoms = case.get('symptoms', [])
            specialist = case.get('specialist')
            urgency = case.get('urgency_level')
            
            if symptoms and specialist and urgency:
                symptoms_list.append(symptoms)
                specialists.append(specialist)
                urgencies.append(urgency)
        
        # One-hot encoding des symptômes
        X_symptoms = self.mlb_symptoms.fit_transform(symptoms_list)
        
        # Targets
        y_specialist = np.array(specialists)
        y_urgency = np.array(urgencies)
        
        # Sauvegarder classes
        self.specialist_classes = list(set(specialists))
        self.urgency_classes = list(set(urgencies))
        
        return X_symptoms, y_specialist, y_urgency
    
    def train_models(self):
        """
        ENTRAÎNE Random Forest pour:
        1. Prédiction spécialiste
        2. Prédiction urgence
        """
        print("🔧 Entraînement des modèles Random Forest...")
        
        # Charger dataset (utiliser l'attribut dataset directement)
        dataset = self.data_loader.dataset
        
        # Préparer features
        X, y_specialist, y_urgency = self.prepare_features(dataset)
        
        print(f"📊 Features shape: {X.shape}")
        print(f"📊 Nombre de symptômes: {len(self.mlb_symptoms.classes_)}")
        print(f"📊 Nombre de spécialistes: {len(self.specialist_classes)}")
        
        # Split train/test (80/20)
        X_train, X_test, y_spec_train, y_spec_test, y_urg_train, y_urg_test = \
            train_test_split(X, y_specialist, y_urgency, 
                           test_size=0.2, random_state=42, stratify=y_specialist)
        
        print(f"📊 Train set: {len(X_train)} cas")
        print(f"📊 Test set: {len(X_test)} cas")
        
        # ========================================
        # MODÈLE 1: PRÉDICTION SPÉCIALISTE
        # ========================================
        print("\n📊 Entraînement modèle spécialiste...")
        
        self.model_specialist = RandomForestClassifier(
            n_estimators=100,        # 100 arbres
            max_depth=20,            # Profondeur max
            min_samples_split=10,    # Min échantillons pour split
            min_samples_leaf=5,      # Min échantillons par feuille
            random_state=42,
            n_jobs=-1,               # Utiliser tous les CPU
            class_weight='balanced'  # Équilibrer classes déséquilibrées
        )
        
        self.model_specialist.fit(X_train, y_spec_train)
        
        # Évaluation
        y_pred_spec = self.model_specialist.predict(X_test)
        accuracy_spec = accuracy_score(y_spec_test, y_pred_spec)
        
        print(f"✅ Accuracy spécialiste: {accuracy_spec:.2%}")
        print("\n📊 Rapport détaillé:")
        print(classification_report(y_spec_test, y_pred_spec, zero_division=0))
        
        # ========================================
        # MODÈLE 2: PRÉDICTION URGENCE
        # ========================================
        print("\n📊 Entraînement modèle urgence...")
        
        self.model_urgency = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        self.model_urgency.fit(X_train, y_urg_train)
        
        # Évaluation
        y_pred_urg = self.model_urgency.predict(X_test)
        accuracy_urg = accuracy_score(y_urg_test, y_pred_urg)
        
        print(f"✅ Accuracy urgence: {accuracy_urg:.2%}")
        print("\n📊 Rapport détaillé:")
        print(classification_report(y_urg_test, y_pred_urg, zero_division=0))
        
        # ========================================
        # FEATURE IMPORTANCE
        # ========================================
        print("\n📈 Top 10 symptômes les plus importants:")
        
        feature_names = self.mlb_symptoms.classes_
        importances = self.model_specialist.feature_importances_
        
        # Trier par importance
        indices = np.argsort(importances)[::-1][:10]
        
        for i, idx in enumerate(indices, 1):
            print(f"  {i}. {feature_names[idx]}: {importances[idx]:.4f}")
        
        return {
            'specialist_accuracy': accuracy_spec,
            'urgency_accuracy': accuracy_urg
        }
    
    def reason(self, analysis: dict) -> dict:
        """
        RAISONNEMENT ML (remplace logique actuelle)
        
        Args:
            analysis: Résultat de nlp_analyzer
                {
                    'symptoms': [
                        {'symptom': 'chest pain', 'confidence': 0.95},
                        ...
                    ]
                }
        
        Returns:
            {
                'specialist': str,
                'urgency': str,
                'timing': str,
                'recommendations': list,
                'confidence': float,
                'model_probabilities': dict
            }
        """
        symptoms = analysis.get('symptoms', [])
        
        if not symptoms:
            return self._default_reasoning()
        
        # Extraire noms symptômes
        symptom_names = [s['symptom'] for s in symptoms]
        
        # Transformer en features (one-hot)
        X = self.mlb_symptoms.transform([symptom_names])
        
        # ========================================
        # PRÉDICTION SPÉCIALISTE avec Random Forest
        # ========================================
        specialist_pred = self.model_specialist.predict(X)[0]
        specialist_proba = self.model_specialist.predict_proba(X)[0]
        
        # Confiance = probabilité prédiction
        specialist_confidence = specialist_proba.max()
        
        # Top 3 spécialistes avec probabilités
        specialist_classes = self.model_specialist.classes_
        top_3_indices = np.argsort(specialist_proba)[::-1][:3]
        top_3_specialists = {
            specialist_classes[i]: float(specialist_proba[i])
            for i in top_3_indices
        }
        
        # ========================================
        # PRÉDICTION URGENCE avec Random Forest
        # ========================================
        urgency_pred = self.model_urgency.predict(X)[0]
        urgency_proba = self.model_urgency.predict_proba(X)[0]
        urgency_confidence = urgency_proba.max()
        
        # Mapper urgence EN → FR
        urgency_mapping = {
            'High': 'ÉLEVÉE',
            'Moderate': 'MODÉRÉE',
            'Low': 'FAIBLE'
        }
        urgency_fr = urgency_mapping.get(urgency_pred, 'MODÉRÉE')
        
        # ========================================
        # TIMING & RECOMMANDATIONS
        # ========================================
        timing = self._determine_timing(urgency_fr)
        recommendations = self._generate_recommendations(
            symptom_names, 
            specialist_pred
        )
        
        # ========================================
        # CONFIANCE GLOBALE
        # ========================================
        # Moyenne des confiances symptômes + modèle
        symptom_confidences = [s.get('confidence', 0.5) for s in symptoms]
        avg_symptom_conf = np.mean(symptom_confidences)
        
        global_confidence = (avg_symptom_conf + specialist_confidence) / 2 * 100
        
        return {
            'specialist': specialist_pred,
            'urgency': urgency_fr,
            'timing': timing,
            'recommendations': recommendations,
            'confidence': global_confidence,
            'symptoms': symptoms,
            'possible_diseases': analysis.get('possible_diseases', {}),
            'model_probabilities': {
                'top_3_specialists': top_3_specialists,
                'specialist_confidence': float(specialist_confidence),
                'urgency_confidence': float(urgency_confidence)
            }
        }
    
    def _determine_timing(self, urgency: str) -> str:
        """Détermine timing selon urgence"""
        timing_map = {
            'VITALE': 'IMMÉDIAT (appeler urgences)',
            'ÉLEVÉE': "Aujourd'hui même",
            'MODÉRÉE': '24-48 heures',
            'FAIBLE': 'Cette semaine'
        }
        return timing_map.get(urgency, '24-48 heures')
    
    def _generate_recommendations(self, symptoms: list, specialist: str) -> list:
        """Génère recommandations par spécialiste"""
        recommendations_map = {
            'Cardiologue': [
                'Repos complet',
                'Éviter tout effort physique',
                'Ne pas fumer',
                'Surveiller la pression artérielle'
            ],
            'Neurologue': [
                'Repos dans endroit calme et sombre',
                'Hydratation régulière',
                'Éviter les écrans',
                'Noter symptômes et fréquence'
            ],
            'Dentiste': [
                'Éviter aliments chauds/froids',
                'Brossage doux',
                'Bain de bouche antiseptique',
                'Ne pas mâcher côté douloureux'
            ],
            'Gastro-entérologue': [
                'Alimentation légère',
                'Hydratation régulière',
                'Éviter aliments épicés',
                'Repos digestif'
            ],
            'Dermatologue': [
                'Ne pas gratter',
                'Garder peau propre et sèche',
                'Éviter exposition soleil',
                'Consulter si persistance'
            ],
            'Rhumatologue': [
                'Repos articulaire',
                'Application froid/chaud',
                'Éviter mouvements brusques',
                'Mobilisation douce'
            ],
            'Urologue': [
                'Hydratation abondante',
                'Uriner fréquemment',
                'Éviter rétention urine',
                'Hygiène intime'
            ],
            'Ophtalmologue': [
                'Repos visuel',
                'Éviter écrans',
                'Lumière tamisée',
                'Ne pas frotter yeux'
            ],
            'Pneumologue': [
                'Repos',
                'Hydratation',
                'Éviter irritants',
                'Position semi-assise'
            ],
            'ORL': [
                'Repos vocal si nécessaire',
                'Hydratation',
                'Éviter changements température',
                'Humidification air'
            ],
            'Médecin généraliste': [
                'Repos',
                'Hydratation',
                'Surveiller évolution',
                'Noter symptômes'
            ]
        }
        
        return recommendations_map.get(
            specialist,
            ['Consulter un professionnel de santé']
        )
    
    def _default_reasoning(self) -> dict:
        """Raisonnement par défaut si aucun symptôme"""
        return {
            'specialist': 'Médecin généraliste',
            'urgency': 'MODÉRÉE',
            'timing': '24-48 heures',
            'recommendations': [
                'Consulter un médecin généraliste',
                'Noter tous les symptômes',
                'Surveiller évolution'
            ],
            'confidence': 0.0,
            'symptoms': [],
            'possible_diseases': {},
            'model_probabilities': {}
        }
    
    def save_model(self, path: str):
        """Sauvegarde modèles entraînés"""
        model_data = {
            'model_specialist': self.model_specialist,
            'model_urgency': self.model_urgency,
            'mlb_symptoms': self.mlb_symptoms,
            'specialist_classes': self.specialist_classes,
            'urgency_classes': self.urgency_classes
        }
        
        joblib.dump(model_data, path)
        print(f"✅ Modèles sauvegardés: {path}")
    
    def load_model(self, path: str):
        """Charge modèles depuis fichier"""
        model_data = joblib.load(path)
        
        self.model_specialist = model_data['model_specialist']
        self.model_urgency = model_data['model_urgency']
        self.mlb_symptoms = model_data['mlb_symptoms']
        self.specialist_classes = model_data['specialist_classes']
        self.urgency_classes = model_data['urgency_classes']
        
        print(f"✅ Modèles chargés: {path}")
    
    def evaluate_on_test_set(self, test_cases: list) -> dict:
        """
        Évalue modèle sur set de tests
        
        Args:
            test_cases: [
                {
                    'symptoms': ['chest pain'],
                    'expected_specialist': 'Cardiologue',
                    'expected_urgency': 'High'
                },
                ...
            ]
        
        Returns:
            Métriques de performance
        """
        correct_specialist = 0
        correct_urgency = 0
        total = len(test_cases)
        
        for case in test_cases:
            # Préparer input
            analysis = {'symptoms': [
                {'symptom': s, 'confidence': 1.0} 
                for s in case['symptoms']
            ]}
            
            # Prédiction
            result = self.reason(analysis)
            
            # Vérifier specialist
            if result['specialist'] == case['expected_specialist']:
                correct_specialist += 1
            
            # Vérifier urgence
            urgency_map = {
                'High': 'ÉLEVÉE',
                'Moderate': 'MODÉRÉE',
                'Low': 'FAIBLE'
            }
            expected_urg = urgency_map.get(case['expected_urgency'], 'MODÉRÉE')
            
            if result['urgency'] == expected_urg:
                correct_urgency += 1
        
        return {
            'specialist_accuracy': correct_specialist / total,
            'urgency_accuracy': correct_urgency / total,
            'total_tests': total
        }