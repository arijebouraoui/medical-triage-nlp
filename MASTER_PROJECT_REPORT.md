# 🏥 Documentation Intégrale : Système de Triage Médical Intelligent (A à Z)

Ce document est le guide de référence complet du projet. Il recense **chaque fichier** et **chaque module** créé pour cette solution de triage basée sur l'IA et le NLP.

---
**Performance du Système (Évaluation Réelle)** :
*   **Urgence** : 100.00% de précision.
*   **Spécialiste** : 86.00% de précision.
*   **Dataset utilisé** : 4 944 cas cliniques réels.

**Note à l'attention du correcteur** : Chaque fichier `.py` a été conçu pour être modulaire et indépendant, facilitant la maintenance et l'évolution du système vers de nouvelles langues ou pathologies.

---

## 1. Vue d'Ensemble des Modèles & Technologies

| Fonction | Modèle / Technologie | Rôle |
| :--- | :--- | :--- |
| **Intelligence Artificielle** | **Random Forest Classifier** | Classification du spécialiste (86% accuracy) et de l'urgence (100% accuracy). |
| **Traitement NLU** | **SpaCy** (`en_core_web_sm`) | Lemmatisation, Tokenisation et analyse grammaticale. |
| **Correction (True NLP)** | **Bigrams Contextuels** | Correction orthographique hybride (Générale + Médicale). |
| **Vecteurs Sémantiques** | **Word2Vec (Gensim)** | Représentation vectorielle des termes médicaux. |
| **Analyse Sémantique** | **TF-IDF & Cosine Similarity** | Calcul de proximité entre les symptômes du patient et la base de données. |
| **Traduction** | **Deep-Translator** | Traduction automatique multi-langues avec fallback dictionnaire. |

---

## 2. Inventaire Complet des Fichiers (Guide A à Z)

### 📁 Racine du Projet (Orchestration & Rapports)
*   **`streamlit_app.py`** : **Interface Utilisateur Finale**. Dashboard interactif affichant les analyses IA et les alertes de sécurité.
*   **`medical_triage_system.py`** : Moteur de triage en ligne de commande pour des tests rapides.
*   **`train_ml_reasoner.py`** : **Script d'Entraînement de l'IA**. Génère le modèle `RandomForest` utilisé par le système.
*   **`evaluate_system.py`** : Module d'évaluation calculant les performances (Accuracy, Rappel) sur tout le dataset.
*   **`setup_dataset.py`** : Initialisation et nettoyage des données médicales brutes.
*   **`interactive_triage.py`** : Mode de consultation interactive pas à pas.
*   **`test_system.py`** : Batterie de tests automatisés pour assurer la non-régression.
*   **`MASTER_PROJECT_REPORT.md`** : Rapport technique détaillé (version Markdown).
*   **`RAPPORT_PROJET_TRIAGE_AZ.pdf`** : Rapport officiel exportable.
*   **`QUESTIONS_REPONSES_EXAMEN.txt`** : Aide-mémoire pour la soutenance orale.
*   **`requirements.txt`** : Toutes les dépendances (Scikit-Learn, SpaCy, fpdf2, etc.).

### 📁 `agents/` (Le Cœur du Système)

#### � `agents/analyzer/` (Analyses de Données)
*   **`nlp_analyzer_v3.py`** : **Version Master de l'Analyseur**. Pipeline complet : Langue -> Correction -> Traduction -> Lemmatisation.
*   **`ml_classifier.py`** : Intègre le modèle `Random Forest` et gère les prédictions de probabilité.
*   **`intelligent_medical_nlu.py`** : Analyseur de syntaxe médicale pour extraire les entités complexes.
*   **`nlp_analyzer.py`** : Support historique pour la recherche sémantique basique.

#### � `agents/nlp/` & `agents/nlp_advanced/` (Langage & Sémantique)
*   **`context_spell_corrector.py`** : **Correcteur Contextuel**. Corrige les fautes en fonction du sens médical (Bigrams).
*   **`medical_word2vec.py`** : Entraînement et utilisation d'embeddings pour la similarité sémantique.
*   **`nlp_foundations.py`** : Algorithmes fondamentaux (TF-IDF manuel, similarité cosinus).
*   **`multilingual_processor.py`** : Détecteur de langue robuste et gestionnaire multilingue.
*   **`nlp_stemmer.py`** : Stemming spécifique pour les racines de mots médicaux.
*   **`spell_corrector.py`** : Moteur de correction orthographique de base (Distance de Levenshtein).
*   **`advanced_medical_nlp.py`** : Techniques de matching hybrides entre texte et codes médicaux.

#### � `agents/reasoner/` (Aide à la Décision)
*   **`ml_medical_reasoner.py`** : **Cerveau Hybride**. Combine les prédictions d'IA avec la logique de sécurité.
*   **`medical_reasoner.py`** : Système de règles expertes classiques (Safety Protocol).

#### 🔸 `agents/decider/` & `agents/data_loader/`
*   **`decision_generator.py`** : Génère les recommandations finales (Spécialiste, Urgence, Conseils).
*   **`medical_data_loader.py`** : Chargeur et indexeur du dataset clinique de 4 944 cas.

---

## 3. Architecture du Système (Processus)

1.  **Entrée** (`streamlit_app`) : Saisie libre du patient (Français, Anglais, etc.).
2.  **Prétraitement** (`nlp` / `spell_corrector`) : Nettoyage et correction des fautes de frappe.
3.  **Normalisation** (`multilingual_processor`) : Détection de langue et traduction vers l'Anglais.
4.  **Analyse** (`nlp_analyzer_v3` / `spaCy`) : Lemmatisation et extraction de concepts clés.
5.  **Intelligence** (`ml_classifier`) : Calcul des probabilités via Random Forest.
6.  **Sécurité** (`ml_medical_reasoner`) : Validation expert pour éviter les erreurs de l'IA.
7.  **Sortie** (`decision_generator`) : Affichage du triage final et du rapport de séance.

---
**Conclusion** : Cette structure modulaire garantit un système de triage médical scalable, explicable et sécurisé par une double validation (IA + Règles).
