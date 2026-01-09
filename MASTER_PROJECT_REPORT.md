# 🏥 Documentation Complète : Projet Triage Médical Intelligent (A à Z)

Ce document offre une vision exhaustive de l'architecture, des technologies et des modèles utilisés dans ce projet de triage médical basé sur le Traitement du Langage Naturel (NLP) et l'IA.

---

## 1. Vue d'Ensemble Technologique (La "Stack")

Le projet est divisé en modules spécialisés, chacun utilisant des bibliothèques de pointe :

| Tâche NLP | Bibliothèque / Modèle | Rôle |
| :--- | :--- | :--- |
| **Interface & Visualisation** | **Streamlit** | Interface web interactive et dashboard de santé IA. |
| **Compréhension (NLU)** | **SpaCy** (`en_core_web_sm`) | Lemmatisation, Tokenisation et analyse grammaticale. |
| **Intelligence Artificielle** | **Scikit-Learn** | Modèle `RandomForest` pour la classification spécialiste/urgence. |
| **Vecteurs (Embeddings)** | **TF-IDF Vectorizer** | Transformation du texte en vecteurs numériques basés sur la fréquence. |
| **Traduction** | **Deep-Translator** | Traducteur multi-moteurs (Google Translate par défaut). |
| **Correction** | **Pyspellchecker** | Correction orthographique basée sur des dictionnaires orfficiels. |
| **Logique de Correction** | **Bigrams / Context-Aware** | Correction intelligente basée sur l'ordre des mots (N-Grams). |

---

## 2. Rôle des Fichiers (Structure du Projet)

### 📁 Racine du Projet
*   **`streamlit_app.py`** : **Le Cœur de l'Interface**. Orchestre la saisie patient, appelle l'analyseur, et affiche les résultats (y compris le "Cerveau de l'IA" et les alertes de sécurité).
*   **`medical_triage_system.py`** : Point d'entrée pour la version Terminal/Console du système.
*   **`evaluate_system.py`** : Script de test de performance qui calcule la précision du modèle sur l'ensemble du dataset.
*   **`requirements.txt`** : Liste de toutes les bibliothèques Python nécessaires au projet.

### 📁 `agents/` (La Core Logic)
L'intelligence est divisée en "Agents" spécialisés :

#### 1. `agents/analyzer/` (Compréhension & Prédiction)
*   **`nlp_analyzer_v3.py`** : **Le Chef d'Orchestre NLP**. Gère le pipeline : Détection langue -> Correction -> Traduction -> Lemmatisation -> Extraction de symptômes.
*   **`ml_classifier.py`** : **Le Modèle Prédictif**. Contient la classe `MedicalMLClassifier` qui entraîne et utilise le modèle `RandomForest` pour prédire le spécialiste et l'urgence.
*   **`intelligent_medical_nlu.py`** : Module avancé pour la reconnaissance d'entités médicales complexes.

#### 2. `agents/nlp/` (Traitement du Langage)
*   **`context_spell_corrector.py`** : **L'Expert en Correction**. Utilise une approche hybride (Dictionnaire + Contexte médical) pour corriger les fautes (ex: "havee" -> "have").
*   **`multilingual_processor.py`** : Gère les spécificités linguistiques pour le Français, l'Anglais et l'Arabe.

#### 3. `agents/reasoner/` (Aide à la Décision)
*   **`medical_reasoner.py`** : **Le Cerveau Expert**. Combine les prédictions de l'IA avec des **règles médicales de sécurité**. C'est lui qui outrepasse l'IA si un symptôme vital (ex: douleur cardiaque) est détecté.

#### 4. `agents/decider/` (Génération des Sorties)
*   **`decision_generator.py`** : Génère les recommandations finales (Délai d'attente, numéros d'urgence selon le pays).

#### 5. `agents/data_loader/` (Gestion des Données)
*   **`medical_data_loader.py`** : Charge et indexe le dataset JSON pour une recherche ultra-rapide des symptômes.

---

## 3. Données & Modèles par Fonction

### 🧠 Modèle pour la Compréhension (NLU)
*   **Bibliothèque** : `SpaCy`.
*   **Logic** : Utilise la **Lemmatisation** pour transformer "teeth", "tooth", "dent" en un seul concept racine.
*   **Data** : S'appuie sur un index de 143 symptômes uniques appris depuis le dataset.

### 🤖 Modèle pour la Prédiction (AI/ML)
*   **Algorithme** : `Random Forest Classifier`.
*   **Pourquoi ?** Robuste, gère bien les données textuelles après vectorisation, et peu sensible au sur-apprentissage sur les petits datasets.
*   **Data** : Entraîné sur **4 944 cas cliniques** réels.

### ✍️ Modèle pour la Correction (Spell Check)
*   **Algorithme** : `Levenshtein Distance + Bigrams`.
*   **Process** : 
    1. Génère des candidats proches.
    2. Utilise les **Bigrams** (mots côte-à-côte) pour choisir le plus probable (ex: "my heart" au lieu de "my hear").
*   **Multilingue** : Gère FR et EN simultanément.

### 🌍 Modèle pour la Traduction
*   **Moteur** : `Google Translate API` (via `deep-translator`).
*   **Fallback** : Un dictionnaire manuel de 100+ termes médicaux critiques pour fonctionner même sans connexion stable.

---

## 4. Qu'avons-nous fait exactement ? (Résumé des étapes)

1.  **Uniformisation Multilingue** : Le système détecte la langue du patient et convertit tout en un "format neutre" (Anglais Lemmatisé) pour une analyse constante.
2.  **Correction Contextuelle** : Création d'un correcteur qui comprend que "ceour" en français doit être "coeur" avant même la traduction.
3.  **IA Hybride** : Passage d'un système à 100% de règles à un système **AI-Driven** (Random Forest) sécurisé par des **Safety Rules** (Règles métiers).
4.  **UI Professionnelle** : Mise en place d'un tableau de bord Streamlit qui explique en temps réel **comment** l'IA a pris sa décision (AI vs Protocol).
5.  **Dictionnaire Médical Étendu** : Création d'une base de connaissances de 100+ organes et symptômes traduits manuellement pour une précision maximale.

---

**Le résultat final est un système industriel capable de trier des patients en moins d'une seconde avec une sécurité médicale garantie.**
