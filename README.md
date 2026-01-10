# 🏥 Système de Triage Médical Intelligent

![Python](https://img.shields.io/badge/python-3.11-blue)
![Accuracy](https://img.shields.io/badge/accuracy-97.27%25-green)
![ML](https://img.shields.io/badge/ML-Random%20Forest-orange)
![Status](https://img.shields.io/badge/status-deployed-success)
![Streamlit](https://img.shields.io/badge/streamlit-1.31.0-red)

> Système intelligent de triage médical utilisant le traitement du langage naturel (NLP) et le Machine Learning pour recommander le spécialiste approprié et le niveau d'urgence en fonction des symptômes du patient.

**🌐 Démo Live:** [https://medical-triage-nlp.streamlit.app](https://medical-triage-nlp.streamlit.app)

**Auteur:** Arije Bouraoui  
**Version:** 4.0 ML Production  
**Date:** Janvier 2026

---

## 📋 Table des matières

- [✨ Aperçu](#-aperçu)
- [🎯 Fonctionnalités](#-fonctionnalités)
- [🏗️ Architecture](#️-architecture)
- [🤖 Modèles & Technologies](#-modèles--technologies)
- [📊 Performances](#-performances)
- [🚀 Installation](#-installation)
- [💻 Utilisation](#-utilisation)
- [📁 Structure du Projet](#-structure-du-projet)
- [🌐 Déploiement](#-déploiement)
- [📚 Documentation](#-documentation)

---

## ✨ Aperçu

Ce système analyse automatiquement les symptômes décrits en **langage naturel** (français, anglais, arabe, espagnol) et recommande le **spécialiste médical** approprié ainsi que le **niveau d'urgence**.

### 🎬 Exemple d'utilisation

```
Input:  "j'ai mal au coeur et je respire difficilement"

Output: 
✅ Spécialiste: Cardiologue (99% confiance)
🚨 Urgence: ÉLEVÉE
⏰ Délai: Aujourd'hui même
📞 SAMU: 190
💡 Recommandations: Repos complet, éviter tout effort physique...

📊 Top 3 Spécialistes:
  • Cardiologue: 75.7%
  • Dentiste: 9.9%
  • Pneumologue: 6.1%
```

---

## 🎯 Fonctionnalités

### 🌍 Multilingue
- ✅ **Français** - "j'ai mal à la tête"
- ✅ **Anglais** - "i have a headache"
- ✅ **Arabe** - "أنا أعاني من صداع"
- ✅ **Espagnol** - "me duele la cabeza"

### 🧠 Intelligence Artificielle
- **Random Forest ML** - 97.27% accuracy sur prédiction spécialiste
- **NLP Avancé** - spaCy, Word2Vec, correction orthographique contextuelle
- **Matching Sémantique** - Détection intelligente des symptômes (TF-IDF + Cosine Similarity)
- **Protocole de Sécurité** - Double validation (IA + Règles expertes médicales)

### 📊 Base de Connaissances
- **4,944 cas médicaux** réels
- **143 symptômes** uniques
- **47 maladies** différentes
- **11 spécialistes** médicaux

### 🎨 Interface Utilisateur
- Interface web moderne (Streamlit)
- Toggle ML / Règles classiques
- Affichage des probabilités Top 3
- Historique des consultations
- Numéros d'urgence par pays (5 pays)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│         INPUT: Texte Patient                    │
│      "j'ai mal au coeur depuis 2h"              │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│     1. NLP ANALYZER (Analyse Linguistique)      │
├─────────────────────────────────────────────────┤
│ • Détection langue (Regex/Fasttext)             │
│ • Correction orthographique (Bigrams+PyEnchant) │
│ • Traduction multi-langues (Deep-Translator)    │
│ • Lemmatisation (spaCy)                         │
│ • Extraction concepts médicaux (NER)            │
│ • Similarité sémantique (Word2Vec CBOW)         │
│ • Matching symptômes (TF-IDF + Cosine)          │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│    2. ML REASONER (Raisonnement Hybride)        │
├─────────────────────────────────────────────────┤
│ • Random Forest (100 arbres, 143 features)      │
│ • Prédiction spécialiste (97.27% accuracy)      │
│ • Prédiction urgence (98.58% accuracy)          │
│ • Protocole de sécurité médical                 │
│ • Validation par règles expertes               │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│   3. DECISION GENERATOR (Génération Rapport)    │
├─────────────────────────────────────────────────┤
│ • Templates bilingues (FR/EN)                   │
│ • Numéros urgence par pays                      │
│ • Recommandations par spécialiste              │
│ • Formatage professionnel                       │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│          OUTPUT: Rapport Médical                │
│   Spécialiste | Urgence | Délai | Numéros      │
└─────────────────────────────────────────────────┘
```

---

## 🤖 Modèles & Technologies

### Machine Learning & NLP

| Tâche | Modèle/Technique | Bibliothèque | Performance |
|-------|------------------|--------------|-------------|
| **Raisonnement IA** | Random Forest (100 arbres) | scikit-learn | 97.27% accuracy |
| **Compréhension Texte** | spaCy CNN | spaCy 3.8.11 | Lemmatisation 97.5% |
| **Similarité Sémantique** | Word2Vec CBOW (100 dims) | gensim 4.3.2 | 243 termes médicaux |
| **Matching Symptômes** | TF-IDF + Cosine Similarity | scikit-learn | Seuil 0.70 |
| **Traduction** | Google GNMT API | deep-translator | 4 langues |
| **Correction Ortho** | Bigrams + Hunspell | PyEnchant + Custom | 60+ corrections/min |
| **Détection Langue** | Regex + Fasttext | Custom | FR/EN/AR/ES |

### Bibliothèques Principales

**NLP & Texte:**
- `spacy==3.8.11` - Pipeline NLP principal
- `gensim==4.3.2` - Word embeddings (Word2Vec)
- `pyenchant==3.2.2` - Correction orthographique
- `deep-translator==1.11.4` - Traduction multilingue
- `pyspellchecker==0.8.1` - Dictionnaire médical

**Machine Learning:**
- `scikit-learn==1.4.0` - Random Forest, TF-IDF, métriques
- `joblib==1.3.2` - Sauvegarde modèles
- `numpy==1.26.4` - Calculs numériques

**Interface & Déploiement:**
- `streamlit==1.31.0` - Interface web interactive
- `pandas==2.2.0` - Manipulation données

---

## 📊 Performances

### 🎯 Métriques Globales

```
┌──────────────────────────────────────────────┐
│       RANDOM FOREST PERFORMANCES             │
├──────────────────────────────────────────────┤
│ Accuracy Spécialiste:    97.27% (962/989)   │
│ Accuracy Urgence:        98.58% (975/989)   │
│                                              │
│ Precision moyenne:       98%                 │
│ Recall moyen:            97%                 │
│ F1-Score moyen:          97%                 │
│                                              │
│ ✅ Overfitting Check:    0.23% gap          │
│    Train accuracy:       ~97.5%              │
│    Test accuracy:        97.27%              │
└──────────────────────────────────────────────┘
```

### 📈 Performances par Spécialiste (Top 5)

| Spécialiste | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| **Cardiologue** | 99% | 99% | 99% | 140 cas |
| **Dermatologue** | 98% | 99% | 98% | 228 cas |
| **Gastro-entérologue** | 97% | 99% | 98% | 281 cas |
| **Rhumatologue** | 100% | 96% | 98% | 124 cas |
| **Neurologue** | 94% | 94% | 94% | 53 cas |

### 🌳 Feature Importance (Top 10)

```
1. chest_pain        6.07%  ████████████
2. vomiting          4.10%  ████████
3. mucoid_sputum     3.78%  ███████
4. skin_rash         3.57%  ███████
5. breathlessness    2.98%  ██████
6. depression        2.92%  ██████
7. itching           2.86%  █████
8. cough             2.85%  █████
9. watering_eyes     2.83%  █████
10. swollen_legs     2.83%  █████
```

### 📊 Dataset

- **Train set:** 3,955 cas (80%)
- **Test set:** 989 cas (20%)
- **Features:** 143 symptômes (one-hot encoding)
- **Classes:** 11 spécialistes + 4 niveaux d'urgence
- **Stratification:** Équilibrée par classe

### 📉 Graphiques de Performance

Les visualisations suivantes sont disponibles dans `reports/figures/`:
- `overall_performance.png` - Performance globale
- `urgency_accuracy.png` - Précision par urgence
- `urgency_heatmap.png` - Matrice de confusion

---

## 🚀 Installation

### Prérequis

- Python 3.11+
- pip
- Git
- 2 GB RAM minimum

### Installation Locale

```bash
# 1. Cloner le repository
git clone https://github.com/arijebouraoui/medical-triage-nlp.git
cd medical-triage-nlp

# 2. Créer environnement virtuel
python -m venv venv

# 3. Activer environnement
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 4. Installer les dépendances
pip install -r requirements.txt

# 5. Télécharger modèles spaCy
python -m spacy download en_core_web_sm

# 6. Entraîner le modèle Random Forest (optionnel)
python train_ml_reasoner.py
```

### Installation Rapide

```bash
git clone https://github.com/arijebouraoui/medical-triage-nlp.git
cd medical-triage-nlp
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## 💻 Utilisation

### 🌐 Interface Web (Recommandé)

```bash
streamlit run streamlit_app.py
```

L'interface sera accessible sur: `http://localhost:8501`

### 🖥️ Ligne de commande

```bash
python medical_triage_system.py
```

### 🐍 API Programmatique

```python
from agents.analyzer.nlp_analyzer_v3 import MedicalNLPAnalyzer
from agents.reasoner.ml_medical_reasoner import MLMedicalReasoner
from agents.decider.decision_generator import DecisionGenerator

# Initialiser
analyzer = MedicalNLPAnalyzer('data/processed/dataset_processed.json')
reasoner = MLMedicalReasoner(model_path='models/random_forest_reasoner.pkl')
decider = DecisionGenerator(patient_country="Tunisie")

# Analyser
analysis = analyzer.analyze("j'ai mal à la tête depuis 2 jours")
reasoning = reasoner.reason(analysis)
report = decider.generate_decision(reasoning)

print(report)
```

### 🧪 Tests

```bash
# Tests automatisés
python test_system.py

# Évaluation complète
python evaluate_system.py
```

---

## 📁 Structure du Projet

```
medical-triage-nlp/
│
├── 📄 README.md                    # Documentation principale
├── 📄 requirements.txt             # Dépendances Python
├── 📄 packages.txt                 # Dépendances système
├── 📄 .gitignore                   # Fichiers à ignorer
│
├── 🎨 streamlit_app.py             # Interface web Streamlit
├── 🧪 test_system.py               # Tests automatisés
├── 🤖 train_ml_reasoner.py         # Entraînement Random Forest
├── 📊 evaluate_system.py           # Évaluation performances
├── 🔧 setup_dataset.py             # Configuration dataset
├── 💬 interactive_triage.py        # Mode interactif CLI
├── 🏥 medical_triage_system.py     # Système CLI principal
│
├── 🤖 agents/                      # Modules intelligents
│   ├── analyzer/
│   │   ├── nlp_analyzer_v3.py      # ⭐ Pipeline NLP complet
│   │   └── intelligent_medical_nlu.py
│   ├── reasoner/
│   │   ├── ml_medical_reasoner.py  # ⭐ Raisonnement ML hybride
│   │   └── medical_reasoner.py     # Règles classiques
│   ├── decider/
│   │   └── decision_generator.py   # Génération rapports
│   ├── data_loader/
│   │   └── medical_data_loader.py  # Chargement données
│   ├── nlp/
│   │   └── spell_corrector.py      # Correction orthographique
│   └── nlp_advanced/
│       ├── medical_word2vec.py     # ⭐ Word2Vec embeddings
│       └── nlp_foundations.py      # TF-IDF, similarité
│
├── 📊 data/                        # Données médicales
│   └── processed/
│       └── dataset_processed.json  # 4,944 cas (8.2 MB)
│
├── 🧠 models/                      # Modèles ML
│   ├── README.md                   # Documentation modèle
│   └── random_forest_reasoner.pkl  # ⭐ Modèle entraîné (1.5 MB)
│
└── 📈 reports/                     # Rapports & graphiques
    └── figures/
        ├── overall_performance.png
        ├── urgency_accuracy.png
        └── urgency_heatmap.png
```

**Légende:**
- ⭐ = Fichiers clés du système
- 📄 = Documentation
- 🤖 = Agents intelligents
- 📊 = Données

---

## 🌐 Déploiement

### ☁️ Streamlit Cloud (Déployé)

**🌐 App Live:** [https://medical-triage-nlp.streamlit.app](https://medical-triage-nlp.streamlit.app)

Le projet est déjà déployé sur Streamlit Cloud avec:
- ✅ Auto-training du modèle au premier lancement
- ✅ Support multilingue complet
- ✅ Interface responsive
- ✅ Disponible 24/7

### 🐳 Docker (Local)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN python -m spacy download en_core_web_sm
COPY . .
CMD ["streamlit", "run", "streamlit_app.py"]
```

```bash
docker build -t medical-triage .
docker run -p 8501:8501 medical-triage
```

### 🚀 Heroku

```bash
# Créer Procfile
echo "web: streamlit run streamlit_app.py --server.port=$PORT" > Procfile

# Déployer
heroku create medical-triage-nlp
git push heroku main
```

---

## 📚 Documentation

### 📖 Guides Techniques

- **NLP Pipeline:** Architecture complète du traitement linguistique
- **Random Forest:** Détails du modèle ML et hyperparamètres
- **Safety Protocol:** Règles expertes de validation médicale
- **Multilingual:** Gestion des 4 langues supportées

### 🎓 Concepts NLP Intégrés

1. **Preprocessing:** Tokenisation (NLTK, spaCy), Normalisation, Stopwords
2. **Word Embeddings:** Word2Vec CBOW (243 mots, 100 dims)
3. **Techniques Avancées:** Lemmatisation, POS Tagging, NER, TF-IDF
4. **Machine Learning:** Random Forest, One-hot encoding, Stratified split

### 🔬 Métriques Détaillées

**Accuracy:** `(TP + TN) / (TP + TN + FP + FN)`
**Precision:** `TP / (TP + FP)`
**Recall:** `TP / (TP + FN)`
**F1-Score:** `2 × (Precision × Recall) / (Precision + Recall)`

---

## 🛡️ Protocole de Sécurité

Le système implémente un **protocole de sécurité médical** en 3 niveaux:

1. **Validation IA:** Random Forest prédit spécialiste et urgence
2. **Règles Expertes:** 17 priorités médicales valident les prédictions
3. **Override de Sécurité:** Symptômes critiques forcent URGENCE VITALE

**Exemple:** "douleur thoracique" → Override automatique vers Cardiologue + URGENCE ÉLEVÉE, même si l'IA hésite.

---

## 📊 Statistiques du Projet

```
📝 Lignes de code:       ~6,000
🐍 Fichiers Python:      18 modules
📦 Dépendances:          14 bibliothèques
💾 Taille totale:        ~210 MB
🧠 Modèles ML:           3 (spaCy, Word2Vec, Random Forest)
📊 Dataset:              4,944 cas médicaux
⏱️  Temps traitement:    ~2.6 secondes/cas
🎯 Accuracy:             97.27% (spécialiste)
🚨 Accuracy urgence:     98.58%
🌍 Langues:              4 (FR, EN, AR, ES)
```

---

## 🤝 Contribution

Les contributions sont bienvenues! Pour contribuer:

1. Fork le projet
2. Créer une branche (`git checkout -b feature/AmazingFeature`)
3. Commit (`git commit -m 'Add AmazingFeature'`)
4. Push (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

---

## 📄 License

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

---

## 👤 Auteur

**Arije Bouraoui**

- 🌐 GitHub: [@arijebouraoui](https://github.com/arijebouraoui)
- 📧 Email: arije.bouraoui@polytechnicien.tn
- 💼 LinkedIn: [Arije Bouraoui](https://www.linkedin.com/in/arije-bouraoui-882675365/)

---

## 🙏 Remerciements

- **Encadrant:** Dr Nizar Omheni
- **Dataset:** Kaggle Medical Transcriptions (4,944 cas cliniques)
- **Bibliothèques:** spaCy, scikit-learn, Streamlit, gensim
- **Inspiration:** Systèmes de triage médicaux professionnels
- **Hébergement:** Streamlit Community Cloud



---

## 📞 Support

Pour toute question ou problème:

1. 📖 Consulter la [documentation](https://github.com/arijebouraoui/medical-triage-nlp)
2. 🐛 Ouvrir une [issue](https://github.com/arijebouraoui/medical-triage-nlp/issues)
3. 🌐 Tester l'app: [https://medical-triage-nlp.streamlit.app](https://medical-triage-nlp.streamlit.app)

---

## ⭐ Si ce projet vous aide, donnez-lui une étoile sur GitHub!

[![GitHub stars](https://img.shields.io/github/stars/arijebouraoui/medical-triage-nlp?style=social)](https://github.com/arijebouraoui/medical-triage-nlp/stargazers)

---

<div align="center">

**🏥 Système de Triage Médical Intelligent**

**NLP Avancé • Random Forest ML • 97% Accuracy • Production Ready**

**🌐 [Live Demo](https://medical-triage-nlp.streamlit.app) | 📂 [GitHub](https://github.com/arijebouraoui/medical-triage-nlp)**

---

Fait avec ❤️ par Arije Bouraoui • Janvier 2026

</div>