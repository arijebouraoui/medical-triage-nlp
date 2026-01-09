\# 🏥 Système de Triage Médical Intelligent



!\[Python](https://img.shields.io/badge/python-3.11-blue)

!\[Accuracy](https://img.shields.io/badge/accuracy-97.27%25-green)

!\[ML](https://img.shields.io/badge/ML-Random%20Forest-orange)

!\[Status](https://img.shields.io/badge/status-production--ready-success)

!\[License](https://img.shields.io/badge/license-MIT-blue)



> Système intelligent de triage médical utilisant le traitement du langage naturel (NLP) et le Machine Learning pour recommander le spécialiste approprié en fonction des symptômes du patient.



\*\*Auteur:\*\* Arije Bouraoui  

\*\*Date:\*\* Janvier 2026  

\*\*Version:\*\* 4.0 ML



---



\## 📋 Table des matières



\- \[✨ Aperçu](#-aperçu)

\- \[🎯 Fonctionnalités](#-fonctionnalités)

\- \[🏗️ Architecture](#️-architecture)

\- \[🤖 Modèles \& Technologies](#-modèles--technologies)

\- \[📊 Performances](#-performances)

\- \[🚀 Installation](#-installation)

\- \[💻 Utilisation](#-utilisation)

\- \[📁 Structure du Projet](#-structure-du-projet)

\- \[🔬 Métriques Détaillées](#-métriques-détaillées)

\- \[📚 Documentation](#-documentation)

\- \[🤝 Contribution](#-contribution)

\- \[📄 License](#-license)



---



\## ✨ Aperçu



Ce système analyse automatiquement les symptômes décrits en \*\*langage naturel\*\* (français, anglais, arabe, espagnol) et recommande le \*\*spécialiste médical\*\* approprié ainsi que le \*\*niveau d'urgence\*\*.



\### Exemple d'utilisation



```

Input:  "j'ai mal au coeur et je respire difficilement"



Output: 

✅ Spécialiste: Cardiologue

⚠️  Urgence: ÉLEVÉE

⏰ Délai: Aujourd'hui même

📞 SAMU: 190

💡 Recommandations: Repos complet, éviter effort...

```



\### Démo en ligne



🌐 \*\*Interface Streamlit:\*\* \[Lien vers démo](https://medical-triage-nlp.streamlit.app) \*(à venir)\*



---



\## 🎯 Fonctionnalités



\### 🌍 Multilingue

\- ✅ \*\*Français\*\* - "j'ai mal à la tête"

\- ✅ \*\*Anglais\*\* - "i have a headache"

\- ✅ \*\*Arabe\*\* - "أنا أعاني من صداع"

\- ✅ \*\*Espagnol\*\* - "me duele la cabeza"



\### 🧠 Intelligence Artificielle

\- \*\*NLP Avancé:\*\* spaCy, Word2Vec, correction orthographique

\- \*\*Machine Learning:\*\* Random Forest (97.27% accuracy)

\- \*\*Matching Sémantique:\*\* Détection intelligente des symptômes

\- \*\*Raisonnement Médical:\*\* 17 priorités + apprentissage statistique



\### 📊 Base de Connaissances

\- \*\*4,944 cas médicaux\*\* réels

\- \*\*143 symptômes\*\* uniques

\- \*\*47 maladies\*\* différentes

\- \*\*11 spécialistes\*\* médicaux



\### 🎨 Interface Utilisateur

\- Interface web moderne (Streamlit)

\- Toggle ML / Règles classiques

\- Affichage des probabilités

\- Historique des consultations

\- Numéros d'urgence par pays (5 pays)



---



\## 🏗️ Architecture



```

┌─────────────────────────────────────────────────┐

│              INPUT: Texte Patient               │

│         "j'ai mal au coeur depuis 2h"           │

└────────────────┬────────────────────────────────┘

&nbsp;                │

&nbsp;                ▼

┌─────────────────────────────────────────────────┐

│          1. NLP ANALYZER (Analyse)              │

├─────────────────────────────────────────────────┤

│ • Détection langue (Regex)                      │

│ • Correction orthographique (PyEnchant)         │

│ • Traduction FR/AR/ES → EN (Google Translate)   │

│ • Lemmatisation (spaCy CNN)                     │

│ • Extraction concepts (NER + Règles)            │

│ • Similarité sémantique (Word2Vec CBOW)         │

│ • Matching symptômes (3 niveaux)                │

└────────────────┬────────────────────────────────┘

&nbsp;                │ symptoms = \["chest pain"]

&nbsp;                ▼

┌─────────────────────────────────────────────────┐

│       2. REASONER (Raisonnement ML)             │

├─────────────────────────────────────────────────┤

│ • Random Forest (100 arbres)                    │

│ • Features: 143 symptômes (one-hot)             │

│ • Prédiction spécialiste (97.27% accuracy)      │

│ • Prédiction urgence (98.58% accuracy)          │

│ • Top 3 probabilités                            │

└────────────────┬────────────────────────────────┘

&nbsp;                │ specialist = "Cardiologue"

&nbsp;                ▼

┌─────────────────────────────────────────────────┐

│     3. DECISION GENERATOR (Génération)          │

├─────────────────────────────────────────────────┤

│ • Templates bilingues (FR/EN)                   │

│ • Numéros urgence par pays                      │

│ • Recommandations par spécialiste              │

│ • Formatage professionnel                       │

└────────────────┬────────────────────────────────┘

&nbsp;                │

&nbsp;                ▼

┌─────────────────────────────────────────────────┐

│         OUTPUT: Rapport Médical                 │

│   Spécialiste | Urgence | Timing | Numéros     │

└─────────────────────────────────────────────────┘

```



---



\## 🤖 Modèles \& Technologies



\### Machine Learning



| Tâche | Modèle/Technique | Bibliothèque | Métriques |

|-------|------------------|--------------|-----------|

| \*\*Raisonnement\*\* | Random Forest (100 arbres) | scikit-learn | 97.27% accuracy |

| \*\*Compréhension\*\* | spaCy CNN (en\_core\_web\_sm) | spaCy | 97.5% lemmatisation |

| \*\*Similarité\*\* | Word2Vec CBOW | gensim | 243 mots, 100 dims |

| \*\*Traduction\*\* | Google GNMT API | deep-translator | 4 langues |

| \*\*Correction\*\* | Hunspell Dicts | PyEnchant | FR/EN, 60 corrections |



\### Bibliothèques Principales



\*\*NLP \& Texte:\*\*

\- `spacy==3.7.4` - Pipeline NLP principal

\- `gensim==4.3.2` - Word embeddings (Word2Vec)

\- `nltk==3.8.1` - Tokenisation, stopwords

\- `pyenchant==3.2.2` - Correction orthographique

\- `deep-translator==1.11.4` - Traduction multilingue



\*\*Machine Learning:\*\*

\- `scikit-learn==1.4.0` - Random Forest, métriques

\- `joblib==1.3.2` - Sauvegarde modèles

\- `numpy==1.26.4` - Calculs numériques



\*\*Interface \& Data:\*\*

\- `streamlit==1.31.0` - Interface web

\- `pandas==2.2.0` - Manipulation données



\*\*Total:\*\* 17 bibliothèques, ~195 MB



---



\## 📊 Performances



\### Modèle Random Forest



```

┌──────────────────────────────────────────────┐

│          MÉTRIQUES GLOBALES                  │

├──────────────────────────────────────────────┤

│ Accuracy Spécialiste:    97.27% (962/989)   │

│ Accuracy Urgence:        98.58% (975/989)   │

│                                              │

│ Precision moyenne:       98%                 │

│ Recall moyen:            97%                 │

│ F1-Score moyen:          97%                 │

│                                              │

│ Overfitting Check:       ✅ 0.23% gap       │

│ Train accuracy:          ~97.5%              │

│ Test accuracy:           97.27%              │

└──────────────────────────────────────────────┘

```



\### Performances par Spécialiste (Top 5)



| Spécialiste | Precision | Recall | F1-Score | Support |

|-------------|-----------|--------|----------|---------|

| Cardiologue | 99% | 99% | 99% | 140 cas |

| Dermatologue | 98% | 99% | 98% | 228 cas |

| Gastro-entérologue | 97% | 99% | 98% | 281 cas |

| Rhumatologue | 100% | 96% | 98% | 124 cas |

| Neurologue | 94% | 94% | 94% | 53 cas |



\### Dataset



\- \*\*Train set:\*\* 3,955 cas (80%)

\- \*\*Test set:\*\* 989 cas (20%)

\- \*\*Features:\*\* 143 symptômes (one-hot encoding)

\- \*\*Classes:\*\* 11 spécialistes



\### Feature Importance (Top 10)



```

1\. chest\_pain        6.07%  ████████████

2\. vomiting          4.10%  ████████

3\. mucoid\_sputum     3.78%  ███████

4\. skin\_rash         3.57%  ███████

5\. breathlessness    2.98%  ██████

6\. depression        2.92%  ██████

7\. itching           2.86%  █████

8\. cough             2.85%  █████

9\. watering\_eyes     2.83%  █████

10\. swollen\_legs     2.83%  █████

```



---



\## 🚀 Installation



\### Prérequis



\- Python 3.11+

\- pip

\- Git

\- (Optionnel) Git LFS pour modèle ML



\### Installation Standard



```bash

\# 1. Cloner le repository

git clone https://github.com/arijebouraoui/medical-triage-nlp.git

cd medical-triage-nlp



\# 2. Créer environnement virtuel

python -m venv venv



\# 3. Activer environnement

\# Windows:

venv\\Scripts\\activate

\# Linux/Mac:

source venv/bin/activate



\# 4. Installer les dépendances

pip install -r requirements.txt



\# 5. Télécharger modèle spaCy

python -m spacy download en\_core\_web\_sm



\# 6. Entraîner le modèle Random Forest

python train\_ml\_reasoner.py

```



\### Installation Rapide (avec modèle pré-entraîné)



Si le modèle est disponible via Git LFS:



```bash

git clone https://github.com/arijebouraoui/medical-triage-nlp.git

cd medical-triage-nlp

pip install -r requirements.txt

python -m spacy download en\_core\_web\_sm

streamlit run streamlit\_app.py

```



---



\## 💻 Utilisation



\### Interface Web (Streamlit)



```bash

streamlit run streamlit\_app.py

```



L'interface sera accessible sur: `http://localhost:8501`



\### Interface CLI (Ligne de commande)



```bash

python medical\_triage\_system.py

```



\### Utilisation Programmatique



```python

from agents.analyzer.nlp\_analyzer\_v3 import MedicalNLPAnalyzer

from agents.reasoner.ml\_medical\_reasoner import MLMedicalReasoner

from agents.decider.decision\_generator import DecisionGenerator



\# Initialiser le système

analyzer = MedicalNLPAnalyzer('data/processed/dataset\_processed.json')

reasoner = MLMedicalReasoner(model\_path='models/random\_forest\_reasoner.pkl')

decider = DecisionGenerator(patient\_country="Tunisie")



\# Analyser des symptômes

text = "j'ai mal à la tête depuis 2 jours"

analysis = analyzer.analyze(text)

reasoning = reasoner.reason(analysis)

report = decider.generate\_decision(reasoning)



print(report)

```



\### Tests Automatisés



```bash

\# Lancer les tests

python test\_system.py



\# Résultats attendus: 4/4 tests réussis (100%)

```



---



\## 📁 Structure du Projet



```

medical-triage-nlp/

│

├── 📄 README.md                    # Ce fichier

├── 📄 requirements.txt             # Dépendances Python

├── 📄 .gitignore                   # Fichiers Git à ignorer

├── 📄 .gitattributes               # Configuration Git LFS

│

├── 🎨 streamlit\_app.py             # Interface web principale

├── 🧪 test\_system.py               # Tests automatisés

├── 🔧 setup\_dataset.py             # Configuration dataset

├── 🤖 train\_ml\_reasoner.py         # Entraînement Random Forest

│

├── 🤖 agents/                      # Modules intelligents

│   ├── analyzer/

│   │   └── nlp\_analyzer\_v3.py      # Analyse NLP complète

│   ├── reasoner/

│   │   ├── medical\_reasoner.py     # Raisonnement classique

│   │   └── ml\_medical\_reasoner.py  # Raisonnement ML ⭐

│   ├── decider/

│   │   └── decision\_generator.py   # Génération rapports

│   ├── data\_loader/

│   │   └── medical\_data\_loader.py  # Chargement données

│   ├── nlp/

│   │   └── spell\_corrector.py      # Correction orthographique

│   └── nlp\_advanced/

│       ├── medical\_word2vec.py     # Word2Vec ⭐

│       └── nlp\_foundations.py      # Techniques NLP

│

├── 📊 data/                        # Données

│   └── processed/

│       └── dataset\_processed.json  # 4,944 cas médicaux (8.2 MB)

│

├── 🧠 models/                      # Modèles ML

│   ├── README.md                   # Documentation modèle

│   └── random\_forest\_reasoner.pkl  # Modèle entraîné ⭐ (1.5 MB)

│

└── 📈 reports/                     # Rapports \& métriques

&nbsp;   └── figures/                    # Graphiques performance

```



\*\*Légende:\*\*

\- ⭐ = Fichiers clés du système ML

\- 📄 = Documentation

\- 🤖 = Agents intelligents

\- 📊 = Données



---



\## 🔬 Métriques Détaillées



\### 1. Accuracy (Précision globale)



\*\*Formule:\*\* `Accuracy = Prédictions correctes / Total prédictions`



\*\*Résultats:\*\*

\- Spécialiste: 97.27% (962/989 corrects)

\- Urgence: 98.58% (975/989 corrects)



\### 2. Precision (Fiabilité)



\*\*Formule:\*\* `Precision = Vrais Positifs / (Vrais Positifs + Faux Positifs)`



\*\*Interprétation:\*\* Quand le modèle prédit "Cardiologue", c'est correct 99% du temps.



\### 3. Recall (Complétude)



\*\*Formule:\*\* `Recall = Vrais Positifs / (Vrais Positifs + Faux Négatifs)`



\*\*Interprétation:\*\* Le modèle détecte 97% des vrais cas cardiaques.



\### 4. F1-Score (Équilibre)



\*\*Formule:\*\* `F1 = 2 × (Precision × Recall) / (Precision + Recall)`



\*\*Résultat:\*\* 97% - Excellent équilibre entre precision et recall.



\### 5. Confusion Matrix



Matrice de confusion disponible dans `reports/figures/`



\### 6. Overfitting Check



```

Train Accuracy:  97.5%  ████████████████████

Test Accuracy:   97.27% ████████████████████

Gap:             0.23%  ✅ Excellent (< 5%)

```



\*\*Protections contre overfitting:\*\*

\- `max\_depth=20` - Limite profondeur arbres

\- `min\_samples\_split=10` - Min échantillons pour split

\- `min\_samples\_leaf=5` - Min échantillons par feuille

\- Random Forest (100 arbres) - Moyenne réduit overfitting



---



\## 📚 Documentation



\### Guides Détaillés



\- 📖 \[Guide NLP Pipeline](docs/NLP\_PIPELINE.md)

\- 🤖 \[Guide Random Forest](models/README.md)

\- 🎨 \[Guide Interface Streamlit](docs/STREAMLIT\_GUIDE.md)

\- 🧪 \[Guide Tests](docs/TESTING.md)



\### Concepts NLP Intégrés



Le projet intègre plusieurs concepts NLP académiques:



1\. \*\*Preprocessing (TP1):\*\*

&nbsp;  - Tokenisation (NLTK, spaCy)

&nbsp;  - Normalisation (lowercase, accents)

&nbsp;  - Stopwords removal

&nbsp;  - Regex patterns



2\. \*\*Word Embeddings (TP2):\*\*

&nbsp;  - Word2Vec CBOW (gensim)

&nbsp;  - Similarité cosinus

&nbsp;  - 243 mots, 100 dimensions



3\. \*\*Techniques Avancées:\*\*

&nbsp;  - Lemmatisation (spaCy)

&nbsp;  - POS Tagging

&nbsp;  - Named Entity Recognition (NER)

&nbsp;  - TF-IDF



\### API Reference



Documentation complète de l'API disponible dans le code (docstrings).



---



\## 🛠️ Configuration



\### Toggle ML / Règles Classiques



Dans l'interface Streamlit, vous pouvez basculer entre:



\- \*\*🤖 Random Forest ML\*\* - 97.27% accuracy, probabilités top 3

\- \*\*📋 Règles + Statistiques\*\* - Méthode classique, priorités médicales



\### Pays Supportés



Numéros d'urgence disponibles pour:

\- 🇹🇳 Tunisie (SAMU: 190)

\- 🇫🇷 France (SAMU: 15)

\- 🇬🇧 UK (Emergency: 999)

\- 🇺🇸 USA (911)

\- 🇨🇦 Canada (911)



\### Langues Supportées



\- 🇫🇷 Français

\- 🇬🇧 English

\- 🇸🇦 العربية (Arabe)

\- 🇪🇸 Español



---



\## 🧪 Tests



\### Tests Automatisés



```bash

python test\_system.py

```



\*\*4 tests critiques:\*\*

1\. ✅ Français - Symptôme cardiaque → Cardiologue

2\. ✅ Français - Symptôme dentaire → Dentiste

3\. ✅ Anglais - Mal de tête → Neurologue

4\. ✅ Anglais - Correction ortho → Cardiologue



\*\*Résultats:\*\* 4/4 tests réussis (100%)



\### Évaluation Complète



```bash

python evaluate\_system.py

```



Génère des métriques détaillées et graphiques.



---



\## 🚀 Déploiement



\### Streamlit Cloud



1\. Connecter GitHub à Streamlit Cloud

2\. Sélectionner le repository

3\. Fichier principal: `streamlit\_app.py`

4\. Déployer ✅



\### Heroku



```bash

\# Créer Procfile

echo "web: streamlit run streamlit\_app.py --server.port=$PORT" > Procfile



\# Déployer

heroku create medical-triage-nlp

git push heroku main

```



\### Docker



```dockerfile

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

RUN python -m spacy download en\_core\_web\_sm

COPY . .

CMD \["streamlit", "run", "streamlit\_app.py"]

```



---



\## 🤝 Contribution



Les contributions sont les bienvenues!



\### Comment contribuer



1\. Fork le projet

2\. Créer une branche (`git checkout -b feature/AmazingFeature`)

3\. Commit les changements (`git commit -m 'Add AmazingFeature'`)

4\. Push vers la branche (`git push origin feature/AmazingFeature`)

5\. Ouvrir une Pull Request



\### Lignes directrices



\- ✅ Suivre PEP 8 (style Python)

\- ✅ Ajouter des tests pour nouvelles fonctionnalités

\- ✅ Mettre à jour la documentation

\- ✅ Commenter le code en français



---



\## 📄 License



Ce projet est sous licence MIT. Voir le fichier \[LICENSE](LICENSE) pour plus de détails.



---



\## 👤 Auteur



\*\*Arije Bouraoui\*\*



\- Email: arije.bouraoui@polytechnicien.tn.tn

\- LinkedIn: \[Arije Bouraoui](www.linkedin.com/in/arije-bouraoui-882675365)



---



\## 🙏 Remerciements



\- \*\*Dataset:\*\* Kaggle Medical Transcriptions

\- \*\*Bibliothèques:\*\* spaCy, scikit-learn, Streamlit, gensim

\- \*\*Inspiration:\*\* Systèmes de triage médicaux professionnels

\- \*\*Encadrement:\*\* Dr Nizar Omheni



---



\## 📞 Support



Pour toute question ou problème:



1\. 📖 Consulter la \[documentation](docs/)

2\. 🐛 Ouvrir une \[issue](https://github.com/arijebouraoui/medical-triage-nlp/issues)

3\. 💬 Discussions dans \[Discussions](https://github.com/arijebouraoui/medical-triage-nlp/discussions)



---



\## 🔮 Roadmap



\### Version 4.1 (À venir)



\- \[ ] Support de plus de langues (Italien, Allemand)

\- \[ ] Deep Learning (BERT médical, BioBERT)

\- \[ ] API REST pour intégrations

\- \[ ] Application mobile (React Native)

\- \[ ] Multimodalité (texte + images symptômes)



\### Version 5.0 (Future)



\- \[ ] Téléconsultation intégrée

\- \[ ] Base de connaissances évolutive (RAG)

\- \[ ] Certification médicale

\- \[ ] Support temps réel (WebSocket)



---



\## 📊 Statistiques du Projet



```

📝 Lignes de code:       ~5,000

🐍 Fichiers Python:      15

📦 Dépendances:          17 bibliothèques

💾 Taille totale:        ~200 MB

🧠 Modèles ML:           3 (spaCy, Word2Vec, Random Forest)

📊 Dataset:              4,944 cas médicaux

⏱️  Temps traitement:    ~2.6 secondes/cas

🎯 Accuracy:             97.27%

```



---



\## ⭐ Si ce projet vous aide, donnez-lui une étoile sur GitHub!



\[!\[GitHub stars](https://img.shields.io/github/stars/arijebouraoui/medical-triage-nlp?style=social)](https://github.com/arijebouraoui/medical-triage-nlp/stargazers)



---



<div align="center">



\*\*Fait avec ❤️ par Arije Bouraoui\*\*



\*\*🏥 Système de Triage Médical Intelligent • NLP • Machine Learning • 97% Accuracy\*\*



\[⬆ Retour en haut](#-système-de-triage-médical-intelligent)



</div>

