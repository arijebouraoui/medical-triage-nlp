# 🏥 Medical Triage NLP System

Système de triage médical intelligent basé sur du NLP avancé et du machine learning.

## ✨ Caractéristiques

- **🧠 True NLP**: Apprentissage automatique du dataset, ZERO hardcoding
- **🌍 Multilingue**: Français, Anglais, Arabe
- **📊 Data-Driven**: 4920+ cas médicaux
- **🎯 Intelligent**: Détection automatique des spécialistes
- **💊 Complet**: Symptômes → Maladies → Spécialiste → Recommandations

## 🚀 Installation Rapide
```bash
# Activer environnement
venv\Scripts\activate

# Installer dépendances
pip install -r requirements.txt
python -m spacy download en_core_web_md

# Setup dataset
python setup_dataset.py

# Tester
python test_system.py

# Lancer
streamlit run streamlit_app.py
```

## 📊 Architecture
```
medical-triage-nlp/
├── agents/
│   ├── analyzer/nlp_analyzer_v3.py      # Système NLP principal
│   ├── reasoner/medical_reasoner.py     # Raisonnement (apprend du dataset)
│   ├── decider/decision_generator.py    # Génération décisions
│   └── nlp/spell_corrector.py           # Correction orthographique
├── data/processed/dataset_processed.json
├── streamlit_app.py
├── setup_dataset.py
└── test_system.py
```

## 🎯 TRUE NLP - Pas de hardcoding!

Le système **apprend automatiquement** du dataset:
- Symptôme → Spécialiste (appris, pas hardcodé)
- Symptôme → Urgence (appris, pas hardcodé)
- Traduction multilingue automatique

## 📝 Licence

MIT License

## 👥 Auteur

Arije Bouraoui