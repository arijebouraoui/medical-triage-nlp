"""
TRUE NLP SYSTEM 
==========================================

"""

import os
import sys
from typing import List, Dict, Set, Tuple
from collections import defaultdict
import re
import unicodedata
import html

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from agents.data_loader.medical_data_loader import MedicalDataLoader
from agents.data_loader.medical_data_loader import MedicalDataLoader
from agents.nlp.context_spell_corrector import ContextSpellCorrector

try:
    from agents.analyzer.ml_classifier import MedicalMLClassifier
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


try:
    from deep_translator import GoogleTranslator
    TRANSLATOR = True
except ImportError:
    TRANSLATOR = False

try:
    import spacy
    SPACY = True
except ImportError:
    SPACY = False

try:
    from agents.nlp_advanced.nlp_foundations import NLPFoundations
    NLP_FOUNDATIONS = True
except:
    NLP_FOUNDATIONS = False

try:
    from agents.nlp_advanced.medical_word2vec import MedicalWord2Vec
    WORD2VEC = True
except:
    WORD2VEC = False


class MedicalNLPAnalyzer:
    """
    Système NLP médical COMPLET basé sur les TPs
    =============================================
    """
    
    def __init__(self, data_path: str = "data/processed/dataset_processed.json", use_spacy: bool = True):
        print("\n" + "="*70)
        print("🧠 TRUE NLP SYSTEM - VERSION FINALE COMPLÈTE")
        print("="*70)
        
        # Data loader
        self.data_loader = MedicalDataLoader(data_path)
        medical_vocab = self.data_loader.get_all_symptoms()
        
        # Spell corrector (True NLP - Context Aware)
        self.spell_corrector = ContextSpellCorrector()
        
        # Load dataset corpus for spell checker training
        print("   📚 Entraînement Correcteur Contextuel...")
        corpus = [case.get('patient_text', '') for case in self.data_loader.dataset if case.get('patient_text')]
        self.spell_corrector.train(corpus)
        
        # Traducteurs
        if TRANSLATOR:
            self.translators = {
                'fr': GoogleTranslator(source='fr', target='en'),
                'es': GoogleTranslator(source='es', target='en'),
                'ar': GoogleTranslator(source='ar', target='en'),
            }
            print("✅ Traducteurs multi-langues")
        else:
            self.translators = {}
        
        # spaCy
        self.nlp_en = None
        if SPACY:
            try:
                self.nlp_en = spacy.load("en_core_web_sm")
                print("✅ spaCy EN chargé")
            except:
                print("⚠️  spaCy EN non disponible")
        
        # NLP Foundations
        if NLP_FOUNDATIONS:
            self.nlp_foundations = NLPFoundations()
            print("✅ NLP Foundations")
        else:
            self.nlp_foundations = None
        
        # Word2Vec (TP2)
        if WORD2VEC:
            self.word2vec = MedicalWord2Vec(data_path)
            self.word2vec.train_cbow(vector_size=100, window=5, epochs=5)
            print("✅ Word2Vec (TP2)")
        else:
            self.word2vec = None

        # ML Classifier (True NLP)
        if ML_AVAILABLE:
            self.ml_classifier = MedicalMLClassifier()
            print("✅ ML Classifier (True NLP)")
        else:
            self.ml_classifier = None
        
        # Apprentissage
        print("\n📚 Apprentissage...")
        self._learn_from_dataset()
        
        # Entraînement ML
        if self.ml_classifier:
            self.ml_classifier.train(self.data_loader.dataset)
            
        self._build_multilingual_medical_dict()
        
        # TP1: Stopwords FR/EN
        self._build_stopwords()
        
        self.session_history = defaultdict(list)
        
        print("\n✅ SYSTÈME PRÊT!")
        print("="*70)
    
    # ========================================================================
    # TP1: PREPROCESSING & NORMALISATION
    # ========================================================================
    
    def _build_stopwords(self):
        """TP1: Stopwords FR/EN"""
        self.stopwords_en = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
            'you', 'your', 'yours', 'yourself', 'yourselves',
            'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
            'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
            'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
            'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
            'against', 'between', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
            'over', 'under', 'again', 'further', 'then', 'once'
        }
        
        self.stopwords_fr = {
            'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'au', 'aux',
            'je', 'tu', 'il', 'elle', 'nous', 'vous', 'ils', 'elles',
            'me', 'te', 'se', 'mon', 'ton', 'son', 'ma', 'ta', 'sa',
            'mes', 'tes', 'ses', 'notre', 'votre', 'leur', 'nos', 'vos', 'leurs',
            'ce', 'cet', 'cette', 'ces', 'qui', 'que', 'quoi', 'dont', 'où',
            'et', 'ou', 'mais', 'donc', 'or', 'ni', 'car',
            'à', 'dans', 'par', 'pour', 'en', 'vers', 'avec', 'sans', 'sous', 'sur',
            'ai', 'as', 'a', 'avons', 'avez', 'ont',
            'suis', 'es', 'est', 'sommes', 'êtes', 'sont',
        }
        
        print(f"   ✅ Stopwords: {len(self.stopwords_en)} EN + {len(self.stopwords_fr)} FR")
    
    def to_lower(self, text: str) -> str:
        """TP1 Ex1a: Minuscules"""
        return text.lower()
    
    def remove_html(self, text: str) -> str:
        """TP1 Ex1b: Supprimer balises HTML et décoder entités"""
        # Supprimer balises
        text = re.sub(r'<[^>]+>', '', text)
        # Décoder entités HTML
        text = html.unescape(text)
        return text
    
    def normalize_quotes_dashes(self, text: str) -> str:
        """TP1 Ex1c: Normaliser apostrophes/tirets"""
        # Apostrophes
        text = re.sub(r'[''`]', "'", text)
        # Tirets
        text = re.sub(r'[–—]', '-', text)
        # Espaces insécables
        text = re.sub(r'\xa0', ' ', text)
        return text
    
    def strip_accents(self, text: str) -> str:
        """TP1 Ex1d: Supprimer accents (via unicodedata)"""
        # Normaliser en NFD (décomposition)
        nfd = unicodedata.normalize('NFD', text)
        # Filtrer les marques diacritiques
        without_accents = ''.join(c for c in nfd if unicodedata.category(c) != 'Mn')
        return without_accents
    
    def extract_emails(self, text: str) -> List[str]:
        """TP1 Ex5a: Extraire emails"""
        pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.findall(pattern, text)
    
    def extract_urls(self, text: str) -> List[str]:
        """TP1 Ex5b: Extraire URLs"""
        pattern = r'https?://[^\s]+|www\.[^\s]+'
        return re.findall(pattern, text)
    
    def extract_phones(self, text: str) -> List[str]:
        """TP1 Ex5d: Extraire téléphones"""
        patterns = [
            r'\+\d{1,3}\s?\d{2}\s?\d{3}\s?\d{3}',  # +216 22 345 678
            r'\d{2}-\d{3}-\d{3}',  # 71-123-456
            r'\(\d{5}\)\s?\d{2}\s?\d{3}\s?\d{3}',  # (00216) 93 111 222
        ]
        phones = []
        for pattern in patterns:
            phones.extend(re.findall(pattern, text))
        return phones
    
    def extract_hashtags_mentions(self, text: str) -> Tuple[List[str], List[str]]:
        """TP1 Ex5e: Extraire hashtags & mentions"""
        hashtags = re.findall(r'#\w+', text)
        mentions = re.findall(r'@\w+', text)
        return hashtags, mentions
    
    def remove_urls(self, text: str) -> str:
        """TP1 Ex1e: Supprimer URLs"""
        text = re.sub(r'https?://[^\s]+', '', text)
        text = re.sub(r'www\.[^\s]+', '', text)
        return text
    
    def remove_emails(self, text: str) -> str:
        """TP1 Ex1e: Supprimer emails"""
        return re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
    
    def remove_phones(self, text: str) -> str:
        """TP1 Ex1e: Supprimer téléphones"""
        text = re.sub(r'\+\d{1,3}\s?\d{2}\s?\d{3}\s?\d{3}', '', text)
        text = re.sub(r'\d{2}-\d{3}-\d{3}', '', text)
        text = re.sub(r'\(\d{5}\)\s?\d{2}\s?\d{3}\s?\d{3}', '', text)
        return text
    
    def remove_extra_spaces(self, text: str) -> str:
        """TP1 Ex1h: Supprimer espaces multiples"""
        return re.sub(r'\s+', ' ', text).strip()
    
    def regex_tokenize(self, text: str) -> List[str]:
        """TP1 Ex3a: Tokenisation regex (conserve #hashtags/@mentions)"""
        # Pattern: mots, hashtags, mentions
        pattern = r'#\w+|@\w+|\w+'
        tokens = re.findall(pattern, text.lower())
        # Filtrer tokens < 2 caractères (sauf hashtags/mentions)
        return [t for t in tokens if len(t) >= 2 or t[0] in ['#', '@']]
    
    def filter_stopwords(self, tokens: List[str], lang: str = 'en') -> List[str]:
        """TP1 Ex3b: Filtrer stopwords"""
        stopwords = self.stopwords_en if lang == 'en' else self.stopwords_fr
        return [t for t in tokens if t not in stopwords]
    
    def clean_text_pipeline(self, text: str) -> str:
        """
        TP1 Ex2: Pipeline de nettoyage
        ================================
        Ordre motivé:
        1. HTML (avant tout, pour éviter interférences)
        2. Minuscules (normalisation)
        3. Quotes/dashes (normalisation)
        4. URLs/emails/phones (suppression données sensibles)
        5. Espaces multiples (nettoyage final)
        """
        text = self.remove_html(text)
        text = self.to_lower(text)
        text = self.normalize_quotes_dashes(text)
        text = self.remove_urls(text)
        text = self.remove_emails(text)
        text = self.remove_phones(text)
        text = self.remove_extra_spaces(text)
        return text
    
    # ========================================================================
    # DICTIONNAIRE MÉDICAL ENRICHI
    # ========================================================================
    
    def _build_multilingual_medical_dict(self):
        """Dictionnaire médical multilingue ENRICHI (100+ mots)"""
        
        print("📖 Construction dictionnaire médical...")
        
        self.medical_dict = {'en': {}, 'fr': {}, 'es': {}, 'ar': {}}
        
        # Vocabulaire médical anglais COMPLET
        medical_words_en = [
            # Organes & parties du corps
            'stomach', 'heart', 'head', 'chest', 'back', 'knee', 'eye', 'eyes',
            'throat', 'skin', 'leg', 'legs', 'arm', 'arms', 'hand', 'hands',
            'foot', 'feet', 'neck', 'shoulder', 'shoulders',
            'tooth', 'teeth', 'gum', 'gums', 'mouth', 'tongue', 'jaw',
            'ear', 'ears', 'nose', 'lung', 'lungs', 'liver', 'kidney', 'kidneys',
            'brain', 'bone', 'bones', 'muscle', 'muscles', 'joint', 'joints',
            'elbow', 'wrist', 'ankle', 'hip', 'spine', 'abdomen', 'belly',
            
            # Symptômes
            'pain', 'ache', 'aching', 'hurt', 'hurts', 'hurting', 'sore',
            'fever', 'feverish', 'cough', 'coughing',
            'bleeding', 'bleed', 'swelling', 'swollen',
            'nausea', 'nauseous', 'vomiting', 'vomit',
            'dizziness', 'dizzy', 'fatigue', 'tired', 'weakness', 'weak',
            'itching', 'itchy', 'rash', 'redness', 'red',
            'burning', 'numbness', 'numb', 'tingling',
            
            # Symptômes composés
            'headache', 'toothache', 'stomachache', 'backache',
            'earache', 'sore throat',
        ]
        
        for word_en in medical_words_en:
            self.medical_dict['en'][word_en] = word_en
        
        # Traductions françaises COMPLÈTES
        manual_fr = {
            # Organes
            'estomac': 'stomach', 'coeur': 'heart', 'cœur': 'heart',
            'tête': 'head', 'tete': 'head', 'poitrine': 'chest',
            'dos': 'back', 'genou': 'knee', 'genoux': 'knees',
            'œil': 'eye', 'oeil': 'eye', 'yeux': 'eyes',
            'gorge': 'throat', 'peau': 'skin',
            'jambe': 'leg', 'jambes': 'legs', 'bras': 'arm',
            'main': 'hand', 'mains': 'hands', 'pied': 'foot', 'pieds': 'feet',
            'cou': 'neck', 'épaule': 'shoulder', 'epaule': 'shoulder',
            'épaules': 'shoulders', 'epaules': 'shoulders',
            
            # Bouche/dents
            'dent': 'tooth', 'dents': 'teeth',
            'gencive': 'gum', 'gencives': 'gums',
            'bouche': 'mouth', 'langue': 'tongue',
            'mâchoire': 'jaw', 'machoire': 'jaw',
            
            # Autres organes
            'oreille': 'ear', 'oreilles': 'ears', 'nez': 'nose',
            'poumon': 'lung', 'poumons': 'lungs',
            'foie': 'liver', 'rein': 'kidney', 'reins': 'kidneys',
            'cerveau': 'brain', 'os': 'bone', 'muscle': 'muscle',
            'articulation': 'joint', 'coude': 'elbow',
            'poignet': 'wrist', 'cheville': 'ankle', 'hanche': 'hip',
            'colonne': 'spine', 'ventre': 'belly', 'abdomen': 'abdomen',
            
            # Symptômes
            'douleur': 'pain', 'mal': 'ache', 'souffrance': 'pain',
            'fièvre': 'fever', 'fievre': 'fever',
            'toux': 'cough', 'saignement': 'bleeding', 'saigner': 'bleed',
            'gonflement': 'swelling', 'enflure': 'swelling', 'gonflé': 'swollen',
            'nausée': 'nausea', 'nausee': 'nausea',
            'vomissement': 'vomiting', 'vomir': 'vomit',
            'vertige': 'dizziness', 'étourdissement': 'dizziness',
            'fatigue': 'fatigue', 'fatigué': 'tired',
            'faiblesse': 'weakness', 'faible': 'weak',
            'démangeaison': 'itching', 'demangeaison': 'itching',
            'éruption': 'rash', 'eruption': 'rash',
            'rougeur': 'redness', 'rouge': 'red',
            'brûlure': 'burning', 'brulure': 'burning',
            'engourdissement': 'numbness', 'engourdi': 'numb',
            
            # Mots de liaison fréquents (pour aider la traduction mot-à-mot)
            "j'ai": "i have", "jai": "i have",
            "au": "in", "aux": "in", "à": "in",
            "les": "the", "le": "the", "la": "the",
            "un": "a", "une": "a",
            "mes": "my", "mon": "my", "ma": "my",
        }
        
        for fr, en in manual_fr.items():
            self.medical_dict['fr'][fr] = en
        
        print(f"   ✅ Dictionnaire EN: {len(self.medical_dict['en'])} mots")
        print(f"   ✅ Dictionnaire FR: {len(self.medical_dict['fr'])} mots")
    
    # ========================================================================
    # APPRENTISSAGE DU DATASET
    # ========================================================================
    
    def _learn_from_dataset(self):
        """Apprend symptômes et patterns du dataset"""
        
        all_symptoms = self.data_loader.get_all_symptoms()
        
        self.symptom_index = {}
        self.body_parts = set()
        self.symptom_types = set()
        self.symptom_patterns = []
        
        for symptom in all_symptoms:
            if not self.nlp_en:
                continue
            
            doc = self.nlp_en(symptom)
            
            structure = {
                'original': symptom,
                'tokens': [t.text.lower() for t in doc],
                'lemmas': [t.lemma_.lower() for t in doc],
                'nouns': [t.lemma_.lower() for t in doc if t.pos_ == 'NOUN'],
            }
            
            self.symptom_index[symptom] = structure
            
            for noun in structure['nouns']:
                self.body_parts.add(noun)
            
            types = ['pain', 'ache', 'aching', 'hurt', 'hurts', 'hurting', 'sore',
                    'bleeding', 'bleed', 'swelling', 'swollen', 'redness', 'red',
                    'itching', 'itchy', 'fever', 'cough', 'coughing',
                    'nausea', 'nauseous', 'vomiting', 'vomit']
            for token in structure['tokens']:
                if token in types:
                    self.symptom_types.add(token)
            
            pattern = self._extract_pattern(structure)
            if pattern:
                self.symptom_patterns.append({'symptom': symptom, 'pattern': pattern})
        
        # Body parts essentielles
        essential_body_parts = [
            'tooth', 'teeth', 'gum', 'gums', 'heart', 'chest', 'lung', 'lungs',
            'stomach', 'abdomen', 'belly', 'head', 'brain', 'eye', 'eyes', 'ear', 'ears',
            'throat', 'nose', 'mouth', 'skin', 'hair', 'bone', 'muscle', 'joint',
            'kidney', 'kidneys', 'liver', 'bladder',
        ]
        
        for body_part in essential_body_parts:
            self.body_parts.add(body_part)
        
        print(f"   ✅ {len(self.symptom_index)} symptômes")
        print(f"   ✅ {len(self.body_parts)} parties du corps")
    
    def _extract_pattern(self, structure: Dict) -> Dict:
        """Extrait pattern d'un symptôme"""
        body_parts = structure['nouns']
        symptom_types = [t for t in structure['tokens']
                        if t in ['pain', 'ache', 'bleeding', 'swelling', 'fever', 'cough']]
        
        if body_parts and symptom_types:
            return {'body_parts': body_parts, 'types': symptom_types}
        
        if 'headache' in structure['tokens']:
            return {'body_parts': ['head'], 'types': ['ache', 'pain']}
        
        return None
    
    # ========================================================================
    # ANALYSE PRINCIPALE
    # ========================================================================
    
    def analyze(self, patient_text: str, session_id: str = None) -> Dict:
        """Analyse INTELLIGENTE avec preprocessing TP1 + Word2Vec TP2"""
        
        if session_id is None:
            session_id = f"session_{id(patient_text)}"
        
        print(f"\n{'='*70}")
        print(f"🧠 ANALYSE: '{patient_text}'")
        print(f"{'='*70}")
        
        # 1. Détection langue
        detected_lang = self._detect_language(patient_text)
        print(f"\n1️⃣  Langue: {detected_lang.upper()}")
        
        # 2. Correction orthographique (Contexte)
        # On corrige D'ABORD, dans la langue détectée
        corrected_text, corrections = self.spell_corrector.correct_text(patient_text, detected_lang)
        if corrections:
            print(f"2️⃣  Correction ({detected_lang.upper()}): {len(corrections)}")
            print(f"    '{patient_text}' → '{corrected_text}'")
            
        # 3. Traduction intelligente (du texte corrigé)
        translated = self._smart_translate(corrected_text, detected_lang)
        print(f"3️⃣  Traduction: '{translated}'")
        
        # 3b. Re-correction en Anglais (si traduit)
        if detected_lang != 'en':
             translated, en_corrections = self.spell_corrector.correct_text(translated, 'en')
             if en_corrections:
                 print(f"    Re-correction EN: {len(en_corrections)}")
                 corrections.extend(en_corrections)
        
        # 4. Lemmatization
        # On utilise 'translated' qui est maintenant le texte EN corrigé
        lemmatized, doc = self._lemmatize(translated)
        print(f"4️⃣  Lemmatization: '{lemmatized}'")
        
        # 5. Extraction concepts
        patient_concepts = self._extract_concepts(doc, translated)
        print(f"5️⃣  Concepts:")
        if patient_concepts['body_parts']:
            print(f"    Body parts: {patient_concepts['body_parts']}")
        if patient_concepts['symptom_types']:
            print(f"    Types: {patient_concepts['symptom_types']}")
        
        # 6. TF-IDF
        tfidf_scores = {}
        if self.nlp_foundations:
            corpus = [turn['text'] for turn in self.session_history[session_id]]
            corpus.append(lemmatized)
            tfidf_scores = self.nlp_foundations.compute_tfidf(lemmatized, corpus)
        
        # 7. Word2Vec (TP2)
        word2vec_similar = {}
        if self.word2vec:
            tokens = lemmatized.split()
            for token in set(tokens):
                if len(token) > 2:
                    similar = self.word2vec.get_similar_words(token, 'cbow', topn=3)
                    if similar:
                        word2vec_similar[token] = similar
        
        # 8. MATCHING
        print(f"\n8️⃣  MATCHING:")
        symptoms = self._match_symptoms(lemmatized, doc, patient_concepts, tfidf_scores, word2vec_similar)
        
        print(f"\n✅ {len(symptoms)} symptôme(s):")
        for s in symptoms:
            print(f"   • {s['symptom']} ({s['method']}, {s['confidence']:.0%})")
        
        # Maladies
        diseases = self.data_loader.find_diseases_by_symptoms([s['symptom'] for s in symptoms])
        
        # ML Prediction
        ml_results = {}
        if self.ml_classifier:
            ml_results = self.ml_classifier.predict(patient_text)
            if ml_results:
                # FIX: Nettoyage encodage
                for key in ['ml_specialist', 'ml_urgency']:
                    if key in ml_results:
                        val = ml_results[key]
                        val = val.replace('Ã‰', 'É').replace('Ã¨', 'è').replace('Ã', 'à')
                        # Correctif spécifique pour MODÉRÉE qui casse souvent
                        if "MOD" in val and "R" in val and "E" in val: 
                            if "ELEV" not in val: # Pas ELEVEE
                                val = "URGENCE MODÉRÉE"
                        
                        ml_results[key] = val

                print(f"\n🧠 Intelligence Artificielle (ML):")
                print(f"   • Spécialiste suggéré: {ml_results['ml_specialist']} ({ml_results['ml_specialist_confidence']:.0%})")
                print(f"   • Urgence estimée: {ml_results['ml_urgency']} ({ml_results['ml_urgency_confidence']:.0%})")
        
        # Historique
        self.session_history[session_id].append({'text': lemmatized, 'symptoms': symptoms})
        
        result = {
            'symptoms': symptoms,
            'all_session_symptoms': self._get_all_session_symptoms(session_id),
            'possible_diseases': dict(list(diseases.items())[:5]),
            'corrections': corrections,
            'language': detected_lang,
            'session_id': session_id,
            'processed_text': lemmatized,
            **ml_results,  # Ajouter les résultats ML
            'statistics': {
                'total_symptoms_found': len(symptoms),
                'total_corrections': len(corrections),
                'total_diseases_matched': len(diseases)
            },
            'patient_text': patient_text,
            'detected_language': detected_lang,
            'original_language': detected_lang,
            'emergency_numbers': self._get_emergency_numbers_by_lang(detected_lang)
        }
        
        print(f"\n{'='*70}\n")
        
        return result
    
    def _smart_translate(self, text: str, base_lang: str) -> str:
        """Traduction intelligente mot par mot"""
        
        # Si multilingue ou mixe détecté, on force la traduction vers l'Anglais (Base Model)
        
        # Si c'est déjà de l'anglais, on retourne
        if base_lang == 'en':
             return text
             
        # 1. Google Translate (Priorité)
        if base_lang in self.translators:
             try:
                 translated = self.translators[base_lang].translate(text)
                 print(f"    🌐 Google Translate ({base_lang}->en): '{text}' → '{translated}'")
                 return translated.lower()
             except Exception as e:
                 print(f"    ⚠️ Erreur traduction Google: {e}")
        
        # 2. Fallback Dictionnaire Manuel (Si Google échoue ou pas dispo)
        print("    📖 Fallback Dictionnaire...")
        words = text.lower().split()
        translated_words = []
        
        for word in words:
            clean_word = re.sub(r'[^\wàâäéèêëïîôùûüÿç]', '', word)
            if not clean_word:
                translated_words.append(word)
                continue
                
            # Dictionnaire médical
            translated = None
            for lang, lang_dict in self.medical_dict.items():
                if clean_word in lang_dict:
                    translated = lang_dict[clean_word]
                    break
            
            if translated:
                translated_words.append(translated)
            else:
                translated_words.append(clean_word)
                
        return ' '.join(translated_words)
    
    def _detect_language(self, text: str) -> str:
        """Détecte la langue du texte"""
        text_lower = text.lower()
        
        if re.search(r'[\u0600-\u06FF]', text):
            return 'ar'
        
        if "'" in text and any(x in text_lower for x in ["j'", "d'", "l'", "m'", "t'", "s'"]):
            return 'fr'
        
        if re.search(r'[àâäéèêëïîôùûüÿç]', text_lower):
            return 'fr'
        
        if re.search(r'[áéíóúñü]', text_lower):
            return 'es'
        
        fr_words = ['je', 'suis', 'ai', 'mon', 'ma', 'au', 'aux', 'du', 'mal', 'douleur']
        if any(word in text_lower.split() for word in fr_words):
            return 'fr'
        
        return 'en'
    
    def _lemmatize(self, text: str):
        """Lemmatization avec spaCy"""
        if not self.nlp_en:
            return text, None
        
        doc = self.nlp_en(text)
        lemmas = [token.lemma_ for token in doc]
        
        return ' '.join(lemmas), doc
    
    def _extract_concepts(self, doc, text: str) -> Dict:
        """Extrait concepts du texte"""
        concepts = {
            'body_parts': [],
            'symptom_types': [],
            'all_nouns': [],
            'all_tokens': []
        }
        
        if not doc:
            return concepts
        
        symptom_type_words = {
            'pain', 'ache', 'aching', 'hurt', 'hurts', 'hurting', 'sore',
            'bleeding', 'bleed', 'swelling', 'swollen',
            'redness', 'red', 'itching', 'itchy',
            'fever', 'cough', 'coughing',
            'nausea', 'nauseous', 'vomiting', 'vomit'
        }
        
        not_body_parts = {
            'pain', 'ache', 'hurt', 'sore',
            'bleeding', 'swelling', 'redness', 'fever', 'cough', 'nausea'
        }
        
        for token in doc:
            lemma = token.lemma_.lower()
            text_token = token.text.lower()
            
            if lemma in self.body_parts and lemma not in not_body_parts:
                concepts['body_parts'].append(lemma)
            
            if lemma in self.symptom_types or text_token in symptom_type_words:
                concepts['symptom_types'].append(lemma)
            
            if token.pos_ == 'NOUN' and lemma not in not_body_parts:
                concepts['all_nouns'].append(lemma)
            
            concepts['all_tokens'].append(lemma)
        
        return concepts
    
    def _match_symptoms(self, text: str, doc, patient_concepts: Dict,
                       tfidf_scores: Dict, word2vec_similar: Dict) -> List[Dict]:
        """Match symptômes avec le dataset"""
        matched = []
        
        # Match exact
        for symptom in self.symptom_index:
            if symptom in text:
                matched.append({'symptom': symptom, 'method': 'exact', 'confidence': 1.0})
                print(f"   ✓ Exact: '{symptom}'")
        
        # Match sémantique
        patient_body = set(patient_concepts['body_parts'])
        patient_types = set(patient_concepts['symptom_types'])
        
        # Équivalences
        body_equivalents = {
            'heart': ['heart', 'chest'],
            'stomach': ['stomach', 'abdomen', 'belly'],
            'head': ['head', 'brain'],
            'eye': ['eye', 'eyes'],
            'tooth': ['tooth', 'teeth', 'gum', 'gums'],
            'teeth': ['tooth', 'teeth', 'gum', 'gums'],
            'gum': ['tooth', 'teeth', 'gum', 'gums'],
        }
        
        extended_body = set()
        for body in patient_body:
            extended_body.add(body)
            if body in body_equivalents:
                extended_body.update(body_equivalents[body])
        
        patient_body = extended_body
        
        # pain/ache équivalents
        if 'pain' in patient_concepts['all_tokens']:
            patient_types.update(['pain', 'ache'])
        if 'ache' in patient_concepts['all_tokens']:
            patient_types.update(['pain', 'ache'])
        if 'hurt' in patient_concepts['all_tokens']:
            patient_types.update(['pain', 'ache', 'hurt'])
        
        if any(w in patient_concepts['all_tokens'] for w in ['have', 'feel']):
            if patient_body:
                patient_types.update(['pain', 'ache'])
        
        if patient_body and patient_types:
            print(f"   🎯 Sémantique: {patient_body} + {patient_types}")
            
            for pattern in self.symptom_patterns:
                symptom = pattern['symptom']
                pattern_body = set(pattern['pattern']['body_parts'])
                pattern_types = set(pattern['pattern']['types'])
                
                if patient_body.intersection(pattern_body) and patient_types.intersection(pattern_types):
                    if not any(m['symptom'] == symptom for m in matched):
                        matched.append({'symptom': symptom, 'method': 'semantic', 'confidence': 0.95})
                        print(f"   ✓ Sémantique: '{symptom}'")
            
            # FIX: Fallback critique pour douleurs coeur/poitrine
            if 'heart' in patient_body or 'chest' in patient_body:
                if 'pain' in patient_types or 'ache' in patient_types:
                    if not any(m['symptom'] == 'chest pain' for m in matched):
                        matched.append({'symptom': 'chest pain', 'method': 'critical_fallback', 'confidence': 1.0})
                        print(f"   ⚠️ CRITICAL: Heart/Chest Pain detected -> Force 'chest pain'")
        
        # Lemma match
        patient_lemmas = set(patient_concepts['all_tokens'])
        
        for symptom, structure in self.symptom_index.items():
            symptom_lemmas = set(structure['lemmas'])
            
            if symptom_lemmas.issubset(patient_lemmas):
                if not any(m['symptom'] == symptom for m in matched):
                    matched.append({'symptom': symptom, 'method': 'lemma', 'confidence': 0.90})
                    print(f"   ✓ Lemma: '{symptom}'")
        
        # Dédupliquer
        unique = {}
        for m in matched:
            s = m['symptom']
            if s not in unique or m['confidence'] > unique[s]['confidence']:
                unique[s] = m
        
        return sorted(unique.values(), key=lambda x: x['confidence'], reverse=True)[:5]
    
    def _get_emergency_numbers_by_lang(self, lang: str) -> Dict:
        """
        CORRECTION: Retourne numéros d'urgence selon la langue
        ========================================================
        """
        # Mapping langue → pays par défaut
        country_by_lang = {
            'fr': 'France',
            'ar': 'Tunisie',
            'en': 'USA',
            'es': 'Spain',
        }
        
        # Numéros d'urgence par pays
        emergency_by_country = {
            'Tunisie': {'samu': '190', 'urgences': '197', 'police': '197', 'pompiers': '198'},
            'France': {'samu': '15', 'urgences': '112', 'police': '17', 'pompiers': '18'},
            'UK': {'emergency': '999', 'urgences': '112', 'police': '999', 'ambulance': '999'},
            'USA': {'emergency': '911', 'police': '911', 'ambulance': '911', 'fire': '911'},
            'Canada': {'emergency': '911', 'police': '911', 'ambulance': '911', 'fire': '911'},
        }
        
        country = country_by_lang.get(lang, 'USA')
        return emergency_by_country.get(country, emergency_by_country['USA'])
    
    def get_session_summary(self, session_id: str) -> Dict:
        """Retourne un résumé complet de la session"""
        history = self.session_history.get(session_id, [])
        all_symptoms = self._get_all_session_symptoms(session_id)
        
        # Récupérer maladies possibles basées sur TOUS les symptômes
        symptom_names = [s['symptom'] for s in all_symptoms]
        diseases = self.data_loader.find_diseases_by_symptoms(symptom_names)
        
        return {
            'total_turns': len(history),
            'total_symptoms': len(all_symptoms),
            'symptoms': all_symptoms,
            'possible_diseases': dict(list(diseases.items())[:5])
        }

    def _get_all_session_symptoms(self, session_id: str) -> List[Dict]:
        """Récupère tous les symptômes de la session"""
        all_symptoms = []
        seen = set()
        for turn in self.session_history[session_id]:
            for symptom in turn['symptoms']:
                s = symptom['symptom']
                if s not in seen:
                    all_symptoms.append(symptom)
                    seen.add(s)
        return all_symptoms


# Alias
TrueNLPAnalyzer = MedicalNLPAnalyzer
CompleteNLPAnalyzer = MedicalNLPAnalyzer
DataDrivenNLPAnalyzer = MedicalNLPAnalyzer