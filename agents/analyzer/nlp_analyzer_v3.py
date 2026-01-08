"""
TRUE NLP SYSTEM - Vraiment intelligent
=======================================
Détecte et traduit automatiquement les mots de n'importe quelle langue
"""

import os
import sys
from typing import List, Dict, Set
from collections import defaultdict
import re

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from agents.data_loader.medical_data_loader import MedicalDataLoader
from agents.nlp.spell_corrector import SpellCorrector

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


class TrueNLPAnalyzer:
    """Système NLP vraiment intelligent"""
    
    def __init__(self, data_path: str = "data/processed/dataset_processed.json"):
        print("\n" + "="*70)
        print("🧠 TRUE NLP SYSTEM - VRAIMENT INTELLIGENT")
        print("="*70)
        
        # Data
        self.data_loader = MedicalDataLoader(data_path)
        medical_vocab = self.data_loader.get_all_symptoms()
        self.spell_corrector = SpellCorrector(medical_vocab)
        
        # Traducteurs Google pour TOUTES les langues
        if TRANSLATOR:
            self.translators = {
                'fr': GoogleTranslator(source='fr', target='en'),
                'es': GoogleTranslator(source='es', target='en'),
                'ar': GoogleTranslator(source='ar', target='en'),
                'de': GoogleTranslator(source='de', target='en'),
                'it': GoogleTranslator(source='it', target='en'),
            }
            print("✅ Traducteurs multi-langues")
        else:
            self.translators = {}
        
        # spaCy
        self.nlp_en = None
        if SPACY:
            try:
                self.nlp_en = spacy.load("en_core_web_md")
                print("✅ spaCy chargé")
            except:
                try:
                    self.nlp_en = spacy.load("en_core_web_sm")
                    print("✅ spaCy (small) chargé")
                except:
                    pass
        
        # NLP Foundations
        if NLP_FOUNDATIONS:
            self.nlp_foundations = NLPFoundations()
            print("✅ NLP Foundations")
        else:
            self.nlp_foundations = None
        
        # Word2Vec
        if WORD2VEC:
            self.word2vec = MedicalWord2Vec(data_path)
            self.word2vec.train_cbow(vector_size=100, window=5, epochs=5)
            print("✅ Word2Vec")
        else:
            self.word2vec = None
        
        # APPRENTISSAGE
        print("\n📚 Apprentissage...")
        self._learn_from_dataset()
        
        # Dictionnaire médical MULTILINGUE
        self._build_multilingual_medical_dict()
        
        self.session_history = defaultdict(list)
        
        print("\n✅ SYSTÈME PRÊT!")
        print("="*70)
    
    def _build_multilingual_medical_dict(self):
        """Construit dictionnaire médical multilingue automatiquement"""
        
        print("📖 Construction dictionnaire multilingue...")
        
        self.medical_dict = {
            'en': {},  # anglais (base)
            'fr': {},  # français
            'es': {},  # espagnol
            'ar': {},  # arabe
        }
        
        # Mots médicaux courants à traduire
        medical_words_en = [
            'stomach', 'heart', 'head', 'chest', 'back', 'knee', 'eye', 'eyes',
            'throat', 'skin', 'leg', 'arm', 'hand', 'foot', 'neck', 'shoulder',
            'pain', 'ache', 'hurt', 'fever', 'cough', 'bleeding', 'swelling',
            'nausea', 'vomiting', 'dizziness', 'fatigue',
        ]
        
        # Traduire automatiquement
        for word_en in medical_words_en:
            self.medical_dict['en'][word_en] = word_en
            
            # Traduire en français
            if 'fr' in self.translators:
                try:
                    # Utiliser traducteur inverse
                    translator_en_fr = GoogleTranslator(source='en', target='fr')
                    word_fr = translator_en_fr.translate(word_en).lower()
                    self.medical_dict['fr'][word_fr] = word_en
                except:
                    pass
            
            # Traductions manuelles importantes (fallback)
            manual_fr = {
                'stomach': 'estomac', 'heart': 'coeur', 'head': 'tête',
                'chest': 'poitrine', 'pain': 'douleur', 'ache': 'mal',
            }
            if word_en in manual_fr:
                self.medical_dict['fr'][manual_fr[word_en]] = word_en
        
        print(f"   ✅ Dictionnaire: {len(self.medical_dict['fr'])} mots FR")
    
    def _learn_from_dataset(self):
        """Apprend du dataset"""
        
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
                self.symptom_patterns.append({
                    'symptom': symptom,
                    'pattern': pattern
                })
        
        # AJOUTER MANUELLEMENT les body parts manquantes importantes
        essential_body_parts = [
            'tooth', 'teeth', 'gum', 'gums',  # Dentaire
            'heart', 'chest', 'lung', 'lungs',  # Cardio/Pulmonaire
            'stomach', 'abdomen', 'belly',  # Digestif
            'head', 'brain',  # Neurologie
            'eye', 'eyes', 'ear', 'ears',  # Sensoriel
            'throat', 'nose', 'mouth',  # ORL
            'skin', 'hair',  # Dermatologie
            'bone', 'muscle', 'joint',  # Musculo-squelettique
            'kidney', 'kidneys', 'liver', 'bladder',  # Organes
        ]
        
        for body_part in essential_body_parts:
            self.body_parts.add(body_part)
        
        print(f"   ✅ {len(self.symptom_index)} symptômes")
        print(f"   ✅ {len(self.body_parts)} parties du corps (+ essentielles)")
    
    def _extract_pattern(self, structure: Dict) -> Dict:
        body_parts = structure['nouns']
        symptom_types = [t for t in structure['tokens'] 
                        if t in ['pain', 'ache', 'bleeding', 'swelling', 'fever', 'cough']]
        
        if body_parts and symptom_types:
            return {'body_parts': body_parts, 'types': symptom_types}
        
        if 'headache' in structure['tokens']:
            return {'body_parts': ['head'], 'types': ['ache', 'pain']}
        
        return None
    
    def analyze(self, patient_text: str, session_id: str = None) -> Dict:
        """Analyse INTELLIGENTE"""
        
        if session_id is None:
            session_id = f"session_{id(patient_text)}"
        
        print(f"\n{'='*70}")
        print(f"🧠 ANALYSE: '{patient_text}'")
        print(f"{'='*70}")
        
        # 1. Détecter langue globale
        detected_lang = self._detect_language(patient_text)
        print(f"\n1️⃣  Langue globale: {detected_lang.upper()}")
        
        # 2. TRANSLATION MOT PAR MOT (intelligent!)
        translated = self._smart_translate(patient_text, detected_lang)
        print(f"2️⃣  Traduction intelligente: '{translated}'")
        
        # 3. Correction
        corrected, corrections = self.spell_corrector.correct_text(translated, 'en')
        if corrections:
            print(f"3️⃣  Corrections: {len(corrections)}")
        
        # 4. Lemmatization
        lemmatized, doc = self._lemmatize(corrected)
        print(f"4️⃣  Lemmatization: '{lemmatized}'")
        
        # 5. Extraction concepts
        patient_concepts = self._extract_concepts(doc, corrected)
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
        
        # 7. Word2Vec
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
        symptoms = self._match_symptoms(
            lemmatized,
            doc,
            patient_concepts,
            tfidf_scores,
            word2vec_similar
        )
        
        print(f"\n✅ {len(symptoms)} symptôme(s):")
        for s in symptoms:
            print(f"   • {s['symptom']} ({s['method']}, {s['confidence']:.0%})")
        
        # Maladies
        diseases = self.data_loader.find_diseases_by_symptoms([s['symptom'] for s in symptoms])
        
        # Historique
        self.session_history[session_id].append({
            'text': lemmatized,
            'symptoms': symptoms
        })
        
        result = {
            'symptoms': symptoms,
            'all_session_symptoms': self._get_all_session_symptoms(session_id),
            'possible_diseases': dict(list(diseases.items())[:5]),
            'corrections': corrections,
            'language': detected_lang,
            'session_id': session_id,
            'processed_text': lemmatized,
            'statistics': {
                'total_symptoms_found': len(symptoms),
                'total_corrections': len(corrections),
                'total_diseases_matched': len(diseases)
            },
            'patient_text': patient_text,
            'detected_language': detected_lang,
            'original_language': detected_lang,
            'emergency_numbers': self._get_emergency_numbers(detected_lang)
        }
        
        print(f"\n{'='*70}\n")
        
        return result
    
    def _smart_translate(self, text: str, base_lang: str) -> str:
        """Traduction INTELLIGENTE mot par mot"""
        
        words = text.lower().split()
        translated_words = []
        
        for word in words:
            # Nettoyer
            clean_word = re.sub(r'[^\wàâäéèêëïîôùûüÿç]', '', word)
            
            if not clean_word:
                translated_words.append(word)
                continue
            
            # Si le mot est déjà anglais (dans body_parts ou symptom_types)
            if clean_word in self.body_parts or clean_word in self.symptom_types:
                translated_words.append(clean_word)
                continue
            
            # Chercher dans dictionnaire médical
            translated = None
            for lang, lang_dict in self.medical_dict.items():
                if clean_word in lang_dict:
                    translated = lang_dict[clean_word]
                    print(f"    📖 Dict: '{clean_word}' → '{translated}'")
                    break
            
            if translated:
                translated_words.append(translated)
                continue
            
            # Essayer traduction Google du mot seul
            if base_lang != 'en' and base_lang in self.translators:
                try:
                    translated = self.translators[base_lang].translate(clean_word).lower()
                    if translated != clean_word:
                        print(f"    🌐 Google: '{clean_word}' → '{translated}'")
                        translated_words.append(translated)
                        continue
                except:
                    pass
            
            # Garder mot original
            translated_words.append(clean_word)
        
        return ' '.join(translated_words)
    
    def _detect_language(self, text: str) -> str:
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
        if not self.nlp_en:
            return text, None
        
        doc = self.nlp_en(text)
        lemmas = [token.lemma_ for token in doc]
        
        return ' '.join(lemmas), doc
    
    def _extract_concepts(self, doc, text: str) -> Dict:
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
        matched = []
        
        # Match exact
        for symptom in self.symptom_index:
            if symptom in text:
                matched.append({
                    'symptom': symptom,
                    'method': 'exact',
                    'confidence': 1.0
                })
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
            'tooth': ['tooth', 'teeth', 'gum', 'gums'],  # Dentaire
            'teeth': ['tooth', 'teeth', 'gum', 'gums'],
            'gum': ['tooth', 'teeth', 'gum', 'gums'],
        }
        
        extended_body = set()
        for body in patient_body:
            extended_body.add(body)
            if body in body_equivalents:
                extended_body.update(body_equivalents[body])
        
        patient_body = extended_body
        
        # pain/ache/hurt équivalents
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
                        matched.append({
                            'symptom': symptom,
                            'method': 'semantic',
                            'confidence': 0.95
                        })
                        print(f"   ✓ Sémantique: '{symptom}'")
        
        # Lemma match
        patient_lemmas = set(patient_concepts['all_tokens'])
        
        for symptom, structure in self.symptom_index.items():
            symptom_lemmas = set(structure['lemmas'])
            
            if symptom_lemmas.issubset(patient_lemmas):
                if not any(m['symptom'] == symptom for m in matched):
                    matched.append({
                        'symptom': symptom,
                        'method': 'lemma',
                        'confidence': 0.90
                    })
                    print(f"   ✓ Lemma: '{symptom}'")
        
        # Partial match
        for symptom, structure in self.symptom_index.items():
            symptom_words = set(structure['tokens'])
            patient_words = set(patient_concepts['all_tokens'])
            
            common = symptom_words.intersection(patient_words)
            stopwords = {'the', 'a', 'an', 'in', 'on', 'at', 'to', 'of', 'my'}
            common_meaningful = common - stopwords
            
            if len(common_meaningful) >= 2:
                if not any(m['symptom'] == symptom for m in matched):
                    matched.append({
                        'symptom': symptom,
                        'method': 'partial',
                        'confidence': 0.85
                    })
                    print(f"   ✓ Partial: '{symptom}'")
        
        unique = {}
        for m in matched:
            s = m['symptom']
            if s not in unique or m['confidence'] > unique[s]['confidence']:
                unique[s] = m
        
        return sorted(unique.values(), key=lambda x: x['confidence'], reverse=True)[:5]
    
    def _get_emergency_numbers(self, lang: str) -> Dict:
        if lang == 'fr' or lang == 'ar':
            return {'samu': '190', 'urgences': '197', 'police': '197', 'pompiers': '198'}
        return {'emergency': '112'}
    
    def _get_all_session_symptoms(self, session_id: str) -> List[Dict]:
        all_symptoms = []
        seen = set()
        for turn in self.session_history[session_id]:
            for symptom in turn['symptoms']:
                s = symptom['symptom']
                if s not in seen:
                    all_symptoms.append(symptom)
                    seen.add(s)
        return all_symptoms
    
    def get_session_summary(self, session_id: str) -> Dict:
        all_symptoms = self._get_all_session_symptoms(session_id)
        diseases = self.data_loader.find_diseases_by_symptoms([s['symptom'] for s in all_symptoms])
        return {
            'session_id': session_id,
            'total_turns': len(self.session_history[session_id]),
            'total_symptoms': len(all_symptoms),
            'symptoms': all_symptoms,
            'possible_diseases': dict(list(diseases.items())[:10])
        }
    
    def clear_session(self, session_id: str):
        if session_id in self.session_history:
            del self.session_history[session_id]


# Alias
MedicalNLPAnalyzer = TrueNLPAnalyzer
CompleteNLPAnalyzer = TrueNLPAnalyzer