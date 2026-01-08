"""
Advanced Medical NLP Engine
============================
Système NLP médical avancé utilisant:
- spaCy + medspaCy pour analyse médicale
- Word embeddings pour similarité sémantique
- Lemmatization et normalisation
- Dictionnaire médical étendu
- Matching intelligent multi-niveaux
"""

import os
import sys
from typing import List, Dict, Tuple, Set
import re

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Imports spaCy et medspaCy
try:
    import spacy
    from spacy.matcher import PhraseMatcher
    import medspacy
    from medspacy.ner import TargetRule
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("⚠️  spaCy ou medspaCy non disponible")


class AdvancedMedicalNLP:
    """Moteur NLP médical avancé"""
    
    def __init__(self):
        """Initialise le moteur NLP"""
        
        print("\n🔬 Initialisation Advanced Medical NLP...")
        
        # Charger spaCy et medspaCy
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                # Essayer le modèle médical d'abord
                try:
                    self.nlp = spacy.load("en_core_sci_md")
                    print("   ✅ Modèle médical chargé (en_core_sci_md)")
                except:
                    # Sinon modèle standard
                    self.nlp = spacy.load("en_core_web_md")
                    print("   ✅ Modèle standard chargé (en_core_web_md)")
                
                # Ajouter medspaCy
                try:
                    self.nlp = medspacy.load()
                    print("   ✅ medspaCy chargé")
                except:
                    print("   ⚠️  medspaCy non chargé, utilisation spaCy seul")
                
            except Exception as e:
                print(f"   ⚠️  Erreur chargement: {e}")
                self.nlp = None
        
        # Dictionnaire de normalisation médical (ÉTENDU)
        self.medical_normalizations = {
            # Dents
            'teeth': 'tooth',
            'tooth': 'tooth',
            'toothache': 'toothache',
            'dental': 'tooth',
            'gum': 'gum',
            'gums': 'gum',
            
            # Douleur
            'hurt': 'pain',
            'hurts': 'pain',
            'hurting': 'pain',
            'ache': 'pain',
            'aching': 'pain',
            'painful': 'pain',
            'sore': 'pain',
            
            # Saignement
            'blood': 'bleeding',
            'bleed': 'bleeding',
            'bleeding': 'bleeding',
            'hemorrhage': 'bleeding',
            
            # Coeur
            'heart': 'heart',
            'cardiac': 'heart',
            'chest': 'chest',
            
            # Estomac
            'stomach': 'stomach',
            'belly': 'stomach',
            'abdomen': 'stomach',
            'abdominal': 'stomach',
            'tummy': 'stomach',
            
            # Tête
            'head': 'head',
            'headache': 'headache',
            'migraine': 'migraine',
            
            # Symptômes communs
            'nausea': 'nausea',
            'nauseous': 'nausea',
            'vomit': 'vomiting',
            'vomiting': 'vomiting',
            'throw up': 'vomiting',
            'fever': 'fever',
            'temperature': 'fever',
            'hot': 'fever',
            'cough': 'cough',
            'coughing': 'cough',
            'dizzy': 'dizziness',
            'dizziness': 'dizziness',
            'weak': 'weakness',
            'weakness': 'weakness',
            'tired': 'fatigue',
            'fatigue': 'fatigue',
        }
        
        # Mapping symptômes → termes médicaux
        self.symptom_mappings = {
            'tooth pain': ['toothache', 'dental pain', 'tooth ache'],
            'tooth bleeding': ['gum bleeding', 'bleeding gums', 'gingival bleeding'],
            'chest pain': ['heart pain', 'cardiac pain', 'angina'],
            'stomach pain': ['abdominal pain', 'belly pain', 'gastric pain'],
            'head pain': ['headache', 'cephalalgia'],
        }
        
        # Patterns de symptômes composés
        self.compound_patterns = [
            # Pattern: [body_part] + [pain/hurt/ache]
            (r'\b(tooth|teeth|head|stomach|chest|heart|back|neck|knee)\b.*\b(hurt|pain|ache|sore)\b', 
             lambda m: f"{self._normalize(m.group(1))} pain"),
            
            # Pattern: [body_part] + [bleed/bleeding]
            (r'\b(tooth|teeth|gum|nose|stomach)\b.*\b(bleed|bleeding|blood)\b',
             lambda m: f"{self._normalize(m.group(1))} bleeding"),
            
            # Pattern: my [body_part] hurts
            (r'my\s+(\w+)\s+hurts?',
             lambda m: f"{self._normalize(m.group(1))} pain"),
        ]
        
        print("   ✅ Dictionnaires médicaux chargés")
    
    def analyze(self, text: str) -> Dict:
        """
        Analyse NLP complète d'un texte médical
        
        Returns:
            Dict avec:
            - normalized_terms: termes normalisés
            - lemmas: lemmes
            - entities: entités médicales
            - symptoms: symptômes détectés
            - embeddings: vecteurs si disponibles
        """
        
        result = {
            'original_text': text,
            'normalized_terms': [],
            'lemmas': [],
            'entities': [],
            'symptoms': [],
            'tokens': [],
            'compound_symptoms': []
        }
        
        text_lower = text.lower()
        
        # ÉTAPE 1: Détection de patterns composés
        print(f"\n🔍 Analyse: '{text}'")
        print("   📋 ÉTAPE 1: Détection patterns composés")
        
        for pattern, extract_func in self.compound_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                symptom = extract_func(match)
                result['compound_symptoms'].append(symptom)
                print(f"      ✅ Pattern détecté: '{match.group()}' → '{symptom}'")
        
        # ÉTAPE 2: Normalisation des mots
        print("   📋 ÉTAPE 2: Normalisation")
        words = text_lower.split()
        for word in words:
            normalized = self._normalize(word)
            if normalized != word:
                result['normalized_terms'].append({
                    'original': word,
                    'normalized': normalized
                })
                print(f"      '{word}' → '{normalized}'")
        
        # ÉTAPE 3: Analyse spaCy (lemmatization, POS, entities)
        if self.nlp:
            print("   📋 ÉTAPE 3: Analyse spaCy/medspaCy")
            doc = self.nlp(text)
            
            # Lemmas
            for token in doc:
                if not token.is_stop and not token.is_punct:
                    result['lemmas'].append({
                        'text': token.text,
                        'lemma': token.lemma_,
                        'pos': token.pos_
                    })
                    result['tokens'].append(token.text)
            
            # Entités médicales
            for ent in doc.ents:
                result['entities'].append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                })
                print(f"      🏥 Entité: '{ent.text}' ({ent.label_})")
        
        # ÉTAPE 4: Extraction de symptômes
        print("   📋 ÉTAPE 4: Extraction symptômes")
        symptoms = self._extract_symptoms(text_lower, result)
        result['symptoms'] = symptoms
        
        for symptom in symptoms:
            print(f"      💊 Symptôme: {symptom['term']} (méthode: {symptom['method']}, conf: {symptom['confidence']:.0%})")
        
        return result
    
    def _normalize(self, word: str) -> str:
        """Normalise un mot médical"""
        word = word.lower().strip()
        return self.medical_normalizations.get(word, word)
    
    def _extract_symptoms(self, text: str, analysis: Dict) -> List[Dict]:
        """Extrait les symptômes avec plusieurs méthodes"""
        
        symptoms = []
        seen = set()
        
        # Méthode 1: Patterns composés détectés
        for compound in analysis['compound_symptoms']:
            if compound not in seen:
                symptoms.append({
                    'term': compound,
                    'method': 'compound_pattern',
                    'confidence': 0.95
                })
                seen.add(compound)
        
        # Méthode 2: Termes normalisés + contexte
        normalized_words = set()
        for norm in analysis['normalized_terms']:
            normalized_words.add(norm['normalized'])
        
        # Chercher dans les mappings
        for symptom_key, variations in self.symptom_mappings.items():
            symptom_words = set(symptom_key.split())
            
            # Si tous les mots du symptôme sont dans le texte normalisé
            if symptom_words.issubset(normalized_words):
                if symptom_key not in seen:
                    symptoms.append({
                        'term': symptom_key,
                        'method': 'normalized_mapping',
                        'confidence': 0.9
                    })
                    seen.add(symptom_key)
        
        # Méthode 3: Entités médicales de medspaCy
        for entity in analysis['entities']:
            entity_text = entity['text'].lower()
            if entity_text not in seen:
                symptoms.append({
                    'term': entity_text,
                    'method': 'medspacy_entity',
                    'confidence': 0.85
                })
                seen.add(entity_text)
        
        # Méthode 4: Lemmas avec contexte médical
        if self.nlp:
            lemma_text = ' '.join([l['lemma'] for l in analysis['lemmas']])
            
            # Chercher patterns dans les lemmes
            for pattern, extract_func in self.compound_patterns:
                matches = re.finditer(pattern, lemma_text)
                for match in matches:
                    symptom = extract_func(match)
                    if symptom not in seen:
                        symptoms.append({
                            'term': symptom,
                            'method': 'lemma_pattern',
                            'confidence': 0.8
                        })
                        seen.add(symptom)
        
        # Trier par confiance
        symptoms.sort(key=lambda x: x['confidence'], reverse=True)
        
        return symptoms
    
    def find_similar_symptoms(self, text: str, symptom_list: List[str], threshold: float = 0.7) -> List[Tuple[str, float]]:
        """
        Trouve les symptômes similaires en utilisant word embeddings
        
        Args:
            text: Texte du patient
            symptom_list: Liste de symptômes connus
            threshold: Seuil de similarité (0-1)
        
        Returns:
            Liste de (symptôme, score de similarité)
        """
        
        if not self.nlp or not self.nlp.vocab.vectors.shape[0]:
            print("   ⚠️  Embeddings non disponibles")
            return []
        
        similar = []
        doc = self.nlp(text)
        
        for symptom in symptom_list:
            symptom_doc = self.nlp(symptom)
            
            # Calculer similarité
            similarity = doc.similarity(symptom_doc)
            
            if similarity >= threshold:
                similar.append((symptom, similarity))
                print(f"   🎯 Similarité: '{symptom}' = {similarity:.2f}")
        
        # Trier par similarité
        similar.sort(key=lambda x: x[1], reverse=True)
        
        return similar


# ==============================================================================
# TEST
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 TEST ADVANCED MEDICAL NLP")
    print("="*70)
    
    nlp_engine = AdvancedMedicalNLP()
    
    test_cases = [
        "my teeth hurt and they bleed",
        "i have pain in my heart",
        "my stomach hurts and i feel nauseous",
        "severe headache with vomiting",
        "chest pain radiating to arm"
    ]
    
    for i, test_text in enumerate(test_cases, 1):
        print(f"\n{'#'*70}")
        print(f"TEST {i}/{len(test_cases)}")
        print(f"{'#'*70}")
        
        result = nlp_engine.analyze(test_text)
        
        print(f"\n📊 RÉSULTATS:")
        print(f"   Symptômes détectés: {len(result['symptoms'])}")
        for symptom in result['symptoms']:
            print(f"      • {symptom['term']} ({symptom['method']}, {symptom['confidence']:.0%})")
        
        if result['normalized_terms']:
            print(f"\n   Normalisations: {len(result['normalized_terms'])}")
            for norm in result['normalized_terms'][:5]:
                print(f"      • {norm['original']} → {norm['normalized']}")
        
        if result['entities']:
            print(f"\n   Entités médicales: {len(result['entities'])}")
            for ent in result['entities']:
                print(f"      • {ent['text']} ({ent['label']})")
    
    print("\n" + "="*70)
    print("✅ TESTS TERMINÉS")
    print("="*70)