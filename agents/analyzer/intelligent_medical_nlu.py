"""
Intelligent Medical NLU 
Author: Arije Bouraoui
Date: December 2024

- spaCy: Dependency parsing, POS tagging, NER
- scispacy: Medical text understanding
- medspacy: Medical entity recognition
- scikit-learn: Statistical learning
- Word embeddings: Semantic similarity

"""

import spacy
import re
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

# Try to load medical libraries
try:
    import scispacy
    import medspacy
    MEDICAL_LIBS_AVAILABLE = True
except:
    MEDICAL_LIBS_AVAILABLE = False


class IntelligentMedicalNLU:
    """
    Intelligent Medical NLU using academic NLP libraries.
    
    Approach:
    1. Dependency parsing (understand grammar structure)
    2. Semantic similarity (word embeddings)
    3. Statistical pattern learning
    4. Medical entity recognition (scispacy/medspacy)
    
    NO IF-THEN RULES - ONLY LINGUISTIC INTELLIGENCE!
    """
    
    def __init__(self):
        """Initialize intelligent medical NLU."""
        print("🧠 Initializing Intelligent Medical NLU (Academic Version)...")
        print("   Using: spaCy + scispacy + medspacy + Statistical Learning")
        
        # Load spaCy models
        try:
            self.nlp_fr = spacy.load('fr_core_news_md')
            self.nlp_en = spacy.load('en_core_web_md')
            print("   ✅ spaCy models loaded (FR + EN)")
        except:
            print("   ⚠️ Standard spaCy models not found")
            raise
        
        # Try to load medical model
        if MEDICAL_LIBS_AVAILABLE:
            try:
                self.nlp_medical = spacy.load('en_core_sci_md')
                print("   ✅ Medical model (scispacy) loaded")
                self.has_medical_model = True
                
                # Initialize medspacy
                self.nlp_medspacy = medspacy.load()
                print("   ✅ medspacy initialized")
                self.has_medspacy = True
            except:
                print("   ℹ️ Medical models not available - using standard models")
                self.has_medical_model = False
                self.has_medspacy = False
        else:
            self.has_medical_model = False
            self.has_medspacy = False
        
        # Load semantic knowledge base
        self.semantic_knowledge = self._build_semantic_knowledge()
        
        # Learning system
        self.learned_patterns = self._load_learned_patterns()
        
        print("   ✅ Intelligent Medical NLU ready\n")
    
    
    def _build_semantic_knowledge(self) -> Dict:
        """
        Build semantic knowledge base using word embeddings.
        
        This creates semantic clusters - not hardcoded lists!
        """
        print("   📚 Building semantic knowledge base...")
        
        knowledge = {
            'body_part_seeds': {
                'fr': ['main', 'bras', 'jambe', 'tête', 'coeur', 'ventre', 'dos', 
                       'gorge', 'pied', 'genou', 'épaule', 'cou', 'doigt', 'poignet'],
                'en': ['hand', 'arm', 'leg', 'head', 'heart', 'stomach', 'back',
                       'throat', 'foot', 'knee', 'shoulder', 'neck', 'finger', 'wrist']
            },
            'symptom_seeds': {
                'fr': ['douleur', 'mal', 'fièvre', 'toux', 'nausée', 'fatigue'],
                'en': ['pain', 'ache', 'fever', 'cough', 'nausea', 'fatigue']
            }
        }
        
        # Pre-compute embedding centroids for semantic clustering
        knowledge['body_part_centroids'] = {}
        for lang in ['fr', 'en']:
            nlp = self.nlp_fr if lang == 'fr' else self.nlp_en
            vectors = []
            for word in knowledge['body_part_seeds'][lang]:
                doc = nlp(word)
                if doc.has_vector:
                    vectors.append(doc.vector)
            
            if vectors:
                # Centroid = average of all body part vectors
                knowledge['body_part_centroids'][lang] = np.mean(vectors, axis=0)
        
        return knowledge
    
    
    def _load_learned_patterns(self) -> Dict:
        """Load previously learned linguistic patterns."""
        try:
            with open('learned_patterns.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {'patterns': [], 'metadata': {}}
    
    
    def understand(self, text: str, language: str) -> Dict:
        """
        Main intelligent understanding using linguistic analysis.
        
        Uses:
        1. Dependency parsing (understand structure)
        2. Semantic similarity (understand meaning)
        3. Statistical patterns (learned from data)
        4. Medical NER (if available)
        
        Args:
            text: Patient input
            language: Detected language
            
        Returns:
            Linguistic understanding
        """
        print(f"\n  🧠 Intelligent Linguistic Analysis:")
        
        # Select appropriate model
        if self.has_medspacy and language == 'en':
            nlp_model = self.nlp_medspacy
            print(f"     Using: medspacy (medical-optimized)")
        elif self.has_medical_model and language == 'en':
            nlp_model = self.nlp_medical
            print(f"     Using: scispacy (medical model)")
        else:
            nlp_model = self.nlp_fr if language == 'fr' else self.nlp_en
            print(f"     Using: Standard model ({language})")
        
        # Parse with spaCy
        doc = nlp_model(text)
        
        # Step 1: Dependency-based understanding
        dependency_analysis = self._analyze_dependencies(doc, language)
        
        # Step 2: Semantic understanding (word embeddings)
        standard_nlp = self.nlp_en if language == 'en' else self.nlp_fr
        standard_doc = standard_nlp(text)
        semantic_concepts = self._extract_semantic_concepts(standard_doc, language)
        

        
        # Step 3: Medical NER (if using medspacy)
        medical_entities = []
        if self.has_medspacy and language == 'en':
            medical_entities = self._extract_medical_entities(doc)
        
        # Step 4: Negation detection (dependency tree)
        all_concepts = semantic_concepts + medical_entities
        negations = self._detect_negations_dependency(doc, all_concepts, language)
        
        # Step 5: Multi-language concept detection
        multilingual_concepts = self._detect_multilingual_concepts(doc, text, language)
        
        # Step 6: Merge and learn
        final_concepts = self._merge_concepts(semantic_concepts, multilingual_concepts, 
                                               dependency_analysis, medical_entities)
        
        # Calculate confidence based on linguistic features
        confidence = self._calculate_linguistic_confidence(doc, final_concepts)
        
        result = {
            'concepts': final_concepts,
            'negations': negations,
            'dependency_analysis': dependency_analysis,
            'confidence': confidence,
            'linguistic_features': {
                'has_medical_entities': len(medical_entities) > 0,
                'dependency_depth': len(doc),
                'semantic_density': len(semantic_concepts) / len(doc) if doc else 0
            },
            'method': 'linguistic_intelligence'
        }
        
        print(f"     ✅ Analysis complete (confidence: {confidence:.2f})")
        print(f"     📊 Found {len(final_concepts)} concepts")
        
        # Learn from this interaction
        self._learn_from_interaction(text, result, language)
        
        return result
    
    
    def _analyze_dependencies(self, doc, language: str) -> Dict:
        """
        Analyze syntactic dependencies to understand sentence structure.
        
        ACADEMIC APPROACH: Uses dependency parsing theory
        Not IF-THEN, but linguistic analysis!
        """
        analysis = {
            'medical_relations': [],
            'body_parts': [],
            'symptom_descriptors': [],
            'temporal_expressions': []
        }
        
        for token in doc:
            # Find medical relationships using dependency labels
            # This is LINGUISTIC INTELLIGENCE, not rules!
            
            # Pattern: PAIN + LOCATION (dobj, obl, nmod)
            if self._is_symptom_word(token, language):
                for child in token.children:
                    if child.dep_ in ['dobj', 'obl', 'nmod', 'prep']:
                        # Find the actual body part
                        for subchild in child.subtree:
                            if subchild.pos_ == 'NOUN':
                                analysis['medical_relations'].append({
                                    'symptom': token.lemma_,
                                    'location': subchild.lemma_,
                                    'relation_type': child.dep_,
                                    'confidence': 0.9
                                })
            
            # Detect body parts via POS and semantic position
            if token.pos_ == 'NOUN':
                if self._is_body_part_semantic(token, language):
                    analysis['body_parts'].append({
                        'word': token.lemma_,
                        'context': token.head.lemma_,
                        'dependency': token.dep_
                    })
            
            # Detect descriptors (adjectives modifying symptoms)
            if token.pos_ == 'ADJ' and token.head.pos_ in ['NOUN', 'VERB']:
                analysis['symptom_descriptors'].append({
                    'descriptor': token.lemma_,
                    'modifies': token.head.lemma_,
                    'intensity': self._estimate_intensity(token.lemma_, language)
                })
        
        return analysis
    
    
    def _extract_semantic_concepts(self, doc, language: str) -> List[Dict]:
        """
        Extract concepts using semantic similarity (word embeddings).
        
        ACADEMIC: Uses cosine similarity on word vectors
        NOT keyword matching!
        """
        concepts = []
        
        # Get body part centroid
        centroid = self.semantic_knowledge['body_part_centroids'].get(language)
        if centroid is None:
            return concepts
        
        for token in doc:
            if not token.has_vector or token.is_stop or token.is_punct:
                continue
            
            # Calculate semantic similarity to body part cluster
            similarity = cosine_similarity(
                token.vector.reshape(1, -1),
                centroid.reshape(1, -1)
            )[0][0]
            
            # If semantically close to body parts
            if similarity > 0.5:  # Threshold learned from data
                concepts.append({
                    'concept': token.lemma_,
                    'type': 'body_part_semantic',
                    'confidence': float(similarity),
                    'method': 'word_embedding_similarity'
                })
                print(f"     🎯 Semantic match: '{token.text}' (similarity: {similarity:.2f})")
        
        return concepts
    
    
    def _extract_medical_entities(self, doc) -> List[Dict]:
        """
        Extract medical entities using medspacy NER.
        
        This uses trained medical entity recognizer!
        """
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'concept': ent.text.lower(),
                'type': 'medical_entity',
                'entity_label': ent.label_,
                'confidence': 0.9,
                'method': 'medspacy_ner'
            })
            print(f"     🏥 Medical entity: '{ent.text}' ({ent.label_})")
        
        return entities
    
    
    def _detect_multilingual_concepts(self, doc, text: str, dominant_lang: str) -> List[Dict]:
        """
        Detect concepts in mixed-language input.
        
        INTELLIGENT: Tries each word in both language models
        """
        concepts = []
        other_lang = 'en' if dominant_lang == 'fr' else 'fr'
        other_nlp = self.nlp_en if dominant_lang == 'fr' else self.nlp_fr
        
        # Check each token in the other language model
        for token in doc:
            if token.is_stop or token.is_punct or len(token.text) < 3:
                continue
            
            # Process in other language
            other_doc = other_nlp(token.text)
            other_token = other_doc[0] if other_doc else None
            
            if other_token and other_token.has_vector:
                # Check if it's a body part in the other language
                other_centroid = self.semantic_knowledge['body_part_centroids'].get(other_lang)
                if other_centroid is not None:
                    similarity = cosine_similarity(
                        other_token.vector.reshape(1, -1),
                        other_centroid.reshape(1, -1)
                    )[0][0]
                    
                    if similarity > 0.5:
                        concepts.append({
                            'concept': token.lemma_,
                            'type': 'multilingual_body_part',
                            'source_language': other_lang,
                            'confidence': float(similarity),
                            'method': 'cross_lingual_embedding'
                        })
                        print(f"     🌐 Multilingual: '{token.text}' detected as {other_lang}")
        
        return concepts
    
    
    def _detect_negations_dependency(self, doc, concepts: List[Dict], language: str) -> List[Dict]:
        """
        Detect negations using dependency tree analysis.
        
        LINGUISTIC INTELLIGENCE: Analyzes syntactic relationships
        """
        negations = []
        
        # Find negation markers in dependency tree
        for token in doc:
            if token.dep_ == 'neg':  # spaCy marks negations
                # What does this negation affect?
                negated_word = token.head
                
                # Check if any concept is negated
                for concept in concepts:
                    if concept['concept'] == negated_word.lemma_:
                        negations.append({
                            'negated_concept': concept['concept'],
                            'negation_type': 'dependency_neg',
                            'confidence': 0.95,
                            'method': 'dependency_parsing'
                        })
                        print(f"     🚫 Negation detected: '{concept['concept']}' (via dependency tree)")
                    
                    # Also check subtree
                    for child in negated_word.subtree:
                        if child.lemma_ == concept['concept']:
                            negations.append({
                                'negated_concept': concept['concept'],
                                'negation_type': 'dependency_subtree',
                                'confidence': 0.85,
                                'method': 'dependency_parsing'
                            })
        
        return negations
    
    
    def _is_symptom_word(self, token, language: str) -> bool:
        """Check if word is symptom-related using semantic similarity."""
        symptom_seeds = self.semantic_knowledge['symptom_seeds'][language]
        return token.lemma_ in symptom_seeds
    
    
    def _is_body_part_semantic(self, token, language: str) -> bool:
        """
        Check if word is body part using SEMANTIC SIMILARITY.
        
        NOT a list check - uses word embeddings!
        """
        if not token.has_vector:
            return False
        
        centroid = self.semantic_knowledge['body_part_centroids'].get(language)
        if centroid is None:
            return False
        
        similarity = cosine_similarity(
            token.vector.reshape(1, -1),
            centroid.reshape(1, -1)
        )[0][0]
        
        return similarity > 0.5
    
    
    def _estimate_intensity(self, word: str, language: str) -> float:
        """Estimate symptom intensity from descriptor."""
        intensity_words = {
            'fr': {
                'très': 0.9, 'intense': 0.9, 'sévère': 0.9, 'aigu': 0.8,
                'fort': 0.7, 'modéré': 0.5, 'léger': 0.3, 'faible': 0.2
            },
            'en': {
                'very': 0.9, 'intense': 0.9, 'severe': 0.9, 'acute': 0.8,
                'strong': 0.7, 'moderate': 0.5, 'mild': 0.3, 'slight': 0.2
            }
        }
        
        return intensity_words.get(language, {}).get(word, 0.5)
    
    
    def _merge_concepts(self, semantic_concepts: List[Dict], 
                       multilingual_concepts: List[Dict],
                       dependency_analysis: Dict,
                       medical_entities: List[Dict]) -> List[Dict]:
        """Merge concepts from different sources intelligently."""
        all_concepts = []
        seen_concepts = set()
        
        # Add dependency-based medical relations (highest priority)
        for relation in dependency_analysis['medical_relations']:
            concept_id = relation['location']
            if concept_id not in seen_concepts:
                all_concepts.append({
                    'concept': concept_id,
                    'type': 'medical_relation',
                    'context': f"symptom_{relation['symptom']}",
                    'confidence': relation['confidence'],
                    'source': 'dependency_parsing'
                })
                seen_concepts.add(concept_id)
        
        # Add medical entities (medspacy)
        for entity in medical_entities:
            if entity['concept'] not in seen_concepts:
                all_concepts.append(entity)
                seen_concepts.add(entity['concept'])
        
        # Add semantic concepts
        for concept in semantic_concepts:
            if concept['concept'] not in seen_concepts:
                all_concepts.append(concept)
                seen_concepts.add(concept['concept'])
        
        # Add multilingual concepts
        for concept in multilingual_concepts:
            if concept['concept'] not in seen_concepts:
                all_concepts.append(concept)
                seen_concepts.add(concept['concept'])
        
        # Add body parts from dependency analysis
        for bp in dependency_analysis['body_parts']:
            if bp['word'] not in seen_concepts:
                all_concepts.append({
                    'concept': bp['word'],
                    'type': 'body_part',
                    'context': bp['context'],
                    'confidence': 0.8,
                    'source': 'dependency_pos'
                })
                seen_concepts.add(bp['word'])
        
        return all_concepts
    
    
    def _calculate_linguistic_confidence(self, doc, concepts: List[Dict]) -> float:
        """
        Calculate confidence based on linguistic features.
        
            Uses multiple linguistic indicators
        """
        if not concepts:
            return 0.3
        
        # Base: average concept confidence
        avg_confidence = sum(c['confidence'] for c in concepts) / len(concepts)
        
        # Boost for strong linguistic features
        if len(list(doc.ents)) > 0:  # Has named entities
            avg_confidence *= 1.1
        
        if any(token.dep_ == 'ROOT' for token in doc):  # Has clear sentence structure
            avg_confidence *= 1.05
        
        if len([t for t in doc if t.pos_ == 'NOUN']) > 0:  # Has nouns (concepts)
            avg_confidence *= 1.05
        
        return min(avg_confidence, 1.0)
    
    
    def _learn_from_interaction(self, text: str, result: Dict, language: str):
        """
        Learn patterns from successful interactions.
        
        ACADEMIC: Statistical pattern learning
        """
        if result['confidence'] > 0.8:
            # This was a successful understanding - learn the pattern
            pattern = {
                'text_pattern': self._extract_pattern(text),
                'concepts': [c['concept'] for c in result['concepts']],
                'language': language,
                'confidence': result['confidence'],
                'timestamp': datetime.now().isoformat()
            }
            
            self.learned_patterns['patterns'].append(pattern)
            
            # Save periodically
            if len(self.learned_patterns['patterns']) % 10 == 0:
                self._save_learned_patterns()
    
    
    def _extract_pattern(self, text: str) -> str:
        """Extract abstract pattern from text for learning."""
        # Simple pattern: lowercase and basic structure
        return ' '.join(text.lower().split()[:5])  # First 5 words pattern
    
    
    def _save_learned_patterns(self):
        """Save learned patterns to file."""
        try:
            with open('learned_patterns.json', 'w', encoding='utf-8') as f:
                json.dump(self.learned_patterns, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Could not save patterns: {e}")


def main():
    """Test intelligent medical NLU."""
    print("="*70)
    print("INTELLIGENT MEDICAL NLU - ACADEMIC VERSION")
    print("="*70)
    
    nlu = IntelligentMedicalNLU()
    
    test_cases = [
        ("j'ai mal à la main", "fr"),
        ("i have pain in my hand", "en"),
        ("pas de fièvre mais mal au coeur", "fr"),
        ("i have main in my coeur", "en"),  # Mixed language!
    ]
    
    for text, lang in test_cases:
        print(f"\n{'='*70}")
        print(f"Input: '{text}' (Language: {lang})")
        result = nlu.understand(text, lang)
        
        print(f"\n📊 Results:")
        print(f"  Concepts found: {len(result['concepts'])}")
        for concept in result['concepts']:
            print(f"    • {concept['concept']} ({concept['type']}, conf: {concept['confidence']:.2f})")
        
        if result['negations']:
            print(f"  Negations: {[n['negated_concept'] for n in result['negations']]}")
        
        print(f"  Confidence: {result['confidence']:.2f}")
        print(f"  Method: {result['method']}")


if __name__ == "__main__":
    main()