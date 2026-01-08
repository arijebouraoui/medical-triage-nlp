"""
NLP Stemmer
============
Stemming et lemmatization multilingue pour texte médical
"""

from typing import List, Dict, Tuple
from enum import Enum
import re


class StemmingAlgorithm(Enum):
    """Algorithmes de stemming disponibles"""
    PORTER = 'porter'
    SNOWBALL = 'snowball'
    MEDICAL = 'medical'


class NLPStemmer:
    """Stemmer multilingue pour texte médical"""
    
    def __init__(self):
        """Initialise le stemmer avec les règles"""
        
        # Règles de stemming pour l'anglais (Porter-like)
        self.english_rules = {
            'suffixes': [
                ('ational', 'ate'),
                ('tional', 'tion'),
                ('enci', 'ence'),
                ('anci', 'ance'),
                ('izer', 'ize'),
                ('abli', 'able'),
                ('alli', 'al'),
                ('entli', 'ent'),
                ('eli', 'e'),
                ('ousli', 'ous'),
                ('ization', 'ize'),
                ('ation', 'ate'),
                ('ator', 'ate'),
                ('alism', 'al'),
                ('iveness', 'ive'),
                ('fulness', 'ful'),
                ('ousness', 'ous'),
                ('aliti', 'al'),
                ('iviti', 'ive'),
                ('biliti', 'ble'),
                ('ness', ''),
                ('ing', ''),
                ('ed', ''),
                ('ly', ''),
                ('s', '')
            ]
        }
        
        # Règles pour le français
        self.french_rules = {
            'suffixes': [
                ('issement', 'ir'),
                ('issement', ''),
                ('ication', 'ique'),
                ('atrice', 'ateur'),
                ('ation', 'er'),
                ('ement', ''),
                ('ment', ''),
                ('ence', 'ent'),
                ('ance', 'ant'),
                ('ité', ''),
                ('eux', ''),
                ('euse', ''),
                ('aux', 'al'),
                ('eau', ''),
                ('elle', 'el'),
                ('er', ''),
                ('é', ''),
                ('es', ''),
                ('s', '')
            ]
        }
        
        # Règles pour l'espagnol
        self.spanish_rules = {
            'suffixes': [
                ('amiento', 'ar'),
                ('imiento', 'ir'),
                ('ación', 'ar'),
                ('ición', 'ir'),
                ('adora', 'ador'),
                ('mente', ''),
                ('anza', ''),
                ('eza', ''),
                ('dad', ''),
                ('oso', ''),
                ('osa', ''),
                ('able', ''),
                ('ible', ''),
                ('ante', ''),
                ('ente', ''),
                ('ción', ''),
                ('or', ''),
                ('ar', ''),
                ('er', ''),
                ('ir', ''),
                ('as', ''),
                ('os', ''),
                ('es', ''),
                ('s', '')
            ]
        }
        
        # Termes médicaux à ne PAS stemmer
        self.medical_exceptions = {
            'diabetes', 'asthma', 'arthritis', 'migraine',
            'nausea', 'vertigo', 'eczema', 'psoriasis',
            'hepatitis', 'bronchitis', 'meningitis'
        }
        
        # Irrégularités communes
        self.irregulars = {
            # Anglais
            'running': 'run',
            'running': 'run',
            'lying': 'lie',
            'tying': 'tie',
            'dying': 'die',
            'teeth': 'tooth',
            'feet': 'foot',
            'geese': 'goose',
            'children': 'child',
            'mice': 'mouse',
            
            # Médical
            'vomiting': 'vomit',
            'vomited': 'vomit',
            'breathing': 'breathe',
            'breathed': 'breathe',
            'coughing': 'cough',
            'coughed': 'cough',
            'bleeding': 'bleed',
            'bled': 'bleed'
        }
    
    def stem_word(self, word: str, language: str = 'en') -> str:
        """
        Applique le stemming à un mot
        
        Args:
            word: Mot à stemmer
            language: Code langue (en, fr, es)
        
        Returns:
            Mot stemmé
        """
        word_lower = word.lower()
        
        # Vérifier exceptions médicales
        if word_lower in self.medical_exceptions:
            return word_lower
        
        # Vérifier irrégularités
        if word_lower in self.irregulars:
            return self.irregulars[word_lower]
        
        # Mot trop court
        if len(word_lower) <= 3:
            return word_lower
        
        # Choisir les règles selon la langue
        if language == 'fr':
            rules = self.french_rules
        elif language == 'es':
            rules = self.spanish_rules
        else:  # 'en' par défaut
            rules = self.english_rules
        
        # Appliquer les règles de suffixes
        for suffix, replacement in rules['suffixes']:
            if word_lower.endswith(suffix):
                stem = word_lower[:-len(suffix)] + replacement
                
                # Vérifier que le stem est valide (au moins 2 caractères)
                if len(stem) >= 2:
                    return stem
        
        return word_lower
    
    def stem_text(self, text: str, language: str = 'en') -> List[str]:
        """
        Applique le stemming à un texte complet
        
        Args:
            text: Texte à stemmer
            language: Code langue
        
        Returns:
            Liste de mots stemmés
        """
        # Tokenize
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Stem chaque mot
        stemmed = [self.stem_word(word, language) for word in words]
        
        return stemmed
    
    def lemmatize_medical(self, word: str) -> str:
        """
        Lemmatization spécifique au domaine médical
        
        Args:
            word: Terme médical
        
        Returns:
            Forme lemmatisée
        """
        word_lower = word.lower()
        
        # Pluriels médicaux spéciaux
        medical_plurals = {
            'bacteria': 'bacterium',
            'fungi': 'fungus',
            'nuclei': 'nucleus',
            'stimuli': 'stimulus',
            'diagnoses': 'diagnosis',
            'prognoses': 'prognosis',
            'crises': 'crisis',
            'analyses': 'analysis',
            'vertebrae': 'vertebra',
            'larvae': 'larva'
        }
        
        if word_lower in medical_plurals:
            return medical_plurals[word_lower]
        
        # Termes en -itis (inflammation)
        if word_lower.endswith('itis'):
            return word_lower  # Ne pas modifier
        
        # Termes en -osis (condition)
        if word_lower.endswith('osis'):
            return word_lower
        
        # Termes en -oma (tumeur)
        if word_lower.endswith('oma'):
            return word_lower
        
        # Sinon, appliquer stemming normal
        return self.stem_word(word_lower)
    
    def get_root_and_variations(self, word: str, language: str = 'en') -> Dict[str, List[str]]:
        """
        Trouve la racine d'un mot et génère ses variations
        
        Args:
            word: Mot de base
            language: Langue
        
        Returns:
            Dict avec root et variations
        """
        root = self.stem_word(word, language)
        
        variations = [word.lower(), root]
        
        # Générer variations communes
        if language == 'en':
            variations.extend([
                root + 'ing',
                root + 'ed',
                root + 's',
                root + 'ly',
                root + 'ness'
            ])
        elif language == 'fr':
            variations.extend([
                root + 'er',
                root + 'é',
                root + 'ement',
                root + 's'
            ])
        elif language == 'es':
            variations.extend([
                root + 'ar',
                root + 'ado',
                root + 'ción',
                root + 's'
            ])
        
        return {
            'root': root,
            'variations': list(set(variations))
        }
    
    def compare_stems(self, word1: str, word2: str, language: str = 'en') -> bool:
        """
        Compare deux mots par leur racine
        
        Args:
            word1, word2: Mots à comparer
            language: Langue
        
        Returns:
            True si même racine
        """
        stem1 = self.stem_word(word1, language)
        stem2 = self.stem_word(word2, language)
        
        return stem1 == stem2
    
    def batch_stem(self, words: List[str], language: str = 'en') -> Dict[str, str]:
        """
        Applique le stemming à une liste de mots
        
        Args:
            words: Liste de mots
            language: Langue
        
        Returns:
            Dict {mot_original: mot_stemmé}
        """
        return {word: self.stem_word(word, language) for word in words}


# ==============================================================================
# EXEMPLE D'UTILISATION
# ==============================================================================

if __name__ == "__main__":
    stemmer = NLPStemmer()
    
    print("="*70)
    print("🌱 NLP STEMMER - TEST")
    print("="*70)
    
    # Test 1: Stemming anglais
    print("\n📝 Test 1: Stemming anglais")
    english_words = [
        'running', 'runs', 'runner',
        'walking', 'walked', 'walks',
        'vomiting', 'vomited', 'vomits',
        'breathing', 'breathed', 'breaths',
        'painful', 'painfully', 'painfulness'
    ]
    
    for word in english_words:
        stemmed = stemmer.stem_word(word, 'en')
        print(f"   {word:15} → {stemmed}")
    
    # Test 2: Stemming français
    print("\n📝 Test 2: Stemming français")
    french_words = [
        'douloureux', 'douloureuse', 'douleur',
        'vomissement', 'vomissements', 'vomir',
        'respiration', 'respiratoire', 'respirer',
        'faiblesse', 'faiblement', 'faible'
    ]
    
    for word in french_words:
        stemmed = stemmer.stem_word(word, 'fr')
        print(f"   {word:15} → {stemmed}")
    
    # Test 3: Stemming espagnol
    print("\n📝 Test 3: Stemming espagnol")
    spanish_words = [
        'doloroso', 'dolorosa', 'dolor',
        'vómito', 'vomitar', 'vomitando',
        'respiración', 'respirar', 'respirando',
        'debilidad', 'débil', 'debilitado'
    ]
    
    for word in spanish_words:
        stemmed = stemmer.stem_word(word, 'es')
        print(f"   {word:15} → {stemmed}")
    
    # Test 4: Lemmatization médicale
    print("\n📝 Test 4: Lemmatization médicale")
    medical_terms = [
        'bacteria', 'diagnoses', 'vertebrae',
        'bronchitis', 'arthritis', 'hepatitis'
    ]
    
    for term in medical_terms:
        lemma = stemmer.lemmatize_medical(term)
        print(f"   {term:15} → {lemma}")
    
    # Test 5: Comparaison de racines
    print("\n📝 Test 5: Comparaison de racines")
    pairs = [
        ('running', 'runs'),
        ('painful', 'pain'),
        ('vomiting', 'vomit'),
        ('breathe', 'breathing')
    ]
    
    for w1, w2 in pairs:
        same = stemmer.compare_stems(w1, w2)
        print(f"   '{w1}' ↔ '{w2}': {same}")
    
    # Test 6: Variations de mots
    print("\n📝 Test 6: Génération de variations")
    test_words = ['pain', 'breathe', 'vomit']
    
    for word in test_words:
        result = stemmer.get_root_and_variations(word)
        print(f"\n   Mot: {word}")
        print(f"   Racine: {result['root']}")
        print(f"   Variations: {result['variations'][:5]}")
    
    # Test 7: Stemming de texte complet
    print("\n📝 Test 7: Stemming de texte complet")
    text = "I have been experiencing severe headaches and stomach pains with vomiting"
    stemmed = stemmer.stem_text(text)
    
    print(f"\n   Original: {text}")
    print(f"   Stemmé:   {' '.join(stemmed)}")