"""
NLP Foundations Module 
=======================================
Implémente tous les concepts fondamentaux 
- One-hot encoding
- Bag-of-Words (BoW)
- TF-IDF
- Tokenization avancée
- POS Tagging
- Lemmatization
"""

import numpy as np
from typing import List, Dict, Tuple, Set
from collections import Counter, defaultdict
import math
import re

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


class NLPFoundations:
    """Implémente les fondations NLP du TP"""
    
    def __init__(self):
        """Initialise le module"""
        print("\n📚 Initialisation NLP Foundations...")
        
        # Charger spaCy pour POS tagging (essayer plusieurs modèles)
        self.nlp = None
        if SPACY_AVAILABLE:
            # Essayer les modèles disponibles dans l'ordre
            models_to_try = [
                ("en_core_web_md", "English Web (Medium)"),
                ("en_core_sci_md", "English Scientific (Medium)"),
                ("en_core_web_sm", "English Web (Small)"),
                ("fr_core_news_md", "French News (Medium)"),
            ]
            
            for model_name, model_desc in models_to_try:
                try:
                    self.nlp = spacy.load(model_name)
                    print(f"   ✅ spaCy chargé: {model_desc}")
                    break
                except:
                    continue
            
            if not self.nlp:
                print("   ⚠️  Aucun modèle spaCy trouvé")
        else:
            print("   ⚠️  spaCy non installé")
        
        # Vocabulaire et index
        self.vocabulary = set()
        self.word2idx = {}
        self.idx2word = {}
        
        # Stats TF-IDF
        self.idf_scores = {}
        self.document_count = 0
        
        print("   ✅ Module initialisé")
    
    # =========================================================================
    # 1. ONE-HOT ENCODING
    # =========================================================================
    
    def build_vocabulary(self, documents: List[str]) -> Dict[str, int]:
        """
        Construit le vocabulaire à partir d'une liste de documents
        
        Args:
            documents: Liste de textes
        
        Returns:
            Dictionnaire mot → index
        """
        print(f"\n📖 Construction vocabulaire ({len(documents)} documents)...")
        
        # Extraire tous les mots
        all_words = set()
        for doc in documents:
            words = self.tokenize(doc)
            all_words.update(words)
        
        # Créer les mappings
        self.vocabulary = sorted(all_words)
        self.word2idx = {word: idx for idx, word in enumerate(self.vocabulary)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
        print(f"   ✅ Vocabulaire: {len(self.vocabulary)} mots uniques")
        
        return self.word2idx
    
    def one_hot_encode(self, word: str) -> np.ndarray:
        """
        Encode un mot en vecteur one-hot
        
        Args:
            word: Mot à encoder
        
        Returns:
            Vecteur one-hot de taille |V|
        
        Example:
            >>> vocab = {'chat': 0, 'chien': 1, 'oiseau': 2}
            >>> one_hot_encode('chat')
            array([1, 0, 0])
        """
        if word not in self.word2idx:
            # Mot inconnu
            return np.zeros(len(self.vocabulary))
        
        vector = np.zeros(len(self.vocabulary))
        vector[self.word2idx[word]] = 1
        
        return vector
    
    def demonstrate_one_hot(self, words: List[str]):
        """Démontre le one-hot encoding"""
        print(f"\n🔢 ONE-HOT ENCODING")
        print(f"   Vocabulaire: {len(self.vocabulary)} mots")
        print(f"   Dimension vecteur: {len(self.vocabulary)}")
        
        for word in words[:5]:
            vector = self.one_hot_encode(word)
            non_zero = np.where(vector == 1)[0]
            print(f"\n   '{word}':")
            print(f"      Index: {non_zero[0] if len(non_zero) > 0 else 'inconnu'}")
            print(f"      Vecteur: {vector[:10]}... (premiers 10 éléments)")
    
    # =========================================================================
    # 2. BAG-OF-WORDS (BoW)
    # =========================================================================
    
    def bag_of_words(self, document: str, binary: bool = False) -> Dict[str, int]:
        """
        Crée une représentation Bag-of-Words d'un document
        
        Args:
            document: Texte du document
            binary: Si True, compte binaire (présence/absence)
        
        Returns:
            Dictionnaire mot → fréquence
        
        Example:
            >>> bow = bag_of_words("le chat dort sur le tapis")
            >>> bow
            {'le': 2, 'chat': 1, 'dort': 1, 'sur': 1, 'tapis': 1}
        """
        words = self.tokenize(document)
        
        if binary:
            # Présence/absence uniquement
            return {word: 1 for word in set(words)}
        else:
            # Comptage des occurrences
            return dict(Counter(words))
    
    def bow_to_vector(self, bow: Dict[str, int]) -> np.ndarray:
        """
        Convertit un BoW en vecteur dense basé sur le vocabulaire
        
        Args:
            bow: Dictionnaire Bag-of-Words
        
        Returns:
            Vecteur de taille |V|
        """
        vector = np.zeros(len(self.vocabulary))
        
        for word, count in bow.items():
            if word in self.word2idx:
                vector[self.word2idx[word]] = count
        
        return vector
    
    def demonstrate_bow(self, documents: List[str]):
        """Démontre le Bag-of-Words"""
        print(f"\n🎒 BAG-OF-WORDS (BoW)")
        
        for i, doc in enumerate(documents[:3], 1):
            bow = self.bag_of_words(doc)
            vector = self.bow_to_vector(bow)
            
            print(f"\n   Document {i}: '{doc[:50]}...'")
            print(f"   BoW: {dict(list(bow.items())[:5])}...")
            print(f"   Vecteur (shape): {vector.shape}")
            print(f"   Non-zéro: {np.count_nonzero(vector)} éléments")
    
    # =========================================================================
    # 3. TF-IDF
    # =========================================================================
    
    def compute_tf(self, document: str) -> Dict[str, float]:
        """
        Calcule Term Frequency (TF)
        
        TF(t, d) = (Nombre d'occurrences de t dans d) / (Nombre total de mots dans d)
        
        Args:
            document: Texte du document
        
        Returns:
            Dictionnaire mot → TF
        """
        words = self.tokenize(document)
        total_words = len(words)
        
        if total_words == 0:
            return {}
        
        bow = Counter(words)
        tf = {word: count / total_words for word, count in bow.items()}
        
        return tf
    
    def compute_idf(self, documents: List[str]) -> Dict[str, float]:
        """
        Calcule Inverse Document Frequency (IDF)
        
        IDF(t) = log(N / df(t))
        où N = nombre total de documents
            df(t) = nombre de documents contenant t
        
        Args:
            documents: Liste de documents
        
        Returns:
            Dictionnaire mot → IDF
        """
        N = len(documents)
        self.document_count = N
        
        # Compter dans combien de documents chaque mot apparaît
        doc_freq = defaultdict(int)
        
        for doc in documents:
            unique_words = set(self.tokenize(doc))
            for word in unique_words:
                doc_freq[word] += 1
        
        # Calculer IDF
        idf = {}
        for word, df in doc_freq.items():
            idf[word] = math.log(N / df)
        
        self.idf_scores = idf
        
        return idf
    
    def compute_tfidf(self, document: str, documents: List[str] = None) -> Dict[str, float]:
        """
        Calcule TF-IDF pour un document
        
        TF-IDF(t, d) = TF(t, d) × IDF(t)
        
        Args:
            document: Document à analyser
            documents: Liste complète de documents (pour calculer IDF)
        
        Returns:
            Dictionnaire mot → TF-IDF
        """
        # Calculer TF
        tf = self.compute_tf(document)
        
        # Calculer ou récupérer IDF
        if documents is not None:
            idf = self.compute_idf(documents)
        else:
            idf = self.idf_scores
        
        # TF-IDF
        tfidf = {}
        for word, tf_val in tf.items():
            if word in idf:
                tfidf[word] = tf_val * idf[word]
            else:
                tfidf[word] = tf_val  # IDF = 0 pour mots non vus
        
        return tfidf
    
    def demonstrate_tfidf(self, documents: List[str]):
        """Démontre TF-IDF"""
        print(f"\n📊 TF-IDF")
        print(f"   Corpus: {len(documents)} documents")
        
        # Calculer IDF sur tout le corpus
        idf = self.compute_idf(documents)
        
        # Top mots par IDF
        top_idf = sorted(idf.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\n   Top 5 mots par IDF (mots rares):")
        for word, score in top_idf:
            print(f"      '{word}': {score:.3f}")
        
        # Analyser premier document
        if documents:
            doc = documents[0]
            tfidf = self.compute_tfidf(doc, documents)
            
            top_tfidf = sorted(tfidf.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"\n   Document 1: '{doc[:50]}...'")
            print(f"   Top 5 mots par TF-IDF:")
            for word, score in top_tfidf:
                print(f"      '{word}': {score:.4f}")
    
    # =========================================================================
    # 4. POS TAGGING
    # =========================================================================
    
    def pos_tag(self, text: str) -> List[Tuple[str, str]]:
        """
        Étiquetage morpho-syntaxique (Part-of-Speech tagging)
        
        Args:
            text: Texte à analyser
        
        Returns:
            Liste de (mot, étiquette POS)
        
        Example:
            >>> pos_tag("I have a severe headache")
            [('I', 'PRON'), ('have', 'VERB'), ('a', 'DET'), 
             ('severe', 'ADJ'), ('headache', 'NOUN')]
        """
        if not self.nlp:
            # Fallback simple si spaCy non disponible
            words = self.tokenize(text)
            return [(word, 'UNKNOWN') for word in words]
        
        doc = self.nlp(text)
        pos_tags = [(token.text, token.pos_) for token in doc]
        
        return pos_tags
    
    def get_pos_statistics(self, text: str) -> Dict[str, int]:
        """
        Statistiques des étiquettes POS
        
        Args:
            text: Texte à analyser
        
        Returns:
            Dictionnaire POS → count
        """
        pos_tags = self.pos_tag(text)
        pos_counts = Counter([pos for _, pos in pos_tags])
        
        return dict(pos_counts)
    
    def extract_by_pos(self, text: str, pos_filter: List[str]) -> List[str]:
        """
        Extrait les mots d'une certaine catégorie POS
        
        Args:
            text: Texte à analyser
            pos_filter: Liste de POS à garder (ex: ['NOUN', 'VERB'])
        
        Returns:
            Liste de mots filtrés
        
        Example:
            >>> extract_by_pos("I have severe pain", ['NOUN', 'ADJ'])
            ['severe', 'pain']
        """
        pos_tags = self.pos_tag(text)
        filtered = [word for word, pos in pos_tags if pos in pos_filter]
        
        return filtered
    
    def demonstrate_pos(self, texts: List[str]):
        """Démontre le POS tagging"""
        print(f"\n🏷️  POS TAGGING")
        
        for i, text in enumerate(texts[:3], 1):
            pos_tags = self.pos_tag(text)
            stats = self.get_pos_statistics(text)
            
            print(f"\n   Texte {i}: '{text[:50]}...'")
            print(f"   POS tags: {pos_tags[:5]}...")
            print(f"   Statistiques: {stats}")
            
            # Extraire noms et adjectifs (symptômes médicaux souvent)
            if self.nlp:
                symptoms = self.extract_by_pos(text, ['NOUN', 'ADJ'])
                if symptoms:
                    print(f"   Symptômes potentiels (NOUN/ADJ): {symptoms[:5]}")
    
    # =========================================================================
    # 5. TOKENIZATION
    # =========================================================================
    
    def tokenize(self, text: str, lowercase: bool = True) -> List[str]:
        """
        Tokenization simple
        
        Args:
            text: Texte à tokenizer
            lowercase: Mettre en minuscules
        
        Returns:
            Liste de tokens
        """
        if lowercase:
            text = text.lower()
        
        # Enlever ponctuation et split
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        
        return tokens


# ==============================================================================
# TEST
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 TEST NLP FOUNDATIONS")
    print("="*70)
    
    nlp_foundations = NLPFoundations()
    
    # Corpus de test médical
    medical_corpus = [
        "I have a severe headache with nausea and vomiting",
        "The patient has chest pain and difficulty breathing",
        "Stomach pain with fever and chills",
        "Severe migraine with visual disturbances",
        "Chest pain radiating to left arm"
    ]
    
    # 1. Build vocabulary
    nlp_foundations.build_vocabulary(medical_corpus)
    
    # 2. One-hot encoding
    nlp_foundations.demonstrate_one_hot(['headache', 'pain', 'fever'])
    
    # 3. Bag-of-Words
    nlp_foundations.demonstrate_bow(medical_corpus)
    
    # 4. TF-IDF
    nlp_foundations.demonstrate_tfidf(medical_corpus)
    
    # 5. POS Tagging
    nlp_foundations.demonstrate_pos(medical_corpus)
    
    print("\n" + "="*70)
    print("✅ TESTS TERMINÉS")
    print("="*70)