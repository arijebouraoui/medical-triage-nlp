"""
Medical Word2Vec Module
========================
Implémente Word2Vec (CBOW et Skip-gram) sur dataset médical:
- Entraînement CBOW
- Entraînement Skip-gram
- Similarité cosinus
- Analogies médicales
- Hypothèse distributionnelle
"""

import os
import sys
import json
from typing import List, Dict, Tuple
import numpy as np

# Ajouter le chemin du projet
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from gensim.models import Word2Vec
    from gensim.models.callbacks import CallbackAny2Vec
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    print("⚠️  gensim non disponible. Installez avec: pip install gensim")


class MedicalWord2Vec:
    """Entraîne Word2Vec sur données médicales"""
    
    def __init__(self, data_path: str = "data/processed/dataset_processed.json"):
        """
        Initialise le module Word2Vec médical
        
        Args:
            data_path: Chemin vers dataset médical
        """
        print("\n🧬 Initialisation Medical Word2Vec...")
        
        self.data_path = data_path
        self.corpus = []
        self.cbow_model = None
        self.skipgram_model = None
        
        # Charger et préparer le corpus
        self._load_medical_corpus()
        
        print(f"   ✅ Corpus chargé: {len(self.corpus)} phrases")
    
    def _load_medical_corpus(self):
        """Charge le corpus médical depuis le dataset"""
        print(f"   📖 Chargement corpus depuis {self.data_path}...")
        
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extraire les textes patients
            for case in data:
                if 'patient_text' in case:
                    # Tokenizer le texte
                    text = case['patient_text'].lower()
                    tokens = text.split()
                    
                    if len(tokens) > 2:  # Phrases avec au moins 3 mots
                        self.corpus.append(tokens)
            
            print(f"      ✅ {len(self.corpus)} phrases extraites")
            
            # Statistiques
            total_words = sum(len(sentence) for sentence in self.corpus)
            unique_words = len(set(word for sentence in self.corpus for word in sentence))
            
            print(f"      📊 Mots totaux: {total_words}")
            print(f"      📊 Vocabulaire: {unique_words} mots uniques")
            
        except FileNotFoundError:
            print(f"      ⚠️  Fichier non trouvé: {self.data_path}")
            print(f"      📝 Utilisation corpus de démonstration...")
            
            # Corpus de démo médical
            demo_corpus = [
                "severe headache with nausea and vomiting",
                "chest pain radiating to left arm",
                "stomach pain with fever and chills",
                "difficulty breathing and chest tightness",
                "severe migraine with visual disturbances",
                "abdominal pain with bloating and gas",
                "sharp chest pain when breathing",
                "chronic headache lasting several days",
                "nausea vomiting and diarrhea",
                "pain in lower back radiating to leg"
            ]
            
            self.corpus = [sentence.split() for sentence in demo_corpus]
    
    # =========================================================================
    # CBOW (Continuous Bag-of-Words)
    # =========================================================================
    
    def train_cbow(self, 
                   vector_size: int = 100,
                   window: int = 5,
                   min_count: int = 1,
                   epochs: int = 10,
                   workers: int = 4) -> Word2Vec:
        """
        Entraîne un modèle CBOW
        
        CBOW prédit le mot central à partir du contexte
        Context: [w_{t-k}, ..., w_{t-1}, w_{t+1}, ..., w_{t+k}] → Target: w_t
        
        Args:
            vector_size: Dimension des vecteurs
            window: Taille de fenêtre contextuelle
            min_count: Fréquence minimale d'un mot
            epochs: Nombre d'époques
            workers: Threads parallèles
        
        Returns:
            Modèle Word2Vec CBOW entraîné
        """
        if not GENSIM_AVAILABLE:
            print("❌ gensim non disponible")
            return None
        
        print(f"\n🎯 Entraînement CBOW:")
        print(f"   • Vector size: {vector_size}")
        print(f"   • Window: {window}")
        print(f"   • Min count: {min_count}")
        print(f"   • Epochs: {epochs}")
        
        # sg=0 pour CBOW (sg=1 pour Skip-gram)
        self.cbow_model = Word2Vec(
            sentences=self.corpus,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            sg=0,  # CBOW
            epochs=epochs
        )
        
        vocab_size = len(self.cbow_model.wv)
        print(f"   ✅ CBOW entraîné ({vocab_size} mots dans vocabulaire)")
        
        return self.cbow_model
    
    # =========================================================================
    # SKIP-GRAM
    # =========================================================================
    
    def train_skipgram(self,
                       vector_size: int = 100,
                       window: int = 5,
                       min_count: int = 1,
                       epochs: int = 10,
                       workers: int = 4) -> Word2Vec:
        """
        Entraîne un modèle Skip-gram
        
        Skip-gram prédit le contexte à partir du mot central
        Target: w_t → Context: [w_{t-k}, ..., w_{t-1}, w_{t+1}, ..., w_{t+k}]
        
        Args:
            vector_size: Dimension des vecteurs
            window: Taille de fenêtre contextuelle
            min_count: Fréquence minimale d'un mot
            epochs: Nombre d'époques
            workers: Threads parallèles
        
        Returns:
            Modèle Word2Vec Skip-gram entraîné
        """
        if not GENSIM_AVAILABLE:
            print("❌ gensim non disponible")
            return None
        
        print(f"\n🎯 Entraînement Skip-gram:")
        print(f"   • Vector size: {vector_size}")
        print(f"   • Window: {window}")
        print(f"   • Min count: {min_count}")
        print(f"   • Epochs: {epochs}")
        
        # sg=1 pour Skip-gram
        self.skipgram_model = Word2Vec(
            sentences=self.corpus,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            sg=1,  # Skip-gram
            epochs=epochs
        )
        
        vocab_size = len(self.skipgram_model.wv)
        print(f"   ✅ Skip-gram entraîné ({vocab_size} mots dans vocabulaire)")
        
        return self.skipgram_model
    
    # =========================================================================
    # SIMILARITÉ COSINUS
    # =========================================================================
    
    def get_similar_words(self, word: str, model_type: str = 'cbow', topn: int = 5) -> List[Tuple[str, float]]:
        """
        Trouve les mots les plus similaires
        
        Utilise similarité cosinus: cos(θ) = (A · B) / (||A|| ||B||)
        
        Args:
            word: Mot de référence
            model_type: 'cbow' ou 'skipgram'
            topn: Nombre de résultats
        
        Returns:
            Liste de (mot, similarité)
        
        Example:
            >>> get_similar_words('headache', 'cbow', 3)
            [('migraine', 0.85), ('pain', 0.78), ('severe', 0.65)]
        """
        model = self.cbow_model if model_type == 'cbow' else self.skipgram_model
        
        if not model:
            print(f"❌ Modèle {model_type} non entraîné")
            return []
        
        if word not in model.wv:
            print(f"⚠️  Mot '{word}' non dans vocabulaire")
            return []
        
        similar = model.wv.most_similar(word, topn=topn)
        
        return similar
    
    def compare_similarity(self, word1: str, word2: str, model_type: str = 'cbow') -> float:
        """
        Calcule la similarité entre deux mots
        
        Args:
            word1: Premier mot
            word2: Deuxième mot
            model_type: 'cbow' ou 'skipgram'
        
        Returns:
            Score de similarité (0-1)
        """
        model = self.cbow_model if model_type == 'cbow' else self.skipgram_model
        
        if not model:
            return 0.0
        
        if word1 not in model.wv or word2 not in model.wv:
            return 0.0
        
        similarity = model.wv.similarity(word1, word2)
        
        return similarity
    
    def demonstrate_similarity(self, words: List[str], model_type: str = 'cbow'):
        """Démontre la similarité cosinus"""
        print(f"\n📐 SIMILARITÉ COSINUS ({model_type.upper()}):")
        
        for word in words:
            similar = self.get_similar_words(word, model_type, topn=5)
            
            if similar:
                print(f"\n   Mots similaires à '{word}':")
                for sim_word, score in similar:
                    print(f"      • {sim_word}: {score:.3f}")
    
    # =========================================================================
    # ANALOGIES
    # =========================================================================
    
    def solve_analogy(self, 
                      positive: List[str], 
                      negative: List[str],
                      model_type: str = 'cbow',
                      topn: int = 1) -> List[Tuple[str, float]]:
        """
        Résout une analogie: A - B + C = ?
        
        Exemple médical:
            headache - head + chest = chest pain
            fever - high + low = hypothermia
        
        Args:
            positive: Mots positifs [C, ...]
            negative: Mots négatifs [A, B, ...]
            model_type: 'cbow' ou 'skipgram'
            topn: Nombre de résultats
        
        Returns:
            Liste de (mot, score)
        
        Example:
            >>> solve_analogy(['chest'], ['headache', 'head'])
            [('pain', 0.75)]
        """
        model = self.cbow_model if model_type == 'cbow' else self.skipgram_model
        
        if not model:
            print(f"❌ Modèle {model_type} non entraîné")
            return []
        
        # Vérifier que tous les mots sont dans le vocabulaire
        all_words = positive + negative
        for word in all_words:
            if word not in model.wv:
                print(f"⚠️  Mot '{word}' non dans vocabulaire")
                return []
        
        try:
            result = model.wv.most_similar(
                positive=positive,
                negative=negative,
                topn=topn
            )
            return result
        except Exception as e:
            print(f"❌ Erreur analogie: {e}")
            return []
    
    def demonstrate_analogies(self, model_type: str = 'cbow'):
        """Démontre les analogies médicales"""
        print(f"\n🔄 ANALOGIES MÉDICALES ({model_type.upper()}):")
        
        # Exemples d'analogies médicales
        analogies = [
            ("headache - head + chest", ['chest'], ['headache', 'head']),
            ("pain + severe", ['pain', 'severe'], []),
            ("nausea + vomiting", ['nausea', 'vomiting'], []),
        ]
        
        for description, positive, negative in analogies:
            result = self.solve_analogy(positive, negative, model_type, topn=3)
            
            if result:
                print(f"\n   {description}:")
                for word, score in result:
                    print(f"      → {word} ({score:.3f})")
    
    # =========================================================================
    # COMPARAISON CBOW vs SKIP-GRAM
    # =========================================================================
    
    def compare_models(self, test_words: List[str]):
        """Compare CBOW vs Skip-gram"""
        print(f"\n⚖️  COMPARAISON CBOW vs SKIP-GRAM:")
        
        for word in test_words:
            print(f"\n   Mot: '{word}'")
            
            # CBOW
            cbow_similar = self.get_similar_words(word, 'cbow', topn=3)
            print(f"      CBOW: {[w for w, s in cbow_similar]}")
            
            # Skip-gram
            sg_similar = self.get_similar_words(word, 'skipgram', topn=3)
            print(f"      Skip-gram: {[w for w, s in sg_similar]}")


# ==============================================================================
# TEST
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 TEST MEDICAL WORD2VEC")
    print("="*70)
    
    w2v = MedicalWord2Vec()
    
    # Entraîner CBOW
    w2v.train_cbow(vector_size=50, window=3, epochs=20)
    
    # Entraîner Skip-gram
    w2v.train_skipgram(vector_size=50, window=3, epochs=20)
    
    # Test mots
    test_words = ['pain', 'headache', 'chest', 'severe']
    
    # Similarités
    w2v.demonstrate_similarity(test_words, 'cbow')
    w2v.demonstrate_similarity(test_words, 'skipgram')
    
    # Analogies
    w2v.demonstrate_analogies('cbow')
    w2v.demonstrate_analogies('skipgram')
    
    # Comparaison
    w2v.compare_models(['pain', 'chest'])
    
    print("\n" + "="*70)
    print("✅ TESTS TERMINÉS")
    print("="*70)