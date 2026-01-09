import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

try:
    from spellchecker import SpellChecker
    HAS_PYSPELLCHECKER = True
except ImportError:
    HAS_PYSPELLCHECKER = False

class ContextSpellCorrector:
    """
    Correcteur orthographique Hybride (True NLP).
    1. Bibliothéque Standard (pyspellchecker) pour vocabulaire large.
    2. N-Grams (Bigrams) pour le contexte médical spécifique.
    """
    
    def __init__(self):
        self.vocab = Counter()
        self.bigrams = Counter()
        self.total_words = 0
        
        # Initialisation correcteur standard
        self.spell_checkers = {}
        if HAS_PYSPELLCHECKER:
            try:
                self.spell_checkers['en'] = SpellChecker(language='en')
                self.spell_checkers['en'].word_frequency.load_words(['arrhythmia', 'tachycardia', 'dyspnea'])
                
                self.spell_checkers['fr'] = SpellChecker(language='fr')
                self.spell_checkers['fr'].word_frequency.load_words(['coeur', 'tête', 'ventre'])
            except Exception as e:
                print(f"⚠️ Erreur init corrections: {e}")
        self.total_words = 0
        
        # Mots médicaux critiques (Priorité absolue)
        self.medical_priority = {
            'headache', 'stomachache', 'toothache', 'backache',
            'pain', 'ache', 'hurt', 'sore', 'fever',
            'heart', 'chest', 'stomach', 'belly', 'head', 'tooth', 'teeth',
            'throat', 'lung', 'arm', 'leg', 'foot', 'eye', 'ear',
            'nausea', 'vomiting', 'dizziness', 'bleeding'
        }
        
        # Mots à NE PAS CORRIGER (Stopwords anglais courants)
        self.protected_words = {
            'i', 'a', 'an', 'am', 'is', 'are', 'was', 'were',
            'in', 'on', 'at', 'to', 'for', 'of', 'with', 'from',
            'the', 'this', 'that', 'these', 'those',
            'my', 'your', 'his', 'her', 'its', 'our', 'their',
            'and', 'but', 'or', 'so', 'if', 'when', 'where',
            'have', 'has', 'had', 'do', 'does', 'did',
            'he', 'she', 'it', 'we', 'they', 'you', 'me',
        }

    def train(self, texts: List[str]):
        """Apprend les probabilités des mots et des contextes depuis le dataset"""
        print("   📚 Entraînement du correcteur contextuel (N-Grams)...")
        
        for text in texts:
            # Tokenization simple
            words = self._tokenize(text)
            self.vocab.update(words)
            self.total_words += len(words)
            
            # Construction des Bigrams (Mot précédent -> Mot actuel)
            for i in range(len(words) - 1):
                bigram = (words[i], words[i+1])
                self.bigrams[bigram] += 1
                
        print(f"      ✅ Vocabulaire: {len(self.vocab)} mots")
        print(f"      ✅ Contextes appris: {len(self.bigrams)} bigrams")

    def correct_text(self, text: str, lang: str = 'en') -> Tuple[str, List[Dict]]:
        """Corrige le texte en utilisant le contexte + Pyspellchecker (Multi-langue)"""
        words = self._tokenize(text)
        corrected_words = []
        corrections = []
        
        # Sélectionner le bon correcteur
        checker = self.spell_checkers.get(lang, self.spell_checkers.get('en'))
        
        for i, word in enumerate(words):
            # 1. Si le mot est protégé, on ne touche pas
            if word in self.protected_words:
                 corrected_words.append(word)
                 continue
                 
            # 2. Si le mot est connu (Médical ou Standard), on le garde
            is_known_med = (self.vocab[word] > 5)
            is_known_std = False
            
            if checker:
                is_known_std = (word in checker)
            
            if is_known_med or is_known_std:
                 corrected_words.append(word)
                 continue
            
            # 3. Mot inconnu -> Correction
            prev_word = corrected_words[-1] if i > 0 else None
            
            candidates = []
            if checker:
                cands = checker.candidates(word)
                if cands:
                    candidates = list(cands)
            
            # Fallback si pyspellchecker échoue ou n'est pas là
            if not candidates:
                candidates = self._get_candidates(word)
            
            best_candidate = self._choose_best_candidate(candidates, prev_word)
            
            if best_candidate != word:
                corrections.append({'original': word, 'corrected': best_candidate, 'type': 'spelling'})
            
            corrected_words.append(best_candidate)
            
        return ' '.join(corrected_words), corrections

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())

    def _get_candidates(self, word: str) -> List[str]:
        """Génère des candidats à distance d'édition 1 ou 2"""
        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        
        # Distance 1
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        
        dist1 = set(deletes + transposes + replaces + inserts)
        
        # Filtrer par vocabulaire connu pour réduire l'espace
        known_dist1 = [w for w in dist1 if w in self.vocab or w in self.medical_priority]
        if known_dist1:
            return known_dist1
            
        return [word] # Fallback

    def _choose_best_candidate(self, candidates: List[str], prev_word: str) -> str:
        """Choisit le meilleur candidat basé sur la probabilité Unigram + Bigram"""
        if not candidates:
            return ""
            
        best_word = candidates[0]
        max_score = -1
        
        for cand in candidates:
            # Score Unigram (Fréquence globale)
            unigram_score = self.vocab[cand] / self.total_words if self.total_words > 0 else 0
            
            # Score Bigram (Contexte)
            bigram_score = 0
            if prev_word:
                bigram_count = self.bigrams[(prev_word, cand)]
                prev_count = self.vocab[prev_word]
                if prev_count > 0:
                    bigram_score = bigram_count / prev_count
            
            # Score total (Poids fort sur le contexte et les mots médicaux)
            medical_bonus = 100 if cand in self.medical_priority else 1
            
            # Formule: Probabilité combinée * Bonus médical
            total_score = (unigram_score + (bigram_score * 50)) * medical_bonus
            
            if total_score > max_score:
                max_score = total_score
                best_word = cand
                
        return best_word

    def _check_contextual_typo(self, word: str, prev_word: str) -> str:
        """Détecte si un mot valide est improbable dans ce contexte (ex: 'hear' après 'my')"""
        if not prev_word:
            return None
            
        # Si le bigram actuel n'a jamais été vu
        if self.bigrams[(prev_word, word)] == 0:
            # On regarde si une variante proche (dist 1) ferait sens ici
            candidates = self._get_candidates(word)
            for cand in candidates:
                if cand == word: continue
                # Si le candidat est médical ou a un fort bigram
                if self.bigrams[(prev_word, cand)] > 0 or cand in self.medical_priority:
                     # On vérifie si le candidat est nettement plus probable
                     if self.vocab[cand] > 0: # Sanity check
                         return cand
        return None
