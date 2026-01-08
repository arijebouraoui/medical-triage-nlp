"""
Spell Corrector FINAL - VERSION PARFAITE
=========================================
Corrections intelligentes FR/EN avec priorité médicale
"""

from typing import List, Tuple, Dict
import re

try:
    import enchant
    ENCHANT_AVAILABLE = True
except ImportError:
    ENCHANT_AVAILABLE = False


class SpellCorrector:
    """Correcteur orthographique médical intelligent"""
    
    def __init__(self, medical_vocab: List[str]):
        """Initialise le correcteur"""
        
        print("   📚 Correcteur orthographique...")
        
        # Vocabulaire médical du dataset
        self.medical_vocab = set(word.lower() for word in medical_vocab)
        
        # Mots médicaux ABSOLUMENT PRIORITAIRES
        self.medical_priority = {
            'pain', 'ache', 'hurt', 'sore',
            'fever', 'cough', 'nausea', 'vomiting',
            'bleeding', 'swelling', 'redness', 'itching',
            'dizziness', 'fatigue', 'weakness',
            'headache', 'toothache', 'stomachache',
            'chest', 'heart', 'lung', 'stomach', 'abdomen',
            'head', 'eye', 'eyes', 'ear', 'nose', 'throat',
            'tooth', 'teeth', 'gum', 'gums',
            'arm', 'leg', 'hand', 'foot', 'knee',
            'back', 'neck', 'shoulder',
        }
        
        # Dictionnaires
        self.dictionaries = {}
        
        if ENCHANT_AVAILABLE:
            try:
                self.dictionaries['en'] = enchant.Dict("en_US")
                print(f"      ✅ Dictionnaire EN")
            except:
                pass
            
            try:
                self.dictionaries['fr'] = enchant.Dict("fr_FR")
                print(f"      ✅ Dictionnaire FR")
            except:
                pass
        
        # Mots communs (ne PAS corriger)
        self.common_words = {
            # Anglais de base
            'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'have', 'has', 'had', 'am', 'is', 'are', 'was', 'were',
            'my', 'your', 'his', 'her', 'its', 'our', 'their',
            'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'from',
            'and', 'but', 'or', 'so', 'if', 'when', 'where', 'why', 'how',
            
            # Français de base
            'je', 'tu', 'il', 'elle', 'nous', 'vous', 'ils', 'elles',
            'ai', 'as', 'a', 'avons', 'avez', 'ont',
            'suis', 'es', 'est', 'sommes', 'êtes', 'sont',
            'mon', 'ma', 'mes', 'ton', 'ta', 'tes', 'son', 'sa', 'ses',
            'le', 'la', 'les', 'un', 'une', 'des',
            'au', 'aux', 'du', 'de', 'dans', 'sur', 'avec', 'pour', 'par',
            'et', 'ou', 'mais', 'donc', 'car',
            
            # Mots médicaux FR
            'mal', 'douleur', 'ventre', 'tête', 'coeur',
        }
    
    def correct_text(self, text: str, language: str = None) -> Tuple[str, List[Dict]]:
        """Corrige le texte"""
        
        if language is None:
            language = self._detect_language(text)
        
        words = text.split()
        corrected_words = []
        corrections = []
        
        for word in words:
            # Nettoyer (garder apostrophes)
            clean_word = word.lower()
            
            # Gérer apostrophes françaises (j'ai, d'un, etc.)
            if "'" in clean_word:
                parts = clean_word.split("'")
                if len(parts) == 2:
                    # Ne PAS corriger les mots avec apostrophes
                    corrected_words.append(word)
                    continue
            
            # Enlever ponctuation pour test
            test_word = re.sub(r'[^\w\']', '', clean_word)
            
            if not test_word or len(test_word) <= 2:
                corrected_words.append(word)
                continue
            
            # VÉRIFICATION 1: Mot médical prioritaire proche?
            medical_correction = self._check_medical_priority(test_word)
            if medical_correction:
                corrected_words.append(medical_correction)
                if medical_correction != test_word:
                    corrections.append({
                        'original': word,
                        'corrected': medical_correction,
                        'type': 'medical_priority'
                    })
                continue
            
            # VÉRIFICATION 2: Mot commun? (ne PAS corriger)
            if test_word in self.common_words:
                corrected_words.append(word)
                continue
            
            # VÉRIFICATION 3: Mot médical du dataset?
            if test_word in self.medical_vocab:
                corrected_words.append(word)
                continue
            
            # VÉRIFICATION 4: Correct dans dictionnaire?
            if self._is_correct(test_word, language):
                corrected_words.append(word)
                continue
            
            # VÉRIFICATION 5: Chercher correction
            corrected = self._find_correction(test_word, language)
            
            if corrected and corrected != test_word:
                corrected_words.append(corrected)
                corrections.append({
                    'original': word,
                    'corrected': corrected,
                    'type': 'standard'
                })
            else:
                corrected_words.append(word)
        
        corrected_text = ' '.join(corrected_words)
        
        return corrected_text, corrections
    
    def _detect_language(self, text: str) -> str:
        """Détecte la langue"""
        
        text_lower = text.lower()
        words = text_lower.split()
        
        # Marqueurs français FORTS
        fr_strong = ["j'ai", "je", "suis", "mon", "ma", "mes", "au", "aux", "du"]
        en_strong = ["i", "have", "my", "the", "is", "are"]
        
        fr_score = sum(2 for marker in fr_strong if marker in words or marker in text_lower)
        en_score = sum(1 for marker in en_strong if marker in words)
        
        # Accents français
        if re.search(r'[àâäéèêëïîôùûüÿç]', text_lower):
            fr_score += 5
        
        # Apostrophes françaises
        if "'" in text and any(x in text_lower for x in ["j'", "d'", "l'", "m'", "t'", "s'"]):
            fr_score += 3
        
        return 'fr' if fr_score > en_score else 'en'
    
    def _check_medical_priority(self, word: str) -> str:
        """Vérifie proximité avec mots médicaux prioritaires"""
        
        if word in self.medical_priority:
            return word
        
        # Chercher distance = 1 uniquement
        for med_word in self.medical_priority:
            if self._levenshtein_distance(word, med_word) == 1:
                return med_word
        
        return None
    
    def _is_correct(self, word: str, language: str) -> bool:
        """Vérifie si mot correct"""
        
        # Dictionnaire de la langue
        if language in self.dictionaries:
            if self.dictionaries[language].check(word):
                return True
        
        # Essayer l'autre langue (textes mixtes)
        other_lang = 'fr' if language == 'en' else 'en'
        if other_lang in self.dictionaries:
            if self.dictionaries[other_lang].check(word):
                return True
        
        return False
    
    def _find_correction(self, word: str, language: str) -> str:
        """Trouve la meilleure correction"""
        
        # Si le mot est correct dans SA langue, ne PAS corriger
        if language in self.dictionaries:
            if self.dictionaries[language].check(word):
                return word  # Mot correct dans sa langue!
        
        candidates = []
        
        # 1. Mots communs (priorité absolue)
        for common in self.common_words:
            dist = self._levenshtein_distance(word, common)
            if dist <= 2:
                candidates.append((common, dist, 0))
        
        # 2. Mots médicaux prioritaires
        for med_word in self.medical_priority:
            dist = self._levenshtein_distance(word, med_word)
            if dist == 1:
                candidates.append((med_word, dist, 1))
        
        # 3. Vocabulaire médical
        for med_word in self.medical_vocab:
            dist = self._levenshtein_distance(word, med_word)
            if dist <= 2:
                candidates.append((med_word, dist, 2))
        
        # 4. Dictionnaire principal
        if language in self.dictionaries:
            try:
                suggestions = self.dictionaries[language].suggest(word)
                for sug in suggestions[:3]:
                    dist = self._levenshtein_distance(word, sug.lower())
                    if dist <= 1:
                        candidates.append((sug.lower(), dist, 3))
            except:
                pass
        
        if not candidates:
            return word
        
        # Trier: distance, puis priorité
        candidates.sort(key=lambda x: (x[1], x[2]))
        
        return candidates[0][0]
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Distance de Levenshtein"""
        
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]