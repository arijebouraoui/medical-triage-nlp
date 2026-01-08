"""
Multilingual Processor - ULTRA VERSION
=======================================
Améliorations:
- Détection langue mixte (EN + FR mots)
- Traduction mot-par-mot même si langue détectée = EN
- Dictionnaire médical étendu
"""

from typing import Dict, List, Tuple
from enum import Enum
import re


class Language(Enum):
    """Langues supportées"""
    FRENCH = 'fr'
    ENGLISH = 'en'
    ARABIC = 'ar'
    SPANISH = 'es'
    UNKNOWN = 'unknown'


class MultilingualProcessor:
    """Processeur multilingue avec support mixte"""
    
    def __init__(self):
        """Initialise le processeur avec dictionnaires étendus"""
        
        # Marqueurs de langue
        self.language_markers = {
            Language.FRENCH: {
                'words': ['je', 'suis', 'mal', 'douleur', 'tête', 'fièvre', 
                         'estomac', 'nausée', 'vomissement', 'aussi', 'très',
                         'depuis', 'jours', 'coeur', 'ventre', 'gorge', 'dos',
                         'mon', 'ma', 'mes', 'le', 'la', 'les', 'un', 'une'],
                'patterns': ['j\'ai', 'je me sens', 'ça fait mal', 'j\'ai mal']
            },
            Language.ENGLISH: {
                'words': ['pain', 'headache', 'stomach', 'nausea', 'vomiting',
                         'fever', 'cough', 'dizzy', 'weak', 'have', 'feel',
                         'my', 'the', 'a', 'an', 'is', 'am', 'are'],
                'patterns': ['i have', 'i feel', 'my head', 'my stomach', 'my heart']
            },
            Language.ARABIC: {
                'words': ['صداع', 'حمى', 'ألم', 'غثيان', 'قيء', 'معدة', 'رأس'],
                'patterns': ['عندي', 'أشعر', 'يؤلمني']
            },
            Language.SPANISH: {
                'words': ['dolor', 'cabeza', 'estómago', 'náusea', 'vómito',
                         'fiebre', 'tos', 'mareo', 'débil', 'tengo', 'siento'],
                'patterns': ['tengo', 'me duele', 'siento']
            }
        }
        
        # Dictionnaire de traduction FR → EN (TRÈS ÉTENDU)
        self.fr_to_en = {
            # Symptômes de base
            'mal': 'pain',
            'douleur': 'pain',
            'douleurs': 'pain',
            'tête': 'head',
            'mal de tête': 'headache',
            'maux de tête': 'headache',
            'douleur de tête': 'headache',
            'migraine': 'migraine',
            'migraines': 'migraine',
            
            # Corps (COMPLET)
            'coeur': 'heart',
            'cœur': 'heart',
            'estomac': 'stomach',
            'ventre': 'stomach',
            'abdomen': 'abdomen',
            'poitrine': 'chest',
            'thorax': 'chest',
            'dos': 'back',
            'gorge': 'throat',
            'genou': 'knee',
            'genoux': 'knee',
            'bras': 'arm',
            'jambe': 'leg',
            'jambes': 'legs',
            'pied': 'foot',
            'pieds': 'feet',
            'main': 'hand',
            'mains': 'hands',
            'yeux': 'eyes',
            'œil': 'eye',
            'oreille': 'ear',
            'oreilles': 'ears',
            'nez': 'nose',
            'bouche': 'mouth',
            'dents': 'teeth',
            'dent': 'tooth',
            'gencives': 'gums',
            'langue': 'tongue',
            
            # Symptômes
            'fièvre': 'fever',
            'température': 'fever',
            'nausée': 'nausea',
            'nausées': 'nausea',
            'vomissement': 'vomiting',
            'vomissements': 'vomiting',
            'vomir': 'vomiting',
            'toux': 'cough',
            'vertige': 'dizziness',
            'vertiges': 'dizziness',
            'étourdissement': 'dizziness',
            'étourdissements': 'dizziness',
            'faiblesse': 'weakness',
            'fatigue': 'fatigue',
            'saignement': 'bleeding',
            'saignements': 'bleeding',
            'sang': 'blood',
            'brûlure': 'burning',
            'brûlures': 'burning',
            'démangeaison': 'itching',
            'démangeaisons': 'itching',
            'gonflement': 'swelling',
            'enflure': 'swelling',
            
            # Intensité
            'très': 'very',
            'fort': 'strong',
            'forte': 'strong',
            'intense': 'intense',
            'sévère': 'severe',
            'léger': 'mild',
            'légère': 'mild',
            'aigu': 'acute',
            'aiguë': 'acute',
            'chronique': 'chronic',
            
            # Localisations
            'au': 'in',
            'à': 'in',
            'dans': 'in',
            'de': 'of',
            'du': 'of the',
            'de la': 'of the',
            'des': 'of the',
            
            # Verbes
            'avoir': 'have',
            'ai': 'have',
            'suis': 'am',
            'je': 'i',
            'mon': 'my',
            'ma': 'my',
            'mes': 'my',
            
            # Temps
            'depuis': 'since',
            'jours': 'days',
            'jour': 'day',
            'heures': 'hours',
            'heure': 'hour',
            'semaine': 'week',
            'semaines': 'weeks',
            'mois': 'month',
            
            # Autres
            'aussi': 'also',
            'et': 'and',
            'avec': 'with',
            'sans': 'without',
        }
        
        # ES → EN
        self.es_to_en = {
            'dolor': 'pain',
            'cabeza': 'head',
            'dolor de cabeza': 'headache',
            'estómago': 'stomach',
            'corazón': 'heart',
            'náusea': 'nausea',
            'vómito': 'vomiting',
            'fiebre': 'fever',
            'tos': 'cough',
            'mareo': 'dizziness',
            'tengo': 'i have',
            'me duele': 'my pain',
        }
        
        # Numéros d'urgence par pays
        self.emergency_numbers = {
            'Tunisie': {
                'samu': '190',
                'urgences': '197',
                'police': '197',
                'pompiers': '198'
            },
            'France': {
                'samu': '15',
                'urgences': '112',
                'police': '17',
                'pompiers': '18'
            },
            'Maroc': {
                'samu': '150',
                'urgences': '141',
                'police': '19',
                'pompiers': '15'
            }
        }
    
    def detect_language(self, text: str) -> Language:
        """
        Détecte la langue d'un texte (AMÉLIORÉ pour mixte)
        """
        text_lower = text.lower()
        
        # Compter les marqueurs
        scores = {lang: 0 for lang in Language if lang != Language.UNKNOWN}
        
        for lang, markers in self.language_markers.items():
            # Mots
            for word in markers['words']:
                if word in text_lower:
                    scores[lang] += 2
            
            # Patterns
            for pattern in markers['patterns']:
                if pattern in text_lower:
                    scores[lang] += 3
        
        # Caractères arabes
        if re.search(r'[\u0600-\u06FF]', text):
            scores[Language.ARABIC] += 10
        
        # Accents français
        if re.search(r'[àâäéèêëïîôùûüÿç]', text_lower):
            scores[Language.FRENCH] += 3
        
        # Accents espagnols
        if re.search(r'[áéíóúñü]', text_lower):
            scores[Language.SPANISH] += 2
        
        # Retourner la langue avec le score le plus élevé
        if max(scores.values()) > 0:
            detected = max(scores.items(), key=lambda x: x[1])[0]
            return detected
        
        return Language.ENGLISH  # Défaut = anglais
    
    def translate_to_english(self, text: str, source_lang: Language = None) -> str:
        """
        Traduit vers l'anglais (AMÉLIORÉ - mot-par-mot même si EN)
        """
        
        if source_lang is None:
            source_lang = self.detect_language(text)
        
        text_lower = text.lower()
        translated = text_lower
        
        # TOUJOURS essayer de traduire les mots FR/ES même si langue = EN
        # (pour gérer les textes mixtes comme "i have pain in my coeur")
        
        # Étape 1: Traduire expressions multi-mots FRANÇAIS
        for french, english in sorted(self.fr_to_en.items(), key=lambda x: len(x[0]), reverse=True):
            if ' ' in french:  # Expressions multi-mots
                if french in translated:
                    translated = translated.replace(french, english)
        
        # Étape 2: Traduire mots simples FRANÇAIS
        words = translated.split()
        translated_words = []
        
        for word in words:
            # Nettoyer le mot
            clean_word = re.sub(r'[^\w]', '', word)
            
            # Chercher traduction
            if clean_word in self.fr_to_en:
                translated_words.append(self.fr_to_en[clean_word])
            elif clean_word in self.es_to_en:
                translated_words.append(self.es_to_en[clean_word])
            else:
                translated_words.append(word)
        
        translated = ' '.join(translated_words)
        
        # Étape 3: Nettoyer
        translated = re.sub(r'\s+', ' ', translated).strip()
        
        return translated
    
    def get_emergency_info(self, language: Language) -> Dict[str, str]:
        """Retourne les numéros d'urgence"""
        
        # Mapping langue → pays par défaut
        country_map = {
            Language.FRENCH: 'Tunisie',
            Language.ENGLISH: 'France',
            Language.ARABIC: 'Tunisie',
            Language.SPANISH: 'France'
        }
        
        country = country_map.get(language, 'Tunisie')
        
        return self.emergency_numbers.get(
            country,
            self.emergency_numbers['Tunisie']
        )