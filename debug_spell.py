
from agents.analyzer.nlp_analyzer_v3 import MedicalNLPAnalyzer
import sys

# Suppress other prints if possible, but we want to see the spell checker debugs
print("Initializing Analyzer...")
analyzer = MedicalNLPAnalyzer()

text = "i have ache in my heart"
print(f"\nTesting: '{text}'")

# Access spell corrector directly
corrected, corrections = analyzer.spell_corrector.correct_text(text)
print(f"Result: '{corrected}'")
print(f"Corrections_List: {corrections}")
