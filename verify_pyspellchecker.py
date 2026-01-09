
from agents.nlp.context_spell_corrector import ContextSpellCorrector

print("Initializing ContextSpellCorrector...")
corrector = ContextSpellCorrector()

if hasattr(corrector, 'std_spell'):
    print("✅ Pyspellchecker loaded")
else:
    print("⚠️ Pyspellchecker NOT loaded (using fallback)")

tests = [
    "i havee a pain",
    "pain in my heart",
    "j'ai mal"
]

for t in tests:
    corrected, _ = corrector.correct_text(t)
    print(f"Original: '{t}' -> Corrected: '{corrected}'")
