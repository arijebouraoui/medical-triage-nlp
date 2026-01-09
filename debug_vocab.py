
from agents.data_loader.medical_data_loader import MedicalDataLoader
from agents.nlp.context_spell_corrector import ContextSpellCorrector

print("Loading data...")
loader = MedicalDataLoader("data/processed/dataset_processed.json")
corrector = ContextSpellCorrector()

# Train
corpus = [case.get('patient_text', '') for case in loader.dataset if case.get('patient_text')]
corrector.train(corpus)

print(f"\nVocab size: {len(corrector.vocab)}")
print(f"Top 10 words: {corrector.vocab.most_common(10)}")

check_words = ['have', 'pain', 'heart', 'chest', 'my', 'is', 'i']
print("\nChecking common words:")
for w in check_words:
    print(f"'{w}': {corrector.vocab[w]}")

print(f"\nCorrection test 'havee': {corrector.correct_text('havee')}")
