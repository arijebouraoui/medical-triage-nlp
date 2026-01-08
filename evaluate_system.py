"""
System Evaluation Script
Tests the NLP triage system on dataset and calculates accuracy
"""

from medical_triage_system import MedicalTriageAI
from data_loader import MedicalDatasetLoader
import json


def extract_urgency_from_report(report: str) -> str:
    """Extract urgency level from system report."""
    if "URGENCE VITALE" in report or "ATTENTION - URGENCE VITALE" in report:
        return "URGENCE VITALE"
    elif "URGENCE ÉLEVÉE" in report or "URGENCE ELEVEE" in report:
        return "URGENCE ÉLEVÉE"
    elif "URGENCE MODÉRÉE" in report or "URGENCE MODEREE" in report:
        return "URGENCE MODÉRÉE"
    elif "NON URGENT" in report:
        return "NON URGENT"
    else:
        return "UNKNOWN"


def evaluate_system(num_samples: int = 50):
    """
    Evaluate NLP system on test dataset.
    
    Args:
        num_samples: Number of samples to test (50 for quick test, -1 for all)
    """
    
    print("\n" + "="*70)
    print("SYSTEM EVALUATION ON KAGGLE DATASET")
    print("="*70 + "\n")
    
    # Load dataset
    print("Step 1: Loading dataset...")
    loader = MedicalDatasetLoader("data/raw/dataset.csv")
    loader.load_disease_symptom_dataset()
    loader.preprocess_for_nlp()
    
    # Get statistics
    stats = loader.get_statistics()
    print(f"\nDataset Overview:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Urgency distribution:")
    for urgency, count in stats['urgency_distribution'].items():
        print(f"    {urgency}: {count}")
    
    # Split train/test
    train_data, test_data = loader.split_train_test(test_size=0.2)
    
    # Use subset for testing
    if num_samples > 0 and num_samples < len(test_data):
        test_data = test_data[:num_samples]
    
    print(f"\nStep 2: Evaluating on {len(test_data)} test samples...")
    print("(This may take a few minutes...)\n")
    
    # Initialize system
    system = MedicalTriageAI(patient_country="France")
    
    # Evaluate
    correct = 0
    total = 0
    results = []
    
    print("-" * 70)
    
    for i, sample in enumerate(test_data, 1):
        patient_text = sample['patient_text']
        true_urgency = sample['urgency_level']
        
        try:
            # Progress indicator
            if i % 10 == 0:
                print(f"Progress: {i}/{len(test_data)} samples processed... ({correct}/{total} correct so far)")
            
            # Get prediction (suppress verbose output)
            import sys
            from io import StringIO
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            report = system.analyze_and_respond(patient_text)
            
            sys.stdout = old_stdout
            
            # Extract predicted urgency
            predicted_urgency = extract_urgency_from_report(report)
            
            # Check correctness
            is_correct = (predicted_urgency == true_urgency)
            
            if is_correct:
                correct += 1
            
            total += 1
            
            results.append({
                'text': patient_text,
                'disease': sample.get('disease', ''),
                'true': true_urgency,
                'predicted': predicted_urgency,
                'correct': is_correct
            })
        
        except Exception as e:
            print(f"Error on sample {i}: {e}")
            continue
    
    # Calculate metrics
    accuracy = (correct / total * 100) if total > 0 else 0
    
    # Print results
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"\nOverall Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    # Results by urgency level
    print("\nAccuracy by Urgency Level:")
    print("-" * 70)
    for urgency in ["URGENCE VITALE", "URGENCE ÉLEVÉE", "URGENCE MODÉRÉE", "NON URGENT"]:
        urgency_results = [r for r in results if r['true'] == urgency]
        if urgency_results:
            urgency_correct = sum(1 for r in urgency_results if r['correct'])
            urgency_total = len(urgency_results)
            urgency_acc = (urgency_correct / urgency_total * 100) if urgency_total > 0 else 0
            print(f"  {urgency:20s}: {urgency_acc:5.1f}% ({urgency_correct:3d}/{urgency_total:3d})")
    
    # Show example predictions
    print("\nSample Predictions:")
    print("-" * 70)
    
    # Show correct and incorrect examples
    correct_examples = [r for r in results if r['correct']][:3]
    incorrect_examples = [r for r in results if not r['correct']][:3]
    
    print("\n✅ Correct Predictions:")
    for i, result in enumerate(correct_examples, 1):
        print(f"\n  {i}. Disease: {result['disease']}")
        print(f"     Symptoms: {result['text'][:60]}...")
        print(f"     Expected: {result['true']}")
        print(f"     Got: {result['predicted']}")
    
    if incorrect_examples:
        print("\n❌ Incorrect Predictions:")
        for i, result in enumerate(incorrect_examples, 1):
            print(f"\n  {i}. Disease: {result['disease']}")
            print(f"     Symptoms: {result['text'][:60]}...")
            print(f"     Expected: {result['true']}")
            print(f"     Got: {result['predicted']}")
    
    # Save detailed results
    output_file = 'data/processed/evaluation_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'by_urgency': {
                urgency: {
                    'correct': sum(1 for r in results if r['true'] == urgency and r['correct']),
                    'total': sum(1 for r in results if r['true'] == urgency),
                    'accuracy': (sum(1 for r in results if r['true'] == urgency and r['correct']) / 
                                sum(1 for r in results if r['true'] == urgency) * 100) 
                                if sum(1 for r in results if r['true'] == urgency) > 0 else 0
                }
                for urgency in ["URGENCE VITALE", "URGENCE ÉLEVÉE", "URGENCE MODÉRÉE", "NON URGENT"]
            },
            'results': results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    
    print("\n" + "="*70)
    print(f"FINAL ACCURACY: {accuracy:.2f}%")
    print("="*70 + "\n")
    
    return accuracy, results


if __name__ == "__main__":
    # Run evaluation on 50 samples (change to -1 for all 984 samples)
    print("\nStarting evaluation on 50 test samples...")
    print("(Use num_samples=-1 to test all 984 samples)\n")
    
    accuracy, results = evaluate_system(num_samples=50)