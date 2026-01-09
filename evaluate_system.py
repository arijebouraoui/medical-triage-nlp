"""
System Evaluation Script
Tests the NLP triage system on dataset and calculates accuracy
"""

from medical_triage_system import MedicalTriageAI
from agents.data_loader.medical_data_loader import MedicalDataLoader as MedicalDatasetLoader
import json


def extract_urgency_from_report(report: str) -> str:
    """Extract urgency level from system report."""
    if "URGENCE VITALE" in report or "IMMÉDIAT" in report:
        return "URGENCE VITALE"
    elif "URGENCE ÉLEVÉE" in report or "Aujourd'hui même" in report:
        return "URGENCE ÉLEVÉE"
    elif "URGENCE MODÉRÉE" in report or "24-48 heures" in report:
        return "URGENCE MODÉRÉE"
    elif "NON URGENT" in report or "Cette semaine" in report:
        return "NON URGENT"
    else:
        return "URGENCE MODÉRÉE"  # Default fallback


def extract_specialist_from_report(report: str) -> str:
    """Extract specialist from system report."""
    import re
    # Look for "Consultation: SpecialistName" or "SPÉCIALISTE RECOMMANDÉ" blocks
    match = re.search(r"Consultation:\s*(.*)", report)
    if match:
        result = match.group(1).strip()
        # Clean up possible trailing formatting
        result = result.split("\n")[0].strip()
        return result
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
    print("Step 1: Loading dataset...")
    loader = MedicalDatasetLoader("data/processed/dataset_processed.json")
    
    # Get statistics
    stats = loader.get_statistics()
    print(f"\nDataset Overview:")
    print(f"  Total samples: {stats['total_cases']}")
    print(f"  Urgency distribution:")
    for urgency, count in stats['urgency_distribution'].items():
        print(f"    {urgency}: {count}")
    
    # Split train/test (Manual split since loader doesn't have it)
    import random
    all_data = loader.dataset.copy()
    random.shuffle(all_data)
    
    split_idx = int(len(all_data) * 0.8)
    train_data = all_data[:split_idx]
    test_data = all_data[split_idx:]
    
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
        if i == 1:
            print(f"DEBUG: Sample keys: {sample.keys()}")
        
        patient_text = sample.get('patient_text')
        if not patient_text:
            print(f"WARNING: Sample {i} missing patient_text. Keys: {sample.keys()}")
            continue
            
        true_urgency = sample.get('urgency_level', 'UNKNOWN')
        # Normalize encoding issues in ground truth
        if true_urgency == "URGENCE MODÃ‰RÃ‰E":
            true_urgency = "URGENCE MODÉRÉE"
        elif true_urgency == "URGENCE Ã‰LEVÃ‰E":
            true_urgency = "URGENCE ÉLEVÉE"

        
        try:
            # Progress indicator
            if i % 10 == 0:
                print(f"Progress: {i}/{len(test_data)} samples processed... ({correct}/{total} correct so far)")
            
            # Get prediction (suppress verbose output to keep console clean)
            import sys
            from io import StringIO
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            # Use verbose=False to get cleaner log in evaluate
            report = system.analyze_and_respond(patient_text, verbose=False)
            
            sys.stdout = old_stdout
            
            # Extract predicted urgency and specialist
            predicted_urgency = extract_urgency_from_report(report)
            predicted_specialist = extract_specialist_from_report(report)
            true_specialist = sample.get('specialist', 'UNKNOWN')
            
            # Check correctness
            urgency_correct = (predicted_urgency == true_urgency)
            specialist_correct = (predicted_specialist == true_specialist) or (true_specialist in predicted_specialist)
            
            if urgency_correct:
                correct += 1
            
            total += 1
            
            results.append({
                'text': patient_text,
                'disease': sample.get('disease', ''),
                'true_urgency': true_urgency,
                'predicted_urgency': predicted_urgency,
                'urgency_correct': urgency_correct,
                'true_specialist': true_specialist,
                'predicted_specialist': predicted_specialist,
                'specialist_correct': specialist_correct
            })
        
        except Exception as e:
            # Safely log error
            try:
                with open("error_log.txt", "a", encoding='utf-8') as f:
                    f.write(f"Error on sample {i}: {str(e)}\n")
            except:
                pass
            print(f"Error on sample {i}")
            continue
    
    # Calculate metrics
    accuracy = (correct / total * 100) if total > 0 else 0
    spec_accuracy = (sum(1 for r in results if r['specialist_correct']) / total * 100) if total > 0 else 0
    
    # Print results
    print("\n" + "="*70)
    print("📋 RAPPORT DE PERFORMANCE DU SYSTÈME")
    print("="*70)
    print(f"\n✅ PRÉCISION URGENCE : {accuracy:.2f}% ({correct}/{total})")
    print(f"✅ PRÉCISION SPÉCIALISTE : {spec_accuracy:.2f}%")
    
    # Results by urgency level
    print("\nAnalyse par niveau d'urgence :")
    print("-" * 70)
    for urgency in ["URGENCE VITALE", "URGENCE ÉLEVÉE", "URGENCE MODÉRÉE", "NON URGENT"]:
        urgency_results = [r for r in results if r['true_urgency'] == urgency]
        if urgency_results:
            u_correct = sum(1 for r in urgency_results if r['urgency_correct'])
            u_total = len(urgency_results)
            u_acc = (u_correct / u_total * 100)
            print(f"  {urgency:20s}: {u_acc:5.1f}% ({u_correct:3d}/{u_total:3d})")
    
    # Show example predictions
    print("\nExemples de prédictions :")
    print("-" * 70)
    
    for i, result in enumerate(results[:5], 1):
        status = "✅" if result['urgency_correct'] and result['specialist_correct'] else "⚠️"
        print(f"\n  {i}. {status} Maladie: {result['disease']}")
        print(f"     Symptômes: {result['text'][:60]}...")
        print(f"     Urgence - Attendu: {result['true_urgency']} | Prédit: {result['predicted_urgency']}")
        print(f"     Spécialiste - Attendu: {result['true_specialist']} | Prédit: {result['predicted_specialist']}")
    
    # Save detailed results
    output_file = 'data/processed/evaluation_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'overall_urgency_accuracy': accuracy,
            'overall_specialist_accuracy': spec_accuracy,
            'total_tested': total,
            'results': results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Résultats détaillés sauvegardés dans: {output_file}")
    print("\n" + "="*70)
    print(f"Taux de succès global : {accuracy:.2f}%")
    print("="*70 + "\n")
    
    return accuracy, results


if __name__ == "__main__":
    # Run evaluation on 50 samples (change to -1 for all 984 samples)
    print("\nStarting evaluation on 50 test samples...")
    print("(Use num_samples=-1 to test all 984 samples)\n")
    
    accuracy, results = evaluate_system(num_samples=50)