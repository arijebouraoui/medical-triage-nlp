import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def generate_plots(results_path='data/processed/evaluation_results.json', output_dir='reports/figures'):
    """Génère les graphiques de performance du système"""
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Charger les résultats
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data['results']
    df = pd.DataFrame(results)
    
    # Configuration du style
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    
    # --- PLOT 1: Performance Globale ---
    print("Génération du graphique de performance globale...")
    metrics = {
        'Urgence': data['overall_urgency_accuracy'],
        'Spécialiste': data['overall_specialist_accuracy']
    }
    
    plt.figure()
    colors = ['#ff4b4b', '#0068c9']
    bars = plt.bar(metrics.keys(), metrics.values(), color=colors)
    plt.ylim(0, 110)
    plt.ylabel('Précision (%)')
    plt.title('Performance Globale du Système (N=200)', fontsize=14, fontweight='bold')
    
    # Ajouter les valeurs au-dessus des barres
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.1f}%', ha='center', fontweight='bold')
        
    plt.savefig(f'{output_dir}/overall_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # --- PLOT 2: Précision par Niveau d'Urgence ---
    print("Génération du graphique par niveau d'urgence...")
    urgency_stats = df.groupby('true_urgency')['urgency_correct'].mean() * 100
    
    plt.figure()
    urgency_order = ['URGENCE VITALE', 'URGENCE ÉLEVÉE', 'URGENCE MODÉRÉE', 'NON URGENT']
    # Filter to only existing ones in test set
    existing_order = [u for u in urgency_order if u in urgency_stats.index]
    
    sns.barplot(x=urgency_stats.index, y=urgency_stats.values, order=existing_order, palette='viridis')
    plt.ylim(0, 110)
    plt.ylabel('Précision (%)')
    plt.xlabel("Niveau d'Urgence Réel")
    plt.title('Précision par Catégorie d\'Urgence', fontsize=14, fontweight='bold')
    
    for i, v in enumerate(urgency_stats[existing_order]):
        plt.text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
        
    plt.savefig(f'{output_dir}/urgency_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # --- PLOT 3: Répartition des prédictions (Confusion d'urgence simplifiée) ---
    print("Génération de la matrice de répartition...")
    plt.figure(figsize=(8, 6))
    confusion = pd.crosstab(df['true_urgency'], df['predicted_urgency'], normalize='index') * 100
    sns.heatmap(confusion, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': 'Rerépartition (%)'})
    plt.title('Matrice de Confusion : Urgence (%)', fontsize=14, fontweight='bold')
    plt.ylabel('Attendu')
    plt.xlabel('Prédit')
    plt.savefig(f'{output_dir}/urgency_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✅ Graphiques générés dans : {output_dir}/")
    print(f"1. overall_performance.png")
    print(f"2. urgency_accuracy.png")
    print(f"3. urgency_heatmap.png")

if __name__ == "__main__":
    generate_plots()
