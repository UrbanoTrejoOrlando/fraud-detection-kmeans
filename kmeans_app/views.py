from django.shortcuts import render
from django.http import JsonResponse
import json
from .ml_utils import FraudDataSimulator, KMeansAnalyzer

# Variable global para almacenar los datos generados
simulated_df = None

def index(request):
    """Página principal"""
    return render(request, 'kmeans_app/index.html')

def dataset_info(request):
    """Información del dataset"""
    global simulated_df
    
    if simulated_df is None:
        simulator = FraudDataSimulator()
        simulated_df = simulator.generate_simulated_data(n_samples=5000)
    
    info = FraudDataSimulator.get_dataset_info(simulated_df)
    
    context = {
        'dataset_info': info,
        'num_samples': info['num_samples'],
        'num_features': info['num_features'],
        'num_normal': info['num_normal'],
        'num_fraud': info['num_fraud'],
        'fraud_percentage': round(info['fraud_percentage'], 3),
    }
    
    return render(request, 'kmeans_app/dataset_info.html', context)

def visualizations(request):
    """Visualizaciones del dataset"""
    global simulated_df
    
    if simulated_df is None:
        simulator = FraudDataSimulator()
        simulated_df = simulator.generate_simulated_data(n_samples=5000)
    
    # Generar algunas visualizaciones básicas
    import matplotlib.pyplot as plt
    import seaborn as sns
    import io
    import base64
    import matplotlib
    matplotlib.use('Agg')
    
    # Gráfico de distribución de clases
    plt.figure(figsize=(10, 6))
    class_counts = simulated_df['Class'].value_counts()
    plt.bar(['Normales (0)', 'Fraudulentas (1)'], class_counts.values, 
            color=['blue', 'red'])
    plt.title('Distribución de Transacciones Normales vs Fraudulentas')
    plt.ylabel('Número de Transacciones')
    
    for i, v in enumerate(class_counts.values):
        plt.text(i, v + 10, str(v), ha='center')
    
    buffer1 = io.BytesIO()
    plt.savefig(buffer1, format='png', dpi=100, bbox_inches='tight')
    buffer1.seek(0)
    class_dist_img = base64.b64encode(buffer1.getvalue()).decode('utf-8')
    plt.close()
    
    # Gráfico de distribución de montos
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    simulated_df['Amount'].hist(bins=50, color='skyblue', edgecolor='black')
    plt.title('Distribución de Montos')
    plt.xlabel('Monto')
    plt.ylabel('Frecuencia')
    
    plt.subplot(1, 2, 2)
    simulated_df[simulated_df['Amount'] < 1000]['Amount'].hist(
        bins=50, color='lightgreen', edgecolor='black'
    )
    plt.title('Distribución de Montos (< 1000)')
    plt.xlabel('Monto')
    plt.ylabel('Frecuencia')
    
    plt.tight_layout()
    buffer2 = io.BytesIO()
    plt.savefig(buffer2, format='png', dpi=100, bbox_inches='tight')
    buffer2.seek(0)
    amount_dist_img = base64.b64encode(buffer2.getvalue()).decode('utf-8')
    plt.close()
    
    context = {
        'class_distribution': class_dist_img,
        'amount_distribution': amount_dist_img,
        'num_samples': len(simulated_df),
    }
    
    return render(request, 'kmeans_app/visualizations.html', context)

# En kmeans_app/views.py, modificar la función kmeans_analysis:

def kmeans_analysis(request):
    """Análisis con KMeans"""
    global simulated_df
    
    if simulated_df is None:
        # Redirigir o generar datos automáticamente
        simulator = FraudDataSimulator()
        simulated_df = simulator.generate_simulated_data(n_samples=5000)
    
    # Obtener parámetros del formulario con valores por defecto
    try:
        n_clusters = int(request.GET.get('n_clusters', 2))
    except ValueError:
        n_clusters = 2
    
    try:
        max_clusters = int(request.GET.get('max_clusters', 10))
    except ValueError:
        max_clusters = 10
    
    # Asegurar que max_clusters sea mayor que n_clusters
    if max_clusters <= n_clusters:
        max_clusters = n_clusters + 1
    
    # Realizar análisis KMeans
    analyzer = KMeansAnalyzer(n_clusters=n_clusters)
    
    try:
        # Preparar datos
        X_scaled, y_true = analyzer.prepare_data(simulated_df)
        
        # Aplicar PCA
        X_pca = analyzer.apply_pca(X_scaled)
        
        # Aplicar KMeans
        cluster_labels, centroids = analyzer.fit_kmeans(X_pca)
        
        # Evaluar clusters
        evaluation = analyzer.evaluate_clusters(X_pca, cluster_labels)
        
        # Generar gráficos
        cluster_plot = analyzer.plot_clusters(X_pca, cluster_labels, centroids)
        elbow_plot, inertias = analyzer.plot_elbow_method(X_pca, max_clusters)
        
        # Estadísticas de clusters
        cluster_stats = {}
        for i in range(n_clusters):
            cluster_mask = (cluster_labels == i)
            cluster_stats[f'Cluster {i}'] = {
                'size': int(cluster_mask.sum()),
                'percentage': round((cluster_mask.sum() / len(cluster_labels)) * 100, 2)
            }
        
        context = {
            'cluster_plot': cluster_plot,
            'elbow_plot': elbow_plot,
            'n_clusters': n_clusters,
            'max_clusters': max_clusters,
            'silhouette_score': round(evaluation['silhouette_score'], 3),
            'purity': round(evaluation['purity'] * 100, 2),
            'inertia': round(evaluation['inertia'], 2),
            'cluster_stats': cluster_stats,
            'inertias': [round(i, 2) for i in inertias],
        }
        
    except Exception as e:
        # En caso de error, mostrar página con valores por defecto
        print(f"Error en KMeans: {str(e)}")
        context = {
            'n_clusters': n_clusters,
            'max_clusters': max_clusters,
            'silhouette_score': 0.0,
            'purity': 0.0,
            'inertia': 0.0,
            'cluster_stats': {},
            'inertias': [],
        }
    
    return render(request, 'kmeans_app/kmeans_result.html', context)

def generate_simulated_data(request):
    """Genera nuevos datos simulados"""
    global simulated_df
    
    n_samples = int(request.GET.get('n_samples', 5000))
    fraud_ratio = float(request.GET.get('fraud_ratio', 0.0017))
    
    simulator = FraudDataSimulator()
    simulated_df = simulator.generate_simulated_data(
        n_samples=n_samples, 
        fraud_ratio=fraud_ratio
    )
    
    info = FraudDataSimulator.get_dataset_info(simulated_df)
    
    return JsonResponse({
        'status': 'success',
        'num_samples': info['num_samples'],
        'num_fraud': info['num_fraud'],
        'num_normal': info['num_normal'],
        'fraud_percentage': round(info['fraud_percentage'], 3),
    })