from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
import json
import time
from .ml_utils import FraudDataSimulator, KMeansAnalyzer

# Variable global para almacenar los datos generados
simulated_df = None

def index(request):
    """Página principal"""
    return render(request, 'kmeans_app/index.html')

def dataset_info(request):
    """Información del dataset - Corregido"""
    global simulated_df
    
    if simulated_df is None:
        simulator = FraudDataSimulator()
        simulated_df = simulator.generate_simulated_data(n_samples=2000)
    
    info = FraudDataSimulator.get_dataset_info(simulated_df)
    
    # Preparar datos para la tabla
    table_data = []
    if 'describe' in info and info['describe']:
        for column, stats in info['describe'].items():
            table_data.append({
                'column': column,
                'count': stats.get('count', 0),
                'mean': stats.get('mean', 0),
                'std': stats.get('std', 0),
                'min': stats.get('min', 0),
                'max': stats.get('max', 0),
            })
    
    context = {
        'dataset_info': info,
        'num_samples': info['num_samples'],
        'num_features': info['num_features'],
        'num_normal': info['num_normal'],
        'num_fraud': info['num_fraud'],
        'fraud_percentage': round(info['fraud_percentage'], 3),
        'table_data': table_data,
        'columns': info.get('columns', []),
    }
    
    return render(request, 'kmeans_app/dataset_info.html', context)

def visualizations(request):
    """Visualizaciones del dataset - Optimizado"""
    global simulated_df
    
    if simulated_df is None:
        simulator = FraudDataSimulator()
        simulated_df = simulator.generate_simulated_data(n_samples=2000)
    
    # Generar visualizaciones simplificadas
    import matplotlib.pyplot as plt
    import io
    import base64
    import matplotlib
    matplotlib.use('Agg')
    
    # Gráfico de distribución de clases (simplificado)
    plt.figure(figsize=(8, 5))
    class_counts = simulated_df['Class'].value_counts()
    colors = ['#3498db', '#e74c3c']
    plt.bar(['Normales (0)', 'Fraudulentas (1)'], class_counts.values, color=colors)
    plt.title('Distribución de Transacciones')
    plt.ylabel('Número de Transacciones')
    
    buffer1 = io.BytesIO()
    plt.savefig(buffer1, format='png', dpi=80, bbox_inches='tight')
    buffer1.seek(0)
    class_dist_img = base64.b64encode(buffer1.getvalue()).decode('utf-8')
    plt.close()
    
    context = {
        'class_distribution': class_dist_img,
        'num_samples': len(simulated_df),
        'num_normal': class_counts.get(0, 0),
        'num_fraud': class_counts.get(1, 0),
    }
    
    return render(request, 'kmeans_app/visualizations.html', context)

def kmeans_analysis(request):
    """Análisis con KMeans - Optimizado para Render"""
    global simulated_df
    
    if simulated_df is None:
        # Redirigir o generar datos automáticamente
        simulator = FraudDataSimulator()
        simulated_df = simulator.generate_simulated_data(n_samples=2000)
    
    # Obtener parámetros del formulario con valores por defecto
    try:
        n_clusters = int(request.GET.get('n_clusters', 2))
    except ValueError:
        n_clusters = 2
    
    try:
        max_clusters = int(request.GET.get('max_clusters', 6))  # Reducido
    except ValueError:
        max_clusters = 6
    
    # Limitar valores para Render
    n_clusters = max(2, min(n_clusters, 5))  # Máximo 5 clusters
    max_clusters = max(3, min(max_clusters, 8))  # Máximo 8 para método del codo
    
    # Asegurar que max_clusters sea mayor que n_clusters
    if max_clusters <= n_clusters:
        max_clusters = n_clusters + 1
    
    context = {
        'n_clusters': n_clusters,
        'max_clusters': max_clusters,
        'silhouette_score': 0.0,
        'purity': 0.0,
        'inertia': 0.0,
        'cluster_stats': {},
        'inertias': [],
    }
    
    try:
        # Configurar timeout para Render
        start_time = time.time()
        
        # Realizar análisis KMeans
        analyzer = KMeansAnalyzer(n_clusters=n_clusters)
        
        # Preparar datos
        X_scaled, y_true = analyzer.prepare_data(simulated_df)
        
        # Verificar timeout
        if time.time() - start_time > 10:  # 10 segundos límite
            raise TimeoutError("Timeout preparando datos")
        
        # Aplicar PCA
        X_pca = analyzer.apply_pca(X_scaled)
        
        # Verificar timeout
        if time.time() - start_time > 15:  # 15 segundos límite
            raise TimeoutError("Timeout aplicando PCA")
        
        # Aplicar KMeans
        cluster_labels, centroids = analyzer.fit_kmeans(X_pca)
        
        # Verificar timeout
        if time.time() - start_time > 20:  # 20 segundos límite total
            raise TimeoutError("Timeout ejecutando KMeans")
        
        # Evaluar clusters
        evaluation = analyzer.evaluate_clusters(X_pca, cluster_labels)
        
        # Generar gráficos (solo si no hay timeout)
        try:
            cluster_plot = analyzer.plot_clusters(X_pca, cluster_labels, centroids)
            elbow_plot, inertias = analyzer.plot_elbow_method(X_pca, max_clusters)
            
            context['cluster_plot'] = cluster_plot
            context['elbow_plot'] = elbow_plot
            context['inertias'] = [round(i, 2) for i in inertias]
        except:
            # Si fallan los gráficos, continuar sin ellos
            pass
        
        # Estadísticas de clusters
        cluster_stats = {}
        for i in range(n_clusters):
            cluster_mask = (cluster_labels == i)
            cluster_stats[f'Cluster {i}'] = {
                'size': int(cluster_mask.sum()),
                'percentage': round((cluster_mask.sum() / len(cluster_labels)) * 100, 2)
            }
        
        # Actualizar contexto con resultados
        context.update({
            'silhouette_score': round(evaluation['silhouette_score'], 3),
            'purity': round(evaluation['purity'] * 100, 2),
            'inertia': round(evaluation['inertia'], 2),
            'cluster_stats': cluster_stats,
            'success': True,
        })
        
    except TimeoutError as e:
        context['error'] = "El análisis tardó demasiado. Reduciendo complejidad..."
        context['success'] = False
        
    except Exception as e:
        context['error'] = f"Error en el análisis: {str(e)}"
        context['success'] = False
        print(f"Error en KMeans: {str(e)}")
    
    return render(request, 'kmeans_app/kmeans_result.html', context)

def generate_simulated_data(request):
    """Genera nuevos datos simulados - Optimizado"""
    global simulated_df
    
    try:
        n_samples = int(request.GET.get('n_samples', 2000))  # Reducido
        fraud_ratio = float(request.GET.get('fraud_ratio', 0.0017))
        
        # Limitar para Render
        n_samples = min(n_samples, 3000)  # Máximo 3000 muestras
        
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
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        })

from django.http import JsonResponse

def dataset_info_api(request):
    global simulated_df

    if simulated_df is None:
        simulator = FraudDataSimulator()
        simulated_df = simulator.generate_simulated_data(n_samples=2000)

    info = FraudDataSimulator.get_dataset_info(simulated_df)

    return JsonResponse({
        "success": True,
        "num_samples": info["num_samples"],
        "num_features": info["num_features"],
        "num_normal": info["num_normal"],
        "num_fraud": info["num_fraud"],
        "fraud_percentage": round(info["fraud_percentage"], 3),
        "columns": info.get("columns", [])
    })


from django.http import JsonResponse

def kmeans_api(request):
    global simulated_df

    if simulated_df is None:
        simulator = FraudDataSimulator()
        simulated_df = simulator.generate_simulated_data(n_samples=2000)

    try:
        n_clusters = int(request.GET.get('n_clusters', 2))
        n_clusters = max(2, min(n_clusters, 5))

        analyzer = KMeansAnalyzer(n_clusters=n_clusters)
        X_scaled, y_true = analyzer.prepare_data(simulated_df)
        X_pca = analyzer.apply_pca(X_scaled)

        cluster_labels, centroids = analyzer.fit_kmeans(X_pca)
        evaluation = analyzer.evaluate_clusters(X_pca, cluster_labels)

        cluster_plot = analyzer.plot_clusters(X_pca, cluster_labels, centroids)

        cluster_stats = {}
        for i in range(n_clusters):
            size = int((cluster_labels == i).sum())
            cluster_stats[f"Cluster {i}"] = size

        return JsonResponse({
            "success": True,
            "n_clusters": n_clusters,
            "silhouette": round(evaluation["silhouette_score"], 3),
            "purity": round(evaluation["purity"] * 100, 2),
            "inertia": round(evaluation["inertia"], 2),
            "cluster_stats": cluster_stats,
            "cluster_plot": cluster_plot,  # 👈 base64 PNG
        })

    except Exception as e:
        return JsonResponse({
            "success": False,
            "error": str(e)
        })
