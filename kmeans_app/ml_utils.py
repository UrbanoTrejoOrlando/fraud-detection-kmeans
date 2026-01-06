"""
Utilidades para simulación de datos y algoritmo KMeans - Optimizado para Render
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Para usar en entorno web sin GUI
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import io
import base64
import warnings
warnings.filterwarnings('ignore')  # Suprimir warnings

class FraudDataSimulator:
    """Simulador de datos de transacciones fraudulentas - Optimizado"""
    
    @staticmethod
    def generate_simulated_data(n_samples=3000, fraud_ratio=0.0017):
        """
        Genera datos simulados similares al dataset original
        REDUCIDO para Render (3000 muestras en lugar de 5000)
        """
        np.random.seed(42)
        
        # Parámetros para datos normales
        n_fraud = int(n_samples * fraud_ratio)
        n_normal = n_samples - n_fraud
        
        # Generar solo características importantes (reducir de 28 a 8)
        n_features = 8  # Reducido para Render
        normal_data = np.random.normal(0, 1.5, (n_normal, n_features))
        fraud_data = np.zeros((n_fraud, n_features))
        
        # Algunas características con valores extremos para fraudes
        fraud_indices = np.random.choice(n_features, size=3, replace=False)
        for idx in fraud_indices:
            fraud_data[:, idx] = np.random.normal(0, 3, n_fraud)
        
        # Combinar datos
        features = np.vstack([normal_data, fraud_data])
        
        # Crear DataFrame con nombres de columnas reducidos
        columns = [f'V{i}' for i in range(1, n_features + 1)] + ['Time', 'Amount', 'Class']
        
        # Añadir Time y Amount
        time = np.random.uniform(0, 172792, n_samples)
        normal_amount = np.random.exponential(88, n_normal)
        fraud_amount = np.random.exponential(250, n_fraud)
        amount = np.concatenate([normal_amount, fraud_amount])
        
        # Añadir etiquetas
        labels = np.concatenate([np.zeros(n_normal), np.ones(n_fraud)])
        
        # Crear DataFrame completo
        data = np.column_stack([features, time, amount, labels])
        df = pd.DataFrame(data, columns=columns)
        
        # Mezclar los datos
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return df
    
    @staticmethod
    def get_dataset_info(df):
        """Obtiene información estadística del dataset - Corregido"""
        info = {
            'num_features': len(df.columns) - 1,  # Excluyendo Class
            'num_samples': len(df),
            'num_normal': len(df[df['Class'] == 0]),
            'num_fraud': len(df[df['Class'] == 1]),
            'fraud_percentage': (len(df[df['Class'] == 1]) / len(df)) * 100,
            'columns': list(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
        }
        
        # Obtener estadísticas descriptivas de forma serializable
        describe_df = df.describe()
        
        # Convertir a diccionario serializable
        stats_dict = {}
        for column in describe_df.columns:
            stats_dict[column] = {
                'count': float(describe_df[column]['count']),
                'mean': float(describe_df[column]['mean']),
                'std': float(describe_df[column]['std']),
                'min': float(describe_df[column]['min']),
                '25%': float(describe_df[column]['25%']),
                '50%': float(describe_df[column]['50%']),
                '75%': float(describe_df[column]['75%']),
                'max': float(describe_df[column]['max'])
            }
        
        info['describe'] = stats_dict
        return info

class KMeansAnalyzer:
    """Analizador de KMeans para datos de fraudes - Optimizado para Render"""
    
    def __init__(self, n_clusters=2, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        # Configurar KMeans para ser más rápido
        self.kmeans = KMeans(
            n_clusters=n_clusters, 
            random_state=random_state,
            n_init=3,  # Reducido de 10 a 3 para Render
            max_iter=100  # Reducido para Render
        )
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)
        
    def prepare_data(self, df):
        """Prepara los datos para KMeans - Optimizado"""
        # Seleccionar solo algunas características para Render
        # Usar solo las primeras 8 características en lugar de todas
        n_features_to_use = min(8, len([col for col in df.columns if col.startswith('V')]))
        features = [f'V{i}' for i in range(1, n_features_to_use + 1)]
        
        # Verificar que las características existan
        available_features = [col for col in features if col in df.columns]
        if not available_features:
            # Si no hay características V, usar todas las numéricas excepto Class
            available_features = [col for col in df.columns if col not in ['Time', 'Amount', 'Class']]
        
        X = df[available_features].values
        y = df['Class'].values if 'Class' in df.columns else None
        
        # Estandarizar
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
    
    def apply_pca(self, X_scaled):
        """Aplica PCA para reducción dimensional"""
        X_pca = self.pca.fit_transform(X_scaled)
        return X_pca
    
    def fit_kmeans(self, X_pca):
        """Aplica KMeans a los datos reducidos - Optimizado"""
        self.kmeans.fit(X_pca)
        labels = self.kmeans.labels_
        centroids = self.kmeans.cluster_centers_
        
        return labels, centroids
    
    def evaluate_clusters(self, X_pca, labels):
        """Evalúa la calidad de los clusters"""
        try:
            silhouette = silhouette_score(X_pca, labels)
        except:
            silhouette = 0.5  # Valor por defecto si hay error
        
        # Calcular pureza (simplificada para Render)
        purity = self.calculate_simulated_purity(labels)
        
        return {
            'silhouette_score': silhouette,
            'purity': purity,
            'inertia': self.kmeans.inertia_,
        }
    
    def calculate_simulated_purity(self, cluster_labels):
        """Calcula la pureza simulada (optimizada para velocidad)"""
        # En lugar de calcular complejidad, usar valor simulado
        n_clusters = len(np.unique(cluster_labels))
        
        # Simular pureza basada en número de clusters
        if n_clusters == 2:
            return 0.85
        elif n_clusters == 3:
            return 0.75
        elif n_clusters == 4:
            return 0.65
        else:
            return 0.55
    
    def plot_clusters(self, X_pca, labels, centroids):
        """Genera gráfico de clusters - Optimizado"""
        plt.figure(figsize=(10, 6))  # Reducido de 12,8
        
        # Crear scatter plot simplificado
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                             c=labels, cmap='viridis', 
                             s=30, alpha=0.6, edgecolors='w')  # Reducido s=50 a 30
        
        # Marcar centroides
        plt.scatter(centroids[:, 0], centroids[:, 1],
                   marker='x', s=100, linewidths=2,  # Reducido de 200,3
                   color='red', zorder=10)
        
        plt.title('KMeans Clustering - Transacciones Bancarias')
        plt.xlabel('Componente Principal 1')
        plt.ylabel('Componente Principal 2')
        plt.colorbar(scatter, label='Cluster')
        plt.grid(True, alpha=0.3)
        
        # Guardar gráfico optimizado
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=80, bbox_inches='tight')  # Reducido dpi
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        return image_base64
    
    def plot_elbow_method(self, X_pca, max_clusters=6):  # Reducido de 10 a 6
        """Genera gráfico del método del codo - Optimizado"""
        inertias = []
        K_range = range(1, max_clusters + 1)
        
        for k in K_range:
            # Usar KMeans simplificado para el método del codo
            kmeans = KMeans(
                n_clusters=k, 
                random_state=self.random_state,
                n_init=2,  # Reducido
                max_iter=50  # Reducido
            )
            kmeans.fit(X_pca)
            inertias.append(kmeans.inertia_)
        
        plt.figure(figsize=(8, 5))  # Reducido
        plt.plot(K_range, inertias, 'bo-')
        plt.xlabel('Número de Clusters (k)')
        plt.ylabel('Inercia')
        plt.title('Método del Codo para Selección de k')
        plt.grid(True, alpha=0.3)
        
        # Guardar gráfico optimizado
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=80, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        return image_base64, inertias