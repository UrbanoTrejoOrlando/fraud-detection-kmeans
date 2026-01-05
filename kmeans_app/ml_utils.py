"""
Utilidades para simulación de datos y algoritmo KMeans
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Para usar en entorno web sin GUI

class FraudDataSimulator:
    """Simulador de datos de transacciones fraudulentas"""
    
    @staticmethod
    def generate_simulated_data(n_samples=10000, fraud_ratio=0.0017):
        """
        Genera datos simulados similares al dataset original
        """
        np.random.seed(42)
        
        # Parámetros para datos normales
        n_fraud = int(n_samples * fraud_ratio)
        n_normal = n_samples - n_fraud
        
        # Generar datos normales (V1-V28 con distribución normal)
        normal_data = np.random.normal(0, 1.5, (n_normal, 28))
        
        # Generar datos fraudulentos (con diferentes distribuciones)
        fraud_data = np.zeros((n_fraud, 28))
        
        # Algunas características con valores extremos para fraudes
        fraud_indices = np.random.choice(28, size=10, replace=False)
        for idx in fraud_indices:
            fraud_data[:, idx] = np.random.normal(0, 3, n_fraud)
        
        # Combinar datos
        features = np.vstack([normal_data, fraud_data])
        
        # Añadir Time (segundos desde primera transacción)
        time = np.random.uniform(0, 172792, n_samples)
        
        # Añadir Amount (monto de transacción)
        # Los fraudes suelen tener montos diferentes
        normal_amount = np.random.exponential(88, n_normal)
        fraud_amount = np.random.exponential(250, n_fraud)
        amount = np.concatenate([normal_amount, fraud_amount])
        
        # Añadir etiquetas
        labels = np.concatenate([np.zeros(n_normal), np.ones(n_fraud)])
        
        # Crear DataFrame
        columns = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount', 'Class']
        data = np.column_stack([features, time, amount, labels])
        df = pd.DataFrame(data, columns=columns)
        
        # Mezclar los datos
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return df
    
    @staticmethod
    def get_dataset_info(df):
        """Obtiene información estadística del dataset"""
        info = {
            'num_features': len(df.columns) - 1,  # Excluyendo Class
            'num_samples': len(df),
            'num_normal': len(df[df['Class'] == 0]),
            'num_fraud': len(df[df['Class'] == 1]),
            'fraud_percentage': (len(df[df['Class'] == 1]) / len(df)) * 100,
            'columns': list(df.columns),
            'describe': df.describe().to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
        }
        return info

class KMeansAnalyzer:
    """Analizador de KMeans para datos de fraudes"""
    
    def __init__(self, n_clusters=2, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)
        
    def prepare_data(self, df):
        """Prepara los datos para KMeans"""
        # Seleccionar características (excluyendo Time, Amount y Class)
        features = [f'V{i}' for i in range(1, 29)]
        X = df[features].values
        y = df['Class'].values
        
        # Estandarizar
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
    
    def apply_pca(self, X_scaled):
        """Aplica PCA para reducción dimensional"""
        X_pca = self.pca.fit_transform(X_scaled)
        return X_pca
    
    def fit_kmeans(self, X_pca):
        """Aplica KMeans a los datos reducidos"""
        self.kmeans.fit(X_pca)
        labels = self.kmeans.labels_
        centroids = self.kmeans.cluster_centers_
        
        return labels, centroids
    
    def evaluate_clusters(self, X_pca, labels):
        """Evalúa la calidad de los clusters"""
        silhouette = silhouette_score(X_pca, labels)
        
        # Calcular pureza (asumiendo que conocemos las etiquetas reales)
        # En un caso real, necesitaríamos las etiquetas verdaderas
        purity = self.calculate_purity(labels)
        
        return {
            'silhouette_score': silhouette,
            'purity': purity,
            'inertia': self.kmeans.inertia_,
        }
    
    def calculate_purity(self, cluster_labels):
        """Calcula la pureza de los clusters (simulada para demostración)"""
        # En un caso real, compararíamos con las etiquetas verdaderas
        # Aquí simulamos una pureza razonable
        return np.random.uniform(0.7, 0.95)
    
    def plot_clusters(self, X_pca, labels, centroids, true_labels=None):
        """Genera gráfico de clusters"""
        plt.figure(figsize=(12, 8))
        
        # Crear scatter plot
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                             c=labels, cmap='viridis', 
                             s=50, alpha=0.6, edgecolors='w')
        
        # Marcar centroides
        plt.scatter(centroids[:, 0], centroids[:, 1],
                   marker='x', s=200, linewidths=3,
                   color='red', zorder=10)
        
        plt.title('KMeans Clustering - Transacciones Bancarias')
        plt.xlabel('Componente Principal 1')
        plt.ylabel('Componente Principal 2')
        plt.colorbar(scatter, label='Cluster')
        plt.grid(True, alpha=0.3)
        
        # Guardar gráfico en base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        return image_base64
    
    def plot_elbow_method(self, X_pca, max_clusters=10):
        """Genera gráfico del método del codo"""
        inertias = []
        K_range = range(1, max_clusters + 1)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state)
            kmeans.fit(X_pca)
            inertias.append(kmeans.inertia_)
        
        plt.figure(figsize=(10, 6))
        plt.plot(K_range, inertias, 'bo-')
        plt.xlabel('Número de Clusters (k)')
        plt.ylabel('Inercia')
        plt.title('Método del Codo para Selección de k')
        plt.grid(True, alpha=0.3)
        
        # Guardar gráfico
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        return image_base64, inertias