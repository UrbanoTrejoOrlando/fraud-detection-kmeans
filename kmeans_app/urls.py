from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('dataset-info/', views.dataset_info, name='dataset_info'),
    path('visualizations/', views.visualizations, name='visualizations'),
    path('kmeans-analysis/', views.kmeans_analysis, name='kmeans_analysis'),
    path('generate-data/', views.generate_simulated_data, name='generate_data'),
]