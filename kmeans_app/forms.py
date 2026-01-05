from django import forms

class KMeansForm(forms.Form):
    n_clusters = forms.IntegerField(
        label='Número de Clusters (k)',
        min_value=2,
        max_value=20,
        initial=2,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    
    max_clusters = forms.IntegerField(
        label='Máximo k para método del codo',
        min_value=3,
        max_value=20,
        initial=10,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    
    random_state = forms.IntegerField(
        label='Semilla aleatoria',
        min_value=0,
        initial=42,
        required=False,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )