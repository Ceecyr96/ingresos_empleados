import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Cargar y preparar datos
df = pd.read_csv('data.csv')
X = df[['feature1', 'feature2', 'feature3']]  # Seleccionar características relevantes

# Normalizar datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Resultados
print(f'Varianza Explicada por Componentes: {pca.explained_variance_ratio_}')

