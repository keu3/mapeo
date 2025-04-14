import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Cargar dataset
df = pd.read_csv("nombre_del_dataset.csv")

# Convertir DateTime a formato de fecha y hora
df['DateTime'] = pd.to_datetime(df['DateTime'])

# Crear nuevas columnas útiles a partir de la fecha
df['hour'] = df['DateTime'].dt.hour
df['dayofweek'] = df['DateTime'].dt.dayofweek
df['month'] = df['DateTime'].dt.month

# Eliminar columnas que no se usarán directamente
df = df.drop(['DateTime', 'ID'], axis=1)

# Convertir columna Junction a variables numéricas (One Hot Encoding)
df = pd.get_dummies(df, columns=['Junction'])

# Variables predictoras (X) y variable objetivo (y)
X = df.drop('Vehicles', axis=1)
y = df['Vehicles']

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenamiento con Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predicción y evaluación
y_pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))
