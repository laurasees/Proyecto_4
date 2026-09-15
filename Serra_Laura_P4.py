# %%
%pip install matplotlib numpy pandas openpyxl seaborn --break-system-packages
#%%
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# %% 
# ### 1. Carga de Datos desde el Escritorio, Limpieza de Cabeceras y Unión

# %%
import os
import pandas as pd

# 1. Definir rutas al Escritorio
desktop_path = os.path.expanduser('~/Desktop')
csv_file = os.path.join(desktop_path, 'bank-additional.csv')
excel_file = os.path.join(desktop_path, 'customer-details.xlsx')

# 2. Cargar CSV detectando el separador automáticamente
df_bank = pd.read_csv(csv_file, sep=None, engine='python')

# Limpiar nombres de columnas (eliminar comillas simples, dobles y espacios adicionales)
df_bank.columns = df_bank.columns.str.replace('"', '').str.replace("'", '').str.strip()

print("--- Columnas en df_bank tras limpiar cabeceras ---")
print(list(df_bank.columns))

# 3. Cargar y combinar las 3 hojas del Excel
xls = pd.ExcelFile(excel_file)
sheets_list = []
for s in xls.sheet_names:
    df_temp = pd.read_excel(xls, sheet_name=s)
    sheets_list.append(df_temp)

df_customers = pd.concat(sheets_list, ignore_index=True)

# Limpiar nombres de columnas en el DataFrame de clientes
df_customers.columns = df_customers.columns.str.replace('"', '').str.replace("'", '').str.strip()

print("\n--- Columnas en df_customers tras limpiar cabeceras ---")
print(list(df_customers.columns))

# 4. Unir ambos DataFrames usando 'id_' e 'ID'
df_merged = pd.merge(
    df_bank, 
    df_customers, 
    left_on='id_', 
    right_on='ID', 
    how='inner'
)

print("\n--- Dataframe Final Unificado ---")
print(f"Dimensiones tras la unión (merge): {df_merged.shape}")
df_merged.head(3)






# %% [markdown]
# ### 2. Limpieza, Transformación y Feature Engineering (Corregido)

# %%
# 1. Eliminar columnas redundantes
cols_to_drop = ['Unnamed: 0_x', 'Unnamed: 0_y', 'ID']
df_clean = df_merged.drop(columns=[c for c in cols_to_drop if c in df_merged.columns]).copy()

# 2. Manejo de valores nulos (Imputación de Income con la mediana)
df_clean['Income'] = df_clean['Income'].fillna(df_clean['Income'].median())

# Forzar tipo numérico en columnas binarias (default, housing, loan)
for col in ['default', 'housing', 'loan']:
    if col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0).astype(int)

# 3. Conversión de formatos de fecha
df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')
df_clean['Dt_Customer'] = pd.to_datetime(df_clean['Dt_Customer'], errors='coerce')

# 4. Estandarización de la variable objetivo 'y' -> 'y_binary' (1 = Sí, 0 = No)
# Mapeo universal que convierte cadenas 'yes'/'no', números o flotantes
mapping = {'yes': 1, 'no': 0, '1': 1, '0': 0, 1: 1, 0: 0, 1.0: 1, 0.0: 0}
df_clean['y_binary'] = df_clean['y'].astype(str).str.lower().str.strip().map(mapping).fillna(0).astype(int)

# 5. Feature Engineering
# A) Total de menores en el hogar
df_clean['total_children'] = df_clean['Kidhome'].fillna(0) + df_clean['Teenhome'].fillna(0)

# B) Antigüedad del cliente en años
ref_date = pd.to_datetime('2015-01-01')
df_clean['customer_seniority_years'] = (ref_date - df_clean['Dt_Customer']).dt.days // 365

print("--- Dataframe Limpio y Transformado ---")
print(f"Dimensiones finales: {df_clean.shape}")
print("\nConteo de valores nulos pendientes:")
print(df_clean.isnull().sum()[df_clean.isnull().sum() > 0])
print("\nDistribución de la variable objetivo (y_binary):")
print(df_clean['y_binary'].value_counts(normalize=True) * 100)





# %% 
# ### 2.1 Ajuste de Formatos y Tratamiento de Nulos Restantes 

# %%
# 1. Asegurar que las columnas numéricas sean de tipo float/int (convirtiendo comas a puntos si las hay)
num_target_cols = ['age', 'euribor3m', 'cons.price.idx']

for col in num_target_cols:
    if col in df_clean.columns:
        if df_clean[col].dtype == object:
            df_clean[col] = df_clean[col].astype(str).str.replace(',', '.')
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

# 2. Imputación de variables numéricas con la mediana
df_clean['age'] = df_clean['age'].fillna(df_clean['age'].median())
df_clean['euribor3m'] = df_clean['euribor3m'].fillna(df_clean['euribor3m'].median())
df_clean['cons.price.idx'] = df_clean['cons.price.idx'].fillna(df_clean['cons.price.idx'].median())

# 3. Imputación de variables categóricas
df_clean['job'] = df_clean['job'].fillna(df_clean['job'].mode()[0])
df_clean['marital'] = df_clean['marital'].fillna(df_clean['marital'].mode()[0])
df_clean['education'] = df_clean['education'].fillna('unknown')

print("--- Verificación Final de Limpieza ---")
print("Nulos restantes en el dataset:")
print(df_clean[['age', 'euribor3m', 'cons.price.idx', 'job', 'marital', 'education']].isnull().sum())


# %% 
# ### 2.2 Corrección de cons.price.idx e Imputación Final

# %%
# Restaurar cons.price.idx convirtiendo formatos de coma/texto si aplica
if 'cons.price.idx' in df_merged.columns:
    df_clean['cons.price.idx'] = pd.to_numeric(
        df_merged['cons.price.idx'].astype(str).str.replace(',', '.'), 
        errors='coerce'
    )
    df_clean['cons.price.idx'] = df_clean['cons.price.idx'].fillna(df_clean['cons.price.idx'].median())

print("Nulos en cons.price.idx:", df_clean['cons.price.idx'].isnull().sum())



# %% 
# ### 3. Análisis Descriptivo y Métricas Estadísticas

# %%
# 1. Resumen Estadístico de Variables Numéricas
num_cols = ['age', 'Income', 'duration', 'campaign', 'pdays', 'euribor3m', 'total_children']
num_cols_valid = [c for c in num_cols if c in df_clean.columns]

print("=== RESUMEN ESTADÍSTICO DE VARIABLES NUMÉRICAS ===")
stats_df = df_clean[num_cols_valid].describe().T[['mean', 'std', '50%', 'min', 'max']]
stats_df.columns = ['Media', 'Desv. Estándar', 'Mediana', 'Mínimo', 'Máximo']
print(stats_df.round(2))

# 2. Tasa de Conversión por Ocupación (Job)
print("\n=== TASA DE CONVERSIÓN POR OCUPACIÓN ===")
job_conv = df_clean.groupby('job')['y_binary'].agg(
    Total='count', 
    Suscripciones='sum', 
    Tasa_Conversion='mean'
).sort_values(by='Tasa_Conversion', ascending=False)
job_conv['Tasa_Conversion (%)'] = (job_conv['Tasa_Conversion'] * 100).round(2)
print(job_conv[['Total', 'Suscripciones', 'Tasa_Conversion (%)']])



# %% 
# ### 4. Visualización Exploratoria de Datos (EDA)

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar el estilo visual general
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Distribución de Edad por Respuesta a la Campaña
sns.boxplot(
    data=df_clean, x='y_binary', y='age', hue='y_binary', palette='Set2', legend=False, ax=axes[0, 0]
)
axes[0, 0].set_title('Distribución de Edad según Suscripción del Depósito', fontsize=13, fontweight='bold')
axes[0, 0].set_xticklabels(['No Suscrito (0)', 'Suscrito (1)'])
axes[0, 0].set_xlabel('Resultado de la Campaña')
axes[0, 0].set_ylabel('Edad del Cliente')

# 2. Tasa de Conversión por Ocupación (Job)
job_order = df_clean.groupby('job')['y_binary'].mean().sort_values(ascending=False).index
sns.barplot(
    data=df_clean, x='y_binary', y='job', order=job_order, errorbar=None, palette='viridis', hue='job', legend=False, ax=axes[0, 1]
)
axes[0, 1].set_title('Tasa Promedio de Conversión por Tipo de Trabajo', fontsize=13, fontweight='bold')
axes[0, 1].set_xlabel('Tasa de Conversión (Proporción)')
axes[0, 1].set_ylabel('Ocupación')

# 3. Relación de la Duración de Llamada con el Éxito de Conversión
sns.kdeplot(
    data=df_clean, x='duration', hue='y_binary', common_norm=False, fill=True, palette='tab10', ax=axes[1, 0]
)
axes[1, 0].set_xlim(0, 1500)
axes[1, 0].set_title('Densidad de Duración del Contacto Telefónico (Segundos)', fontsize=13, fontweight='bold')
axes[1, 0].set_xlabel('Duración de la Llamada (s)')
axes[1, 0].set_ylabel('Densidad')

# 4. Contexto Macroeconómico: Euribor 3 Meses vs Suscripción
sns.histplot(
    data=df_clean, x='euribor3m', hue='y_binary', multiple='stack', bins=25, palette='crest', ax=axes[1, 1]
)
axes[1, 1].set_title('Distribución del Euribor 3M y Contratación del Producto', fontsize=13, fontweight='bold')
axes[1, 1].set_xlabel('Euribor 3 Meses (%)')
axes[1, 1].set_ylabel('Cantidad de Contactos')

plt.tight_layout()
plt.show()
# %%
