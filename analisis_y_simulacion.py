# FASE 1 Y 2: CARGA, LIMPIEZA Y EDA COMPLETO

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix

# 1. Carga directa desde GitHub
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)

# 2. Limpieza e Ingeniería de Variables
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
df['Churn_Num'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Crear segmentos de antigüedad y cantidad de servicios contratados
df['Segmento_Antiguedad'] = pd.cut(df['tenure'], bins=[-1, 12, 24, 48, 72], 
                                   labels=['0-12 meses', '13-24 meses', '25-48 meses', '49-72 meses'])
servicios_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
df['Total_Servicios'] = (df[servicios_cols] == 'Yes').sum(axis=1)

print(f"Registros totales: {len(df)} | Tasa Churn Global: {df['Churn_Num'].mean():.2%}")

# 3. EDA Visual (Panel de 6 gráficos)
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(3, 2, figsize=(16, 15))

# G1: Tasa de Churn por Tipo de Contrato
sns.barplot(data=df, x='Contract', y='Churn_Num', ax=axes[0, 0], palette='viridis', errorbar=None)
axes[0, 0].set_title('1. Tasa de Churn por Tipo de Contrato', fontweight='bold')
axes[0, 0].set_ylabel('Proporción de Abandono')

# G2: Tasa de Churn por Servicio de Internet
sns.barplot(data=df, x='InternetService', y='Churn_Num', ax=axes[0, 1], palette='magma', errorbar=None)
axes[0, 1].set_title('2. Tasa de Churn por Servicio de Internet', fontweight='bold')
axes[0, 1].set_ylabel('Proporción de Abandono')

# G3: Distribución de Cargos Mensuales (KDE) según Churn
sns.kdeplot(data=df, x='MonthlyCharges', hue='Churn', fill=True, ax=axes[1, 0], palette={'Yes': '#ef4444', 'No': '#3b82f6'})
axes[1, 0].set_title('3. Densidad de Cargos Mensuales por Estado de Churn', fontweight='bold')

# G4: Tasa de Churn por Cohorte de Antigüedad
sns.barplot(data=df, x='Segmento_Antiguedad', y='Churn_Num', ax=axes[1, 1], palette='coolwarm', errorbar=None)
axes[1, 1].set_title('4. Abandono según Antigüedad del Cliente', fontweight='bold')
axes[1, 1].set_ylabel('Proporción de Abandono')

# G5: Impacto de Servicios Adicionales en la Retención
sns.lineplot(data=df, x='Total_Servicios', y='Churn_Num', marker='o', ax=axes[2, 0], color='#dc2626', linewidth=2.5)
axes[2, 0].set_title('5. Tasa de Churn vs. Cantidad de Servicios Contratados', fontweight='bold')
axes[2, 0].set_ylabel('Proporción de Abandono')

# G6: Mapa de Calor de Correlaciones Numéricas
corr_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Total_Servicios', 'SeniorCitizen', 'Churn_Num']
sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f", cmap='RdBu_r', center=0, ax=axes[2, 1])
axes[2, 1].set_title('6. Matriz de Correlación de Variables Clave', fontweight='bold')

plt.tight_layout()
plt.savefig('eda_completo.png', dpi=300)
plt.show()


# MODELADO PREDICTIVO Y VALIDACIÓN

features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Total_Servicios', 'Contract', 
            'InternetService', 'PaymentMethod', 'TechSupport', 'OnlineSecurity', 'PaperlessBilling']
X = pd.get_dummies(df[features], drop_first=True)
y = df['Churn_Num']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

modelo = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)
y_prob = modelo.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_prob)

print(f"\nROC-AUC en Prueba: {auc_score:.4f}")
print("\nReporte de Clasificación:\n", classification_report(y_test, y_pred))

# Visualización del Modelo: Curva ROC + Importancia de Variables
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[0].plot(fpr, tpr, label=f'Random Forest (AUC = {auc_score:.3f})', color='#2563eb', lw=2.5)
axes[0].plot([0, 1], [0, 1], 'k--', lw=1)
axes[0].set_title('Curva ROC del Modelo Predictivo', fontweight='bold')
axes[0].set_xlabel('Tasa de Falsos Positivos')
axes[0].set_ylabel('Tasa de Verdaderos Positivos')
axes[0].legend()

importancias = pd.Series(modelo.feature_importances_, index=X.columns).sort_values(ascending=True).tail(10)
importancias.plot(kind='barh', ax=axes[1], color='#10b981')
axes[1].set_title('Top 10 Variables Más Influyentes en el Churn', fontweight='bold')

plt.tight_layout()
plt.savefig('evaluacion_modelo.png', dpi=300)
plt.show()

# Exportar dataset para Dash
df['Prob_Churn'] = modelo.predict_proba(X)[:, 1]
df['CLV_Estimado'] = df['MonthlyCharges'] * df['tenure']
df.to_csv("telco_procesado.csv", index=False)


# FASE 4: SIMULACIÓN Y ANÁLISIS DE SENSIBILIDAD

alto_riesgo = df[df['Prob_Churn'] >= 0.60].copy()
clientes_riesgo = len(alto_riesgo)
ingreso_promedio = alto_riesgo['MonthlyCharges'].mean()

def simular_estrategia(tasa_retencion, descuento_pct, costo_operativo=15, meses=12):
    retenidos = clientes_riesgo * tasa_retencion
    ingresos_salvados = retenidos * (ingreso_promedio * (1 - descuento_pct)) * meses
    costo_total = (clientes_riesgo * costo_operativo) + (retenidos * ingreso_promedio * descuento_pct * meses)
    beneficio_neto = ingresos_salvados - costo_total
    roi = (beneficio_neto / costo_total) * 100
    return round(beneficio_neto, 2), round(roi, 2)

# Matriz de sensibilidad (Descuento vs. Tasa de retención)
tasas = [0.20, 0.30, 0.35, 0.45, 0.55]
descuentos = [0.05, 0.10, 0.15, 0.20, 0.25]
matriz_bn = pd.DataFrame(index=[f"Retención {int(t*100)}%" for t in tasas],
                         columns=[f"Dcto {int(d*100)}%" for d in descuentos], dtype=float)

for t in tasas:
    for d in descuentos:
        bn, _ = simular_estrategia(t, d)
        matriz_bn.loc[f"Retención {int(t*100)}%", f"Dcto {int(d*100)}%"] = bn

plt.figure(figsize=(10, 6))
sns.heatmap(matriz_bn, annot=True, fmt=",.0f", cmap="YlGnBu", cbar_kws={'label': 'Beneficio Neto Anual ($)'})
plt.title('Análisis de Sensibilidad: Beneficio Neto ($) según Retención y Descuento', fontweight='bold')
plt.tight_layout()
plt.savefig('sensibilidad_financiera.png', dpi=300)
plt.show()