import dash
from dash import dcc, html, dash_table, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Cargar datos procesados desde el notebook
df = pd.read_csv("telco_procesado.csv")

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div(style={'fontFamily': 'Segoe UI, Arial, sans-serif', 'backgroundColor': '#f3f4f6', 'padding': '20px'}, children=[
    
    # Encabezado Principal
    html.Div([
        html.H1("Churn y optimización de ingresos recurrentes en telecomunicaciones", style={'color': '#111827', 'marginBottom': '4px'}),
        html.P("Diagnóstico descriptivo, scoring predictivo de abandono y simulación financiera de retención", style={'color': '#4b5563', 'marginTop': 0})
    ], style={'backgroundColor': 'white', 'padding': '18px', 'borderRadius': '8px', 'marginBottom': '15px', 'boxShadow': '0 1px 3px rgba(0,0,0,0.1)'}),

    # Panel Global de Filtros
    html.Div([
        html.Div([
            html.Label("Tipo de contrato:", style={'fontWeight': 'bold', 'fontSize': '13px'}),
            dcc.Dropdown(id='filtro-contrato',
                         options=[{'label': 'Todos los contratos', 'value': 'Todos'}] + [{'label': c, 'value': c} for c in df['Contract'].unique()],
                         value='Todos', clearable=False)
        ], style={'width': '24%', 'display': 'inline-block', 'marginRight': '1.3%'}),

        html.Div([
            html.Label("Servicio de internet:", style={'fontWeight': 'bold', 'fontSize': '13px'}),
            dcc.Dropdown(id='filtro-internet',
                         options=[{'label': 'Todos los servicios', 'value': 'Todos'}] + [{'label': i, 'value': i} for i in df['InternetService'].unique()],
                         value='Todos', clearable=False)
        ], style={'width': '24%', 'display': 'inline-block', 'marginRight': '1.3%'}),

        html.Div([
            html.Label("Cohorte de antigüedad:", style={'fontWeight': 'bold', 'fontSize': '13px'}),
            dcc.Dropdown(id='filtro-cohorte',
                         options=[{'label': 'Todas las cohortes', 'value': 'Todos'}] + [{'label': s, 'value': s} for s in df['Segmento_Antiguedad'].dropna().unique()],
                         value='Todos', clearable=False)
        ], style={'width': '24%', 'display': 'inline-block', 'marginRight': '1.3%'}),

        html.Div([
            html.Label("Umbral riesgo crítico (Prob. Churn):", style={'fontWeight': 'bold', 'fontSize': '13px'}),
            dcc.Slider(id='slider-riesgo', min=0.3, max=0.9, step=0.1, value=0.6,
                       marks={i/10: f"{int(i*10)}%" for i in range(3, 10)})
        ], style={'width': '24%', 'display': 'inline-block', 'verticalAlign': 'top'})
    ], style={'backgroundColor': 'white', 'padding': '15px', 'borderRadius': '8px', 'marginBottom': '15px', 'boxShadow': '0 1px 3px rgba(0,0,0,0.1)'}),

    # Fila de 5 Tarjetas KPI
    html.Div(id='kpi-container', style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '15px'}),

    # Navegación por Pestañas (Narrativa Visual)
    dcc.Tabs(id='tabs-principales', value='tab-1', children=[
        
        # PESTAÑA 1: DIAGNÓSTICO EJECUTIVO
        dcc.Tab(label='1. Diagnóstico ejecutivo (EDA)', value='tab-1', children=[
            html.Div([
                html.Div([dcc.Graph(id='g-contrato')], style={'width': '49.5%', 'display': 'inline-block', 'backgroundColor': 'white', 'borderRadius': '8px', 'marginTop': '15px'}),
                html.Div([dcc.Graph(id='g-pago')], style={'width': '49.5%', 'display': 'inline-block', 'float': 'right', 'backgroundColor': 'white', 'borderRadius': '8px', 'marginTop': '15px'}),
                html.Div([dcc.Graph(id='g-antiguedad')], style={'width': '100%', 'backgroundColor': 'white', 'borderRadius': '8px', 'marginTop': '15px'})
            ])
        ]),

        # PESTAÑA 2: SEGMENTACIÓN Y RIESGO PREDICTIVO
        dcc.Tab(label='2. Perfil de riesgo y clientes críticos', value='tab-2', children=[
            html.Div([
                html.Div([dcc.Graph(id='g-scatter')], style={'width': '49.5%', 'display': 'inline-block', 'backgroundColor': 'white', 'borderRadius': '8px', 'marginTop': '15px'}),
                html.Div([dcc.Graph(id='g-boxplot')], style={'width': '49.5%', 'display': 'inline-block', 'float': 'right', 'backgroundColor': 'white', 'borderRadius': '8px', 'marginTop': '15px'}),
                html.Div([
                    html.H4("Top 15 clientes en riesgo crítico priorizados por ingreso mensual", style={'padding': '15px 15px 5px 15px', 'margin': 0}),
                    html.Div(id='tabla-criticos', style={'padding': '15px'})
                ], style={'backgroundColor': 'white', 'borderRadius': '8px', 'marginTop': '15px'})
            ])
        ]),

        # PESTAÑA 3: SIMULADOR DE DECISIONES
        dcc.Tab(label='3. Simulador financiero y sensibilidad', value='tab-3', children=[
            html.Div([
                html.Div([
                    html.H4("Parámetros de la estrategia de retención", style={'marginTop': 0}),
                    html.Label("Descuento Mensual Ofrecido (%):"),
                    dcc.Slider(id='sim-dcto', min=0, max=30, step=5, value=10, marks={i: f"{i}%" for i in range(0, 35, 5)}),
                    html.Br(),
                    html.Label("Tasa de retención esperada (%):"),
                    dcc.Slider(id='sim-ret', min=10, max=60, step=5, value=35, marks={i: f"{i}%" for i in range(10, 65, 10)}),
                    html.Br(),
                    html.Label("Costo operativo por cliente contactado ($):"),
                    dcc.Slider(id='sim-costo', min=5, max=40, step=5, value=15, marks={i: f"${i}" for i in range(5, 45, 5)}),
                    html.Div(id='resumen-decision', style={'marginTop': '20px', 'padding': '12px', 'backgroundColor': '#eff6ff', 'borderLeft': '4px solid #2563eb', 'borderRadius': '4px'})
                ], style={'width': '34%', 'display': 'inline-block', 'verticalAlign': 'top', 'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '8px', 'marginTop': '15px'}),

                html.Div([
                    html.Div([dcc.Graph(id='g-sim-barras')], style={'backgroundColor': 'white', 'borderRadius': '8px', 'marginBottom': '15px'}),
                    html.Div([dcc.Graph(id='g-sim-sensibilidad')], style={'backgroundColor': 'white', 'borderRadius': '8px'})
                ], style={'width': '64%', 'display': 'inline-block', 'float': 'right', 'marginTop': '15px'})
            ])
        ])
    ])
])

@app.callback(
    [Output('kpi-container', 'children'),
     Output('g-contrato', 'figure'),
     Output('g-pago', 'figure'),
     Output('g-antiguedad', 'figure'),
     Output('g-scatter', 'figure'),
     Output('g-boxplot', 'figure'),
     Output('tabla-criticos', 'children'),
     Output('g-sim-barras', 'figure'),
     Output('g-sim-sensibilidad', 'figure'),
     Output('resumen-decision', 'children')],
    [Input('filtro-contrato', 'value'),
     Input('filtro-internet', 'value'),
     Input('filtro-cohorte', 'value'),
     Input('slider-riesgo', 'value'),
     Input('sim-dcto', 'value'),
     Input('sim-ret', 'value'),
     Input('sim-costo', 'value')]
)
def actualizar_todo(contrato, internet, cohorte, umbral, dcto_pct, ret_pct, costo_op):
    dff = df.copy()
    if contrato != 'Todos':
        dff = dff[dff['Contract'] == contrato]
    if internet != 'Todos':
        dff = dff[dff['InternetService'] == internet]
    if cohorte != 'Todos':
        dff = dff[dff['Segmento_Antiguedad'] == cohorte]

    # KPIs
    n_total = len(dff)
    churn_rate = (dff['Churn_Num'].mean() * 100) if n_total > 0 else 0
    en_riesgo = dff[dff['Prob_Churn'] >= umbral]
    n_riesgo = len(en_riesgo)
    mrr_riesgo = en_riesgo['MonthlyCharges'].sum()
    clv_prom = dff['CLV_Estimado'].mean() if n_total > 0 else 0

    card_s = {'backgroundColor': 'white', 'padding': '14px', 'borderRadius': '8px', 'width': '18.8%', 'textAlign': 'center', 'boxShadow': '0 1px 3px rgba(0,0,0,0.1)'}
    kpis = [
        html.Div([html.Div("Total Clientes", style={'color': '#6b7280', 'fontSize': '13px'}), html.H2(f"{n_total:,}", style={'margin': '5px 0'})], style=card_s),
        html.Div([html.Div("Tasa Churn Histórica", style={'color': '#6b7280', 'fontSize': '13px'}), html.H2(f"{churn_rate:.1f}%", style={'margin': '5px 0', 'color': '#dc2626'})], style=card_s),
        html.Div([html.Div(f"Clientes Riesgo (≥{int(umbral*100)}%)", style={'color': '#6b7280', 'fontSize': '13px'}), html.H2(f"{n_riesgo:,}", style={'margin': '5px 0', 'color': '#d97706'})], style=card_s),
        html.Div([html.Div("MRR Mensual en Riesgo", style={'color': '#6b7280', 'fontSize': '13px'}), html.H2(f"${mrr_riesgo:,.0f}", style={'margin': '5px 0', 'color': '#2563eb'})], style=card_s),
        html.Div([html.Div("CLV Promedio", style={'color': '#6b7280', 'fontSize': '13px'}), html.H2(f"${clv_prom:,.0f}", style={'margin': '5px 0', 'color': '#059669'})], style=card_s),
    ]

    # Pestaña 1: Gráficos EDA
    g1 = px.histogram(dff, x='Contract', color='Churn', barmode='group', title="Distribución de abandono por tipo de contrato",
                      color_discrete_map={'Yes': '#ef4444', 'No': '#3b82f6'})
    g1.update_layout(height=320, margin=dict(t=40, b=20, l=20, r=20))

    g2 = px.histogram(dff, x='PaymentMethod', color='Churn', barmode='group', title="Abandono por Método de Pago",
                      color_discrete_map={'Yes': '#ef4444', 'No': '#3b82f6'})
    g2.update_layout(height=320, margin=dict(t=40, b=20, l=20, r=20))

    g3 = px.histogram(dff, x='tenure', color='Churn', nbins=36, barmode='overlay', opacity=0.75,
                      title="Curva de Supervivencia: Distribución de Antigüedad (Meses)",
                      color_discrete_map={'Yes': '#ef4444', 'No': '#10b981'})
    g3.update_layout(height=300, margin=dict(t=40, b=20, l=20, r=20))

    # Pestaña 2: Segmentación y Tabla
    muestra = dff.sample(min(1000, len(dff)), random_state=42) if len(dff) > 0 else dff
    g4 = px.scatter(muestra, x='tenure', y='MonthlyCharges', color='Prob_Churn', color_continuous_scale='RdYlGn_r',
                    title="Mapa de riesgo predictivo: antigüedad vs. cargo Mensual")
    g4.update_layout(height=340, margin=dict(t=40, b=20, l=20, r=20))

    g5 = px.box(dff, x='InternetService', y='MonthlyCharges', color='Churn',
                title="Dispersión de facturación mensual por tipo de internet",
                color_discrete_map={'Yes': '#ef4444', 'No': '#3b82f6'})
    g5.update_layout(height=340, margin=dict(t=40, b=20, l=20, r=20))

    top_df = en_riesgo.sort_values(by=['Prob_Churn', 'MonthlyCharges'], ascending=False)[
        ['customerID', 'Contract', 'InternetService', 'tenure', 'MonthlyCharges', 'Prob_Churn']
    ].head(15).copy()
    top_df['Prob_Churn'] = (top_df['Prob_Churn'] * 100).round(1).astype(str) + '%'
    top_df['MonthlyCharges'] = '$' + top_df['MonthlyCharges'].round(2).astype(str)

    tabla = dash_table.DataTable(
        data=top_df.to_dict('records'),
        columns=[{'name': i, 'id': i} for i in top_df.columns],
        style_cell={'textAlign': 'center', 'padding': '8px', 'fontSize': '13px'},
        style_header={'backgroundColor': '#1f2937', 'color': 'white', 'fontWeight': 'bold'}
    )

    # Pestaña 3: Simulación y Sensibilidad
    ing_prom = en_riesgo['MonthlyCharges'].mean() if n_riesgo > 0 else 0
    perdida_base = -(mrr_riesgo * 12)
    retenidos = n_riesgo * (ret_pct / 100)
    ing_salvados = retenidos * (ing_prom * (1 - dcto_pct / 100)) * 12
    costo_camp = (n_riesgo * costo_op) + (retenidos * ing_prom * (dcto_pct / 100) * 12)
    bn_actual = ing_salvados - costo_camp
    roi_actual = (bn_actual / costo_camp * 100) if costo_camp > 0 else 0

    df_sim = pd.DataFrame({
        'Concepto': ['Pérdida Base (Sin Acción)', 'Ingresos Brutos Salvados', 'Costo Campaña + Descuento', 'Beneficio Neto Anual'],
        'USD': [perdida_base, ing_salvados, -costo_camp, bn_actual]
    })
    g6 = px.bar(df_sim, x='Concepto', y='USD', text_auto='.2s', color='Concepto',
                title=f"Impacto Financiero Anual (ROI Estimado: {roi_actual:.1f}%)",
                color_discrete_sequence=['#ef4444', '#3b82f6', '#f59e0b', '#10b981'])
    g6.update_layout(showlegend=False, height=280, margin=dict(t=40, b=20, l=20, r=20))

    # Curva de sensibilidad comparando 3 niveles de descuento
    tasas_grid = np.linspace(0.10, 0.60, 11)
    sens_rows = []
    for d_alt in [5, 10, 20]:
        for t_alt in tasas_grid:
            ret_a = n_riesgo * t_alt
            ing_a = ret_a * (ing_prom * (1 - d_alt / 100)) * 12
            cost_a = (n_riesgo * costo_op) + (ret_a * ing_prom * (d_alt / 100) * 12)
            sens_rows.append({'Tasa Retención (%)': t_alt * 100, 'Beneficio Neto ($)': ing_a - cost_a, 'Estrategia': f'Descuento {d_alt}%'})

    g7 = px.line(pd.DataFrame(sens_rows), x='Tasa Retención (%)', y='Beneficio Neto ($)', color='Estrategia', markers=True,
                 title="Análisis de Sensibilidad: Beneficio Neto según Efectividad de Retención")
    g7.add_hline(y=0, line_dash="dash", line_color="gray")
    g7.update_layout(height=280, margin=dict(t=40, b=20, l=20, r=20))

    resumen = html.Div([
        html.Strong("Diagnóstico de Viabilidad:"),
        html.P(f"Interviniendo {n_riesgo:,} clientes críticos con {dcto_pct}% de descuento, se proyecta retener a {int(retenidos):,} clientes, generando un beneficio neto de ${bn_actual:,.2f} (ROI: {roi_actual:.1f}%).", style={'marginBottom': 0, 'fontSize': '13px'})
    ])

    return kpis, g1, g2, g3, g4, g5, tabla, g6, g7, resumen

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8050)