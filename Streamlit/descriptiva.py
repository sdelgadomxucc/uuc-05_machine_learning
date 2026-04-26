# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import io

# Configuración de la página
st.set_page_config(
    page_title="Análisis Estadístico Descriptivo",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e9ecef;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

def load_sample_data(dataset_choice):
    """Carga datos de ejemplo según la selección"""
    if dataset_choice == "Iris":
        df = sns.load_dataset('iris')
    elif dataset_choice == "Titanic":
        df = sns.load_dataset('titanic')
    elif dataset_choice == "Diamantes":
        df = sns.load_dataset('diamonds').sample(1000)
    elif dataset_choice == "Ventas":
        # Datos de ventas simulados
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        df = pd.DataFrame({
            'fecha': np.random.choice(dates, 500),
            'producto': np.random.choice(['Producto A', 'Producto B', 'Producto C'], 500),
            'categoria': np.random.choice(['Electrónicos', 'Ropa', 'Hogar'], 500),
            'ventas': np.random.normal(1000, 300, 500),
            'precio': np.random.normal(50, 15, 500),
            'cantidad': np.random.randint(1, 100, 500),
            'rating': np.random.uniform(1, 5, 500)
        })
    else:
        # Datos aleatorios generales
        np.random.seed(42)
        df = pd.DataFrame({
            'edad': np.random.normal(35, 10, 1000),
            'ingresos': np.random.normal(50000, 15000, 1000),
            'satisfaccion': np.random.randint(1, 11, 1000),
            'categoria': np.random.choice(['A', 'B', 'C'], 1000),
            'puntuacion': np.random.normal(75, 15, 1000),
            'horas_estudio': np.random.normal(20, 5, 1000),
            'calificacion': np.random.normal(80, 10, 1000)
        })
    return df

def display_data_preview(df):
    """Muestra vista previa e información del dataset"""
    st.markdown("### Vista Previa de los Datos")
    
    # Métricas rápidas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Filas", df.shape[0])
    with col2:
        st.metric("📈 Columnas", df.shape[1])
    with col3:
        st.metric("🔍 Valores Nulos", df.isnull().sum().sum())
    with col4:
        st.metric("💾 Memoria", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Pestañas para exploración detallada
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Primeras Filas", "📊 Últimas Filas", "🔍 Info", "📈 Descripción"])
    
    with tab1:
        st.dataframe(df.head(10), use_container_width=True)
        
    with tab2:
        st.dataframe(df.tail(10), use_container_width=True)
        
    with tab3:
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())
        
    with tab4:
        st.dataframe(df.describe(), use_container_width=True)

def display_descriptive_stats(df, selected_columns):
    """Muestra estadísticas descriptivas para las columnas seleccionadas"""
    st.markdown("### 📋 Resumen Estadístico Completo")
    
    # Calcular estadísticas extendidas
    stats_df = df[selected_columns].describe().T
    stats_df['varianza'] = df[selected_columns].var()
    
    # Calcular moda de forma segura
    mode_values = []
    for col in selected_columns:
        mode_result = df[col].mode()
        if not mode_result.empty:
            mode_values.append(mode_result.iloc[0])
        else:
            mode_values.append(np.nan)
    
    stats_df['moda'] = mode_values
    stats_df['asimetría'] = df[selected_columns].skew()
    stats_df['curtosis'] = df[selected_columns].kurtosis()
    stats_df['rango'] = df[selected_columns].max() - df[selected_columns].min()
    stats_df['rango_intercuartil'] = stats_df['75%'] - stats_df['25%']
    
    # Formatear números
    formatted_stats = stats_df.round(3)
    st.dataframe(formatted_stats, use_container_width=True)
    
    # Análisis detallado por variable
    st.markdown("### 🔍 Análisis Detallado por Variable")
    
    for col in selected_columns:
        with st.expander(f"📊 Análisis Detallado de **{col}**", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📐 Media", f"{df[col].mean():.2f}")
                st.metric("🎯 Mediana", f"{df[col].median():.2f}")
                mode_val = df[col].mode()
                st.metric("⭐ Moda", f"{mode_val.iloc[0] if not mode_val.empty else 'N/A'}")
                
            with col2:
                st.metric("📏 Desv. Estándar", f"{df[col].std():.2f}")
                st.metric("📊 Varianza", f"{df[col].var():.2f}")
                st.metric("📐 Rango", f"{df[col].max() - df[col].min():.2f}")
                
            with col3:
                st.metric("📉 Mínimo", f"{df[col].min():.2f}")
                st.metric("📈 Máximo", f"{df[col].max():.2f}")
                st.metric("🎯 Rango Intercuartil", f"{df[col].quantile(0.75) - df[col].quantile(0.25):.2f}")
                
            with col4:
                skew_val = df[col].skew()
                kurt_val = df[col].kurtosis()
                st.metric("↔️ Asimetría", f"{skew_val:.2f}")
                st.metric("📊 Curtosis", f"{kurt_val:.2f}")
                st.metric("🔢 Count", f"{df[col].count()}")
            
            # Interpretación de asimetría y curtosis
            interpret_skew = "Simétrica" if abs(skew_val) < 0.5 else "Sesgo derecho" if skew_val > 0 else "Sesgo izquierdo"
            interpret_kurt = "Mesocúrtica (normal)" if abs(kurt_val) < 1 else "Leptocúrtica (picuda)" if kurt_val > 0 else "Platicúrtica (plana)"
            
            st.markdown(f"""
            <div class="info-box">
            <strong>📝 Interpretación de {col}:</strong><br>
            • <strong>Asimetría ({skew_val:.2f}):</strong> {interpret_skew}<br>
            • <strong>Curtosis ({kurt_val:.2f}):</strong> {interpret_kurt}<br>
            • <strong>Forma:</strong> La distribución presenta {interpret_skew.lower()} y es {interpret_kurt.lower()}
            </div>
            """, unsafe_allow_html=True)

def create_histogram(df, column, bins):
    """Crea un histograma interactivo"""
    fig = px.histogram(
        df, 
        x=column, 
        nbins=bins,
        title=f"📈 Histograma de {column}",
        template="plotly_white",
        color_discrete_sequence=['#1f77b4'],
        opacity=0.8
    )
    fig.update_layout(
        xaxis_title=column,
        yaxis_title="Frecuencia",
        showlegend=False
    )
    return fig

def create_density_plot(df, column):
    """Crea un gráfico de densidad"""
    fig = px.histogram(
        df, 
        x=column, 
        marginal="rug",
        hover_data=df.columns,
        title=f"📊 Distribución de Densidad de {column}",
        color_discrete_sequence=['#2e86ab'],
        opacity=0.7
    )
    fig.update_layout(
        xaxis_title=column,
        yaxis_title="Densidad"
    )
    return fig

def create_box_plot(df, numeric_col, categorical_col=None):
    """Crea diagrama de caja"""
    if categorical_col:
        fig = px.box(
            df, 
            x=categorical_col, 
            y=numeric_col,
            title=f"📦 Diagrama de Caja de {numeric_col} por {categorical_col}",
            color=categorical_col
        )
    else:
        fig = px.box(
            df, 
            y=numeric_col,
            title=f"📦 Diagrama de Caja de {numeric_col}",
            color_discrete_sequence=['#ff7f0e']
        )
    return fig

def create_scatter_plot(df, x_col, y_col, color_col=None, trendline=False):
    """Crea gráfico de dispersión"""
    fig = px.scatter(
        df, 
        x=x_col, 
        y=y_col, 
        color=color_col if color_col else None,
        title=f"🔵 Dispersión: {x_col} vs {y_col}",
        trendline="ols" if trendline else None,
        opacity=0.6
    )
    return fig

def create_bar_plot(df, column):
    """Crea gráfico de barras para variables categóricas"""
    value_counts = df[column].value_counts().reset_index()
    value_counts.columns = [column, 'count']
    
    fig = px.bar(
        value_counts, 
        x=column, 
        y='count',
        title=f"📊 Distribución de {column}",
        color=column,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.update_layout(showlegend=False)
    return fig

def create_correlation_heatmap(df, columns):
    """Crea heatmap de correlación"""
    corr_matrix = df[columns].corr()
    
    fig = px.imshow(
        corr_matrix,
        title="🔥 Matriz de Correlación",
        color_continuous_scale="RdBu_r",
        aspect="auto",
        text_auto=True
    )
    fig.update_layout(
        xaxis_title="Variables",
        yaxis_title="Variables"
    )
    return fig

def create_line_plot(df, x_col, y_col, color_col=None):
    """Crea gráfico de líneas"""
    if color_col:
        fig = px.line(
            df, 
            x=x_col, 
            y=y_col, 
            color=color_col,
            title=f"📈 Tendencia de {y_col} por {x_col}"
        )
    else:
        fig = px.line(
            df, 
            x=x_col, 
            y=y_col,
            title=f"📈 Tendencia de {y_col} por {x_col}"
        )
    return fig

def main():
    """Función principal de la aplicación"""
    
    # Título principal
    st.markdown('<h1 class="main-header">📊 Dashboard de Análisis Estadístico Descriptivo</h1>', unsafe_allow_html=True)
    
    # Sidebar para navegación
    st.sidebar.title("🧭 Navegación")
    section = st.sidebar.radio(
        "Selecciona una sección:",
        ["🏠 Inicio", "📁 Carga de Datos", "📈 Estadística Descriptiva", "📊 Visualizaciones", "📋 Reporte Completo"]
    )
    
    # Sección de Inicio
    if section == "🏠 Inicio":
        display_home_section()
    
    # Sección de Carga de Datos
    elif section == "📁 Carga de Datos":
        display_data_loading_section()
    
    # Sección de Estadística Descriptiva
    elif section == "📈 Estadística Descriptiva":
        display_statistics_section()
    
    # Sección de Visualizaciones
    elif section == "📊 Visualizaciones":
        display_visualizations_section()
    
    # Sección de Reporte Completo
    elif section == "📋 Reporte Completo":
        display_report_section()
    
    # Pie de página
    display_footer()

def display_home_section():
    """Muestra la sección de inicio"""
    st.markdown("""
    ## 🏠 Bienvenido al Dashboard de Análisis Estadístico
    
    Esta aplicación interactiva te permite realizar análisis estadísticos descriptivos completos 
    y crear visualizaciones profesionales de tus datos.
    
    ### 🎯 **Funcionalidades Principales:**
    
    #### 📁 **Carga de Datos**
    - Carga archivos CSV, Excel o usa datasets de ejemplo
    - Vista previa y exploración rápida de los datos
    - Información detallada del dataset
    
    #### 📈 **Estadística Descriptiva**
    - Medidas de tendencia central (media, mediana, moda)
    - Medidas de dispersión (desviación estándar, varianza, rango)
    - Medidas de forma (asimetría, curtosis)
    - Análisis detallado por variable
    
    #### 📊 **Visualizaciones Interactivas**
    - Histogramas y gráficos de densidad
    - Diagramas de caja (boxplots)
    - Gráficos de dispersión
    - Gráficos de barras
    - Heatmaps de correlación
    - Gráficos de líneas
    
    #### 📋 **Reporte Completo**
    - Resumen ejecutivo automático
    - Análisis integrado de todas las variables
    - Recomendaciones y observaciones
    
    ### 🚀 **Cómo comenzar:**
    
    1. **Ve a la sección 'Carga de Datos'**
    2. **Selecciona tu fuente de datos** (archivo propio o ejemplo)
    3. **Explora las estadísticas** en la sección correspondiente
    4. **Crea visualizaciones** interactivas
    5. **Genera tu reporte** final
    
    ### 📚 **Conceptos Estadísticos Incluidos:**
    
    - **Asimetría**: Mide la simetría de la distribución
    - **Curtosis**: Mide el "pico" de la distribución  
    - **Correlación**: Relación lineal entre variables
    - **Valores atípicos**: Datos inusuales en la distribución
    - **Distribución**: Forma en que se dispersan los datos
    
    ### 💡 **Consejos:**
    - Comienza con los datos de ejemplo para familiarizarte
    - Usa las explicaciones incluidas en cada gráfico
    - Exporta tus resultados tomando capturas de pantalla
    """)

def display_data_loading_section():
    """Muestra la sección de carga de datos"""
    st.markdown('<h2 class="section-header">📁 Carga y Exploración de Datos</h2>', unsafe_allow_html=True)
    
    # Opciones de carga de datos
    data_option = st.radio(
        "Selecciona la fuente de datos:",
        ["📊 Usar Datos de Ejemplo", "📤 Cargar Archivo Propio"],
        horizontal=True
    )
    
    df = pd.DataFrame()
    
    if data_option == "📊 Usar Datos de Ejemplo":
        dataset_choice = st.selectbox(
            "Selecciona el conjunto de datos de ejemplo:",
            ["Iris", "Titanic", "Diamantes", "Ventas", "Datos Aleatorios"]
        )
        
        if st.button("🔄 Cargar Datos de Ejemplo"):
            with st.spinner("Cargando datos..."):
                df = load_sample_data(dataset_choice)
                st.success(f"✅ Datos de **{dataset_choice}** cargados exitosamente!")
        
    else:
        uploaded_file = st.file_uploader(
            "📤 Carga tu archivo de datos",
            type=['csv', 'xlsx', 'xls'],
            help="Formatos soportados: CSV, Excel (xlsx, xls)"
        )
        
        if uploaded_file is not None:
            try:
                with st.spinner("Procesando archivo..."):
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                st.success("✅ Archivo cargado exitosamente!")
            except Exception as e:
                st.error(f"❌ Error al cargar el archivo: {e}")
                st.info("💡 Asegúrate de que el archivo no esté corrupto y tenga el formato correcto.")
    
    # Mostrar información del dataset si está cargado
    if not df.empty:
        display_data_preview(df)
        
        # Guardar el dataframe en session state
        st.session_state['df'] = df
        st.session_state['columns'] = df.columns.tolist()
        st.session_state['numeric_columns'] = df.select_dtypes(include=[np.number]).columns.tolist()
        st.session_state['categorical_columns'] = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        st.markdown("### 🎯 Próximos Pasos")
        st.info("""
        Los datos han sido cargados exitosamente. Ahora puedes:
        - **Analizar estadísticas descriptivas** en la sección correspondiente
        - **Crear visualizaciones** interactivas de tus datos
        - **Generar un reporte completo** del análisis
        """)

def display_statistics_section():
    """Muestra la sección de estadística descriptiva"""
    st.markdown('<h2 class="section-header">📈 Análisis Estadístico Descriptivo</h2>', unsafe_allow_html=True)
    
    if 'df' not in st.session_state or st.session_state['df'].empty:
        st.warning("⚠️ Por favor, carga datos primero en la sección 'Carga de Datos'")
        return
    
    df = st.session_state['df']
    numeric_columns = st.session_state['numeric_columns']
    
    if not numeric_columns:
        st.error("❌ No hay columnas numéricas para analizar")
        return
    
    # Selección de columnas para análisis
    st.markdown("### 🎯 Selección de Variables")
    selected_columns = st.multiselect(
        "Selecciona las columnas numéricas para analizar:",
        numeric_columns,
        default=numeric_columns[:min(3, len(numeric_columns))],
        help="Selecciona al menos una variable numérica para el análisis"
    )
    
    if not selected_columns:
        st.info("👆 Por favor, selecciona al menos una variable numérica para continuar con el análisis.")
        return
    
    # Mostrar estadísticas descriptivas
    display_descriptive_stats(df, selected_columns)
    
    # Análisis de correlación si hay múltiples variables
    if len(selected_columns) >= 2:
        st.markdown("### 🔗 Análisis de Correlación")
        corr_matrix = df[selected_columns].corr()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # CORRECCIÓN: Pasar df como primer parámetro
            fig = create_correlation_heatmap(df, selected_columns)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 💡 Interpretación de Correlaciones")
            st.markdown("""
            - **+1.0**: Correlación positiva perfecta
            - **+0.7 a +0.9**: Correlación positiva fuerte
            - **+0.4 a +0.6**: Correlación positiva moderada
            - **-0.3 a +0.3**: Correlación débil o nula
            - **-0.4 a -0.6**: Correlación negativa moderada
            - **-0.7 a -0.9**: Correlación negativa fuerte
            - **-1.0**: Correlación negativa perfecta
            """)

def display_visualizations_section():
    """Muestra la sección de visualizaciones"""
    st.markdown('<h2 class="section-header">📊 Visualización de Datos</h2>', unsafe_allow_html=True)
    
    if 'df' not in st.session_state or st.session_state['df'].empty:
        st.warning("⚠️ Por favor, carga datos primero en la sección 'Carga de Datos'")
        return
    
    df = st.session_state['df']
    numeric_columns = st.session_state['numeric_columns']
    categorical_columns = st.session_state['categorical_columns']
    
    # Selección de tipo de gráfico
    chart_type = st.selectbox(
        "🎨 Selecciona el tipo de gráfico:",
        [
            "📈 Histograma",
            "📊 Gráfico de Densidad",
            "📦 Diagrama de Caja",
            "🔵 Gráfico de Dispersión",
            "📊 Gráfico de Barras",
            "🔥 Heatmap de Correlación",
            "📈 Gráfico de Líneas"
        ]
    )
    
    # Configuración común
    st.markdown("### ⚙️ Configuración del Gráfico")
    
    if chart_type == "📈 Histograma":
        col1, col2 = st.columns(2)
        with col1:
            hist_column = st.selectbox("Selecciona columna:", numeric_columns)
        with col2:
            bins = st.slider("Número de bins:", 5, 100, 30)
        
        if hist_column:
            fig = create_histogram(df, hist_column, bins)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            <div class="info-box">
            <strong>📝 ¿Qué es un Histograma?</strong><br>
            Un histograma muestra la distribución de una variable numérica dividiendo los datos en intervalos (bins) 
            y contando cuántas observaciones caen en cada intervalo. Es útil para identificar:<br>
            • La forma de la distribución (normal, sesgada, bimodal)<br>
            • Valores atípicos<br>
            • La dispersión de los datos<br>
            • La tendencia central
            </div>
            """, unsafe_allow_html=True)
    
    elif chart_type == "📊 Gráfico de Densidad":
        density_column = st.selectbox("Selecciona columna:", numeric_columns)
        if density_column:
            fig = create_density_plot(df, density_column)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            <div class="info-box">
            <strong>📝 ¿Qué es un Gráfico de Densidad?</strong><br>
            Un gráfico de densidad muestra la distribución de probabilidad de una variable continua. 
            Es similar a un histograma pero suavizado, lo que facilita ver la forma general de la distribución.<br>
            • <strong>Ventaja:</strong> No depende del número de bins seleccionado<br>
            • <strong>Uso:</strong> Ideal para comparar distribuciones<br>
            • <strong>Interpretación:</strong> El área bajo la curva suma 1 (100%)
            </div>
            """, unsafe_allow_html=True)
    
    elif chart_type == "📦 Diagrama de Caja":
        col1, col2 = st.columns(2)
        with col1:
            box_column = st.selectbox("Variable numérica:", numeric_columns)
        with col2:
            if categorical_columns:
                group_column = st.selectbox("Variable categórica (opcional):", [""] + categorical_columns)
            else:
                group_column = ""
        
        if box_column:
            fig = create_box_plot(df, box_column, group_column if group_column else None)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            <div class="info-box">
            <strong>📝 ¿Qué es un Diagrama de Caja?</strong><br>
            Un diagrama de caja (boxplot) muestra la distribución de datos a través de sus cuartiles:<br>
            • <strong>Caja:</strong> Representa el 50% central de los datos (Q1 a Q3)<br>
            • <strong>Línea interior:</strong> Es la mediana (Q2)<br>
            • <strong>Bigotes:</strong> Muestran el rango de datos típicos (1.5 * IQR)<br>
            • <strong>Puntos:</strong> Son valores atípicos (outliers)<br>
            • <strong>IQR:</strong> Rango intercuartílico (Q3 - Q1)
            </div>
            """, unsafe_allow_html=True)
    
    elif chart_type == "🔵 Gráfico de Dispersión":
        col1, col2 = st.columns(2)
        with col1:
            scatter_x = st.selectbox("Variable X:", numeric_columns)
        with col2:
            scatter_y = st.selectbox("Variable Y:", numeric_columns)
        
        color_options = [""] + categorical_columns + numeric_columns
        color_column = st.selectbox("Color por (opcional):", color_options)
        
        show_trendline = st.checkbox("📈 Mostrar línea de tendencia")
        
        if scatter_x and scatter_y:
            fig = create_scatter_plot(df, scatter_x, scatter_y, color_column if color_column else None, show_trendline)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            <div class="info-box">
            <strong>📝 ¿Qué es un Gráfico de Dispersión?</strong><br>
            Un gráfico de dispersión muestra la relación entre dos variables numéricas:<br>
            • <strong>Eje X:</strong> Variable independiente<br>
            • <strong>Eje Y:</strong> Variable dependiente<br>
            • <strong>Puntos:</strong> Cada punto representa una observación<br>
            • <strong>Uso:</strong> Identificar correlaciones, tendencias y valores atípicos<br>
            • <strong>Línea de tendencia:</strong> Muestra la dirección general de la relación
            </div>
            """, unsafe_allow_html=True)
    
    elif chart_type == "📊 Gráfico de Barras":
        if categorical_columns:
            bar_column = st.selectbox("Selecciona columna categórica:", categorical_columns)
            if bar_column:
                fig = create_bar_plot(df, bar_column)
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("""
                <div class="info-box">
                <strong>📝 ¿Qué es un Gráfico de Barras?</strong><br>
                Un gráfico de barras muestra la frecuencia o proporción de categorías en una variable cualitativa:<br>
                • <strong>Altura de barras:</strong> Representa la cantidad en cada categoría<br>
                • <strong>Uso:</strong> Comparar frecuencias entre categorías<br>
                • <strong>Ventaja:</strong> Fácil de interpretar y comparar<br>
                • <strong>Variantes:</strong> Barras horizontales, apiladas, agrupadas
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("❌ No hay columnas categóricas para mostrar en gráfico de barras")
    
    elif chart_type == "🔥 Heatmap de Correlación":
        if len(numeric_columns) >= 2:
            selected_for_corr = st.multiselect(
                "Selecciona columnas para correlación:",
                numeric_columns,
                default=numeric_columns[:min(6, len(numeric_columns))]
            )
            
            if len(selected_for_corr) >= 2:
                # CORRECCIÓN: Pasar df como primer parámetro
                fig = create_correlation_heatmap(df, selected_for_corr)
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("""
                <div class="info-box">
                <strong>📝 ¿Qué es un Heatmap de Correlación?</strong><br>
                Un heatmap de correlación muestra las relaciones lineales entre variables numéricas:<br>
                • <strong>Colores:</strong> Rojo (correlación negativa), Azul (correlación positiva)<br>
                • <strong>Valores:</strong> -1 (negativa perfecta) a +1 (positiva perfecta)<br>
                • <strong>0:</strong> No hay correlación lineal<br>
                • <strong>Uso:</strong> Identificar relaciones fuertes entre variables<br>
                • <strong>Precaución:</strong> Correlación no implica causalidad
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("❌ Se necesitan al menos 2 columnas numéricas para el heatmap")
    
    elif chart_type == "📈 Gráfico de Líneas":
        if len(numeric_columns) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                line_x = st.selectbox("Variable X (eje horizontal):", numeric_columns)
            with col2:
                line_y = st.selectbox("Variable Y (eje vertical):", numeric_columns)
            
            color_options = [""] + categorical_columns
            line_color = st.selectbox("Color por categoría (opcional):", color_options)
            
            if line_x and line_y:
                # Ordenar por X para mejor visualización
                temp_df = df.sort_values(by=line_x)
                fig = create_line_plot(temp_df, line_x, line_y, line_color if line_color else None)
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("""
                <div class="info-box">
                <strong>📝 ¿Qué es un Gráfico de Líneas?</strong><br>
                Un gráfico de líneas muestra la evolución de una variable en función de otra:<br>
                • <strong>Eje X:</strong> Variable independiente (generalmente tiempo)<br>
                • <strong>Eje Y:</strong> Variable dependiente<br>
                • <strong>Líneas:</strong> Conectan puntos en orden secuencial<br>
                • <strong>Uso:</strong> Mostrar tendencias a lo largo del tiempo<br>
                • <strong>Ventaja:</strong> Ideal para series temporales y datos secuenciales
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("❌ Se necesitan al menos 2 columnas numéricas para el gráfico de líneas")

def display_report_section():
    """Muestra la sección de reporte completo"""
    st.markdown('<h2 class="section-header">📋 Reporte de Análisis Completo</h2>', unsafe_allow_html=True)
    
    if 'df' not in st.session_state or st.session_state['df'].empty:
        st.warning("⚠️ Por favor, carga datos primero en la sección 'Carga de Datos'")
        return
    
    df = st.session_state['df']
    numeric_columns = st.session_state['numeric_columns']
    categorical_columns = st.session_state['categorical_columns']
    
    # Generar reporte automático
    st.markdown("### 📊 Resumen Ejecutivo del Dataset")
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Total de Observaciones", df.shape[0])
    with col2:
        st.metric("📈 Total de Variables", df.shape[1])
    with col3:
        st.metric("🔢 Variables Numéricas", len(numeric_columns))
    with col4:
        st.metric("🏷️ Variables Categóricas", len(categorical_columns))
    
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("🔍 Valores Nulos", df.isnull().sum().sum())
    with col6:
        st.metric("📐 Filas Duplicadas", df.duplicated().sum())
    with col7:
        st.metric("💾 Memoria Usada", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    with col8:
        # Verificar si hay columnas de fecha
        date_columns = df.select_dtypes(include=['datetime64']).columns
        if len(date_columns) > 0:
            date_col = date_columns[0]
            date_range = f"{df[date_col].min().date()} a {df[date_col].max().date()}"
        else:
            date_range = "N/A"
        st.metric("📅 Rango Temporal", date_range)
    
    # Análisis de variables numéricas
    if numeric_columns:
        st.markdown("### 📈 Análisis de Variables Numéricas")
        
        for col in numeric_columns:
            with st.expander(f"🔍 Análisis de **{col}**", expanded=False):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Estadísticas resumidas
                    st.write("**📋 Estadísticas Clave:**")
                    stats_summary = {
                        'Medida': ['Media', 'Mediana', 'Desv. Estándar', 'Mínimo', 'Máximo', 'Asimetría', 'Curtosis'],
                        'Valor': [
                            f"{df[col].mean():.2f}",
                            f"{df[col].median():.2f}",
                            f"{df[col].std():.2f}",
                            f"{df[col].min():.2f}",
                            f"{df[col].max():.2f}",
                            f"{df[col].skew():.2f}",
                            f"{df[col].kurtosis():.2f}"
                        ]
                    }
                    st.table(pd.DataFrame(stats_summary))
                    
                    # Detección de outliers usando IQR
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                    
                    st.metric("📊 Valores Atípicos", len(outliers))
                
                with col2:
                    # Gráfico combinado
                    fig = px.histogram(df, x=col, title=f"Distribución de {col}", marginal="box")
                    st.plotly_chart(fig, use_container_width=True)
    
    # Análisis de variables categóricas
    if categorical_columns:
        st.markdown("### 🏷️ Análisis de Variables Categóricas")
        
        for col in categorical_columns:
            with st.expander(f"📊 Análisis de **{col}**", expanded=False):
                value_counts = df[col].value_counts()
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**📋 Distribución de Frecuencias:**")
                    st.dataframe(value_counts, use_container_width=True)
                
                with col2:
                    fig = px.pie(values=value_counts.values, names=value_counts.index, 
                               title=f"Distribución de {col}")
                    st.plotly_chart(fig, use_container_width=True)
    
    # Recomendaciones y observaciones
    st.markdown("### 💡 Recomendaciones y Observaciones")
    
    recommendations = []
    
    # Análisis de asimetría
    if numeric_columns:
        skewed_vars = []
        for col in numeric_columns:
            skewness = df[col].skew()
            if abs(skewness) > 1:
                skewed_vars.append((col, skewness))
        
        if skewed_vars:
            recommendations.append("""
            **⚠️ Variables con alta asimetría detectadas:**
            Considera transformaciones (log, raíz cuadrada) para estas variables:
            """ + "\n".join([f"- {var}: asimetría = {skew:.2f}" for var, skew in skewed_vars]))
    
    # Análisis de valores nulos
    null_counts = df.isnull().sum()
    high_null_vars = null_counts[null_counts > 0]
    if not high_null_vars.empty:
        recommendations.append("""
        **🔍 Valores nulos encontrados:**
        Considera estrategias de imputación para:
        """ + "\n".join([f"- {var}: {count} valores nulos" for var, count in high_null_vars.items()]))
    
    # Análisis de correlaciones fuertes
    if len(numeric_columns) >= 2:
        corr_matrix = df[numeric_columns].corr()
        strong_corrs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.8:
                    strong_corrs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
        
        if strong_corrs:
            recommendations.append("""
            **🔗 Correlaciones fuertes detectadas:**
            Considera la multicolinealidad en modelos predictivos:
            """ + "\n".join([f"- {var1} y {var2}: {corr:.2f}" for var1, var2, corr in strong_corrs]))
    
    # Mostrar recomendaciones
    if recommendations:
        for rec in recommendations:
            st.warning(rec)
    else:
        st.success("""
        **✅ El dataset parece estar en buen estado:**
        - No se detectaron variables con alta asimetría problemática
        - No hay valores nulos significativos
        - No hay correlaciones extremadamente altas que sugieran multicolinealidad
        """)
    
    # Botón para exportar
    st.markdown("### 📥 Exportar Reporte")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🖨️ Generar Reporte PDF"):
            st.info("""
            ⚠️ Función de exportación PDF en desarrollo.
            
            **Alternativas:**
            - Toma capturas de pantalla de las secciones relevantes
            - Usa la función de impresión de tu navegador
            - Copia las tablas y gráficos manualmente
            """)
    
    with col2:
        if st.button("💾 Exportar Datos Procesados"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Descargar CSV",
                data=csv,
                file_name="datos_procesados.csv",
                mime="text/csv"
            )
    
    with col3:
        if st.button("📊 Exportar Estadísticas"):
            stats_summary = df.describe().T
            csv_stats = stats_summary.to_csv()
            st.download_button(
                label="📥 Descargar Estadísticas",
                data=csv_stats,
                file_name="estadisticas_descriptivas.csv",
                mime="text/csv"
            )

def display_footer():
    """Muestra el pie de página"""
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        <strong>📊 Dashboard de Análisis Estadístico Descriptivo</strong> • 
        Creado con Streamlit • 
        Usa los datos responsablemente 🎯
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()