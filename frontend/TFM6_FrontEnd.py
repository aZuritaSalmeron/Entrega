# =========================================================
# ARCHIVO: TFM6_FrontEnd.py (Interfaz Gráfica Streamlit)
# =========================================================
import streamlit as st
import requests
import time

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Chef IA - TFM", page_icon="👨‍🍳", layout="centered") 
MODO_DEBUG = True  # Cambia a True para que reaparezca el botón de analizar

 
# --- CONFIGURACIÓN DEL SERVIDOR (SIDEBAR) ---
st.sidebar.title("⚙️ Configuración Técnica")
st.sidebar.info("Si ejecutas con Docker Compose, usa el valor por defecto. Si usas Kaggle, pega la URL del túnel:")

# OPTIMIZACIÓN DOCKER: Establecemos la dirección interna del contenedor como valor por defecto
url_base = st.sidebar.text_input(
    "URL del Backend:", 
    value="http://backend:8000", 
    placeholder="http://backend:8000 o https://tu-url.ngrok-free.dev"
)

# Limpiamos la URL y la guardamos en la variable oficial "url_base" que usa tu código
url_base = url_base.replace("\\", "").strip(" /")

# --- NUEVO: SISTEMA DE DETECCIÓN DE CALENTAMIENTO ---
if url_base:
    status_placeholder = st.sidebar.empty() # Creamos un espacio reservado
    status_placeholder.info("⏳ Comprobando el cerebro de la IA...")
    
    try:
        # Hacemos un "ping" rápido a la ruta que acabamos de crear (máximo 3 segundos)
        check = requests.get(f"{url_base.rstrip('/')}/status", timeout=3)
        
        if check.status_code == 200:
            status_placeholder.success("✅ **Modelo cargado y listo.**")
        else:
            status_placeholder.warning("⚠️ El servidor responde, pero la IA dio error.")
            
    except requests.exceptions.RequestException:
        # Si la conexión falla o da timeout, significa que Flask sigue bloqueado en "calentar_modelo()"
        status_placeholder.warning("🔥 **Cargando el modelo...** (Puede tardar 1-2 minutos. Si acaba de encenderse, espera un momento).")

# Construimos la URL final asegurándonos de que termina en /cocinar
BACKEND_URL = f"{url_base.rstrip('/')}/cocinar" if url_base else ""

# --- DICCIONARIO DE PERFILES ---
# Mapeamos lo que ve el usuario (Español) con el JSON técnico que espera el Backend
PERFILES_MAP = {
    "Ninguno (Sin restricciones)": {},
    "Vegano": {"vegan": True},
    "Intolerante a la Lactosa": {"lactose_intolerant": True},
    "Halal": {"halal": True},
    "Diabético": {"diabetic": True},
    "Celiaco":  {"celiac": True}
}

# --- INTERFAZ GRÁFICA ---
st.title("👨‍🍳 Tu Asistente de Cocina Inteligente")

st.markdown("Háblame de forma natural. Dime qué tienes en la nevera y te pensaré un plato.")

# 1. Caja para ingredientes (texto libre)
st.subheader("🛒 ¿Qué ingredientes tienes?")
ingredientes_input = st.text_area(
    "Escribe aquí (ej: Hola, hoy tengo un poco de arroz, dos pechugas de pollo y algo de sal):", 
    height=100,
    placeholder="Tengo arroz, tomate, un bote de garbanzos y cebolla..."
)

# 2. Desplegable para el perfil
st.subheader("🛡️ Perfil Dietético")
perfil_seleccionado = st.selectbox(
    "Selecciona una restricción si la tienes:",
    options=list(PERFILES_MAP.keys()),
    index=0  
)

st.markdown("---")

# --- BOTONES DE ACCIÓN (Usamos columnas para ponerlos lado a lado) ---
col1, col2 = st.columns(2)
# ==========================================
# BOTÓN 1: MODO ANÁLISIS (Debug NLP) - AHORA INVISIBLE
# ==========================================
if MODO_DEBUG:
    if col1.button("🔍 Analizar Frase", use_container_width=True):
        if not url_base:
            st.sidebar.error("⚠️ Falta la URL del Backend.")
        elif not ingredientes_input.strip():
            st.warning("⚠️ Escribe algo en la caja de ingredientes.")
        else:
            with st.spinner("🤖 Analizando texto y detectando conflictos..."):
                try:
                    # Construimos la URL apuntando a la nueva ruta /analizar
                    URL_ANALIZAR = f"{url_base.rstrip('/')}/analizar"
                    payload = {
                        "raw_input": ingredientes_input, 
                        "profile": PERFILES_MAP[perfil_seleccionado]
                    }
                    
                    # ⏱️ Arrancamos el cronómetro de Análisis
                    inicio_tiempo = time.time()  
                    response = requests.post(URL_ANALIZAR, json=payload, timeout=150)
                    # ⏱️ Paramos el cronómetro
                    fin_tiempo = time.time()
                    tiempo_tardado = fin_tiempo - inicio_tiempo  

                    if response.status_code == 200:
                        datos = response.json()
                        
                        # AÑADIDO: Mostramos el tiempo directamente en el título del panel
                        st.info(f"📊 **Resumen del Análisis NLP** (⏱️ {tiempo_tardado:.2f}s):")
                        
                        ingredientes_ok = datos.get("ingredientes_detectados", [])
                        conflictos = datos.get("conflictos_detectados", [])
                        
                        # --- NUEVA LÓGICA: ¿Hay comida para cocinar? ---
                        if not ingredientes_ok:
                            # Si la lista verde está vacía, lanzamos el error crítico
                            st.error("❌ **No es posible generar receta por falta de suficientes ingredientes.**")
                            # (Opcional) Si está vacía porque la IA excluyó todo, mostramos por qué
                            if conflictos:
                                st.error(f"⚠️ **Conflicto con perfil:** Se ha excluido {', '.join(conflictos)}")
                                if datos.get("razonamiento"):
                                    st.info(f"💡 **Motivo de la IA:**\n\n{datos['razonamiento']}")
                        else:
                            # Si SÍ hay ingredientes válidos, mostramos el flujo normal
                            st.success(f"✅ **Ingredientes a usar:** {', '.join(ingredientes_ok)}")
                            if conflictos:
                                st.error(f"⚠️ **Conflicto con perfil:** Se ha excluido {', '.join(conflictos)}")
                                if datos.get("razonamiento"):
                                    st.info(f"💡 **Motivo de la IA:**\n\n{datos['razonamiento']}")
                            # Si hay perfil, hay ingredientes, y NO hay conflictos, entonces sí es seguro
                            elif PERFILES_MAP[perfil_seleccionado]:
                                 st.success("✅ **Perfil Seguro:** No se detectaron conflictos con tu dieta.")
    
                        # Mostramos el payload técnico
                        with st.expander("Ver Payload Técnico enviado a FAISS"):
                            st.json(datos)
                    else:
                        st.error("❌ Error al analizar.")
                except Exception as e:
                    st.error(f"❌ Error de conexión: {e}")
                
# ==========================================
# BOTÓN 2: LÓGICA DEL BOTÓN PRINCIPAL
# ==========================================
if col2.button("🚀 Crear Receta Mágica", use_container_width=True):
    # 1. Validación: Comprobar que el usuario ha puesto la URL de ngrok
    if not url_base:
        st.sidebar.error("⚠️ Falta la URL del Backend.")
        st.warning("👈 Por favor, pega la URL de ngrok en el menú lateral izquierdo antes de continuar.")
    # 2. Validación: Comprobar que hay texto en la caja de ingredientes
    elif not ingredientes_input.strip():
        st.warning("⚠️ ¡Dime al menos un ingrediente para poder cocinar!")
    # 3. Ejecución principal
    else:
        with st.spinner("🧠 Leyendo tu mensaje y pensando una receta..."):
            try:
                perfil_json = PERFILES_MAP[perfil_seleccionado]

                # --- FASE 1: EJECUTAMOS EL ANÁLISIS DE FORMA AUTOMÁTICA ---
                URL_ANALIZAR = f"{url_base.rstrip('/')}/analizar"
                payload_analisis = {
                    "raw_input": ingredientes_input, 
                    "profile": perfil_json
                }
                
                # ⏱️ Cronómetro Fase 1 (Análisis NLP)
                inicio_analisis = time.time()
                resp_analisis = requests.post(URL_ANALIZAR, json=payload_analisis, timeout=150)
                tiempo_analisis = time.time() - inicio_analisis
                
                if resp_analisis.status_code == 200:
                    datos = resp_analisis.json()
                    st.info(f"📊 **Resumen del Análisis NLP** (⏱️ {tiempo_analisis:.2f}s):")
                    
                    ingredientes_ok = datos.get("ingredientes_detectados", [])
                    conflictos = datos.get("conflictos_detectados", [])
                    
                    # Lógica de dibujo de interfaz y comprobación
                    if not ingredientes_ok:
                        st.error("❌ **No es posible generar receta por falta de suficientes ingredientes.**")
                        if conflictos:
                            st.error(f"⚠️ **Conflicto con perfil:** Se ha excluido {', '.join(conflictos)}")
                            if datos.get("razonamiento"):
                                st.info(f"💡 **Motivo de la IA:**\n\n{datos['razonamiento']}")
                        st.stop() # <-- CRÍTICO: Esto detiene la ejecución aquí mismo si no hay ingredientes
                    else:
                        st.success(f"✅ **Ingredientes a usar:** {', '.join(ingredientes_ok)}")
                        if conflictos:
                            st.error(f"⚠️ **Conflicto con perfil:** Se ha excluido {', '.join(conflictos)}")
                            if datos.get("razonamiento"):
                                st.info(f"💡 **Motivo de la IA:**\n\n{datos['razonamiento']}")
                        elif perfil_json:
                             st.success("✅ **Perfil Seguro:** No se detectaron conflictos con tu dieta.")
                else:
                    st.error("❌ Error en el servidor al intentar analizar la frase.")
                    st.stop()

                # --- FASE 2: SI LLEGAMOS AQUÍ, HAY INGREDIENTES, GENERAMOS LA RECETA ---
                #BACKEND_URL = f"{url_base.rstrip('/')}/cocinar"

                # PREPARAR EL PAQUETE (PAYLOAD ORIGINAL)
                payload = {
                    "raw_input": ingredientes_input, 
                    "profile": perfil_json
                }
                
                # ⏱️ Cronómetro Fase 2 (Búsqueda FAISS)
                inicio_generacion = time.time()
                # Llamada al Backend (Localtunnel en Kaggle)
                response = requests.post(BACKEND_URL, json=payload, timeout=150)
                # ⏱️ Paramos cronómetro Fase 2 y calculamos el total
                fin_generacion = time.time()
                tiempo_generacion = fin_generacion - inicio_generacion 
                # Sumamos ambos tiempos
                tiempo_total = tiempo_analisis + tiempo_generacion

                # Procesamos la respuesta
                if response.status_code == 200:
                    receta = response.json()
                    # Mostramos el éxito y el tiempo TOTAL que ha tardado con 2 decimales
                    st.success(f"¡Receta lista! ⏱️ Proceso completo en {tiempo_total:.2f} segundos.")
                    # Mostramos el JSON de la receta en bruto temporalmente
                    st.json(receta) 
                else:
                    st.error(f"❌ Error en el cerebro de la IA (Backend): Código {response.status_code}")
                
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Ocurrió un error de conexión con el servidor backend.")
                st.info(f"💡 Detalle del error: {e}")
                