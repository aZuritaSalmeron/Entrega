# =========================================================
# ARCHIVO: TFM_AZ.py (El Servidor API Flask)
# =========================================================
import os
import re
import json
import threading
import traceback
from flask import Flask, request, jsonify

# Importamos el músculo lógico y las variables de estado desde Motor_RAG.py
from Motor_RAG import (
    df, 
    vocab, 
    run_pipeline_case, 
    translate_to_english, 
    extraer_y_analizar_con_ia,
    log
)

app = Flask(__name__)

        
# --- 2A. NUEVA RUTA: SOLO ANALIZAR (Frontend) ---
@app.route('/analizar', methods=['POST'])
def analizar_ingredientes():
    data = request.json
    if not data:
        return jsonify({"error": "No se recibieron datos en la petición"}), 400
    
    frase_cruda = data.get('raw_input', '')  
    perfil = data.get('profile', {})

    log(f"Flask [/analizar]: Nueva petición de análisis NLP recibida.")
    #print(f"\n🍳 [NUEVA PETICIÓN] Frase: '{frase_cruda}' | Perfil: {perfil}")
    
    try:
        # 🧹 PASO 0.1: LIMPIEZA ANTI-TRUNCAMIENTO (MarianMT no tolera los ".." o "...")
        frase_limpia = re.sub(r'[\.\?!¡]+', ',', frase_cruda) # Cambiamos puntos por comas
        frase_limpia = re.sub(r'\s+', ' ', frase_limpia).strip() # Quitamos dobles espacios
        
        log(f"🍳 antes de traduccion '{frase_limpia}' | Perfil: {perfil}")
        #print(f"\n🍳 antes de traduccion '{frase_limpia}' | Perfil: {perfil}")
        # 🚀 PASO 0: TRADUCCIÓN PREVIA (Igual que en cocinar)
        # Esto asegura que el análisis que ve el usuario sea el mismo que usará FAISS
        traduccion_previa = translate_to_english([frase_limpia])
        frase_usuario_en = traduccion_previa[0] if traduccion_previa else frase_limpia
        log(f"🇬🇧 Traducción para motor interno: '{frase_usuario_en}'")
        #print(f"🇬🇧 Traducción para motor interno: '{frase_usuario_en}'")

        # 1. Llamamos a la Portera (Qwen) con la frase ya en inglés
        analisis = extraer_y_analizar_con_ia(frase_usuario_en, perfil)
        
        # 2. Preparamos la respuesta para la web
        respuesta = {
            "perfil_elegido": perfil,
            "ingredientes_detectados": analisis["aceptados"],  # Vendrán en inglés
            "conflictos_detectados": analisis["conflictos"],   # Vendrán en inglés
            "razonamiento": analisis["razonamiento"]           # Vendrá en español
        }
        return jsonify(respuesta)
        
    except Exception as e:
        log(f"❌ Error interno en /analizar: {str(e)}")
        return jsonify({"error": f"Error en el análisis: {str(e)}"}), 500
    
# ==========================================
# 2B. LA RUTA DE GENERACIÓN REAL (RAG + Pipeline Completo)
# ==========================================
@app.route('/cocinar', methods=['POST'])
def genera_receta():
    data = request.json
    if not data:
        return jsonify({"error": "No se recibieron datos en la petición"}), 400
    
    # 1. Recibimos la petición del Frontend
    frase_cruda = data.get('raw_input', '')  
    perfil = data.get('profile', {})
    
    log(f"Flask [/cocinar]: Iniciando pipeline RAG de generación culinaria...Frase: '{frase_cruda}' | Perfil: {perfil}")
    #print(f"\n🍳 [NUEVA PETICIÓN] Frase: '{frase_cruda}' | Perfil: {perfil}")
    
    try:
        # 🧹 PASO 0.1: LIMPIEZA ANTI-TRUNCAMIENTO (MarianMT no tolera los ".." o "...")
        frase_limpia = re.sub(r'[\.\?!¡]+', ',', frase_cruda) # Cambiamos puntos por comas
        frase_limpia = re.sub(r'\s+', ' ', frase_limpia).strip() # Quitamos dobles espacios
        log(f"Antes de traduccion '{frase_limpia}' | Perfil: {perfil}")
        
        
        # --- 🚀 PASO 0: TRADUCCIÓN TÉCNICA (MarianMT) ---
        # Traducimos la frase completa al inglés antes de que Qwen la analice.
        # Esto garantiza que el Portero trabaje sobre términos ingleses precisos.
        traduccion_previa = translate_to_english([frase_limpia])
        frase_usuario_en = traduccion_previa[0] if traduccion_previa else frase_limpia
        
        log(f"🇬🇧 Traducción para motor interno: '{frase_usuario_en}'")
        #print(f"🇬🇧 Traducción para motor interno: '{frase_usuario_en}'")
        
        # 2. FASE 1: La IA Portera (Qwen) limpia los ingredientes
        analisis = extraer_y_analizar_con_ia(frase_usuario_en, perfil)
        ingredientes_finales = analisis.get("aceptados", [])
        
        
        # Validación: Si no hay ingredientes, abortamos antes de gastar recursos en FAISS
        if not ingredientes_finales:
            return jsonify({
                "error": "No se detectaron ingredientes seguros o suficientes para cocinar.",
                "razonamiento": analisis.get("razonamiento", "")
            }), 400
            
        log(f"✨ [NLP OUTPUT] Ingredientes validados para motor FAISS: {ingredientes_finales}")
        #print(f"✨ [NLP OUTPUT] Ingredientes listos para FAISS: {ingredientes_finales}")
        
        # 3. FASE 2: EL PIPELINE MAESTRO (Tu Celda 8A)
        # Importante: Asumimos que 'df' y 'vocab' ya están cargados en la memoria de Kaggle
        resultado_pipeline = run_pipeline_case(
            df_data=df, 
            vocab_data=vocab,
            user_ingredients=ingredientes_finales,
            user_profile=perfil,
            max_new_ingredients=3  
        )
        
        log("✅ [PIPELINE COMPLETADO] Receta de respuesta empaquetada y enviada.")

        # ---> 💡 EL NUEVO CHIVATO VISUAL PARA KAGGLE <---
        receta_formateada = json.dumps(resultado_pipeline, indent=4, ensure_ascii=False)
        log(f"\n=================== RECETA FINAL A ENVIAR ===================\n{receta_formateada}\n=============================================================\n")
        
        # Inyectamos el razonamiento para que el frontend lo vea
        resultado_pipeline["razonamiento_salud"] = analisis.get("razonamiento", "")        
        # 4. Devolvemos TODO el resultado del pipeline (receta + métricas) a la web
        return jsonify(resultado_pipeline)
        
    except Exception as e:
        error_details = traceback.format_exc()
        log(f"❌ [ERROR CRÍTICO EN SERVIDOR]:\n{error_details}")
        return jsonify({"error": str(e)}), 500
        
# --- RUTA DE ESTADO (Para que Streamlit sepa que ya calentamos) ---
@app.route('/status', methods=['GET'])
def check_status():
    return jsonify({"status": "ready"})
    
# --- 3. LANZAMIENTO DEL SERVIDOR (EN SEGUNDO PLANO) ---
def calentar_modelo():
    """Fuerza una inferencia mínima inicial para cargar los pesos del LLM en la VRAM de la GPU."""
    log("🔥 Calentando el modelo Qwen en la GPU... (esto tardará 1-2 minutos)")
    try:
        # Le enviamos un prompt tonto solo para obligarle a cargar en memoria
        from Motor_RAG import client, MODEL
        modelo_usar = MODEL if 'MODEL' in globals() else 'qwen2.5:7b'
        client.generate(model=modelo_usar, prompt='hola')
        log("✅ Modelo cargado y listo para volar.")
    except Exception as e:
        log(f"⚠️ Error al calentar: {e}") 
        
def run_app():
    calentar_modelo() # <-- Llamamos al calentamiento antes de abrir el puerto
    # Solo escucha en el puerto 8000 de forma interna
    # MODIFICACIÓN CRÍTICA PARA DOCKER: host="0.0.0.0" expone el puerto fuera del contenedor local bridge
    app.run(host="0.0.0.0", port=8000, debug=False, use_reloader=False)

# Esto evita que Kaggle se sature si ejecutas la celda varias veces por error
#try:
#   if thread.is_alive():
#        print("⚠️ El servidor ya está corriendo en segundo plano.")
#except NameError:
#    thread = threading.Thread(target=run_app)
#    thread.start()
#    print("🚀 Servidor Backend iniciado en el puerto 8000. Listo para recibir peticiones.")

if __name__ == '__main__':
    # Esta estructura de hilos mantiene la compatibilidad de ejecución si realizas pruebas locales,
    # pero ejecuta el bucle infinito nativo de producción de forma limpia al invocarse desde consola.
    try:
        if 'thread' in globals() and thread.is_alive():
            log("⚠️ El daemon del servidor web ya está activo en segundo plano.")
        else:
            thread = threading.Thread(target=run_app)
            thread.daemon = True
            thread.start()
            log("🚀 Servidor de Producción Backend desplegado en http://0.0.0.0:8000")
            
            # Bloquea el hilo principal para que el script no termine al correr desde terminal o Docker
            thread.join()
    except KeyboardInterrupt:
        log("🛑 Señal de interrupción recibida. Servidor web Flask apagado.")