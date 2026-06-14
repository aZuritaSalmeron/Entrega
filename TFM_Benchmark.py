import time
import requests
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Supresión de advertencias de visualización para mantener la consola limpia
warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURACIÓN DEL ENTORNO Y DATOS
# ==========================================
# Endpoint del túnel inverso hacia el backend en Kaggle
API_URL = "https://nosological-apryl-hybridizable.ngrok-free.dev/cocinar" 

# Batería de validación de 51 casos
CASES_50 = [
    # ---------- Base (no restrictions) ----------
    {"id": 1,  "ings": ["milk","butter","vanilla","salt"], "profile": {}},
    {"id": 2,  "ings": ["rice","chicken","onion","garlic","salt"], "profile": {}},
    {"id": 3,  "ings": ["pasta","tomato","olive oil","garlic","salt"], "profile": {}},
    {"id": 4,  "ings": ["eggs","cheese","salt","pepper"], "profile": {}},
    {"id": 5,  "ings": ["potato","onion","olive oil","salt"], "profile": {}},
    {"id": 6,  "ings": ["tuna","mayonnaise","lemon","salt"], "profile": {}},
    {"id": 7,  "ings": ["oats","milk","banana","cinnamon"], "profile": {}},
    {"id": 8,  "ings": ["yogurt","honey","nuts"], "profile": {}},
    {"id": 9,  "ings": ["bread","ham","cheese","butter"], "profile": {}},
    {"id":10,  "ings": ["lentils","onion","carrot","garlic","salt"], "profile": {}},

    # ---------- Lactose intolerance ----------
    {"id":11, "ings": ["milk","butter","vanilla","salt"], "profile": {"lactose_intolerant": True}},
    {"id":12, "ings": ["yogurt","banana","honey"], "profile": {"lactose_intolerant": True}},
    {"id":13, "ings": ["cheese","eggs","salt"], "profile": {"lactose_intolerant": True}},
    {"id":14, "ings": ["pasta","tomato","olive oil","garlic"], "profile": {"lactose_intolerant": True}},
    {"id":15, "ings": ["chicken","rice","onion","salt"], "profile": {"lactose_intolerant": True}},
    {"id":16, "ings": ["butter","garlic","salt"], "profile": {"lactose_intolerant": True}},

    # ---------- Vegan ----------
    {"id":17, "ings": ["milk","butter","vanilla","salt"], "profile": {"vegan": True}},
    {"id":18, "ings": ["pasta","tomato","olive oil","garlic","salt"], "profile": {"vegan": True}},
    {"id":19, "ings": ["rice","beans","onion","garlic","salt"], "profile": {"vegan": True}},
    {"id":20, "ings": ["tofu","soy sauce","garlic","ginger"], "profile": {"vegan": True}},
    {"id":21, "ings": ["eggs","cheese","salt"], "profile": {"vegan": True}},
    {"id":22, "ings": ["banana","oats","cinnamon"], "profile": {"vegan": True}},

    # ---------- Vegetarian ----------
    {"id":23, "ings": ["chicken","rice","onion","salt"], "profile": {"vegetarian": True}},
    {"id":24, "ings": ["eggs","cheese","salt","pepper"], "profile": {"vegetarian": True}},
    {"id":25, "ings": ["pasta","tomato","cheese","olive oil"], "profile": {"vegetarian": True}},
    {"id":26, "ings": ["tuna","pasta","tomato"], "profile": {"vegetarian": True}},
    {"id":27, "ings": ["lentils","carrot","onion","salt"], "profile": {"vegetarian": True}},
    {"id":28, "ings": ["mushrooms","garlic","butter","salt"], "profile": {"vegetarian": True}},

    # ---------- Shellfish allergy ----------
    {"id":29, "ings": ["shrimp","rice","saffron","salt"], "profile": {"shellfish_allergy": True}},
    {"id":30, "ings": ["mussels","pasta","garlic","olive oil"], "profile": {"shellfish_allergy": True}},
    {"id":31, "ings": ["fish","lemon","salt"], "profile": {"shellfish_allergy": True}},
    {"id":32, "ings": ["crab","bread","mayonnaise"], "profile": {"shellfish_allergy": True}},
    {"id":33, "ings": ["chicken","rice","onion"], "profile": {"shellfish_allergy": True}},
    {"id":34, "ings": ["shrimp","garlic","butter"], "profile": {"shellfish_allergy": True}},

    # ---------- Halal ----------
    {"id":35, "ings": ["pork","rice","onion","salt"], "profile": {"halal": True}},
    {"id":36, "ings": ["chicken","rice","onion","salt"], "profile": {"halal": True}},
    {"id":37, "ings": ["beef","tomato","onion","salt"], "profile": {"halal": True}},
    {"id":38, "ings": ["wine","garlic","butter"], "profile": {"halal": True}},
    {"id":39, "ings": ["gelatin","fruit"], "profile": {"halal": True}},
    {"id":40, "ings": ["fish","lemon","salt"], "profile": {"halal": True}},

    # ---------- Diabetes ----------
    {"id":41, "ings": ["sugar","milk","butter","vanilla"], "profile": {"diabetic": True}},
    {"id":42, "ings": ["banana","oats","milk"], "profile": {"diabetic": True}},
    {"id":43, "ings": ["chicken","salad","olive oil","salt"], "profile": {"diabetic": True}},
    {"id":44, "ings": ["pasta","tomato","salt"], "profile": {"diabetic": True}},
    {"id":45, "ings": ["yogurt","honey","nuts"], "profile": {"diabetic": True}},
    {"id":46, "ings": ["eggs","spinach","olive oil","salt"], "profile": {"diabetic": True}},

    # ---------- Mixed/edge cases ----------
    {"id":47, "ings": ["milk","butter","vanilla","salt"], "profile": {"vegan": True, "diabetic": True}},
    {"id":48, "ings": ["shrimp","milk","butter"], "profile": {"shellfish_allergy": True, "lactose_intolerant": True}},
    {"id":49, "ings": ["pork","wine","cheese"], "profile": {"halal": True, "lactose_intolerant": True}},
    {"id":50, "ings": ["tomato","onion","garlic","olive oil","salt"], "profile": {"vegan": True, "halal": True}},
    {"id":51, "ings": ["oil", "garlic", "salt"], "profile": {}},
]

# ==========================================
# 2. FUNCIONES DE GENERACIÓN DE GRÁFICOS
# ==========================================
def generar_graficos(metrics, res, path_counts):
    print("\nGenerando gráficos de alta resolución...")
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'figure.dpi': 300})

    # Gráfico 1: Latencia
    labels_lat = ['Media (Avg)', 'Mediana (P50)', 'Percentil 95 (P95)']
    values_lat = [metrics['avg_seconds'], metrics['p50_seconds'], metrics['p95_seconds']]
    
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = sns.barplot(x=labels_lat, y=values_lat, palette="Blues_d", ax=ax)
    ax.set_title('Distribución de la Latencia Computacional (Qwen2.5 7B)', pad=15, fontweight='bold')
    ax.set_ylabel('Tiempo de respuesta (segundos)')
    ax.set_ylim(0, max(values_lat) * 1.2 if max(values_lat) > 0 else 10)
    
    for p in bars.patches:
        ax.annotate(f'{p.get_height():.2f} s', (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 9), textcoords='offset points', fontweight='bold')
    plt.savefig('grafico_latencia_7B.png')
    plt.close()

    # Gráfico 2: Tasas de Éxito
    tasa_exito_total = len(res[res['status'] == 'SUCCESS']) / len(res) if len(res) > 0 else 0
    labels_succ = ['Pre-Validación', 'Post-Validación', 'Éxito Final']
    values_succ = [metrics['valid_pre_rate']*100, metrics['valid_post_rate']*100, tasa_exito_total*100]
    
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = sns.barplot(x=labels_succ, y=values_succ, palette="Greens_d", ax=ax)
    ax.set_title('Tasas de Validación y Cumplimiento de Restricciones', pad=15, fontweight='bold')
    ax.set_ylabel('Porcentaje de Éxito (%)')
    ax.set_ylim(0, 115)
    
    for p in bars.patches:
        ax.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 9), textcoords='offset points', fontweight='bold')
    plt.savefig('grafico_tasas_exito.png')
    plt.close()

    # Gráfico 3: Flujos de ejecución (Paths)
    labels_paths = [k.replace('_', ' ').title() for k, v in path_counts.items() if v > 0]
    sizes_paths = [v for v in path_counts.values() if v > 0]
    
    if len(sizes_paths) > 0:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.pie(sizes_paths, labels=labels_paths, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11})
        ax.axis('equal')
        ax.set_title('Intervención del Sistema de Guardrails', pad=15, fontweight='bold')
        plt.savefig('grafico_flujos_ejecucion.png')
        plt.close()

    print("Gráficos generados: 'grafico_latencia_7B.png', 'grafico_tasas_exito.png', 'grafico_flujos_ejecucion.png'.")

# ==========================================
# 3. EJECUCIÓN DEL BENCHMARK
# ==========================================
def run_benchmark():
    resultados = []
    print(f"Iniciando evaluación rigurosa de {len(CASES_50)} casos a través de ngrok...")

    # Cabeceras requeridas para omitir la página de advertencia de ngrok
    headers = {
        "Content-Type": "application/json",
        "ngrok-skip-browser-warning": "true" 
    }

    for caso in CASES_50:
        payload = {"ingredientes": caso["ings"], "perfil": caso["profile"]}
        
        try:
            response = requests.post(API_URL, json=payload, headers=headers, timeout=180)
            if response.status_code != 200:
                raise ValueError(f"Error HTTP {response.status_code}")
                
            api_data = response.json()
            
            # Extracción de métricas lógicas
            valid_pre = api_data.get("valid_pre", False)
            valid_post = api_data.get("valid_post", False)
            early_exit = api_data.get("early_exit", False)
            title_sanity_exit = api_data.get("title_sanity_exit", False)
            used_fix = api_data.get("used_fix", False)
            used_fallback = api_data.get("used_fallback", False)
            seconds = api_data.get("seconds_total", 0.0)

            # Extracción íntegra de variables de texto (Evita KeyError en Pandas)
            receta_final = api_data.get("recipe_final", {})
            titulo = receta_final.get("title", "Sin Título") if isinstance(receta_final, dict) else "Generación Plana"
            errors_post = api_data.get("errors_post", [])
            invalid_ings_pre = api_data.get("invalid_ings_pre", [])
            title_errs_pre = api_data.get("title_errs_pre", [])

            # Determinación analítica del flujo (Path)
            if early_exit: path = "early_exit"
            elif title_sanity_exit: path = "title_sanity_exit"
            elif used_fix: path = "fixed_generation"
            elif used_fallback: path = "fallback_generation"
            else: path = "standard_generation"
                
            # Determinación analítica del estado final (Status)
            if early_exit or title_sanity_exit: status = "BLOCKED_PRE"
            elif not valid_post: status = "FAILED_POST"
            else: status = "SUCCESS"
                
        except Exception as e:
            valid_pre, valid_post, early_exit, title_sanity_exit, used_fix, used_fallback = [False]*6
            seconds, path, status = 0.0, "system_error", "CRITICAL_ERROR"
            titulo = "Error del servidor"
            errors_post = [str(e)]
            invalid_ings_pre = []
            title_errs_pre = []

        # Estructura completa inyectada en la lista (14 variables requeridas)
        resultados.append({
            "id": caso["id"], 
            "status": status, 
            "path": path, 
            "valid_pre": valid_pre, 
            "valid_post": valid_post, 
            "early_exit": early_exit, 
            "title_sanity_exit": title_sanity_exit, 
            "used_fallback": used_fallback, 
            "used_fix": used_fix, 
            "seconds": seconds,
            "title": titulo,                      
            "errors_post": errors_post,           
            "invalid_ings_pre": invalid_ings_pre, 
            "title_errs_pre": title_errs_pre      
        })
        print(f"[{caso['id']:02d}/{len(CASES_50)}] Procesado. Estado: {status} | Tiempo: {seconds:.2f}s | Ruta: {path}")

    # ==========================================
    # 4. CÁLCULO DE MÉTRICAS E IMPRESIÓN DETALLADA
    # ==========================================
    res = pd.DataFrame(resultados)

    metrics = {
        "n_cases": int(len(res)),
        "valid_pre_rate": float(res["valid_pre"].mean()),
        "valid_post_rate": float(res["valid_post"].mean()),
        "fallback_rate": float(res["used_fallback"].mean()),
        "fix_rate": float(res["used_fix"].mean()),
        "early_exit_rate": float(res["early_exit"].mean()),
        "title_sanity_exit_rate": float(res["title_sanity_exit"].mean()),
        "avg_seconds": float(res["seconds"].mean()),
        "p50_seconds": float(res["seconds"].median()),
        "p95_seconds": float(res["seconds"].quantile(0.95)),
    }

    status_counts = res["status"].value_counts(dropna=False).to_dict()
    path_counts = res["path"].value_counts(dropna=False).to_dict()

    print("\n" + "="*40)
    print("=== INFORME DE MÉTRICAS GLOBALES ===")
    print("="*40)
    print(json.dumps(metrics, indent=2))
    
    print("\n=== DISTRIBUCIÓN DE ESTADOS ===")
    print(json.dumps(status_counts, indent=2))
    
    print("\n=== DISTRIBUCIÓN DE FLUJOS (PATHS) ===")
    print(json.dumps(path_counts, indent=2))

    print("\n=== FALLOS POST-GENERACIÓN (Deberían tender a 0) ===")
    print(res[res["valid_post"] == False][["id","status","title","errors_post","path","seconds"]].head(10).to_string(index=False))

    print("\n=== EJEMPLOS DE SALIDA TEMPRANA (EARLY EXIT) ===")
    print(res[res["early_exit"] == True][["id","status","title","invalid_ings_pre","path","seconds"]].head(5).to_string(index=False))

    print("\n=== EJEMPLOS DE BLOQUEO POR TÍTULO (SANITY EXIT) ===")
    print(res[res["title_sanity_exit"] == True][["id","status","title","title_errs_pre","path","seconds"]].head(5).to_string(index=False))

    print("\n=== TOP 10 CASOS CON MAYOR LATENCIA COMPUTACIONAL ===")
    print(res.sort_values("seconds", ascending=False)[["id", "status", "path", "seconds"]].head(10).to_string(index=False))

    # Invocación final de la generación gráfica para la memoria académica
    generar_graficos(metrics, res, path_counts)
    # Volcado de métricas a un archivo persistente
    with open("registro_metricas_7B.json", "w", encoding="utf-8") as f:
        json.dump({
            "metricas_globales": metrics,
            "distribucion_flujos": path_counts
        }, f, indent=4)
if __name__ == "__main__":
    run_benchmark()