# =========================================================
# ARCHIVO: motor_rag.py (El Cerebro del TFM)
# =========================================================
import os
import ast
import json
import re
import time
import requests
import pandas as pd
import numpy as np
import faiss
import itertools
import ollama
from typing import List, Dict, Any, Set, Tuple
from sentence_transformers import SentenceTransformer
from copy import deepcopy

# =========================================================
# 1. CONFIGURACIÓN Y CONSTANTES (Preparado para Docker)
# =========================================================

# Para Docker: Leemos la IP del host de Ollama desde las variables de entorno. 
# Si estamos en local, usamos localhost.
OLLAMA_HOST_BASE = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_URL  = f"{OLLAMA_HOST_BASE}/api/generate"

# Configuración del modelo en la librería nativa de ollama
client = ollama.Client(host=OLLAMA_HOST_BASE)

#MODEL       = "qwen2.5:3b"
MODEL       = "qwen2.5:7b"
MAX_NEW_INGREDIENTS_DEFAULT = 2
OLLAMA_NUM_CTX = 2048
OLLAMA_NUM_PREDICT = 600
OLLAMA_TEMP = 0.0

# Rutas de datos adaptadas para el entorno local/Docker
DATA_DIR = os.getenv("DATA_DIR", "./data")
SUBSET_PATH = os.path.join(DATA_DIR, 'recipes_full_cleaned.parquet')
VOCAB_PATH = os.path.join(DATA_DIR, 'ingredient_vocab.json')
FAISS_INDEX_PATH = os.path.join(DATA_DIR, 'recipe_index.faiss')
MODEL_NAME = 'all-MiniLM-L6-v2'

# -------------------------
# MAPEO DE RESTRICCIONES Y BÁSICOS
# -------------------------
RESTRICTION_MAP = {
    "diabetic": {"sugar", "honey", "syrup", "condensed milk", "molasses"},
    "lactose_intolerant": {"milk", "cream", "butter", "cheese", "yogurt", "whey", "lactose"},
    "shellfish_allergy": {"shrimp", "prawn", "crab", "lobster", "mussel", "clam", "oyster", "scallop"},
    "vegan": {"meat", "beef", "pork", "chicken", "fish", "egg", "eggs", "milk", "cheese", "honey", "gelatin"},
    "vegetarian": {"meat", "beef", "pork", "chicken", "fish", "shrimp", "bacon", "ham"},
    "halal": {"pork", "bacon", "lard", "wine", "beer", "alcohol", "rum", "vodka"},
    "celiac": {"wheat", "barley", "rye", "oats", "spelt", "kamut", "semolina", "coscous", "bulgur", "triticale", "einkorn", "emmer", "farina", "farro", "malt", "soy", "seitan", "dextrin", "bread", "pita", "ciabatta", "baguettes", "beer"},
}

STAPLES = {
    "salt", "pepper", "water", "oil", "olive oil",
    "sugar", "garlic", "onion", "spice", "herb"
}

BAKING_STAPLES = {
    "flour", "all-purpose flour", "plain flour",
    "sugar", "brown sugar", "powdered sugar", "icing sugar",
    "egg", "eggs","salt", "pepper","water",
    "baking powder", "baking soda","vanilla",
    "yeast", "oil", "olive oil","garlic", "onion",
    "cocoa powder", "soy sauce", "vinegar"
}

TITLE_IMPLY_STAPLES = {
    "cake", "cookie", "cookies", "bread", "fudge", "pudding", "cheesecake", "brownie", "brownies",
    "muffin", "pie", "tart", "pancake", "waffle", "omelette", "custard"
}
STAPLES_MIN = {"flour", "sugar"}
UTILITY_INGREDIENTS = {"water"}
NON_INGREDIENT_TOKENS = {
    "foil", "aluminum", "parchment", "paper", "toothpick", "rack", "pan", "pot", "bowl",
    "skillet", "dish", "tray", "microwave", "oven", "freezer", "fridge", "container",
    "plate", "spoon", "fork", "knife", "sheet", "wrap", "lid", "with", "and", "or",
    "before", "after", "until", "into", "over", "in", "on", "of", "for"
}
COOKING_STOPWORDS = {
    "mix","stir","combine","heat","cook","bake","pan","pot","bowl","oven","serve",
    "add","pour","bring","boil","simmer","until","smooth","heavy","cream","water",
    "foil", "aluminum", "parchment", "paper", "wrap", "plastic", "sheet", "tray",
    "rack", "dish", "knife", "spoon", "fork", "whisk", "microwave", "stove",
    "container", "plate", "with", "and", "or", "before", "after", "then"
}
COMMON_INGREDIENT_WORDS = {
    "sugar", "flour", "oil", "yeast", "egg", "eggs", "baking powder",
    "rice", "pasta", "milk", "butter", "cheese", "yogurt", "heavy cream",
    "chicken", "beef", "pork", "fish", "onion", "garlic", "tomato", "lemon"
}

# =========================================================
# 2. LOGGING Y HERRAMIENTAS JSON
# =========================================================
def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

_TRAILING_COMMAS_RE = re.compile(r",\s*([}\]])")
_JSON_BLOCK_RE = re.compile(r'(\{.*\}|\[.*\])', re.DOTALL)

def cleanup_json_str(s: str) -> str:
    if not s or not isinstance(s, str): return ""
    fence = chr(96) * 3
    s = re.sub(rf"{fence}(?:json)?", "", s, flags=re.IGNORECASE)
    s = s.replace(fence, "").strip()
    match = _JSON_BLOCK_RE.search(s)
    if match: s = match.group(1).strip()
    else: return "{}"
    s = _TRAILING_COMMAS_RE.sub(r"\1", s)
    s = "".join(char for char in s if ord(char) >= 32 or char in "\n\r\t")
    return s.strip()

def parse_llm_json(text: str) -> Any:
    if not text: raise ValueError("La respuesta está vacía.")
    json_str = cleanup_json_str(text)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        try:
            json_str_alt = json_str.replace('\\n', ' ').replace('\\r', '')
            return json.loads(json_str_alt)
        except:
            raise ValueError(f"Fallo crítico al parsear JSON: {e}")

# =========================================================
# 3. CONEXIÓN CON IA Y GENERACIÓN DE PROMPTS
# =========================================================
def ollama_generate(prompt: str, max_tokens: int = None, temperature: float = None, num_ctx: int = None) -> str:
    req = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": int(max_tokens if max_tokens is not None else OLLAMA_NUM_PREDICT),
            "temperature": float(temperature if temperature is not None else OLLAMA_TEMP),
            "num_ctx": int(num_ctx if num_ctx is not None else OLLAMA_NUM_CTX),
        }
    }
    try:
        r = requests.post(OLLAMA_URL, json=req, timeout=600)
        r.raise_for_status()
        return r.json().get("response", "")
    except Exception as e:
        log(f"⚠️ Error Ollama: {e}")
        return ""

def translate_to_english(texts):
    if isinstance(texts, str): texts = [texts]
    traducciones = []
    for texto in texts:
        prompt_traduccion = f"""
        Translate the following Spanish text to English. 
        Respond ONLY with the exact translation. No explanations.
        TEXT: {texto}
        """
        try:
            response = client.generate(model=MODEL, prompt=prompt_traduccion)
            traducciones.append(response['response'].strip())
        except Exception as e:
            log(f"⚠️ Error en traducción: {e}")
            traducciones.append(texto) 
    return traducciones

def extraer_y_analizar_con_ia(frase_usuario, perfil_json):
    perfil_json = perfil_json or {}
    reglas_activas_en = ""
    perfil_lower = {str(k).lower(): v for k, v in perfil_json.items()}
    
    for dieta, activa in perfil_lower.items():
        if activa and dieta in RESTRICTION_MAP:
            prohibidos = ", ".join(RESTRICTION_MAP[dieta])
            reglas_activas_en += f"- STRICT {dieta.upper()} RULE: PROHIBITED: {prohibidos}\n"
            
    if not reglas_activas_en:
        reglas_activas_en = "NONE. NO RESTRICTIONS. ALL INGREDIENTS ARE 100% SAFE AND ACCEPTED."
        
    frase_limpia = re.sub(r'[!¡]+', '', frase_usuario).replace('...', ',')
    prompt = f"""
    ROLE: Nutrition and Food Safety Expert Assistant.
    TASK: Extract ingredients from the input and validate them against dietary restrictions.
    
    USER INPUT (Already in English): "{frase_limpia}"
    
    ⚠️ DIETARY RULES TO ENFORCE (IN ENGLISH):
    {reglas_activas_en}
    
    CRITICAL PROCESSING INSTRUCTIONS:
    0. STRICT COMPARISON: Compare each extracted ingredient STRICTLY against the 'DIETARY RULES' above to determine rejection.
    1. EXTRACT IN ENGLISH: Extract ALL ingredients detected and place them in 'detected_english'.
    2. CLASSIFY IN ENGLISH: Distribute EVERY English ingredient name into "aceptados" (safe) or "conflictos" (prohibited). 
    3. STEP BY STEP (SPANISH): Create a "step by step" field IN SPANISH evaluating each ingredient individually.
    4. REASONING IN SPANISH: Write the "reasoning_spanish" field STRICTLY IN SPANISH. One short sentence per conflict.
    5. EMPTY LIST RULE: If there are no conflicts, "conflictos" MUST be an empty array [].

    TRAINING EXAMPLE (Use as exact JSON template):
    Input: "apple, rice and peanut"
    Rules: - STRICT APPLE_ALLERGY RULE: PROHIBITED: apple
    Output:
{{
        "detected_english": ["rice", "chicken", "potato", "chickpeas"],
        "accepted_english": ["rice", "chicken", "potato", "chickpeas"],
        "conflicts_english": [],
        "reasoning_spanish": "El pollo no está permitido en tu dieta."
    }}
    ---
    NOW ANALYZE THE REAL INPUT, APPLY RULES STRICTLY, AND RETURN THE JSON:
    """
    analisis_final = {"aceptados": [], "conflictos": [], "razonamiento": ""}
    try:
        response = client.generate(model=MODEL, prompt=prompt, format='json')
        datos_ia = json.loads(response['response'].strip())
        
        analisis_final["aceptados"] = datos_ia.get("accepted_english", datos_ia.get("aceptados", []))
        analisis_final["conflictos"] = datos_ia.get("conflicts_english", datos_ia.get("conflictos", datos_ia.get("conflicts", [])))
        analisis_final["razonamiento"] = datos_ia.get("reasoning_spanish", datos_ia.get("razonamiento", datos_ia.get("reasoning", "")))
        return analisis_final
    except Exception as e:
        log(f"⚠️ Error IA Portera: {e}")
        return analisis_final

def build_prompt_en(user_ingredients_en: List[str], fusion_recipes: List[Dict[str, Any]], user_profile: Dict[str, Any], max_new_ingredients: int) -> str:
    inspiration = {"top_retrieved_recipes": fusion_recipes}
    schema = {
        "status": "OK | NEED | NOT_POSSIBLE",
        "title_en": "string",
        "shopping_list": [{"name_en": "string", "reason_en": "string"}],
        "ingredients_used": [{"name_en": "string", "quantity_metric_en": "string"}],
        "steps_en": ["string", "..."],
        "time_minutes": 0,
        "notes_en": "string"
    }

    cuisine_logic_rules = """
UNIVERSAL CUISINE ARCHITECTURE:
- ROLE: You are a "Pantry Chef". Your goal is to find the most COHERENT dish possible.
- SELECTION OVER COMPLETION: You are NOT required to use every available ingredient. Choose only those that pair well.
- FLAVOR INTEGRITY: If an ingredient clashes with a savory base, OMIT IT.
"""
    minimal_rules = """
MINIMAL MODE (priority when ingredients are limited):
- Prefer recipes that can realistically be made with very few ingredients.
- Avoid baked goods unless the required staples are availabe.
""" if len(user_ingredients_en) <= 5 else ""

    title_grounding_rules = """
TITLE GROUNDING (STRICT):
- title_en MUST be a truthful description of the dish that can be made using ONLY ingredients_used.
"""
    diet_enforcement = f"- HEALTH RESTRICTIONS: {json.dumps(user_profile)}. If a recipe violates these and cannot be substituted, status=\"NOT_POSSIBLE\"."
    for diet, is_active in user_profile.items():
        if is_active and str(diet).lower() in RESTRICTION_MAP:
            prohibidos = ", ".join(RESTRICTION_MAP[str(diet).lower()])
            diet_enforcement += f"\n- 🚨 STRICT {diet} RULE: PROHIBITED: {prohibidos}."
        
    return f"""
Return ONLY one valid JSON object. No markdown. No ``` blocks.

{cuisine_logic_rules}
CORE RULES:
- Language: English. Use metric units.
- Available ingredients: {", ".join(user_ingredients_en)}
- You may add at most {max_new_ingredients} ingredients in shopping_list ONLY if status="NEED".
- shopping_list MUST NOT include any ingredient that is already in Available ingredients.
- If status="OK": shopping_list MUST be [].
{diet_enforcement}
- ingredients_used MUST list ONLY the actual ingredients used.
- Steps MUST NOT mention any ingredient outside of ingredients_used.

{minimal_rules}
{title_grounding_rules}

Schema:
{json.dumps(schema, ensure_ascii=False)}

Retrieved recipes (ONLY titles + overlapping ingredients):
{json.dumps(inspiration, ensure_ascii=False)}

Now output the JSON.
""".strip()

def build_json_only_retry_prompt(bad_text: str) -> str:
    return f"Return ONLY a valid JSON object. No markdown. Fix so Python json.loads() can parse it.\n\nINVALID OUTPUT:\n{(bad_text or '')[:2000]}".strip()

# =========================================================
# 4. NORMALIZACIÓN Y LÓGICA RAG (Retrieval y validación)
# =========================================================
def normalize_ing(s: str) -> str: return (s or "").strip().lower()

def canonicalize_ing(ing: str) -> str:
    x = normalize_ing(ing)
    if not x: return ""
    x = re.sub(r's$', '', x)
    x = re.sub(r'\b(chopped|minced|diced|sliced|fresh|frozen|hot|warm|cold|peeled)\b', '', x)
    if "vanilla" in x: return "vanilla"
    if "milk" in x: return "milk"
    if "butter" in x: return "butter"
    if "tomato" in x: return "tomato"
    if "potato" in x: return "potato"
    if "onion" in x: return "onion"
    if "garlic" in x: return "garlic"
    if "cheese" in x: return "cheese"
    if "chicken" in x: return "chicken"
    if "egg" in x: return "egg"
    if "beef" in x: return "beef"
    if "pork" in x: return "pork"
    if x in {"salt", "sea salt", "kosher salt"}: return "salt"
    return " ".join(x.split())

def normalize_user_ingredients(ingredients_en: List[str]) -> Set[str]:
    return {canonicalize_ing(x) for x in ingredients_en if x and str(x).strip()}

def overlap_recall(user_set: set, ner_list) -> float:
    if not user_set: return 0.0
    try: ner_set = set([canonicalize_ing(x) for x in list(ner_list)])
    except: return 0.0
    if len(ner_set) == 0: return 0
    return len(user_set.intersection(ner_set)) / len(user_set) if len(user_set) > 0 else 0

def count_missing_staples_in_recipe(ner_list, available_set: set) -> int:
    if not ner_list: return 0
    try: ner_set = set([canonicalize_ing(x) for x in list(ner_list)])
    except: return 0
    missing_critical = [m for m in (ner_set - available_set) if m not in BAKING_STAPLES]
    return len(missing_critical)

def retrieve_top_k(df_data, user_ingredients_en: list, k: int=3, search_window: int=15, min_recall: float=0.25, max_new_ingredients: int=2, staples_penalty: float=0.15) -> list:
    t0 = time.time()
    user_set = normalize_user_ingredients(user_ingredients_en)
    query_str = ", ".join(user_ingredients_en)
    query_vec = embedder.encode([query_str]).astype('float32')

    distances, candidate_indices = faiss_index.search(query_vec, 100)
    raw_candidates = df_data.iloc[candidate_indices[0]]

    refined_candidates = []
    processed_count = 0

    for _, r in raw_candidates.iterrows():
        if processed_count >= search_window: break
        raw_ner = r["ner"].tolist() if hasattr(r["ner"], "tolist") else list(r["ner"])
        clean_ner = [str(i).lower().strip() for i in raw_ner if isinstance(i, str)]

        rec = overlap_recall(user_set, clean_ner)
        if rec < min_recall: continue

        missing_count = count_missing_staples_in_recipe(clean_ner, user_set)
        if missing_count > max_new_ingredients: continue

        score = rec - (staples_penalty * missing_count)
        refined_candidates.append({
            "score": score, "rec": rec, "missing_staples": missing_count,
            "r": r, "clean_ner": clean_ner
        })
        processed_count += 1

    refined_candidates.sort(key=lambda x: x["score"], reverse=True)
    results = []
    for item in refined_candidates[:k]:
        ner_set = set([canonicalize_ing(x) for x in item["clean_ner"]])
        results.append({
            "title": str(item["r"]["title"]),
            "score": round(float(item["score"]), 3),
            "score_recall": round(float(item["rec"]), 3),
            "missing_staples": int(item["missing_staples"]),
            "common": sorted(list(user_set.intersection(ner_set))),
            "ner": item["clean_ner"]
        })
    return results

def blocked_by_profile(user_profile: dict) -> Set[str]:
    blocked = set()
    if not user_profile: return blocked
    profile_lower = {str(k).lower(): v for k, v in user_profile.items()}
    for condition, terms in RESTRICTION_MAP.items():
        if profile_lower.get(condition) is True:
            blocked |= terms
    return blocked

def validate_recipe(recipe: Dict[str, Any], available_ings: Set[str], max_new_ingredients: int, vocab: Set[str], user_profile: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errors = []
    status = (recipe.get("status") or "").strip()
    shopping_set = {canonicalize_ing(x.get("name_en", "")) for x in recipe.get("shopping_list", []) if isinstance(x, dict)}
    used_set = {canonicalize_ing(x.get("name_en", "")) for x in recipe.get("ingredients_used", []) if isinstance(x, dict)}
    
    allowed_set = set(available_ings) | shopping_set | STAPLES
    blocked_terms = blocked_by_profile(user_profile)
    clash = (used_set | shopping_set).intersection(blocked_terms)
    if clash: errors.append(f"Violates restrictions: {sorted(list(clash))}")
    if status == "OK" and shopping_set: errors.append('status="OK" but shopping_list not empty')
    if status == "NEED" and not shopping_set: errors.append('status="NEED" but shopping_list empty')
    if len(shopping_set) > max_new_ingredients: errors.append(f"Shopping list too long")
    return (len(errors) == 0), errors

def fallback_recipe_en(available_set, user_profile, notes=""):
    return {
        "status": "NOT_POSSIBLE",
        "title_en": "No reliable recipe found",
        "shopping_list": [], "ingredients_used": [], "steps_en": [], "time_minutes": 0,
        "notes_en": notes if notes else "No ha sido posible encontrar una receta."
    }

# =========================================================
# 5. PIPELINE MAESTRO Y CARGA DE DATOS OFFLINE
# =========================================================
def run_pipeline_case(df_data, vocab_data, user_ingredients: list, user_profile: dict, max_new_ingredients: int = 3, k: int = 3) -> dict:
    t0 = time.time()
    available_set = normalize_user_ingredients(user_ingredients)
    
    try:
        top = retrieve_top_k(df_data, user_ingredients, k=k, search_window=15, min_recall=0.15, max_new_ingredients=max_new_ingredients)
    except Exception as e:
        log(f"❌ Retrieval error: {e}")
        top = []

    if not top:
        return {
            "recipe_final": fallback_recipe_en(available_set, user_profile, "No recipes found."),
            "valid_post": True, "errors_post": [], "seconds_total": round(time.time() - t0, 3)
        }

    prompt = build_prompt_en(user_ingredients, top, user_profile, max_new_ingredients)
    raw = ollama_generate(prompt)

    try:
        recipe = parse_llm_json(raw)
    except Exception:
        raw_retry = ollama_generate(build_json_only_retry_prompt(raw), max_tokens=600, temperature=0.0)
        try: recipe = parse_llm_json(raw_retry)
        except: recipe = fallback_recipe_en(available_set, user_profile, "Failed to parse AI response.")

    if recipe.get("status") == "OK": 
        recipe["shopping_list"] = []
    
    ok_post, errors_post = validate_recipe(recipe, available_set, max_new_ingredients, vocab_data, user_profile)

    return {
        "recipe_final": recipe,
        "valid_post": bool(ok_post),
        "errors_post": errors_post,
        "seconds_total": round(time.time() - t0, 3),
    }

# ---------------------------------------------------------
# INICIALIZACIÓN DE LA BASE DE DATOS (Se ejecuta al arrancar)
# ---------------------------------------------------------
df = pd.DataFrame()
vocab = set()
faiss_index = None
embedder = None

def init_database():
    global df, vocab, faiss_index, embedder
    log("Iniciando carga de base de datos offline...")
    
    # Carga Embedder
    embedder = SentenceTransformer(MODEL_NAME)
    
    # Carga Parquet
    if os.path.exists(SUBSET_PATH):
        df = pd.read_parquet(SUBSET_PATH)
        df.columns = [c.lower() for c in df.columns]
        if 'ingredients' in df.columns: df.rename(columns={'ingredients': 'ner'}, inplace=True)
        # Convertir a listas si es necesario
        if not df.empty and str(type(df['ner'].iloc[0])).find('numpy.ndarray') != -1:
            df['ner'] = df['ner'].apply(list)
    else:
        log(f"⚠️ Parquet no encontrado en {SUBSET_PATH}")
        
    # Carga Vocab
    if os.path.exists(VOCAB_PATH):
        with open(VOCAB_PATH, 'r') as f: vocab = set(json.load(f))
    
    # Carga FAISS
    if os.path.exists(FAISS_INDEX_PATH):
        faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    else:
        log(f"⚠️ Índice FAISS no encontrado en {FAISS_INDEX_PATH}")
        
    log("✅ Base de datos cargada y lista para consultas.")

# Lanzar la inicialización en el momento en que Flask importe el archivo
init_database()