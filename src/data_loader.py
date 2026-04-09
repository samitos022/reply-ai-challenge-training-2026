import pandas as pd
import json
import os
import numpy as np

# ==========================================
# LIVELLO 1: LOGICA E MATEMATICA CORE
# ==========================================

def load_json(filepath):
    """Utility per caricare JSON in modo sicuro."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)



def get_trend_label(series):
    """
    Confronta la media storica (tutto tranne ultimi 3) con la media recente (ultimi 3).
    Ritorna: media_storica, media_recente, etichetta_trend
    """
    if len(series) < 4:
        avg = np.mean(series) if len(series) > 0 else 0
        return avg, avg, "STABLE (Not enough data)"
    
    baseline = np.mean(series[:-3])
    recent = np.mean(series[-3:])
    
    # Evitiamo divisioni per zero
    if baseline == 0:
        return baseline, recent, "STABLE"
        
    delta_pct = ((recent - baseline) / baseline) * 100
    
    if delta_pct <= -25: return baseline, recent, "SEVERE DROP"
    if delta_pct <= -10: return baseline, recent, "Moderate Drop"
    if delta_pct >= 25:  return baseline, recent, "SEVERE SPIKE"
    if delta_pct >= 10:  return baseline, recent, "Moderate Spike"
    
    return baseline, recent, "STABLE"

def check_escalation(events):
    """
    Controlla se gli eventi medici recenti sono più gravi del solito.
    """
    severity = {
        'routine check-up': 1, 'lifestyle coaching session': 1,
        'preventive screening': 2, 'specialist consultation': 4,
        'follow-up assessment': 5
    }
    scores = [severity.get(e, 0) for e in events]
    
    if len(scores) < 3: return "None"
    
    recent_severity = np.mean(scores[-3:])
    if recent_severity >= 4:
        return "CRITICAL (Specialist loop detected)"
    return "Normal Routine"


# ==========================================
# LIVELLO 2: ESTRAZIONE DATI PANDAS -> DICT
# ==========================================

def extract_status_features(status_df, citizen_id):
    """Filtra i dati di salute e restituisce un dizionario strutturato."""
    df = status_df[status_df['CitizenID'] == citizen_id].copy()
    if df.empty:
        return {"error": "No status data"}

    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values('Timestamp')
    
    events = df['EventType'].tolist()
    
    base_pa, rec_pa, trend_pa = get_trend_label(df['PhysicalActivityIndex'].values)
    base_sq, rec_sq, trend_sq = get_trend_label(df['SleepQualityIndex'].values)
    base_ee, rec_ee, trend_ee = get_trend_label(df['EnvironmentalExposureLevel'].values)

    return {
        "escalation_flag": check_escalation(events),
        "recent_events": events[-3:],
        "physical_activity": {"recent": round(rec_pa, 1), "trend": trend_pa},
        "sleep_quality": {"recent": round(rec_sq, 1), "trend": trend_sq},
        "environmental_stress": {"recent": round(rec_ee, 1), "trend": trend_ee}
    }

def extract_location_features(locations_df, citizen_id, home_city):
    """Filtra i dati di geolocalizzazione e restituisce un dizionario."""
    df = locations_df[locations_df['user_id'] == citizen_id].copy()
    if df.empty:
        return {"error": "No location data"}

    cities_visited = df['city'].unique().tolist()
    travels = [c for c in cities_visited if c != home_city]
    
    # Rilevamento di isolamento recente (si muoveva e ora si è fermato?)
    isolation = "No"
    if len(df) > 10:
        first_half_cities = df.iloc[:len(df)//2]['city'].nunique()
        last_5_records_cities = df.tail(5)['city'].nunique()
        if first_half_cities > 1 and last_5_records_cities == 1:
            isolation = "YES (Sudden confinement)"

    return {
        "unique_cities": len(cities_visited),
        "travels": travels,
        "isolation_detected": isolation
    }


# ==========================================
# LIVELLO 3: FORMATTAZIONE PER L'LLM
# ==========================================

def format_citizen_record(c_id, profile, status, loc):
    """
    Trasforma i dizionari in una stringa pulita YAML-style,
    perfetta per far ragionare un LLM senza sprecare token.
    """
    return f"""CITIZEN: {c_id}
PROFILE:
  Age: {profile['age']}
  Job: {profile['job']}
  Home City: {profile['city']}
HEALTH_TRENDS:
  Escalation: {status.get('escalation_flag', 'N/A')}
  Recent_Events: {status.get('recent_events', 'N/A')}
  Physical_Activity: {status.get('physical_activity', 'N/A')}
  Sleep_Quality: {status.get('sleep_quality', 'N/A')}
  Stress_Exposure: {status.get('environmental_stress', 'N/A')}
MOBILITY_TRENDS:
  Unique_Cities: {loc.get('unique_cities', 'N/A')}
  Isolation_Detected: {loc.get('isolation_detected', 'N/A')}
  Travels: {loc.get('travels', 'None')}
"""


# ==========================================
# ENTRY POINT PRINCIPALE
# ==========================================

def load_and_preprocess_data():
    """Legge i CSV e orchestra l'estrazione per tutti i cittadini."""
    print("[*] Caricamento Dati e Feature Engineering...")
    base_dir = "data/inputs/public_lev_1"
    
    users = load_json(os.path.join(base_dir, "users.json"))
    status_df = pd.read_csv(os.path.join(base_dir, "status.csv"))
    
    with open(os.path.join(base_dir, "locations.json"), 'r') as f:
        locations_df = pd.DataFrame(json.load(f))
    
    summaries = {}
    CURRENT_YEAR = 2026 
    
    for user in users:
        c_id = user['user_id']
        home_city = user.get('residence', {}).get('city', 'Unknown')
        
        profile_dict = {
            "age": CURRENT_YEAR - user.get('birth_year', 2000),
            "job": user.get('job', 'Unknown'),
            "city": home_city
        }
        
        status_dict = extract_status_features(status_df, c_id)
        loc_dict = extract_location_features(locations_df, c_id, home_city)
        
        # Generiamo il testo finale per l'LLM
        summaries[c_id] = format_citizen_record(c_id, profile_dict, status_dict, loc_dict)
        
    return summaries


# ==========================================
# AREA DI TEST (Eseguibile direttamente)
# ==========================================
if __name__ == "__main__":
    # Se esegui questo file con `python src/data_loader.py` dal terminale, 
    # farà un test stampando il risultato, così vedi esattamente cosa manderai all'LLM.
    
    # Assicurati di lanciare il comando dalla root del progetto (Sandbox_2026)
    print("=== TEST DATA LOADER ===")
    dati_pronti = load_and_preprocess_data()
    
    print(f"\nGenerati riassunti per {len(dati_pronti)} cittadini.\n")
    
    # Stampiamo il primo cittadino come esempio
    primo_id = list(dati_pronti.keys())[0] # Stampiamo il cittadino WNACROYX (che sappiamo avere problemi)
    print("ESEMPIO DI PAYLOAD PER LLM:")
    print("-" * 40)
    print(dati_pronti[primo_id])
    print("-" * 40)