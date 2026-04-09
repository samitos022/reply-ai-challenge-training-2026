import os
from src.tracker import generate_session_id, langfuse_client
from src.data_loader import load_and_preprocess_data
from src.agents import evaluate_citizen

def main():
    print("========================================")
    print(" REPLY MIRROR - THE EYE (AI SYSTEM) ")
    print("========================================\n")
    
    # 1. Generazione ID Sessione per il tracciamento dei costi
    session_id = generate_session_id()
    print(f"[*] Session ID generato: {session_id}")
    print(f"[*] Usa 'python check_traces.py {session_id}' a fine run per i costi.\n")
    
    # 2. Caricamento Dati
    citizens_data = load_and_preprocess_data()
    total_citizens = len(citizens_data)
    print(f"\n[*] Trovati {total_citizens} cittadini da analizzare.")
    
    # 3. Analisi Multi-Agente
    citizens_needing_intervention = []
    
    print("[*] Avvio analisi AI...\n")
    for i, (citizen_id, data) in enumerate(citizens_data.items(), 1):
        print(f"  Analisi {i}/{total_citizens} -> Cittadino: {citizen_id}...", end=" ")
        
        # Chiamata all'agente (tracciata su Langfuse in automatico)
        prediction = evaluate_citizen(session_id, citizen_id, data)
        
        if prediction == 1:
            citizens_needing_intervention.append(citizen_id)
            print("Esito: 1 (Intervento Richiesto)")
        else:
            print("Esito: 0 (Monitoraggio Standard)")

    # 4. Assicura che i dati di Langfuse vengano inviati al server
    print("\n[*] Sincronizzazione telemetria costi con Langfuse...")
    langfuse_client.flush()
    
    # 5. Generazione Output File
    output_dir = "data/outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "submission.txt")
    
    print(f"[*] Generazione file di output in {output_file}...")
    with open(output_file, "w") as f:
        for cid in citizens_needing_intervention:
            f.write(f"{cid}\n")
            
    print("\n========================================")
    print(" ESECUZIONE COMPLETATA CON SUCCESSO! ")
    print("========================================")

if __name__ == "__main__":
    main()