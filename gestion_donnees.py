import yfinance as yf
from pathlib import Path

def telecharger_historique(ticker):
    """
    Télécharge l'historique d'une action et le sauvegarde en CSV
    directement dans le dossier courant du projet.
    """

    dossier = Path(__file__).parent 

    chemin_fichier = dossier / f"{ticker}.csv"

    action = yf.Ticker(ticker)
    print(f"Connexion à Yahoo Finance pour {ticker}...")
    
    # On récupère 2 ans d'historique
    df = action.history(period="2y") 

    if df.empty:
        print(f"Erreur : Aucune donnée trouvée pour {ticker}.")
        return False

    df.to_csv(chemin_fichier)
    print(f"Succès : Données de {ticker} sauvegardées sous le nom '{chemin_fichier}'")
    return True

# --- TEST DU MODULE ---
if __name__ == "__main__":
    print("--- Téléchargement des données ---")
    telecharger_historique("AAPL")
    telecharger_historique("MSFT")
