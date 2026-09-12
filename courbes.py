import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# Remplacez 'modele_actif' par le vrai nom du fichier Python où se trouve votre classe
from modeles import ActifFinancier 

def tracer_graphiques_complets(ticker):
    print(f"--- Préparation des graphiques pour {ticker} ---")
    
    # 1. Initialisation et calculs via VOTRE classe
    action = ActifFinancier(ticker)
    if not action.charger_donnees():
        return

    # On lance vos méthodes pour générer les colonnes
    action.calculer_moyenne_mobile(fenetre=20)
    action.calculer_EMA(fenetre=20)
    action.calculer_rendements()
    action.calculer_volatilite_historique(fenetre=20)
    action.calculer_rsi(fenetre=14)
    action.calculer_macd()
    
    # On récupère le tableau final rempli de données
    df = action.historique.copy()
    
    # Optionnel : on peut limiter l'affichage aux 250 derniers jours (1 an) pour plus de lisibilité
    df = df.tail(252)

    # 2. TRACÉ DES GRAPHIQUES (4 fenêtres empilées)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 12), sharex=True, 
                                             gridspec_kw={'height_ratios': [3, 1, 1, 1]})
    fig.suptitle(f'Tableau de Bord Technique - {ticker}', fontsize=16, fontweight='bold')

    # --- Graphique 1 : Le Prix et les Moyennes ---
    ax1.plot(df.index, df["Close"], label="Prix (Close)", color="black", linewidth=1.5)
    ax1.plot(df.index, df["SMA_20"], label="SMA 20", color="blue", linestyle="--", alpha=0.7)
    ax1.plot(df.index, df["EMA_20"], label="EMA 20", color="orange", linestyle="-.", alpha=0.7)
    ax1.set_ylabel("Prix ($)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # --- Graphique 2 : Le MACD (Vos colonnes personnalisées) ---
    ax2.plot(df.index, df["MACD"], label="MACD", color="blue")
    ax2.plot(df.index, df["MACD_signal"], label="Signal", color="orange")
    # Pour l'histogramme, on met en vert si positif, rouge si négatif
    couleurs_macd = ['green' if val >= 0 else 'red' for val in df["MACD_hist"]]
    ax2.bar(df.index, df["MACD_hist"], color=couleurs_macd, alpha=0.5, label="Histogramme")
    ax2.set_ylabel("MACD")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    # --- Graphique 3 : Le RSI ---
    ax3.plot(df.index, df["RSI"], label="RSI (14j)", color="purple")
    ax3.axhline(70, color='red', linestyle='--', alpha=0.5) # Zone de surachat
    ax3.axhline(30, color='green', linestyle='--', alpha=0.5) # Zone de survente
    ax3.fill_between(df.index, y1=30, y2=70, color='purple', alpha=0.05) # Grise la zone neutre
    ax3.set_ylabel("RSI")
    ax3.set_ylim(0, 100)
    ax3.legend(loc="upper left")
    ax3.grid(True, alpha=0.3)

    # --- Graphique 4 : La Volatilité ---
    ax4.plot(df.index, df["Volatilite_20j"], label="Volatilité (20j)", color="red")
    ax4.set_ylabel("Volatilité")
    ax4.legend(loc="upper left")
    ax4.grid(True, alpha=0.3)
    
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    fig.autofmt_xdate(rotation=45)

    # Ajustement final et affichage
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    tracer_graphiques_complets("AAPL")
