import pandas as pd
from pathlib import Path 
import numpy as np 

class ActifFinancier: 
    def __init__(self, ticker):
        self.ticker = ticker
        self.historique = None
    
    def charger_donnees(self):
        """
        Cherche le fichier CSV correspondant au ticker et lit ses données.
        """
        # On trouve dynamiquement le dossier actuel
        dossier = Path(__file__).parent 
        chemin_fichier = dossier / f"{self.ticker}.csv"
        
        if not chemin_fichier.exists():
            print(f"Erreur : Le fichier {chemin_fichier} est introuvable.")
            print(f"Avez-vous bien lancé gestion_donnees.py pour {self.ticker} ?")
            return False
            
        # --- LECTURE DE FICHIER (Figure imposée validée !) creation d'un DATAFRAME ---
        self.historique = pd.read_csv(chemin_fichier, index_col="Date", parse_dates=True)
        
        print(f"Succès : {len(self.historique)} jours de cotation chargés pour {self.ticker}.")
        return True

    def get_dernier_prix(self):
        """
        Renvoie le prix de clôture ('Close') le plus récent.
        """
        if self.historique is not None and not self.historique.empty:
            return self.historique['Close'].iloc[-1]
        return 0.0
    
    def calculer_moyenne_mobile(self, fenetre=20):
        """
        Calcule la moyenne mobile simple sur une fenêtre donnée.
        """

        if self.historique is not None:
            # On crée un nom de colonne dynamique, ex: "SMA_20"
            nom_colonne = f"SMA_{fenetre}"
            
            # rolling(window) crée la fenêtre glissante, mean() calcule la moyenne
            self.historique[nom_colonne] = self.historique['Close'].rolling(window=fenetre).mean()
            print(f"Indicateur ajouté pour {self.ticker} : {nom_colonne}")

    def calculer_EMA(self, fenetre=20):
        """
        Calcule la moyenne mobile exponentielle sur une fenêtre donnée.
        EMA = (Prix d'aujourd'hui * alpha) + (EMA d'hier * (1 - alpha)) avec alpha = 2 / (fenetre + 1)
        """
        if self.historique is not None: 
            alpha=2/(fenetre+1)  # Coefficient de lissage pour EMA
            prix=self.historique['Close'].tolist()
            ema=[prix[0]]  # Initialisation de l'EMA avec le premier prix

            for i in range(1, len(prix)):
                ema.append(prix[i] * alpha + ema[i-1] * (1 - alpha))
            self.historique[f"EMA_{fenetre}"] = ema


    def calculer_rendements(self):
        """
        Calcule le rendement quotidien basé sur les prix de clôture.
        """
        if self.historique is not None:
            prix_aujourd_hui = self.historique['Close']
            prix_hier = self.historique['Close'].shift(1)  
            self.historique['Rendements'] = (prix_aujourd_hui - prix_hier) / prix_hier
            print(f"Rendements calculé pour {self.ticker}.")

    def calculer_volatilite_historique(self, fenetre=20):
        """
        Calcule la volatilité annualisée glissante basée sur les rendements.
        """
        if self.historique is not None:
            if 'Rendements' not in self.historique.columns:
                print(f" Erreur : Il faut calculer les rendements avant la volatilité pour {self.ticker}.")
                return
    
        # --- ÉTAPE 1 : Écart-type quotidien (Moving Standard Deviation) ---
        # Mesure de la dispersion des rendements autour de leur moyenne sur N jours.
        sigma_quotidien = self.historique['Rendements'].rolling(window=fenetre).std()
        
        # --- ÉTAPE 2 : Variance quotidienne ---
        variance_quotidienne = sigma_quotidien ** 2
        
        # --- ÉTAPE 3 : Variance annuelle ---
        # On multiplie par 252 car les variances s'additionnent sur le temps (hypothèse IID : indépendants et identiquement distribués).
        variance_annuelle = variance_quotidienne * 252
        
        # --- ÉTAPE 4 : Volatilité annuelle (Retour à l'écart-type) ---
        nom_colonne = f"Volatilite_{fenetre}j"
        self.historique[nom_colonne] = np.sqrt(variance_annuelle)
        
        print(f" Volatilité calculée pour {self.ticker} via la variance (Annualisation par racine de T).")

    def calculer_rsi(self, fenetre=14):
        """
        Calcule le RSI (Momentum - Oscillateur 0 à 100).
        """
        if self.historique is not None:
            # 1. Différence de prix par rapport à la veille
            delta = self.historique['Close'].diff()
            
            # 2. On sépare les jours de hausse (gains) et de baisse (pertes)
            gains = delta.clip(lower=0)
            pertes = -delta.clip(upper=0)
            
            # 3. Moyenne exponentielle des gains et des pertes (formule stricte de Wilder)
            moyenne_gains = gains.ewm(alpha=1/fenetre, adjust=False).mean()
            moyenne_pertes = pertes.ewm(alpha=1/fenetre, adjust=False).mean()
            
            # 4. Calcul du ratio (Relative Strength) puis du RSI final
            rs = moyenne_gains / moyenne_pertes
            self.historique['RSI'] = 100 - (100 / (1 + rs))
            print(f"Indicateur ajouté pour {self.ticker} : RSI ({fenetre} jours)")
    

    def entrainer_ia(self, alpha=1.0):
        if self.historique is None: return

        # 1. Préparation des données (Feature Engineering)
        df = self.historique.copy()
        df['Dist_SMA'] = (df['Close'] - df['SMA_20']) / df['SMA_20'] # Distance relative au SMA 20 jours
        df['Dist_EMA'] = (df['Close'] - df['EMA_20']) / df['EMA_20'] # Distance relative à l'EMA 20 jours
        df['Target_Ret'] = df['Rendements'].shift(-1) # Rendement du jour suivant comme cible (Objectif de prédiction de l'IA)
        
        features = ['Dist_SMA', 'Dist_EMA', 'Volatilite_20j', 'Rendements', 'RSI']
        data = df[features + ['Target_Ret']].dropna()
        
        X = data[features].values
        y = data['Target_Ret'].values

        # 2. STANDARDISATION MANUELLE (Crucial pour Ridge)
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0)
        X_std = (X - self.mu) / self.sigma

        # Ajouter une colonne de 1 pour l'Intercept (biais)
        X_std = np.column_stack([np.ones(X_std.shape[0]), X_std])

        # 3. L'ÉQUATION NORMALE DE RIDGE
        # On ne pénalise pas le premier coefficient (l'intercept)
        n_features = X_std.shape[1]
        A = alpha * np.eye(n_features)
        A[0, 0] = 0 

        # Formule : (X.T @ X + alpha*I)^-1 @ X.T @ y
        self.theta = np.linalg.inv(X_std.T @ X_std + A) @ X_std.T @ y
        
        print(f" Modèle Ridge NumPy entraîné. Coefficients : {self.theta}")

    def predire_prochain_rendement(self, x_input):
        """
        x_input : liste des dernières valeurs [Dist_SMA, Dist_EMA, Vol, Rend, RSI]
        """
        # 1. Standardisation avec les paramètres d'entraînement
        x_std = (x_input - self.mu) / self.sigma
        
        # 2. Ajout du 1 pour l'intercept
        x_final = np.insert(x_std, 0, 1)
        
        # 3. Produit scalaire
        return np.dot(x_final, self.theta)
    
# --- TEST DU MODULE ---
if __name__ == "__main__":
    action = ActifFinancier("AAPL")
    
    if action.charger_donnees():
        # On appelle nos nouvelles méthodes !
        action.calculer_moyenne_mobile(fenetre=20)  # Moyenne sur 1 mois
        action.calculer_moyenne_mobile(fenetre=50)  # Moyenne sur 2 mois et demi
        action.calculer_rendements()
        action.calculer_EMA(fenetre=20)  # EMA sur 1 mois
        action.calculer_volatilite_historique(fenetre=20)  # Volatilité sur 1 mois
        action.calculer_rsi(fenetre=14)  # RSI sur 2 semaines
        
        # On affiche les 5 dernières lignes pour vérifier que les colonnes ont bien été ajoutées
        print("\nAperçu des données avec les nouveaux indicateurs :")
        print(action.historique[['Close', 'SMA_20', 'SMA_50', 'Rendements', 'EMA_20', 'Volatilite_20j', 'RSI']])
