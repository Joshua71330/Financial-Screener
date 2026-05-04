import pandas as pd
from pathlib import Path 
import numpy as np 
from algos_IA import Regression_Ridge

class ActifFinancier: 
    def __init__(self, ticker):
        self.ticker = ticker
        self.historique = None
        self.IA= None  
        self.features_list = ['Dist_SMA', 'Dist_EMA', 'Volatilite_20j', 'RSI', 'Volume_zscore']
        self.mu = None
        self.sigma = None

    def charger_donnees(self):
        """
        Cherche le fichier CSV correspondant au ticker et lit ses données.
        """
        dossier = Path(__file__).parent 
        chemin_fichier = dossier / f"{self.ticker}.csv"
        
        if not chemin_fichier.exists():
            print(f"Erreur : Le fichier {chemin_fichier} est introuvable.")
            print(f"Avez-vous bien lancé gestion_donnees.py pour {self.ticker} ?")
            return False
            
        # creation d'un DATAFRAME 
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
            nom_colonne = f"SMA_{fenetre}"
            self.historique[nom_colonne] = self.historique['Close'].rolling(window=fenetre).mean()
            print(f"Indicateur ajouté pour {self.ticker} : {nom_colonne}")

    def calculer_EMA(self, fenetre=20, colonne='Close'):
        """
        Calcule la moyenne mobile exponentielle sur une fenêtre donnée.
        EMA = (Prix d'aujourd'hui * alpha) + (EMA d'hier * (1 - alpha)) avec alpha = 2 / (fenetre + 1)
        """
        if self.historique is not None: 
            alpha=2/(fenetre+1)  # Coefficient de lissage pour EMA
            serie=self.historique[colonne].tolist()
            ema=[serie[0]]  # Initialisation de l'EMA avec le premier prix

            for i in range(1, len(serie)):
                ema.append(serie[i] * alpha + ema[i-1] * (1 - alpha))
            self.historique[f"EMA_{fenetre}"] = ema

    def calculer_macd(self, span_court=12, span_long=26, span_signal=9):
        self.calculer_EMA(fenetre=span_court)   #  EMA_12
        self.calculer_EMA(fenetre=span_long)    #  EMA_26
        self.historique['MACD'] = self.historique['EMA_12'] - self.historique['EMA_26']
        self.calculer_EMA(fenetre=span_signal, colonne='MACD')  # crée EMA_9 du MACD = signal
        self.historique['MACD_signal'] = self.historique['EMA_9']
        self.historique['MACD_hist']   = self.historique['MACD'] - self.historique['MACD_signal']


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
        Calcule le RSI avec une boucle manuelle type EMA (sans ewm de Pandas).
        """
        if self.historique is not None:
            # 1. Différence de prix par rapport à la veille
            delta = self.historique['Close'].diff()
            
            # 2. On sépare les gains et les pertes
            # On utilise fillna(0) pour que la première case (qui est vide) devienne un 0
            gains = delta.clip(lower=0).fillna(0).tolist()
            pertes = (-delta.clip(upper=0)).fillna(0).tolist()
            
            # Note : Pour le RSI (formule de Wilder), le vrai alpha est 1/fenetre. 
            # (Si c'était un vrai EMA classique, ça serait 2/(fenetre+1))
            alpha = 1 / fenetre
            
            moyenne_gains = [gains[0]]
            moyenne_pertes = [pertes[0]]
            
            for i in range(1, len(gains)):
                mg = gains[i] * alpha + moyenne_gains[i-1] * (1 - alpha)
                moyenne_gains.append(mg)
                
                mp = pertes[i] * alpha + moyenne_pertes[i-1] * (1 - alpha)
                moyenne_pertes.append(mp)
            
            # 4. On reconvertit nos listes en colonnes Pandas pour le calcul final
            mg_series = pd.Series(moyenne_gains, index=self.historique.index)
            mp_series = pd.Series(moyenne_pertes, index=self.historique.index)
            
            # 5. Calcul du ratio et du RSI final
            rs = mg_series / mp_series
            self.historique['RSI'] = 100 - (100 / (1 + rs))
            
            print(f"Indicateur ajouté pour {self.ticker} : RSI ({fenetre} jours)")

    def calculer_volume_zscore(self, fenetre=20):
        """
        Z-score du volume : mesure si le volume d'aujourd'hui est anormal.
        Un Z-score > 2 signifie un volume très inhabituel (potentiel signal fort).
        """
        if self.historique is not None:
            moyenne_vol = self.historique['Volume'].rolling(window=fenetre).mean()
            std_vol     = self.historique['Volume'].rolling(window=fenetre).std()
            self.historique['Volume_zscore'] = (self.historique['Volume'] - moyenne_vol) / std_vol
            print(f"Indicateur ajouté pour {self.ticker} : Volume Z-score ({fenetre}j)")
    
    def preparer_donnees_IA(self):
            """
            Calcule les distances relatives, crée la Target (le futur) et nettoie le tableau.
            """
            if self.historique is None:
                return None
                
            df = self.historique.copy()
            
            # 1. Création des distances relatives (normalisation des prix)
            df['Dist_SMA'] = (df['Close'] - df['SMA_20']) / df['SMA_20']
            df['Dist_EMA'] = (df['Close'] - df['EMA_20']) / df['EMA_20']
            
            # 2. La cible (Target) : On remonte le rendement de DEMAIN sur la ligne d'aujourd'hui
            df['Target_Ret'] = df['Rendements'].shift(-1)
            
            # 3. Le coup de balai
            colonnes_utiles = self.features_list + ['Target_Ret']
            data_propre = df[colonnes_utiles].dropna()
            
            return data_propre


    def trouver_meilleur_alpha(self):
        """
        Teste une grille de valeurs Alpha pour trouver celle qui 
        maximise l'Accuracy Directionnelle (Hit Ratio) sur le Test Set.
        """
        data = self.preparer_donnees_IA()
        if data is None or data.empty:
            return
            
        X_brut = data[self.features_list].values
        y = data['Target_Ret'].values
        
        # Split (80% passé, 20% futur)
        split_idx = int(len(X_brut) * 0.8)
        X_train_brut, X_test_brut = X_brut[:split_idx], X_brut[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Standardisation stricte (calculée QUE sur le train)
        mu_train = np.mean(X_train_brut, axis=0)
        sigma_train = np.std(X_train_brut, axis=0)
        X_train = (X_train_brut - mu_train) / sigma_train
        X_test = (X_test_brut - mu_train) / sigma_train
        
        # --- GRILLE DE RECHERCHE (Grid Search) ---
        alphas_a_tester = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        
        meilleur_alpha = None
        meilleur_hit_ratio = 0.0
        
        print("\n=== DÉBUT DU GRID SEARCH POUR ALPHA ===")
        for a in alphas_a_tester:
            # 1. On crée une IA de test avec cet alpha
            test_IA = Regression_Ridge(alpha=a)
            
            # 2. On l'entraîne
            test_IA.fit(X_train, y_train)
            
            # 3. On calcule son Hit Ratio sur les données qu'elle ne connaît pas (Test)
            y_pred = test_IA.predict(X_test)
            hit_ratio = np.sum(np.sign(y_test) == np.sign(y_pred)) / len(y_test)
            
            print(f"Test Alpha = {a:8.3f} --> Hit Ratio = {hit_ratio*100:.2f} %")
            
            # 4. On sauvegarde le roi de la colline
            if hit_ratio > meilleur_hit_ratio:
                meilleur_hit_ratio = hit_ratio
                meilleur_alpha = a
                
        print(f">>> Alpha optimal retenu = {meilleur_alpha} <<<")
        
        # On met à jour l'IA "officielle" de l'actif avec le paramètre gagnant
        self.IA = Regression_Ridge(alpha=meilleur_alpha)

    
    def entrainer_IA(self):
        data = self.preparer_donnees_IA()
        if data is None or data.empty:
            print("Erreur : Pas de données pour entraîner l'IA.")
            return
            
        X_brut = data[self.features_list].values
        y = data['Target_Ret'].values
        
        # 1. On définit l'index de coupure (80% passé / 20% futur)
        split_idx = int(len(X_brut) * 0.8)
        
        # 2. CALIBRAGE  : On calcule mu et sigma uniquement sur le passé
        self.mu = np.mean(X_brut[:split_idx], axis=0)
        self.sigma = np.std(X_brut[:split_idx], axis=0)
        
        # 3. On applique cette échelle à toutes les données
        X_std = (X_brut - self.mu) / self.sigma
        
        # 4. On sépare maintenant en Train et Test
        X_train, X_test = X_std[:split_idx], X_std[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # 5. Entraînement final avec l'Alpha optimal déjà trouvé
        print(f"\n--- Entraînement final de l'IA pour {self.ticker} ---")
        self.IA.fit(X_train, y_train)
        
        # 6. Évaluation sur les données de Test (les 20% que l'IA découvre)
        self.IA.score(X_test, y_test)
    
    def predire_demain(self):

        if self.mu is None or self.sigma is None:
            print("Erreur : L'IA doit être entraînée avant de prédire.")
            return
            
        df = self.historique.copy()
        
        # On calcule les indicateurs d'aujourd'hui
        df['Dist_SMA'] = (df['Close'] - df['SMA_20']) / df['SMA_20']
        df['Dist_EMA'] = (df['Close'] - df['EMA_20']) / df['EMA_20']
        
        # On isole la TOUTE DERNIÈRE LIGNE de notre tableau (les données du jour)
        derniere_ligne = df[self.features_list].iloc[-1].values
        
        X_demain_std = (derniere_ligne - self.mu) / self.sigma
        
        # reshape(1, -1) : L'IA attend un "tableau" de lignes. On la force à voir 1 ligne.
        X_demain_std = X_demain_std.reshape(1, -1)
        
        # On lance la prédiction
        prediction = self.IA.predict(X_demain_std)[0]
        
        signe = "HAUSSE" if prediction > 0 else "BAISSE"
        print(f"\n PRÉDICTION POUR DEMAIN ({self.ticker}) : Potentielle {signe}")
        
        return prediction
    
# --- TEST DU MODULE ---
if __name__ == "__main__":
    action = ActifFinancier("AAPL")
    
    if action.charger_donnees():
        action.calculer_moyenne_mobile(fenetre=20)  # Moyenne sur 1 mois
        action.calculer_moyenne_mobile(fenetre=50)  # Moyenne sur 2 mois et demi
        action.calculer_rendements()
        action.calculer_EMA(fenetre=20,colonne='Close')  # EMA sur 1 mois
        action.calculer_volatilite_historique(fenetre=20)  # Volatilité sur 1 mois
        action.calculer_rsi(fenetre=14)  # RSI sur 2 semaines
        action.calculer_macd()
        action.calculer_volume_zscore(fenetre=20)  # Volume Z-score sur 1 mois
        
        print("\nAperçu des données avec les nouveaux indicateurs :")
        print(action.historique[['Close', 'SMA_20', 'SMA_50', 'Rendements', 'EMA_20', 'Volatilite_20j', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist']].tail())

        action.trouver_meilleur_alpha()
        # 2. Entraînement de l'IA (va afficher les stats R2, MAE, Hit Ratio)
        action.entrainer_IA()
        action.predire_demain()
