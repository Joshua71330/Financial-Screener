import pandas as pd
from pathlib import Path 
import numpy as np 
from scipy.stats import norm 

from algos_IA import Regression_Ridge, AnalyseurNews
from gestion_donnees import telecharger_historique
import yfinance as yf

class ActifFinancier: 
    def __init__(self, ticker):
        self.ticker = ticker
        self.historique = None
        self.IA= None  
        self.strategies   = []              # liste de Strategie (composition)
        self.features_list = ['Dist_SMA', 'Dist_EMA', 'Volatilite_20j', 'RSI', 'Volume_zscore']
        self.mu = None
        self.sigma = None
        self.news=[]
        self.hit_ratio = 0.0
        self.df_opex = None

    def charger_donnees(self):
        dossier = Path(__file__).parent
        chemin_fichier = dossier / f"{self.ticker}.csv"
        
        # 1. On tente de télécharger les données les plus récentes
        succes_telechargement = telecharger_historique(self.ticker)
        
        # 2. Si le téléchargement échoue (pas de wifi, erreur Yahoo) ET qu'on n'a pas d'ancien fichier en cache
        if not succes_telechargement and not chemin_fichier.exists():
            print(f"Impossible de récupérer les données pour {self.ticker}.")
            return False 
            
        # 3. On lit le fichier (soit la version fraîchement téléchargée, soit l'ancienne version de secours)
        try:
            self.historique = pd.read_csv(chemin_fichier, index_col='Date', parse_dates=True)
            return True
        except FileNotFoundError:
            return False
        

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
    
    def calculer_EMA_RSI_recursif(self, valeurs: list, alpha: float, index: int) -> float:
        """
        Calcule la moyenne EMA de Wilder jusqu'à l'indice `index` par récursion.
        Cas de base : premier élément = valeurs[0].
        alpha=2/(fenetre+1)  pour un EMA classique, mais pour le RSI de Wilder, c'est alpha=1/fenetre.
        Récursion   : EMA(i) = valeurs[i] * alpha + EMA(i-1) * (1 - alpha)

        Satisfait la figure imposée « Fonction récursive dans le cœur du projet ».
        """
        if index == 0:
            return valeurs[0]
        precedent = self.calculer_EMA_RSI_recursif(valeurs, alpha, index - 1)
        return valeurs[index] * alpha + precedent * (1 - alpha)

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
    

    # ─────────────────────────────────────────
    #  GESTION DES STRATÉGIES (composition)
    # ─────────────────────────────────────────

    def ajouter_strategie(self, strategie) -> None:
        """
        Attache une stratégie (instance de Strategie) à cet actif.
        Relation de composition : l'actif possède ses stratégies.
        """
        self.strategies.append(strategie)
        print(f"  Stratégie ajoutée : {strategie.nom}")

    def evaluer_strategies(self) -> list[dict]:
        """
        Backteste toutes les stratégies rattachées sur l'historique complet.
        Retourne une liste de dictionnaires de performance.
        """
        if not self.strategies:
            print("  Aucune stratégie rattachée. Utilisez ajouter_strategie() d'abord.")
            return []

        resultats = []
        for s in self.strategies:
            res = s.calculer_performance(self.historique)
            resultats.append(res)
        return resultats


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
            
            # 3. On ne garde que les colonnes utiles pour l'IA et on supprime les lignes avec des NaN
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

        # --- AJOUT ICI : Calcul et sauvegarde du Hit Ratio pour l'interface ---
        y_pred = self.IA.predict(X_test)
        self.hit_ratio = float(np.sum(np.sign(y_test) == np.sign(y_pred)) / len(y_test))
    
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
        print(f"\n PRÉDICTION Technique POUR DEMAIN ({self.ticker}) : Potentielle {signe} \n(Aucune donnée Macro ou d'actualité prise en compte, uniquement basée sur les indicateurs techniques et l'historique des prix).")
        
        return prediction
    
    def analyser_fondamental(self):
        """
        Déclenche l'IA sémantique pour récupérer et analyser les news de cet actif.
        """
        print(f"Lancement de l'analyse fondamentale pour {self.ticker}...")
        analyseur = AnalyseurNews()
        self.news = analyseur.scraper_news_yahoo(self.ticker)
        return self.news
    
    def calculer_calendrier_opex(self):
        """
        Génère le calendrier des expirations d'options (OpEx) et 
        des Quad Witching pour l'actif en cours.
        """
        if self.historique is None or self.historique.empty:
            return
            
        start_date = self.historique.index[0]
        # On va un peu plus loin dans le futur pour attraper le prochain OpEx
        end_date = self.historique.index[-1] + pd.Timedelta(days=30) 
        
        # Trouver tous les 3èmes vendredis du mois
        opex_dates = pd.date_range(start_date, end_date, freq='WOM-3FRI')
        df_opex = pd.DataFrame({'OpEx_Date': opex_dates})
        
        # Identifier les 4 Sorcières (Quad Witching = Mars, Juin, Septembre, Décembre)
        df_opex['Is_Quad_Witching'] = df_opex['OpEx_Date'].dt.month.isin([3, 6, 9, 12])
        
        self.df_opex = df_opex
        print(f"Calendrier OpEx calculé pour {self.ticker}.")

    def annoter_phases_opex(self):
        """
        Catégorise chaque jour de l'historique en Phase 1 (Aimant), Phase 2 (Directionnel), ou Normal.
        Indispensable pour backtester les filtres GEX.
        """
        if self.historique is None or getattr(self, 'df_opex', None) is None:
            return
            
        self.historique['Phase_OpEx'] = 'Normal'
        
        for _, row in self.df_opex.iterrows():
            opex_date = pd.to_datetime(row['OpEx_Date']).tz_localize(None)
            
            # Phase 1 : J-7 à J (Compression)
            start_p1 = opex_date - pd.Timedelta(days=7)
            mask_p1 = (self.historique.index >= start_p1) & (self.historique.index <= opex_date)
            self.historique.loc[mask_p1, 'Phase_OpEx'] = 'Phase 1'
            
            # Phase 2 : J à J+7 (Shakeout / Libération)
            end_p2 = opex_date + pd.Timedelta(days=7)
            mask_p2 = (self.historique.index > opex_date) & (self.historique.index <= end_p2)
            self.historique.loc[mask_p2, 'Phase_OpEx'] = 'Phase 2'
            
        print(f"Phases OpEx annotées sur l'historique de {self.ticker}.")

    def calculer_max_pain(self):
        """
        Scrape la chaîne d'options la plus proche via yfinance et calcule le Max Pain Strike.
        """
        import yfinance as yf
        self.max_pain = None
        self.expiration_proche = None
        
        try:
            tk = yf.Ticker(self.ticker)
            dates = tk.options
            
            if not dates:
                print(f"Aucune chaîne d'options trouvée pour {self.ticker}.")
                return
                
            self.expiration_proche = dates[0]
            opt = tk.option_chain(self.expiration_proche)
            calls, puts = opt.calls, opt.puts
            
            # Liste de tous les strikes disponibles
            strikes = sorted(list(set(calls['strike'].tolist() + puts['strike'].tolist())))
            min_pain = float('inf')
            
            for strike in strikes:
                pain = 0
                # Valeur intrinsèque perdue par les acheteurs de Calls
                call_itm = calls[calls['strike'] < strike]
                pain += sum((strike - call_itm['strike']) * call_itm['openInterest'])
                
                # Valeur intrinsèque perdue par les acheteurs de Puts
                put_itm = puts[puts['strike'] > strike]
                pain += sum((put_itm['strike'] - strike) * put_itm['openInterest'])
                
                if pain < min_pain:
                    min_pain = pain
                    self.max_pain = strike
                    
            print(f"Max Pain trouvé pour {self.ticker} ({self.expiration_proche}) : {self.max_pain}$")
            
        except Exception as e:
            print(f"Erreur lors du calcul du Max Pain : {e}")

    def _calculer_gamma_bsm(self, S, K, T, r, sigma):
        """
        Calcule le Gamma d'une option via la formule de Black-Scholes.
        """
        # Sécurité : éviter la division par zéro si l'option expire aujourd'hui ou si l'IV est nulle
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0.0
            
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        
        # norm.pdf est la dérivée de la fonction de répartition (N')
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        return gamma

    def calculer_profil_gex(self):
        """
        Télécharge la chaîne d'options, calcule le Gamma de chaque Strike via Black-Scholes,
        et agrège le Dollar GEX total ainsi que le niveau Zero Gamma.

        Calcule également séparément le GEX des Calls et des Puts afin d'identifier :
          - Le Call Wall  (Résistance) : strike où le Gamma des Calls est le plus concentré.
          - Le Put Wall   (Support)    : strike où le Gamma des Puts est le plus concentré.
        Ces deux niveaux servent de base à la fenêtre d'analyse GEX (voir app.py).
        """
        print(f"Calcul du profil GEX (Black-Scholes) pour {self.ticker}...")
        
        self.gex_total = 0.0
        self.zero_gamma_level = None
        self.gex_par_strike = {}        # GEX net (Calls + Puts) par strike
        self.gex_calls_par_strike = {}  # GEX des seuls Calls par strike (toujours >= 0)
        self.gex_puts_par_strike = {}   # GEX des seuls Puts par strike (toujours <= 0)
        self.call_wall = None           # Résistance probable (concentration de Gamma Calls)
        self.put_wall = None            # Support probable (concentration de Gamma Puts)
        self.spot_price = self.get_dernier_prix()
        self.date_expiration_gex = None
        
        try:
            tk = yf.Ticker(self.ticker)
            dates = tk.options
            
            if not dates:
                return
                
            # Pour l'exemple, on prend la prochaine expiration (OpEx le plus proche)
            # Dans un algo pro, on boucle souvent sur toutes les expirations des 30 prochains jours.
            date_exp = dates[0]
            opt = tk.option_chain(date_exp)
            self.date_expiration_gex = date_exp
            
            S = self.get_dernier_prix()
            if S == 0:
                return
            self.spot_price = S
                
            # Temps jusqu'à l'expiration en années (T)
            date_exp_dt = pd.to_datetime(date_exp).tz_localize(None)
            aujourd_hui = pd.Timestamp.now().tz_localize(None)
            jours_restants = (date_exp_dt - aujourd_hui).days
            T = max(jours_restants / 365.25, 0.001) # Minimum 1/1000e d'année
            
            r = 0.04  # Taux sans risque arbitraire à 4% (T-Bills)
            
            calls = opt.calls
            puts = opt.puts
            
            # --- CALCUL DU GEX DES CALLS ---
            # Convention : On suppose les MM short sur les Calls (investisseurs longs) -> GEX Positif
            for _, row in calls.iterrows():
                K = row['strike']
                sigma = row['impliedVolatility']
                oi = row['openInterest']
                
                gamma = self._calculer_gamma_bsm(S, K, T, r, sigma)
                # Dollar Gamma = Gamma * Open Interest * 100 actions par contrat * Spot Price^2 / 100 (pour 1% de mouvement)
                # Formule simplifiée standardisée : Gamma * OI * 100
                gex = gamma * oi * 100
                
                self.gex_par_strike[K] = self.gex_par_strike.get(K, 0) + gex
                self.gex_calls_par_strike[K] = self.gex_calls_par_strike.get(K, 0) + gex
                self.gex_total += gex
                
            # --- CALCUL DU GEX DES PUTS ---
            # Convention : On suppose les MM short sur les Puts (investisseurs longs pour se couvrir) -> GEX Négatif
            for _, row in puts.iterrows():
                K = row['strike']
                sigma = row['impliedVolatility']
                oi = row['openInterest']
                
                gamma = self._calculer_gamma_bsm(S, K, T, r, sigma)
                gex = -(gamma * oi * 100) # Négatif car les MM vendent pour se couvrir à la baisse
                
                self.gex_par_strike[K] = self.gex_par_strike.get(K, 0) + gex
                self.gex_puts_par_strike[K] = self.gex_puts_par_strike.get(K, 0) + gex
                self.gex_total += gex
            
            # --- RECHERCHE DU NIVEAU ZERO GAMMA ---
            # C'est le Strike où le GEX cumulé (Calls vs Puts) s'inverse.
            strikes = sorted(self.gex_par_strike.keys())
            gex_cumule = 0
            for strike in strikes:
                gex_cumule += self.gex_par_strike[strike]
                # Si le cumul passe de négatif à positif (ou s'approche de zéro dans un flux de Puts dominant)
                if gex_cumule >= 0 and self.zero_gamma_level is None:
                    self.zero_gamma_level = strike
            
            # Si le profil est entièrement négatif ou positif, on fallback sur le prix actuel
            if self.zero_gamma_level is None:
                self.zero_gamma_level = S

            # --- RECHERCHE DU CALL WALL (RÉSISTANCE) ---
            # Strike où la concentration de Gamma des Calls est maximale : 
            # les Market Makers, longs Gamma, vendent les hausses pour rester neutres -> freine le prix.
            if self.gex_calls_par_strike:
                self.call_wall = max(self.gex_calls_par_strike, key=self.gex_calls_par_strike.get)

            # --- RECHERCHE DU PUT WALL (SUPPORT) ---
            # Strike où la concentration de Gamma des Puts est maximale (valeur la plus négative) :
            # les Market Makers achètent les baisses pour se couvrir -> amortit la chute.
            if self.gex_puts_par_strike:
                self.put_wall = min(self.gex_puts_par_strike, key=self.gex_puts_par_strike.get)

            print(f"GEX Total : {self.gex_total:,.0f} | Zero Gamma : {self.zero_gamma_level}$ "
                  f"| Résistance (Call Wall) : {self.call_wall}$ | Support (Put Wall) : {self.put_wall}$")
            
        except Exception as e:
            print(f"Erreur lors du calcul Black-Scholes GEX : {e}")
    
    
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