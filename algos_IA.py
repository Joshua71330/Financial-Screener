import numpy as np
import urllib.request
import urllib.error
import re
import json

class Regression_Ridge:
    #y=theta*X + alpha*||theta||^2
    def __init__(self,alpha):
        self.alpha = alpha
        self.theta = None  # Coefficients du modèle
    
    def fit(self,X,y):
        X_biais=np.column_stack([np.ones(X.shape[0]),X]) # Ajout de la colonne de biais
        n_features=X_biais.shape[1]

        I=np.eye(n_features)
        I[0,0]=0 # On penalise pas le biais.

        # Formule de la régression Ridge : theta = (X^T X + alpha*I)^-1 X^T y
        self.theta=np.linalg.inv(X_biais.T @ X_biais + self.alpha*I) @ X_biais.T @ y
        
    def predict(self,X):
        if self.theta is None:
            raise ValueError("Le modèle n'est pas encore entraîné.")
        
        X_biais=np.column_stack([np.ones(X.shape[0]),X]) # Ajout de la colonne de biais
        return X_biais @ self.theta
    

    def score(self, X, y_vrai):
        """
        Calcule et affiche les performances du modèle.
        """
        y_pred = self.predict(X)
        
        # 1. Calcul du R2
        ss_res = np.sum((y_vrai - y_pred) ** 2) # Somme des erreurs au carré
        ss_tot = np.sum((y_vrai - np.mean(y_vrai)) ** 2) # Variance totale
        r2 = 1 - (ss_res / ss_tot)
        
        # 2. Calcul de la MAE
        mae = np.mean(np.abs(y_vrai - y_pred))
        
        # 3. Accuracy Directionnelle (Hit Ratio)
        # On compare le signe de la prédiction avec le signe réel
        signe_vrai = np.sign(y_vrai)
        signe_pred = np.sign(y_pred)
        # Les cas où les signes sont égaux (ignorer les zéros pour simplifier)
        bonnes_directions = np.sum(signe_vrai == signe_pred)
        hit_ratio = bonnes_directions / len(y_vrai)
        
        print("=== ÉVALUATION DE L'IA RIDGE ===")
        print(f"R2 Score         : {r2:.4f} (Attention, normal qu'il soit proche de 0 en finance)")
        print(f"MAE              : {mae:.4f} ({mae*100:.2f} % d'erreur moyenne)")
        print(f"Hit Ratio (Signe): {hit_ratio*100:.2f} % de bonnes directions")
        
        return r2, mae, hit_ratio
    

class AnalyseurNews:
    def __init__(self):
        self.mots_positifs = {
            # Mouvements et Tendance
            'accelerate', 'accelerates', 'accelerated', 'appreciate', 'appreciated', 'boom', 'booming', 
            'boost', 'boosted', 'breakout', 'breakthrough', 'bull', 'bullish', 'bulls', 'climb', 'climbed', 
            'double', 'doubled', 'gain', 'gained', 'gains', 'grow', 'growing', 'growth', 'high', 'higher', 
            'hot', 'jump', 'jumped', 'jumps', 'leap', 'leaps', 'momentum', 'peak', 'peaks', 'rally', 'rallied', 
            'rebound', 'recovers', 'recovery', 'skyrocket', 'skyrocketed', 'skyrockets', 'soar', 'soared', 
            'soars', 'surge', 'surged', 'surges', 'triple', 'tripled', 'uptrend', 'uptrends',
            
            # Fondamental, Résultats & Qualificatifs
            'achieve', 'achieved', 'advantage', 'attractive', 'beat', 'beats', 'beaten', 'beneficial', 'benefit', 
            'confident', 'deal', 'dividend', 'earnings', 'enhance', 'enhanced', 'exceed', 'exceeded', 'excel', 
            'excellent', 'expand', 'expanded', 'expansion', 'favorable', 'improve', 'improved', 'improvement', 
            'increase', 'increased', 'innovative', 'lucrative', 'optimism', 'optimistic', 'partnership', 'profit', 
            'profitability', 'profitable', 'profits', 'promising', 'record', 'revenue', 'revenues', 'reward', 
            'stellar', 'strong', 'stronger', 'succeed', 'success', 'successful', 'thrive', 'thriving', 'yield',
            
            # Recommandations
            'buy', 'outperform', 'outperformed', 'raise', 'raised', 'raises', 'target', 'upgrade', 'upgraded', 'upgrades', 'upside'
        }

        self.mots_negatifs = {
            # Mouvements et Tendance baissière
            'bear', 'bearish', 'collapse', 'collapses', 'collapsed', 'crash', 'crashed', 'crashes', 'decline', 
            'declined', 'declines', 'decrease', 'decreased', 'decreases', 'drop', 'dropped', 'drops', 'fall', 
            'falling', 'falls','loser','down', 'low', 'lower', 'lowest', 'plunge', 'plunged', 'plunges', 'sink', 'sinks', 
            'slide', 'slides', 'slump', 'slumps', 'tumble', 'tumbled', 'tumbles', 'weak', 'weaken', 'weakness',
            
            # Fondamental, Alertes & Pertes
            'bankrupt', 'bankruptcy', 'crisis', 'cut', 'cuts', 'debt', 'default', 'defaulted', 'deficit', 
            'delay', 'delayed', 'depress', 'depressed', 'disappoint', 'disappointed', 'disappointing', 
            'down', 'downgrade', 'downgraded', 'downgrades', 'fail', 'failed', 'failure', 'inflation', 
            'loss', 'losses', 'miss', 'missed', 'misses', 'negative', 'pessimistic', 'recession', 'risk', 
            'sell', 'shortfall', 'struggle', 'struggling', 'suspend', 'suspended', 'underperform', 
            'underperformed', 'warning', 'warns', 'worse', 'worst',
            
            # Légal & Scandales
            'breach', 'charge', 'charged', 'convict', 'convicted', 'fine', 'fines', 'fraud', 'fraudulent', 
            'guilty', 'illegal', 'investigate', 'investigation', 'investigating', 'lawsuit', 'litigation', 
            'penalty', 'probe', 'sue', 'sued', 'violation'
        }
    def _nettoyer_texte(self, texte):
        """Transforme le texte en minuscules et ne garde que les mots."""
        texte = texte.lower()
        # L'expression régulière \b\w+\b extrait uniquement les mots (vire la ponctuation)
        mots = re.findall(r'\b\w+\b', texte)
        return mots

    def calculer_sentiment(self, texte):
        """Calcule un score basé sur l'apparition des mots de ton lexique."""
        mots = self._nettoyer_texte(texte)
        if not mots:
            return "⚪ NEUTRE", 0.0

        score_positif = sum(1 for mot in mots if mot in self.mots_positifs)
        score_negatif = sum(1 for mot in mots if mot in self.mots_negatifs)
        
        score_total = score_positif - score_negatif
        
        # Calcul d'un pourcentage de "confiance" 
        mots_trouves = score_positif + score_negatif
        confiance = (mots_trouves / len(mots)) * 100 if len(mots) > 0 else 0

        if score_total > 0:
            return "🟢 POSITIF", confiance
        elif score_total < 0:
            return "🔴 NÉGATIF", confiance
        else:
            return "⚪ NEUTRE", confiance

    def scraper_news_yahoo(self, ticker):
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={ticker}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(url, headers=headers)
        
        resultats = []
        
        # Note : On ne met pas le try/except ici pour laisser les erreurs remonter à l'interface !
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            news_brutes = data.get('news', [])
            
            for article in news_brutes[:5]:
                titre = article.get('title', '')
                lien = article.get('link', '')
                
                if titre:
                    sentiment, confiance = self.calculer_sentiment(titre)
                    resultats.append({
                        'titre': titre,
                        'sentiment': sentiment,
                        'confiance': confiance,
                        'lien': lien
                    })
                    
        return resultats
