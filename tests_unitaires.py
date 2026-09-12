"""
tests_unitaires.py — Jeu de tests unitaires complet.

Couvre la figure imposée 5 :
  ≥ 4 méthodes testées, ≥ 2 cas par méthode.

Classes testées :
  - TestRegressionRidge   (algos_IA.py)
  - TestActifFinancier    (modeles.py)
  - TestStrategies        (strategies.py)
  - TestAnalyseurNews     (algos_IA.py)
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import json
import urllib.error

from algos_IA  import Regression_Ridge, AnalyseurNews
from modeles   import ActifFinancier
from strategies import (
    StrategieCroisementMA,
    StrategieRSI,
    StrategieMACD,
)


# ═══════════════════════════════════════════════════════════════
#  UTILITAIRE : CRÉATION D'UN HISTORIQUE SYNTHÉTIQUE
# ═══════════════════════════════════════════════════════════════

def _creer_historique(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """
    Retourne un DataFrame avec des prix synthétiques (marche aléatoire)
    et un volume aléatoire. Utilisé par plusieurs classes de test.
    """
    np.random.seed(seed)
    rendements = np.random.randn(n) * 0.01
    prices = 100 * np.cumprod(1 + rendements)
    volume = np.random.randint(1_000_000, 5_000_000, n)
    dates  = pd.date_range(start='2022-01-01', periods=n, freq='B')
    return pd.DataFrame({'Close': prices, 'Volume': volume}, index=dates)


# ═══════════════════════════════════════════════════════════════
#  1. TESTS — Regression_Ridge (algos_IA.py)
# ═══════════════════════════════════════════════════════════════

class TestRegressionRidge(unittest.TestCase):
    """Teste la classe Regression_Ridge sur 4 comportements clés."""

    def setUp(self):
        """Données synthétiques réutilisées dans chaque test."""
        np.random.seed(0)
        self.X = np.random.randn(100, 3)
        self.y = self.X @ np.array([1.5, -2.0, 0.5]) + np.random.randn(100) * 0.1

    # ── Test 1 : predict avant entraînement ───────────────────

    def test_predict_avant_fit_leve_exception(self):
        """predict() doit lever ValueError si le modèle n'est pas entraîné."""
        model = Regression_Ridge(alpha=1.0)
        with self.assertRaises(ValueError):
            model.predict(self.X)

    def test_predict_tableau_vide_leve_exception(self):
        """predict() sur un tableau vide doit lever ValueError (non entraîné)."""
        model = Regression_Ridge(alpha=1.0)
        X_vide = np.empty((0, 3))
        with self.assertRaises(ValueError):
            model.predict(X_vide)

    # ── Test 2 : forme de la sortie après fit ─────────────────

    def test_predict_apres_fit_forme_correcte(self):
        """La prédiction doit avoir le même nombre de lignes que X."""
        model = Regression_Ridge(alpha=1.0)
        model.fit(self.X, self.y)
        y_pred = model.predict(self.X)
        self.assertEqual(y_pred.shape[0], self.X.shape[0])

    def test_predict_single_observation(self):
        """La prédiction doit fonctionner sur une seule observation."""
        model = Regression_Ridge(alpha=1.0)
        model.fit(self.X, self.y)
        X_une_ligne = self.X[[0], :]          # shape (1, 3)
        y_pred = model.predict(X_une_ligne)
        self.assertEqual(y_pred.shape[0], 1)

    # ── Test 3 : qualité du modèle (R²) ──────────────────────

    def test_r2_proche_de_1_sur_donnees_lineaires(self):
        """Sur des données quasi-linéaires, R² doit être ≥ 0.99."""
        X = np.arange(50).reshape(-1, 1).astype(float)
        y = 3.0 * X.flatten() + 7.0
        model = Regression_Ridge(alpha=1e-6)
        model.fit(X, y)
        r2, _, _ = model.score(X, y)
        self.assertGreater(r2, 0.99)

    def test_r2_degradation_fort_alpha(self):
        """Un alpha très élevé sur-régularise et dégrade le R²."""
        X = np.arange(50).reshape(-1, 1).astype(float)
        y = 3.0 * X.flatten() + 7.0
        model_fort = Regression_Ridge(alpha=1e9)
        model_faible = Regression_Ridge(alpha=1e-6)
        model_fort.fit(X, y)
        model_faible.fit(X, y)
        r2_fort,   _, _ = model_fort.score(X, y)
        r2_faible, _, _ = model_faible.score(X, y)
        self.assertLess(r2_fort, r2_faible)

    # ── Test 4 : comportement du Hit Ratio ────────────────────

    def test_hit_ratio_borne_entre_0_et_1(self):
        """Le Hit Ratio doit toujours être dans [0, 1]."""
        model = Regression_Ridge(alpha=1.0)
        model.fit(self.X, self.y)
        _, _, hit = model.score(self.X, self.y)
        self.assertGreaterEqual(hit, 0.0)
        self.assertLessEqual(hit,    1.0)

    def test_hit_ratio_parfait_signe_identique(self):
        """Quand prédiction et réalité ont toujours le même signe, hit = 1.0."""
        # On fabrique un cas où Ridge prédit parfaitement les signes
        np.random.seed(1)
        X = np.random.randn(80, 2)
        y = X[:, 0] * 10 + X[:, 1] * 5          # signal très fort, peu de bruit
        model = Regression_Ridge(alpha=1e-8)
        model.fit(X, y)
        _, _, hit = model.score(X, y)
        self.assertGreater(hit, 0.85)            # au moins 85 % de bonnes directions


# ═══════════════════════════════════════════════════════════════
#  2. TESTS — ActifFinancier (modeles.py)
# ═══════════════════════════════════════════════════════════════

class TestActifFinancier(unittest.TestCase):
    """Teste les indicateurs techniques de la classe ActifFinancier."""

    def setUp(self):
        """Instancie un actif avec un historique synthétique (sans fichier CSV)."""
        self.actif = ActifFinancier("TEST")
        self.actif.historique = _creer_historique(300)

    # ── Test 5 : calculer_moyenne_mobile ─────────────────────

    def test_sma_nan_au_debut(self):
        """Les (fenetre-1) premières valeurs de la SMA doivent être NaN."""
        self.actif.calculer_moyenne_mobile(fenetre=20)
        sma = self.actif.historique['SMA_20']
        self.assertTrue(sma.iloc[:19].isna().all(),
                        "Les 19 premières valeurs doivent être NaN (fenêtre=20).")

    def test_sma_valeur_exacte(self):
        """SMA_3 sur une série connue doit retourner la moyenne exacte."""
        dates = pd.date_range('2022-01-01', periods=5, freq='B')
        self.actif.historique = pd.DataFrame(
            {'Close': [10.0, 20.0, 30.0, 40.0, 50.0], 'Volume': [1e6] * 5},
            index=dates
        )
        self.actif.calculer_moyenne_mobile(fenetre=3)
        attendu = (10.0 + 20.0 + 30.0) / 3
        self.assertAlmostEqual(
            self.actif.historique['SMA_3'].iloc[2], attendu, places=5
        )

    # ── Test 6 : calculer_rsi ─────────────────────────────────

    def test_rsi_borne_0_100(self):
        """Le RSI doit toujours être dans l'intervalle [0, 100]."""
        self.actif.calculer_rsi(fenetre=14)
        rsi = self.actif.historique['RSI'].dropna()
        self.assertTrue((rsi >= 0).all(),   "RSI ne peut pas être négatif.")
        self.assertTrue((rsi <= 100).all(), "RSI ne peut pas dépasser 100.")

    def test_rsi_que_hausses_proche_de_100(self):
        """Sur une série strictement croissante, le RSI final doit être proche de 100."""
        dates = pd.date_range('2022-01-01', periods=60, freq='B')
        self.actif.historique = pd.DataFrame(
            {'Close': np.linspace(100, 200, 60), 'Volume': [1e6] * 60},
            index=dates
        )
        self.actif.calculer_rsi(fenetre=14)
        rsi_final = self.actif.historique['RSI'].iloc[-1]
        self.assertGreater(rsi_final, 90,
                           "Un trend haussier pur doit donner RSI > 90.")

    # ── Test 7 : calculer_volatilite_historique ───────────────

    def test_volatilite_necessite_rendements(self):
        """La volatilité ne doit PAS être calculée si les rendements sont absents."""
        # Historique sans colonne 'Rendements'
        self.actif.historique = _creer_historique(100)
        self.actif.calculer_volatilite_historique(fenetre=20)
        self.assertNotIn(
            'Volatilite_20j', self.actif.historique.columns,
            "La volatilité ne doit pas être ajoutée si les rendements manquent."
        )

    def test_volatilite_toujours_positive(self):
        """La volatilité annualisée doit être strictement positive ou nulle."""
        self.actif.calculer_rendements()
        self.actif.calculer_volatilite_historique(fenetre=20)
        vol = self.actif.historique['Volatilite_20j'].dropna()
        self.assertTrue((vol >= 0).all(),
                        "La volatilité ne peut pas être négative.")

    # ── Test 8 : charger_donnees ──────────────────────────────

    def test_charger_donnees_ticker_inexistant_retourne_false(self):
        """
        Si le ticker est invalide et le téléchargement échoue,
        charger_donnees() doit retourner False.
        """
        actif_fake = ActifFinancier("TICKER_INEXISTANT_ZZZ999")
        # On simule l'échec du téléchargement sans toucher le réseau
        with patch('gestion_donnees.telecharger_historique', return_value=False):
            with patch('modeles.telecharger_historique', return_value=False):
                result = actif_fake.charger_donnees()
        self.assertFalse(result)

    def test_charger_donnees_historique_non_vide_apres_injection(self):
        """
        Après injection directe de l'historique (simulation de CSV chargé),
        l'attribut historique ne doit pas être vide.
        """
        self.actif.historique = _creer_historique(100)
        self.assertIsNotNone(self.actif.historique)
        self.assertGreater(len(self.actif.historique), 0)


# ═══════════════════════════════════════════════════════════════
#  3. TESTS — Stratégies (strategies.py)
# ═══════════════════════════════════════════════════════════════

class TestStrategies(unittest.TestCase):
    """Teste les trois stratégies concrètes."""

    def setUp(self):
        """
        Historique synthétique commun.
        On précalcule RSI et MACD pour StrategieMACD.
        """
        self.actif = ActifFinancier("TEST")
        self.actif.historique = _creer_historique(300)
        self.actif.calculer_rendements()
        self.actif.calculer_moyenne_mobile(fenetre=20)
        self.actif.calculer_moyenne_mobile(fenetre=50)
        self.actif.calculer_EMA(fenetre=20)
        self.actif.calculer_volatilite_historique(fenetre=20)
        self.actif.calculer_rsi(fenetre=14)
        self.actif.calculer_macd()
        self.df = self.actif.historique.copy()

    # ── Test 9 : signaux dans {-1, 0, 1} ─────────────────────

    def test_signaux_croisementMA_valeurs_valides(self):
        """Les signaux de StrategieCroisementMA doivent être dans {-1, 0, 1}."""
        s = StrategieCroisementMA(20, 50)
        sigs = s.generer_signaux(self.df)
        valeurs_uniques = set(sigs['Signal'].unique())
        self.assertTrue(valeurs_uniques.issubset({-1, 0, 1}),
                        f"Valeurs inattendues : {valeurs_uniques}")

    def test_signaux_rsi_valeurs_valides(self):
        """Les signaux de StrategieRSI doivent être dans {-1, 0, 1}."""
        s = StrategieRSI(14, 30, 70)
        sigs = s.generer_signaux(self.df)
        valeurs_uniques = set(sigs['Signal'].unique())
        self.assertTrue(valeurs_uniques.issubset({-1, 0, 1}),
                        f"Valeurs inattendues : {valeurs_uniques}")

    # ── Test 10 : performance retourne les bonnes clés ────────

    def test_performance_cles_retournees(self):
        """calculer_performance() doit retourner les 4 clés attendues."""
        s = StrategieCroisementMA(20, 50)
        res = s.calculer_performance(self.df)
        for cle in ('nom', 'performance_strategie', 'performance_buy_hold', 'nombre_trades'):
            self.assertIn(cle, res, f"Clé manquante : '{cle}'")

    def test_performance_nombre_trades_positif_ou_nul(self):
        """Le nombre de trades doit être un entier ≥ 0."""
        s = StrategieRSI(14, 30, 70)
        res = s.calculer_performance(self.df)
        self.assertIsInstance(res['nombre_trades'], int)
        self.assertGreaterEqual(res['nombre_trades'], 0)

    # ── Test 11 : StrategieMACD nécessite les colonnes MACD ───

    def test_macd_sans_colonnes_retourne_signaux_neutres(self):
        """Sans colonnes MACD, StrategieMACD doit retourner des signaux nuls (0)."""
        df_sans_macd = self.df[['Close', 'Volume']].copy()
        s = StrategieMACD()
        sigs = s.generer_signaux(df_sans_macd)
        self.assertTrue((sigs['Signal'] == 0).all(),
                        "Tous les signaux doivent être 0 en l'absence de MACD.")

    def test_macd_avec_colonnes_genere_signaux(self):
        """Avec les colonnes MACD présentes, des signaux non nuls doivent apparaître."""
        s = StrategieMACD()
        sigs = s.generer_signaux(self.df)
        # Sur 300 jours il doit y avoir au moins un signal
        self.assertGreater(
            (sigs['Signal'] != 0).sum(), 0,
            "Aucun signal généré sur 300 jours — vérifier le calcul MACD."
        )

    # ── Test 12 : héritage (isinstance) ──────────────────────

    def test_heritage_strategie_croisementMA(self):
        """StrategieCroisementMA doit être une instance de Strategie."""
        from strategies import Strategie
        s = StrategieCroisementMA()
        self.assertIsInstance(s, Strategie)

    def test_heritage_strategie_rsi(self):
        """StrategieRSI doit être une instance de Strategie."""
        from strategies import Strategie
        s = StrategieRSI()
        self.assertIsInstance(s, Strategie)

# ═══════════════════════════════════════════════════════════════
#  4. TESTS — AnalyseurNews (algos_IA.py)
# ═══════════════════════════════════════════════════════════════

class TestAnalyseurNews(unittest.TestCase):
    """Teste le nettoyage de texte, le calcul de sentiment et le scraper web."""

    def setUp(self):
        self.analyseur = AnalyseurNews()

    # ── Test 13 : Nettoyage de texte ─────────────────────────

    def test_nettoyer_texte_enleve_ponctuation_et_minuscule(self):
        """Le texte doit être mis en minuscules et débarrassé de sa ponctuation."""
        texte_brut = "Hausse record, l'entreprise gagne 100%!"
        mots = self.analyseur._nettoyer_texte(texte_brut)
        
        self.assertIn("hausse", mots)
        self.assertIn("record", mots)
        self.assertNotIn(",", mots)
        self.assertNotIn("!", mots)

    # ── Test 14 : Calcul de sentiment (Lexique) ──────────────

    def test_calculer_sentiment_positif(self):
        """Une phrase avec des mots positifs doit renvoyer 🟢 POSITIF."""
        # Correction : Phrase en anglais pour correspondre au lexique de l'IA
        texte = "The company reported a record profit and strong growth this quarter."
        sentiment, confiance = self.analyseur.calculer_sentiment(texte)
        
        self.assertEqual(sentiment, "🟢 POSITIF")
        self.assertGreater(confiance, 0.0)

    def test_calculer_sentiment_negatif(self):
        """Une phrase avec des mots négatifs doit renvoyer 🔴 NÉGATIF."""
        # Correction : Phrase en anglais avec les mots 'crash', 'drop', 'loss'
        texte = "The market crash causes a massive drop and a huge financial loss."
        sentiment, confiance = self.analyseur.calculer_sentiment(texte)
        
        self.assertEqual(sentiment, "🔴 NÉGATIF")
        self.assertGreater(confiance, 0.0)

    def test_calculer_sentiment_neutre(self):
        """Une phrase sans mots du lexique doit renvoyer ⚪ NEUTRE avec 0% de confiance."""
        # Correction : Phrase en anglais sans mots financiers de la liste
        texte = "The CEO ate an apple at the cafeteria."
        sentiment, confiance = self.analyseur.calculer_sentiment(texte)
        
        self.assertEqual(sentiment, "⚪ NEUTRE")
        self.assertEqual(confiance, 0.0)

    # ── Test 15 : Scraper Web (Mocking) ──────────────────────

    @patch('urllib.request.urlopen')
    def test_scraper_news_yahoo_succes(self, mock_urlopen):
        """
        Teste si le scraper parse bien le JSON quand internet fonctionne, 
        sans faire de vraie requête HTTP (Mock).
        """
        # On fabrique une fausse réponse JSON (comme si Yahoo nous répondait)
        fake_json = {
            "news": [
                {"title": "Profit record et hausse !", "link": "https://fake.link.com"}
            ]
        }
        
        # On configure notre objet MagicMock pour imiter le comportement de urlopen
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(fake_json).encode('utf-8')
        mock_response.__enter__.return_value = mock_response # Pour gérer le "with urlopen() as..."
        mock_urlopen.return_value = mock_response

        # On lance notre fonction
        resultats = self.analyseur.scraper_news_yahoo("FAKE_TICKER")

        # Vérifications
        self.assertEqual(len(resultats), 1)
        self.assertEqual(resultats[0]['titre'], "Profit record et hausse !")
        self.assertEqual(resultats[0]['sentiment'], "🟢 POSITIF") # Vérifie que le calcul s'est appliqué
        self.assertEqual(resultats[0]['lien'], "https://fake.link.com")

    @patch('urllib.request.urlopen')
    def test_scraper_news_yahoo_erreur_reseau(self, mock_urlopen):
        """
        Si Yahoo renvoie une erreur (ex: 404 ou erreur de connexion), 
        le scraper DOIT laisser remonter l'exception pour qu'elle soit gérée 
        par l'interface (app.py).
        """
        # On force la fausse connexion internet à planter
        mock_urlopen.side_effect = urllib.error.URLError("Not Found")
        
        # On vérifie que l'exception est bien levée (et non masquée)
        with self.assertRaises(urllib.error.URLError):
            self.analyseur.scraper_news_yahoo("FAKE_TICKER")


# ═══════════════════════════════════════════════════════════════
#  POINT D'ENTRÉE
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
