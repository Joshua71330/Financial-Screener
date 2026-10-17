# 📊 Financial Screener — Analyse technique & prédiction par IA

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![PyQt6](https://img.shields.io/badge/GUI-PyQt6-41CD52?logo=qt)](https://pypi.org/project/PyQt6/)
[![Status](https://img.shields.io/badge/status-projet%20acad%C3%A9mique-orange)]()

> 🇫🇷 Application de bureau pour l'analyse technique et la prédiction algorithmique d'actifs financiers, avec un moteur d'IA développé from scratch.
> 🇬🇧 Desktop application for technical analysis and algorithmic prediction of financial assets, powered by a from-scratch AI engine.

**[🇫🇷 Version française](#-français)** · **[🇬🇧 English version](#-english)**

---

## 🇫🇷 Français

### Aperçu

Financial Screener est une application de bureau écrite entièrement en Python qui combine **analyse technique classique** (moyennes mobiles, RSI, MACD, volatilité) et un **modèle d'intelligence artificielle développé en interne** (régression Ridge implémentée à la main avec NumPy) pour évaluer des opportunités sur les marchés actions.

Le rapport technique complet (méthodologie, formules mathématiques, architecture) est disponible dans [`Rapport Screener_IA_FINAL-1.pdf`](./Rapport%20Screener_IA_FINAL-1.pdf).

### Fonctionnalités

- **Interface graphique (PyQt6)** : recherche d'un ticker, tableaux de bord et graphiques interactifs (zoom, survol).
- **Indicateurs techniques** : SMA, EMA, RSI, MACD, volatilité historique annualisée.
- **Téléchargement de données** automatique via `yfinance`, avec mise en cache locale en CSV.
- **Modèle d'IA maison** : régression Ridge codée sans framework de ML (forme fermée, NumPy), pour prédire le rendement du lendemain à partir de features techniques (distance à la SMA/EMA, volatilité 20 j, RSI de Wilder, z-score du volume). Découpage chronologique 80/20, standardisation calculée uniquement sur l'entraînement, choix de la régularisation par grille, évaluation par R², MAE et taux de bonne direction (hit ratio).
- **Stratégies de trading** orientées objet (classe abstraite `Strategie` + sous-classes croisement de moyennes mobiles, RSI, MACD), backtestées contre le buy & hold : rendement, nombre de trades, **ratio de Sharpe** annualisé et **max drawdown**. Les ordres sont exécutés le lendemain du signal pour éviter le biais de look-ahead.
- **Analyse des options (GEX)** : profil d'exposition gamma des market makers calculé à partir du gamma de Black–Scholes sur la chaîne d'options, niveau zero-gamma, max pain et calendrier des échéances (OpEx), utilisés comme filtre de régime pour la stratégie RSI.
- **Analyseur de news** (`AnalyseurNews`) pour intégrer un signal texte à l'analyse.
- **Suite de tests unitaires** (`unittest`) couvrant le modèle de régression, la classe `ActifFinancier` et les stratégies.

### Stack technique

| Domaine | Outils |
|---|---|
| Langage | Python 3.10+ |
| Interface | PyQt6, Matplotlib (embarqué via `FigureCanvasQTAgg`) |
| Données | pandas, NumPy, yfinance |
| IA / quant | Régression Ridge implémentée à la main (NumPy), gamma de Black–Scholes (SciPy) |
| Tests | `unittest` |

### Structure du projet

```
Financial-Screener/
├── main.py                 # Point d'entrée : lance l'interface PyQt6
├── app.py                  # Fenêtre principale / contrôleur de l'interface
├── modeles.py               # Classe ActifFinancier (orchestration données + IA)
├── strategies.py             # Stratégies de trading (héritage : MA, RSI, MACD)
├── algos_IA.py               # Régression Ridge + analyseur de news
├── gestion_donnees.py        # Téléchargement et cache des données (yfinance)
├── courbes.py                 # Script autonome de tracé de tableaux de bord
├── tests_unitaires.py        # Suite de tests unitaires (30 tests)
├── docs/architecture.md      # Diagramme de classes (Mermaid)
└── Rapport Screener_IA_FINAL-1.pdf   # Rapport technique complet
```

### Installation

```bash
git clone https://github.com/Joshua71330/Financial-Screener.git
cd Financial-Screener
pip install -r requirements.txt
```

### Utilisation

```bash
python main.py
```

Lance l'interface graphique : entrez un ticker (ex. `AAPL`, `MSFT`), l'application télécharge l'historique, calcule les indicateurs et affiche le tableau de bord.

Pour lancer les tests :

```bash
python -m unittest tests_unitaires -v
```

### Auteurs

Projet réalisé par **Kéziah Daull** et **Joshua Fadel**.

---

## 🇬🇧 English

### Overview

Financial Screener is a desktop application, written entirely in Python, that combines **classic technical analysis** (moving averages, RSI, MACD, volatility) with an **in-house artificial intelligence model** (a Ridge regression implemented from scratch with NumPy) to evaluate opportunities in stock markets.

The full technical report (methodology, math, architecture) is available in [`Rapport Screener_IA_FINAL-1.pdf`](./Rapport%20Screener_IA_FINAL-1.pdf).

### Features

- **Graphical interface (PyQt6)**: ticker search, dashboards and interactive charts (zoom, hover tooltips).
- **Technical indicators**: SMA, EMA, RSI, MACD, annualized historical volatility.
- **Automatic data download** via `yfinance`, with local CSV caching.
- **Custom AI model**: a Ridge regression built without any ML framework (closed form, NumPy), predicting next-day returns from technical features (distance to SMA/EMA, 20-day volatility, Wilder RSI, volume z-score). Chronological 80/20 split, standardization fitted on the training set only, grid search over the regularization strength, evaluation with R², MAE and directional hit ratio.
- **Object-oriented trading strategies** (abstract `Strategie` base class + moving-average-crossover, RSI, and MACD subclasses), backtested against buy & hold: total return, number of trades, annualized **Sharpe ratio** and **maximum drawdown**. Orders are executed the day after the signal to avoid look-ahead bias.
- **Options analytics (GEX)**: dealer gamma exposure profile computed from Black–Scholes gamma over the option chain, zero-gamma level, max pain and expiry (OpEx) calendar, used as a regime filter for the RSI strategy.
- **News analyzer** (`AnalyseurNews`) to fold a text-based signal into the analysis.
- **Unit test suite** (`unittest`) covering the regression model, the `ActifFinancier` class, and the strategies.

### Tech stack

| Area | Tools |
|---|---|
| Language | Python 3.10+ |
| UI | PyQt6, Matplotlib (embedded via `FigureCanvasQTAgg`) |
| Data | pandas, NumPy, yfinance |
| AI / quant | Hand-rolled Ridge regression (NumPy), Black–Scholes gamma (SciPy) |
| Testing | `unittest` |

### Project structure

```
Financial-Screener/
├── main.py                 # Entry point: launches the PyQt6 UI
├── app.py                  # Main window / UI controller
├── modeles.py               # ActifFinancier class (data + AI orchestration)
├── strategies.py             # Trading strategies (inheritance: MA, RSI, MACD)
├── algos_IA.py               # Ridge regression + news analyzer
├── gestion_donnees.py        # Data download and caching (yfinance)
├── courbes.py                 # Standalone dashboard-plotting script
├── tests_unitaires.py        # Unit test suite (30 tests)
├── docs/architecture.md      # Class diagram (Mermaid)
└── Rapport Screener_IA_FINAL-1.pdf   # Full technical report
```

### Installation

```bash
git clone https://github.com/Joshua71330/Financial-Screener.git
cd Financial-Screener
pip install -r requirements.txt
```

### Usage

```bash
python main.py
```

Launches the GUI: enter a ticker (e.g. `AAPL`, `MSFT`), the app downloads history, computes indicators, and displays the dashboard.

Run the tests:

```bash
python -m unittest tests_unitaires -v
```

### Authors

Built by **Kéziah Daull** and **Joshua Fadel**.
