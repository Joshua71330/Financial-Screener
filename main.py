"""
main.py — Point d'entrée de l'application Screener IA.

Ce script initialise l'application graphique. Toutes les opérations d'analyse 
(téléchargement des données, calcul des indicateurs, backtesting et IA) 
sont déléguées au contrôleur de l'interface (app.py).

Auteur : Kéziah Daull & Joshua Fadel 
"""

import sys
from PyQt6.QtWidgets import QApplication

# Import de la fenêtre principale depuis le fichier d'interface
from app import ScreenerWindow 

def main():
    """
    Initialise la boucle d'événements PyQt6 et lance l'interface.
    """
    # 1. Création de l'application Qt (Obligatoire avant tout widget)
    app = QApplication(sys.argv)
    
    # 2. Instanciation de la fenêtre principale
    fenetre = ScreenerWindow()
    
    # 3. Affichage à l'écran
    fenetre.show()
    
    # 4. Lancement de la boucle d'événements de l'application
    sys.exit(app.exec())


# ─────────────────────────────────────────────
#  POINT D'ENTRÉE
# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()
