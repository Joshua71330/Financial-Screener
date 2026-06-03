import pandas as pd
import numpy as np

import io
from contextlib import redirect_stdout

import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, QLineEdit, 
                             QPushButton, QVBoxLayout, QWidget, QStackedWidget, 
                             QHBoxLayout, QMessageBox, QSpacerItem, QSizePolicy,
                             QTextEdit, QTabWidget, QTextBrowser, QCheckBox) 

from strategies import StrategieCroisementMA, StrategieRSI, StrategieMACD
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QFontDatabase
import urllib.error

import matplotlib as mpl
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from modeles import ActifFinancier 






# --- Classe pour gérer le canevas Multi-Graphiques ---
class DashboardCanvas(FigureCanvas):
    
    def __init__(self, parent=None, width=10, height=8, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_facecolor('#0c0c0c')
        
        super(DashboardCanvas, self).__init__(self.fig)
        self.fig.canvas.mpl_connect('scroll_event', self.zoom_molette)
        self.fig.canvas.mpl_connect('button_press_event', self.clic_presse)
        self.fig.canvas.mpl_connect('button_release_event', self.clic_relache)
        self.fig.canvas.mpl_connect('motion_notify_event', self.mouvement_souris)
        
        self.pan_axes = None
        self.press_x = None
        self.press_y = None
        
        self.df_courant = None
        # --- NOUVEAU : Listes vides, elles seront remplies dynamiquement ---
        self.axes = [] 
        self.types_axes = [] 
        self.x_dates_num = None
        
        self.tooltip = self.fig.text(0.0, 0.0, "", va="bottom", ha="left",
                                     fontsize=8, color="white",
                                     bbox=dict(boxstyle="round,pad=0.2", fc="#2A2A2A", ec="#555555", alpha=0.9),
                                     zorder=100, visible=False)



    def zoom_molette(self, event):
        # 1. On ignore si la souris n'est pas au-dessus d'un graphique (ex: dans les marges)
        if event.inaxes is None:
            return

        # 2. L'axe sur lequel se trouve la souris
        ax = event.inaxes

        # 3. Définir l'intensité du zoom (1.2 = 20% par cran de molette)
        facteur_base = 1.2
        if event.step > 0:
            # Molette vers le haut = Zoom in (on réduit la plage)
            facteur = 1 / facteur_base
        else:
            # Molette vers le bas = Zoom out (on agrandit la plage)
            facteur = facteur_base

        # 4. Récupérer les limites X actuelles et la position de la souris
        x_min, x_max = ax.get_xlim()
        x_souris = event.xdata

        # 5. Mathématiques : Calculer les nouvelles limites
        # On calcule la nouvelle largeur totale du graphique
        nouvelle_largeur = (x_max - x_min) * facteur
        
        # On calcule où se trouve la souris en pourcentage (0 = tout à gauche, 1 = tout à droite)
        position_relative = (x_souris - x_min) / (x_max - x_min)

        # On applique les nouvelles limites pour que la souris reste exactement au même endroit
        ax.set_xlim([
            x_souris - nouvelle_largeur * position_relative, 
            x_souris + nouvelle_largeur * (1 - position_relative)
        ])

        # 6. Redessiner le canevas de manière optimisée
        self.fig.canvas.draw_idle()



    def clic_presse(self, event):
            """Déclenché quand on clique sur le graphique."""
            if event.button == 1 and event.inaxes is not None:
                self.pan_axes = event.inaxes
                
                # On stocke les coordonnées en convertissant les pixels absolus de la fenêtre
                # vers les unités mathématiques du graphique cliqué.
                inv = self.pan_axes.transData.inverted()
                self.press_x, self.press_y = inv.transform((event.x, event.y))



    def clic_relache(self, event):
        """Déclenché quand on relâche le clic."""
        if event.button == 1:
            self.pan_axes = None



    def mouvement_souris(self, event):
        """Gère le glissement (drag) ET l'affichage des valeurs au survol."""
        
        # ==========================================
        # MODE 1 : DRAG (Si on maintient le clic)
        # ==========================================
        if self.pan_axes is not None:
            if event.x is None or event.y is None: return
            inv = self.pan_axes.transData.inverted()
            x_data, y_data = inv.transform((event.x, event.y))
            dx = x_data - self.press_x
            dy = y_data - self.press_y
            xlim = self.pan_axes.get_xlim()
            ylim = self.pan_axes.get_ylim()
            self.pan_axes.set_xlim(xlim[0] - dx, xlim[1] - dx)
            self.pan_axes.set_ylim(ylim[0] - dy, ylim[1] - dy)
            
            # Cacher le tooltip pendant qu'on drag
            if self.tooltip.get_visible(): self.tooltip.set_visible(False)
            self.fig.canvas.draw_idle()
            return

        # ==========================================
        # MODE 2 : SURVOL (Si on bouge sans cliquer)
        # ==========================================
        # Si la souris sort du graphique ou qu'on n'a pas encore de données : on cache la bulle
        if event.inaxes is None or self.df_courant is None:
            if self.tooltip.get_visible():
                self.tooltip.set_visible(False)
                self.fig.canvas.draw_idle()
            return

        ax = event.inaxes
        
        try:
            # 1. Convertir la position de la souris (X) en date
            # date_souris = pd.to_datetime(mdates.num2date(event.xdata)).tz_localize(None)
            
            # 2. Trouver l'index de la date la plus proche dans le DataFrame
            # index_propre = self.df_courant.index.tz_localize(None)
            # idx = index_propre.get_indexer([date_souris], method='nearest')[0]
            
            idx = np.abs(self.x_dates_num - event.xdata).argmin()
            
            # Récupérer la ligne de données et formater la vraie date
            row = self.df_courant.iloc[idx]
            date_reelle = self.df_courant.index[idx].strftime('%d %b %Y')
            
            # ---> CORRECTION : ON DÉFINIT L'INDEX DU GRAPHIQUE ICI <---
            index_graphique = self.axes.tolist().index(ax)

            # =========================================================
            # CALCUL DE LA DISTANCE Y POUR CACHER LA BULLE
            # =========================================================
            # --- NOUVEAU : Détection dynamique du type de graphique ---
            index_graphique = self.axes.index(ax)
            type_ax = self.types_axes[index_graphique]

            y_courbe = None
            if type_ax == 'prix' and "Close" in row: y_courbe = row['Close']
            elif type_ax == 'vol' and "Volatilite_20j" in row: y_courbe = row['Volatilite_20j']
            elif type_ax == 'macd' and "MACD" in row: y_courbe = row['MACD']
            elif type_ax == 'rsi' and "RSI" in row: y_courbe = row['RSI']

            if y_courbe is not None:
                y_min, y_max = ax.get_ylim()
                tolerance = (y_max - y_min) * 0.05 
                if abs(event.ydata - y_courbe) > tolerance:
                    if self.tooltip.get_visible():
                        self.tooltip.set_visible(False)
                        self.fig.canvas.draw_idle()
                    return 
            
            lignes = [f"{date_reelle}"]
            if type_ax == 'prix':
                lignes.append(f"Prix : {row['Close']:.2f} $")
                if "SMA_20" in row: lignes.append(f"SMA 20 : {row['SMA_20']:.2f}")
                if "EMA_20" in row: lignes.append(f"EMA 20 : {row['EMA_20']:.2f}")
            elif type_ax == 'vol':
                if "Volatilite_20j" in row: lignes.append(f"Volatilité : {row['Volatilite_20j']:.4f}")
            elif type_ax == 'macd':
                if "MACD" in row: lignes.append(f"MACD : {row['MACD']:.2f}")
                if "MACD_signal" in row: lignes.append(f"Signal : {row['MACD_signal']:.2f}")
            elif type_ax == 'rsi':
                if "RSI" in row: lignes.append(f"RSI : {row['RSI']:.2f}")

            # 4. Mettre à jour le texte
            nouveau_texte = "\n".join(lignes)
            
            # 5. Positionner le Tooltip
            fig_w, fig_h = self.fig.get_size_inches() * self.fig.dpi
            pos_x = (event.x + 15) / fig_w
            pos_y = (event.y + 15) / fig_h
            
            if pos_x > 0.8: pos_x = (event.x - 120) / fig_w
            if pos_y > 0.8: pos_y = (event.y - 80) / fig_h
            
            nouvelle_position = (pos_x, pos_y)

            # OPTIMISATION : Ne redessiner que si nécessaire
            texte_actuel = self.tooltip.get_text()
            position_actuelle = self.tooltip.get_position()
            
            if not self.tooltip.get_visible() or texte_actuel != nouveau_texte or position_actuelle != nouvelle_position:
                self.tooltip.set_text(nouveau_texte)
                self.tooltip.set_position(nouvelle_position)
                self.tooltip.set_visible(True)
                self.fig.canvas.draw_idle() # Appel de rendu uniquement s'il y a un changement
                
            self.tooltip.set_position((pos_x, pos_y))
            self.tooltip.set_visible(True)
            self.fig.canvas.draw_idle()
            
        except Exception as e:
            # Petite astuce : pendant le développement, il vaut mieux afficher l'erreur 
            # dans la console au lieu de 'pass', pour repérer ce genre de bugs plus vite !
            # print(f"Erreur de tooltip : {e}")
            pass






# --- Fenêtre Principale ---

# Force l'utilisation des dates modernes pour éviter le bug de 1970
mpl.rcParams['date.converter'] = 'auto'

# --- Fenêtre Principale ---
class ScreenerWindow(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Scamming Land Screener")
        self.setGeometry(100, 100, 1200, 750) 
        
        # Style Global "Dark Mode"
        # Style Global "Dark Mode"
        self.setStyleSheet("""
            QWidget { 
                font-family: 'Google Sans', sans-serif; 
                background-color: #0c0c0c; 
                color: white; 
            }
            /* Style pour la zone de texte normale */
            QLineEdit { 
                border: 1px solid #555555; 
                border-radius: 4px; 
                padding: 5px; 
                background-color: #1E1E1E; 
                color: white; 
            }
            /* Nouveau : Style spécifique pour le Placeholder. 
               Notez qu'il faut cibler QLineEdit et utiliser qproperty-placeholderText
               si on veut être très précis, mais dans les versions récentes de PyQt6,
               on utilise la pseudo-classe dynamique. 
            */
            QLineEdit[text=""] {
                color: #b4b4b4; /* Couleur du placeholder quand le champ est vide (ex: gris clair) */
            }
        
            
            QPushButton { border: 1px solid #555555; border-radius: 4px; padding: 8px 15px; background-color: #2A2A2A; color: white; }
            QPushButton:hover { background-color: #3A3A3A; }
            QLabel { color: white; }
        """)

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        # Variables pour stocker les données courantes
        self.df_complet = None
        self.ticker_actuel = ""

        # Construction des pages
        self.creer_page_accueil()     # Index 0
        self.creer_page_dashboard()   # Index 1
        self.creer_page_ia_report()   # Index 2 <--- NOUVEAU
        
        self.stacked_widget.setCurrentIndex(0)
        
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
        # 2. On lui donne le focus, retirant ainsi le curseur de la barre de recherche
        self.setFocus()


    def mousePressEvent(self, event):
        """
        Désélectionne la barre de recherche (ou tout autre élément) 
        lorsqu'on clique n'importe où ailleurs dans la fenêtre.
        """
        # Si un élément (comme la barre de recherche) possède actuellement le curseur...
        widget_actif = self.focusWidget()
        if widget_actif:
            # ... on lui retire le focus.
            widget_actif.clearFocus()
            
        # On laisse ensuite PyQt gérer le clic normalement pour le reste de l'interface
        super().mousePressEvent(event)


    # ================= PAGE 0 : ACCUEIL =================
    def creer_page_accueil(self):
        page = QWidget()
        layout_principal = QVBoxLayout(page)
        layout_principal.addStretch(1)

        self.titre_page1 = QLabel("Scamming Land Screener")
        self.titre_page1.setFont(QFont("Google Sans", 20, QFont.Weight.Medium))
        self.titre_page1.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout_principal.addWidget(self.titre_page1)

        layout_principal.addSpacing(12)

        # 1. On crée un QWidget qui servira de "boîte" visible
        boite_recherche = QWidget()
        
        # 2. On applique le style spécifique à CETTE boîte
        boite_recherche.setStyleSheet("""
            QWidget {
                background-color: #181818; /* Un fond très légèrement gris pour la distinguer */
                border: 2px solid transparent; /* Bordure transparente (supprimée visuellement) */
                border-radius: 20px; /* Rayon de courbure des coins (15px = très arrondi) */
            }
        """)
        
        # 3. On crée la barre de recherche
        self.input_ticker = QLineEdit()
        self.input_ticker.setFont(QFont("Google Sans", 15, QFont.Weight.Normal))
        self.input_ticker.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.input_ticker.setPlaceholderText("Ticker de l'entreprise")
        self.input_ticker.setFixedWidth(300)
        self.input_ticker.returnPressed.connect(self.lancer_analyse)
        
        # On s'assure que la barre de recherche elle-même n'a pas de style qui rentre en conflit
        # (elle hérite par défaut du style global, mais on peut le forcer ici si besoin)
        self.input_ticker.setStyleSheet("""
            QLineEdit {
                background-color: transparent; /* Fond transparent pour voir la boîte en dessous */
                border: none; /* Pas de bordure sur la zone de saisie elle-même */
            }
        """)

        # 4. On utilise le layout_boite pour placer la barre DANS le conteneur stylisé
        layout_boite = QHBoxLayout(boite_recherche) # Le layout appartient à 'boite_recherche'
        
        # On réduit les marges internes du layout pour que la barre occupe bien l'espace
        layout_boite.setContentsMargins(10, 5, 10, 5) 
        
        layout_boite.addWidget(self.input_ticker)
        
        # --- FIN DU NOUVEAU BLOC ---


        # 5. On place maintenant ce widget conteneur (la boîte) au centre de la page
        layout_centrage = QHBoxLayout()
        layout_centrage.addStretch()
        layout_centrage.addWidget(boite_recherche) # On ajoute le widget complet
        layout_centrage.addStretch()

        layout_principal.addLayout(layout_centrage)
        
        layout_principal.addStretch(1)
        self.stacked_widget.addWidget(page)
    


    # ================= PAGE 1 : DASHBOARD =================
    def creer_page_dashboard(self):
        page = QWidget()
        # Layout principal horizontal (Gauche: Barre, Droite: Contenu)
        layout_global = QHBoxLayout(page)
        layout_global.setContentsMargins(0, 0, 0, 0)
        layout_global.setSpacing(0)
        
        # ================= SIDEBAR (Barre verticale gauche) =================
        sidebar = QWidget()
        sidebar.setFixedWidth(230)
        sidebar.setStyleSheet("""
            QWidget { background-color: #1a1a1a; border-radius: 0px; }
            QLabel { font-weight: bold; font-size: 14px; margin-top: 10px; color: #17b978; padding-left: 5px; }
            QCheckBox { font-size: 13px; padding: 5px; color: white; }
            QCheckBox::indicator { width: 16px; height: 16px; border-radius: 3px; border: 1px solid #555; }
            QCheckBox::indicator:checked { background-color: #17b978; }
            QPushButton { margin: 5px; padding: 8px; border-radius: 5px; }
        """)
        layout_sidebar = QVBoxLayout(sidebar)
        
        # --- Sélection des graphiques ---
        layout_sidebar.addWidget(QLabel("GRAPHIQUES AFFICHÉS"))
        
        self.chk_prix = QCheckBox("Prix, SMA & EMA")
        self.chk_prix.setChecked(True)
        self.chk_vol = QCheckBox("Volatilité (20j)")
        self.chk_vol.setChecked(True)
        self.chk_macd = QCheckBox("MACD & Signal")
        self.chk_macd.setChecked(True)
        self.chk_rsi = QCheckBox("RSI (14j)")
        self.chk_rsi.setChecked(True)
        
        # Connexion aux mises à jour
        self.chk_prix.stateChanged.connect(self.actualiser_graphiques)
        self.chk_vol.stateChanged.connect(self.actualiser_graphiques)
        self.chk_macd.stateChanged.connect(self.actualiser_graphiques)
        self.chk_rsi.stateChanged.connect(self.actualiser_graphiques)
        
        layout_sidebar.addWidget(self.chk_prix)
        layout_sidebar.addWidget(self.chk_vol)
        layout_sidebar.addWidget(self.chk_macd)
        layout_sidebar.addWidget(self.chk_rsi)
        
        layout_sidebar.addSpacing(15)
        
        # --- Sélection de la période ---
        layout_sidebar.addWidget(QLabel("PÉRIODE D'ANALYSE"))
        
        btn_1m = QPushButton("1 Mois")
        btn_3m = QPushButton("3 Mois")
        btn_6m = QPushButton("6 Mois")
        btn_1a = QPushButton("1 An")
        
        btn_1m.clicked.connect(lambda: self.changer_periode(21))
        btn_3m.clicked.connect(lambda: self.changer_periode(63))
        btn_6m.clicked.connect(lambda: self.changer_periode(126))
        btn_1a.clicked.connect(lambda: self.changer_periode(252))
        
        layout_sidebar.addWidget(btn_1m)
        layout_sidebar.addWidget(btn_3m)
        layout_sidebar.addWidget(btn_6m)
        layout_sidebar.addWidget(btn_1a)
        
        layout_sidebar.addStretch()
        
        # Bouton retour en bas
        btn_accueil = QPushButton("← Autre actif")
        btn_accueil.setStyleSheet("background-color: #e74c3c; color: white; font-weight: bold;")
        btn_accueil.clicked.connect(self.retour_accueil)
        layout_sidebar.addWidget(btn_accueil)
        
        
        # ================= ZONE PRINCIPALE (Droite) =================
        zone_droite = QWidget()
        layout_droite = QVBoxLayout(zone_droite)
        layout_droite.setContentsMargins(15, 15, 15, 15)
        
        self.label_titre_dashboard = QLabel("Tableau de bord technique")
        self.label_titre_dashboard.setFont(QFont("Google Sans", 18, QFont.Weight.Bold))
        self.label_titre_dashboard.setAlignment(Qt.AlignmentFlag.AlignHCenter) 
        layout_droite.addWidget(self.label_titre_dashboard)

        self.label_prediction = QLabel("Prédiction IA : En attente...")
        self.label_prediction.setFont(QFont("Google Sans", 14, QFont.Weight.Bold))
        self.label_prediction.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout_droite.addWidget(self.label_prediction)
        
        # ✅ AJOUTEZ CES 3 LIGNES ICI POUR CORRIGER L'ERREUR :
        self.label_ia_news = QLabel("") # Texte vide par défaut
        self.label_ia_news.setFont(QFont("Google Sans", 11, QFont.Weight.Normal))
        self.label_ia_news.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout_droite.addWidget(self.label_ia_news)
        
        self.affichage_news = QTextBrowser()
        self.affichage_news.setReadOnly(True)
        self.affichage_news.setMaximumHeight(120) # Limite la hauteur pour ne pas écraser les graphiques
        self.affichage_news.setStyleSheet("""
            QTextBrowser {
                background-color: #121212; 
                color: #e0e0e0; 
                border: 1px solid #333333; 
                border-radius: 6px;
                padding: 8px;
            }
        """)
        # Par défaut, on peut le cacher s'il n'y a pas de news, ou le laisser visible.
        # self.affichage_news.hide() 
        layout_droite.addWidget(self.affichage_news)
        
        
        # Bouton IA
        layout_navigation_ia = QHBoxLayout()
        self.btn_switch_to_ia = QPushButton("Consulter le Rapport Détaillé IA 🤖")
        self.btn_switch_to_ia.setFont(QFont("Google Sans", 11, QFont.Weight.Bold))
        self.btn_switch_to_ia.setStyleSheet("background-color: #1e3d59; border: 1px solid #17b978;")
        self.btn_switch_to_ia.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(2))
        layout_navigation_ia.addStretch()
        layout_navigation_ia.addWidget(self.btn_switch_to_ia)
        layout_navigation_ia.addStretch()
        layout_droite.addLayout(layout_navigation_ia)
        
        # Canvas Graphique
        self.canvas = DashboardCanvas(self, width=10, height=8, dpi=100)
        layout_droite.addWidget(self.canvas)
        
        # --- Assemblage des deux parties ---
        layout_global.addWidget(sidebar)
        layout_global.addWidget(zone_droite)
        
        self.stacked_widget.addWidget(page)


    # ================= PAGE 2 : RAPPORT IA (NOUVELLE PAGE) =================
    def creer_page_ia_report(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        
        layout.addSpacing(15)
        
        self.label_titre_ia = QLabel("Rapport d'Optimisation & d'Évaluation de l'IA")
        self.label_titre_ia.setFont(QFont("Google Sans", 16, QFont.Weight.Bold))
        self.label_titre_ia.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.label_titre_ia)
        
        layout.addSpacing(15)
        
        # Le widget d'affichage de texte (Console Graphique)
        self.console_ia_text = QTextEdit()
        self.console_ia_text.setReadOnly(True)
        # Utilisation de la police Courier pour conserver l'alignement parfait du tableau
        self.console_ia_text.setFont(QFont("Courier", 11))
        self.console_ia_text.setStyleSheet("""
            QTextEdit {
                background-color: #0B0B0B;
                color: #FFFFFF;
                border: 1px solid #333333;
                border-radius: 6px;
                padding: 15px;
            }
        """)
        layout.addWidget(self.console_ia_text)
        
        layout.addSpacing(15)
        
        # Barre de boutons de navigation basse
        layout_nav_basse = QHBoxLayout()
        btn_retour_graph = QPushButton("← Retourner aux Graphiques")
        btn_retour_graph.setFont(QFont("Google Sans", 10, QFont.Weight.Bold))
        btn_retour_graph.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(1))
        
        btn_nouveau_ticker = QPushButton("🏠 Analyser un autre actif")
        btn_nouveau_ticker.clicked.connect(self.retour_accueil)
        
        layout_nav_basse.addStretch()
        layout_nav_basse.addWidget(btn_retour_graph)
        layout_nav_basse.addSpacing(20)
        layout_nav_basse.addWidget(btn_nouveau_ticker)
        layout_nav_basse.addStretch()
        layout.addLayout(layout_nav_basse)
        
        layout.addSpacing(15)
        self.stacked_widget.addWidget(page)



    # ================= LOGIQUE GLOBALE =================
    def retour_accueil(self):
        self.input_ticker.clear()
        self.stacked_widget.setCurrentIndex(0)
        self.input_ticker.setFocus()



    def lancer_analyse(self):
        ticker = self.input_ticker.text().strip().upper()
        if not ticker: return

        # 1. Traitement des données via ta classe
        action = ActifFinancier(ticker)
        if not action.charger_donnees():
            QMessageBox.warning(self, "Erreur", f"Impossible de charger les données pour le ticker : {ticker}")
            return

        # ---> CORRECTION DU BUG 1970 : On force l'index en format datetime de Pandas <---
        # Assure-toi que "import pandas as pd" est bien au début de ton fichier app.py
        # ---> CORRECTION DU BUG 1970 ET DES FUSEAUX HORAIRES <---
        action.historique.index = pd.to_datetime(action.historique.index, utc=True).tz_localize(None)
        # ---------------------------------------------------------------------------------

        action.calculer_moyenne_mobile(fenetre=20)
        action.calculer_EMA(fenetre=20)
        action.calculer_rendements()
        action.calculer_volatilite_historique(fenetre=20)
        action.calculer_rsi(fenetre=14)
        action.calculer_macd()
        
        # NOUVEAU : Calcul indispensable pour les features de l'IA
        action.calculer_volume_zscore(fenetre=20) 

        # =========================================================
        # BACKTESTING DES STRATÉGIES (Transféré depuis main.py)
        # =========================================================
        strategies = [
            StrategieCroisementMA(fenetre_courte=20, fenetre_longue=50),
            StrategieRSI(fenetre=14, seuil_survente=30, seuil_surachat=70),
            StrategieMACD(),
        ]

        for s in strategies:
            action.ajouter_strategie(s)

        resultats = action.evaluer_strategies()

        # Formatage du texte pour qu'il s'affiche proprement dans l'interface
        en_tete = f"{'Stratégie':<30} {'Rendement':>10} {'B&H':>8} {'Trades':>7}\n"
        texte_final = en_tete + "-" * 58 + "\n"
        
        for r in resultats:
            signe = "+" if r['performance_strategie'] >= 0 else ""
            texte_final += (
                f"{r['nom']:<30} "
                f"{signe}{r['performance_strategie']:>8.2f}%"
                f"{'+' if r['performance_buy_hold'] >= 0 else ''}"
                f"{r['performance_buy_hold']:>7.2f}%"
                f"{r['nombre_trades']:>7}\n"
            )
        

        # Envoi du texte formaté vers le widget de l'interface
        # self.console_strategies.setText(texte_final)
        # =========================================================

        # --- NOUVEAU : Exécution de l'IA ---
        # On met un curseur d'attente pour l'utilisateur
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        
        # ================= AJOUTER CECI (Le buffer de capture) =================
        capture_buffer = io.StringIO()
        
        try:
            with redirect_stdout(capture_buffer):
                # Tout ce qui est affiché par ces fonctions sera intercepté silencieusement
                action.trouver_meilleur_alpha()
                action.entrainer_IA()
                prediction = action.predire_demain()
            
            # Récupération de la chaîne de caractères et envoi sur la Page 2 (Rapport IA)
            texte_final_ia = capture_buffer.getvalue()
            self.console_ia_text.setText(texte_final_ia)
            # =======================================================================
            # CONDITION D'AFFICHAGE SÉCURISÉ (HIT RATIO > 50%)
            # =======================================================================
            
            # On vérifie que la variable existe (sécurité) et on la récupère
            hit_ratio_actuel = getattr(action, 'hit_ratio', 0)
            
            if prediction is not None and hit_ratio_actuel > 0.50:
                # L'IA est performante : on affiche la direction et le score
                if prediction > 0:
                    self.label_prediction.setText(f"Prédiction IA : 📈 HAUSSE (precision : {hit_ratio_actuel*100:.1f}%)")
                    self.label_prediction.setStyleSheet("color: #27ae60; font-weight: bold;") 
                else:
                    self.label_prediction.setText(f"Prédiction IA : 📉 BAISSE (precision :{hit_ratio_actuel*100:.1f}%)")
                    self.label_prediction.setStyleSheet("color: #e74c3c; font-weight: bold;") 
            else:
                # L'IA est mauvaise ou égale à 50% : on bloque l'affichage
                if hit_ratio_actuel > 0:
                    texte_blocage = f"🔒 IA non fiable (Précision : {hit_ratio_actuel*100:.1f}%)"
                else:
                    texte_blocage = "🔒 Prédiction IA indisponible"
                    
                self.label_prediction.setText(texte_blocage)
                # Style grisé / désactivé pour montrer que l'option est bloquée
                self.label_prediction.setStyleSheet("""
                    color: #888888; 
                    background-color: #2A2A2A; 
                    padding: 5px 15px; 
                    border-radius: 6px;
                    border: 1px solid #444444;
                """)
                
        except Exception as e:
            self.console_ia_text.setText(f"Erreur lors de la capture du flux IA :\n{str(e)}")
        finally:
            QApplication.restoreOverrideCursor()

        
        # =========================================================
        # EXÉCUTION DE L'ANALYSE FONDAMENTALE
        # =========================================================
        self.label_ia_news.setText("🔍 Analyse Lexicale en cours...")
        QApplication.processEvents() 
        
        try:
            # Si cette ligne plante (pas d'internet ou bloqué par Yahoo), 
            # le code saute directement au bloc "except" tout en bas !
            action.analyser_fondamental()
            
            html_content = f"<h2 style='color: #17b978;'>Dernières actualités pour {ticker}</h2><hr style='border-color: #555;'>"
            
            # Si on arrive ici, la connexion a RÉUSSI. 
            # Donc si c'est vide, c'est qu'il n'y a VRAIMENT pas de news.
            if not action.news:
                html_content += f"<p style='color: #A0A0A0;'>✅ Connexion réussie, mais aucune actualité récente n'a été trouvée pour le ticker {ticker}.</p>"
            else:
                for article in action.news:
                    couleur_sentiment = "white"
                    if "POSITIF" in article['sentiment']: couleur_sentiment = "#27ae60"
                    elif "NÉGATIF" in article['sentiment']: couleur_sentiment = "#e74c3c"
                    
                    html_content += f"""
                    <div style='margin-bottom: 20px; padding: 15px; background-color: #1E1E1E; border: 1px solid #333; border-radius: 8px;'>
                        <h3 style='margin-top: 0;'><a href='{article['lien']}' style='color: #3498db; text-decoration: none;'>{article['titre']}</a></h3>
                        <p style='margin: 5px 0; font-size: 14px;'>
                            Sentiment : <b style='color: {couleur_sentiment};'>{article['sentiment']}</b> 
                            <span style='color: #888;'>| Confiance : {article['confiance']:.1f}%</span>
                        </p>
                    </div>
                    """
            self.affichage_news.setHtml(html_content)
            self.label_ia_news.setText("✅ Analyse Macro terminée.")
            
        except urllib.error.HTTPError as e:
            # Spécifique au refus d'accès (Ex: Yahoo t'a banni temporairement ou l'URL est morte)
            html_content = f"<h2 style='color: #e74c3c;'>❌ Accès Refusé</h2><hr style='border-color: #555;'>"
            html_content += f"<p>Yahoo Finance a refusé la requête (Erreur HTTP {e.code}). Votre User-Agent a peut-être été détecté ou bloqué.</p>"
            self.affichage_news.setHtml(html_content)
            self.label_ia_news.setText("❌ Échec : Accès refusé.")
            
        except Exception as e:
            # Autres erreurs (Ex: pas d'internet du tout)
            html_content = f"<h2 style='color: #e74c3c;'>❌ Erreur de Connexion</h2><hr style='border-color: #555;'>"
            html_content += f"<p>Impossible de joindre le serveur. Vérifiez votre connexion internet.<br><br><small>Détail : {repr(e)}</small></p>"
            self.affichage_news.setHtml(html_content)
            self.label_ia_news.setText("❌ Échec de l'analyse macro.")
        # =========================================================

        # On sauvegarde le DataFrame complet et le ticker pour le filtrage temporel
        self.df_complet = action.historique.copy()
        self.ticker_actuel = ticker

        # Mise à jour du titre
        self.label_titre_dashboard.setText(f"Tableau de bord technique : {ticker}")
        
        # === AJOUTER CECI : Mise à jour du titre du rapport IA ===
        self.label_titre_ia.setText(f"Rapport d'Analyse IA Dédié : {ticker}")

        # On affiche par défaut la dernière année (252 jours)
        self.changer_periode(252)

        # On affiche la page du Dashboard
        self.stacked_widget.setCurrentIndex(1)



    # Méthode pour filtrer par période et redessiner ---
    # Dans la classe ScreenerWindow, ajoutez self.jours_actuels = 252 dans le __init__
    
    def changer_periode(self, jours):
        self.jours_actuels = jours
        self.actualiser_graphiques()

    def actualiser_graphiques(self):
        if self.df_complet is None: return
        
        # 1. Préparer les données
        # (Si on n'a pas défini self.jours_actuels dans le init, on le sécurise ici)
        jours = getattr(self, 'jours_actuels', 252) 
        df = self.df_complet.tail(jours)
        self.canvas.df_courant = df
        
        # 2. Réinitialiser la grille
        self.canvas.fig.clear()
        
        # 3. Lister ce qu'il faut afficher
        actifs = []
        if self.chk_prix.isChecked(): actifs.append('prix')
        if self.chk_vol.isChecked(): actifs.append('vol')
        if self.chk_macd.isChecked(): actifs.append('macd')
        if self.chk_rsi.isChecked(): actifs.append('rsi')
        
        n_plots = len(actifs)
        
        if n_plots == 0: # Si l'utilisateur a tout décoché
            self.canvas.axes = []
            self.canvas.types_axes = []
            self.canvas.draw()
            return
            
        # Donner plus d'espace au graphique des prix s'il est présent
        ratios = [2.5 if a == 'prix' else 1 for a in actifs]
        
        # 4. Créer les sous-graphiques à la volée
        axes_crees = self.canvas.fig.subplots(n_plots, 1, sharex=True, gridspec_kw={'height_ratios': ratios})
        
        # Matplotlib renvoie un seul objet (pas un tableau) s'il n'y a qu'un plot
        if n_plots == 1:
            self.canvas.axes = [axes_crees]
        else:
            self.canvas.axes = axes_crees.tolist()
            
        self.canvas.types_axes = actifs
        
        # 5. Dessiner les courbes sur les bons axes
        for ax, type_ax in zip(self.canvas.axes, actifs):
            if type_ax == 'prix': self.dessiner_prix(ax, df, self.ticker_actuel)
            elif type_ax == 'vol': self.dessiner_volatilite(ax, df, self.ticker_actuel)
            elif type_ax == 'macd': self.dessiner_macd(ax, df, self.ticker_actuel)
            elif type_ax == 'rsi': self.dessiner_rsi(ax, df, self.ticker_actuel)
            
        self.formater_axe_x(self.canvas.axes[-1])
        self.canvas.x_dates_num = mdates.date2num(df.index)
        
        # Ajustement de l'espace pour éviter les marges disgracieuses
        self.canvas.fig.subplots_adjust(hspace=0.1, bottom=0.15, left=0.08, right=0.95, top=0.95)
        self.canvas.draw()
        
        

    # ================= MÉTHODES DE DESSIN =================
    def formater_axe_x(self, ax):
        # 1. Le locator calcule dynamiquement l'espacement idéal selon le zoom
        locator = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        
        # 2. Le formatter automatique adapte le texte selon l'échelle du locator
        formatter = mdates.AutoDateFormatter(locator)
        
        # 3. Personnalisation des formats d'affichage selon le niveau de zoom
        # (les clés sont des intervalles en jours)
        formatter.scaled[1] = '%d %b %Y'   # Vue détaillée (ex: 13 Mai 2026)
        formatter.scaled[30] = '%b %Y'     # Vue mensuelle (ex: Mai 2026)
        formatter.scaled[365] = '%Y'       # Vue annuelle (ex: 2026)
        
        ax.xaxis.set_major_formatter(formatter)
        
        # Inclinaison pour éviter que les textes se chevauchent
        for label in ax.get_xticklabels():
            label.set_rotation(45)



    def dessiner_prix(self, ax, df, ticker):
        ax.clear()
        
        # --- STYLE DU FOND ET DES AXES ---
        ax.set_facecolor('#1e1e1e')  # Gris assez foncé pour le fond du graphique
        ax.tick_params(colors='lightgray') # Chiffres des axes en gris clair
        for spine in ax.spines.values():
            spine.set_color('#2d2d2d') # Bordures discrètes
        
        # 1. Préparation des données pour les bougies
        # Largeur des bougies (environ 0.6 unité de jour)
        width = 0.6
        
        # On sépare les jours de hausse (vert) et de baisse (rouge)
        hausse = df[df.Close >= df.Open]
        baisse = df[df.Close < df.Open]
        
        # 2. Dessin des mèches (High et Low) de la même couleur que le corps
        # On trace les mèches vertes pour la hausse
        ax.vlines(hausse.index, hausse.Low, hausse.High, color='#27ae60', linewidth=1.5)
        # On trace les mèches rouges pour la baisse
        ax.vlines(baisse.index, baisse.Low, baisse.High, color='#e74c3c', linewidth=1.5)
        
        # 3. Dessin des corps (Open et Close)
        # Pour la hausse : corps vert, bordure verte
        ax.bar(hausse.index, hausse.Close - hausse.Open, width, 
               bottom=hausse.Open, color='#27ae60', edgecolor='#27ae60', linewidth=1)
        
        # Pour la baisse : corps rouge, bordure rouge
        ax.bar(baisse.index, baisse.Open - baisse.Close, width, 
               bottom=baisse.Close, color='#e74c3c', edgecolor='#e74c3c', linewidth=1)

        # 4. Superposition des indicateurs (SMA/EMA)
        if "SMA_20" in df.columns:
            ax.plot(df.index, df["SMA_20"], label="SMA 20", color="blue", linestyle="--", alpha=0.8)
        if "EMA_20" in df.columns:
            ax.plot(df.index, df["EMA_20"], label="EMA 20", color="orange", linestyle="-.", alpha=0.8)
            
        ax.set_ylabel("Prix ($)")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)



    def dessiner_volatilite(self, ax, df, ticker):
        ax.clear()
        
         # --- STYLE DU FOND ET DES AXES ---
        ax.set_facecolor('#1e1e1e')  # Gris assez foncé pour le fond du graphique
        ax.tick_params(colors='lightgray') # Chiffres des axes en gris clair
        for spine in ax.spines.values():
            spine.set_color('#2d2d2d') # Bordures discrètes
        
        if "Volatilite_20j" in df.columns:
            ax.plot(df.index, df["Volatilite_20j"], label="Volatilité (20j)", color="red")
        ax.set_ylabel("Volatilité")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)



    def dessiner_macd(self, ax, df, ticker):
        ax.clear()
        
         # --- STYLE DU FOND ET DES AXES ---
        ax.set_facecolor('#1e1e1e')  # Gris assez foncé pour le fond du graphique
        ax.tick_params(colors='lightgray') # Chiffres des axes en gris clair
        for spine in ax.spines.values():
            spine.set_color('#2d2d2d') # Bordures discrètes
        
        if "MACD" in df.columns and "MACD_signal" in df.columns:
            ax.plot(df.index, df["MACD"], label="MACD", color="blue")
            ax.plot(df.index, df["MACD_signal"], label="Signal", color="orange")
            if "MACD_hist" in df.columns:
                couleurs = ['green' if val >= 0 else 'red' for val in df["MACD_hist"]]
                ax.bar(df.index, df["MACD_hist"], color=couleurs, alpha=0.5, label="Histogramme")
        ax.set_ylabel("MACD")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)



    def dessiner_rsi(self, ax, df, ticker):
        ax.clear()
        
         # --- STYLE DU FOND ET DES AXES ---
        ax.set_facecolor('#1e1e1e')  # Gris assez foncé pour le fond du graphique
        ax.tick_params(colors='lightgray') # Chiffres des axes en gris clair
        for spine in ax.spines.values():
            spine.set_color('#2d2d2d') # Bordures discrètes
        
        if "RSI" in df.columns:
            ax.plot(df.index, df["RSI"], label="RSI", color="purple")
            ax.axhline(70, color='red', linestyle='--', alpha=0.5)
            ax.axhline(30, color='green', linestyle='--', alpha=0.5)
            ax.fill_between(df.index, y1=30, y2=70, color='purple', alpha=0.05)
        ax.set_ylabel("RSI")
        ax.set_ylim(0, 100)
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)






if __name__ == "__main__":
    app = QApplication(sys.argv)
    fenetre = ScreenerWindow()
    fenetre.show()
    sys.exit(app.exec())
