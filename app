# =================================================================================================
# ======================================= IMPORTATIONS ============================================
# =================================================================================================

import pandas as pd
import numpy as np
import io
import sys
import urllib.error
from contextlib import redirect_stdout

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, 
    QVBoxLayout, QWidget, QStackedWidget, QHBoxLayout, QMessageBox, 
    QSpacerItem, QSizePolicy, QTextEdit, QTabWidget, QTextBrowser, 
    QCheckBox
) 
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QFontDatabase

# Importations spécifiques au projet (Doivent exister dans ton répertoire)
from strategies import StrategieCroisementMA, StrategieRSI, StrategieMACD
from modeles import ActifFinancier 


# =================================================================================================
# ============================= PARAMÈTRES GLOBAUX MATPLOTLIB =====================================
# =================================================================================================

# Force l'utilisation des dates modernes pour éviter le bug de 1970
mpl.rcParams['date.converter'] = 'auto'


# =================================================================================================
# ============================ CLASSE : DASHBOARD CANVAS ==========================================
# Gère l'affichage, le zoom, le survol et l'interaction avec les graphiques Matplotlib
# =================================================================================================

class DashboardCanvas(FigureCanvas):
    
    def __init__(self, parent=None, width=10, height=8, dpi=100):
        
        # Initialisation de la figure Matplotlib
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_facecolor('#0c0c0c')
        
        super(DashboardCanvas, self).__init__(self.fig)
        
        # Connexion des événements souris
        self.fig.canvas.mpl_connect('scroll_event', self.zoom_molette)
        self.fig.canvas.mpl_connect('button_press_event', self.clic_presse)
        self.fig.canvas.mpl_connect('button_release_event', self.clic_relache)
        self.fig.canvas.mpl_connect('motion_notify_event', self.mouvement_souris)
        
        # Variables d'état pour les interactions
        self.pan_axes = None
        self.press_x = None
        self.press_y = None
        self.df_courant = None
        
        # Listes pour stocker les axes créés dynamiquement
        self.axes = [] 
        self.types_axes = [] 
        self.x_dates_num = None
        
        # Tooltip pour afficher les informations au survol
        self.tooltip = self.fig.text(
            0.0, 0.0, "", 
            va="bottom", ha="left",
            fontsize=8, color="white",
            bbox=dict(boxstyle="round,pad=0.2", fc="#2A2A2A", ec="#555555", alpha=0.9),
            zorder=100, visible=False
        )


    # ---------------------------------------------------------------------------------------------
    # GESTION DU ZOOM (MOLETTE SOURIS)
    # ---------------------------------------------------------------------------------------------
    def zoom_molette(self, event):
        
        if event.inaxes is None:
            return

        ax = event.inaxes
        facteur_base = 1.2
        
        if event.step > 0:
            facteur = 1 / facteur_base
        else:
            facteur = facteur_base

        x_min, x_max = ax.get_xlim()
        x_souris = event.xdata

        nouvelle_largeur = (x_max - x_min) * facteur
        position_relative = (x_souris - x_min) / (x_max - x_min)

        ax.set_xlim([
            x_souris - nouvelle_largeur * position_relative, 
            x_souris + nouvelle_largeur * (1 - position_relative)
        ])

        self.fig.canvas.draw_idle()


    # ---------------------------------------------------------------------------------------------
    # GESTION DU CLIC (DÉBUT DU PANNING)
    # ---------------------------------------------------------------------------------------------
    def clic_presse(self, event):
        
        if event.button == 1 and event.inaxes is not None:
            self.pan_axes = event.inaxes
            
            inv = self.pan_axes.transData.inverted()
            self.press_x, self.press_y = inv.transform((event.x, event.y))


    # ---------------------------------------------------------------------------------------------
    # GESTION DU RELÂCHEMENT DU CLIC (FIN DU PANNING)
    # ---------------------------------------------------------------------------------------------
    def clic_relache(self, event):
        
        if event.button == 1:
            self.pan_axes = None


    # ---------------------------------------------------------------------------------------------
    # GESTION DES MOUVEMENTS DE SOURIS (PANNING + TOOLTIP)
    # ---------------------------------------------------------------------------------------------
    def mouvement_souris(self, event):
        
        # --- MODE DRAG (Déplacement du graphique) ---
        if self.pan_axes is not None:
            
            if event.x is None or event.y is None: 
                return
                
            inv = self.pan_axes.transData.inverted()
            x_data, y_data = inv.transform((event.x, event.y))
            
            dx = x_data - self.press_x
            dy = y_data - self.press_y
            
            xlim = self.pan_axes.get_xlim()
            ylim = self.pan_axes.get_ylim()
            
            self.pan_axes.set_xlim(xlim[0] - dx, xlim[1] - dx)
            self.pan_axes.set_ylim(ylim[0] - dy, ylim[1] - dy)
            
            if self.tooltip.get_visible(): 
                self.tooltip.set_visible(False)
                
            self.fig.canvas.draw_idle()
            return


        # --- MODE SURVOL (Affichage des données) ---
        if event.inaxes is None or self.df_courant is None:
            
            if self.tooltip.get_visible():
                self.tooltip.set_visible(False)
                self.fig.canvas.draw_idle()
            return

        ax = event.inaxes
        
        try:
            # Récupération de l'index le plus proche
            idx = np.abs(self.x_dates_num - event.xdata).argmin()
            row = self.df_courant.iloc[idx]
            date_reelle = self.df_courant.index[idx].strftime('%d %b %Y')
            
            index_graphique = self.axes.index(ax)
            type_ax = self.types_axes[index_graphique]

            y_courbe = None
            
            # Détermination de la valeur Y en fonction du type de graphique
            if type_ax == 'prix' and "Close" in row: 
                y_courbe = row['Close']
            elif type_ax == 'vol' and "Volatilite_20j" in row: 
                y_courbe = row['Volatilite_20j']
            elif type_ax == 'macd' and "MACD" in row: 
                y_courbe = row['MACD']
            elif type_ax == 'rsi' and "RSI" in row: 
                y_courbe = row['RSI']

            # Vérification de la tolérance pour l'affichage de la bulle
            if y_courbe is not None:
                y_min, y_max = ax.get_ylim()
                tolerance = (y_max - y_min) * 0.05 
                
                if abs(event.ydata - y_courbe) > tolerance:
                    if self.tooltip.get_visible():
                        self.tooltip.set_visible(False)
                        self.fig.canvas.draw_idle()
                    return 
            
            # Construction du texte du tooltip
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

            nouveau_texte = "\n".join(lignes)
            
            # Positionnement du tooltip
            fig_w, fig_h = self.fig.get_size_inches() * self.fig.dpi
            pos_x = (event.x + 15) / fig_w
            pos_y = (event.y + 15) / fig_h
            
            if pos_x > 0.8: pos_x = (event.x - 120) / fig_w
            if pos_y > 0.8: pos_y = (event.y - 80) / fig_h
            
            nouvelle_position = (pos_x, pos_y)
            texte_actuel = self.tooltip.get_text()
            position_actuelle = self.tooltip.get_position()
            
            # Mise à jour si nécessaire
            if not self.tooltip.get_visible() or texte_actuel != nouveau_texte or position_actuelle != nouvelle_position:
                self.tooltip.set_text(nouveau_texte)
                self.tooltip.set_position(nouvelle_position)
                self.tooltip.set_visible(True)
                self.fig.canvas.draw_idle() 
                
        except Exception:
            pass



# =================================================================================================
# ============================ CLASSE PRINCIPALE : L'APPLICATION ==================================
# Interface globale gérant les différentes vues (Accueil, Dashboard, Rapport IA)
# =================================================================================================

class ScreenerWindow(QMainWindow):
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Scamming Land Screener")
        self.setGeometry(100, 100, 1300, 800) 
        
        # Application du style global (Dark Mode)
        self.setStyleSheet("""
            QWidget { 
                font-family: 'Google Sans', sans-serif; 
                background-color: #0c0c0c; 
                color: white; 
            }
            QLineEdit { 
                border: 0px transparent #555555; 
                border-radius: 4px; 
                padding: 5px; 
                background-color: #1E1E1E; 
                color: white; 
            }
            QLineEdit[text=""] {
                color: #b4b4b4; 
            }
            QPushButton { 
                border: 1px solid #555555; 
                border-radius: 4px; 
                padding: 8px 15px; 
                background-color: #FFFFFF; 
                color: white; 
            }
            QPushButton:hover { 
                background-color: #3A3A3A; 
            }
            QLabel { 
                color: white; 
            }
        """)

        # Gestionnaire de pages
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        # Variables de stockage des données
        self.df_complet = None
        self.ticker_actuel = ""
        self.jours_actuels = 252

        # Création des différentes vues
        self.creer_page_accueil()     # Index 0
        self.creer_page_dashboard()   # Index 1
        self.creer_page_ia_report()   # Index 2
        
        self.stacked_widget.setCurrentIndex(0)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFocus()


    # ---------------------------------------------------------------------------------------------
    # GESTION DU FOCUS SOURIS
    # ---------------------------------------------------------------------------------------------
    def mousePressEvent(self, event):
        
        widget_actif = self.focusWidget()
        
        if widget_actif:
            widget_actif.clearFocus()
            
        super().mousePressEvent(event)


    # =============================================================================================
    # ============================ PAGE 0 : ACCUEIL ===============================================
    # =============================================================================================
    def creer_page_accueil(self):
        
        page = QWidget()
        layout_principal = QVBoxLayout(page)
        
        layout_principal.addStretch(1)

        # Titre Principal
        self.titre_page1 = QLabel("Scamming Land Screener")
        self.titre_page1.setFont(QFont("Google Sans", 20, QFont.Weight.Medium))
        self.titre_page1.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout_principal.addWidget(self.titre_page1)

        layout_principal.addSpacing(12)

        # Boîte de recherche stylisée
        boite_recherche = QWidget()
        boite_recherche.setStyleSheet("""
            QWidget {
                background-color: #181818; 
                border: 0px solid transparent; 
                border-radius: 20px; 
            }
        """)
        
        # Barre de texte pour le Ticker
        self.input_ticker = QLineEdit()
        self.input_ticker.setFont(QFont("Google Sans", 15, QFont.Weight.Normal))
        self.input_ticker.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.input_ticker.setPlaceholderText("Ticker de l'entreprise")
        self.input_ticker.setFixedWidth(300)
        self.input_ticker.returnPressed.connect(self.lancer_analyse)
        
        self.input_ticker.setStyleSheet("""
            QLineEdit {
                background-color: transparent; 
                border: none; 
            }
        """)

        # Layout de la boîte
        layout_boite = QHBoxLayout(boite_recherche) 
        layout_boite.setContentsMargins(10, 5, 10, 5) 
        layout_boite.addWidget(self.input_ticker)
        
        # Centrage de la boîte
        layout_centrage = QHBoxLayout()
        layout_centrage.addStretch()
        layout_centrage.addWidget(boite_recherche) 
        layout_centrage.addStretch()

        layout_principal.addLayout(layout_centrage)
        layout_principal.addStretch(1)
        
        self.stacked_widget.addWidget(page)
    

    # =============================================================================================
    # ============================ PAGE 1 : DASHBOARD =============================================
    # Contient la barre latérale avec toutes les infos, et la zone de droite avec le graphe
    # =============================================================================================
    def creer_page_dashboard(self):
        
        page = QWidget()
        layout_global = QHBoxLayout(page)
        layout_global.setContentsMargins(0, 0, 0, 0)
        layout_global.setSpacing(0)
        
        # -----------------------------------------------------------------------------------------
        # BARRE LATÉRALE (Désormais plus large pour accueillir les textes)
        # -----------------------------------------------------------------------------------------
        sidebar = QWidget()
        # Élargissement de la sidebar pour afficher correctement les actualités
        sidebar.setFixedWidth(350)
        sidebar.setStyleSheet("""
            QWidget { 
                background-color: #1a1a1a; 
                border-radius: 0px; 
                padding-left: 10px;
                padding-right: 10px;
            }
            QLabel { 
                font-weight: bold;
                font-size: 15px; 
                margin-top: 10px;
                color: white; 
            }
            QCheckBox { 
                font-size: 15px;
                padding: 10px; 
                color: grey; 
            }
            QCheckBox::indicator { 
                width: 10px; 
                height: 10px; 
                border-radius: 3px; 
                border: 1px solid #555; 
            }
            QCheckBox::indicator:checked { 
                background-color: #17b978; 
            }
            QPushButton { 
                margin: 5px; 
                border-radius: 5px; 
            }
        """)
        
        layout_sidebar = QVBoxLayout(sidebar)
        
        # --- SECTION : INFORMATIONS IA ET PRÉDICTIONS ---
        layout_sidebar.addWidget(QLabel("ANALYSE INTELLIGENCE ARTIFICIELLE"))
        
        self.label_prediction = QLabel("Prédiction IA : En attente...")
        self.label_prediction.setFont(QFont("Google Sans", 12, QFont.Weight.Bold))
        # Alignement à gauche pour bien s'intégrer dans la barre
        self.label_prediction.setAlignment(Qt.AlignmentFlag.AlignLeft)
        # Modification de la couleur du texte par défaut
        self.label_prediction.setStyleSheet("color: white;")
        layout_sidebar.addWidget(self.label_prediction)

        self.btn_switch_to_ia = QPushButton("Consulter le Rapport Détaillé IA 🤖")
        self.btn_switch_to_ia.setFont(QFont("Google Sans", 10, QFont.Weight.Bold))
        self.btn_switch_to_ia.setStyleSheet("background-color: #1e3d59; border: 1px solid #17b978;")
        self.btn_switch_to_ia.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(2))
        layout_sidebar.addWidget(self.btn_switch_to_ia)
        # --- SECTION : PERFORMANCES STRATÉGIES ---
        layout_sidebar.addWidget(QLabel("BACKTESTING STRATÉGIES"))
        
        self.affichage_perf = QTextBrowser()
        self.affichage_perf.setReadOnly(True)
        self.affichage_perf.setMaximumHeight(120) 
        self.affichage_perf.setStyleSheet("""
            QTextBrowser {
                background-color: #121212; 
                color: #e0e0e0; 
                border: 1px solid #333333; 
                border-radius: 6px;
                padding: 8px;
            }
        """)
        layout_sidebar.addWidget(self.affichage_perf)
        
        layout_sidebar.addSpacing(15)
        
        # --- SECTION : ACTUALITÉS ET FONDAMENTAL ---
        layout_sidebar.addWidget(QLabel("ACTUALITÉS ET SENTIMENT"))

        self.label_ia_news = QLabel("Statut : En attente...") 
        self.label_ia_news.setFont(QFont("Google Sans", 10, QFont.Weight.Normal))
        self.label_ia_news.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.label_ia_news.setStyleSheet("color: #b4b4b4; margin-top: 0px;")
        layout_sidebar.addWidget(self.label_ia_news)
        
        self.affichage_news = QTextBrowser()
        self.affichage_news.setReadOnly(True)
        self.affichage_news.setOpenExternalLinks(True)
        # On donne une hauteur confortable mais limitée à la zone d'actualité
        self.affichage_news.setMaximumHeight(200) 
        self.affichage_news.setStyleSheet("""
            QTextBrowser {
                background-color: #121212; 
                color: #e0e0e0; 
                border: 1px solid #333333; 
                border-radius: 6px;
                padding: 8px;
            }
            a {
                color: #3498db;
                text-decoration: none;
            }
        """)
        layout_sidebar.addWidget(self.affichage_news)

        layout_sidebar.addSpacing(15)
        
        # --- SECTION : CONTRÔLE DES GRAPHIQUES ---
        layout_sidebar.addWidget(QLabel("GRAPHIQUES AFFICHÉS"))
        
        self.chk_prix = QCheckBox("Prix, SMA & EMA")
        self.chk_prix.setChecked(True)
        self.chk_vol = QCheckBox("Volatilité (20j)")
        self.chk_vol.setChecked(True)
        self.chk_macd = QCheckBox("MACD & Signal")
        self.chk_macd.setChecked(True)
        self.chk_rsi = QCheckBox("RSI (14j)")
        self.chk_rsi.setChecked(True)
        
        self.chk_prix.stateChanged.connect(self.actualiser_graphiques)
        self.chk_vol.stateChanged.connect(self.actualiser_graphiques)
        self.chk_macd.stateChanged.connect(self.actualiser_graphiques)
        self.chk_rsi.stateChanged.connect(self.actualiser_graphiques)
        
        layout_sidebar.addWidget(self.chk_prix)
        layout_sidebar.addWidget(self.chk_vol)
        layout_sidebar.addWidget(self.chk_macd)
        layout_sidebar.addWidget(self.chk_rsi)
        
        layout_sidebar.addSpacing(15)
        
        # --- SECTION : CONTRÔLE DE LA PÉRIODE ---
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
        
        # --- BOUTON DE RETOUR (Bas de la barre latérale) ---
        btn_accueil = QPushButton("Changer d'actif")
        btn_accueil.setStyleSheet("background-color: #e74c3c; color: white; font-weight: bold;")
        btn_accueil.clicked.connect(self.retour_accueil)
        layout_sidebar.addWidget(btn_accueil)
        
        
        # -----------------------------------------------------------------------------------------
        # ZONE PRINCIPALE (Droite) : Uniquement le titre et les graphiques
        # -----------------------------------------------------------------------------------------
        zone_droite = QWidget()
        layout_droite = QVBoxLayout(zone_droite)
        layout_droite.setContentsMargins(10, 40, 10, 5)
        
        layout_droite.setSpacing(0)
        
        self.label_titre_dashboard = QLabel("Tableau de bord technique")
        self.label_titre_dashboard.setFont(QFont("Google Sans", 20, QFont.Weight.Medium))
        self.label_titre_dashboard.setAlignment(Qt.AlignmentFlag.AlignHCenter) 
        layout_droite.addWidget(self.label_titre_dashboard)
        
        layout_droite.addSpacing(0)

        # Ajout du Canva contenant les graphes Matplotlib
        self.canvas = DashboardCanvas(self, width=10, height=10, dpi=100)
        layout_droite.addWidget(self.canvas)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout_droite.addWidget(self.canvas)
        
        # Assemblage final de la page
        layout_global.addWidget(sidebar)
        layout_global.addWidget(zone_droite)
        
        self.stacked_widget.addWidget(page)


    # =============================================================================================
    # ============================ PAGE 2 : RAPPORT IA ============================================
    # =============================================================================================
    def creer_page_ia_report(self):
        
        page = QWidget()
        layout = QVBoxLayout(page)
        
        layout.addSpacing(15)
        
        self.label_titre_ia = QLabel("Rapport d'Optimisation & d'Évaluation de l'IA")
        self.label_titre_ia.setFont(QFont("Google Sans", 16, QFont.Weight.Bold))
        self.label_titre_ia.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.label_titre_ia)
        
        layout.addSpacing(15)
        
        # Console texte pour l'affichage des logs bruts de l'IA
        self.console_ia_text = QTextEdit()
        self.console_ia_text.setReadOnly(True)
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
        
        # Boutons de navigation inférieurs
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



    # =============================================================================================
    # ============================ LOGIQUE GLOBALE ================================================
    # Navigation, Traitement et Préparation des données
    # =============================================================================================

    def retour_accueil(self):
        
        self.input_ticker.clear()
        self.stacked_widget.setCurrentIndex(0)
        self.input_ticker.setFocus()


    def lancer_analyse(self):
        
        ticker = self.input_ticker.text().strip().upper()
        if not ticker: return

        # -----------------------------------------------------------------------------------------
        # CHARGEMENT ET CALCUL DES INDICATEURS
        # -----------------------------------------------------------------------------------------
        action = ActifFinancier(ticker)
        
        if not action.charger_donnees():
            QMessageBox.warning(self, "Erreur", f"Impossible de charger les données pour le ticker : {ticker}")
            return

        # Correction des fuseaux horaires (Bug 1970)
        action.historique.index = pd.to_datetime(action.historique.index, utc=True).tz_localize(None)

        action.calculer_moyenne_mobile(fenetre=20)
        action.calculer_EMA(fenetre=20)
        action.calculer_rendements()
        action.calculer_volatilite_historique(fenetre=20)
        action.calculer_rsi(fenetre=14)
        action.calculer_macd()
        action.calculer_volume_zscore(fenetre=20) 


        # -----------------------------------------------------------------------------------------
        # BACKTESTING DES STRATÉGIES
        # -----------------------------------------------------------------------------------------
        strategies = [
            StrategieCroisementMA(fenetre_courte=20, fenetre_longue=50),
            StrategieRSI(fenetre=14, seuil_survente=30, seuil_surachat=70),
            StrategieMACD(),
        ]

        for s in strategies:
            action.ajouter_strategie(s)

        resultats = action.evaluer_strategies()


        # -----------------------------------------------------------------------------------------
        # EXÉCUTION DU MODÈLE IA
        # -----------------------------------------------------------------------------------------
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        capture_buffer = io.StringIO()
        
        try:
            # Capture silencieuse des impressions pour le rapport détaillé
            with redirect_stdout(capture_buffer):
                action.trouver_meilleur_alpha()
                action.entrainer_IA()
                prediction = action.predire_demain()
            
            texte_final_ia = capture_buffer.getvalue()
            self.console_ia_text.setText(texte_final_ia)
            
            # Gestion de l'affichage en fonction de la fiabilité du modèle
            hit_ratio_actuel = getattr(action, 'hit_ratio', 0)
            
            if prediction is not None and hit_ratio_actuel > 0.50:
                if prediction > 0:
                    self.label_prediction.setText(f"Prédiction IA :\n📈 HAUSSE (Précision : {hit_ratio_actuel*100:.1f}%)")
                    self.label_prediction.setStyleSheet("color: #27ae60; font-weight: bold; margin-top:0px;") 
                else:
                    self.label_prediction.setText(f"Prédiction IA :\n📉 BAISSE (Précision :{hit_ratio_actuel*100:.1f}%)")
                    self.label_prediction.setStyleSheet("color: #e74c3c; font-weight: bold; margin-top:0px;") 
            else:
                if hit_ratio_actuel > 0:
                    texte_blocage = f"🔒 IA non fiable\n(Précision : {hit_ratio_actuel*100:.1f}%)"
                else:
                    texte_blocage = "🔒 Prédiction IA\nindisponible"
                    
                self.label_prediction.setText(texte_blocage)
                self.label_prediction.setStyleSheet("""
                    color: #888888; 
                    background-color: #2A2A2A; 
                    padding: 5px; 
                    border-radius: 6px;
                    border: 1px solid #444444;
                    margin-top:0px;
                """)
                
        except Exception as e:
            self.console_ia_text.setText(f"Erreur lors de la capture du flux IA :\n{str(e)}")
            
        finally:
            QApplication.restoreOverrideCursor()

        
        # -----------------------------------------------------------------------------------------
        # ANALYSE FONDAMENTALE (SCRAPING ACTUALITÉS)
        # -----------------------------------------------------------------------------------------
        self.label_ia_news.setText("🔍 Analyse Lexicale en cours...")
        QApplication.processEvents() 
        
        try:
            action.analyser_fondamental()
            
            # Note : on utilise un style plus petit/compact pour la sidebar
            html_content = f"<h4 style='color: #17b978; margin:0;'>Actualités {ticker}</h4><hr style='border-color: #555;'>"
            
            if not action.news:
                html_content += f"<p style='color: #A0A0A0; font-size: 12px;'> ✅ Aucune actualité récente trouvée.</p>"
            else:
                for article in action.news:
                    # Définition de l'émoji en fonction du sentiment
                    emoji = "⚪"
                    if "POSITIF" in article['sentiment']: emoji = "🟢"
                    elif "NÉGATIF" in article['sentiment']: emoji = "🔴"
                    
                    # Génération d'une seule ligne simple et robuste
                    html_content += f"""
                    <p style="margin-top: 5px; margin-bottom: 5px;">
                        <span style="font-size: 12px;">{emoji}</span>
                        <a href="{article['lien']}" style="font-size: 13px; font-weight: bold; text-decoration: none; color: #3498db;">
                            {article['titre']}
                        </a>
                    </p>
                    <hr style="background-color: #333333; height: 1px; border: none; margin-top: 8px; margin-bottom: 8px;">
                    """
            self.affichage_news.setHtml(html_content)
            self.label_ia_news.setText("✅ Analyse Macro terminée.")
            
        except urllib.error.HTTPError as e:
            html_content = f"<h4 style='color: #e74c3c;'>❌ Accès Refusé</h4><hr style='border-color: #555;'>"
            html_content += f"<p style='font-size:12px;'>Rejeté par Yahoo (Erreur HTTP {e.code}).</p>"
            self.affichage_news.setHtml(html_content)
            self.label_ia_news.setText("❌ Échec : Accès refusé.")
            
        except Exception as e:
            html_content = f"<h4 style='color: #e74c3c;'>❌ Erreur de Connexion</h4><hr style='border-color: #555;'>"
            html_content += f"<p style='font-size:12px;'>Impossible de joindre le serveur.</p>"
            self.affichage_news.setHtml(html_content)
            self.label_ia_news.setText("❌ Échec de l'analyse macro.")

        # -----------------------------------------------------------------------------------------
        # FINALISATION ET MISE À JOUR VISUELLE
        # -----------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------------------
        # FINALISATION ET MISE À JOUR VISUELLE
        # -----------------------------------------------------------------------------------------
        self.df_complet = action.historique.copy()
        
        # 1. Sauvegarde des signaux pour Matplotlib
        self.df_complet['Signal_MA'] = strategies[0].signaux['Signal']
        self.df_complet['Signal_RSI'] = strategies[1].signaux['Signal']
        self.df_complet['Signal_MACD'] = strategies[2].signaux['Signal']
        
        self.ticker_actuel = ticker

        self.label_titre_dashboard.setText(f"Tableau de bord technique : {ticker}")
        self.label_titre_ia.setText(f"Rapport d'Analyse IA Dédié : {ticker}")

        # 2. Affichage des performances des stratégies
        html_perf = f"<h4 style='color: #3498db; margin:0;'>Rendements & Risques (vs B&H)</h4><hr style='border-color: #555;'>"
        for res in resultats:
            couleur = "#27ae60" if res['performance_strategie'] >= 0 else "#e74c3c"
            # Mise en valeur si notre Max Drawdown est meilleur (plus petit) que le marché
            couleur_dd = "#2ecc71" if res['max_dd_strat'] < res['max_dd_bh'] else "#e0e0e0"
            
            html_perf += f"""
            <div style='margin-bottom: 8px; font-size: 11px;'>
                <b>{res['nom']}</b> : <span style='color: {couleur}; font-weight: bold;'>{res['performance_strategie']}%</span> 
                <span style='color: #888;'>(B&H: {res['performance_buy_hold']}%)</span><br>
                <span style='color: #AAA;'>↳ Trades : {res['nombre_trades']} | Sharpe : {res['sharpe_ratio']} </span><br>
                <span style='color: #AAA;'>↳ Pire chute (Max DD) : <b style='color: {couleur_dd};'>-{res['max_dd_strat']}%</b> vs -{res['max_dd_bh']}% (B&H)</span>
            </div>
            """
        self.affichage_perf.setHtml(html_perf)

        # On appelle le rafraîchissement avec la durée sélectionnée (252 jours par défaut)
        self.changer_periode(self.jours_actuels)
        self.stacked_widget.setCurrentIndex(1)

    # ---------------------------------------------------------------------------------------------
    # GESTION DES PÉRIODES TEMPORELLES
    # ---------------------------------------------------------------------------------------------
    def changer_periode(self, jours):
        
        self.jours_actuels = jours
        self.actualiser_graphiques()


    # ---------------------------------------------------------------------------------------------
    # ACTUALISATION DE L'AFFICHAGE MATPLOTLIB
    # ---------------------------------------------------------------------------------------------
    def actualiser_graphiques(self):
        
        if self.df_complet is None: 
            return
        
        jours = getattr(self, 'jours_actuels', 252) 
        df = self.df_complet.tail(jours)
        self.canvas.df_courant = df
        
        self.canvas.fig.clear()
        
        # Liste des composants actifs
        actifs = []
        if self.chk_prix.isChecked(): actifs.append('prix')
        if self.chk_vol.isChecked(): actifs.append('vol')
        if self.chk_macd.isChecked(): actifs.append('macd')
        if self.chk_rsi.isChecked(): actifs.append('rsi')
        
        n_plots = len(actifs)
        
        # Si rien n'est sélectionné, on vide le canevas
        if n_plots == 0: 
            self.canvas.axes = []
            self.canvas.types_axes = []
            self.canvas.draw()
            return
            
        # Poids de la hauteur (Le prix prend plus de place)
        ratios = [2.5 if a == 'prix' else 1 for a in actifs]
        
        axes_crees = self.canvas.fig.subplots(
            n_plots, 1, 
            sharex=True, 
            gridspec_kw={'height_ratios': ratios}
        )
        
        # Matplotlib retourne un array s'il y a plusieurs plots, ou un objet seul
        if n_plots == 1:
            self.canvas.axes = [axes_crees]
        else:
            self.canvas.axes = axes_crees.tolist()
            
        self.canvas.types_axes = actifs
        
        # Dessin sur les sous-graphes
        for ax, type_ax in zip(self.canvas.axes, actifs):
            if type_ax == 'prix': self.dessiner_prix(ax, df, self.ticker_actuel)
            elif type_ax == 'vol': self.dessiner_volatilite(ax, df, self.ticker_actuel)
            elif type_ax == 'macd': self.dessiner_macd(ax, df, self.ticker_actuel)
            elif type_ax == 'rsi': self.dessiner_rsi(ax, df, self.ticker_actuel)
            
        self.formater_axe_x(self.canvas.axes[-1])
        self.canvas.x_dates_num = mdates.date2num(df.index)
        
        # Ajustement des marges internes pour la beauté de l'interface
        self.canvas.fig.subplots_adjust(hspace=0.1, bottom=0.15, left=0.08, right=0.95, top=0.95)
        self.canvas.draw()
        

    # =============================================================================================
    # ============================ MÉTHODES DE DESSIN =============================================
    # Sous-routines gérant spécifiquement la construction géométrique des courbes
    # =============================================================================================

    def formater_axe_x(self, ax):
        
        locator = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        
        formatter = mdates.AutoDateFormatter(locator)
        
        formatter.scaled[1] = '%d %b %Y'  
        formatter.scaled[30] = '%b %Y'    
        formatter.scaled[365] = '%Y'      
        
        ax.xaxis.set_major_formatter(formatter)
        
        for label in ax.get_xticklabels():
            label.set_rotation(45)


    # ---------------------------------------------------------------------------------------------
    def dessiner_prix(self, ax, df, ticker):
        
        ax.clear()
        
        # Apparence du fond
        ax.set_facecolor('#1e1e1e')  
        ax.tick_params(colors='lightgray') 
        for spine in ax.spines.values():
            spine.set_color('#2d2d2d') 
        
        width = 0.6
        
        hausse = df[df.Close >= df.Open]
        baisse = df[df.Close < df.Open]
        
        # Mèches et corps pour les bougies haussières
        ax.vlines(hausse.index, hausse.Low, hausse.High, color='#27ae60', linewidth=1.5)
        ax.bar(hausse.index, hausse.Close - hausse.Open, width, 
               bottom=hausse.Open, color='#27ae60', edgecolor='#27ae60', linewidth=1)
        
        # Mèches et corps pour les bougies baissières
        ax.vlines(baisse.index, baisse.Low, baisse.High, color='#e74c3c', linewidth=1.5)
        ax.bar(baisse.index, baisse.Open - baisse.Close, width, 
               bottom=baisse.Close, color='#e74c3c', edgecolor='#e74c3c', linewidth=1)

        # Indicateurs superposés
        if "SMA_20" in df.columns:
            ax.plot(df.index, df["SMA_20"], label="SMA 20", color="blue", linestyle="--", alpha=0.8)
            
        if "EMA_20" in df.columns:
            ax.plot(df.index, df["EMA_20"], label="EMA 20", color="orange", linestyle="-.", alpha=0.8)
        
        # Affichage des signaux d'achat/vente (Stratégie MA)
        if "Signal_MA" in df.columns:
            achats = df[df['Signal_MA'] == 1]
            ventes = df[df['Signal_MA'] == -1]
            
            if not achats.empty:
                # Flèche verte en dessous du prix bas
                ax.scatter(achats.index, achats['Low'] * 0.96, marker='^', color='#2ecc71', s=120, label='Achat (MA)', zorder=5)
            if not ventes.empty:
                # Flèche rouge au dessus du prix haut
                ax.scatter(ventes.index, ventes['High'] * 1.04, marker='v', color='#e74c3c', s=120, label='Vente (MA)', zorder=5)

        ax.set_ylabel("Prix ($)")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)


    # ---------------------------------------------------------------------------------------------
    def dessiner_volatilite(self, ax, df, ticker):
        
        ax.clear()
        
        ax.set_facecolor('#1e1e1e')  
        ax.tick_params(colors='lightgray') 
        for spine in ax.spines.values():
            spine.set_color('#2d2d2d') 
        
        if "Volatilite_20j" in df.columns:
            ax.plot(df.index, df["Volatilite_20j"], label="Volatilité (20j)", color="red")
            
        ax.set_ylabel("Volatilité")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)


    # ---------------------------------------------------------------------------------------------
    def dessiner_macd(self, ax, df, ticker):
        
        ax.clear()
        
        ax.set_facecolor('#1e1e1e')  
        ax.tick_params(colors='lightgray') 
        for spine in ax.spines.values():
            spine.set_color('#2d2d2d') 
        
        if "MACD" in df.columns and "MACD_signal" in df.columns:
            
            ax.plot(df.index, df["MACD"], label="MACD", color="blue")
            ax.plot(df.index, df["MACD_signal"], label="Signal", color="orange")
            
            if "MACD_hist" in df.columns:
                couleurs = ['green' if val >= 0 else 'red' for val in df["MACD_hist"]]
                ax.bar(df.index, df["MACD_hist"], color=couleurs, alpha=0.5, label="Histogramme")
                
        ax.set_ylabel("MACD")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)


    # ---------------------------------------------------------------------------------------------
    def dessiner_rsi(self, ax, df, ticker):
        
        ax.clear()
        
        ax.set_facecolor('#1e1e1e') 
        ax.tick_params(colors='lightgray') 
        for spine in ax.spines.values():
            spine.set_color('#2d2d2d') 
        
        if "RSI" in df.columns:
            ax.plot(df.index, df["RSI"], label="RSI", color="purple")
            ax.axhline(70, color='red', linestyle='--', alpha=0.5)
            ax.axhline(30, color='green', linestyle='--', alpha=0.5)
            ax.fill_between(df.index, y1=30, y2=70, color='purple', alpha=0.05)
        # Affichage des signaux MACD
        if "Signal_MACD" in df.columns:
            achats = df[df['Signal_MACD'] == 1]
            ventes = df[df['Signal_MACD'] == -1]
            if not achats.empty:
                ax.scatter(achats.index, achats['MACD'], marker='^', color='#2ecc71', s=100, zorder=5)
            if not ventes.empty:
                ax.scatter(ventes.index, ventes['MACD'], marker='v', color='#e74c3c', s=100, zorder=5)
        
        # Affichage des signaux RSI
        if "Signal_RSI" in df.columns:
            achats = df[df['Signal_RSI'] == 1]
            ventes = df[df['Signal_RSI'] == -1]
            if not achats.empty:
                ax.scatter(achats.index, achats['RSI'], marker='^', color='#2ecc71', s=100, zorder=5)
            if not ventes.empty:
                ax.scatter(ventes.index, ventes['RSI'], marker='v', color='#e74c3c', s=100, zorder=5)
        
        ax.set_ylabel("RSI")
        ax.set_ylim(0, 100)
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)



# =================================================================================================
# ============================ DÉMARRAGE DE L'APPLICATION =========================================
# =================================================================================================

if __name__ == "__main__":
    
    app = QApplication(sys.argv)
    fenetre = ScreenerWindow()
    fenetre.show()
    sys.exit(app.exec())
