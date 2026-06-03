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
                             QTextEdit)

from strategies import StrategieCroisementMA, StrategieRSI, StrategieMACD
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

import matplotlib as mpl
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# IMPORTANT : Importez votre classe
from modeles import ActifFinancier 






# --- Classe pour gérer le canevas Multi-Graphiques ---
class DashboardCanvas(FigureCanvas):
    
    def __init__(self, parent=None, width=10, height=8, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        
        # 4 sous-graphiques empilés. sharex=True pour lier l'axe des dates.
        # height_ratios: donne 3x plus d'espace au prix par rapport aux autres.
        self.axes = self.fig.subplots(4, 1, sharex=True, gridspec_kw={'height_ratios': [2, 1, 1, 1]})
        
        self.fig.subplots_adjust(hspace=0.1, bottom=0.1)
        
        # ---> NOUVEAU : Fond noir pour la figure complète <---
        self.fig.patch.set_facecolor('black')
        
        super(DashboardCanvas, self).__init__(self.fig)
        self.fig.canvas.mpl_connect('scroll_event', self.zoom_molette)
        self.fig.canvas.mpl_connect('button_press_event', self.clic_presse)
        self.fig.canvas.mpl_connect('button_release_event', self.clic_relache)
        self.fig.canvas.mpl_connect('motion_notify_event', self.mouvement_souris)
        
        self.pan_axes = None
        self.press_x = None
        self.press_y = None
        
        # --- Variables pour le Tooltip (Survol) ---
        self.df_courant = None # Va contenir le DataFrame
        # Création d'une boîte de texte flottante, initialement invisible
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
            y_courbe = None
            if index_graphique == 0 and "Close" in row: y_courbe = row['Close']
            elif index_graphique == 1 and "Volatilite_20j" in row: y_courbe = row['Volatilite_20j']
            elif index_graphique == 2 and "MACD" in row: y_courbe = row['MACD']
            elif index_graphique == 3 and "RSI" in row: y_courbe = row['RSI']

            if y_courbe is not None:
                # On récupère la hauteur totale de l'axe survolé
                y_min, y_max = ax.get_ylim()
                # On définit une zone d'accroche (5% de la hauteur du graphique)
                tolerance = (y_max - y_min) * 0.05 
                
                # Si la souris est trop haute ou trop basse par rapport à la courbe
                if abs(event.ydata - y_courbe) > tolerance:
                    if self.tooltip.get_visible():
                        self.tooltip.set_visible(False)
                        self.fig.canvas.draw_idle()
                    return # On arrête ici, on n'affiche pas la bulle
            # =========================================================
            
            lignes = [f"{date_reelle}"]
            
            if index_graphique == 0:
                lignes.append(f"Prix : {row['Close']:.2f} $")
                if "SMA_20" in row: lignes.append(f"SMA 20 : {row['SMA_20']:.2f}")
                if "EMA_20" in row: lignes.append(f"EMA 20 : {row['EMA_20']:.2f}")
            elif index_graphique == 1:
                if "Volatilite_20j" in row: lignes.append(f"Volatilité : {row['Volatilite_20j']:.4f}")
            elif index_graphique == 2:
                if "MACD" in row: lignes.append(f"MACD : {row['MACD']:.2f}")
                if "MACD_signal" in row: lignes.append(f"Signal : {row['MACD_signal']:.2f}")
            elif index_graphique == 3:
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
        self.setStyleSheet("""
            QWidget { background-color: black; color: white; }
            QLineEdit { border: 1px solid #555555; border-radius: 4px; padding: 5px; background-color: #1E1E1E; color: white; }
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



    # ================= PAGE 0 : ACCUEIL =================
    def creer_page_accueil(self):
        page = QWidget()
        layout_principal = QVBoxLayout(page)
        layout_principal.addStretch(1)

        self.titre_page1 = QLabel("Scamming Land Screener")
        self.titre_page1.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        self.titre_page1.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout_principal.addWidget(self.titre_page1)

        layout_principal.addSpacing(12)

        label_ticker = QLabel("Ticker de l'entreprise :")
        label_ticker.setFont(QFont("Arial", 16))
        label_ticker.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout_principal.addWidget(label_ticker)

        layout_principal.addSpacing(12)

        layout_boite = QHBoxLayout()
        self.input_ticker = QLineEdit()
        self.input_ticker.setFont(QFont("Arial", 12))
        self.input_ticker.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.input_ticker.setPlaceholderText("")
        self.input_ticker.setFixedWidth(300)
        self.input_ticker.returnPressed.connect(self.lancer_analyse)

        layout_boite.addStretch()
        layout_boite.addWidget(self.input_ticker)
        layout_boite.addStretch()
        layout_principal.addLayout(layout_boite)

        layout_principal.addStretch(1)
        self.stacked_widget.addWidget(page)



    # ================= PAGE 1 : DASHBOARD =================
    def creer_page_dashboard(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        layout.addStretch(1)

        # Titre de la page 
        self.label_titre_dashboard = QLabel("Tableau de bord technique")
        self.label_titre_dashboard.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        self.label_titre_dashboard.setAlignment(Qt.AlignmentFlag.AlignHCenter) 
        layout.addWidget(self.label_titre_dashboard)

        # --- NOUVEAU : Label pour la prédiction IA ---
        self.label_prediction = QLabel("Prédiction IA : En attente...")
        self.label_prediction.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.label_prediction.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.label_prediction)
        
        # ================= AJOUTER CECI =================
        layout_navigation_ia = QHBoxLayout()
        self.btn_switch_to_ia = QPushButton("Consulter le Rapport Détaillé IA 🤖")
        self.btn_switch_to_ia.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        self.btn_switch_to_ia.setStyleSheet("background-color: #1e3d59; border: 1px solid #17b978;")
        self.btn_switch_to_ia.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(2))
        layout_navigation_ia.addStretch()
        layout_navigation_ia.addWidget(self.btn_switch_to_ia)
        layout_navigation_ia.addStretch()
        layout.addLayout(layout_navigation_ia)
        # ================================================

        # --- NOUVEAU : Zone d'affichage des backtests de stratégies ---
        self.console_strategies = QTextEdit()
        self.console_strategies.setReadOnly(True)
        # On utilise une police à chasse fixe (Courier) pour que les colonnes du tableau soient parfaitement alignées, comme dans un terminal.
        self.console_strategies.setFont(QFont("Courier", 10)) 
        self.console_strategies.setMaximumHeight(120)
        self.console_strategies.setStyleSheet("""
            QTextEdit {
                background-color: #121212; 
                color: #A0A0A0; 
                border: 1px solid #333333; 
                border-radius: 4px;
                padding: 5px;
            }
        """)
        layout.addWidget(self.console_strategies)
        
        layout.addSpacing(10)

        # --- NOUVEAU : Boutons de filtrage temporel ---
        layout_filtres = QHBoxLayout()
        btn_1m = QPushButton("1 Mois")
        btn_3m = QPushButton("3 Mois")
        btn_6m = QPushButton("6 Mois")
        btn_1a = QPushButton("1 An")

        # 1 mois ~ 21 jours de bourse
        btn_1m.clicked.connect(lambda: self.afficher_periode(21))
        btn_3m.clicked.connect(lambda: self.afficher_periode(63))
        btn_6m.clicked.connect(lambda: self.afficher_periode(126))
        btn_1a.clicked.connect(lambda: self.afficher_periode(252))

        layout_filtres.addStretch()
        layout_filtres.addWidget(QLabel("Période :"))
        layout_filtres.addWidget(btn_1m)
        layout_filtres.addWidget(btn_3m)
        layout_filtres.addWidget(btn_6m)
        layout_filtres.addWidget(btn_1a)
        layout_filtres.addStretch()
        
        layout.addLayout(layout_filtres)

        espaceur_dynamique = QSpacerItem(0, 12, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        layout.addSpacerItem(espaceur_dynamique)

        # Création du canvas multi-courbes
        self.canvas = DashboardCanvas(self, width=10, height=8, dpi=100)
        
        layout_canvas = QHBoxLayout()
        layout_canvas.addStretch()
        layout_canvas.addWidget(self.canvas)
        layout_canvas.addStretch()
        layout.addLayout(layout_canvas)

        espaceur_dynamique = QSpacerItem(0, 12, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        layout.addSpacerItem(espaceur_dynamique)

        # Bouton de retour
        layout_boutons = QHBoxLayout()
        btn_accueil = QPushButton("← Analyser un autre actif")
        btn_accueil.clicked.connect(self.retour_accueil)
        
        layout_boutons.addStretch()
        layout_boutons.addWidget(btn_accueil)
        layout_boutons.addStretch()

        layout.addLayout(layout_boutons)
        layout.addStretch(1)

        self.stacked_widget.addWidget(page)



    # ================= PAGE 2 : RAPPORT IA (NOUVELLE PAGE) =================
    def creer_page_ia_report(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        
        layout.addSpacing(15)
        
        self.label_titre_ia = QLabel("Rapport d'Optimisation & d'Évaluation de l'IA")
        self.label_titre_ia.setFont(QFont("Arial", 16, QFont.Weight.Bold))
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
        btn_retour_graph.setFont(QFont("Arial", 10, QFont.Weight.Bold))
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
        self.console_strategies.setText(texte_final)
        # =========================================================

        # Envoi du texte formaté vers le widget de l'interface
        self.console_strategies.setText(texte_final)
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
            
            if prediction is not None:
                if prediction > 0:
                    self.label_prediction.setText(f"Prédiction IA pour demain : 📈 HAUSSE")
                    self.label_prediction.setStyleSheet("color: #27ae60;") # Vert
                else:
                    self.label_prediction.setText(f"Prédiction IA pour demain : 📉 BAISSE")
                    self.label_prediction.setStyleSheet("color: #e74c3c;") # Rouge
            else:
                self.label_prediction.setText("Prédiction IA indisponible")
                self.label_prediction.setStyleSheet("color: orange;")
                
        except Exception as e:
            self.console_ia_text.setText(f"Erreur lors de la capture du flux IA :\n{str(e)}")
        finally:
            QApplication.restoreOverrideCursor()

        # On sauvegarde le DataFrame complet et le ticker pour le filtrage temporel
        self.df_complet = action.historique.copy()
        self.ticker_actuel = ticker

        # Mise à jour du titre
        self.label_titre_dashboard.setText(f"Tableau de bord technique : {ticker}")
        
        # === AJOUTER CECI : Mise à jour du titre du rapport IA ===
        self.label_titre_ia.setText(f"Rapport d'Analyse IA Dédié : {ticker}")

        # On affiche par défaut la dernière année (252 jours)
        self.afficher_periode(252)

        # On affiche la page du Dashboard
        self.stacked_widget.setCurrentIndex(1)



    # --- NOUVEAU : Méthode pour filtrer par période et redessiner ---
    def afficher_periode(self, jours_bourse):
        if self.df_complet is None:
            return
            
        # On découpe le DataFrame selon le nombre de jours demandés
        df = self.df_complet.tail(jours_bourse)
        
        # On met à jour le DataFrame du canvas pour le tooltip
        self.canvas.df_courant = df

        ax_prix, ax_vol, ax_macd, ax_rsi = self.canvas.axes

        self.dessiner_prix(ax_prix, df, self.ticker_actuel)
        self.dessiner_volatilite(ax_vol, df, self.ticker_actuel)
        self.dessiner_macd(ax_macd, df, self.ticker_actuel)
        self.dessiner_rsi(ax_rsi, df, self.ticker_actuel)

        self.formater_axe_x(ax_rsi)
        
        self.canvas.x_dates_num = mdates.date2num(df.index)

        self.canvas.fig.tight_layout()
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
