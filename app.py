# =================================================================================================
# ======================================= IMPORTATIONS ============================================
# =================================================================================================

import pandas as pd
import numpy as np
import io
import sys
import urllib.error

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, 
    QVBoxLayout, QWidget, QStackedWidget, QHBoxLayout, QMessageBox, 
    QSpacerItem, QSizePolicy, QTextEdit, QTabWidget, QTextBrowser, 
    QCheckBox, QRadioButton, QScrollArea, QProgressBar, QDialog, QFrame
)

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
    QCheckBox, QRadioButton
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
            fontsize=8, color="#f0f0f0",
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
# ============================ CLASSE : GEX CANVAS =================================================
# Canevas Matplotlib dédié au profil de Gamma Exposure (graphique en barres horizontales par strike)
# =================================================================================================

class GexCanvas(FigureCanvas):
    """
    Affiche le profil de Gamma Exposure (GEX) par strike sous forme de barres horizontales.
    Les barres vertes représentent un GEX net positif (zone de compression / résistance probable),
    les barres rouges un GEX net négatif (zone d'expansion / support probable).
    Le survol d'une barre affiche une bulle d'information avec le strike et la valeur exacte.
    """

    def __init__(self, parent=None, width=7, height=7, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_facecolor('#0c0c0c')
        super().__init__(self.fig)

        self.ax = self.fig.add_subplot(111)

        # Données courantes (mises à jour à chaque dessin)
        self.strikes = []
        self.valeurs = []

        self.fig.canvas.mpl_connect('motion_notify_event', self.survol_barre)

        self.tooltip = self.fig.text(
            0.0, 0.0, "",
            va="bottom", ha="left",
            fontsize=9, color="#f0f0f0",
            bbox=dict(boxstyle="round,pad=0.3", fc="#2A2A2A", ec="#555555", alpha=0.95),
            zorder=100, visible=False
        )

    # ---------------------------------------------------------------------------------------------
    def survol_barre(self, event):

        if event.inaxes != self.ax or not self.strikes:
            if self.tooltip.get_visible():
                self.tooltip.set_visible(False)
                self.fig.canvas.draw_idle()
            return

        try:
            idx = int(round(event.ydata))
            if idx < 0 or idx >= len(self.strikes):
                raise IndexError

            # Tolérance verticale : on n'affiche la bulle que si on survole bien la barre
            if abs(event.ydata - idx) > 0.45:
                if self.tooltip.get_visible():
                    self.tooltip.set_visible(False)
                    self.fig.canvas.draw_idle()
                return

            strike = self.strikes[idx]
            valeur = self.valeurs[idx]
            nature = "Compression (Résistance)" if valeur >= 0 else "Expansion (Support)"

            texte = f"Strike : {strike:.1f} $\nGEX net : {valeur:,.2f} M$\n{nature}"

            fig_w, fig_h = self.fig.get_size_inches() * self.fig.dpi
            pos_x = (event.x + 15) / fig_w
            pos_y = (event.y + 15) / fig_h
            if pos_x > 0.7: pos_x = (event.x - 140) / fig_w
            if pos_y > 0.85: pos_y = (event.y - 70) / fig_h

            nouvelle_position = (pos_x, pos_y)
            if not self.tooltip.get_visible() or self.tooltip.get_text() != texte or self.tooltip.get_position() != nouvelle_position:
                self.tooltip.set_text(texte)
                self.tooltip.set_position(nouvelle_position)
                self.tooltip.set_visible(True)
                self.fig.canvas.draw_idle()

        except Exception:
            if self.tooltip.get_visible():
                self.tooltip.set_visible(False)
                self.fig.canvas.draw_idle()


# =================================================================================================
# ============================ CLASSE : FENÊTRE GEX (CLICABLE) ====================================
# Fenêtre indépendante détaillant le profil complet du Gamma Exposure, avec les niveaux clés
# de Résistance (Call Wall), de Support (Put Wall), de Zero Gamma et de Max Pain.
# =================================================================================================

class FenetreGEX(QDialog):
    """
    Fenêtre clicable (ouverte via un bouton du tableau de bord) qui affiche :
      - Un graphique en barres horizontales du GEX net par strike.
      - Des repères horizontaux pour le Spot, la Résistance, le Support, le Zero Gamma et le Max Pain.
      - Un panneau latéral résumant ces niveaux et le régime de volatilité actuel.
    """

    def __init__(self, ticker, donnees_gex, parent=None):
        super().__init__(parent)

        self.setWindowTitle(f"Profil Gamma Exposure (GEX) — {ticker}")
        self.resize(1050, 740)
        self.setStyleSheet("""
            QDialog { background-color: #0c0c0c; }
            QLabel { color: #f0f0f0; }
        """)

        d = donnees_gex

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(8)

        # --- En-tête ---
        titre = QLabel(f"📊 Profil de Gamma Exposure — {ticker}")
        titre.setFont(QFont("Google Sans", 17, QFont.Weight.Bold))
        titre.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(titre)

        date_exp = d.get('date_expiration')
        sous_titre = QLabel(f"Échéance analysée : {date_exp if date_exp else 'Indisponible'}")
        sous_titre.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        sous_titre.setStyleSheet("color: #888888; font-size: 11px; font-weight: normal; margin-top: 0px;")
        layout.addWidget(sous_titre)

        ligne_separatrice = QFrame()
        ligne_separatrice.setFrameShape(QFrame.Shape.HLine)
        ligne_separatrice.setStyleSheet("background-color: #2A2A2A; max-height: 1px; border: none;")
        layout.addWidget(ligne_separatrice)
        layout.addSpacing(8)

        # --- Zone centrale : graphique (gauche) + panneau de niveaux clés (droite) ---
        zone_centrale = QHBoxLayout()
        zone_centrale.setSpacing(15)

        self.canvas = GexCanvas(self, width=7, height=7, dpi=100)
        zone_centrale.addWidget(self.canvas, 3)

        panneau = QTextBrowser()
        panneau.setReadOnly(True)
        panneau.setFixedWidth(270)
        panneau.setStyleSheet("""
            QTextBrowser {
                background-color: #181818;
                color: #e0e0e0;
                border: 1px solid #333333;
                border-radius: 8px;
                padding: 12px;
            }
        """)
        panneau.setHtml(self._construire_html_niveaux(d))
        zone_centrale.addWidget(panneau, 1)

        layout.addLayout(zone_centrale)

        # --- Bouton de fermeture ---
        layout_btn = QHBoxLayout()
        layout_btn.addStretch()
        btn_fermer = QPushButton("Fermer")
        btn_fermer.setFixedWidth(130)
        btn_fermer.clicked.connect(self.close)
        layout_btn.addWidget(btn_fermer)
        layout_btn.addStretch()
        layout.addLayout(layout_btn)

        self._dessiner_profil(d)

    # ---------------------------------------------------------------------------------------------
    @staticmethod
    def _fmt_prix(valeur):
        return f"{valeur:.1f} $" if valeur is not None else "N/A"

    # ---------------------------------------------------------------------------------------------
    def _construire_html_niveaux(self, d):

        gex_par_strike = d.get('gex_par_strike') or {}
        spot       = d.get('spot')
        max_pain   = d.get('max_pain')
        zero_gamma = d.get('zero_gamma')
        call_wall  = d.get('call_wall')
        put_wall   = d.get('put_wall')
        gex_total  = d.get('gex_total')

        if not gex_par_strike:
            return """
            <h3 style='color:#3498db; margin-top:0;'>Niveaux Clés</h3>
            <hr style='border-color:#444;'>
            <p style='color:#e74c3c; font-size:13px;'>
            ❌ Aucune chaîne d'options disponible pour cet actif.<br><br>
            Le profil GEX ne peut pas être calculé (ticker sans options listées, ou chaîne vide côté Yahoo Finance).
            </p>
            """

        if gex_total is not None and gex_total > 0:
            regime, couleur_regime = "🟢 Compression (Pro-MM)", "#27ae60"
        elif gex_total is not None and gex_total < 0:
            regime, couleur_regime = "🔴 Expansion (Anti-MM)", "#e74c3c"
        else:
            regime, couleur_regime = "Indisponible", "#888888"

        gex_total_txt = f"{gex_total / 1_000_000:,.2f} M$" if gex_total is not None else "N/A"

        return f"""
        <h3 style='color:#3498db; margin-top:0;'>Niveaux Clés</h3>
        <hr style='border-color:#444;'>
        <p><b style='color:#f0f0f0;'>📍 Prix Spot :</b><br>{self._fmt_prix(spot)}</p>
        <p><b style='color:#2ecc71;'>🟩 Résistance (Call Wall) :</b><br>{self._fmt_prix(call_wall)}</p>
        <p><b style='color:#e74c3c;'>🟥 Support (Put Wall) :</b><br>{self._fmt_prix(put_wall)}</p>
        <p><b style='color:#bdc3c7;'>⚖️ Zero Gamma Level :</b><br>{self._fmt_prix(zero_gamma)}</p>
        <p><b style='color:#f39c12;'>🎯 Max Pain :</b><br>{self._fmt_prix(max_pain)}</p>
        <hr style='border-color:#444;'>
        <p><b style='color:{couleur_regime};'>Régime actuel :</b><br>{regime}</p>
        <p style='font-size:11px; color:#888;'>GEX Total : {gex_total_txt}</p>
        <hr style='border-color:#444;'>
        <p style='font-size:11px; color:#999; line-height:1.4;'>
        💡 <b>Lecture :</b> Au-dessus du Zero Gamma, les Market Makers amortissent les mouvements 
        (volatilité écrasée). En-dessous, ils les amplifient (risque d'accélération). La Résistance 
        et le Support marquent les strikes où leur couverture est la plus dense.
        </p>
        """

    # ---------------------------------------------------------------------------------------------
    def _dessiner_profil(self, d):

        ax = self.canvas.ax
        ax.clear()
        ax.set_facecolor('#0c0c0c')
        ax.tick_params(colors='#f0f0f0')
        for spine in ax.spines.values():
            spine.set_color('#333333')

        gex_par_strike = d.get('gex_par_strike') or {}

        if not gex_par_strike:
            ax.text(0.5, 0.5, "Aucune donnée d'options disponible",
                    color='#888888', fontsize=12, ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            self.canvas.draw()
            return

        spot = d.get('spot')

        # On limite l'affichage à une fenêtre pertinente autour du prix actuel pour garder
        # un graphique lisible, tout en s'assurant d'inclure les niveaux clés (murs, max pain...).
        niveaux_cles = [v for v in [spot, d.get('call_wall'), d.get('put_wall'),
                                     d.get('zero_gamma'), d.get('max_pain')] if v]
        tous_strikes = sorted(gex_par_strike.keys())

        if niveaux_cles and spot:
            marge = max(spot * 0.25, 5)
            borne_min = min(niveaux_cles) - marge
            borne_max = max(niveaux_cles) + marge
            strikes_filtres = [s for s in tous_strikes if borne_min <= s <= borne_max]
            strikes = strikes_filtres if strikes_filtres else tous_strikes
        else:
            strikes = tous_strikes

        valeurs = [gex_par_strike[k] / 1_000_000 for k in strikes]  # en millions de $

        self.canvas.strikes = strikes
        self.canvas.valeurs = valeurs

        couleurs = ['#2ecc71' if v >= 0 else '#e74c3c' for v in valeurs]
        y_pos = list(range(len(strikes)))

        ax.barh(y_pos, valeurs, color=couleurs, edgecolor='none', height=0.7, zorder=3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{s:.1f}" for s in strikes], fontsize=8)
        ax.axvline(0, color='#555555', linewidth=1, zorder=2)

        # --- Repères horizontaux pour les niveaux clés (indexés sur la position du strike le + proche) ---
        def _ligne_niveau(valeur, couleur, nom):
            if valeur is None or not strikes:
                return
            idx_proche = min(range(len(strikes)), key=lambda i: abs(strikes[i] - valeur))
            ax.axhline(idx_proche, color=couleur, linestyle='--', linewidth=1.7,
                       alpha=0.9, zorder=4, label=f"{nom} ({valeur:.1f} $)")

        _ligne_niveau(spot, '#f0f0f0', "Spot")
        _ligne_niveau(d.get('call_wall'), '#2ecc71', "Résistance")
        _ligne_niveau(d.get('put_wall'), '#e74c3c', "Support")
        _ligne_niveau(d.get('zero_gamma'), '#3498db', "Zero Gamma")
        _ligne_niveau(d.get('max_pain'), '#f39c12', "Max Pain")

        ax.set_xlabel("Gamma Exposure Net (Millions $ par 1% de mouvement)", color='#f0f0f0', fontsize=10)
        ax.set_ylabel("Strike ($)", color='#f0f0f0', fontsize=10)
        ax.set_title("Profil GEX par Strike — Vert = Résistance probable | Rouge = Support probable",
                     color='#f0f0f0', fontsize=11)

        ax.legend(loc='upper right', fontsize=8, facecolor='#181818', edgecolor='#333333',
                  labelcolor='#f0f0f0', framealpha=0.95)
        ax.grid(True, axis='x', color='#181818', linewidth=1.0)

        self.canvas.fig.subplots_adjust(left=0.16, right=0.97, top=0.93, bottom=0.09)
        self.canvas.draw()


# =================================================================================================
# ============================ CLASSE PRINCIPALE : L'APPLICATION ==================================
# Interface globale gérant les différentes vues (Accueil, Dashboard, Rapport IA)
# =================================================================================================

class ScreenerWindow(QMainWindow):
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Scamming Land Screener")
        self.setWindowTitle("Scamming Land Screener")
        self.resize(1200, 800) # Taille de repli si le plein écran est quitté
        self.showMaximized()   # <--- Force le plein écran dynamique
        
        # Application du style global (Dark Mode)
        self.setStyleSheet("""
            QWidget { 
                font-family: 'Google Sans', sans-serif; 
                background-color: #0c0c0c; 
                color: #f0f0f0; 
            }
            QLineEdit { 
                border: 0px transparent #555555; 
                border-radius: 4px; 
                padding: 5px; 
                background-color: #1E1E1E; 
                color: #f0f0f0; 
            }
            QLineEdit[text=""] {
                color: #b4b4b4; 
            }
            QPushButton { 
                border: 1px solid #555555; 
                border-radius: 4px; 
                padding: 8px 15px; 
                background-color: #2A2A2A; /* <-- CORRIGÉ : Gris foncé au lieu de Blanc */
                color: #f0f0f0; 
            }
            QPushButton:hover { 
                background-color: #3A3A3A; 
            }
            QLabel { 
                color: #f0f0f0; 
            }
            /* --- STYLE DES SCROLLBARS --- */
            QScrollBar:vertical {
                border: none;
                background-color: #121212;
                width: 8px; /* Un poil plus large pour laisser l'arrondi s'exprimer */
                border-radius: 3px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background-color: #444444;
                min-height: 30px;
                border-radius: 3px;
                margin: 1px; /* LA CLÉ EST ICI : cela décolle le curseur des bords et révèle l'arrondi */
            }
            QScrollBar::handle:vertical:hover {
                background-color: #666666;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px; 
                border: none; /* Empêche tout artefact visuel aux extrémités */
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none; 
            }
        """)

        # Gestionnaire de pages
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        # Variables de stockage des données
        self.df_complet = None
        self.ticker_actuel = ""
        self.jours_actuels = 252
        
        # --- AJOUT ICI : Définition des polices Matplotlib ---
        self.police_axes = {
            'family': 'Google Sans', # Ou 'sans-serif' si Google Sans n'est pas reconnue par Matplotlib
            'weight': 'regular',        # C'est ici que tu forces le poids (bold, medium, etc.)
            'size': 12,
            'color': '#f0f0f0'
        }
        
        self.police_legendes = {
            'family': 'Google Sans',
            'weight': 'regular',
            'size': 10,
        }

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
                border-radius: 23px; 
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
                color: #f0f0f0;
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
    # =============================================================================================
    # ============================ PAGE 1 : DASHBOARD =============================================
    # =============================================================================================
    def creer_page_dashboard(self):
        
        page = QWidget()
        layout_global = QHBoxLayout(page)
        layout_global.setContentsMargins(0, 0, 0, 0)
        layout_global.setSpacing(0)
        
        # -----------------------------------------------------------------------------------------
        # BARRE LATÉRALE AVEC SCROLL (La correction est ici)
        # -----------------------------------------------------------------------------------------
        # 1. Création du conteneur de défilement principal
        zone_scroll = QScrollArea()
        zone_scroll.setFixedWidth(370) # Un peu plus large pour inclure la barre de scroll
        zone_scroll.setWidgetResizable(True)
        zone_scroll.setStyleSheet("QScrollArea { border: none; background-color: #181818; }")
        
        # 2. Ton widget sidebar classique (qui va se glisser à l'intérieur du scroll)
        sidebar = QWidget()
        sidebar.setStyleSheet("""
            QWidget { 
                background-color: #181818; 
                border-radius: 0px; 
                padding-left: 10px;
                padding-right: 10px;
            }
            QLabel { 
                font-weight: bold;
                font-size: 15px; 
                margin-top: 10px;
                color: #f0f0f0; 
            }
            QCheckBox, QRadioButton { 
                font-size: 13.5px;
                padding: 3px; 
                color: #787878;
            }
            QCheckBox::indicator, QRadioButton::indicator { 
                width: 10px; 
                height: 10px; 
                border-radius: 6px; 
                border: 1px solid #787878; 
            }
            QCheckBox::indicator:checked, QRadioButton::indicator:checked { 
                background-color: #787878; 
            }
            QPushButton { 
                margin: 5px; 
                border-radius: 10px; 
            }
        """)
        
        layout_sidebar = QVBoxLayout(sidebar)

        # Petit utilitaire pour créer des séparateurs visuels entre les sections de la sidebar
        def _separateur():
            ligne = QFrame()
            ligne.setFrameShape(QFrame.Shape.HLine)
            ligne.setStyleSheet("background-color: #2A2A2A; max-height: 1px; border: none; margin: 6px 0px;")
            return ligne
        
        # --- SECTION : INFORMATIONS IA ET PRÉDICTIONS ---
        layout_sidebar.addWidget(QLabel("ANALYSE INTELLIGENCE ARTIFICIELLE"))
        
        self.label_prediction = QLabel("Prédiction IA : En attente...")
        self.label_prediction.setFont(QFont("Google Sans", 12, QFont.Weight.Bold))
        self.label_prediction.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.label_prediction.setStyleSheet("color: #f0f0f0;")
        layout_sidebar.addWidget(self.label_prediction)

        self.btn_switch_to_ia = QPushButton("Consulter le Rapport Détaillé IA 🤖")
        self.btn_switch_to_ia.setFont(QFont("Google Sans", 10, QFont.Weight.Bold))
        self.btn_switch_to_ia.setStyleSheet("background-color: #1e3d59; border: 1px solid #17b978; color: white;")
        self.btn_switch_to_ia.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(2))
        layout_sidebar.addWidget(self.btn_switch_to_ia)
        
        # (À l'intérieur de creer_page_dashboard, sous le bouton btn_switch_to_ia)
        layout_sidebar.addSpacing(15)
        layout_sidebar.addWidget(_separateur())
        
        layout_sidebar.addWidget(QLabel("MÉCANIQUE OPEX & GEX"))

        self.affichage_gex = QTextBrowser()
        self.affichage_gex.setReadOnly(True)
        self.affichage_gex.setMinimumHeight(140)
        self.affichage_gex.setMaximumHeight(220)
        self.affichage_gex.setStyleSheet("""
            QTextBrowser {
                background-color: #121212; 
                color: #e0e0e0; 
                border: 1px solid #333333; 
                border-radius: 6px;
                padding: 8px;
            }
        """)
        self.affichage_gex.setHtml("<p style='color:#888; font-size:12px;'>Phase actuelle : en attente d'analyse...</p>")
        layout_sidebar.addWidget(self.affichage_gex)

        self.btn_profil_gex = QPushButton("📊 Voir le Profil GEX (Support / Résistance)")
        self.btn_profil_gex.setFont(QFont("Google Sans", 10, QFont.Weight.Bold))
        self.btn_profil_gex.setStyleSheet("""
            QPushButton { background-color: #2a1f0d; border: 1px solid #f39c12; color: #f39c12; }
            QPushButton:hover { background-color: #3a2a10; }
        """)
        self.btn_profil_gex.clicked.connect(self.ouvrir_fenetre_gex)
        layout_sidebar.addWidget(self.btn_profil_gex)

        layout_sidebar.addSpacing(10)
        layout_sidebar.addWidget(_separateur())

        # --- SECTION : PERFORMANCES STRATÉGIES ---
        layout_sidebar.addWidget(QLabel("BACKTESTING STRATÉGIES"))
        
        self.affichage_perf = QTextBrowser()
        self.affichage_perf.setReadOnly(True)
        self.affichage_perf.setMinimumHeight(180)
        self.affichage_perf.setMaximumHeight(360)
        self.affichage_perf.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
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
        layout_sidebar.addWidget(_separateur())
        
        # --- SECTION : ACTUALITÉS ET FONDAMENTAL ---
        layout_sidebar.addWidget(QLabel("ACTUALITÉS ET SENTIMENT"))
        
        self.affichage_news = QTextBrowser()
        self.affichage_news.setReadOnly(True)
        self.affichage_news.setOpenExternalLinks(True)
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
        layout_sidebar.addWidget(_separateur())
        
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
        self.chk_opex = QCheckBox("Phases OpEx (Gamma)")
        self.chk_opex.setChecked(True)
        self.chk_opex.setStyleSheet("color: #f39c12; font-weight: bold;")
        self.chk_opex.stateChanged.connect(self.actualiser_graphiques)
        layout_sidebar.addWidget(self.chk_opex)

        self.chk_gex_walls = QCheckBox("Murs GEX (Résistance/Support)")
        self.chk_gex_walls.setChecked(True)
        self.chk_gex_walls.setStyleSheet("color: #2ecc71; font-weight: bold;")
        self.chk_gex_walls.stateChanged.connect(self.actualiser_graphiques)
        layout_sidebar.addWidget(self.chk_gex_walls)
        
        self.chk_prix.stateChanged.connect(self.actualiser_graphiques)
        self.chk_vol.stateChanged.connect(self.actualiser_graphiques)
        self.chk_macd.stateChanged.connect(self.actualiser_graphiques)
        self.chk_rsi.stateChanged.connect(self.actualiser_graphiques)
        
        layout_sidebar.addWidget(self.chk_prix)
        layout_sidebar.addWidget(self.chk_vol)
        layout_sidebar.addWidget(self.chk_macd)
        layout_sidebar.addWidget(self.chk_rsi)
        
        layout_sidebar.addSpacing(15)
        layout_sidebar.addWidget(_separateur())
        
        # --- SECTION : CONTRÔLE DE LA PÉRIODE ---
        layout_sidebar.addWidget(QLabel("PÉRIODE D'ANALYSE"))
        
        btn_1m = QRadioButton("1 Mois")
        btn_3m = QRadioButton("3 Mois")
        btn_6m = QRadioButton("6 Mois")
        btn_1a = QRadioButton("1 An")
        
        btn_1a.setChecked(True)
        
        btn_1m.toggled.connect(lambda checked: self.changer_periode(21) if checked else None)
        btn_3m.toggled.connect(lambda checked: self.changer_periode(63) if checked else None)
        btn_6m.toggled.connect(lambda checked: self.changer_periode(126) if checked else None)
        btn_1a.toggled.connect(lambda checked: self.changer_periode(252) if checked else None)
        
        layout_sidebar.addWidget(btn_1m)
        layout_sidebar.addWidget(btn_3m)
        layout_sidebar.addWidget(btn_6m)
        layout_sidebar.addWidget(btn_1a)
        
        layout_sidebar.addStretch()
        
        # --- BOUTON DE RETOUR (Bas de la barre latérale) ---
        btn_accueil = QPushButton("Changer d'actif")
        btn_accueil.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c; 
                color: #f0f0f0; 
                border-radius: 15px; 
                font-weight: bold; 
                font-size: 15px; 
                padding: 6px; 
                border: none;
            }
            QPushButton:hover {
                background-color: #c0392b; 
            }
        """)
        btn_accueil.clicked.connect(self.retour_accueil)
        layout_sidebar.addWidget(btn_accueil)
        
        
        # -----------------------------------------------------------------------------------------
        # ASSEMBLAGE DE LA ZONE GAUCHE (Scroll) ET DROITE (Graphiques)
        # -----------------------------------------------------------------------------------------
        # On insère la sidebar dans la zone de scroll
        zone_scroll.setWidget(sidebar)
        
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
        
        # Assemblage final de la page (On ajoute zone_scroll au lieu de sidebar)
        layout_global.addWidget(zone_scroll)
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
        
        btn_nouveau_ticker = QPushButton("Analyser un autre actif")
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
        
        # --- NOUVEAUX APPELS GEX (Avant les stratégies pour qu'elles puissent filtrer) ---
        action.calculer_calendrier_opex()
        action.annoter_phases_opex()
        action.calculer_max_pain()
        action.calculer_profil_gex()


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
            
            if prediction is not None:
                if prediction > 0:
                    self.label_prediction.setText(f"📈 (IA) Tendance Haussière\n Edge Statistique : +{(hit_ratio_actuel * 100) - 50:.2f}%")
                    self.label_prediction.setStyleSheet("color: #27ae60; font-weight: bold; margin-top:0px;") 
                else:
                    self.label_prediction.setText(f"📉 (IA) Tendance Baissière \n Edge Statistique : {(hit_ratio_actuel*100) - 50:.2f}%")
                    self.label_prediction.setStyleSheet("color: #e74c3c; font-weight: bold; margin-top:0px;") 
            else:
                # Ne s'affiche que si le modèle a complètement planté et n'a sorti aucune valeur
                self.label_prediction.setText("❌ Prédiction IA\néchouée")
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
        QApplication.processEvents() 
        
        try:
            action.analyser_fondamental()
            
            html_content = f"<h4 style='color: #17b978; margin:0;'>Actualités {ticker}</h4><hr style='border-color: #555;'>"
            
            if not action.news:
                html_content += f"<p style='color: #A0A0A0; font-size: 12px;'> ✅ Aucune actualité récente trouvée.</p>"
            else:
                for article in action.news:
                    emoji = "⚪"
                    if "POSITIF" in article['sentiment']: emoji = "🟢"
                    elif "NÉGATIF" in article['sentiment']: emoji = "🔴"
                    
                    html_content += f"""
                    <p style="margin-top: 5px; margin-bottom: 5px;">
                        <span style="font-size: 12px;">{emoji}</span>
                        <a href="{article['lien']}" style="font-size: 12px; font-weight: bold; text-decoration: none; color: #3498db;">
                            {article['titre']}
                        </a>
                    </p>
                    <hr style="background-color: #333333; height: 1px; border: none; margin-top: 8px; margin-bottom: 8px;">
                    """
            self.affichage_news.setHtml(html_content)
            
        except urllib.error.HTTPError as e:
            html_content = f"<h4 style='color: #e74c3c;'>❌ Accès Refusé</h4><hr style='border-color: #555;'>"
            html_content += f"<p style='font-size:12px;'>Rejeté par Yahoo (Erreur HTTP {e.code}).</p>"
            self.affichage_news.setHtml(html_content)
            
        except Exception as e:
            html_content = f"<h4 style='color: #e74c3c;'>❌ Erreur de Connexion</h4><hr style='border-color: #555;'>"
            html_content += f"<p style='font-size:12px;'>Impossible de joindre le serveur.</p>"
            self.affichage_news.setHtml(html_content)


        # -----------------------------------------------------------------------------------------
        # FINALISATION ET MISE À JOUR VISUELLE
        # -----------------------------------------------------------------------------------------
        self.df_complet = action.historique.copy()
        
        # Sécurité : vérifier que l'objet possède l'attribut avant de le copier
        if getattr(action, 'df_opex', None) is not None:
            self.df_opex_courant = action.df_opex.copy()
        
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
            couleur_dd = "#2ecc71" if res['max_dd_strat'] < res['max_dd_bh'] else "#e0e0e0"
            
            html_perf += f"""
            <div style='margin-bottom: 8px; font-size: 12px;'>
                <b>{res['nom']}</b> : <span style='color: {couleur}; font-weight: bold;'>{res['performance_strategie']}%</span> 
                <span style='color: #888;'>(B&H: {res['performance_buy_hold']}%)</span><br>
                <span style='color: #AAA;'>↳ Trades : {res['nombre_trades']} | Sharpe : {res['sharpe_ratio']} </span><br>
                <span style='color: #AAA;'>↳ Pire chute (Max DD) : <b style='color: {couleur_dd};'>-{res['max_dd_strat']}%</b> vs -{res['max_dd_bh']}% (B&H)</span>
            </div>
            """
        self.affichage_perf.setHtml(html_perf)

        # 3. --- MISE À JOUR DU PANNEAU GEX AVEC BLACK-SCHOLES ---
        # On conserve les données brutes sur l'instance pour la fenêtre dédiée et le graphique de prix.
        self.gex_par_strike_courant       = getattr(action, 'gex_par_strike', {}) or {}
        self.gex_calls_par_strike_courant = getattr(action, 'gex_calls_par_strike', {}) or {}
        self.gex_puts_par_strike_courant  = getattr(action, 'gex_puts_par_strike', {}) or {}
        self.call_wall_courant            = getattr(action, 'call_wall', None)
        self.put_wall_courant             = getattr(action, 'put_wall', None)
        self.zero_gamma_courant           = getattr(action, 'zero_gamma_level', None)
        self.max_pain_courant             = getattr(action, 'max_pain', None)
        self.spot_price_courant           = getattr(action, 'spot_price', None)
        self.gex_total_courant            = getattr(action, 'gex_total', None)
        self.date_expiration_gex_courant  = getattr(action, 'date_expiration_gex', None)

        if 'Phase_OpEx' in action.historique.columns:
            phase_du_jour = action.historique['Phase_OpEx'].iloc[-1]

            html_gex = f"<p style='margin:0 0 6px 0;'>📅 <b>Cycle :</b> {phase_du_jour}</p>"

            if self.max_pain_courant is not None:
                html_gex += f"<p style='margin:0 0 6px 0;'>🎯 <b>Max Pain :</b> {self.max_pain_courant:.1f} $</p>"

            if self.gex_total_courant is not None and self.gex_par_strike_courant:
                gex_m = self.gex_total_courant / 1_000_000
                couleur_gex = "#27ae60" if self.gex_total_courant > 0 else "#e74c3c"
                emoji_gex = "🟢" if self.gex_total_courant > 0 else "🔴"

                html_gex += f"<p style='margin:0 0 6px 0;'>{emoji_gex} <b>GEX Total :</b> <span style='color:{couleur_gex};'>{gex_m:,.2f} M$</span></p>"
                html_gex += f"<p style='margin:0 0 6px 0;'>⚖️ <b>Zero Gamma :</b> {self.zero_gamma_courant:.1f} $</p>"

                if self.call_wall_courant is not None:
                    html_gex += f"<p style='margin:0 0 6px 0; color:#2ecc71;'>🟩 <b>Résistance (Call Wall) :</b> {self.call_wall_courant:.1f} $</p>"
                if self.put_wall_courant is not None:
                    html_gex += f"<p style='margin:0 0 6px 0; color:#e74c3c;'>🟥 <b>Support (Put Wall) :</b> {self.put_wall_courant:.1f} $</p>"

                if self.gex_total_courant > 0:
                    html_gex += "<p style='margin:6px 0 0 0; font-size:11px; color:#f39c12;'>⚠️ MMs en Contre-Tendance. La volatilité est écrasée.</p>"
                else:
                    html_gex += "<p style='margin:6px 0 0 0; font-size:11px; color:#f39c12;'>🚀 MMs en Pro-Tendance. Risque d'accélération directionnelle.</p>"
            else:
                html_gex += "<p style='margin:6px 0 0 0; font-size:11px; color:#888;'>Aucune chaîne d'options disponible pour ce ticker.</p>"

            self.affichage_gex.setHtml(html_gex)

        # 4. Dessin final et bascule d'affichage
        self.changer_periode(self.jours_actuels)
        self.stacked_widget.setCurrentIndex(1)

    # ---------------------------------------------------------------------------------------------
    # OUVERTURE DE LA FENÊTRE DÉDIÉE AU PROFIL GEX (Support / Résistance)
    # ---------------------------------------------------------------------------------------------
    def ouvrir_fenetre_gex(self):

        if not getattr(self, 'ticker_actuel', ""):
            QMessageBox.information(self, "Aucune analyse", "Lance d'abord une analyse sur un ticker.")
            return

        if not getattr(self, 'gex_par_strike_courant', None):
            QMessageBox.information(
                self, "GEX indisponible",
                f"Aucune chaîne d'options n'est disponible pour {self.ticker_actuel}.\n"
                "Le profil GEX ne peut pas être calculé pour cet actif."
            )
            return

        donnees_gex = {
            'gex_par_strike': self.gex_par_strike_courant,
            'spot':           self.spot_price_courant,
            'max_pain':       self.max_pain_courant,
            'zero_gamma':     self.zero_gamma_courant,
            'call_wall':      self.call_wall_courant,
            'put_wall':       self.put_wall_courant,
            'gex_total':      self.gex_total_courant,
            'date_expiration': self.date_expiration_gex_courant,
        }

        # On garde une référence pour éviter que la fenêtre soit détruite par le garbage collector
        self._fenetre_gex = FenetreGEX(self.ticker_actuel, donnees_gex, parent=self)
        self._fenetre_gex.show()

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
        ax.set_facecolor('#0c0c0c')  
        ax.tick_params(colors='#f0f0f0') 
        for spine in ax.spines.values():
            spine.set_color('#181818')
            spine.set_linewidth(1.0) 
        
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

        # On mémorise l'échelle Y "naturelle" du prix avant d'ajouter les repères GEX,
        # afin de ne jamais déformer le graphique si un mur se trouve loin du prix actuel.
        y_min_prix, y_max_prix = ax.get_ylim()

        if hasattr(self, 'chk_opex') and self.chk_opex.isChecked() and hasattr(self, 'df_opex_courant'):
            
            # Variables pour éviter de dupliquer la légende 100 fois
            legende_p1_ajoutee = False
            legende_p2_ajoutee = False
            
            for _, row in self.df_opex_courant.iterrows():
                opex_date = row['OpEx_Date']
                is_quad = row['Is_Quad_Witching']
                
                # On ne dessine que si l'OpEx est proche des dates affichées sur le graph
                if opex_date < df.index[0] - pd.Timedelta(days=15) or opex_date > df.index[-1] + pd.Timedelta(days=15):
                    continue
                
                # Phase 1 : Max Pain (Aimant) -> J-7 à Jour J
                phase1_start = opex_date - pd.Timedelta(days=7)
                
                # Phase 2 : Shakeout (Directionnel) -> Jour J à J+7
                phase2_end = opex_date + pd.Timedelta(days=7)
                
                # Couleurs : Plus intenses si c'est les 4 Sorcières (Quad Witching)
                color_p1 = '#e67e22' if is_quad else '#f1c40f' # Orange / Jaune
                color_p2 = '#8e44ad' if is_quad else '#3498db' # Violet / Bleu
                alpha_val = 0.2 if is_quad else 0.1
                
                # 1. Dessiner la zone de Compression (Max Pain)
                label_p1 = "Phase 1: Compression (GEX+)" if not legende_p1_ajoutee else ""
                ax.axvspan(phase1_start, opex_date, color=color_p1, alpha=alpha_val, lw=0, label=label_p1)
                legende_p1_ajoutee = True
                
                # 2. Dessiner la zone de Libération (Shakeout)
                label_p2 = "Phase 2: Shakeout (GEX-)" if not legende_p2_ajoutee else ""
                ax.axvspan(opex_date, phase2_end, color=color_p2, alpha=alpha_val, lw=0, label=label_p2)
                legende_p2_ajoutee = True
                
                # 3. La ligne rouge exacte du 3ème vendredi (OpEx)
                ax.axvline(opex_date, color='#e74c3c', linestyle=':', linewidth=1.5, alpha=0.8)

        # --- REPÈRES DE RÉSISTANCE / SUPPORT ISSUS DU PROFIL GEX ---
        if hasattr(self, 'chk_gex_walls') and self.chk_gex_walls.isChecked():

            call_wall  = getattr(self, 'call_wall_courant', None)
            put_wall   = getattr(self, 'put_wall_courant', None)
            zero_gamma = getattr(self, 'zero_gamma_courant', None)
            max_pain   = getattr(self, 'max_pain_courant', None)

            if call_wall is not None:
                ax.axhline(call_wall, color='#2ecc71', linestyle='--', linewidth=1.6, alpha=0.85, zorder=4,
                          label=f"Résistance (Call Wall) : {call_wall:.1f} $")
            if put_wall is not None:
                ax.axhline(put_wall, color='#e74c3c', linestyle='--', linewidth=1.6, alpha=0.85, zorder=4,
                          label=f"Support (Put Wall) : {put_wall:.1f} $")
            if zero_gamma is not None:
                ax.axhline(zero_gamma, color='#3498db', linestyle=':', linewidth=1.4, alpha=0.7, zorder=4,
                          label=f"Zero Gamma : {zero_gamma:.1f} $")
            if max_pain is not None:
                ax.axhline(max_pain, color='#f39c12', linestyle=':', linewidth=1.4, alpha=0.6, zorder=4,
                          label=f"Max Pain : {max_pain:.1f} $")

            # On élargit légèrement l'échelle si un niveau est proche du prix, mais on l'empêche
            # de partir trop loin si un mur se trouve à un strike éloigné (lisibilité du chandelier).
            niveaux = [v for v in [call_wall, put_wall, zero_gamma, max_pain] if v is not None]
            marge_prix = (y_max_prix - y_min_prix) * 0.6
            for niveau in niveaux:
                if y_min_prix - marge_prix <= niveau <= y_max_prix + marge_prix:
                    y_min_prix = min(y_min_prix, niveau)
                    y_max_prix = max(y_max_prix, niveau)

            padding = (y_max_prix - y_min_prix) * 0.05
            ax.set_ylim(y_min_prix - padding, y_max_prix + padding)

        ax.set_ylabel("Prix ($)", color="#f0f0f0", fontdict=self.police_axes)
        ax.legend(
            loc="upper left", 
            prop=self.police_legendes, 
            labelcolor="#f0f0f0", 
            facecolor='#181818',  # La couleur de la plaque (ici le gris très foncé de ton fond)
            edgecolor='#181818',  # La couleur de la bordure de la plaque
            framealpha=1.0,        # Légère transparence (1 = 100% opaque, 0 = 100% transparent)
            fontsize=8
            )
        ax.grid(True, color='#181818', linewidth=1.0)


    # ---------------------------------------------------------------------------------------------
    def dessiner_volatilite(self, ax, df, ticker):
        
        ax.clear()
        
        ax.set_facecolor('#0c0c0c')  
        ax.tick_params(colors='#f0f0f0') 
        for spine in ax.spines.values():
            spine.set_color('#181818')
            spine.set_linewidth(1.0) 
        
        if "Volatilite_20j" in df.columns:
            ax.plot(df.index, df["Volatilite_20j"], label="Volatilité (20j)", color="red")
            
        ax.set_ylabel("Volatilité", color="#f0f0f0", fontdict=self.police_axes)
        ax.legend(
            loc="upper left", 
            prop=self.police_legendes, 
            labelcolor="#f0f0f0", 
            facecolor='#181818',  # La couleur de la plaque (ici le gris très foncé de ton fond)
            edgecolor='#181818',  # La couleur de la bordure de la plaque
            framealpha=1.0        # Légère transparence (1 = 100% opaque, 0 = 100% transparent)
            )
        ax.grid(True, color='#181818', linewidth=1.0)


    # ---------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------
    def dessiner_macd(self, ax, df, ticker):
        
        ax.clear()
        
        ax.set_facecolor('#0c0c0c')  
        ax.tick_params(colors='#f0f0f0') 
        for spine in ax.spines.values():
            spine.set_color('#181818')
            spine.set_linewidth(1.0) 
        
        if "MACD" in df.columns and "MACD_signal" in df.columns:
            
            ax.plot(df.index, df["MACD"], label="MACD", color="blue")
            ax.plot(df.index, df["MACD_signal"], label="Signal", color="orange")
            
            if "MACD_hist" in df.columns:
                couleurs = ['green' if val >= 0 else 'red' for val in df["MACD_hist"]]
                ax.bar(df.index, df["MACD_hist"], color=couleurs, alpha=0.5, label="Histogramme")
                
        if "Signal_MACD" in df.columns:
            achats = df[df['Signal_MACD'] == 1]
            ventes = df[df['Signal_MACD'] == -1]
            if not achats.empty:
                ax.scatter(achats.index, achats['MACD'], marker='^', color='#2ecc71', s=120, zorder=5)
            if not ventes.empty:
                ax.scatter(ventes.index, ventes['MACD'], marker='v', color='#e74c3c', s=120, zorder=5)
                
        ax.set_ylabel("MACD", color="#f0f0f0", fontdict=self.police_axes)
        ax.legend(
            loc="upper left", 
            prop=self.police_legendes, 
            labelcolor="#f0f0f0", 
            facecolor='#181818', 
            edgecolor='#181818', 
            framealpha=1.0        
            )
        ax.grid(True, color='#181818', linewidth=1.0)


    # ---------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------
    def dessiner_rsi(self, ax, df, ticker):
        
        ax.clear()
        
        ax.set_facecolor('#0c0c0c')  
        ax.tick_params(colors='#f0f0f0') 
        for spine in ax.spines.values():
            spine.set_color('#181818')
            spine.set_linewidth(1.0)
        
        if "RSI" in df.columns:
            ax.plot(df.index, df["RSI"], label="RSI", color="purple")
            ax.axhline(70, color='red', linestyle='--', alpha=0.5)
            ax.axhline(30, color='green', linestyle='--', alpha=0.5)
            ax.fill_between(df.index, y1=30, y2=70, color='purple', alpha=0.05)
            
        # --- CORRECTION : Seuls les signaux RSI sont affichés ici ---
        if "Signal_RSI" in df.columns:
            achats = df[df['Signal_RSI'] == 1]
            ventes = df[df['Signal_RSI'] == -1]
            
            # Petite astuce : on décale légèrement les flèches (-5 / +5) pour qu'elles 
            # ne cachent pas la courbe violette
            if not achats.empty:
                ax.scatter(achats.index, achats['RSI'] - 5, marker='^', color='#2ecc71', s=120, zorder=5)
            if not ventes.empty:
                ax.scatter(ventes.index, ventes['RSI'] + 5, marker='v', color='#e74c3c', s=120, zorder=5)
        
        ax.set_ylabel("RSI", color="#f0f0f0", fontdict=self.police_axes)
        ax.set_ylim(0, 100)
        ax.legend(
            loc="upper left", 
            prop=self.police_legendes, 
            labelcolor="#f0f0f0", 
            facecolor='#181818',
            edgecolor='#181818',
            framealpha=1.0
        )
        ax.grid(True, color='#181818', linewidth=1.0)


# =================================================================================================
# ============================ DÉMARRAGE DE L'APPLICATION =========================================
# =================================================================================================

if __name__ == "__main__":
    
    app = QApplication(sys.argv)
    fenetre = ScreenerWindow()
    fenetre.show()
    sys.exit(app.exec())