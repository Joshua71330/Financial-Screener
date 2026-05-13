import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, QLineEdit, 
                             QPushButton, QVBoxLayout, QWidget, QStackedWidget, 
                             QHBoxLayout, QMessageBox, QSpacerItem, QSizePolicy)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# --- Importation de tes modules ---
from gestion_donnees import telecharger_historique
from modeles import ActifFinancier


# =====================================================================
# CANEVAS MULTI-GRAPHIQUES (Avec Bougies et Survol)
# =====================================================================
class DashboardCanvas(FigureCanvas):
    def __init__(self, parent=None, width=10, height=8, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.subplots(4, 1, sharex=True, gridspec_kw={'height_ratios': [2, 1, 1, 1]})
        self.fig.subplots_adjust(hspace=0.1, bottom=0.1) 
        super(DashboardCanvas, self).__init__(self.fig)
        
        self.fig.canvas.mpl_connect('scroll_event', self.zoom_molette)
        self.fig.canvas.mpl_connect('button_press_event', self.clic_presse)
        self.fig.canvas.mpl_connect('button_release_event', self.clic_relache)
        self.fig.canvas.mpl_connect('motion_notify_event', self.mouvement_souris)
        
        self.pan_axes = None
        self.press_x = None; self.press_y = None
        self.df_courant = None 
        
        # Tooltip flottant
        self.tooltip = self.fig.text(0.0, 0.0, "", va="bottom", ha="left",
                                     fontsize=8, 
                                     bbox=dict(boxstyle="round,pad=0.2", fc="#f8f9fa", ec="#cccccc", alpha=0.9),
                                     zorder=100, visible=False)

    def zoom_molette(self, event):
        if event.inaxes is None: return
        ax = event.inaxes
        facteur_base = 1.2
        facteur = 1 / facteur_base if event.step > 0 else facteur_base
        x_min, x_max = ax.get_xlim()
        x_souris = event.xdata
        nouvelle_largeur = (x_max - x_min) * facteur
        position_relative = (x_souris - x_min) / (x_max - x_min)
        ax.set_xlim([x_souris - nouvelle_largeur * position_relative, x_souris + nouvelle_largeur * (1 - position_relative)])
        self.fig.canvas.draw_idle()

    def clic_presse(self, event):
        if event.button == 1 and event.inaxes is not None:
            self.pan_axes = event.inaxes
            inv = self.pan_axes.transData.inverted()
            self.press_x, self.press_y = inv.transform((event.x, event.y))

    def clic_relache(self, event):
        if event.button == 1: self.pan_axes = None

    def mouvement_souris(self, event):
        if self.pan_axes is not None:
            if event.x is None or event.y is None: return
            inv = self.pan_axes.transData.inverted()
            x_data, y_data = inv.transform((event.x, event.y))
            dx = x_data - self.press_x
            dy = y_data - self.press_y
            xlim = self.pan_axes.get_xlim(); ylim = self.pan_axes.get_ylim()
            self.pan_axes.set_xlim(xlim[0] - dx, xlim[1] - dx)
            self.pan_axes.set_ylim(ylim[0] - dy, ylim[1] - dy)
            if self.tooltip.get_visible(): self.tooltip.set_visible(False)
            self.fig.canvas.draw_idle()
            return

        if event.inaxes is None or self.df_courant is None:
            if self.tooltip.get_visible():
                self.tooltip.set_visible(False)
                self.fig.canvas.draw_idle()
            return

        ax = event.inaxes
        try:
            date_souris = pd.to_datetime(mdates.num2date(event.xdata)).tz_localize(None)
            index_propre = self.df_courant.index.tz_localize(None)
            idx = index_propre.get_indexer([date_souris], method='nearest')[0]
            row = self.df_courant.iloc[idx]
            date_reelle = self.df_courant.index[idx].strftime('%d %b %Y')
            index_graphique = self.axes.tolist().index(ax)

            lignes = [f"{date_reelle}"]
            
            if index_graphique == 0:
                if all(col in row for col in ['Open', 'High', 'Low', 'Close']):
                    lignes.append(f"O: {row['Open']:.2f} | H: {row['High']:.2f}")
                    lignes.append(f"L: {row['Low']:.2f} | C: {row['Close']:.2f}")
                else:
                    lignes.append(f"Prix : {row['Close']:.2f} $")
                if "SMA_20" in row: lignes.append(f"SMA 20 : {row['SMA_20']:.2f}")
            elif index_graphique == 1:
                if "Volatilite_20j" in row: lignes.append(f"Volatilité : {row['Volatilite_20j']:.4f}")
            elif index_graphique == 2:
                if "MACD" in row: lignes.append(f"MACD : {row['MACD']:.2f}")
            elif index_graphique == 3:
                if "RSI" in row: lignes.append(f"RSI : {row['RSI']:.2f}")

            self.tooltip.set_text("\n".join(lignes))
            fig_w, fig_h = self.fig.get_size_inches() * self.fig.dpi
            pos_x, pos_y = (event.x + 15) / fig_w, (event.y + 15) / fig_h
            if pos_x > 0.8: pos_x = (event.x - 120) / fig_w
            if pos_y > 0.8: pos_y = (event.y - 80) / fig_h
            self.tooltip.set_position((pos_x, pos_y))
            self.tooltip.set_visible(True)
            self.fig.canvas.draw_idle()
        except:
            pass


# =====================================================================
# FENÊTRE PRINCIPALE
# =====================================================================
class ScreenerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Screener IA")
        self.setGeometry(50, 50, 1300, 900)
        
        self.setStyleSheet("""
            QWidget { background-color: white; color: black; }
            QLineEdit { border: 2px solid #007BFF; border-radius: 5px; padding: 8px; }
            QPushButton { border: 1px solid #007BFF; border-radius: 5px; padding: 8px 15px; background-color: #F0F8FF; font-weight: bold;}
            QPushButton:hover { background-color: #D0E8FF; }
        """)

        self.df_complet = None
        self.ticker_courant = ""

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        self.creer_page_accueil()
        self.creer_page_dashboard()
        self.stacked_widget.setCurrentIndex(0)


    # ─── PAGE 1 : ACCUEIL ────────────────────────────────────────────
    def creer_page_accueil(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addStretch()

        titre = QLabel("SCREENER TECHNIQUE & IA")
        titre.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        titre.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(titre)

        self.input_ticker = QLineEdit()
        self.input_ticker.setFont(QFont("Arial", 14))
        self.input_ticker.setPlaceholderText("Ex: AAPL, TSLA, MSFT...")
        self.input_ticker.setFixedWidth(300)
        self.input_ticker.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.input_ticker.returnPressed.connect(self.lancer_analyse)

        boite = QHBoxLayout()
        boite.addStretch()
        boite.addWidget(self.input_ticker)
        boite.addStretch()
        layout.addLayout(boite)

        layout.addStretch()
        self.stacked_widget.addWidget(page)


    # ─── PAGE 2 : DASHBOARD ──────────────────────────────────────────
    def creer_page_dashboard(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        # En-tête : Titre et Prédiction IA
        layout_header = QHBoxLayout()
        self.label_titre = QLabel("Tableau de bord")
        self.label_titre.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        
        self.label_ia = QLabel("Prédiction IA : En attente...")
        self.label_ia.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.label_ia.setStyleSheet("color: #333; background-color: #EEE; padding: 5px; border-radius: 5px;")
        
        layout_header.addWidget(self.label_titre)
        layout_header.addStretch()
        layout_header.addWidget(self.label_ia)
        layout.addLayout(layout_header)

        # Boutons d'échelle de temps
        layout_echelles = QHBoxLayout()
        layout_echelles.addWidget(QLabel("Échelle de temps : "))
        
        echelles = [("1 Mois", 21), ("3 Mois", 63), ("6 Mois", 126), ("1 An", 252), ("2 Ans", 504), ("Max", None)]
        for texte, jours in echelles:
            btn = QPushButton(texte)
            # Utilisation de lambda pour passer le paramètre 'jours' à la fonction
            btn.clicked.connect(lambda checked, j=jours: self.changer_echelle(j))
            layout_echelles.addWidget(btn)
        
        layout_echelles.addStretch()
        layout.addLayout(layout_echelles)

        # Graphiques
        self.canvas = DashboardCanvas(self, width=12, height=7, dpi=100)
        layout.addWidget(self.canvas)

        # Zone inférieure : Bouton retour
        layout_bas = QHBoxLayout()
        layout_bas.addStretch()
        
        btn_retour = QPushButton("← Nouvelle Analyse")
        btn_retour.clicked.connect(self.retour_accueil)
        btn_retour.setMinimumHeight(40)
        btn_retour.setMinimumWidth(200)
        layout_bas.addWidget(btn_retour)

        layout_bas.addStretch()
        layout.addLayout(layout_bas)
        self.stacked_widget.addWidget(page)


    # ─── LOGIQUE MÉTIER ──────────────────────────────────────────────
    def retour_accueil(self):
        self.input_ticker.clear()
        self.stacked_widget.setCurrentIndex(0)
        self.input_ticker.setFocus()

    def lancer_analyse(self):
        ticker = self.input_ticker.text().strip().upper()
        if not ticker: return
        self.ticker_courant = ticker

        # 1. Téléchargement et chargement (via gestion_donnees)
        telecharger_historique(ticker)
        actif = ActifFinancier(ticker)
        if not actif.charger_donnees():
            QMessageBox.warning(self, "Erreur", f"Données introuvables pour {ticker}.")
            return

        # 2. Calcul des indicateurs (modeles.py)
        actif.calculer_moyenne_mobile(fenetre=20)
        actif.calculer_moyenne_mobile(fenetre=50)
        actif.calculer_EMA(fenetre=20)
        actif.calculer_rendements()
        actif.calculer_volatilite_historique(fenetre=20)
        actif.calculer_rsi(fenetre=14)
        actif.calculer_macd()
        actif.calculer_volume_zscore(fenetre=20)

        self.df_complet = actif.historique.copy()

        # 3. Entraînement IA
        self.label_ia.setText("IA en cours d'entraînement...")
        self.label_ia.setStyleSheet("color: orange; background-color: #EEE; padding: 5px; border-radius: 5px;")
        QApplication.processEvents() # Force la mise à jour de l'affichage
        
        actif.trouver_meilleur_alpha()
        actif.entrainer_IA()
        pred = actif.predire_demain()

        if pred is not None:
            if pred > 0:
                self.label_ia.setText("Prédiction IA Demain : 📈 HAUSSE")
                self.label_ia.setStyleSheet("color: white; background-color: #28a745; padding: 5px; border-radius: 5px;")
            else:
                self.label_ia.setText("Prédiction IA Demain : 📉 BAISSE")
                self.label_ia.setStyleSheet("color: white; background-color: #dc3545; padding: 5px; border-radius: 5px;")

        # 4. Affichage final (Par défaut sur 1 An = 252 jours)
        self.label_titre.setText(f"Analyse Technique : {ticker}")
        self.changer_echelle(252) 
        self.stacked_widget.setCurrentIndex(1)


    # ─── GESTION DE L'ÉCHELLE DE TEMPS ───────────────────────────────
    def changer_echelle(self, jours):
        if self.df_complet is None or self.df_complet.empty: return
        
        if jours is None: # Cas "Max"
            df = self.df_complet.copy()
        else:
            df = self.df_complet.tail(jours).copy()

        self.canvas.df_courant = df
        ax_prix, ax_vol, ax_macd, ax_rsi = self.canvas.axes

        self.dessiner_prix(ax_prix, df)
        self.dessiner_volatilite(ax_vol, df)
        self.dessiner_macd(ax_macd, df)
        self.dessiner_rsi(ax_rsi, df)

        self.formater_axe_x(ax_rsi)
        self.canvas.fig.tight_layout()
        self.canvas.draw()


    # ─── MÉTHODES DE DESSIN ──────────────────────────────────────────
    def formater_axe_x(self, ax):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
        # Limite le nombre d'étiquettes de date pour éviter que ça se chevauche
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        for label in ax.get_xticklabels():
            label.set_rotation(30)

    def dessiner_prix(self, ax, df):
        ax.clear()
        if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            jours_hausse = df[df['Close'] >= df['Open']]
            jours_baisse = df[df['Close'] < df['Open']]
            largeur_corps = 0.8
            
            # Bougies vertes
            ax.vlines(jours_hausse.index, jours_hausse['Low'], jours_hausse['High'], color='green', linewidth=1)
            ax.bar(jours_hausse.index, jours_hausse['Close'] - jours_hausse['Open'], 
                   bottom=jours_hausse['Open'], color='green', width=largeur_corps)
            
            # Bougies rouges
            ax.vlines(jours_baisse.index, jours_baisse['Low'], jours_baisse['High'], color='red', linewidth=1)
            ax.bar(jours_baisse.index, jours_baisse['Open'] - jours_baisse['Close'], 
                   bottom=jours_baisse['Close'], color='red', width=largeur_corps)
        else:
            ax.plot(df.index, df["Close"], color="black")

        if "SMA_20" in df.columns: ax.plot(df.index, df["SMA_20"], label="SMA 20", color="blue", alpha=0.5)
        if "EMA_20" in df.columns: ax.plot(df.index, df["EMA_20"], label="EMA 20", color="orange", alpha=0.5)
        ax.set_ylabel("Prix ($)")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

    def dessiner_volatilite(self, ax, df):
        ax.clear()
        if "Volatilite_20j" in df.columns:
            ax.plot(df.index, df["Volatilite_20j"], color="brown")
        ax.set_ylabel("Volatilité")
        ax.grid(True, alpha=0.3)

    def dessiner_macd(self, ax, df):
        ax.clear()
        if "MACD" in df.columns and "MACD_signal" in df.columns:
            ax.plot(df.index, df["MACD"], label="MACD", color="blue", linewidth=1)
            ax.plot(df.index, df["MACD_signal"], label="Signal", color="orange", linewidth=1)
            if "MACD_hist" in df.columns:
                couleurs = ['green' if val >= 0 else 'red' for val in df["MACD_hist"]]
                ax.bar(df.index, df["MACD_hist"], color=couleurs, alpha=0.5)
        ax.set_ylabel("MACD")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

    def dessiner_rsi(self, ax, df):
        ax.clear()
        if "RSI" in df.columns:
            ax.plot(df.index, df["RSI"], color="purple")
            ax.axhline(70, color='red', linestyle='--', alpha=0.5)
            ax.axhline(30, color='green', linestyle='--', alpha=0.5)
            ax.fill_between(df.index, y1=30, y2=70, color='purple', alpha=0.05)
        ax.set_ylabel("RSI")
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    fenetre = ScreenerWindow()
    fenetre.show()
    sys.exit(app.exec())
    
