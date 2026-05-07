import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, QLineEdit, 
                             QPushButton, QVBoxLayout, QWidget, QStackedWidget, 
                             QHBoxLayout, QMessageBox, QSpacerItem, QSizePolicy)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

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
        super(DashboardCanvas, self).__init__(self.fig)
        self.fig.canvas.mpl_connect('scroll_event', self.zoom_molette)
        
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



# --- Fenêtre Principale ---
class ScreenerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Scamming Land Screener")
        self.setGeometry(100, 100, 1200, 750) # Fenêtre légèrement agrandie pour le multi-courbes
        
        # Style Global
        self.setStyleSheet("""
            QWidget { background-color: white; color: black; }
            QLineEdit { border: 1px solid #CCCCCC; border-radius: 4px; padding: 5px; }
            QPushButton { border: 1px solid black; border-radius: 4px; padding: 8px 15px; background-color: #F8F8F8; }
            QPushButton:hover { background-color: #E0E0E0; }
        """)

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        # Construction des pages
        self.creer_page_accueil()     # Index 0
        self.creer_page_dashboard()   # Index 1 (Dashboard unique avec 4 courbes)
        
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

        # Bouton lancer (Optionnel, au cas où l'utilisateur ne fait pas "Entrée")
        # layout_btn = QHBoxLayout()
        # btn_lancer = QPushButton("Lancer l'Analyse")
        # btn_lancer.clicked.connect(self.lancer_analyse)
        # layout_btn.addStretch()
        # layout_btn.addWidget(btn_lancer)
        # layout_btn.addStretch()
        # layout_principal.addSpacing(20)
        # layout_principal.addLayout(layout_btn)

        layout_principal.addStretch(1)
        self.stacked_widget.addWidget(page)



    # ================= PAGE 1 : DASHBOARD (4 COURBES) =================
    def creer_page_dashboard(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        # 1. RESSORT HAUT : Pousse tout le contenu vers le bas
        layout.addStretch(1)

        espaceur_haut = QSpacerItem(0, 24, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        layout.addSpacerItem(espaceur_haut)

        # Titre de la page 
        self.label_titre_dashboard = QLabel("Tableau de bord technique")
        self.label_titre_dashboard.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        self.label_titre_dashboard.setAlignment(Qt.AlignmentFlag.AlignHCenter) # Centré horizontalement
        layout.addWidget(self.label_titre_dashboard)

        # 2. ESPACE FIXE : Crée une respiration entre le titre et le graphique
        espaceur_dynamique = QSpacerItem(0, 12, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        layout.addSpacerItem(espaceur_dynamique)

        # Création du canvas multi-courbes
        self.canvas = DashboardCanvas(self, width=10, height=8, dpi=100)
        
        # 3. CENTRAGE HORIZONTAL DU CANVAS (Optionnel mais recommandé)
        # On l'enferme dans une boîte horizontale avec des ressorts sur les côtés
        layout_canvas = QHBoxLayout()
        layout_canvas.addStretch()
        layout_canvas.addWidget(self.canvas)
        layout_canvas.addStretch()
        layout.addLayout(layout_canvas)

        espaceur_dynamique = QSpacerItem(0, 12, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        layout.addSpacerItem(espaceur_dynamique)

        # Espace avant les boutons
        layout.addSpacing(12)

        # Bouton de retour
        layout_boutons = QHBoxLayout()
        btn_accueil = QPushButton("← Analyser un autre actif")
        btn_accueil.clicked.connect(self.retour_accueil)
        
        layout_boutons.addStretch()
        layout_boutons.addWidget(btn_accueil)
        layout_boutons.addStretch()

        layout.addLayout(layout_boutons)

        espaceur_dynamique = QSpacerItem(0, 24, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        layout.addSpacerItem(espaceur_dynamique)

        # 4. RESSORT BAS : Pousse tout le contenu vers le haut
        layout.addStretch(1)

        self.stacked_widget.addWidget(page)



    # ================= LOGIQUE GLOBALE =================
    def retour_accueil(self):
        self.input_ticker.clear()
        self.stacked_widget.setCurrentIndex(0)
        self.input_ticker.setFocus()

    def lancer_analyse(self):
        ticker = self.input_ticker.text().strip().upper()
        if not ticker: return

        # 1. Traitement des données via votre classe
        action = ActifFinancier(ticker)
        if not action.charger_donnees():
            QMessageBox.warning(self, "Erreur", f"Impossible de charger les données pour le ticker : {ticker}")
            return

        action.calculer_moyenne_mobile(fenetre=20)
        action.calculer_EMA(fenetre=20)
        action.calculer_rendements()
        action.calculer_volatilite_historique(fenetre=20)
        action.calculer_rsi(fenetre=14)
        action.calculer_macd()

        # On isole la dernière année de cotation (~252 jours)
        df = action.historique.copy().tail(252)

        # Mise à jour du titre
        self.label_titre_dashboard.setText(f"Tableau de bord technique : {ticker}")

        # 2. Répartition des graphiques dans le bon ordre (Prix -> Volatilité -> MACD -> RSI)
        ax_prix, ax_vol, ax_macd, ax_rsi = self.canvas.axes

        self.dessiner_prix(ax_prix, df, ticker)
        self.dessiner_volatilite(ax_vol, df, ticker)
        self.dessiner_macd(ax_macd, df, ticker)
        self.dessiner_rsi(ax_rsi, df, ticker)

        # On formatte l'axe des dates uniquement sur le graphique du bas (RSI)
        self.formater_axe_x(ax_rsi)

        # Ajustement des marges pour un rendu propre
        self.canvas.fig.tight_layout()
        self.canvas.draw()

        # 3. On affiche la page du Dashboard
        self.stacked_widget.setCurrentIndex(1)



    # ================= MÉTHODES DE DESSIN =================
    def formater_axe_x(self, ax):
        """Formatte l'axe des X pour afficher proprement les dates."""
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        for label in ax.get_xticklabels():
            label.set_rotation(45)

    def dessiner_prix(self, ax, df, ticker):
        ax.clear()
        ax.plot(df.index, df["Close"], label="Prix (Close)", color="black", linewidth=1.5)
        if "SMA_20" in df.columns:
            ax.plot(df.index, df["SMA_20"], label="SMA 20", color="blue", linestyle="--", alpha=0.7)
        if "EMA_20" in df.columns:
            ax.plot(df.index, df["EMA_20"], label="EMA 20", color="orange", linestyle="-.", alpha=0.7)
        ax.set_ylabel("Prix ($)")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

    def dessiner_volatilite(self, ax, df, ticker):
        ax.clear()
        if "Volatilite_20j" in df.columns:
            ax.plot(df.index, df["Volatilite_20j"], label="Volatilité (20j)", color="red")
        ax.set_ylabel("Volatilité")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

    def dessiner_macd(self, ax, df, ticker):
        ax.clear()
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
