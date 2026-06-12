# Système de trading semi-automatisé ICT/SMC — Documentation

## 1. Présentation

Ce projet est un système de trading **semi-automatisé** : le logiciel détecte
les configurations de marché selon la méthodologie ICT/SMC (Smart Money
Concepts), génère des signaux d'entrée, et **un humain confirme chaque
trade** avant toute exécution. Il se compose de deux applications reposant
sur un cœur commun :

- **Le bot de signaux** (à venir) : surveille le marché en continu, envoie
  une alerte Telegram quand une configuration apparaît, et attend la
  confirmation du trader (boutons ✅ Exécuter / ❌ Ignorer).
- **L'outil d'analyse** (livré) : un tableau de bord web pour visualiser les
  détections sur graphique, régler les paramètres de la stratégie et lancer
  des backtests.

**Principe fondamental :** le cœur fait tout le raisonnement (données,
détections, règles, backtest) ; les deux applications ne font que consommer
ses résultats. Aucune logique de trading ne vit dans les interfaces.

## 2. Avertissement important

Ce logiciel exécute **les règles du trader** avec discipline et constance.
Il ne garantit aucun profit. Le trading comporte un risque réel de perte en
capital ; la majorité des traders particuliers actifs perdent de l'argent.
Le rôle du backtester est précisément de vérifier, sur données historiques
et avec des hypothèses pessimistes, si une stratégie possède un avantage
**avant** de risquer de l'argent réel. Les résultats passés ne préjugent
pas des résultats futurs.

## 3. Installation

Prérequis : Python 3.10 ou plus récent.

```bash
cd trading-system
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## 4. Utilisation

### 4.1 Démo graphique (détections sur graphique)

```bash
python run_demo.py                        # données synthétiques, hors ligne
python run_demo.py --live EURUSD=X 15m    # données réelles (Yahoo Finance)
python run_demo.py --live BTC-EUR 1h      # exemple crypto en euros
```

Produit `chart.html` : chandeliers japonais avec les zones FVG (rectangles
verts/rouges), les étiquettes de structure (HH/HL/LH/LL), les cassures
BOS/CHoCH et les signaux de la stratégie (triangles ; survoler pour voir
entrée, stop-loss, take-profit et ratio risque/gain).

Format des symboles Yahoo : le forex prend le suffixe `=X` (`EURUSD=X`,
`GBPUSD=X`), la crypto s'écrit `BTC-USD`, `BTC-EUR`, `ETH-EUR`. Les unités
de temps disponibles : `5m`, `15m`, `1h`, `1d` (historique intrajournalier
limité à ~60 jours par Yahoo).

### 4.2 Backtest

```bash
python run_backtest.py                       # hors ligne
python run_backtest.py --live EURUSD=X 15m   # données réelles
```

Affiche les métriques dans le terminal et produit `backtest_report.html`
(courbe de capital, drawdown, résultat R de chaque trade) ainsi que
`trades.csv` (le journal complet des trades simulés).

### 4.3 Tableau de bord

```bash
streamlit run dashboard/app.py
```

S'ouvre dans le navigateur (http://localhost:8501). La barre latérale permet
de choisir la source de données, le symbole, l'unité de temps, et de régler
en direct les « boutons de réglage » de la stratégie. Trois onglets :
Graphique, Backtest, Signaux. Le tableau de bord est **en lecture seule** :
aucun bouton d'ordre, par conception.

### 4.4 Tests automatisés

```bash
python -m pytest tests/ -v
```

## 5. Les détecteurs

### 5.1 Fair Value Gap (FVG)

Motif à 3 bougies : il y a un FVG haussier lorsque le plus bas de la bougie
3 reste au-dessus du plus haut de la bougie 1 (et symétriquement pour le
baissier). La zone du gap est délimitée par ces deux prix. Paramètre
réglable `min_displacement_atr` : la bougie centrale (l'impulsion) doit
avoir un corps d'au moins X fois l'ATR — cela filtre les micro-gaps dus au
bruit. Le système suit chaque zone dans le temps : **mitigée** au premier
retour du prix dans la zone, **comblée** quand le prix la traverse
entièrement.

### 5.2 Structure de marché

Détection des points pivots (swing highs/lows) par fenêtre glissante
(paramètre `lookback` : plus il est grand, plus la structure est filtrée),
étiquetage HH/HL/LH/LL, puis détection des cassures : **BOS** (continuation
de tendance) et **CHoCH** (changement de caractère, retournement potentiel),
validées à la clôture de bougie.

Point essentiel : un point pivot n'est **connaissable** que `lookback`
bougies après s'être formé. Le système n'utilise jamais une information
avant la date où elle était réellement disponible (aucun « regard vers le
futur »). Cette règle est verrouillée par des tests automatisés.

### 5.3 Stratégie d'exemple

`FVGRetestStrategy` (échafaudage, **pas encore les règles du client**) :
CHoCH + FVG dans le même sens + retour du prix dans la zone ⇒ signal avec
entrée au bord de la zone, stop au-delà, objectif à ratio fixe. Tous les
seuils sont des paramètres, sans valeurs magiques dans le code.

## 6. Le backtester

Hypothèses volontairement **pessimistes** — un backtest doit sous-promettre :

- Un signal devient un ordre limite actif **à partir de la bougie suivante**
  uniquement ; l'ordre expire s'il n'est pas exécuté sous N bougies.
- Spread complet + slippage facturés à l'entrée.
- Si une même bougie touche le stop ET l'objectif, le trade est compté
  **perdant** (l'ordre des événements à l'intérieur d'une bougie est
  inconnaissable en OHLC).
- Une seule position à la fois (cohérent avec la confirmation humaine).
- Taille de position : % fixe du capital courant risqué par trade
  (capitalisation composée).

Métriques produites : nombre de trades, taux de réussite, profit factor,
espérance en R, rendement net, drawdown maximal, ordres expirés, signaux
ignorés (position déjà ouverte).

## 7. Qualité et tests

16 tests automatisés sur des bougies construites à la main dont la bonne
réponse est connue d'avance : zones FVG aux prix exacts, étiquettes de
structure sur les bonnes bougies, interdiction du regard vers le futur,
P&L du backtester exact au centime, et un test de démarrage complet du
tableau de bord. Le déterminisme est aussi testé : mêmes bougies en entrée
⇒ signaux identiques en sortie, à chaque exécution.

## 8. Limites actuelles (assumées)

- Données Yahoo : pas de vrai spread bid/ask, ~60 jours d'historique
  intrajournalier — suffisant pour la validation visuelle, pas pour un
  backtest sérieux (un flux de meilleure qualité sera branché ensuite ;
  l'architecture le permet en changeant une seule classe).
- La stratégie d'exemple est bruyante : les filtres de confluence (kill
  zones, biais de l'unité de temps supérieure, prises de liquidité)
  arriveront avec le cahier des charges du trader.
- Pas encore de bot Telegram ni d'exécution broker.

## 9. Feuille de route

1. **Bot Telegram** : planificateur ⇒ stratégies ⇒ alerte avec boutons
   ✅ / ❌, expiration des signaux, journalisation complète en base.
2. **Règles réelles du client** : nouvelle classe de stratégie issue de la
   spécification validée avec lui (session de validation sur graphique).
3. **Module d'exécution** broker/exchange — en dernier, d'abord en compte
   de démonstration (paper trading), puis capital réel réduit.

## 10. Dépannage

- `No data returned` : format de symbole incorrect (le forex exige `=X`) ou
  limite d'historique atteinte — essayer `1h` ou `1d`.
- `ModuleNotFoundError: core` : lancer les commandes depuis la racine du
  projet (le dossier contenant `run_demo.py`).
- Graphique vide dans le navigateur : la page charge plotly.js depuis un
  CDN, une connexion internet est nécessaire à la première ouverture.
- Données périmées : supprimer le dossier `data_cache/`.
