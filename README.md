# Project Description

## Définitions 

**Option**
Les options sont des produits financiers dits « dérivés » donnant le droit d'acheter ou de vendre une quantité d'actifs sous-jacents (actions, devises, etc.) pendant une période et à un prix convenus à l'avance. Vous payez une prime (prix de l'option) pour acquérir ce droit. 

**Close du sous-jacent**  
Le "close" du sous-jacent fait référence au prix de clôture de l'actif sous-jacent à la fin de la journée de trading. C'est le dernier prix auquel le sous-jacent (par exemple une action ou un indice) a été échangé avant la fermeture du marché.

**Volatilité**  
La volatilité mesure l'ampleur des fluctuations du prix d'un actif sur une période donnée.

**Strike Price (Prix d'exercice)**  
Le prix d'exercice est le prix prédéterminé auquel le détenteur d'une option peut acheter (call) ou vendre (put) le sous-jacent lorsqu'il exerce son option. Par exemple si on a un call avec un strike de 100 €, le détenteur peut acheter le sous-jacent à 100 €, peu importe son prix actuel.
  
**Bid**  
Le "bid" est le prix le plus élevé qu'un acheteur est prêt à payer pour une option ou un actif. 

**Ask**  
Le "ask" est le prix le plus bas auquel un vendeur est prêt à céder une option ou un actif.

**Spread**  
Le spread est la différence entre le bid et l'ask :  
Il reflète la liquidité et les coûts de transaction sur le marché. Un spread large indique souvent une faible liquidité.

**Performance du sous-jacent**  
La performance du sous-jacent désigne le pourcentage de variation de son prix sur une période donnée :  
Elle mesure le rendement ou la perte liée à l'évolution du sous-jacent.

**Moyenne X jours du sous-jacent**  
La moyenne X jours correspond à la moyenne arithmétique des prix du sous-jacent sur les X derniers jours. 
C'est un indicateur de tendance utilisé pour lisser les variations et identifier des patterns.

## Le modèle de Black Scholes 

Le modèle de Black-Scholes, permet de calculer le prix théorique d'une option européenne (qui ne peut être exercée qu'à l'échéance) sur un actif sous-jacent (comme une action).
C'est un outil mathématique utilisé principalement pour évaluer le prix des options financières. 

Hypothèses :
1. Le prix de l'actif sous-jacent suit un mouvement brownien géométrique, avec une volatilité constante.
2. Le marché est liquide, sans coûts de transaction ni taxes.
3. Il n'y a pas d'arbitrage possible (gains sans risque).
4. Le taux d'intérêt sans risque est constant.
5. Les dividendes sont nuls (ou constants dans les extensions du modèle).

 Formule de base pour une option d'achat (call) :
Le prix d'un call \( C \) est donné par :

![Equation](https://latex.codecogs.com/svg.image?%20C=S_0%5Ccdot%20N(d_1)-K%5Ccdot%20e%5E%7B-rT%7D%5Ccdot%20N(d_2))

Où :
- ![Equation](https://latex.codecogs.com/svg.image?S_0) : prix actuel de l'actif sous-jacent
- ![Equation](https://latex.codecogs.com/svg.image?K) : prix d'exercice de l'option
- ![Equation](https://latex.codecogs.com/svg.image?T) : temps jusqu'à l'échéance (en années)
- ![Equation](https://latex.codecogs.com/svg.image?r) : taux d'intérêt sans risque
- ![Equation](https://latex.codecogs.com/svg.image?%5Csigma) : volatilité du prix de l'actif sous-jacent
- ![Equation](https://latex.codecogs.com/svg.image?N(x)) : fonction de répartition de la loi normale standard (probabilité cumulée)
- ![Equation](https://latex.codecogs.com/svg.image?d_1) et ![Equation](https://latex.codecogs.com/svg.image?d_2) sont calculés comme suit :
  -   ![Equation](https://latex.codecogs.com/svg.image?d_1=%5Cfrac%7B%5Cln(S_0/K)+(r+%5Csigma%5E2/2)T%7D%7B%5Csigma%5Csqrt%7BT%7D)
  -   ![Equation](https://latex.codecogs.com/svg.image?d_2=d_1-%5Csigma%5Csqrt%7BT%7D)

 Intuition derrière la formule :
- ![Equation](https://latex.codecogs.com/svg.image?N(d_1)) représente la probabilité ajustée au risque que l'option soit exercée.
- ![Equation](https://latex.codecogs.com/svg.image?N(d_2)) pondère la valeur actualisée du prix d'exercice.
- La première partie ![Equation](https://latex.codecogs.com/svg.image?%20S_0%5Ccdot%20N(d_1))représente la probabilité que l'option soit exercée à l'échéance, en fonction du prix actuel de l'action, de la volatilité et du temps restant. Plus cette probabilité est élevée, plus l'option vaut cher.
- La seconde partie ![Equation](https://latex.codecogs.com/svg.image?-K%5Ccdot%20e%5E%7B-rT%7D%5Ccdot%20N(d_2))correspond à la valeur actualisée du paiement à l'échéance de l'option, en tenant compte du taux d'intérêt sans risque et du prix d'exercice. Elle reflète le coût d'opportunité de détenir l'option plutôt qu'un investissement sans risque.

## Problématique

Comment prédire au mieux le prix d'une option? 

## Project Scope

Dans ce projet nous cherchons à prédire le prix d'une option call en fonction de données macroéconomiques et financières. La première étape est de constituer un fichier CSV agrégeant toutes les données nécessaires à la prédiction du prix d'une option. Après avoir visualisé ces différentes séries temporelles et étudier leurs caractéristiques avec le modèle ARIMA, nous avons cherché à les prédire à l'aide de 2 modèles : une méthode de boosting par gradient (XGBoost et LightGBM) et des réseaux de neuronnes récurrents (LSTM) pour répondre à la problématique et comparer avec le modèle théorique de Black Sholes.   

## Nos données 

Pour répondre à notre problématique, nous avons du récupérer plusieurs catégories de données issues de différentes sources : 

## Les données des sous-jacent et des calls

Nous avons utilisé la bibliothèque Yahoo Finance, yfinance, pour accéder aux données. Cependant, l'historique des prix des options d'achat (calls) ne remonte pas à plus de deux ans. Nous avons donc dû traiter un problème de série temporelle avec un volume limité de données historiques. Pour remédier à cette contrainte, deux approches étaient envisageables :

1 : Prendre l'historique de tous les calls accessibles sur la librairie d'une même entreprise, ce qui nous permettait d'avoir plusieurs séries temporelles : pour les N dates de maturités il y a n séries temporelles différentes avec seulement le strike price et le prix du call qui varient pour ces n séries temporelles. 

2 : Prendre plusieurs entreprises et ne choisir qu'un historique de calls pour chaque entreprise, ce qui nous donnait donc x séries temporelles avec pour chaque série temporelle des aractéristiques différentes.

Nous avons opté pour l'option 2, car l'option 1 risquait d’introduire une compensation artificielle au manque de données. En effet, les séries temporelles issues d'une même entreprise, avec des dates d'échéance identiques mais des prix d'exercice différents, sont susceptibles d'évoluer de manière très similaire. Il est peu probable que leurs trajectoires soient complètement décorrélées.

L'option 2, bien qu'imparfaite, permet de limiter ce biais. Néanmoins les entreprises évoluant dans un contexte économique similaire on peut supposer que le cours des actions entre les différentes entreprises n'est pas complétement décorrélé. L'idéal aurait été de disposer d'une seule série temporelle sur une période bien plus longue. Cependant, nous n'avons pas trouvé d'alternative satisfaisante compte tenu des limitations d'accès aux données historiques.

Afin de maximiser notre nombre de données, il a donc fallu regarder pour chaque entreprise pour quel call nous avions l'hitsorique le plus long et le choisir lui. 

La bibliothèque yahoofinance ne fonctionnanant pas toujours après 2 ou 3 récupérations de données (il faut relancer le kernel sinon il y a des erreurs)  le script Notebook_v5_VR qui génère automatiquement la table finale ne fonctionne pas toujours puisque nous avons étudié 7 entreprises.

Nous avons donc du générer les tables de manière individuelle pour chaque entreprise ce qui prend 15 minutes/entreprise ( avec le script :  script_data/call_{entreprise}_v2 ) qui génère un fichier csv stocké dans le SSPcloud et accessible via "FILE_PATH_S3 = f"{MY_BUCKET}/result/data_{ticker}_filtre2.csv" comme précisé dans les scripts ). Le script "script_dataframe_final_v2.ipybn" permet à partir de ces fichiers dans SSPcloud de générer la table finale.

## Les données macroéconomiques

Pour enrichir notre analyse et répondre efficacement à notre problématique, nous avons intégré plusieurs variables macroéconomiques.

Nous avons utilisé la bibliothèque yfinance pour récupérer divers indices boursiers ainsi que l'indicateur de volatilité du marché américain (VIX).
Par ailleurs, nous avons aussi exploité l'API de la Banque Fédérale de Saint Louis (Missouri, USA) pour accéder à des données macroéconomiques complémentaires, notamment :
- Les taux sans risque de différents pays
- Le taux de chômage américain
- Le PIB des États-Unis
- Un indice représentant l'inflation (IPC)

Cette combinaison de données financières et macroéconomiques nous a permis d'approfondir notre analyse tout en garantissant une vue d'ensemble cohérente sur les différents facteurs influençant notre sujet.

## Project Structure

```
.
├── README.md
│ 
├── 04_modele_gradient_boosting_visualisation  #contient les scripts qui executent notre modèle et la visualisation pour comparer nos résultats aux valeurs réelles
│   ├──   prediction_v2.py  #script qui fait tourner un modèle light gbm et xgboost et renvoie le dataframe de script_data_frame_final_v2 en rajoutant ces predictions en colonne
    └──   script_data_frame_final_v2.ipynb #script qui appelle les données de script_data dans le ssp_cloud ( dans result ) --> Renvoie un data frame nommé data_v2 sur SSP Cloud
    └──   visualisation_finale_v2.ipynb  #appelle prediction_v2 et affiche les predicitons xgboost et lightgbm sur le port 5050
│ 
├── script_data
        └── call_aapl.ipynb  #fichier python qui récupère l'historique du call avec le plus d'historique et les données du sous jacent de l'entreprise en question --> renvoie un dataframe data_entreprise_filtre2.csv sur SSP Cloud. On a affiché dans ce script le graphe qui montre la réprtition de nombre d'historique que l'on a pour chaque call
        │
.       .
.       .
.       .
        │
        └── call_tsla.py
│       
├── 01_script_data.ipynb #Script qui permet d'automatiser la création du dataframe prend plus d'une heure à tourner (script commenté) avec toutes les entreprises en même temps (cf readme/Nos données ) --> renvoie un           dataframe data_valv2 dans SSP Cloud
│   Nous avons affiché à la fin de ce script les prévisions attendues par le modèle de Black Sholes en les superposant aux données réelles.     
│   
└── 02_analyse_data.ipynb #Note book qui appelle data_valv2 et qui analyse nos données dont certaines sur les call et sous jacent avec dash (sur les ports 5050,5051 et 5052 ) afin de faire une visualisation dynamique 
│   
└── 03_modele_arima_analyse_serie.ipynb #Note book qui fait l'étude classique des séries temporelles ( en prenant l'exemple d'une entreprise ) en regardant la stationnarité, la saisonnalité,... et qui explique pourquoi nous n'avons pas opté pour ce choix lors de nos modèles
│   
└── 05_LSTM.ipynb  #Fichier qui fais tourner un réseau de neurones récurrents pour prédire le prix du call et qui affiche les résultats pour une entreprise (AAPL, nous n'avons pas fait de visualisation dynamique comme précisé dans le notebook car il n'est pas opérationnel et nous n'avons pas encore trouvé pourquoi.
│   
└── visualisation_dynamique_local : dossier qui contient tous les éléments pour faire tourner nos visualisations dash en local si jamais ça ne fonctionne pas sur SSP Cloud ( cf Running the Project ) 


```
