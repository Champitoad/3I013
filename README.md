# 3I013

Projet d'initiation à la recherche, sujet "Environnement pour l'apprentissage de l'exploration visuelle d'une image" effectué dans le cadre de l'UE 3I013 de la L3 Informatique de l'UPMC.

## Installation

```
$ git submodule init
$ git submodule update
```

### Dépendances

```
# pip install gym tensorflow
```

Dépendances optionnelles pour stocker les résultats des expériences et générer des graphiques :

```
# pip install pandas matplotlib
```

## Utilisation

#### `train_subnet.py`

Module permettant d'entraîner les 10 prédicteurs (classe `agent.autoencoder.Predicter`) sur l'environnement `NumGrid` construit à partir du set d'entraînement de MNIST. Pour lancer l'entraînement :

```
python3 train_subnet.py
```

Chaque prédicteur est entraîné en parallèle dans un processus, et les modèles sont stockés dans le dossier `models`.

#### `interface.py`

Module permettant de tester les performances de l'agent sur l'environnement `NumGrid` construit à partir du set de test de MNIST. Les résultats expérimentaux sont stockés dans le dossier `results`.

#### `consts.py`

Paramètres communs à la phase d'apprentissage et de test de l'agent.

#### `plots.py`

Module pour générer des graphiques à partir des résultats expérimentaux.
