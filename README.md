# Projet RANZCR CLip - Catheter and Line Position Challenge
### Guillaume Picart, Samy Dougui, Guillaume Herry    
<br> 

Lien vers le challenge: https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification/overview <br>
Lien vers le OneDrive contenant les données, la vidéo, ainsi que nos meilleurs modèles sous format pth: https://centralesupelec-my.sharepoint.com/:f:/g/personal/guillaume_herry_student-cs_fr/EsxX2WNX-cBGgPHV2-fmJVYBjDSu6oAAqpx4BZqTOOngKA?e=3mmDN1
<br><br>

**Description du repository:** <br>
- *Dossier EDA*: <br>
    - **exploration_des_données.ipynb** : Notebook contenant les scripts ayant permis de réaliser l'exploration des données dans notre présentation (graphes et annotations).

- *Dossier models*: <br>
    
    - **main.py** : Fichier principal<br>
    - **models.py** : Il contient les classes des modèles que nous avons utilisés <br>
    - **utils.py** : Il contient toutes les fonctions que nous utilisons
    - **Testing.ipynb** : Exemple pour tester les modèles que nous avons entrainé
    - **Training.ipynb** : Exemple pour entrainer un des modèles
    
    - Exemples d'utilisation:
        
        - Assurez-vous que les librairies inclues dans le *requirements.txt* sont installées

        - Pour lancer l'entrainement du modèle resnet200d en affichant les logs ```python3 main.py --model resnet --mode train --verbose```

        - Pour tester le modèle EfficientNetB2 sans afficher les logs ```python3 main.py --model efficientnet --mode test```

        - Pour visualiser l'ensemble des arguments possibles ```python3 main.py -h```
        


**Data:** <br>
Les données étant trés volumineuses (+ de 10 Go), nous ne pouvons pas les importer sur notre repository, vous pouvez y accéder de deux manières: <br>
- Directement via le challenge Kaggle: https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification/data (bouton *download all*)
- Via le Onedrive que nous avons créé et qui contient l'ensemble des données (lien à la deuxième ligne du ReadMe).

Afin de tester notre code avec les données, il faut donc:
- *exploration_des_données.ipynb*: Compléter la variable "BASE_DIR" (seconde cellule), en y mettant le PATH menant aux données (train.csv, train_annotations.csv et les dossiers contenant les jpg) <br>
- *Scripts Python*: Préciser dans la fonction ```get_config()``` le chemin vers les données<br>


<br><br>
**Classement final du challenge:** <br> 1230 ème / 1500 <br>
Nous obtenons un score de 92.5%, le meilleur score étant à 97%. Le classement nous semble biaisé car des modèles autour de 96,5% étaient publics, et donc recopiés par beaucoup d'équipes.

<br><br>
**Vidéo de présentation:** <br>
Vous pouvez y accéder via le OneDrive que nous avons créé (lien à la seconde ligne du ReadMe).





