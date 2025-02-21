import pickle
from nltk.tree import Tree

# Remplace 'chemin/vers/ton_fichier.pkl' par le chemin de ton fichier pickle
pkl_file = r'ward\cifar.rebuild_hierarchy.pkl'


with open(pkl_file, 'rb') as f:
    hierarchy = pickle.load(f)

print("Type de l'objet chargé :", type(hierarchy))

# Si c'est un arbre NLTK, tu peux afficher sa structure de manière plus lisible :
if isinstance(hierarchy, Tree):
    print("\nAffichage de l'arbre (hiérarchie) :")
    hierarchy.pretty_print()
    
    # Pour obtenir la liste des feuilles (les classes finales)
    print("\nListe des feuilles :", hierarchy.leaves())
else:
    # Sinon, affiche simplement l'objet
    print(hierarchy)
