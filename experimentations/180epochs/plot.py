import os
import json
import numpy as np
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def load_epoch_results(folder):
    """Charge les données d'entraînement depuis all_epoch_results.json dans le dossier donné."""
    path = os.path.join(folder, 'all_epoch_results.json')
    with open(path, 'r') as f:
        data = json.load(f)
    epochs     = [d['epoch'] for d in data]
    train_loss = [d['train_loss'] for d in data]
    val_loss   = [d['val_loss'] for d in data]
    train_acc  = [d['train_accuracy'] for d in data]
    val_acc    = [d['val_accuracy'] for d in data]
    return epochs, train_loss, val_loss, train_acc, val_acc

def load_final_test_results(folder):
    """Charge les données de test final depuis final_test_results.json dans le dossier donné."""
    path = os.path.join(folder, 'final_test_results.json')
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def create_figure(models_epoch, models_test, colors, model_names):
    """
    Crée une figure matplotlib contenant :
    - Un graphique de la loss (train et validation)
    - Un graphique de l'accuracy (train et validation)
    - Un graphique en barres pour les distances hiérarchiques moyennes (test final)
    """
    fig = Figure(figsize=(10, 10))
    
    # Création des sous-graphiques (3 lignes, 1 colonne)
    ax_loss = fig.add_subplot(311)
    ax_acc  = fig.add_subplot(312)
    ax_bar  = fig.add_subplot(313)
    
    # Dictionnaire : index du modèle -> liste d'éléments graphiques
    model_artists = {i: [] for i in range(len(models_epoch))}
    
    # --- Graphique 1 : Loss ---
    for i, data in enumerate(models_epoch):
        epochs, train_loss, val_loss, train_acc, val_acc = data
        col = colors[i]
        l_train, = ax_loss.plot(epochs, train_loss,
                                label=f'{model_names[i]} - Train Loss',
                                color=col, linestyle='-')
        l_val, = ax_loss.plot(epochs, val_loss,
                              label=f'{model_names[i]} - Val Loss',
                              color=col, linestyle='--')
        model_artists[i].extend([l_train, l_val])
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_title('Loss (Train et Validation) par Epoch')
    ax_loss.legend(loc='upper left')
    
    # --- Graphique 2 : Accuracy ---
    for i, data in enumerate(models_epoch):
        epochs, train_loss, val_loss, train_acc, val_acc = data
        col = colors[i]
        a_train, = ax_acc.plot(epochs, train_acc,
                               label=f'{model_names[i]} - Train Acc',
                               color=col, linestyle='-')
        a_val, = ax_acc.plot(epochs, val_acc,
                             label=f'{model_names[i]} - Val Acc',
                             color=col, linestyle='--')
        model_artists[i].extend([a_train, a_val])
    ax_acc.set_xlabel('Epoch')
    ax_acc.set_ylabel('Accuracy (%)')
    ax_acc.set_title('Accuracy (Train et Validation) par Epoch')
    ax_acc.legend(loc='upper left')
    
    # --- Graphique 3 : Bar chart pour les résultats de test final (Distance hiérarchique moyenne) ---
    n_levels = len(models_test[0]['hierarchical_distance_avg'])
    levels = np.arange(n_levels)
    bar_width = 0.15  # Largeur des barres
    for i, test_data in enumerate(models_test):
        col = colors[i]
        distance_avg = test_data['hierarchical_distance_avg']
        bars = ax_bar.bar(levels + i * bar_width, distance_avg,
                          width=bar_width, label=model_names[i], color=col)
        for bar in bars:
            model_artists[i].append(bar)
    ax_bar.set_xlabel('Niveau hiérarchique')
    ax_bar.set_ylabel('Distance moyenne')
    ax_bar.set_title("Distance Hiérarchique Moyenne (Test Final)")
    ax_bar.set_xticks(levels + 2 * bar_width)
    ax_bar.set_xticklabels([f'Niveau {j+1}' for j in range(n_levels)])
    ax_bar.legend(loc='upper left')
    
    fig.tight_layout()
    
    return fig, model_artists, ax_loss, ax_acc, ax_bar

def main(model_dirs):
    # Génération dynamique des couleurs en fonction du nombre de dossiers
    num_models = len(model_dirs)
    cmap = cm.get_cmap('tab10') if num_models <= 10 else cm.get_cmap('tab20')
    colors = [mcolors.to_hex(cmap(i)) for i in range(num_models)]
    
    # Nom des modèles = nom du dossier
    model_names = [os.path.basename(os.path.normpath(folder)) for folder in model_dirs]
    
    # Charger les données pour chaque modèle
    models_epoch = []
    models_test  = []
    for folder in model_dirs:
        models_epoch.append(load_epoch_results(folder))
        models_test.append(load_final_test_results(folder))
    
    # Créer la figure matplotlib
    fig, model_artists, ax_loss, ax_acc, ax_bar = create_figure(
        models_epoch, models_test, colors, model_names
    )
    
    # --- Interface Tkinter ---
    root = tk.Tk()
    root.title("Comparaison des modèles")
    root.geometry("1920x1080")
    
    # Cadre principal à 3 lignes :
    #   ligne 0 : barre de checkboxes,
    #   ligne 1 : zone scrollable de la figure,
    #   ligne 2 : affichage des résultats finaux (tableau)
    main_frame = tk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True)
    main_frame.rowconfigure(1, weight=1)
    main_frame.columnconfigure(0, weight=1)
    
    # ----- Barre des checkboxes (ligne 0) -----
    top_frame = tk.Frame(main_frame)
    top_frame.grid(row=0, column=0, sticky='nw')
    
    var_list = []
    
    # ----- Zone scrollable pour la figure (ligne 1) -----
    canvas_frame = tk.Frame(main_frame)
    canvas_frame.grid(row=1, column=0, sticky='nsew')
    
    canvas = tk.Canvas(canvas_frame)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    v_scrollbar = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=canvas.yview)
    v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.configure(yscrollcommand=v_scrollbar.set)
    
    figure_frame = tk.Frame(canvas)
    canvas.create_window((0, 0), window=figure_frame, anchor='nw')
    
    fig_canvas = FigureCanvasTkAgg(fig, master=figure_frame)
    fig_canvas.draw()
    fig_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def update_scrollregion(event=None):
        canvas.configure(scrollregion=canvas.bbox("all"))
    figure_frame.bind("<Configure>", update_scrollregion)
    
    # ----- Zone pour afficher les résultats finaux (ligne 2) -----
    bottom_frame = tk.Frame(main_frame)
    bottom_frame.grid(row=2, column=0, sticky='nsew', pady=10)
    
    # Création d'un Treeview pour afficher les résultats finaux de test de chaque modèle
    # Colonnes : modèle, test_loss, test_accuracy, hierarchical_distance_avg (top-1),
    # hierarchical_precision (top-1) et hierarchical_mAP (top-1)
    tree = ttk.Treeview(bottom_frame, columns=("model", "test_loss", "test_accuracy", "hd_avg_top1", "hd_prec_top1", "hd_map_top1"), show='headings', height=5)
    tree.heading("model", text="Modèle")
    tree.heading("test_loss", text="Test Loss")
    tree.heading("test_accuracy", text="Test Accuracy (%)")
    tree.heading("hd_avg_top1", text="Hier. Dist Avg (Top-1)")
    tree.heading("hd_prec_top1", text="Hier. Prec (Top-1)")
    tree.heading("hd_map_top1", text="Hier. mAP (Top-1)")
    
    tree.column("model", width=150, anchor='center')
    tree.column("test_loss", width=100, anchor='center')
    tree.column("test_accuracy", width=120, anchor='center')
    tree.column("hd_avg_top1", width=150, anchor='center')
    tree.column("hd_prec_top1", width=150, anchor='center')
    tree.column("hd_map_top1", width=150, anchor='center')
    
    # Insertion des données pour chaque modèle
    for i, test_data in enumerate(models_test):
        tree.insert("", "end", values=(
            model_names[i],
            f"{test_data['test_loss']:.2f}",
            f"{test_data['test_accuracy']:.2f}",
            f"{test_data['hierarchical_distance_avg'][0]:.2f}" if 'hierarchical_distance_avg' in test_data and len(test_data['hierarchical_distance_avg']) > 0 else 'N/A',
            f"{test_data['hierarchical_precision'][0]:.2f}" if 'hierarchical_precision' in test_data and len(test_data['hierarchical_precision']) > 0 else 'N/A',
            f"{test_data['hierarchical_mAP'][0]:.2f}" if 'hierarchical_mAP' in test_data and len(test_data['hierarchical_mAP']) > 0 else 'N/A'
        ))
    
    tree.pack(fill=tk.X, padx=10)
    
    # ----------------------------------------------------------------
    # Fonction de mise à jour dynamique des légendes
    # ----------------------------------------------------------------
    def update_legends():
        # --- 1) Graphique de la Loss ---
        lines_loss = [line for line in ax_loss.lines if line.get_visible()]
        ax_loss.legend(lines_loss, [line.get_label() for line in lines_loss],
                       loc='upper left')
        
        # --- 2) Graphique de l'Accuracy ---
        lines_acc = [line for line in ax_acc.lines if line.get_visible()]
        ax_acc.legend(lines_acc, [line.get_label() for line in lines_acc],
                      loc='upper left')
        
        # --- 3) Graphique Barres (Test Final) ---
        handles_bar, labels_bar = ax_bar.get_legend_handles_labels()
        visible_handles = []
        visible_labels  = []
        for h, lbl in zip(handles_bar, labels_bar):
            if hasattr(h, 'get_visible'):
                if h.get_visible():
                    visible_handles.append(h)
                    visible_labels.append(lbl)
            else:
                if any(p.get_visible() for p in h):
                    visible_handles.append(h)
                    visible_labels.append(lbl)
        ax_bar.legend(visible_handles, visible_labels, loc='upper left')
        fig_canvas.draw()
    
    # ----------------------------------------------------------------
    # Fonction de bascule de visibilité (toggle)
    # ----------------------------------------------------------------
    def toggle_model(i):
        visible = var_list[i].get()
        for artist in model_artists[i]:
            artist.set_visible(visible)
        update_legends()
    
    # ----- Création des cases à cocher (horizontales) -----
    for i, name in enumerate(model_names):
        var = tk.BooleanVar(value=True)
        var_list.append(var)
        cb = tk.Checkbutton(top_frame, text=name, variable=var,
                            command=lambda i=i: toggle_model(i))
        cb.pack(side=tk.LEFT, padx=5)
    
    root.mainloop()

if __name__ == '__main__':
    # Détermine le dossier du script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Recherche tous les sous-dossiers du dossier du script
    all_subdirs = [os.path.join(script_dir, d) for d in os.listdir(script_dir)
                   if os.path.isdir(os.path.join(script_dir, d))]
    
    # Ne conserver que ceux contenant les fichiers JSON requis
    model_dirs = [folder for folder in all_subdirs
                  if os.path.exists(os.path.join(folder, 'all_epoch_results.json')) and
                     os.path.exists(os.path.join(folder, 'final_test_results.json'))]
    
    if not model_dirs:
        print("Aucun dossier contenant les fichiers JSON requis n'a été trouvé.")
    else:
        main(model_dirs)
