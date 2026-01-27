import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

# FONCTION DE RÉTRACTION
def apply_retraction(points, center, strength, radius_influence):
    """
    Simule la contraction des tissus vers un centre (brûlure).
    """
    vectors = points - center
    distances = np.linalg.norm(vectors, axis=1)
    
    # Sécurité division par zéro
    distances[distances == 0] = 0.001
    normalized_vectors = vectors / distances[:, None]
    
    # Formule Gaussienne inversée
    displacement_magnitude = strength * distances * np.exp(- (distances**2) / (2 * radius_influence**2))
    
    # Déplacement VERS le centre
    new_points = points - (normalized_vectors * displacement_magnitude[:, None])
    return new_points

# GENERATION DES FORMES JOUET
# 1) Foie en forme de haricot
def get_liver_shape(n_points=400):
    t = np.linspace(0, 2*np.pi, n_points)
    # Forme "Haricot"
    x = 10 * (np.cos(t) + 0.2 * np.cos(2*t))
    y = 8 * (np.sin(t) - 0.1 * np.sin(2*t))
    return np.column_stack((x, y))

# 2) Zone d'abalation ou tumeur en forme de cercle
def get_circle(center, radius, n_points=100):
    t = np.linspace(0, 2*np.pi, n_points)
    x = center[0] + radius * np.cos(t)
    y = center[1] + radius * np.sin(t)
    return np.column_stack((x, y))

# 3) Vaisseau en arc de cercle
def get_arc_vessel(center_ablation, radius_vessel, start_angle, end_angle, n_points=60):
    """
    Crée un vaisseau principal en arc de cercle autour de l'ablation
    + une branche qui part vers l'extérieur.
    """
    t = np.linspace(np.radians(start_angle), np.radians(end_angle), n_points)
    arc_x = center_ablation[0] + radius_vessel * np.cos(t)
    arc_y = center_ablation[1] + radius_vessel * np.sin(t)
    arc = np.column_stack((arc_x, arc_y))
    
    return arc

# CONVERSION D'UN OBJET NUMY EN POLYDATA
def curve_to_polydata(points_2d, is_closed=False):
    """
    Convertit une liste de points 2D (x, y) en un objet PyVista PolyData (x, y, z=0).
    is_closed=True -> Relie le dernier point au premier (pour le Foie/Ablation)
    is_closed=False -> Laisse ouvert (pour les Vaisseaux)
    """
    n_points = len(points_2d)
    
    # 1. Ajouter la dimension Z=0
    points_3d = np.column_stack((points_2d, np.zeros(n_points)))
    
    # 2. Créer la connectivité (Lignes)
    # Pour PyVista, une ligne se définit par : [Nombre_de_points, idx0, idx1, idx2...]
    if is_closed:
        # On ferme la boucle : 0->1->...->N->0
        # On crée une seule cellule qui contient tous les points
        cells = np.r_[n_points, np.arange(n_points)]
        # Note : PyVista comprend qu'il faut fermer si on utilise 'lines' correctement ou via constructeur
        # Méthode explicite ligne par ligne pour compatibilité maximale :
        lines = np.column_stack((
            np.full(n_points, 2),       # Chaque segment a 2 points
            np.arange(n_points),        # Point A
            np.roll(np.arange(n_points), -1) # Point B (le dernier se lie au premier)
        )).ravel()
    else:
        # Ligne ouverte : 0->1->...->N
        # On relie i à i+1
        lines = np.column_stack((
            np.full(n_points - 1, 2),
            np.arange(n_points - 1),
            np.arange(1, n_points)
        )).ravel()

    # 3. Création de l'objet PolyData
    mesh = pv.PolyData(points_3d)
    mesh.lines = lines
    return mesh

# FUSION DE PLUSIEURS VAISSEAUX
def merge_vessels(list_of_vessel_arrays):
    """Fusionne une liste de vaisseaux (tableaux numpy) en un seul Mesh PolyData"""
    combined_mesh = pv.PolyData()
    for v_points in list_of_vessel_arrays:
        v_mesh = curve_to_polydata(v_points, is_closed=False)
        combined_mesh += v_mesh # Fusion VTK automatique
    return combined_mesh

# VISUALISATEUR AVANT/APRÈS AVEC GRILLE DE DÉFORMATION
def visualiser_zoom_grid(source, target, morphed, window_size=(1200, 600)):
    """
    Affiche une comparaison Avant/Après avec la grille de déformation visible.
    
    Args:
        source: PolyData (Avant déformation)
        target: PolyData (Cible)
        morphed: PolyData (Résultat déformé)
        model: L'objet modèle (pour récupérer le scale)
    """
    source_color = "teal"
    target_color = "red"

    plotter = pv.Plotter(shape=(1, 2), window_size=window_size)

    # --- GAUCHE : AVANT ---
    plotter.subplot(0, 0)
    plotter.add_text("Source non recalée VS cible", font_size=10, color="black")

    # Affichage de la grille en fil de fer (SANS LES POINTS NOIRS)
    if hasattr(source, "control_points"):
        plotter.add_mesh(source.control_points.to_pyvista(), color="black", 
                         style="wireframe", line_width=1, opacity=0.4)
        # J'ai retiré le bloc qui ajoutait les points ici

    plotter.add_mesh(source.to_pyvista(), color=source_color, opacity=0.7, line_width=2, label="Source non recalée")
    plotter.add_mesh(target.to_pyvista(), color=target_color, style="wireframe", line_width=2, label="Cible")
    
    plotter.add_legend()
    plotter.view_xy()
    plotter.enable_2d_style() # Pour bloquer la rotation 3D

    # --- DROITE : APRÈS ---
    plotter.subplot(0, 1)
    plotter.add_text("Source recalée VS cible", font_size=10, color="black")

    if hasattr(morphed, "control_points"):
        # On voit bien comment la grille fine s'est tordue
        plotter.add_mesh(morphed.control_points.to_pyvista(), color="black", 
                         style="wireframe", line_width=1, opacity=0.4)

    plotter.add_mesh(morphed.to_pyvista(), color=source_color, opacity=0.7, line_width=2, label="Source recalée")
    plotter.add_mesh(target.to_pyvista(), color=target_color, style="wireframe", line_width=2, label="Cible")
    
    plotter.add_legend()
    plotter.view_xy()
    plotter.enable_2d_style() # Pour bloquer la rotation 3D

    # Finalisation
    plotter.link_views()
    plotter.show()

def visualiser_validation_only(verite_terrain, resultat_recale, window_size=(1000, 1000)):
    """
    Version ultra-simplifiée.
    Affiche uniquement la superposition : Vérité Terrain (Bleu) vs Résultat (Rouge).
    """
    
    # 1. Conversion de sécurité (accepte Scikit-Shapes ou PyVista)
    # Si c'est du Scikit-Shapes, on convertit. Sinon on garde tel quel.
    pv_verite = verite_terrain.to_pyvista() if hasattr(verite_terrain, "to_pyvista") else verite_terrain
    pv_resultat = resultat_recale.to_pyvista() if hasattr(resultat_recale, "to_pyvista") else resultat_recale

    # 2. Création de la fenêtre
    pl = pv.Plotter(window_size=window_size)
    pl.add_text("VALIDATION FINALE (De-Rétractation)", font_size=12, color="black")
    
    # A. Vérité Terrain (Bleu) en mode "Fil de fer" standard
    pl.add_mesh(pv_verite, color="blue", style="wireframe", line_width=2, label="Vérité Terrain")
    
    # B. Résultat (Rouge) en mode "Surface" transparente
    pl.add_mesh(pv_resultat, color="red", opacity=0.5, show_edges=False, label="Zone Corrigée")
    
    # 3. Affichage 2D
    pl.add_legend()
    pl.view_xy()          # Vue de dessus
    pl.enable_2d_style()  # Bloque la rotation 3D
    
    pl.show()

def visualiser_intervention(
    foie_pre, zone_ablation_pre, tumeur_pre, vaisseaux_pre,
    foie_post, zone_ablation_post, vaisseaux_post,
    figsize=(12, 6)
):
    """
    Affiche une comparaison côte à côte des états pré-opératoire et post-opératoire
    pour le foie, la tumeur, la zone d'ablation et les vaisseaux.
    """
    plt.figure(figsize=figsize)

    # --- GAUCHE : PRÉ-OP ---
    plt.subplot(1, 2, 1)
    plt.title("Pré-op")
    
    # Structures statiques
    plt.plot(foie_pre[:,0], foie_pre[:,1], 'k-', lw=1, label='Foie')
    plt.fill(zone_ablation_pre[:,0], zone_ablation_pre[:,1], 'r', alpha=0.5, label='Zone Ablation (ground truth)')
    plt.fill(tumeur_pre[:,0], tumeur_pre[:,1], 'k', label='Tumeur')

    # Boucle sur les vaisseaux
    for i, v in enumerate(vaisseaux_pre):
        # On ne met le label que sur le premier pour ne pas polluer la légende
        label = "Vaisseau" if i == 0 else ""
        plt.plot(v[:,0], v[:,1], 'b-', lw=3, label=label)

    plt.axis('equal')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    # --- DROITE : POST-OP ---
    plt.subplot(1, 2, 2)
    plt.title("Post-op")
    
    # Structures statiques
    plt.plot(foie_post[:,0], foie_post[:,1], 'k-', lw=1, label='Foie')
    plt.fill(zone_ablation_post[:,0], zone_ablation_post[:,1], 'r', alpha=0.5, label='Zone Ablation')

    # Boucle sur les vaisseaux déformés
    for i, v in enumerate(vaisseaux_post):
        label = "Vaisseau" if i == 0 else ""
        plt.plot(v[:,0], v[:,1], 'b-', lw=3, label=label)

    plt.axis('equal')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()