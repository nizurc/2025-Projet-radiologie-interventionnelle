import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import torch
from tqdm import tqdm
sys.path.insert(0, os.path.abspath('imodal_git'))
import imodal

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

def registration_imodal(foie_pre, foie_post, vaisseaux_pre, vaisseaux_post, zone_ablation_post, params=None):
    """
    Exécute un recalage biomécanique 2D ou 3D avec imodal, basé sur la modélisation de la croissance 
    d'une zone d'ablation, tout en contraignant la déformation avec un réseau vasculaire.

    Paramètres :
    ------------
    foie_pre : ndarray (N, 2)
        Contour du foie cible (Pré-Opératoire / Vérité terrain).
    foie_post : ndarray (M, 2)
        Contour du foie source (Post-Opératoire / Ce qui va être déformé).
    vaisseaux_pre : list de ndarray (K, 2)
        Liste des segments vasculaires cibles (Pré-Opératoire).
    vaisseaux_post : list de ndarray (L, 2)
        Liste des segments vasculaires sources (Post-Opératoire).
    zone_ablation_post : ndarray (P, 2)
        Contour de la zone d'ablation qui agit comme le moteur de la déformation.
    params : dict, optionnel
        Dictionnaire des hyperparamètres physiques. Clés acceptées :
        - 'sigma_local', 'sigma_global' : Échelles géométriques des forces (défaut: 0.5, 2.0).
        - 'nu_local', 'nu_global' : Pénalités de régularisation (défaut: 0.01, 1.0).
        - 'lamb' : Poids de l'attachement géométrique (défaut: 1.0).

    Retours :
    ---------
    final_deformed_source_foie : ndarray (M, 2)
        Coordonnées du foie recalé.
    final_deformed_vaisseaux : list de ndarray (L, 2)
        Liste des coordonnées des vaisseaux recalés.
    final_deformed_za : ndarray (P, 2)
        Coordonnées de la zone d'ablation dilatée.
    final_local_force : float
        Intensité optimale trouvée pour le module local.
    final_global_force : float
        Intensité optimale trouvée pour le module global.
    loss_history : list de float
        Historique de la fonction de coût à chaque itération L-BFGS.
    """
    
    if params is None:
        params = {}

    sigma_local = params.get('sigma_local', 0.5)
    nu_local = params.get('nu_local', 0.01)
    sigma_global = params.get('sigma_global', 2.0)
    nu_global = params.get('nu_global', 1.0)
    lamb = params.get('lamb', 1.0)

    # POINTS SOURCE (POST-OP) - CE QUI BOUGE
    source_foie = torch.tensor(foie_post, dtype=torch.get_default_dtype(), requires_grad=True)
    source_vaisseaux = [torch.tensor(v, dtype=torch.get_default_dtype(), requires_grad=True) for v in vaisseaux_post]

    # POINTS CIBLE (PRE-OP) - CE QU'ON VEUT ATTEINDRE
    target_foie = torch.tensor(foie_pre, dtype=torch.get_default_dtype())
    target_vaisseaux = [torch.tensor(v, dtype=torch.get_default_dtype()) for v in vaisseaux_pre]

    # MOTEUR (ZONE D'ABLATION POST-OP)
    moteur_za = torch.tensor(zone_ablation_post, dtype=torch.get_default_dtype())

    # MODELE DE CROISSANCE IMODAL
    N = moteur_za.shape[0]
    d = foie_post.shape[1]
    p = 1
    C = torch.zeros(N, d, p)
    C[:, :, 0] = 1.0 # Croissance isotrope
    # Gestion des rotations selon la dimension
    if d == 2:
        rot_init = torch.stack([imodal.Utilities.rot2d(0.)] * N)
    elif d == 3:
        rot_init = torch.stack([torch.eye(3, dtype=torch.get_default_dtype())] * N)
    else:
        raise ValueError("La dimension des données doit être 2D ou 3D.")
    gd_init = (moteur_za, rot_init)
    mod_local = imodal.DeformationModules.ImplicitModule1(d, N, sigma=sigma_local, C=C, nu=nu_local, gd=gd_init) # Module Local
    mod_global = imodal.DeformationModules.ImplicitModule1(d, N, sigma=sigma_global, C=C, nu=nu_global, gd=gd_init) # Module Global

    # ON DEFINIT LES PARAMETRES DU RECALAGE
    opt_control_local = torch.zeros(1, requires_grad=True) 
    opt_control_global = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([opt_control_local, opt_control_global], lr=1.0, max_iter=20, history_size=10, line_search_fn='strong_wolfe')
    attachment_foie = imodal.Attachment.VarifoldAttachment(d, [0.5, 2.0])
    attachment_vaisseau = imodal.Attachment.VarifoldAttachment(d, [0.5])

    # RECALAGE
    total_calls = optimizer.defaults.get("max_eval", None) or optimizer.defaults.get("max_iter", None)
    pbar = tqdm(total=total_calls, desc="registration_imodal (LBFGS)")
    closure_calls = 0
    loss_history = []
    def closure():
        nonlocal closure_calls
        optimizer.zero_grad()

        # 1. Mise à jour des contrôles
        mod_local.fill_controls(opt_control_local)
        mod_global.fill_controls(opt_control_global)
        
        # 2. Calcul de la vitesse
        vitesse_foie = mod_local(source_foie) + mod_global(source_foie)
        
        # 3. Déformation
        deformed_source_foie = source_foie + vitesse_foie
        
        # 4. Calcul de l'erreur (Loss)
        input_deformed_foie = deformed_source_foie.unsqueeze(0)
        input_target_foie = target_foie.unsqueeze(0)
        
        loss_foie = attachment_foie(input_deformed_foie, input_target_foie)

        loss_vaisseaux = 0.0
        for v_source, v_target in zip(source_vaisseaux, target_vaisseaux):
            # 1. Calcul de la vitesse pour CE vaisseau spécifique
            vitesse_v = mod_local(v_source) + mod_global(v_source)
            # 2. Déformation
            deformed_v = v_source + vitesse_v
            # 3. Ajout de son erreur à l'erreur totale des vaisseaux
            loss_vaisseaux += attachment_vaisseau(deformed_v.unsqueeze(0), v_target.unsqueeze(0))
        
        # Régularisation
        loss_reg = mod_local.cost() + mod_global.cost()
        
        loss = loss_foie + loss_vaisseaux + lamb * loss_reg
        
        loss.backward()
        closure_calls += 1
        if pbar.total is None:
            pbar.update(1)
        elif pbar.n < pbar.total:
            pbar.update(1)
        pbar.set_postfix(loss=f"{loss.item():.4e}", calls=closure_calls)

        loss_history.append(loss.item())
        return loss

    optimizer.step(closure) # lancement de l'optimisation

    # RÉSULTATS
    final_local_force = opt_control_local.item()
    final_global_force = opt_control_global.item()
    with torch.no_grad():
        # 1. On s'assure que les modules ont les bons contrôles
        mod_local.fill_controls(opt_control_local)
        mod_global.fill_controls(opt_control_global)
        
        # 2. Calcul de la déformation pour le FOIE entier (Source)
        vitesse_foie = mod_local(source_foie) + mod_global(source_foie)
        final_deformed_source_foie = source_foie + vitesse_foie
        
        # 3. Calcul de la déformation pour l'ABLATION (Moteur)
        vitesse_za = mod_local(moteur_za) + mod_global(moteur_za)
        final_deformed_za = moteur_za + vitesse_za
        
        # 4. Calcul de la déformation pour les VAISSEAUX
        final_deformed_vaisseaux = []
        for seg in source_vaisseaux:
            vitesse_v = mod_local(seg) + mod_global(seg)
            deformed_v = seg + vitesse_v
            final_deformed_vaisseaux.append(deformed_v.detach().cpu().numpy())

    return (
        final_deformed_source_foie.detach().cpu().numpy(),  
        final_deformed_vaisseaux,                           
        final_deformed_za.detach().cpu().numpy(),           
        final_local_force, 
        final_global_force, 
        loss_history
    )
