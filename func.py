import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import torch
from tqdm import tqdm
sys.path.insert(0, os.path.abspath('imodal_git'))
import imodal
import imodal.Utilities


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

def _extract_triangle_faces(polydata):
    """
    Extract triangle faces from a triangulated PyVista PolyData.

    Returns
    -------
    np.ndarray
        Triangle connectivity with shape (M, 3), dtype int64.
    """
    faces = np.asarray(polydata.faces, dtype=np.int64)
    if faces.size == 0:
        return np.empty((0, 3), dtype=np.int64)

    tri_faces = []
    i = 0
    while i < faces.size:
        n = int(faces[i])
        if n == 3:
            tri_faces.append(faces[i + 1:i + 4])
        i += n + 1

    if len(tri_faces) == 0:
        return np.empty((0, 3), dtype=np.int64)

    return np.ascontiguousarray(np.vstack(tri_faces), dtype=np.int64)

def _geometry_to_points_and_faces(geometry):
    """
    Convert geometry to points and optional faces.

    Returns
    -------
    tuple[np.ndarray, np.ndarray | None]
        (points, faces). Faces are returned only for mesh inputs.

    """
    if isinstance(geometry, np.ndarray):
        points = np.asarray(geometry, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] not in (2, 3):
            raise ValueError(f"Invalid point shape: shape={points.shape}, expected (N,2) or (N,3).")
        return np.ascontiguousarray(points, dtype=np.float64), None

    if isinstance(geometry, pv.UnstructuredGrid):
        surface = geometry.extract_surface()
        surface = surface.triangulate()
        surface = surface.clean()
        if surface is None:
            raise ValueError("Failed to convert UnstructuredGrid to a triangulated surface.")
    elif isinstance(geometry, pv.PolyData):
        surface = geometry.triangulate()
        surface = surface.clean()
        if surface is None:
            raise ValueError("Failed to clean/triangulate PolyData.")
    else:
        raise TypeError(
            "Unsupported geometry type. Use np.ndarray, "
            "pyvista.PolyData, or pyvista.UnstructuredGrid."
        )

    points = np.asarray(surface.points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] not in (2, 3):
        raise ValueError(f"Invalid point shape: shape={points.shape}, expected (N,2) or (N,3).")

    faces = _extract_triangle_faces(surface)
    if faces.shape[0] == 0:
        result = (np.ascontiguousarray(points, dtype=np.float64), None)
        return result

    result = (np.ascontiguousarray(points, dtype=np.float64), faces)
    return result

def _chunked_varifold_scalar_product_3d(
    centers_x,
    centers_y,
    lengths_x,
    lengths_y,
    normalized_x,
    normalized_y,
    sigma,
    chunk_size,
    chunk_size_y=None,
):
    """
    Memory-efficient varifold scalar product in 3D using chunked pairwise blocks.
    """
    scalar = torch.zeros((), dtype=centers_x.dtype, device=centers_x.device)
    n_x = centers_x.shape[0]
    n_y = centers_y.shape[0]
    if chunk_size_y is None:
        y_block = n_y
    else:
        y_block = max(1, int(chunk_size_y))

    for i in range(0, n_x, chunk_size):
        i_end = min(i + chunk_size, n_x)
        cx = centers_x[i:i_end]
        ux = normalized_x[i:i_end]
        lx = lengths_x[i:i_end].view(-1)

        for j in range(0, n_y, y_block):
            j_end = min(j + y_block, n_y)
            cy = centers_y[j:j_end]
            uy = normalized_y[j:j_end]
            ly = lengths_y[j:j_end].view(-1, 1)

            kernel = imodal.Kernels.K_xy(cx, cy, sigma)
            dot_sq = torch.mm(ux, uy.t()) ** 2
            weighted = kernel * dot_sq
            convolved = torch.mm(weighted, ly).view(-1)
            scalar = scalar + torch.dot(lx, convolved)

    return scalar

def _chunked_varifold_cost_3d(source, target, sigmas, chunk_size, chunk_size_y=None):
    """
    3D varifold cost with chunked scalar products.

    Parameters
    ----------
    source : tuple[Tensor, Tensor]
        (vertices, faces) for source mesh.
    target : tuple[Tensor, Tensor]
        (vertices, faces) for target mesh.
    sigmas : list[float]
        Multi-scale varifold sigmas.
    chunk_size : int
        Chunk size for blockwise kernel computation.
    """
    vertices_source, faces_source = source
    vertices_target, faces_target = target

    centers_source, normals_source, lengths_source = imodal.Utilities.compute_centers_normals_lengths(vertices_source, faces_source)
    centers_target, normals_target, lengths_target = imodal.Utilities.compute_centers_normals_lengths(vertices_target, faces_target)

    eps = torch.finfo(vertices_source.dtype).eps
    normalized_source = normals_source / lengths_source.clamp_min(eps)
    normalized_target = normals_target / lengths_target.clamp_min(eps)

    loss = torch.zeros((), dtype=vertices_source.dtype, device=vertices_source.device)
    for sigma in sigmas:
        tt = _chunked_varifold_scalar_product_3d(
            centers_target,
            centers_target,
            lengths_target,
            lengths_target,
            normalized_target,
            normalized_target,
            sigma,
            chunk_size,
            chunk_size_y,
        )
        ss = _chunked_varifold_scalar_product_3d(
            centers_source,
            centers_source,
            lengths_source,
            lengths_source,
            normalized_source,
            normalized_source,
            sigma,
            chunk_size,
            chunk_size_y,
        )
        st = _chunked_varifold_scalar_product_3d(
            centers_source,
            centers_target,
            lengths_source,
            lengths_target,
            normalized_source,
            normalized_target,
            sigma,
            chunk_size,
            chunk_size_y,
        )
        loss = loss + tt + ss - 2.0 * st

    return loss

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

def registration_imodal(foie_pre, foie_post, vaisseaux_pre, vaisseaux_post, zone_ablation_post, params=None, dtype=torch.float32):
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
    dtype : torch.dtype, optionnel
        Précision des calculs (défaut: torch.float32). Peut être torch.float64 pour plus de précision au prix de la vitesse et de la mémoire.

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
    sigmas_liver = params.get('sigmas_liver', [0.5, 2.0])
    sigmas_vessel = params.get('sigmas_vessel', [0.5])
    chunk_size_varifold = int(params.get('chunk_size_varifold', 2048))
    chunk_size_varifold_y = params.get('chunk_size_varifold_y', None)
    show_progress = bool(params.get('show_progress', True))
    lbfgs_lr = float(params.get('lbfgs_lr', 1.0))
    lbfgs_max_iter = int(params.get('lbfgs_max_iter', 20))
    lbfgs_history_size = int(params.get('lbfgs_history_size', 10))
    lbfgs_line_search = params.get('lbfgs_line_search_fn', 'strong_wolfe')
    lbfgs_max_eval = params.get('lbfgs_max_eval', None)
    requested_device = params.get('device', 'auto')
    clear_cuda_cache = bool(params.get('clear_cuda_cache', False))

    if requested_device is None or requested_device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        try:
            device = torch.device(requested_device)
        except (TypeError, RuntimeError):
            print(f"Invalid device '{requested_device}'. Falling back to auto selection.")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cuda' and not torch.cuda.is_available():
            print("CUDA requested but not available. Falling back to CPU.")
            device = torch.device('cpu')

    if clear_cuda_cache and device.type == 'cuda':
        torch.cuda.empty_cache()

    d = foie_post.shape[1]

    # POINTS SOURCE (POST-OP) - CE QUI BOUGE
    source_foie = torch.tensor(foie_post, dtype=dtype, device=device, requires_grad=True)
    source_vaisseaux = [
        torch.tensor(v, dtype=dtype, device=device, requires_grad=True)
        for v in vaisseaux_post
    ]

    # POINTS CIBLE (PRE-OP) - CE QU'ON VEUT ATTEINDRE
    target_foie = torch.tensor(foie_pre, dtype=dtype, device=device)
    target_vaisseaux = [torch.tensor(v, dtype=dtype, device=device) for v in vaisseaux_pre]

    source_foie_faces = None
    target_foie_faces = None
    source_vaisseaux_faces = None
    target_vaisseaux_faces = None
    if d == 3:
        source_faces_np = params.get('_faces_liver_post', None)
        target_faces_np = params.get('_faces_liver_pre', None)
        source_vessel_faces_np = params.get('_faces_vessels_post', None)
        target_vessel_faces_np = params.get('_faces_vessels_pre', None)

        if source_faces_np is None or target_faces_np is None:
            raise ValueError(
                "3D varifold attachment requires mesh faces for liver source/target. "
                "Use registration_imodal_from_mesh with PyVista meshes."
            )

        source_foie_faces = torch.tensor(source_faces_np, dtype=torch.long, device=device)
        target_foie_faces = torch.tensor(target_faces_np, dtype=torch.long, device=device)

        if source_vessel_faces_np is not None and target_vessel_faces_np is not None:
            if len(source_vessel_faces_np) != len(source_vaisseaux) or len(target_vessel_faces_np) != len(target_vaisseaux):
                raise ValueError("Inconsistent vessel face lists length with vessel point lists.")
            source_vaisseaux_faces = [
                torch.tensor(faces, dtype=torch.long, device=device) if faces is not None else None
                for faces in source_vessel_faces_np
            ]
            target_vaisseaux_faces = [
                torch.tensor(faces, dtype=torch.long, device=device) if faces is not None else None
                for faces in target_vessel_faces_np
            ]
        else:
            source_vaisseaux_faces = [None] * len(source_vaisseaux)
            target_vaisseaux_faces = [None] * len(target_vaisseaux)

    # MOTEUR (ZONE D'ABLATION POST-OP)
    moteur_za = torch.tensor(zone_ablation_post, dtype=dtype, device=device)

    # MODELE DE CROISSANCE IMODAL
    N = moteur_za.shape[0]
    p = 1
    C = torch.zeros(N, d, p, device=device)
    C[:, :, 0] = 1.0 # Croissance isotrope
    # Gestion des rotations selon la dimension
    if d == 2:
        rot_init = torch.stack([imodal.Utilities.rot2d(0.)] * N).to(dtype=dtype, device=device)
    elif d == 3:
        rot_init = torch.stack([torch.eye(3, dtype=dtype, device=device)] * N)
    else:
        raise ValueError("La dimension des données doit être 2D ou 3D.")
    gd_init = (moteur_za, rot_init)
    mod_local = imodal.DeformationModules.ImplicitModule1(d, N, sigma=sigma_local, C=C, nu=nu_local, gd=gd_init) # Module Local
    mod_global = imodal.DeformationModules.ImplicitModule1(d, N, sigma=sigma_global, C=C, nu=nu_global, gd=gd_init) # Module Global
    if hasattr(mod_local, 'to'):
        mod_local = mod_local.to(device)
    if hasattr(mod_global, 'to'):
        mod_global = mod_global.to(device)

    # ON DEFINIT LES PARAMETRES DU RECALAGE
    opt_control_local = torch.zeros(1, device=device, requires_grad=True) 
    opt_control_global = torch.zeros(1, device=device, requires_grad=True)
    optimizer_kwargs = {
        'lr': lbfgs_lr,
        'max_iter': lbfgs_max_iter,
        'history_size': lbfgs_history_size,
        'line_search_fn': lbfgs_line_search,
    }
    if lbfgs_max_eval is not None:
        optimizer_kwargs['max_eval'] = int(lbfgs_max_eval)
    optimizer = torch.optim.LBFGS([opt_control_local, opt_control_global], **optimizer_kwargs)
    attachment_foie = imodal.Attachment.VarifoldAttachment(d, sigmas_liver)
    attachment_vaisseau = imodal.Attachment.VarifoldAttachment(d, sigmas_vessel)

    # RECALAGE
    total_calls = optimizer.defaults.get("max_eval", None) or optimizer.defaults.get("max_iter", None)
    pbar = tqdm(total=total_calls, desc="registration_imodal (LBFGS)", disable=not show_progress)
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
        if d == 3:
            input_deformed_foie = (deformed_source_foie, source_foie_faces)
            input_target_foie = (target_foie, target_foie_faces)
        else:
            input_deformed_foie = deformed_source_foie.unsqueeze(0)
            input_target_foie = target_foie.unsqueeze(0)
        
        if d == 3:
            loss_foie = _chunked_varifold_cost_3d(
                input_deformed_foie,
                input_target_foie,
                sigmas=sigmas_liver,
                chunk_size=chunk_size_varifold,
                chunk_size_y=chunk_size_varifold_y,
            )
        else:
            loss_foie = attachment_foie(input_deformed_foie, input_target_foie)

        loss_vaisseaux = 0.0
        source_vaisseaux_faces_iter = source_vaisseaux_faces if source_vaisseaux_faces is not None else [None] * len(source_vaisseaux)
        target_vaisseaux_faces_iter = target_vaisseaux_faces if target_vaisseaux_faces is not None else [None] * len(target_vaisseaux)

        for v_source, v_target, f_source, f_target in zip(
            source_vaisseaux,
            target_vaisseaux,
            source_vaisseaux_faces_iter,
            target_vaisseaux_faces_iter,
        ):
            # 1. Calcul de la vitesse pour CE vaisseau spécifique
            vitesse_v = mod_local(v_source) + mod_global(v_source)
            # 2. Déformation
            deformed_v = v_source + vitesse_v
            # 3. Ajout de son erreur à l'erreur totale des vaisseaux
            if d == 3 and f_source is not None and f_target is not None:
                loss_vaisseaux += _chunked_varifold_cost_3d(
                    (deformed_v, f_source),
                    (v_target, f_target),
                    sigmas=sigmas_vessel,
                    chunk_size=chunk_size_varifold,
                    chunk_size_y=chunk_size_varifold_y,
                )
            else:
                loss_vaisseaux += attachment_vaisseau(deformed_v.unsqueeze(0), v_target.unsqueeze(0))
        
        # Régularisation
        loss_reg = mod_local.cost() + mod_global.cost()
        
        loss = loss_foie + loss_vaisseaux + lamb * loss_reg
        
        loss.backward()
        closure_calls += 1
        if show_progress:
            if pbar.total is None or pbar.n < pbar.total:
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

def registration_imodal_from_mesh(
    liver_pre,
    liver_post,
    vessels_pre,
    vessels_post,
    ablation_zone_post,
    params=None,
    dtype=torch.float32
):
    """
    Converts PyVista objects (or ndarrays) to NumPy point arrays, then calls
    `registration_imodal` without changing its internal behavior.

    Parameters
    ----------
    liver_pre : PyVista PolyData or UnstructuredGrid or ndarray
        Geometry of the liver in the pre-operative state (target).
    liver_post : PyVista PolyData or UnstructuredGrid or ndarray
        Geometry of the liver in the post-operative state (source).
    vessels_pre : list of PyVista PolyData or UnstructuredGrid or ndarray
        List of geometries for the vessels in the pre-operative state (target).
    vessels_post : list of PyVista PolyData or UnstructuredGrid or ndarray
        List of geometries for the vessels in the post-operative state (source).
    ablation_zone_post : PyVista PolyData or UnstructuredGrid or ndarray
        Geometry of the ablation zone in the post-operative state (moteur de la déformation).
    params : dict, optional
        Additional parameters to pass to `registration_imodal`. Keys accepted:
    dtype : torch.dtype, optional
        Data type for the computations (default: torch.float32). Can be set to torch.float64 for higher precision at the cost of speed and memory.
    """
    if params is None:
        params = {}

    liver_pre_np, liver_pre_faces = _geometry_to_points_and_faces(
        liver_pre
    )
    liver_post_np, liver_post_faces = _geometry_to_points_and_faces(
        liver_post
    )
    ablation_zone_post_np, _ = _geometry_to_points_and_faces(
        ablation_zone_post
    )

    vessels_pre_data = [
        _geometry_to_points_and_faces(v)
        for i, v in enumerate(vessels_pre)
    ]
    vessels_post_data = [
        _geometry_to_points_and_faces(v)
        for i, v in enumerate(vessels_post)
    ]

    vessels_pre_np = [points for points, _faces in vessels_pre_data]
    vessels_post_np = [points for points, _faces in vessels_post_data]
    vessels_pre_faces = [faces for _points, faces in vessels_pre_data]
    vessels_post_faces = [faces for _points, faces in vessels_post_data]

    dim_pre = liver_pre_np.shape[1]
    dim_post = liver_post_np.shape[1]
    dim_ablation = ablation_zone_post_np.shape[1]
    if not (dim_pre == dim_post == dim_ablation):
        raise ValueError(
            f"Inconsistent dimensions: liver_pre={dim_pre}, liver_post={dim_post}, ablation_zone_post={dim_ablation}."
        )

    for i, vessel in enumerate(vessels_pre_np):
        if vessel.shape[1] != dim_pre:
            raise ValueError(f"Dimension of vessels_pre[{i}]={vessel.shape[1]} is inconsistent with liver={dim_pre}.")
    for i, vessel in enumerate(vessels_post_np):
        if vessel.shape[1] != dim_pre:
            raise ValueError(f"Dimension of vessels_post[{i}]={vessel.shape[1]} is inconsistent with liver={dim_pre}.")

    registration_params = dict(params)
    if dim_pre == 3:
        registration_params['_faces_liver_pre'] = liver_pre_faces
        registration_params['_faces_liver_post'] = liver_post_faces
        registration_params['_faces_vessels_pre'] = vessels_pre_faces
        registration_params['_faces_vessels_post'] = vessels_post_faces

    return registration_imodal(
        liver_pre_np,
        liver_post_np,
        vessels_pre_np,
        vessels_post_np,
        ablation_zone_post_np,
        params=registration_params,
        dtype=dtype
    )

def registration_imodal_geodesic_shooting(foie_pre, foie_post, vaisseaux_pre, vaisseaux_post, zone_ablation_post, params=None, dtype=torch.float32):
    """
    Recalage biomécanique 2D ou 3D avec imodal utilisant le shooting géodésique.

    On modélise la rétractation autour de la zone d'ablation comme un module de
    croissance implicite (ImplicitModule1), et on utilise l'infrastructure IMODAL
    (RegistrationModel + Fitter) pour optimiser les moments initiaux via un
    schéma de shooting Hamiltonien.

    Paramètres :
    ------------
    foie_pre : ndarray (N, d)
        Points du foie cible (Pré-Opératoire / Vérité terrain).
    foie_post : ndarray (M, d)
        Points du foie source (Post-Opératoire / Ce qui va être déformé).
    vaisseaux_pre : list de ndarray (K_i, d)
        Liste des repères vasculaires cibles (Pré-Opératoire).
    vaisseaux_post : list de ndarray (L_i, d)
        Liste des repères vasculaires sources (Post-Opératoire).
    zone_ablation_post : ndarray (P, d)
        Points de la zone d'ablation (moteur de la déformation).
    params : dict, optionnel
        Hyperparamètres. Clés acceptées :
        - 'sigma_local'  : échelle du module de croissance local (défaut: auto).
        - 'nu_local'     : régularisation du module local (défaut: 0.001).
        - 'sigma_global' : échelle du module de croissance global (défaut: auto).
        - 'nu_global'    : régularisation du module global (défaut: 0.001).
        - 'coeff_local'  : coefficient de coût du module local (défaut: 0.01).
        - 'coeff_global' : coefficient de coût du module global (défaut: 1.0).
        - 'lamb'         : poids de l'attachement aux données (défaut: 1.0).
        - 'varifold_sigmas_liver' : sigmas varifold pour le foie (défaut: auto).
        - 'vessel_weight': poids de l'attache des vaisseaux (défaut: 100.).
        - 'shoot_solver' : solveur ODE ('euler', 'rk4', etc., défaut: 'euler').
        - 'shoot_it'     : nombre de pas d'intégration (défaut: 10).
        - 'lbfgs_max_iter': itérations LBFGS max (défaut: 100).
        - 'fit_gd'       : optimiser aussi les positions du module (défaut: False).
    dtype : torch.dtype
        Précision des calculs.

    Retours :
    ---------
    final_deformed_liver : ndarray (M, d)
    final_deformed_vessels : list de ndarray
    final_deformed_ablation : ndarray (P, d)
    final_local_force : float (norme des moments locaux)
    final_global_force : float (norme des moments globaux)
    loss_history : list de float
    """
    if params is None:
        params = {}

    d = foie_post.shape[1]

    # --- Device selection ---
    requested_device = params.get('device', 'auto')
    if requested_device is None or requested_device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        try:
            device = torch.device(requested_device)
        except (TypeError, RuntimeError):
            print(f"Invalid device '{requested_device}'. Falling back to auto selection.")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cuda' and not torch.cuda.is_available():
            print("CUDA requested but not available. Falling back to CPU.")
            device = torch.device('cpu')

    clear_cuda_cache = bool(params.get('clear_cuda_cache', False))
    if clear_cuda_cache and device.type == 'cuda':
        torch.cuda.empty_cache()

    show_progress = bool(params.get('show_progress', True))
    random_seed = params.get('random_seed', None)
    if random_seed is not None:
        torch.manual_seed(int(random_seed))

    # --- Auto-scale sigmas based on data extent ---
    all_points = np.vstack([foie_post, zone_ablation_post] + list(vaisseaux_post))
    data_extent = np.ptp(all_points, axis=0)
    data_diag = float(np.linalg.norm(data_extent))

    sigma_local = params.get('sigma_local', data_diag * 0.08)
    nu_local = params.get('nu_local', 0.001)
    coeff_local = params.get('coeff_local', 0.01)
    sigma_global = params.get('sigma_global', data_diag * 0.25)
    nu_global = params.get('nu_global', 0.001)
    coeff_global = params.get('coeff_global', 1.0)
    lamb = params.get('lamb', 1.0)
    vessel_weight = params.get('vessel_weight', 100.)
    shoot_solver = params.get('shoot_solver', 'euler')
    shoot_it = int(params.get('shoot_it', 10))
    lbfgs_max_iter = int(params.get('lbfgs_max_iter', 100))
    fit_gd = params.get('fit_gd', False)

    # Varifold sigmas: auto-scale to ~5% and ~20% of diagonal
    default_varifold_sigmas = [data_diag * 0.05, data_diag * 0.2]
    varifold_sigmas_liver = params.get('varifold_sigmas_liver', default_varifold_sigmas)

    print(f"[registration_imodal] dim={d}, device={device}, dtype={dtype}")
    print(f"  data diagonal={data_diag:.2f}")
    print(f"  sigma_local={sigma_local:.4f}, nu_local={nu_local:.6f}, coeff_local={coeff_local}")
    print(f"  sigma_global={sigma_global:.4f}, nu_global={nu_global:.6f}, coeff_global={coeff_global}")
    print(f"  varifold_sigmas_liver={varifold_sigmas_liver}")
    print(f"  lamb={lamb}, vessel_weight={vessel_weight}")
    print(f"  shoot_solver={shoot_solver}, shoot_it={shoot_it}, lbfgs_max_iter={lbfgs_max_iter}")

    # --- Set IMODAL dtype and backend ---
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    imodal.Utilities.set_compute_backend('torch')

    try:
        # --- Build tensors ---
        source_foie_t = torch.tensor(foie_post, dtype=dtype, device=device)
        target_foie_t = torch.tensor(foie_pre, dtype=dtype, device=device)
        source_vaisseaux_t = [torch.tensor(v, dtype=dtype, device=device) for v in vaisseaux_post]
        target_vaisseaux_t = [torch.tensor(v, dtype=dtype, device=device) for v in vaisseaux_pre]
        moteur_za_t = torch.tensor(zone_ablation_post, dtype=dtype, device=device)

        # Faces for 3D varifold
        source_foie_faces_t = None
        target_foie_faces_t = None
        if d == 3:
            source_faces_np = params.get('_faces_liver_post', None)
            target_faces_np = params.get('_faces_liver_pre', None)
            if source_faces_np is not None:
                source_foie_faces_t = torch.tensor(source_faces_np, dtype=torch.long, device=device)
            if target_faces_np is not None:
                target_foie_faces_t = torch.tensor(target_faces_np, dtype=torch.long, device=device)

        # --- Source deformables (POST-OP, what gets deformed) ---
        deformables_source = []
        deformables_target = []
        attachments = []

        # Liver
        if d == 3 and source_foie_faces_t is not None and target_foie_faces_t is not None:
            deformable_liver_source = imodal.Models.DeformableMesh(source_foie_t, source_foie_faces_t.to(torch.int), label="liver")
            deformable_liver_target = imodal.Models.DeformableMesh(target_foie_t, target_foie_faces_t.to(torch.int), label="liver_target")
            attachments.append(imodal.Attachment.VarifoldAttachment(3, varifold_sigmas_liver, backend='torch'))
        else:
            deformable_liver_source = imodal.Models.DeformablePoints(source_foie_t, label="liver")
            deformable_liver_target = imodal.Models.DeformablePoints(target_foie_t, label="liver_target")
            attachments.append(imodal.Attachment.VarifoldAttachment(d, varifold_sigmas_liver, backend='torch'))
        deformables_source.append(deformable_liver_source)
        deformables_target.append(deformable_liver_target)

        # Vessels (landmarks): use pointwise L2 distance with correspondence
        for i, (v_src, v_tgt) in enumerate(zip(source_vaisseaux_t, target_vaisseaux_t)):
            deformables_source.append(imodal.Models.DeformablePoints(v_src, label=f"vessel_{i}"))
            deformables_target.append(imodal.Models.DeformablePoints(v_tgt, label=f"vessel_{i}_target"))
            attachments.append(imodal.Attachment.EuclideanPointwiseDistanceAttachment(weight=vessel_weight))

        # Ablation zone: transported passively through the flow (no cost)
        deformable_ablation_source = imodal.Models.DeformablePoints(moteur_za_t.clone(), label="ablation")
        deformable_ablation_target = imodal.Models.DeformablePoints(moteur_za_t.clone(), label="ablation_target")
        deformables_source.append(deformable_ablation_source)
        deformables_target.append(deformable_ablation_target)
        attachments.append(imodal.Attachment.NullLoss())

        # --- Deformation modules ---
        N_ablation = moteur_za_t.shape[0]
        C = torch.zeros(N_ablation, d, 1, dtype=dtype, device=device)
        C[:, :, 0] = 1.0  # isotropic growth

        if d == 2:
            rot_init = torch.stack([imodal.Utilities.rot2d(0.)] * N_ablation).to(dtype=dtype, device=device)
        elif d == 3:
            rot_init = torch.stack([torch.eye(3, dtype=dtype, device=device)] * N_ablation)
        else:
            raise ValueError("Dimension must be 2 or 3.")

        gd_init = (moteur_za_t.clone(), rot_init)

        mod_growth_local = imodal.DeformationModules.ImplicitModule1(
            d, N_ablation, sigma=sigma_local, C=C, nu=nu_local,
            coeff=coeff_local, gd=gd_init
        )
        mod_growth_global = imodal.DeformationModules.ImplicitModule1(
            d, N_ablation, sigma=sigma_global, C=C, nu=nu_global,
            coeff=coeff_global, gd=gd_init
        )

        # Small-scale corrections on the liver source points
        coeff_small = params.get('coeff_small', 1.0)
        sigma_small = params.get('sigma_small', sigma_local * 0.5)
        nu_small = params.get('nu_small', nu_local)
        mod_small = imodal.DeformationModules.ImplicitModule0(
            d, source_foie_t.shape[0], sigma_small, nu=nu_small,
            coeff=coeff_small, gd=source_foie_t.clone()
        )

        global_translation = imodal.DeformationModules.GlobalTranslation(d)

        deformation_modules = [global_translation, mod_growth_local, mod_growth_global, mod_small]

        # fit_gd: whether to also optimize module positions
        fit_gd_list = [False, fit_gd, fit_gd, False]

        # --- Build RegistrationModel ---
        model = imodal.Models.RegistrationModel(
            deformables_source,
            deformation_modules,
            attachments,
            fit_gd=fit_gd_list,
            lam=lamb,
        )
        model.to_device(str(device))

        # --- Fit ---
        costs = {}
        fitter = imodal.Models.Fitter(model, optimizer='torch_lbfgs')

        fitter.fit(
            deformables_target,
            lbfgs_max_iter,
            costs=costs,
            options={
                'shoot_solver': shoot_solver,
                'shoot_it': shoot_it,
                'line_search_fn': 'strong_wolfe',
            },
        )

        # --- Collect loss history ---
        loss_history = []
        if costs:
            keys = list(costs.keys())
            n_iters = len(costs[keys[0]])
            for i in range(n_iters):
                loss_history.append(sum(costs[k][i] for k in keys))

        # --- Extract deformed results ---
        intermediates = {}
        with torch.autograd.no_grad():
            deformed = model.compute_deformed(shoot_solver, shoot_it, intermediates=intermediates)

        # Liver = first deformable
        final_deformed_liver = deformed[0][0].detach().cpu().numpy()

        # Vessels = next deformables
        n_vessels = len(vaisseaux_post)
        final_deformed_vaisseaux = []
        for i in range(n_vessels):
            final_deformed_vaisseaux.append(deformed[1 + i][0].detach().cpu().numpy())

        # Ablation zone = last deformable (transported passively)
        idx_ablation = 1 + n_vessels
        final_deformed_za = deformed[idx_ablation][0].detach().cpu().numpy()

        # Norms of momenta as summary statistics
        n_deformables = len(deformables_source)
        final_states = intermediates['states'][-1]
        local_cotan_norm = 0.
        global_cotan_norm = 0.
        try:
            local_cotan_norm = float(final_states[n_deformables + 1].cotan[0].norm().item())
        except Exception:
            pass
        try:
            global_cotan_norm = float(final_states[n_deformables + 2].cotan[0].norm().item())
        except Exception:
            pass

    finally:
        torch.set_default_dtype(prev_dtype)

    return (
        final_deformed_liver,
        final_deformed_vaisseaux,
        final_deformed_za,
        local_cotan_norm,
        global_cotan_norm,
        loss_history,
    )

def registration_imodal_geodesic_shooting_from_mesh(
    liver_pre,
    liver_post,
    vessels_pre,
    vessels_post,
    ablation_zone_post,
    params=None,
    dtype=torch.float32
):
    """
    Converts PyVista objects (or ndarrays) to NumPy arrays, then calls
    ``registration_imodal`` with proper mesh face information for 3-D
    varifold attachment.

    Parameters
    ----------
    liver_pre : PyVista PolyData or UnstructuredGrid or ndarray
        Geometry of the liver in the pre-operative state (target).
    liver_post : PyVista PolyData or UnstructuredGrid or ndarray
        Geometry of the liver in the post-operative state (source).
    vessels_pre : list of PyVista PolyData or UnstructuredGrid or ndarray
        Vessel landmarks in the pre-operative state (target).
    vessels_post : list of PyVista PolyData or UnstructuredGrid or ndarray
        Vessel landmarks in the post-operative state (source).
    ablation_zone_post : PyVista PolyData or UnstructuredGrid or ndarray
        Ablation zone geometry (moteur de la déformation).
    params : dict, optional
        Additional parameters forwarded to ``registration_imodal_geodesic_shooting``.
    dtype : torch.dtype, optional
        Data type (default: torch.float32).
    """
    if params is None:
        params = {}

    liver_pre_np, liver_pre_faces = _geometry_to_points_and_faces(liver_pre)
    liver_post_np, liver_post_faces = _geometry_to_points_and_faces(liver_post)
    ablation_zone_post_np, _ = _geometry_to_points_and_faces(ablation_zone_post)

    vessels_pre_data = [_geometry_to_points_and_faces(v) for v in vessels_pre]
    vessels_post_data = [_geometry_to_points_and_faces(v) for v in vessels_post]

    vessels_pre_np = [pts for pts, _ in vessels_pre_data]
    vessels_post_np = [pts for pts, _ in vessels_post_data]

    dim = liver_pre_np.shape[1]
    if not (liver_post_np.shape[1] == dim == ablation_zone_post_np.shape[1]):
        raise ValueError(
            f"Inconsistent dimensions: liver_pre={dim}, "
            f"liver_post={liver_post_np.shape[1]}, "
            f"ablation={ablation_zone_post_np.shape[1]}."
        )

    registration_params = dict(params)
    if dim == 3:
        registration_params['_faces_liver_pre'] = liver_pre_faces
        registration_params['_faces_liver_post'] = liver_post_faces

    return registration_imodal_geodesic_shooting(
        liver_pre_np,
        liver_post_np,
        vessels_pre_np,
        vessels_post_np,
        ablation_zone_post_np,
        params=registration_params,
        dtype=dtype,
    )
