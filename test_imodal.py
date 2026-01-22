import sys
import os

# --- LA PARTIE MAGIQUE ---
# On dit à Python : "Regarde dans le dossier 'imodal' qui est juste à côté de moi"
current_dir = os.path.dirname(os.path.abspath(__file__))
path_to_imodal = os.path.join(current_dir, "imodal")

if path_to_imodal not in sys.path:
    sys.path.append(path_to_imodal)
# -------------------------

print("Tentative d'import...")

try:
    import imodal
    # On essaie d'importer un vrai sous-module pour être sûr
    from imodal import Utilities
    
    print("✅ SUCCÈS TOTAL !")
    print(f"Le module est chargé depuis : {imodal.__path__}")
    print("Tu peux commencer ta thèse.")
    
except ImportError as e:
    print(f"❌ Erreur : {e}")
    print(f"Vérifie que le dossier 'imodal' est bien ici : {path_to_imodal}")