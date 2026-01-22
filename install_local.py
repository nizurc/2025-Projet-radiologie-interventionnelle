import sys
import os
from pathlib import Path
import site

def install_local_package(package_name):
    # 1. Trouver où est le code source
    cwd = Path.cwd()
    package_path = cwd / package_name
    
    if not package_path.exists():
        print(f"❌ Erreur : Le dossier '{package_name}' n'existe pas ici.")
        return

    # 2. Trouver le dossier site-packages de ton environnement virtuel
    site_packages = site.getsitepackages()[0]
    
    # 3. Créer le fichier .pth (le lien magique)
    pth_file = Path(site_packages) / f"{package_name}.pth"
    
    try:
        with open(pth_file, "w") as f:
            f.write(str(package_path))
        print(f"✅ SUCCÈS !")
        print(f"Le lien a été créé ici : {pth_file}")
        print(f"Il pointe vers : {package_path}")
        print("-" * 30)
        print(f"Tu peux maintenant faire 'import {package_name}' dans n'importe quel fichier !")
    except Exception as e:
        print(f"❌ Erreur lors de la création du lien : {e}")

if __name__ == "__main__":
    # On installe imodal
    install_local_package("imodal")