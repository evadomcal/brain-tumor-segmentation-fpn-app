# OBTENCIÓN DE DATOS
import shutil
from pathlib import Path
import kagglehub

ruta_descargas = Path("C:\Users\Usuario\Desktop\UNI\5\SEGUNDO_CUATRI\IA\EjerciciosPython/Trabajo/kaggle.json")

# Crear carpeta .kaggle si no existe
carpeta_kaggle = Path("C:\Users\Usuario\Desktop\UNI\5\SEGUNDO_CUATRI\IA\EjerciciosPython/TRABAJO/.kaggle")
carpeta_kaggle.mkdir(exist_ok=True)

# Copiar el archivo
shutil.copy(ruta_descargas, carpeta_kaggle / "kaggle.json")


path = kagglehub.dataset_download("mateuszbuda/lgg-mri-segmentation")

print("Path al dataset:", path)