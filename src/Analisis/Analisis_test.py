import pandas as pd
import subprocess
import os
from pathlib import Path

def limpiar_datos(ruta_csv):
    """Limpia el CSV eliminando columnas repetidas"""
    df = pd.read_csv(ruta_csv)
    
    # Eliminar columnas especificadas ya que estan repetidas
    columnas_eliminar = ['paciente', 'Patient', 
                         'mascara_tiene_tumor', 'tamaño_tumor_pixeles']
    
    columnas_existentes = [col for col in columnas_eliminar if col in df.columns]
    df_limpio = df.drop(columns=columnas_existentes)
    
    # Guardar
    output_path = Path(ruta_csv).parent / "tumores_limpio.csv"
    df_limpio.to_csv(output_path, index=False)
    
    return str(output_path)

def analisis_descriptivo(ruta_csv):
    """Ejecuta análisis en R usando subprocess"""
    
    # Crear directorio para resultados
    output_dir = Path(ruta_csv).parent / "resultados_r"
    output_dir.mkdir(exist_ok=True)
    
    # Ruta al script R
    script_r = Path(__file__).parent / "Script_R_1.R"
    
    # Ejecutar R
    cmd = [
        r'C:\Program Files\R\R-4.5.2\bin\Rscript.exe',
        str(script_r),
        ruta_csv,
        str(output_dir)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Mostrar salida
    print(result.stdout)
    if result.stderr:
        print("Errores R:", result.stderr)
    
    if result.returncode == 0:
        print(f"\n✅ Análisis completado. Resultados en: {output_dir}")
        return str(output_dir)
    else:
        raise Exception(f"Error en R: {result.stderr}")