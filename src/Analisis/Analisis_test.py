#ANALISIS/ANALISIS_TEST

# Este script es el puente de unión entre Python y R. Primero, se encarga de "limpiar" 
# el Excel de los tumores para que solo queden los datos importantes. Después, abre R de forma 
# automática en segundo plano para generar todas las gráficas y estadísticas útiles para nuestro 
# estudio.


# LIBRERÍAS NECESARIAS:
import pandas as pd      # Para manipular las tablas de datos (dataframes)
import subprocess        # Para abrir y ejecutar programas externos (como R).
import os                # Para gestionar carpetas del sistema.
from pathlib import Path   # Para manejar rutas de archivos de forma segura (Linux-Mac-Windows).

# Preparamos los datos para dárselos a R.
def limpiar_datos(ruta_csv):
    """Prepara el CSV para que R no tenga problemas al leerlo"""
    df = pd.read_csv(ruta_csv)
    
    # Identificamos columnas que no aportan información al análisis estadístico
    # (variables repetidas o redundantes).
    columnas_eliminar = ['paciente', 'Patient', 'mascara_tiene_tumor', 'tamaño_tumor_pixeles']
    
    # Filtramos para borrar solo las que realmente existan en el archivo
    columnas_existentes = [col for col in columnas_eliminar if col in df.columns]
    df_limpio = df.drop(columns=columnas_existentes)
    
    # Guardamos el archivo limpio como 'tumores_limpio.csv'
    # Path(ruta_csv).parent = carpeta anterior
    output_path = Path(ruta_csv).parent / "tumores_limpio.csv"  
    df_limpio.to_csv(output_path, index=False)  # creamos el csv en la ruta dada
    
    return str(output_path) # str nos da el formato estándar de ruta

# La siguiente función permite realizar las representaciones gráficas en R
def analisis_descriptivo(ruta_csv):
    """Lanza el motor de R para generar las gráficas estadísticas"""
    
    # Creamos una carpeta 'resultados_r' para guardar los PNG y el informe final
    output_dir = Path(ruta_csv).parent / "resultados_r"
    output_dir.mkdir(exist_ok=True) # exist_ok: si ya existe la carpeta, sigue sin dar error
    
    # Buscamos dónde está el script de R (.R)
    # __file__ es una 'variable mágica': encuentra la ruta del script de R, que debe estar en la misma carpeta.
    script_r = Path(__file__).parent / "Script_R_1.R"
    
    # Configuramos la interacción con R
    # Le decimos: Usa R, ejecuta el script, lee el CSV y guárdalo.
    cmd = [
        r'C:\Program Files\R\R-4.5.2\bin\Rscript.exe', # Ruta al ejecutable de R
        str(script_r),                                 # El código R a ejecutar
        ruta_csv,                                      # El CSV de entrada
        str(output_dir)                                # La carpeta de salida
    ]
    
    # Lanzamos el proceso y esperamos a que termine, mostrándolo por pantalla.
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Mostramos en la terminal de Python lo que R muestra en la consola.
    print(result.stdout)
    if result.stderr:  # recopila los errores de R
        print("Errores R:", result.stderr)
    
    # Verificamos si R terminó con éxito (código 0)
    if result.returncode == 0:
        print(f"\n Análisis completado. Resultados en: {output_dir}")
        return str(output_dir)
    else:
        # Si R falla, lanzamos una excepción para avisar al usuario
        raise Exception(f"Error en R: {result.stderr}")