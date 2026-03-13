import numpy as np
from scipy.io import loadmat
import pandas as pd

# Cargar el archivo .mat
storedStructure = loadmat('/home/pedroj/Desktop/pps/senales_eeg/data/subj013.mat')

# Filtrar solo las variables reales (evitar __header__, __version__, __globals__)
variables = {k: v for k, v in storedStructure.items() if not k.startswith('__')}

# Nombre del archivo Excel de salida
fullFileName = '/home/pedroj/Desktop/pps/red_neuro/prueba_pasar_datos.xlsx'

with pd.ExcelWriter(fullFileName) as writer:
    for var_name, data in variables.items():
        data = np.array(data)
        
        # Si es 1D, convertir a DataFrame con una columna
        if data.ndim == 1:
            df = pd.DataFrame(data, columns=[var_name])
        # Si es 2D, poner columnas numeradas: varname_1, varname_2, ...
        elif data.ndim == 2:
            col_names = [f"{var_name}_{i+1}" for i in range(data.shape[1])]
            df = pd.DataFrame(data, columns=col_names)
        else:
            # Para más dimensiones, aplanar completamente y poner una columna
            df = pd.DataFrame(data.flatten(), columns=[var_name])
        
        # Guardar cada variable en su propia hoja
        df.to_excel(writer, sheet_name=var_name[:31], index=False)  # Excel limita nombres a 31 caracteres

print(f"Archivo Excel creado en {fullFileName} con cada variable en su propia hoja.")
