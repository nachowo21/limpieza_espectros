def editar_fits_(ruta_fits_entrada, ruta_salida):
    import os
    from astropy.wcs import WCS
    from astropy.io import fits
    from astropy.time import Time
    from datetime import datetime
    from natural_cubic_spline import continuum
    import numpy as np
    import shutil
    from scipy.interpolate import UnivariateSpline
    """esta funcion se encargará de editar algunas cosas del archivo fits y luego guardarlo
    inputs:
    ruta_fits_entrada: es la ruta del archivo fits a modificar
    ruta_salida: es la ruta en donde se desea guardar el archivo fits modificado
    high_reject: es un input para la mascara en el clipping iterativo, indica cuantas veces queremos que se desvie el espectro de la 
    desviacion estandar para cortarlo por arriba
    lowreject:es un input para la mascara en el clipping iterativo, indica cuantas veces queremos que se desvie el espectro de la 
    desviacion estandar para cortarlo por abajo
    
    outputs: 
    una serie de printsque dice donde se aloja el nuevo archivo, con normalizacion en la data del archivo y modificaciones en el header
    y el return retorna el path completo del archivo"""

    # primero asegurarse que esté bien copiada la ruta
    if not os.path.exists(ruta_fits_entrada):
        raise FileNotFoundError(f"El archivo de entrada {ruta_fits_entrada} no existe.")

    # obtener el nombre base del archivo
    nombre_base = os.path.basename(ruta_fits_entrada)
    nombre_sin_ext, ext = os.path.splitext(nombre_base)
    nombre_salida = f"{nombre_sin_ext}_modificado{ext}"
    ruta_completa_salida = os.path.join(ruta_salida,nombre_salida)
    # crear directorio de salida si es que no existe
    os.makedirs(ruta_salida, exist_ok = True)
    
    # copiar la ruta del archivo plantilla en la ruta de salida
    shutil.copy(ruta_fits_entrada, ruta_completa_salida)
    print(f"archivo {ruta_fits_entrada}, copiado en {ruta_completa_salida}")

    # abrimos 
    with fits.open(ruta_completa_salida, mode= 'update') as hdul:
        # obtener los datos
        data = hdul[0].data
        HDR = hdul[0].header
        n_pix = data.shape[0]
        w = WCS(HDR, naxis = 1)
        long_ond = w.pixel_to_world(np.arange(n_pix)).value
        date_time = F"{HDR['DATE-OBS']}T{HDR['UT']}"
        t = Time(date_time,format = 'isot',scale='utc')
        dt = t.datetime
        ut_sec =  (dt.hour*3600) + (dt.minute*60) + dt.second + (dt.microsecond/1e6)
        #modificacion del header
        
        HDR['DATE-OBS'] = t.mjd
        HDR['UT'] = ut_sec

        #modificacion del flujo
        flux_norm, y_fit, w_excl, f_excl =\
    		continuum(long_ond,data,low_rej=1.8,high_re=0.0,niter=10,order=3,plots=True)

        hdul[0].data = flux_norm
        
        hdul.flush()
    print(f"Archivo modificado se guardó en: {ruta_completa_salida}")
    return ruta_completa_salida
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def archivo_par_2(ruta_fits, ruta_):
    from astropy.io import fits
    from astropy.wcs import WCS
    from datetime import datetime
    import os 
    import shutil
    import glob
    import re

    """Función para crear archivos .par para la aplicación de molecfit

    Args:
        ruta_fits: Ruta donde se encuentra el archivo FITS a procesar
        ruta_: Ruta donde existe el archivo .par de ejemplo
    """
    
    # Nombre del archivo que sirve para clonar
    nombre_shoter = "molecfit_XSHOOTER_NIR_Pipeline_R71.par"
    base = os.path.basename(ruta_fits)
    nuevo_nombre = os.path.splitext(base)[0] + ".par"
    
    # Clonar el archivo de ejemplo y asignar un nombre al nuevo archivo
    nuevo_output_dir = "/home/nacho/molecfit_test/mis_outputsVAC_star9"
    ruta_original = os.path.join(ruta_, nombre_shoter)
    ruta_nueva = os.path.join(nuevo_output_dir, nuevo_nombre)
    shutil.copy(ruta_original, ruta_nueva)
    print(f'Archivo clonado en {ruta_nueva}')
    
    # buscar el archivo.dat correspondiente 
    ruta_excludes = "/home/nacho/molecfit_test/excludes_star9/"
    match = os.path.basename(ruta_fits)
    m = re.search(r'_m(\d+)_',match)
    orden = m.group(1) if m else None 

    wrange_exclude = "wrange_exclude: none"
    if orden:
        patron = os.path.join(ruta_excludes, f"*m{orden}*.dat")
        candidatos = glob.glob(patron)
        if candidatos:
            ruta_dat = max(candidatos, key= os.path.getctime)
            wrange_exclude = f"wrange_exclude: {ruta_dat}"
            print(f"se encontr exclude de orden m{orden}, en {ruta_dat}")
        else:
            print(f"no se encontró ,dat para el orden m{orden} en {ruta_excludes}")
    else:
        print("no se detetctó orden en el nombre del fits")
        
    moleculas_modelo = "H2O CO2 CH4 O2"  # Podrían ser NO2 y CO
    
    # Diccionario de parámetros a modificar (sin incluir los dos puntos en las claves)
    modificar = {
        "filename": f"filename: {ruta_fits}",
        "listname": "listname: none",
        "wlgtomicron": "wlgtomicron: 0.0001",
        "vac_air": "vac_air: vac",
        "wrange_include": "wrange_include: none",
        "wrange_exclude": wrange_exclude,
        "prange_exclude": "prange_exclude: none",
        "output_dir": f"output_dir: {nuevo_output_dir}",
        "output_name": f"output_name: {nuevo_nombre}",
        "ftol": "ftol: 1e-05",
        "xtol": "xtol: 1e-05",
        "list_molec": f"list_molec: {moleculas_modelo}",
        "fit_molec": "fit_molec: 1 1 1 1",
        "flux_unit": "flux_unit: 0",
        "cont_n": "cont_n: 4",
        "cont_const":"cont_const: 1.0",
        "fit_wlc" : "fit_wlc: 1",
        "wlc_n": "wlc_n: 4",
        "kernmode":"kernmode: 0",
        "fit_res_gauss":"fit_res_gauss: 1",
        "res_gauss" : "res_gauss: 3.7",
        "fit_res_lorentz": "fit_res_lorentz: 0",
        "kernfac" : "kernfac: 300",
        "obsdate_key": "obsdate_key: DATE-OBS",
        "utc_key": "utc_key: UT",
        "telalt_key": "telalt_key: ALTITUDE",
        "rhum_key": "rhum_key: OUT-HUM",
        "pres_key": "pres_key: OUT-PRS",
        "temp_key": "temp_key: OUT-TMP",
        "m1temp_key": "m1temp_key: T_M1AIR",
        "geoelev": "geoelev: 2375",
        "geoelev_key": "geoelev_key: NONE",
        "longitude": "longitude: -70.69239",
        "longitude_key": "longitude_key: NONE",
        "latitude": "latitude: -29.01418",
        "latitude_key": "latitude_key: NONE",
        "slitw": "slitw: 0.29",
        "pixsc": "pixsc: 0.144",
    }
    
    # Leer el archivo
    with open(ruta_nueva, "r") as f:
        lineas = f.readlines()
    
    # Modificar las líneas
    with open(ruta_nueva, "w") as f:
        for linea in lineas:
            # Ignorar líneas vacías o comentarios
            if linea.strip() == "" or linea.strip().startswith("#"):
                f.write(linea)
                continue
            
            # Verificar si la línea corresponde a un parámetro en el diccionario
            written = False
            for parametro, nuevo_valor in modificar.items():
                # Comparar solo la parte inicial de la línea (antes del valor)
                if linea.strip().startswith(parametro + ":") or linea.strip() == parametro:
                    f.write(nuevo_valor + "\n")
                    written = True
                    break
            if not written:
                f.write(linea)
    
    return ruta_nueva
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def ejec_molecfit_o_calctrans(ruta, ruta_molecfit):
    import subprocess
    import os

    proc = subprocess.run([ruta_molecfit, ruta], capture_output = True,text = True)
    if "molecfit" in os.path.basename(ruta_molecfit):
        print(f"se aplica molecfit a {ruta}")
    else:
        print(f"se aplica calctrans a {ruta}")
        
    print("Salida estándar:\n", proc.stdout)
    # print("Errores:\n", proc.stderr)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def visualizar(ruta_original,ruta_modificado):
    from astropy.io import fits
    import matplotlib.pyplot as plt
    from astropy.wcs import WCS
    import numpy as np
    import os 
    import re 
    
    with fits.open(ruta_modificado) as hdul:
        HDR = hdul[1].header
        datos = hdul[1].data
        long_onda = datos['LAMBDA']#*1e4 descomentar para pasar a angstroms
        trans_curv = datos['MTRANS']
        flujo = datos['CFLUX']
    with fits.open(ruta_original) as hdul:
        HDR = hdul[0].header
        data_ = hdul[0].data
        n_pix = data_.shape[0]
        w = WCS(HDR, naxis = 1)
        long_ond2 = w.pixel_to_world(np.arange(n_pix)).value

    
    match = re.search(r'_m(\d+)_',ruta_original)
    orden = match.group(1)
    
    # dividir los ordenes 
    min_lam = np.min(long_onda)
    max_lam = np.max(long_onda)
    bins = np.linspace(min_lam, max_lam, 5)  # 5 puntos para 4 intervalos
    
    # Crear figura con 4 subplots apilados - SIN sharex
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 12))  # sharex eliminado
    axes = [ax1, ax2, ax3, ax4]

    # Graficar cada parte
    for i, ax in enumerate(axes):
        start_lam = bins[i]
        end_lam = bins[i + 1]
        mask = (long_onda >= start_lam) & (long_onda <= end_lam)
        ax.plot(long_onda[mask], data_[mask], color='blue', lw=1, label = "flujo sin calctrans")
        ax.plot(long_onda[mask],flujo[mask],color='red', lw=1, label="flujo de calctrans")
        ax.plot(long_onda[mask],trans_curv[mask], color= 'green', lw = 1, label= "mejor curva de transmision")
        ax.set_ylabel('Flujo normalizado')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True)
        # Opcional: ajustar los límites para que coincidan exactamente con el rango
        ax.set_xlim(start_lam, end_lam)

    # Etiqueta del eje X solo para el último subplot
    ax4.set_xlabel('Longitud de onda (µm)')
    fig.suptitle(f'Espectro corregido - Orden m{orden} ', fontsize=14)

    # Ajustar espaciado
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # ax.set_xlabel('longitud de onda (en micrometros)')
    # guardar las imagenes
    carpeta_img = "/home/nacho/molecfit_test/imagenes_VACstar9/"
    os.makedirs(carpeta_img,exist_ok=True)
    nombre_img = f"orden m{orden}.png"
    
    ruta_img = os.path.join(carpeta_img,nombre_img)
    plt.savefig(ruta_img,format="png",dpi = 150)
    plt.show()
    plt.close(fig)
    
    #vamos a encontrar las diferencias entre los datos de los archivos
    diferencias = flujo == data_
    print(diferencias)
    if np.any(~diferencias):
        print('se encontraron diferencias')
    else:
        print('no se encontraron diferencias')
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def aplicar_todo(ruta_entrada, ruta_salida_modificado, ruta_par, ruta_molecfit, ruta_calctrans ):
    import os
    nuevo_path = "/home/nacho/molecfit_test/mis_outputs_VACstar9/"
    print("Iniciando flujo completo de molecfit :)")
    
    # Paso 1: Editar el archivo FITS
    print("\n[1/4] Editando archivo FITS...")
    ruta_fits_mod = editar_fits_(ruta_entrada, ruta_salida_modificado)
    
    # Paso 2: Crear archivo .par
    print("\n[2/4] Creando archivo .par...")
    ruta_par_creado = archivo_par_2(ruta_fits_mod, ruta_par)
    
    # Paso 3: Ejecutar molecfit 
    print("\n[3/4] Ejecutando molecfit...")
    ejec_molecfit_o_calctrans(ruta_par_creado, ruta_molecfit)
    
    # paso 4: ejecutar calctrans
    print("\n[4/5] Ejecutando calctrans...")
    ejec_molecfit_o_calctrans(ruta_par_creado, ruta_calctrans)

    # Paso 5: Visualizar resultados (solo si se generó un archivo *_calctrans.fits)
    print("\n[5/5] Visualizando resultados...")
    ruta_calctrans = os.path.splitext(ruta_par_creado)[0] + ".par_tac.fits"
    if os.path.exists(ruta_calctrans):
        visualizar(ruta_fits_mod, ruta_calctrans)
    else:
        print(f"No se encontró el archivo {ruta_calctrans}, se omite la visualización.")
    
    print("\n=== Flujo completo finalizado ===")
    return nuevo_path
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def aplicar_todo_todos(carpeta_entrada, carpeta_salida, ruta_par, ruta_molecfit, ruta_calctrans):
    """
    Aplica la secuencia completa (editar_fits_ -> archivo_par_2 -> ejec_molecfit_o_calctrans)
    a todos los archivos .fits dentro de una carpeta.

    Args:
        carpeta_entrada: Carpeta donde están los archivos FITS originales.
        carpeta_salida: Carpeta donde se guardarán los FITS modificados y resultados.
        ruta_par: Carpeta donde se encuentra el archivo .par base de molecfit.
    """
    import os, glob

    archivos = sorted(glob.glob(os.path.join(carpeta_entrada, "*.fits")))
    if not archivos:
        print(f"No se encontraron archivos FITS en {carpeta_entrada}")
        return []

    print(f"\n Se encontraron {len(archivos)} archivos FITS para procesar.\n")

    rutas_finales = []
    for i, ruta_fits in enumerate(archivos, start=1):
        print(f"\n=============================")
        print(f" Procesando archivo {i}/{len(archivos)}:")
        print(f" {os.path.basename(ruta_fits)}")
        print(f"=============================")

        try:
            ruta_final = aplicar_todo(ruta_fits, carpeta_salida, ruta_par,ruta_molecfit,ruta_calctrans)
            rutas_finales.append(ruta_final)
            print(f" Terminado: {os.path.basename(ruta_final)}")

        except Exception as e:
            print(f" Error procesando {ruta_fits}: {e}")
            continue

    print(f"\n Proceso completo. Archivos finales en: {carpeta_salida}\n")
    return rutas_finales
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def generar_excludes(ruta_fits, ruta_excludes, factor_sigma=2, plots=False):
    import os, glob, re
    import numpy as np
    import matplotlib.pyplot as plt
    from astropy.io import fits
    """
    Genera archivos .dat con las longitudes de onda a excluir por aberraciones.

    Parámetros:
    ruta_fits : str
        Carpeta donde están los archivos .par_tac.fits
    ruta_excludes : str
        Carpeta donde se guardarán los .dat
    factor_sigma : float
        Umbral de detección en unidades de la media del flujo
    plots : bool
        Si True, muestra los gráficos de cada orden

    Retorna:
    lista_archivos : list
        Lista con las rutas de los .dat creados/actualizados
    """
    os.makedirs(ruta_excludes, exist_ok=True)
    fits_calctrans = glob.glob(os.path.join(ruta_fits, '*.par_tac.fits'))
    archivos_creados = []

    for i in fits_calctrans:
        with fits.open(i) as hdul:
            datos = hdul[1].data
            long_onda = datos['LAMBDA']
            flujo = datos['CFLUX']

        # Detección de aberraciones
        media = np.mean(flujo)
        mask = np.abs(flujo - media) > factor_sigma * media
        print(media)
        # Crear archivo .dat
        base = os.path.basename(i).replace('.par_tac.fits', '_exclude.dat')
        ruta_dat = os.path.join(ruta_excludes, base)
        
        # Leer rangos existentes si el archivo ya existe
        rangos_existentes = []
        if os.path.exists(ruta_dat):
            with open(ruta_dat, 'r') as f:
                for linea in f:
                    linea = linea.strip()
                    if linea and not linea.startswith('#'):
                        partes = linea.split()
                        if len(partes) >= 2:
                            try:
                                min_exist = float(partes[0])
                                max_exist = float(partes[1])
                                rangos_existentes.append((min_exist, max_exist))
                            except ValueError:
                                continue
        
        nuevo_rango = None
        if np.any(mask):
            min_long = np.min(long_onda[mask]) - 0.0002
            max_long = np.max(long_onda[mask]) + 0.0002
            # Limitar al rango válido
            min_long = max(min_long, np.min(long_onda))
            max_long = min(max_long, np.max(long_onda))
            
            nuevo_rango = (min_long, max_long)
            
            # Combinar rangos existentes con el nuevo
            todos_los_rangos = rangos_existentes + [nuevo_rango]
            
            # Escribir todos los rangos al archivo
            with open(ruta_dat, 'w') as f:
                for rango in todos_los_rangos:
                    f.write(f"{rango[0]:.6f} {rango[1]:.6f}\n")
            
            archivos_creados.append(ruta_dat)
            print(f"Actualizado {ruta_dat} con {len(todos_los_rangos)} rangos")
        else:
            # Si no hay nuevas aberraciones, mantener los rangos existentes
            if rangos_existentes:
                with open(ruta_dat, 'w') as f:
                    for rango in rangos_existentes:
                        f.write(f"{rango[0]:.6f} {rango[1]:.6f}\n")
                archivos_creados.append(ruta_dat)
                print(f"Mantenidos {len(rangos_existentes)} rangos existentes en {ruta_dat}")
            else:
                print(f"Orden {os.path.basename(i)}: sin aberraciones detectadas")

        # Graficar si se desea
        if plots:
            match = re.search(r'_m(\d+)_', i)
            orden = match.group(1) if match else "?"
            
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(long_onda, flujo, 'b-', label='Flujo')
            
            # Mostrar todos los rangos de exclusión
            rangos_a_mostrar = rangos_existentes
            if nuevo_rango:
                rangos_a_mostrar = rangos_existentes + [nuevo_rango]
            
            for j, (min_rango, max_rango) in enumerate(rangos_a_mostrar):
                color = 'red' if j == len(rangos_a_mostrar) - 1 and nuevo_rango else 'orange'
                label = 'Nueva zona excluida' if j == len(rangos_a_mostrar) - 1 and nuevo_rango else 'Zona excluida existente'
                ax.axvspan(min_rango, max_rango, color=color, alpha=0.3, label=label if j == 0 else "")
            
            titulo = f"Orden m{orden}: {len(rangos_a_mostrar)} rangos excluidos"
            ax.set_xlabel("Longitud de onda (µm)")
            ax.set_ylabel("Flujo")
            ax.set_title(titulo)
            ax.legend()
            plt.tight_layout()
            plt.show()

    print(f"\nSe procesaron {len(archivos_creados)} archivos .dat en {ruta_excludes}")
    return archivos_creados
