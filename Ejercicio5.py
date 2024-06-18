# 5 Multiplique dos matrices sparce de más de 1000 filas y columnas de manera paralelo con Python (scypi y multprocessing) y
#c (openMP o MPI). Observación. El paralelismo se realiza por fila o columna, no una multiplicación directa y es a partir del sparce.
from multiprocessing import Pool, cpu_count


# Función para cargar una imagen y convertirla en escala de grises
def imagen_a_EscalaGrises(ruta_imagen):
    imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    return imagen

# Función para convertir una imagen en una matriz dispersa
def imagen_a_MatrizSparce(imagen):
    sparse_matrix = csr_matrix(imagen)
    return sparse_matrix

# Función para redimensionar una imagen si es menor a 1000x1000
def rdimencionarImagen(image, min_size=(1000, 1000)):
    if image.shape[0] < min_size[0] or image.shape[1] < min_size[1]:
        return cv2.resize(image, min_size)
    return image

# Rutas de las imágenes en Google Drive
ruta_imagen1 = '/content/drive/MyDrive/inf_lic_silva/leon1.jpg'
ruta_imagen2 = '/content/drive/MyDrive/inf_lic_silva/leon2.jpg'

# Cargar las imágenes en escala de grises
imagen1 = imagen_a_EscalaGrises(ruta_imagen1)
imagen2 = imagen_a_EscalaGrises(ruta_imagen2)

# Redimensionar las imágenes si es necesario
imagen1 = rdimencionarImagen(imagen1)
imagen2 = rdimencionarImagen(imagen2)

# Convertir las imágenes en matrices dispersas
matrizSparce1 = imagen_a_MatrizSparce(imagen1)
matrizSparce2 = imagen_a_MatrizSparce(imagen2)

# Verificar el tamaño de las matrices dispersas
print(f"Tamaño de la matriz dispersa 1: {matrizSparce1.shape}")
print(f"Tamaño de la matriz dispersa 2: {matrizSparce2.shape}")

# Asegurarse de que las matrices tengan el mismo tamaño para la multiplicación
if matrizSparce1.shape != matrizSparce2.shape:
    print("Las matrices no tienen el mismo tamaño. Redimensionando la segunda matriz para que coincida con la primera.")
    redimencionarImg2 = cv2.resize(imagen2, (imagen1.shape[1], imagen1.shape[0]))
    matrizSparce2 = imagen_a_MatrizSparce(redimencionarImg2)

# Multiplicación paralela por filas usando multiprocessing.Pool y cpu_count
def multiplicarFila(indice_fila):
    res = matrizSparce1.getrow(indice_fila).dot(matrizSparce2)
    return res

if __name__ == "__main__":
    # Determinar el número de procesos a utilizar (por defecto, el número de núcleos de CPU)
    num_procedos = cpu_count()
    print(f"Número de procesos a utilizar: {num_procedos}")

    # Crear un pool de procesos
    with Pool(processes=num_procedos) as pool:
        # Aplicar la función multiplicarFila a cada índice de fila en paralelo
        resultadoParalelo = pool.map(multiplicarFila, range(matrizSparce1.shape[0]))

    # Concatenar resultados en una matriz dispersa COO
    resultadoFila = coo_matrix((0, matrizSparce2.shape[1]))  # Inicializar matriz dispersa COO vacía
    for res in resultadoParalelo:
        resultadoFila = vstack([resultadoFila, res])

    # Convertir el resultado a una matriz densa para visualizar (opcional)
    ResultadoDenso = resultadoFila.toarray()

    # Imprimir resultados
    print("\nMatriz Dispersa 1 (CSR):")
    print(matrizSparce1)

    print("\nMatriz Dispersa 2 (CSR):")
    print(matrizSparce2)

    print("\nResultado de la multiplicación paralela (Matriz Dispersa):")
    print(resultadoFila)

    print("\nResultado de la multiplicación paralela (Matriz Densa):")
    print(ResultadoDenso)

    # Mostrar las imágenes originales y la matriz resultante
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title('imagen 1')
    plt.imshow(imagen1, cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title('imagen 2')
    plt.imshow(imagen2, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title('Producto de Matrices')
    plt.imshow(ResultadoDenso, cmap='gray')

    plt.show()