#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <suitesparse/cs.h>

using namespace cv;

// Función para cargar una imagen y convertirla en escala de grises
Mat imagen_a_EscalaGrises(const char* ruta_imagen) {
    Mat imagen = imread(ruta_imagen, IMREAD_GRAYSCALE);
    return imagen;
}

// Función para redimensionar una imagen si es menor a 1000x1000
Mat redimensionarImagen(Mat image, Size min_size = Size(1000, 1000)) {
    if (image.rows < min_size.height || image.cols < min_size.width) {
        resize(image, image, min_size);
    }
    return image;
}

// Función para convertir una imagen en una matriz dispersa
cs* imagen_a_MatrizSparce(Mat imagen) {
    int rows = imagen.rows;
    int cols = imagen.cols;
    int nnz = countNonZero(imagen);
    cs* T = cs_spalloc(rows, cols, nnz, 1, 1);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (imagen.at<uchar>(i, j) != 0) {
                cs_entry(T, i, j, imagen.at<uchar>(i, j));
            }
        }
    }
    cs* A = cs_compress(T);
    cs_spfree(T);
    return A;
}

// Función para multiplicar una fila de una matriz dispersa
cs* multiplicarFila(cs* A, cs* B, int fila) {
    cs* C = cs_multiply(cs_transpose(cs_spalloc(1, A->n, A->n, 1, 0)), B);
    return C;
}

int main() {
    const char* ruta_imagen1 = "leon1.jpg";
    const char* ruta_imagen2 = "leon2.jpg";

    // Cargar las imágenes en escala de grises
    Mat imagen1 = imagen_a_EscalaGrises(ruta_imagen1);
    Mat imagen2 = imagen_a_EscalaGrises(ruta_imagen2);

    // Redimensionar las imágenes si es necesario
    imagen1 = redimensionarImagen(imagen1);
    imagen2 = redimensionarImagen(imagen2);

    // Convertir las imágenes en matrices dispersas
    cs* matrizSparce1 = imagen_a_MatrizSparce(imagen1);
    cs* matrizSparce2 = imagen_a_MatrizSparce(imagen2);

    // Verificar el tamaño de las matrices dispersas
    printf("Tamaño de la matriz dispersa 1: (%d, %d)\n", matrizSparce1->m, matrizSparce1->n);
    printf("Tamaño de la matriz dispersa 2: (%d, %d)\n", matrizSparce2->m, matrizSparce2->n);

    // Asegurarse de que las matrices tengan el mismo tamaño para la multiplicación
    if (matrizSparce1->m != matrizSparce2->m || matrizSparce1->n != matrizSparce2->n) {
        printf("Las matrices no tienen el mismo tamaño.\n");
        return -1;
    }

    // Multiplicación paralela por filas usando OpenMP
    cs* resultado = cs_spalloc(matrizSparce1->m, matrizSparce2->n, matrizSparce1->m * matrizSparce2->n, 1, 0);

    #pragma omp parallel for
    for (int i = 0; i < matrizSparce1->m; ++i) {
        cs* fila_resultado = multiplicarFila(matrizSparce1, matrizSparce2, i);
        #pragma omp critical
        cs_add(resultado, fila_resultado, 1.0, 1.0);
        cs_spfree(fila_resultado);
    }

    // Convertir el resultado a una matriz densa para visualizar (opcional)
    Mat ResultadoDenso(resultado->m, resultado->n, CV_8UC1, Scalar(0));
    for (int i = 0; i < resultado->nzmax; ++i) {
        ResultadoDenso.at<uchar>(resultado->i[i], resultado->p[i]) = resultado->x[i];
    }

    // Imprimir resultados
    printf("\nResultado de la multiplicación paralela (Matriz Densa):\n");
    std::cout << ResultadoDenso << std::endl;

    // Mostrar las imágenes originales y la matriz resultante
    imshow("Imagen 1", imagen1);
    imshow("Imagen 2", imagen2);
    imshow("Producto de Matrices", ResultadoDenso);
    waitKey(0);

    // Liberar memoria
    cs_spfree(matrizSparce1);
    cs_spfree(matrizSparce2);
    cs_spfree(resultado);

    return 0;
}
