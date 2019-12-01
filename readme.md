Implementaciones de sketch countmin-cu utilizando AVX y CUDA.

## Compilación

Para compilar usar `make`. Usar `make DEBUG=1` para compilar versión de
depuración.

## Script para ejecutar las pruebas

El script `run-eval.py` ejecuta automaticamente pruebas sobre las distintas
implementaciones. Este script requiere un archivo `datasets.json` que describa
los sets de datos a utilizar en las pruebas el cual tiene la siguiente
estructura

```json
{
    "<nombre_del_set_datos>": {
        "tags": ["default"],  // Etiquetas para agrupar sets de datos
        "test_file": "ruta/al/set/de/prueba",
        "control_file": "ruta/al/set/de/prueba",
        "first_length": <menor_largo_de_kmer>,  // i.e. 10
        "thresholds": [<valores>, <de>, <umbrales>]
    },
    // ...
}
```

El script se debe ejecutar con python3

```
python3 run_eval.py runs [--cuda-metrics] [--program-type type1 [...]] \
[--data-tags tag1 [...]]
```

Los argumentos son:

* `runs`: número de ejecuciones para cada caso
* `--cuda-metrics`: activa la recolección de metricas adicionales para los
programas que utilizan CUDA
* `--program-type`: selecciona los programas a ejecutar. Los posibles valores
son `avx`, `cuda`, `default`. En caso de omitir esta opción se ejecutan todos
los programas
* `--data-tags`: selecciona los sets de datos a utilizar. Las etiquetas
corresponden a aquellas especificadas en el valor "tags" de cada set de dato
en el archivo `datasets.json`. Por omisión se ejecutan los sets de datos con
etiqueta "default".

Por ejemplo, el siguiente comando ejecuta unicamente el programa baseline 1 vez
para cada set de datos

```
python3 run_eval.py 1 --program-tags default
```
