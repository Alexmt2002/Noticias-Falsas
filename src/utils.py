import os
import time
import joblib


# =========================
# IMPRIMIR TÍTULOS
# =========================

def print_titulo(texto):

    print("\n========================")
    print(f" {texto}")
    print("========================")


# =========================
# VALIDAR TEXTO
# =========================

def validar_texto(texto):

    if not texto:
        return False

    if not isinstance(texto, str):
        return False

    if not texto.strip():
        return False

    return True


# =========================
# GUARDAR MODELO
# =========================

def guardar_modelo(modelo, ruta):

    os.makedirs(
        os.path.dirname(ruta),
        exist_ok=True
    )

    joblib.dump(
        modelo,
        ruta
    )

    print(f"\nModelo guardado en:\n{ruta}")


# =========================
# CARGAR MODELO
# =========================

def cargar_modelo(ruta):

    if not os.path.exists(ruta):

        raise FileNotFoundError(
            f"No existe el archivo:\n{ruta}"
        )

    modelo = joblib.load(ruta)

    return modelo


# =========================
# MOSTRAR TIEMPO
# =========================

def calcular_tiempo(inicio):

    fin = time.time()

    segundos = fin - inicio

    print(f"\nTiempo: {segundos:.2f} segundos")


# =========================
# MOSTRAR RESULTADO
# =========================

def mostrar_resultado(
    prediccion,
    confianza
):

    print_titulo("RESULTADO")

    print(f"\nPredicción: {prediccion}")

    print(f"Confianza: {confianza:.2f}%")


# =========================
# MOSTRAR MÉTRICAS
# =========================

def guardar_metricas(
    accuracy,
    reporte,
    ruta
):

    with open(
        ruta,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            f"Accuracy: {accuracy:.4f}\n\n"
        )

        f.write(reporte)

    print("\nMétricas guardadas")


# =========================
# TEXTO LARGO
# =========================

def recortar_texto(
    texto,
    max_chars=300
):

    if len(texto) <= max_chars:

        return texto

    return texto[:max_chars] + "..."


# =========================
# MOSTRAR ERROR
# =========================

def mostrar_error(error):

    print("\nERROR:")
    print(error)