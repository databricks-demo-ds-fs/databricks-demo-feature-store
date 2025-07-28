# Databricks notebook source
# MAGIC %pip install faker

# COMMAND ----------

import pyspark.sql.functions as f
from pyspark.sql.window import Window
import numpy as np
import pandas as pd
import pyspark
from pyspark.pandas import DataFrame
from pyspark.sql.types import *
from faker import Faker
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from tqdm.auto import tqdm

# COMMAND ----------

np.random.seed(42)

# COMMAND ----------

# MAGIC %md
# MAGIC Variables Globales

# COMMAND ----------

TOTAL_CLIENTES = 10000  # Número de clientes
MESES_HISTORIA = 24         # Número de meses de historial
PROPORCION_MOROSOS = 0.15    # Proporción de morosos (~15%)
# DATA_LAKE_PATH = "dbfs:/mnt/data_lake"  # Ruta en Databricks (DBFS)
faker = Faker()
np.random.seed(42)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tabla para clientes

# COMMAND ----------

print("Generando tabla: clientes...")
id_cliente = np.arange(1, TOTAL_CLIENTES + 1)
edades = np.random.randint(18, 80, TOTAL_CLIENTES)
generos = np.random.choice(["M", "F"], TOTAL_CLIENTES)
estado_civil = np.random.choice(["Soltero", "Casado", "Divorciado"], TOTAL_CLIENTES)
nivel_educativo = np.random.choice(["Primaria", "Secundaria", "Universitario", "Postgrado"], TOTAL_CLIENTES)
ingresos = np.round(np.random.normal(3000, 2500, TOTAL_CLIENTES).clip(300, 20000), 2)
zona_residencia = np.random.choice(["Urbano", "Rural"], TOTAL_CLIENTES, p=[0.8, 0.2])
ciudades = [faker.city() for _ in range(TOTAL_CLIENTES)]
fecha_apertura = [faker.date_between(start_date="-30y", end_date="today") for _ in range(TOTAL_CLIENTES)]

# Asignación de segmentos según ingresos
segmento_cliente = np.where(
    ingresos >= 10000, "ELITE",
    np.where(
        ingresos >= 4000, "PREMIUM",
        np.where(
            ingresos >= 2500, "PRESTIGE",
            "SILVER"
        )
    )
)
#consolidar
clientes_df = pd.DataFrame({
    "id_cliente": id_cliente,
    "fecha_nacimiento": [faker.date_of_birth(minimum_age=18, maximum_age=80) for _ in range(TOTAL_CLIENTES)],
    "genero": generos,
    "estado_civil": estado_civil,
    "nivel_educativo": nivel_educativo,
    "ingresos_mensuales": ingresos,
    "zona_residencia": zona_residencia,
    "ciudad": ciudades,
    "fecha_apertura": fecha_apertura,
    "segmento_cliente": segmento_cliente
})
# Convertir a Spark DataFrame
clientes_spark_df = spark.createDataFrame(clientes_df)
del clientes_df

# COMMAND ----------

spark.sql("CREATE DATABASE IF NOT EXISTS demo_db")

# COMMAND ----------

clientes_spark_df.write.saveAsTable(
    name="demo_db.clientes",
    path="dbfs:/mnt/SANDBOX/WORKSPACE/SVELIZA/DEMO_PROJECT/clientes",
    mode="overwrite"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tabla de Productos

# COMMAND ----------

id_cliente = np.arange(1, TOTAL_CLIENTES + 1)

# COMMAND ----------

print("Generando tabla: productos...")

productos = [
    ("tarjeta de credito", 0.7, 500, 20000),
    ("credito hipotecario", 0.2, 100000, 2000000),
    ("credito productivo", 0.5, 10000, 500000),
    ("credito vehicular", 0.15, 50000, 800000)
]

historia = []
np.random.seed(42)
fecha_base = pd.Timestamp("2023-07-01")

for idx in tqdm(range(TOTAL_CLIENTES)):
    productos_cliente = []
    for nombre, prob, min_monto, max_monto in productos:
        if np.random.rand() < prob:
            meses = np.random.randint(3, 25)
            inicio = fecha_base + pd.DateOffset(months=np.random.randint(0, 25 - meses))
            deuda = np.round(np.random.uniform(min_monto, max_monto), 2)
            for m in range(meses):
                fecha = inicio + pd.DateOffset(months=m)
                # Simular amortización simple: la deuda disminuye linealmente cada mes
                deuda_mes = max(0, np.round(deuda * (1 - m / meses), 2))
                productos_cliente.append({
                    "id_cliente": id_cliente[idx],
                    "producto": nombre,
                    "fecha": fecha,
                    "monto": deuda_mes
                })
    # Calcular total_productos por mes
    if productos_cliente:
        df_tmp = pd.DataFrame(productos_cliente)
        for fecha, grupo in df_tmp.groupby("fecha"):
            total_productos = grupo.shape[0]
            for row in grupo.itertuples(index=False):
                historia.append({
                    "id_cliente": row.id_cliente,
                    "producto": row.producto,
                    "fecha": row.fecha,
                    "monto": row.monto,
                    "total_productos": total_productos
                })

productos_df = pd.DataFrame(historia)
productos_spark_df = spark.createDataFrame(productos_df)
del productos_df

# COMMAND ----------

productos_spark_df.write.saveAsTable(
    name="demo_db.productos",
    path="dbfs:/mnt/SANDBOX/WORKSPACE/SVELIZA/DEMO_PROJECT/productos",
    mode="overwrite",
    overwriteSchema=True,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pagos (HISTORIAL MENSUAL)

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np

valores = np.round(np.exp(np.random.uniform(np.log(5000), np.log(40000), TOTAL_CLIENTES)), 2)
plt.hist(valores, bins=30)
plt.title("Distribución de creditos asignados")
plt.xlabel("Valor")
plt.ylabel("Frecuencia")
plt.show()

# COMMAND ----------

print("Generando tabla: pagos...")

pagos_list = []
periodos = pd.date_range(end=datetime.today(), periods=MESES_HISTORIA, freq='M')

# Definir fechas de apertura y cierre por cliente
fecha_apertura = pd.to_datetime(np.random.choice(periodos, TOTAL_CLIENTES))
fecha_cierre = fecha_apertura + pd.to_timedelta(np.random.randint(12, 36, TOTAL_CLIENTES) * 30, unit='D')

# Calificación inicial y contador de pagos buenos
calificacion_clientes = np.random.choice(['A', 'B', 'C'], size=TOTAL_CLIENTES)
meses_buen_pago = np.zeros(TOTAL_CLIENTES, dtype=int)

# Mapeo para bajar y subir calificación
bajar = {"A": "B", "B": "C", "C": "D", "D": "E", "E": "E"}
subir = {"E": "D", "D": "C", "C": "B", "B": "A", "A": "A"}

# Valor total del crédito por cliente
valor_total_credito = np.round(np.exp(np.random.uniform(np.log(5000), np.log(40000), TOTAL_CLIENTES)), 2)
# Meses de crédito por cliente
meses_credito = ((fecha_cierre - fecha_apertura) / np.timedelta64(1, 'M')).astype(int) + 1
# Cuota mensual por cliente
cuota_mensual = np.round(valor_total_credito / meses_credito, 2)

# --------------------------
# Aplicar PROPORCION_MOROSOS
# --------------------------
morosos_idx = np.random.choice(
    TOTAL_CLIENTES,
    size=int(TOTAL_CLIENTES * PROPORCION_MOROSOS),
    replace=False
)
buenos_idx = np.setdiff1d(np.arange(TOTAL_CLIENTES), morosos_idx)

for idx, cliente in tqdm(enumerate(id_cliente), total=len(id_cliente)):
    pagos_cliente = []
    ultima_fecha_pago = None
    primera_fecha_no_pago = None
    saldo_anterior = valor_total_credito[idx]

    for periodo in periodos:
        if periodo < fecha_apertura[idx] or periodo > fecha_cierre[idx]:
            continue  # Solo considerar meses dentro del periodo de vida del crédito

        # Probabilidad diferenciada según tipo de cliente
        if idx in morosos_idx:
            cuota_pagada = np.random.choice([True, False], p=[0.75, 0.25])  # morosos: 25% de chance de no pagar
        else:
            cuota_pagada = np.random.choice([True, False], p=[0.98, 0.02])  # buenos: 2% de chance de no pagar

        pago_minimo = np.random.choice([True, False], p=[0.4, 0.6])

        if cuota_pagada:
            monto_pagado = cuota_mensual[idx]
            if primera_fecha_no_pago is not None:
                dias_mora = (periodo - primera_fecha_no_pago).days
                primera_fecha_no_pago = None
            else:
                dias_mora = 0
            ultima_fecha_pago = periodo
            meses_buen_pago[idx] += 1
            if meses_buen_pago[idx] >= 2:
                calificacion_clientes[idx] = subir[calificacion_clientes[idx]]
        else:
            monto_pagado = 0
            if ultima_fecha_pago is not None:
                periodo_date = periodo.date()
                ultima_fecha_pago_date = ultima_fecha_pago.date()
                diff = relativedelta(periodo_date, ultima_fecha_pago_date)
                diff_months = diff.years * 12 + diff.months
                diff_days = diff.days
                if diff_months > 1 or (diff_months == 1 and diff_days > 31):
                    dias_mora = (periodo_date - primera_fecha_no_pago.date()).days if primera_fecha_no_pago is not None else diff_days
                elif diff_months < 1 or (diff_months == 1 and diff_days <= 31):
                    primera_fecha_no_pago = periodo
                    dias_mora = 0
                else:
                    dias_mora = (periodo_date - primera_fecha_no_pago.date()).days if primera_fecha_no_pago is not None else diff_days
            else:
                periodo_date = periodo.date()
                fecha_apertura_date = fecha_apertura[idx].date()
                diff = relativedelta(periodo_date, fecha_apertura_date)
                diff_months = diff.years * 12 + diff.months
                diff_days = diff.days
                if diff_months > 1 or (diff_months == 1 and diff_days > 31):
                    dias_mora = (periodo_date - primera_fecha_no_pago.date()).days if primera_fecha_no_pago is not None else diff_days
                elif diff_months < 1 or (diff_months == 1 and diff_days <= 31):
                    primera_fecha_no_pago = periodo
                    dias_mora = 0
                else:
                    dias_mora = (periodo_date - primera_fecha_no_pago.date()).days if primera_fecha_no_pago is not None else diff_days
            calificacion_clientes[idx] = bajar[calificacion_clientes[idx]]
            meses_buen_pago[idx] = 0

        saldo_total_credito = round(max(saldo_anterior - monto_pagado, 2), 2)
        pagos_cliente.append({
            "id_cliente": cliente,
            "periodo": periodo.strftime("%Y-%m"),
            "saldo_total_credito": saldo_total_credito,
            "cuota_pagada": cuota_pagada,
            "monto_pagado": monto_pagado,
            "dias_mora": dias_mora,
            "pago_minimo": pago_minimo,
            "calificacion_sistema": calificacion_clientes[idx],
            "fecha_apertura": fecha_apertura[idx].date(),
            "fecha_cierre": fecha_cierre[idx].date(),
            "valor_total_credito": valor_total_credito[idx],
            "cuota_mensual": cuota_mensual[idx],
            "meses_credito": meses_credito[idx],
        })
        saldo_anterior = saldo_total_credito

    pagos_list.extend(pagos_cliente)

tabla_pagos_df = pd.DataFrame(pagos_list)
tabla_pagos_spark = spark.createDataFrame(tabla_pagos_df)
del tabla_pagos_df

# COMMAND ----------

tabla_pagos_spark.write.saveAsTable(
    name="demo_db.pagos",
    path="dbfs:/mnt/SANDBOX/WORKSPACE/SVELIZA/DEMO_PROJECT/pagos",
    mode="overwrite",
    overwriteSchema=True,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tabla Buró

# COMMAND ----------

from pyspark.sql import functions as f

pagos_df = spark.table("demo_db.pagos")

buro_spark_df = pagos_df.select(
    "id_cliente",
    "periodo",
    "calificacion_sistema",
    "saldo_total_credito"
).withColumn(
    "buro_score",
    f.when(f.col("calificacion_sistema") == "A", (f.rand() * 200 + 750).cast("int"))
    .when(f.col("calificacion_sistema") == "B", (f.rand() * 100 + 650).cast("int"))
    .when(f.col("calificacion_sistema") == "C", (f.rand() * 100 + 550).cast("int"))
    .when(f.col("calificacion_sistema") == "D", (f.rand() * 100 + 450).cast("int"))
    .otherwise((f.rand() * 150 + 300).cast("int"))
).withColumn(
    "consultas_buro_12m", (f.rand() * 5).cast("int")
).withColumn(
    "deuda_total_bancos", f.round(
      f.col("saldo_total_credito") +
      (f.exp(f.rand() * f.log(f.lit(10000) - f.lit(500) + 1)) + 500 - 1), 
      2
    )
).drop("saldo_total_credito")

# COMMAND ----------

buro_spark_df.write.saveAsTable(
    name="demo_db.buro_credito",
    path="dbfs:/mnt/SANDBOX/WORKSPACE/SVELIZA/DEMO_PROJECT/buro_credito",
    mode="overwrite",
    overwriteSchema=True,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Transacciones (HISTORIAL MENSUAL CONSOLIDADO)

# COMMAND ----------

print("Generando tabla: transacciones...")
transacciones_list = []

for periodo in periodos:
    numero_transacciones = np.random.poisson(15, TOTAL_CLIENTES)
    monto_total_transacciones = np.round(np.random.gamma(shape=2, scale=200, size=TOTAL_CLIENTES), 2)
    recencia_ultima_transaccion = np.random.randint(0, 31, TOTAL_CLIENTES)

    transacciones_df = pd.DataFrame({
        "id_cliente": id_cliente,
        "periodo": periodo.strftime("%Y-%m"),
        "numero_transacciones": numero_transacciones,
        "monto_total_transacciones": monto_total_transacciones,
        "recencia_ultima_transaccion": recencia_ultima_transaccion
    })
    transacciones_list.append(transacciones_df)

# Consolidar en un solo DataFrame
tabla_transacciones_df = pd.concat(transacciones_list, ignore_index=True)
tabla_transacciones_spark = spark.createDataFrame(tabla_transacciones_df)
del tabla_transacciones_df

# COMMAND ----------

tabla_transacciones_spark.write.saveAsTable(
    name="demo_db.transacciones",
    path="dbfs:/mnt/SANDBOX/WORKSPACE/SVELIZA/DEMO_PROJECT/transacciones",
    mode="overwrite"
)