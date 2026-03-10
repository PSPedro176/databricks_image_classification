# Databricks notebook source
# MAGIC %md
# MAGIC # Image Recommendation System — Data Preparation
# MAGIC
# MAGIC This notebook downloads the Fashion MNIST dataset, converts the raw binary files
# MAGIC into Delta tables in Unity Catalog, and verifies the result.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Table of Contents
# MAGIC
# MAGIC 1. **Setup** — Imports, configure Unity Catalog
# MAGIC 2. **Data Preparation** — Download Fashion MNIST, convert to Delta tables

# COMMAND ----------

import os

import numpy as np
import pandas as pd
from PIL import Image

# COMMAND ----------

# DBTITLE 1,Configure Unity Catalog and Volumes
catalog = "image_rec_classic_catalog"
schema = "image_recommendation"
volume = "data"

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.{volume}")

volume_path = f"/Volumes/{catalog}/{schema}/{volume}"

print(f"Using catalog: {catalog}")
print(f"Using schema: {catalog}.{schema}")
print(f"Volume path: {volume_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download Fashion MNIST
# MAGIC
# MAGIC Fetch data from the official Zalando Fashion MNIST repository on GitHub.

# COMMAND ----------

# MAGIC %sh
# MAGIC cd /databricks/driver
# MAGIC wget -q -O test_labels.gz https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-labels-idx1-ubyte.gz
# MAGIC wget -q -O test_images.gz https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-images-idx3-ubyte.gz
# MAGIC wget -q -O train_images.gz https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-images-idx3-ubyte.gz
# MAGIC wget -q -O train_labels.gz https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-labels-idx1-ubyte.gz
# MAGIC
# MAGIC ls -lh /databricks/driver/{train,test}_{images,labels}.gz

# COMMAND ----------

# MAGIC %md
# MAGIC ### Convert raw files to Delta tables
# MAGIC
# MAGIC Adapted from <https://pjreddie.com/projects/mnist-in-csv/>.

# COMMAND ----------

import gzip

datasets = [
    ["test_images.gz", "test_labels.gz", f"{catalog}.{schema}.fmnist_test_data", 10_000, "test"],
    ["train_images.gz", "train_labels.gz", f"{catalog}.{schema}.fmnist_train_data", 60_000, "train"],
]


def convert(
    imgf: str,
    labelf: str,
    outf: str,
    n: int,
    image_dir: str,
) -> None:
    """Read gzipped MNIST binary files, save PNGs to the Volume, and write a Delta table."""
    img_out = f"{volume_path}/images/{image_dir}"
    os.makedirs(img_out, exist_ok=True)

    with gzip.open(f"/databricks/driver/{imgf}", "rb") as img_fp, \
         gzip.open(f"/databricks/driver/{labelf}", "rb") as lbl_fp:
        img_fp.read(16)
        lbl_fp.read(8)
        rows: list[list[int]] = []
        for image_id in range(n):
            label = ord(lbl_fp.read(1))
            pixels = [ord(img_fp.read(1)) for _ in range(784)]
            # Save as PNG to the Volume
            arr = np.array(pixels, dtype=np.uint8).reshape(28, 28)
            Image.fromarray(arr, mode="L").save(f"{img_out}/{image_id}.png")
            rows.append([image_id, label] + pixels)

    columns = ["image_id", "label"] + [f"pixel_{i}" for i in range(784)]
    df = pd.DataFrame(rows, columns=columns)
    spark_df = spark.createDataFrame(df)
    spark_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(outf)

# COMMAND ----------

for dataset in datasets:
    convert(dataset[0], dataset[1], dataset[2], dataset[3], dataset[4])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Class labels
# MAGIC
# MAGIC | Label | Description |
# MAGIC |-------|-------------|
# MAGIC | 0     | T-shirt/top |
# MAGIC | 1     | Trouser     |
# MAGIC | 2     | Pullover    |
# MAGIC | 3     | Dress       |
# MAGIC | 4     | Coat        |
# MAGIC | 5     | Sandal      |
# MAGIC | 6     | Shirt       |
# MAGIC | 7     | Sneaker     |
# MAGIC | 8     | Bag         |
# MAGIC | 9     | Ankle boot  |

# COMMAND ----------

# MAGIC %md
# MAGIC ### Verification — read tables back and print shapes

# COMMAND ----------

train_df = spark.table(f"{catalog}.{schema}.fmnist_train_data")
test_df = spark.table(f"{catalog}.{schema}.fmnist_test_data")

print(f"Train: {train_df.count()} rows, columns: {train_df.columns[:5]}...")
print(f"Test:  {test_df.count()} rows, columns: {test_df.columns[:5]}...")

import subprocess
result = subprocess.run(["ls", f"{volume_path}/images/train/"], capture_output=True, text=True)
print(f"Train images saved: {len(result.stdout.strip().split())} files")
result = subprocess.run(["ls", f"{volume_path}/images/test/"], capture_output=True, text=True)
print(f"Test images saved:  {len(result.stdout.strip().split())} files")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the [Databricks License](https://databricks.com/db-license-source). All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library / data source | description | license | source |
# MAGIC |---|---|---|---|
# MAGIC | tensorflow | package | Apache 2.0 | https://github.com/tensorflow/tensorflow/blob/master/LICENSE |
# MAGIC | fashion-mnist | dataset | MIT | https://github.com/zalandoresearch/fashion-mnist/blob/master/LICENSE |
