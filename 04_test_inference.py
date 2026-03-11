# Databricks notebook source
# MAGIC %md
# MAGIC # Test Inference — Image Recommender
# MAGIC
# MAGIC This notebook verifies the trained model works via two methods:
# MAGIC 1. **Local inference** — Load the pyfunc model from Unity Catalog via MLflow
# MAGIC 2. **Serving endpoint** — Query the Model Serving REST API

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

import numpy as np
import pandas as pd
import mlflow
import requests
import json
from matplotlib import pyplot as plt
from PIL import Image
from mlflow.tracking import MlflowClient

mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

catalog = "image_rec_classic_catalog"
schema = "image_recommendation"
model_name = f"{catalog}.{schema}.image_recommender"
serving_endpoint_name = "image-recommender"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sample Image ID
# MAGIC
# MAGIC We pass an integer `image_id` corresponding to a training image (0–59,999).

# COMMAND ----------

sample_image_id = 42
sample_input = pd.DataFrame({"image_id": [sample_image_id]})

# COMMAND ----------

# MAGIC %md
# MAGIC ### Display the input image

# COMMAND ----------

input_img = Image.open(f"/Volumes/{catalog}/{schema}/data/images/train/{sample_image_id}.png")

plt.figure(figsize=(3, 3))
plt.imshow(input_img, cmap="Greys")
plt.title(f"Input image (id={sample_image_id})")
plt.axis("off")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section A — Load Model from Unity Catalog

# COMMAND ----------

# Use the latest model version (deployed by notebook 03)
client = MlflowClient(registry_uri="databricks-uc")
latest_version = max(
    client.search_model_versions(f"name='{model_name}'"),
    key=lambda v: int(v.version),
).version

model_uri = f"models:/{model_name}/{latest_version}"
print(f"Loading model from: {model_uri}")
loaded_model = mlflow.pyfunc.load_model(model_uri)

# COMMAND ----------

predictions = loaded_model.predict(sample_input)
print("Predictions (5 recommended image IDs):")
display(predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section B — Query Model Serving Endpoint

# COMMAND ----------

workspace_url = (
    dbutils.notebook.entry_point
    .getDbutils().notebook().getContext().apiUrl().get()
)
token = (
    dbutils.notebook.entry_point
    .getDbutils().notebook().getContext().apiToken().get()
)

# COMMAND ----------

endpoint_url = f"{workspace_url}/serving-endpoints/{serving_endpoint_name}/invocations"
print(f"Querying: {endpoint_url}")

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json",
}

payload = {"dataframe_records": [{"image_id": sample_image_id}]}

response = requests.post(endpoint_url, headers=headers, json=payload)

print(f"Status: {response.status_code}")
print(json.dumps(response.json(), indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualize Results
# MAGIC
# MAGIC Load the input image and all 5 recommended images from the Volume PNGs.

# COMMAND ----------

# Get recommended IDs from local prediction
recommended_ids = predictions["recommended_image_id"].tolist()

fig, axes = plt.subplots(1, 6, figsize=(18, 3))

# Input image
input_img = Image.open(f"/Volumes/{catalog}/{schema}/data/images/train/{sample_image_id}.png")
axes[0].imshow(input_img, cmap="Greys")
axes[0].set_title(f"Input\nid={sample_image_id}")
axes[0].axis("off")

# Recommended images
for i, rec_id in enumerate(recommended_ids):
    rec_img = Image.open(f"/Volumes/{catalog}/{schema}/data/images/train/{rec_id}.png")
    axes[i + 1].imshow(rec_img, cmap="Greys")
    axes[i + 1].set_title(f"Rec #{i+1}\nid={rec_id}")
    axes[i + 1].axis("off")

plt.suptitle("Image Recommendations", fontsize=14)
plt.tight_layout()
plt.show()
