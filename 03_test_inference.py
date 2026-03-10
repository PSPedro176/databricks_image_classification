# Databricks notebook source
# MAGIC %md
# MAGIC # Test Inference — Image Recommender
# MAGIC
# MAGIC This notebook verifies the trained model works via two methods:
# MAGIC 1. **Local inference** — Load the pyfunc model from Unity Catalog via MLflow
# MAGIC 2. **Serving endpoint** — Query the Model Serving REST API

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# COMMAND ----------

import pandas as pd
import mlflow

mlflow.set_registry_uri("databricks-uc")

model_name = "image_rec_classic_catalog.image_recommendation.image_recommender"
serving_endpoint_name = "image-recommender"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sample Image
# MAGIC
# MAGIC Base64-encoded 28x28 Fashion MNIST image (a coat).

# COMMAND ----------

sample_b64 = (
    "iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAAAAABXZoBIAAACN0lEQVR4nF3S"
    "z2vaYBgH8DdKNypDZYO2hxWnVDGiESMxIUoStGKCMWgwEpVEtKIyxR848Qet"
    "1FEHPRShp0FPu4wNdtoOg3Wn3Xvb/7Npq0Z93hBe3g/PA3m/AUCrkG/xrlkA"
    "tKjlIQR086359ejhz93v+wcMfr7RMVf9MV8dpKnuz7vk6WQ2Fl9t2GGqd95t"
    "dduicqoOzwqN/hWxxv1gnPZ5vAjK8HyE8FEwglG2lVqJYEgQRUlMyUI5+fFv"
    "OJvA1q0kwpbYVEVMJVg+o0TlYorHGMMSE0ikwIfnYzEy6EcJNJgs47HV3Cjl"
    "m7QtVpfD5nScwLDHySs0Y1yiy2L78cWBIH6MwQM46ie+dgJh8slO/n0yz+4L"
    "MstyUZrj43L1O28djp9QHxOOPv9631EFlqE4qV6pfSuH8JdLPHa8UUdxNR4O"
    "00GKkXIZF2w52t+4wjJBu+0exGGHvUhEr53rFhk0qzLqQ2GnFyNSkgnSr2K"
    "BFoudvsso+WwmW8mfN4xgHdncAESqMSrCEAQpcErRAOk2MwMMQwSCATJEhDx"
    "kZE8L+xHVNIOSbjcM4yhV3wOaLUq47lebhayUrTfqt6YthABeyFXOykVVzsqc"
    "bAa6LeTSASZAzm8W9WOJgy0EoPchX8uXJFFQ8uXLwx3sTCeXF9Pm29ZgMOiZ"
    "gF6bOX9ihVa9NO7k6oqqJJ5BO50Nnk4nJYKLRviG4fFP1qp/O51dXYza18Pu"
    "zc2LnU/R2zHBTcJev81mNS7tP1M4itBUw7AYAAAAAElFTkSuQmCC"
)

sample_input = pd.DataFrame({"input": [sample_b64]})

# COMMAND ----------

# MAGIC %md
# MAGIC ### Display the input image

# COMMAND ----------

import base64
from io import BytesIO
from PIL import Image
from matplotlib import pyplot as plt

input_img = Image.open(BytesIO(base64.b64decode(sample_b64)))

plt.figure(figsize=(3, 3))
plt.imshow(input_img, cmap="Greys")
plt.title("Input image (coat)")
plt.axis("off")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section A — Load Model from Unity Catalog

# COMMAND ----------

# MAGIC %pip install tensorflow_similarity protobuf==3.20.3 tf-keras

# COMMAND ----------

# Register custom objects globally so tf_keras can deserialize the saved model
import tf_keras
from tensorflow_similarity.losses import MultiSimilarityLoss
from tensorflow_similarity.layers import MetricEmbedding
from tensorflow_similarity.models import SimilarityModel

tf_keras.utils.get_custom_objects().update({
    "Similarity>MultiSimilarityLoss": MultiSimilarityLoss,
    "MultiSimilarityLoss": MultiSimilarityLoss,
    "MetricEmbedding": MetricEmbedding,
    "SimilarityModel": SimilarityModel,
})

# COMMAND ----------

model_uri = f"models:/{model_name}/8"
print(f"Loading model from: {model_uri}")
loaded_model = mlflow.pyfunc.load_model(model_uri)

# COMMAND ----------

predictions = loaded_model.predict(sample_input)
print("Predictions (5 nearest-neighbour images as hex):")
display(predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section B — Query Model Serving Endpoint

# COMMAND ----------

import requests
import json

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

payload = {"dataframe_records": [{"input": sample_b64}]}

response = requests.post(endpoint_url, headers=headers, json=payload)

print(f"Status: {response.status_code}")
print(json.dumps(response.json(), indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualize Results
# MAGIC
# MAGIC Decode one of the nearest-neighbour hex images back to a viewable format.

# COMMAND ----------

import numpy as np
from matplotlib import pyplot as plt

# Use the local prediction result
hex_img = predictions.iloc[0, 0]
img_array = np.frombuffer(bytes.fromhex(hex_img), dtype=np.float32).reshape(28, 28)

plt.figure(figsize=(3, 3))
plt.imshow(img_array, cmap="Greys")
plt.title("Nearest neighbour #1")
plt.axis("off")
plt.show()
