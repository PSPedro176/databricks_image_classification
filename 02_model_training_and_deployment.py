# Databricks notebook source
# MAGIC %md
# MAGIC # Image Recommendation System — Model Training & Deployment
# MAGIC
# MAGIC This notebook reads the Fashion MNIST Delta tables created in **01_data_preparation**,
# MAGIC trains a similarity model on a single GPU, registers it to Unity Catalog,
# MAGIC and deploys a Model Serving endpoint.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Table of Contents
# MAGIC
# MAGIC 1. **Setup** — Install libraries, imports, configure Unity Catalog
# MAGIC 2. **Single-GPU Training** — Train a similarity model on one GPU
# MAGIC 3. **Deployment** — Register to UC and create a Model Serving endpoint

# COMMAND ----------

# MAGIC %pip install tensorflow_similarity protobuf==3.20.3 tf-keras

# COMMAND ----------

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from sys import version_info

import numpy as np
import pandas as pd
import mlflow
import mlflow.pyfunc
import requests
import cloudpickle

import tensorflow as tf

# Allow GPU memory growth to prevent cuDNN init errors
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

import tensorflow_similarity as tfsim
from tensorflow_similarity.layers import MetricEmbedding
from tensorflow_similarity.losses import MultiSimilarityLoss
from tensorflow_similarity.models import SimilarityModel
from tensorflow_similarity.samplers import MultiShotMemorySampler
from tensorflow_similarity.samplers import select_examples
from tensorflow_similarity.visualization import viz_neigbors_imgs
from matplotlib import pyplot as plt

# COMMAND ----------

# DBTITLE 1,Set up MLflow experiment
useremail = (
    dbutils.notebook.entry_point
    .getDbutils().notebook().getContext().userName().get()
)
experiment_name = f"/Users/{useremail}/image_recommender"
mlflow.set_experiment(experiment_name)
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# DBTITLE 1,Configure Unity Catalog and Volumes
catalog = "image_rec_classic_catalog"
schema = "image_recommendation"
volume = "data"
volume_path = f"/Volumes/{catalog}/{schema}/{volume}"
model_name = f"{catalog}.{schema}.image_recommender"

print(f"Using catalog: {catalog}")
print(f"Using schema: {catalog}.{schema}")
print(f"Volume path: {volume_path}")
print(f"Model name: {model_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read Delta tables

# COMMAND ----------

train = (
    spark.table(f"{catalog}.{schema}.fmnist_train_data")
    .drop("image_id")
    .toPandas().values
)
test = (
    spark.table(f"{catalog}.{schema}.fmnist_test_data")
    .drop("image_id")
    .toPandas().values
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helper functions

# COMMAND ----------

def get_dataset(
    train: np.ndarray,
    test: np.ndarray,
    rank: int = 0,
    size: int = 1,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Reshape and partition image data for training."""
    np.random.shuffle(train)
    np.random.shuffle(test)

    x_train = train[:, 1:].reshape(-1, 28, 28)
    y_train = train[:, 0].astype(np.int32)
    x_test = test[:, 1:].reshape(-1, 28, 28)
    y_test = test[:, 0].astype(np.int32)

    x_train = x_train[rank::size]
    y_train = y_train[rank::size]
    x_test = x_test[rank::size]
    y_test = y_test[rank::size]

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    return (x_train, y_train), (x_test, y_test)

# COMMAND ----------

def get_model() -> SimilarityModel:
    """Build a simple CNN-based similarity model."""
    inputs = layers.Input(shape=(28, 28, 1))
    x = layers.Rescaling(1 / 255)(inputs)
    x = layers.Conv2D(32, 3, activation="relu")(x)
    x = layers.MaxPool2D(2, 2)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.MaxPool2D(2, 2)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(256, 3, activation="relu")(x)
    x = layers.MaxPool2D(2, 2)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Flatten()(x)
    outputs = MetricEmbedding(256)(x)
    return SimilarityModel(inputs, outputs)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Single-GPU Training

# COMMAND ----------

num_classes = 10

# COMMAND ----------

def train_model(
    train: np.ndarray,
    test: np.ndarray,
    learning_rate: float = 0.001,
) -> SimilarityModel:
    """Train a similarity model on a single GPU."""
    from tensorflow.keras.optimizers import Adam as AdamOpt
    from tensorflow_similarity.losses import MultiSimilarityLoss as MSLoss
    from tensorflow_similarity.samplers import MultiShotMemorySampler as Sampler

    mlflow.tensorflow.autolog()

    (x_train, y_train), (x_test, y_test) = get_dataset(train, test)
    classes = [2, 3, 1, 7, 9, 6, 8, 5, 0, 4]
    num_classes_ = 6
    class_per_batch = num_classes_
    example_per_class = 6
    epochs = 10
    steps_per_epoch = 1000

    sampler = Sampler(
        x_train,
        y_train,
        classes_per_batch=class_per_batch,
        examples_per_class_per_batch=example_per_class,
        class_list=classes[:num_classes_],
        steps_per_epoch=steps_per_epoch,
    )
    model = get_model()
    distance = "cosine"
    loss = MSLoss(distance=distance, reduction="sum_over_batch_size")
    model.compile(optimizer=AdamOpt(learning_rate), loss=loss)
    model.fit(sampler, epochs=epochs, validation_data=(x_test, y_test))
    return model

# COMMAND ----------

model = train_model(train, test, learning_rate=0.001)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train the Final Model
# MAGIC
# MAGIC Train the final model with tuned parameters and build an index for querying.

# COMMAND ----------

(x_train, y_train), (x_test, y_test) = get_dataset(train, test)
classes = [2, 3, 1, 7, 9, 6, 8, 5, 0, 4]
num_classes = 7
classes_per_batch = num_classes
example_per_class = 20
epochs = 20
steps_per_epoch = 1100
learning_rate = 0.0013508067254937172

sampler = MultiShotMemorySampler(
    x_train,
    y_train,
    classes_per_batch=classes_per_batch,
    examples_per_class_per_batch=example_per_class,
    class_list=classes[:num_classes],
    steps_per_epoch=steps_per_epoch,
)
tfsim_model = get_model()
distance = "cosine"
loss = MultiSimilarityLoss(distance=distance, reduction="sum_over_batch_size")
tfsim_model.compile(optimizer=Adam(learning_rate), loss=loss)
tfsim_model.fit(sampler, epochs=epochs, validation_data=(x_test, y_test))

# COMMAND ----------

tfsim_model.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build an Index

# COMMAND ----------

x_index, y_index = select_examples(x_train, y_train, classes, 20)
tfsim_model.reset_index()
tfsim_model.index(x_index, y_index, data=x_index)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualize a sample image

# COMMAND ----------

sample_image = x_index[0]
sample_image = sample_image.reshape(1, sample_image.shape[0], sample_image.shape[1])
plt.imshow(sample_image[0], interpolation="nearest")
plt.show()

# COMMAND ----------

label = y_index[0]
label
# 4 is a Coat — see https://github.com/zalandoresearch/fashion-mnist

# COMMAND ----------

# MAGIC %md
# MAGIC ### Query nearest neighbours

# COMMAND ----------

x_display, y_display = select_examples(x_test, y_test, classes, 1)

nns = np.array(tfsim_model.lookup(x_display, k=5))

for idx in np.argsort(y_display):
    viz_neigbors_imgs(
        x_display[idx], y_display[idx], nns[idx],
        fig_size=(16, 2), cmap="Greys",
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflow Pyfunc Wrapper for Deployment

# COMMAND ----------

import shutil

tfsim_path = "/databricks/driver/models/tfsim.pth"
if os.path.exists(tfsim_path):
    shutil.rmtree(tfsim_path)
tfsim_model.save(tfsim_path)

# COMMAND ----------

artifacts = {"tfsim_model": tfsim_path}

# COMMAND ----------

class TfsimWrapper(mlflow.pyfunc.PythonModel):
    """Pyfunc wrapper that accepts a base64-encoded image and returns
    the 5 nearest-neighbour images as hex-encoded byte strings."""

    def load_context(
        self, context: mlflow.pyfunc.PythonModelContext
    ) -> None:
        import os
        os.environ["TF_USE_LEGACY_KERAS"] = "1"
        import tf_keras
        from tensorflow_similarity.losses import MultiSimilarityLoss
        from tensorflow_similarity.layers import MetricEmbedding
        from tensorflow_similarity.models import SimilarityModel

        self.tfsim_model = tf_keras.models.load_model(
            context.artifacts["tfsim_model"],
            custom_objects={
                "MultiSimilarityLoss": MultiSimilarityLoss,
                "MetricEmbedding": MetricEmbedding,
                "SimilarityModel": SimilarityModel,
            },
        )
        self.tfsim_model.load_index(context.artifacts["tfsim_model"])

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
    ) -> pd.DataFrame:
        from PIL import Image
        import base64
        import io

        raw = model_input["input"][0].encode()
        image = np.array(
            Image.open(io.BytesIO(base64.b64decode(raw)))
        )
        image_reshaped = image.reshape(-1, 28, 28) / 255.0
        images = np.array(self.tfsim_model.lookup(image_reshaped, k=5))
        image_dict = {
            i: images[0][i].data.tostring().hex() for i in range(5)
        }
        return pd.DataFrame.from_dict(image_dict, orient="index")

# COMMAND ----------

PYTHON_VERSION = (
    f"{version_info.major}.{version_info.minor}.{version_info.micro}"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Conda environment for the MLflow model

# COMMAND ----------

conda_env = {
    "channels": ["defaults"],
    "dependencies": [
        f"python={PYTHON_VERSION}",
        "pip",
        {
            "pip": [
                "mlflow",
                f"tensorflow_similarity=={tfsim.__version__}",
                f"tensorflow_cpu=={tf.__version__}",
                f"cloudpickle=={cloudpickle.__version__}",
                "tf-keras",
                "Pillow",
                "protobuf==3.20.3",
            ],
        },
    ],
    "name": "tfsim_env",
}

# COMMAND ----------

mlflow_pyfunc_model_path = "/databricks/driver/models/tfsim_mlflow.pth"
if os.path.exists(mlflow_pyfunc_model_path):
    shutil.rmtree(mlflow_pyfunc_model_path)
mlflow.pyfunc.save_model(
    path=mlflow_pyfunc_model_path,
    python_model=TfsimWrapper(),
    artifacts=artifacts,
    conda_env=conda_env,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test the pyfunc model locally

# COMMAND ----------

img = (
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

data = {"input": [img]}
sample_image = pd.DataFrame.from_dict(data)

# COMMAND ----------

loaded_model = mlflow.pyfunc.load_model(mlflow_pyfunc_model_path)
test_predictions = loaded_model.predict(sample_image)
print(test_predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Infer model signature

# COMMAND ----------

from mlflow.models.signature import infer_signature

signature = infer_signature(sample_image, loaded_model.predict(sample_image))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register Model to Unity Catalog

# COMMAND ----------

with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
        artifact_path="tfsim",
        python_model=TfsimWrapper(),
        artifacts=artifacts,
        conda_env=conda_env,
        signature=signature,
        registered_model_name=model_name,
    )
    model_version = mlflow.register_model(
        f"runs:/{run.info.run_id}/tfsim",
        model_name,
    )
    print(
        f"Model registered: {model_name}, version: {model_version.version}"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Model Serving Endpoint

# COMMAND ----------

serving_endpoint_name = "image-recommender"
workspace_url = (
    dbutils.notebook.entry_point
    .getDbutils().notebook().getContext().apiUrl().get()
)
token = (
    dbutils.notebook.entry_point
    .getDbutils().notebook().getContext().apiToken().get()
)

endpoint_config = {
    "name": serving_endpoint_name,
    "config": {
        "served_entities": [
            {
                "entity_name": model_name,
                "entity_version": model_version.version,
                "workload_size": "Small",
                "scale_to_zero_enabled": True,
            }
        ]
    },
}

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json",
}

response = requests.post(
    f"{workspace_url}/api/2.0/serving-endpoints",
    headers=headers,
    json=endpoint_config,
)

if response.status_code == 200:
    print(f"Endpoint '{serving_endpoint_name}' created successfully")
elif "already exists" in response.text:
    response = requests.put(
        f"{workspace_url}/api/2.0/serving-endpoints/"
        f"{serving_endpoint_name}/config",
        headers=headers,
        json=endpoint_config["config"],
    )
    print(
        f"Endpoint '{serving_endpoint_name}' updated: "
        f"{response.status_code}"
    )
else:
    print(f"Error: {response.status_code} - {response.text}")

print(
    f"Endpoint URL: {workspace_url}/serving-endpoints/"
    f"{serving_endpoint_name}/invocations"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test payload example
# MAGIC
# MAGIC ```json
# MAGIC [{"input": "iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAAAAABXZoBIAAACN0lE..."}]
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the [Databricks License](https://databricks.com/db-license-source). All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library / data source | description | license | source |
# MAGIC |---|---|---|---|
# MAGIC | tensorflow | package | Apache 2.0 | https://github.com/tensorflow/tensorflow/blob/master/LICENSE |
# MAGIC | fashion-mnist | dataset | MIT | https://github.com/zalandoresearch/fashion-mnist/blob/master/LICENSE |
