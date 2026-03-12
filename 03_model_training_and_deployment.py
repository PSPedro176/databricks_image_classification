# Databricks notebook source
# MAGIC %md
# MAGIC # Image Recommendation System — Model Training & Deployment

# COMMAND ----------

# MAGIC %pip install tensorflow_similarity protobuf==3.20.3 tf-keras databricks-vectorsearch
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Packages
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
embeddings_table_name = f"{catalog}.{schema}.image_embeddings"

print(f"Using catalog: {catalog}")
print(f"Using schema: {catalog}.{schema}")
print(f"Volume path: {volume_path}")
print(f"Model name: {model_name}")
print(f"Embeddings table: {embeddings_table_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read Delta tables

# COMMAND ----------

train_pdf = spark.table(f"{catalog}.{schema}.fmnist_train_data").toPandas()
test_pdf = spark.table(f"{catalog}.{schema}.fmnist_test_data").toPandas()

# train keeps image_id (col 0), label (col 1), pixels (cols 2+)
train = train_pdf.values
# test drops image_id — only used for validation, not indexing
test = test_pdf.drop(columns=["image_id"]).values

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helper functions

# COMMAND ----------

def get_dataset(
    train: np.ndarray,
    test: np.ndarray,
    rank: int = 0,
    size: int = 1,
):
    """Reshape and partition image data for training.
    train has columns: image_id, label, pixel_0..pixel_783
    test has columns: label, pixel_0..pixel_783 (no image_id)
    """
    np.random.shuffle(train)
    np.random.shuffle(test)

    id_train = train[:, 0].astype(np.int64)
    x_train = train[:, 2:].reshape(-1, 28, 28)
    y_train = train[:, 1].astype(np.int32)
    x_test = test[:, 1:].reshape(-1, 28, 28)
    y_test = test[:, 0].astype(np.int32)

    id_train = id_train[rank::size]
    x_train = x_train[rank::size]
    y_train = y_train[rank::size]
    x_test = x_test[rank::size]
    y_test = y_test[rank::size]

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    return (x_train, y_train, id_train), (x_test, y_test)

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

def select_examples_with_ids(x, y, ids, classes, n_per_class):
    """Like select_examples but also tracks image_ids."""
    x_out, y_out, id_out = [], [], []
    for cls in classes:
        indices = np.where(y == cls)[0][:n_per_class]
        x_out.append(x[indices])
        y_out.append(y[indices])
        id_out.append(ids[indices])
    return np.concatenate(x_out), np.concatenate(y_out), np.concatenate(id_out)

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

    (x_train, y_train, _), (x_test, y_test) = get_dataset(train, test)
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

# MAGIC %md
# MAGIC ## Train Model

# COMMAND ----------

(x_train, y_train, id_train), (x_test, y_test) = get_dataset(train, test)
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
# MAGIC ## Build an Index and visualize images

# COMMAND ----------

# DBTITLE 1,Index Build
x_index, y_index, id_index = select_examples_with_ids(x_train, y_train, id_train, classes, 20)
tfsim_model.reset_index()
tfsim_model.index(x_index, y_index, data=id_index)

# COMMAND ----------

# DBTITLE 1,Sample image
sample_image = x_index[0]
sample_image = sample_image.reshape(1, sample_image.shape[0], sample_image.shape[1])
plt.imshow(sample_image[0], interpolation="nearest")
plt.show()

# COMMAND ----------

# DBTITLE 1,Label of sample image
label = y_index[0]
label
# 4 is a Coat — see https://github.com/zalandoresearch/fashion-mnist

# COMMAND ----------

# DBTITLE 1,Query nearest neighbours
from tensorflow_similarity.samplers import select_examples
x_display, y_display = select_examples(x_test, y_test, classes, 1)

nns = tfsim_model.lookup(x_display, k=5)

# Build pixel lookup from train_pdf for visualization
_all_ids = train_pdf["image_id"].values.astype(np.int64)
_all_px = train_pdf.drop(columns=["image_id", "label"]).values.reshape(-1, 28, 28).astype(np.uint8)
_id_to_px = dict(zip(_all_ids.tolist(), _all_px))

for idx in np.argsort(y_display):
    fig, axes = plt.subplots(1, 6, figsize=(16, 2))
    axes[0].imshow(x_display[idx], cmap="Greys")
    axes[0].set_title(f"Query (label={y_display[idx]})")
    axes[0].axis("off")
    for i in range(5):
        rec_id = int(nns[idx][i].data)
        axes[i + 1].imshow(_id_to_px[rec_id], cmap="Greys")
        axes[i + 1].set_title(f"id={rec_id} d={nns[idx][i].distance:.3f}")
        axes[i + 1].axis("off")
    plt.tight_layout()
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute Embeddings

# COMMAND ----------

# DBTITLE 1,Inference
embeddings = tfsim_model.predict(x_train)  # shape: (N, 256)
print(f"Computed {embeddings.shape[0]} embeddings of dim {embeddings.shape[1]}")

# COMMAND ----------

# DBTITLE 1,Save to Delta Table
from pyspark.sql.types import (
    StructType, StructField, IntegerType, ArrayType, FloatType,
)

rows = [
    (int(id_train[i]), embeddings[i].tolist())
    for i in range(len(id_train))
]
emb_schema = StructType([
    StructField("image_id", IntegerType(), False),
    StructField("embedding", ArrayType(FloatType()), False),
])
embeddings_df = spark.createDataFrame(rows, schema=emb_schema)
embeddings_df.write.mode("overwrite").saveAsTable(embeddings_table_name)
print(f"Wrote {len(rows)} embeddings to {embeddings_table_name}")

# COMMAND ----------

# DBTITLE 1,Trigger Vector Search index sync
from databricks.vector_search.client import VectorSearchClient

vs_client = VectorSearchClient()
vs_index = vs_client.get_index(
    endpoint_name="image-recommender-vs",
    index_name=f"{catalog}.{schema}.image_embeddings_index",
)
vs_index.sync()
print("Vector Search index sync triggered.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflow Pyfunc Wrapper for Deployment

# COMMAND ----------

# DBTITLE 1,Class definition
class TfsimWrapper(mlflow.pyfunc.PythonModel):
    """Pyfunc wrapper: looks up embedding via Vector Search,
    then finds 5 nearest neighbors."""

    def load_context(self, context):
        from databricks.vector_search.client import VectorSearchClient
        self._index = VectorSearchClient().get_index(
            endpoint_name=self.vs_endpoint_name,
            index_name=self.vs_index_name,
        )

    def predict(self, context, model_input):
        image_id = int(model_input["image_id"].iloc[0])

        # Step 1: Get embedding for this image_id
        lookup = self._index.similarity_search(
            query_vector=[0.0] * 256,
            filters={"image_id": image_id},
            columns=["embedding"],
            num_results=1,
        )
        query_embedding = lookup["result"]["data_array"][0][0]

        # Step 2: Find 5 nearest neighbors (exclude self)
        results = self._index.similarity_search(
            query_vector=query_embedding,
            columns=["image_id"],
            num_results=6,
        )
        recommended_ids = [
            int(row[0]) for row in results["result"]["data_array"]
            if int(row[0]) != image_id
        ][:5]

        return pd.DataFrame({"recommended_image_id": recommended_ids})

# COMMAND ----------

# DBTITLE 1,Conda environment for the model
conda_env = {
    "channels": ["defaults"],
    "dependencies": [
        f"python={version_info.major}.{version_info.minor}.{version_info.micro}",
        "pip",
        {
            "pip": [
                "mlflow",
                f"cloudpickle=={cloudpickle.__version__}",
                "databricks-vectorsearch",
            ],
        },
    ],
    "name": "tfsim_env",
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register Model to Unity Catalog

# COMMAND ----------

# DBTITLE 1,Infer model signature
from mlflow.models.signature import infer_signature
from mlflow.models.resources import DatabricksVectorSearchIndex

# Signature uses image_id only — Vector Search resolves the embedding at serving time
sample_id = 42
sig_input = pd.DataFrame({"image_id": [sample_id]})
signature = infer_signature(sig_input, test_predictions)

# COMMAND ----------

# DBTITLE 1,MLflow logging and UC registration
wrapper = TfsimWrapper()
wrapper.vs_endpoint_name = "image-recommender-vs"
wrapper.vs_index_name = f"{catalog}.{schema}.image_embeddings_index"

with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
        artifact_path="tfsim",
        python_model=wrapper,
        conda_env=conda_env,
        signature=signature,
        input_example=sig_input.to_dict(orient="list"),
        resources=[
            DatabricksVectorSearchIndex(
                index_name=f"{catalog}.{schema}.image_embeddings_index",
            ),
        ],
    )

    model_version = mlflow.register_model(
        f"runs:/{run.info.run_id}/tfsim", model_name
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