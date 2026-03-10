# Databricks notebook source
# MAGIC %md
# MAGIC # Vector Search Setup
# MAGIC
# MAGIC This notebook creates the infrastructure for similarity search:
# MAGIC 1. **Delta table** for embeddings in Unity Catalog (source of truth for VS)
# MAGIC 2. **Vector Search endpoint & Delta Sync index** for ANN similarity search
# MAGIC
# MAGIC Run this **before** notebook 03 (model training), which populates the table.

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

catalog = "image_rec_classic_catalog"
schema = "image_recommendation"
embeddings_table_name = f"{catalog}.{schema}.image_embeddings"
vs_endpoint_name = "image-recommender-vs"
vs_index_name = f"{catalog}.{schema}.image_embeddings_index"

print(f"Embeddings table: {embeddings_table_name}")
print(f"VS endpoint:    {vs_endpoint_name}")
print(f"VS index:       {vs_index_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Embeddings Delta Table

# COMMAND ----------

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {embeddings_table_name} (
        image_id INT NOT NULL,
        embedding ARRAY<FLOAT> NOT NULL
    )
    USING DELTA
    TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")

# Set primary key constraint (required for Vector Search Delta Sync)
spark.sql(
    f"ALTER TABLE {embeddings_table_name} "
    f"ALTER COLUMN image_id SET NOT NULL"
)
try:
    spark.sql(
        f"ALTER TABLE {embeddings_table_name} "
        f"ADD CONSTRAINT image_embeddings_pk PRIMARY KEY (image_id)"
    )
except Exception as e:
    if "already exists" in str(e).lower():
        pass  # PK already set from a previous run
    else:
        raise
print(f"Embeddings table '{embeddings_table_name}' ready (with PK on image_id).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Vector Search Endpoint

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vs_client = VectorSearchClient()

try:
    vs_client.create_endpoint_and_wait(name=vs_endpoint_name, endpoint_type="STANDARD")
    print(f"VS endpoint '{vs_endpoint_name}' created.")
except Exception as e:
    if "already exists" in str(e).lower():
        print(f"VS endpoint '{vs_endpoint_name}' already exists.")
    else:
        raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Delta Sync Index

# COMMAND ----------

try:
    vs_client.create_delta_sync_index_and_wait(
        endpoint_name=vs_endpoint_name,
        index_name=vs_index_name,
        source_table_name=embeddings_table_name,
        primary_key="image_id",
        pipeline_type="TRIGGERED",
        embedding_vector_column="embedding",
        embedding_dimension=256,
    )
    print(f"VS index '{vs_index_name}' created.")
except Exception as e:
    if "already exists" in str(e).lower():
        print(f"VS index '{vs_index_name}' already exists.")
    else:
        raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## Done
# MAGIC
# MAGIC Infrastructure is ready. Run **03_model_training_and_deployment** next to
# MAGIC train the model, compute embeddings, sync the VS index, and deploy.
