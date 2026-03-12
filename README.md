# Image Recommendation System with Model Serving and Vector Search
- Demo inspired in solution accelerator for [image recommendation](https://github.com/databricks-industry-solutions/image-based-recommendations)
- Download to Volume a labeled dataset of 70k small images of pixelated clothes
- Train a model for embedding the images
- Save the embeddings to a Delta Table and set up a Vector Search Index
- Create a custom pyfunc mlflow model that:
  - Receives and image ID and finds the embedding of the corresponding image
  - Calculates the 5 nearest image embeddings of the input ID
  - Returns the ID of the 5 nearest neighbours
- Create a Model Serving Endpoint for your model
- Test your model and endpoint in Notebook 4
###### Keep in mind that notebooks 1 and 3 can take a while to execute:
- Save to Volume 70k images
- Train a model
- Set up a Model Serving Endpoint
