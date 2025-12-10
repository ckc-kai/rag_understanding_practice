from sentence_transformers import SentenceTransformer
import torch
model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="mps")

text = ["A small cat sleeps on the couch.", 
        "The couch has a sleeping kitten."]

emb = model.encode(text)
print(emb.shape) 
