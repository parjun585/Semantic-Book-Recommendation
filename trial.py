from transformers import pipeline
from transformers.modeling_utils import no_init_weights, init_empty_weights

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)
result = classifier("This is a story about a wizard.", candidate_labels=["Fiction", "Nonfiction"])
print(result)
