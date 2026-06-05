import random
import pandas as pd

casual_human = [
    "bro idk what just happened but it was crazy",
    "i literally forgot everything lol",
    "this whole thing makes no sense tbh",
    "man i was just chilling and then this happened",
    "i dont even care anymore honestly",
    "lowkey weird but okay",
    "i was like yeah whatever bro",
    "idk why but this felt off",
    "i just went with it and didnt think much",
    "honestly i didnt expect this at all"
]

structured_human = [
    "I had an unexpected experience that left me confused.",
    "The situation unfolded in a way I did not anticipate.",
    "I found myself reflecting on what had just occurred.",
    "This event made me reconsider my assumptions.",
    "It was a strange but memorable experience."
]

ai_style = [
    "Artificial intelligence systems are designed to process structured data efficiently.",
    "The model demonstrates consistent behavior across different inputs.",
    "This approach ensures reliable and predictable outputs.",
    "The system generates responses based on learned patterns.",
    "This method improves performance across various tasks."
]

data = []

# Generate MORE data (important)
for _ in range(5000):
    data.append((random.choice(casual_human), 0))
    data.append((random.choice(structured_human), 0))
    data.append((random.choice(ai_style), 1))

df = pd.DataFrame(data, columns=["content", "label"])
df.to_csv("datasets/auto_generated.csv", index=False)

print("Better auto dataset created!")