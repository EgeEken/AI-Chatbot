from gpt4all import GPT4All
import time

start = time.time()

llama3 = "Meta-Llama-3-8B-Instruct.Q4_0.gguf"

gpt4all_path="C:\\Users\\EGE\\AppData\\Local\\nomic.ai\\GPT4All"

model = GPT4All(llama3, model_path=gpt4all_path, device="gpu", allow_download=False)

with open("english-common-words.txt") as f:
    words = f.readlines()
    
words = [word.strip() for word in words]

def prompt_generator(word, count=5):
    return f"Generate {count} Q/A exchanges relating to the word {word} that i can use to train my own chatbot. Your response should be in this format:\ni:[input]\no:[output]\nand nothing else, no numbers, just the i: and o: lines"

ind1 = 0
ind2 = len(words)

ind1 = 600
ind2 = 800 # updated (04/05/2024)

chosen_words = words[ind1:ind2]

print(f"Loaded model and prepared word list in {time.time()-start:.2f} seconds")

allstart = time.time()
for i in range(len(chosen_words)):
    with open(f"data.txt", "a") as f:
        f.write("\n\n")
    print(f"  {i+1}/{len(chosen_words)} - Generating data for word {chosen_words[i]} (word id: {ind1+i})")
    start = time.time()
    output = "".join(model.generate(prompt_generator(chosen_words[i], 5), max_tokens=2000, temp=0.4, streaming=False))
    with open(f"data.txt", "a") as f:
        f.write(str(output))
        f.write("\n\n")
    print(f"--- Done in {time.time()-start:.2f} seconds ---")
        
        
print(f"--- All Done in {time.time()-allstart:.2f} seconds ---")