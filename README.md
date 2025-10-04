# AI Chatbot
### An AI Chatbot LLM model with architecture based on the original GPT2 (124M), built and trained entirely locally from scratch.

All of the code and processing of this project was done entirely on my laptop, no cloud computing was used to train or run the models. 

Repository includes the code, a dataset generated using locally run Llama 3-8B for training, and the necessary programs to create said data using a local model, as well as train and run your own models.

I followed and built over Andrej Karpathy's lectures for this project, specifically these two respectively for [V1](https://www.youtube.com/watch?v=kCc8FmEb1nY) and [V2](https://www.youtube.com/watch?v=l8pRSuU81PU)

# Features

## AI Chatbot

Model that reads and responds to your questions. V1 is the model trained from scratch on the generated dataset, V2 is the pretrained GPT2 model that is finetuned on the generated dataset.

### V1 Directly Trained Model
![image](https://github.com/EgeEken/AI-Chatbot/assets/96302110/4f6a94ba-661b-4fba-9594-0d0947127fce)
![image](https://github.com/EgeEken/EgeEken/assets/96302110/51d382e2-4f24-4c51-8d94-508cf3f81aee)
![image](https://github.com/EgeEken/AI-Chatbot/assets/96302110/512f7cb1-dd17-4daf-a476-353b716a6b43)

### V2 Finetuned Model (Displayed on Gradio Interface)
<img width="1090" height="180" alt="image" src="https://github.com/user-attachments/assets/82dcd01c-973d-4bf6-987e-38ea63198f7c" />

<img width="1025" height="596" alt="image" src="https://github.com/user-attachments/assets/816aa571-fdf4-4bd9-a74c-1b9d9d5aa5bf" />


## AI that speaks like you

Feed the model your own data to create a virtual homunculus that aspires to be just like you but fails due to lacking a soul. Finetune the existing GPT2 model to morph it into yourself or someone else using the finetuner.py and placing the data you wish to emulate in the same directory, by default the program will look for a file named "input.txt" but you can modify that in the first few lines of the code.

### Model finetuned on my discord messages in a server between 2019-2024 
<img width="542" height="246" alt="image" src="https://github.com/user-attachments/assets/8ccd2cf7-da0f-4657-adc0-48351fdd9c33" />


