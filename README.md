# AI Chatbot
### An AI Chatbot LLM model with architecture based on the original GPT2 (124M), built and trained entirely locally from scratch.

All of the code and processing of this project was done entirely on my laptop, no cloud computing was used to train or run the models. 

Repository includes the code, a dataset generated using locally run Llama 3-8B for training, and the necessary programs to create said data using a local model, as well as train and run your own models.

I followed and built over Andrej Karpathy's lectures for this project, specifically these two respectively for [V1](https://www.youtube.com/watch?v=kCc8FmEb1nY) and [V2](https://www.youtube.com/watch?v=l8pRSuU81PU)

## Features

### AI Chatbot

Model that reads and responds to your questions. V1 is the model trained from scratch on the generated dataset, V2 is the pretrained GPT2 model that is finetuned on the generated dataset.


### AI that speaks like you

Feed the model your own data to create a virtual homunculus that aspires to be just like you but fails due to lacking a soul. Finetune the existing GPT2 model to morph it into yourself or someone else using the finetuner.py and placing the data you wish to emulate in the same directory, by default the program will look for a file named "input.txt" but you can modify that in the first few lines of the code.
