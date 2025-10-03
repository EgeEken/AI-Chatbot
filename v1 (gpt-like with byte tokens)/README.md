# AI Chatbot
An AI Chatbot LLM model, trained from scratch. Includes my trained models, a dataset generated using locally run Llama 3-8B for training, and the necessary programs to train and run your own models.

I followed and built over [Andrej Karpathy's lecture](https://www.youtube.com/watch?v=kCc8FmEb1nY) for this project


## Results

![image](https://github.com/EgeEken/AI-Chatbot/assets/96302110/4f6a94ba-661b-4fba-9594-0d0947127fce)
![image](https://github.com/EgeEken/EgeEken/assets/96302110/51d382e2-4f24-4c51-8d94-508cf3f81aee)
![image](https://github.com/EgeEken/AI-Chatbot/assets/96302110/512f7cb1-dd17-4daf-a476-353b716a6b43)

## The chatbot

To generate text using any of the models i put here or any of the models you trained with, use the file `Chatbot.py`.
You will be prompted to input the model filename that you want to use, along with the hyperparameters.

Afterwards you can input your questions, and the chatbot will respond!


## The trainer

To train a chatbot using your own data, use the file `Trainer.py`. You will be prompted to input the text filename, the desired output model filename, as well as the necessary hyperparameters. 

The program will then start training and give you an estimate of how long it will take to finish training given your hyperparameters and GPU, it is a very accurate estimate so if it seems too long for you, it probably is, try lowering the layer or iteration counts.


## The dataset (and local dataset generator)

I generated a dataset of questions and answers using locally run Llama 3-8B, prompting it to generate 5 q/a exchanges relating to a given word, i ran this over every word in the "common english words" list i found online from Paris Diderot University, [link](https://python.sdv.univ-paris-diderot.fr/data-files/english-common-words.txt).

The dataset is in the file `data.txt`.

If you want to generate your own dataset using the same method, use the file `Data_generator.py`, you can set the model path if you have another model installed that you would like to use instead of Llama, and the words file if you want it to generate exchanges based on different words.


