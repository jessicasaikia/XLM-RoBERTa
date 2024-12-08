# XLM-RoBERTa
This repository implements a XLM-RoBERTa model for performing Parts-of-Speech (POS) Tagging on Assamese-English code-mixed texts.

## Introduction to Parts-of-Speech (PoS) Tagging
PoS tagging is the process of identifying and labeling grammatical roles of words in texts, supporting applications like machine translation and sentiment analysis. While different languages may have their own PoS tags, I have used my own custom PoS tags for this model. The Table below defines the custom PoS tags used in this model-

![Table](https://github.com/jessicasaikia/hidden-markov-model-HMM/blob/main/Custom%20PoS%20tags%20Table.png)

## About XLM-RoBERTa
It is a transformer-based multilingual pre-trained model that is designed for handling a wide variety of languages in NLP tasks. Trained in over 100 languages, it can understand and generate text without needing translation. XLM-RoBERTa uses a Masked Language Model (MLM) approach, where some words are hidden, and the model predicts them. It generates token embeddings using wordpiece, positional, and segment embeddings. The model’s core relies on self-attention to learn relationships between tokens and improves embeddings through feed-forward networks. During pre-training, it learns tasks like sentence order prediction to align information across languages. For specific tasks like POS tagging, it adds a classification head and uses SoftMax activation to predict labels. The model is fine-tuned with labeled data and optimised with techniques like Adam. During prediction, it generates labels based on its multilingual learning.

**Algorithm**:
1.	The model imports the required libraries and loads the dataset.
2.	Input sentences are tokenised into subwords using the XLM-RoBERTa tokeniser.
3.	Each token is mapped to a contextualised embedding via the XLM-RoBERTa model. 
4.	The model generates embeddings for each token based on its surrounding context (both in Assamese and English).
5.	The contextualised embeddings are passed through a linear classification layer on top of the model. 
6.	This layer predicts a POS tag for each token, such as EN-NOUN, AS-VERB, etc., by using the learned features from the embeddings. 
7.	The model outputs a sequence of POS tags corresponding to the input tokens.
8.	These predicted tags are compared to the true POS tags for evaluation.


## Where should you run this code?
I used Google Colab for this Model.
1. Create a new notebook (or file) on Google Colab.
2. Paste the code.
3. Upload your CSV dataset file to Google Colab.
4. Please make sure that you update the "path for the CSV" part of the code based on your CSV file name and file path.
5. Run the code.
6. The output will be displayed and saved as a different CSV file.

You can also VScode or any other platform (this code is just a Python code)
1. In this case, you will have to make sure you have the necessary libraries installed and datasets loaded correctly.
2. Run the program for the output.

## Additional Notes from me
If you need any help or questions, feel free to reach out to me in the comments or via my socials. My socials are:
- Discord: jessicasaikia
- Instagram: jessicasaikiaa
- LinkedIn: jessicasaikia (www.linkedin.com/in/jessicasaikia-787a771b2)

Additionally, you can find the custom dictionaries that I have used in this project and the dataset in their respective repositories on my profile. Have fun coding and good luck! :D
