---
title: LLM
layout: home
nav_order: 4
---

TODO

(All of this is rough work and not organized well.)

## Key Terminology

### Vector Embedding

Vector Embedding is a technique in machine learning and natural language processing to convert non numeric data types, such as words, sentences, or event tier document into numeric forms. Specially Vector of real numbers.

Key process with in Vector Embedding is to represent in a high dimensional space such as geometric relationship between these vectors capturing semantic relationship.

There are several methods and models for generating vector embedding - Word2Vec, GloVe and FastText among the most popular.

## Text Embedding


## What is Language Model vs Large Language Model ?


There are 2 key properties that distinguish LLms from other language models. One is Quantitative and the other is Qualitative.

- Quantitative; 10-1oo billion parameters.
- Qualitative; primarily lies in their scale, complexity, capabilities, and the resources required for their development and operation. Has enhanced capabilities that can support zero-shot learning which can generalize based on learning and can process unseen classes based on seen classes.


### What does an LLM do anyway ?

AN LLM task is to do word prediction. In other words, given a sentence or sequence of words, it needs predict next word.

Interestingly, this is the same way many Language Models have been trained in the past with GPT. When a language model gets beyond a certain size 10B parameters, these emergent abilities, such as zero shot learning, can start to pop up.

**Size matter in LLM, perhaps training data plays vital roles in knowledge, similar to how human learn based on past experiences, same with LLM.**


### Different Use Cases of LLM

Although there are countless potential use cases for LLM, they can be broadly categorized into three distinct types.

- Prompt Engineering

- Model Fine tuning
  
  This is the process where pre-trained models undergoes a second phase of training, fine tuning, where the trained model learns about domain specific data set.

  This process allows customization of model to meet specific needs or can improve performance based on respective application. 


- 




### What is BERT in LLM ?

BERT is Bidirectional Encoder Representations from Transformers is a new NLP model and a specific type of LLM. This is developed by Google and is bidirectional in nature as the name suggest. 

- This can Read entire sequence of works at once, which allows the model to understand the context of a word based on all of its sourrouding vs just the Left or Right to it. **BiDirectional Context**

- BERT is based on the transformation architecture, which relies on attention mechanisms to capture the influence of different words on each other. Can effectively handle large amount of text and understand the nuanced meaning of words in different contexts.

- 

# Implementing a Simple Language Model

## Step 1: Understand the Basics
- **Learn about NLP**: Understand the basics of Natural Language Processing (NLP).
- **Study Language Models**: Familiarize yourself with language models, especially Transformer-based models like BERT, GPT-2, etc.

## Step 2: Choose a Framework
- **Select an ML Framework**: Use TensorFlow or PyTorch. Both are popular and well-supported.
- **Install Libraries**: Ensure you have the necessary libraries (like Hugging Face’s Transformers) installed.

## Step 3: Choose a Pre-trained Model
- **Start with a Pre-trained Model**: Choose a pre-trained model from Hugging Face’s model hub. Models like GPT-2 are a good starting point.

## Step 4: Prepare Your Dataset
- **Dataset Selection**: Choose a dataset. For language models, a large and diverse text dataset is crucial. You can find datasets on platforms like Kaggle or use open-source corpora.
- **Data Preprocessing**: Clean and preprocess your data. This involves tokenization, removing special characters, and possibly lowercasing the text.

## Step 5: Fine-Tuning the Model
- **Fine-Tuning**: Adapt the pre-trained model to your specific dataset or task. This step involves training the model further on your dataset.
- **Training Environment**: Consider using a GPU or a cloud-based platform for training, as it can be resource-intensive.

## Step 6: Implement Training Code
- **Write Training Script**: Use Python to write a script for model training. Your script will load the model, the tokenizer, process the dataset, and then run the training loop.

## Step 7: Evaluate the Model
- **Testing**: After training, evaluate your model’s performance using a separate test dataset.
- **Metrics**: Use metrics like perplexity for language models to measure performance.

## Step 8: Deployment
- **Deployment Options**: You can deploy your model using a Flask/Django app for a web interface, or use a model serving tool like TensorFlow Serving.
- **Model Serving**: Implement an API endpoint where you can input text and receive model-generated text.

## Step 9: Optimization and Maintenance
- **Monitor Performance**: Keep track of your model's performance and make improvements as needed.
- **Update the Model**: Regularly update the model with new data or refine the model to improve accuracy.

## Technology Stack
- **Programming Language**: Python
- **Frameworks**: TensorFlow or PyTorch, Hugging Face’s Transformers
- **Tools**: Jupyter Notebook for development, Git for version control
- **Deployment**: Flask/Django for web app, Docker for containerization, AWS/GCP for cloud deployment

## Example Code Snippet for Fine-Tuning
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Encode text inputs
inputs = tokenizer("Your input text here", return_tensors="pt")

# Fine-tune the model
outputs = model(**inputs, labels=inputs["input_ids"])
loss = outputs.loss
loss.backward()
```


### Building an LLM From Scratch



Generative Pre Trained Transformer

A type of Auto Regressive langugae model, pre trained on diverse test data nd fine tuned for specific tasks.

Anthropic

Reneforcement learning with Human Feedbac

COnstitutional

WHat is a Token ?

Unit which token operates on; 1 token is ~ 4 characteres/ 0.75 English works.

Every Model uses different Tokenizers.

Embeddings

Numerical Representation of works of phrases in a multidimentzasion vecotr space enabling modelrs to understand context and relationship between peices of text.

COntext Window


GROQ


Temparature faltten probablity 


Seed >> 
Specifically, SEED involves identifying error code generated by LLMs, employing Self-revise for code revision, optimizing the model with revised code, and iteratively adapting the process for continuous improvement. 

https://arxiv.org/abs/2403.00046#:~:text=Specifically%2C%20SEED%20involves%20identifying%20error,the%20process%20for%20continuous%20improvement.

Few shot prompting

Show vs telling;

for more complex tasks specific desired outputs, provide examples in the prompt which demostrate the desired behavior. This gives the model more 

Chain of Thought Prompting

pip instal streamlit


What is an Agent ?

Agenmts are systems that uses LLMS as reasoning enginer to

- Determin which actions to tkae

 analyse the resilt of their action

 determin subsequent 


ReACT

combininf reasoning and action to solve complex task thought action observation loops

AIWORKSHOP for Tabnine code assist


What is Hugging Face pre trained libraries ?


What is RAG ??

LLM Knowledge is limited based on training timeframes and access to data domains and makes them unsuitable to reson about private data or data introduce after the model training cut off date.

Relevant and Augment  Generation

Galileo RAG Bench marking system

NNCF

Quantization


## How to build an AI Businness Application ?

Lets say you have use case for leveraging AI to build a specific Business Process, it can be as simple as Fitness Plan tailored or Meal Planing for Weight Loss or a COMPLEX CRM. AI Can be leveraged based on various different needs, goals is to build systems are intuitive and easy to use, while keeping build and cost associated in a reasonable manner. 

Lets get into the details.

### Reference Architecture


Lets define reference Architecture for a building an AI Based System. 




















