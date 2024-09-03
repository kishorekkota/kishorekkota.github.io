---
title: AWS Lex
layout: home
nav_order: 10
parent: AWS Connect
---

## AWS Lex Bot 

This Document Describe some of the fundamental around AWS Lex Box and how it can be leveraged to implement IVR and IVA solutions for a Contact Center.


### Context

A Contact Center support interaction via two different channels, which are Digital and Voice. Goal for any contact center is to provide self service. Lex is support contact centers to build self service solutions.


### WHAT is Lex ?

Amazon Lex is a fully managed AI service with advanced natural language models to design, build, test, and deploy conversational interfaces for voice and text.

Design and Deploy Omnichannel Conversational AI, without worrying about hardware or infrastructure. Hence Lex can be referred as a Managed Service solution for Conversational AI.

Lex Provides below services.

- NLU - Natural Language Understanding.
- ASR - Automatic Speech Recognition

It provide these functionality by leveraging LLMs. 


Lex allows building below functionality.

- Build Virtual Agents
- Automated informational responses
- Productivity Task Automation



For supporting these; Lex leverages other AWS Service like Amazon Kendra, AWS Lambda, AWS Identity Manager.

Lex Supported Channel include.

- Phone
- Chat
- Messaging
- Website
more..


Amazon Lex can be used with a streaming API or request - response API.

When integrating with Connect, it uses Streaming API.

When using text chatbot, you will use a request - response api.

### Technical Concepts with Lex

* **Intent:** The desired outcome of a user's interaction with Lex. For example, "Order a pizza," "Check account balance," or "Schedule an appointment."
* **Slot:** Information needed to fulfill an intent. For example, in the "Order a pizza" intent, slots might include "pizza type," "size," "toppings," and "delivery address."
* **Utterance:** The way a user expresses their intent. Example: "I want to order a large pepperoni pizza with extra cheese," or "Can I order a pizza?"
* **Fulfillment:** The action taken to fulfill the user's intent. This can involve accessing external systems or databases, generating a response, or transferring the call to a human agent.
* **Dialog Flow:** The series of steps and prompts used by Lex to gather the necessary information from the user to fulfill their intent.

**Integration Process:**

1. **Build Lex Bot:** Create a Lex bot that understands the desired intents and slots relevant to your business needs.
2. **Define Contact Flow:** Create a Contact Flow in Amazon Connect that:
3. **Initiates Lex:** Use the `Invoke AWS Lambda` block to call a Lambda function that starts the Lex interaction.
4. **Processes Lex Output:** Capture the bot's response (e.g., text, speech) and present it to the customer.
5. **Collect User Inputs:** Gather user input and forward it to Lex through a Lambda function.
6. **Handle Fulfillment:** Implement the necessary steps to handle the bot's fulfillment action, such as making API calls or transferring to a human agent.


**Integration Choices**

>Chat
- Lex Can integrate with Amazon Connect for Voice.
- Lex Can integrate with Slack, Facebook Messenger and SMS for Chat.

>Translate
- Amazon Translate can be integrated to Chat / Text interactions.

>ASR and NLU
- Lex Provides both ASR and NLU capabilities.

>Amazon Polly
- Polly is used for Text to Speech with SSML Capabilities.

> Amazon Kendra
- Kendra can support NLP based on Knowledge Based Search.
- Provides Out Of The Box connectors for SaaS solutions like Salesforce and ServiceNOW.

> Dashboard
- Lex can stream logs to S3, which can used to create Dashboard via Athena.

> LLM
- Amazon Bed Rock provides LLM capabilities for NLP and Summarization.

> API Integration
- Amazon Lambda used for integrating with API and Data sources.

> Integration Outside of Amazon Connect
- Amazon Chime Voice Connector and Amazon Chime SDK Public Switched Telephone Network (PSTN) Audio service to access Amazon Lex V2 bot.