---
title: Proactive Customer Engagement
layout: home
nav_order: 9
parent: Large Language Models
---


## Context 

Traditionally, contact center agent requires understanding past interaction to service the agent, this requires quick processing of the data and building muscle on deciphering short codes and quick notes, and tons of activity recording in relation to the Customer Account.

Reducing Agent Call Handling is the front and center of every call center - this not only reduces times spent by agent, also provides excellent Customer Experience, after all no one wants to be on the call longer and hearing hold music often.

In this Blog Post, I will talk thru how we are approaching solving the AHT leveraging AI.


## Data and Context

Most enterprise have very strict compliance requirements, thanks to regulations. These often require establishing various different activity records outside of actual SOR, and customer interaction often recorded in contact history and agent interaction require notes as a wrap up process.

All of these can be used to draw useful insights which can assist with providing predicting customer reason for the call. In most cases, no one will be calling for checking balances, perhaps, a delay in order processing or missing  an order delivery or wrong transaction amount or check bounced. ( I am generalizing these across different industries. These are powerful techniques.)

Data available for insights is plentiful in most enterprises, let dive into different options on how we can draw insights using AI.



## Options

There are several options for building a prediction engine, which can be realtime and batch. In this blog, we will focus on Real Time Prediction using Generative AI & LLM.

Building a prediction engine is much more complex than building a simple classification model, as it requires specialized skill with data engineering, data science and machine learning. The prediction engine requires a lot of data to be trained on, and the model needs to be able to learn from the data in order to make accurate predictions. This is where generative AI comes in, as it can easily draw insights from the data and provide a more accurate prediction based on probabilistic reasoning.

### Generative AI

There are different choices from implementation standpoint that can impact overall design and with some tradeoffs. The most common choices are:

**Use Case:** Use LLM to predict call intent and provide summary of the past interactions. 

### Option 1

Draw Call Prediction with Prompt Based Classification of Data.

Data requires preparation and cleansing, this can done based on specific data format. Most common scenario are removing PII, and supplementing abbreviation with actual text as applicable. This ensures that the data is clean and ready for analysis, which is crucial for accurate predictions.

We do not need to super scientific about data formatting, an LLM is capable of understanding data that simply plugged concatenated together from contact history and account history. Most important thing is to follow your data security and compliance requirements. ( This is same in next option as well.)

Steps to implement the prediction engine include:

1. Data Collection: Gather historical interaction data and relevant customer information.
    - Collect data from various sources, including contact history, account history, and customer interactions.
    - Build knowledge based about abbreviations and acronyms used in the data.
2. Data Preparation: Clean and format the data, ensuring compliance with regulations.
    - Remove PII and sensitive information.
    - Concatenate data from different sources (e.g., contact history, account history) into a single format.
    - Use knowledge base to replace abbreviations and acronyms with their full meanings or build glossary relevant for each customer contact history data
3. Prepare Prompt:
    - Provide a glossary for Prompt to understand the abbreviations and acronyms used in the data.
    - Provide Prompt with data prepared. 
    - Define Task for Prompt to perform.

```JSON
### Glossary
{glossary_block}

### Memo
{memo_text}

### Task
Write a plain-English call summary for the account representative.
- Use the glossary meanings where applicable.
- Keep the summary concise and under 120 words.
- Format the summary as bullet points for clarity.
- Include information that is missing abbreviations.
- Predict customer call intent based on the memo.
- Only respond with the summary and return in 3-5 bullet points.
- Respond with "Call Summary:" and then the summary for most important topics on recent information with respective dates with call intent.

```
4. Draw insights with LLM:
    - Use the LLM to analyze the data and generate a summary of the call intent.
    - The LLM will use the glossary and memo text to understand the context and provide a concise summary.
5. Post-processing:
    - Review the generated summary for accuracy and relevance.
    - Make any necessary adjustments to ensure compliance with regulations.
    - Store the summary in a secure location for future reference.



### Option 2

Draw Call Prediction with Embedding Based Classification of Data using RAG. Many steps are similar to Option 1, but with some differences in the implementation.

In this option most notable difference is that we will be using Embedding based classification of data, which is a more advanced technique that can provide better results in some cases. This option requires more data and processing power, but it can work well for large datasets.

1. Data Collection: Same as Option 1.
2. Data Preparation: Same as Option 1.
3. Fine Tune with Knowledge Base (Working example is for Option 1, Option 2 is WIP)
    - **Build Embedding:** Build a text embedding model to build a vector database for knowledge base. 
    - **Import to Vector DB:** Store the generated embeddings in the vendor database for further analysis and retrieval.
    - **Define Retrieval Tool:** Implement a retrieval tool to query the vector database and fetch relevant embeddings based on user input.

4. Draw insights with LLM:(Very similar to Option 1)
    - Provide LLM with retrieval tool for fetching from vector database.
    - Prompt does not contain glossary, as the glossary is already embedded in the vector database.
    - Prompt will retain other details from Option 1.

```JSON
### Glossary
{glossary_block}

### Memo
{memo_text}

### Task
Write a plain-English call summary for the account representative.
- Use the glossary meanings where applicable.
- Keep the summary concise and under 120 words.
- Format the summary as bullet points for clarity.
- Include information that is missing abbreviations.
- Predict customer call intent based on the memo.
- Only respond with the summary and return in 3-5 bullet points.
- Respond with "Call Summary:" and then the summary for most important topics on recent information with respective dates with call intent.        
```
5. Post-processing: Same as Option 1.


### Summary
Both options have their own advantages and disadvantages. Option 1 is simpler and easier to implement, but it may not provide the same level of accuracy as Option 2. Option 2 is more complex and requires more data and processing power, but it can provide better results in some cases.

Most importantly both of these rely on Prompt Engineering, which is a critical step in the process. The quality of the prompt can significantly impact the performance of the LLM and the accuracy of the predictions.

Some key concepts with Prompting are:

- **Task Definition:** Clearly define the task you want the LLM to perform. This helps the model understand what is expected and improves the quality of the output. In our case, we defined the task as writing a plain-English call summary for the account representative, including specific instructions on formatting and content.

- **Contextual Information:** Provide relevant context to the LLM. This helps the model understand the specific domain and improves its ability to generate accurate predictions. In our case, we provided the glossary and memo text to give the model context about the abbreviations and acronyms used in the data.

- **Formatting:** Define rules to follow for the responses. This includes using bullet points, headings, and other formatting techniques to improve readability. This can go into providing structured response as well, for the above sample, we asked the prompt to provide responses in bullet points and with "Call Summary:".

- **Feedback Loop:** Implement a feedback mechanism to continuously improve the prompt and the model's performance based on user interactions and outcomes. Although we did not cover this in the above options, this is a critical step in the process. This can be done by collecting user feedback on the generated summaries and using that feedback to refine the prompt and improve the model's performance over time. This feedback loop can help in adjusting the prompt to better meet user needs and enhance the overall effectiveness of the predictions. There are some tools off the shelf, such as LangSmith, that can help with this process. However, this can be an easier integration leveraging in-house tools for analytics  whichever is available and practiced in the enterprise.

- **Reevaluation:** Identify quality of response by evaluating against predefined metrics. This can include accuracy, relevance, and reduce hallucination based on enterprise standards. By regularly evaluating the performance of the LLM and the prompt, you can identify areas for improvement and make necessary adjustments to enhance the overall effectiveness of the predictions. These datapoints need to be made available to the data science team for further analysis and improvement of the model. (I will cover this in a future enhancement post.)

### Conclusion

Here are the key decision points for deciding on Option 1 vs Option 2:

- Does your Knowledge Base have tons of data that can be better off with RAG?
- Does your Prompt expect to have a large token size?

If you said yes to any of these, then you might want to consider Option 2. As it reduces token limit going into the Prompt and also allows you to build a knowledge base that can be used for other use cases as well. The knowledge base pipeline can be kept up to date and leveraged for continuous improvement in predictions. Most cases, Option 1 is good enough to get started with, and you can always move to Option 2 later on as your data grows and your needs evolve.


### Working Sample

[Call Intent Predication](https://github.com/kishorekkota/gen_ai/tree/main/llm_summary_with_finetuning)