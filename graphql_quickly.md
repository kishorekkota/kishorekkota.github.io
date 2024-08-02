---
layout: home
nav_order: 10
title: Graph QL
---

## Why I wrote this page ?

I am mostly unhappy with the way books are written, rather prefer articles over book for the reason of not having time to read, TLDR. I do want to read a book when i have time but most of the time, I spend time to find time. 

This page is broken down into 3 three section, each section will have subsection. Any article that has more than three section is consider too long to read - IMO.

- What is the concept Graph QL ?
- What is GraphQL Architecture and Design ?
- What are the important terms and their usage ?
- Diagram representing different usage in Graph QL ?


## What is it ?

GraphQL as the name says, it is built on the concept of Graph to work on Data, with ability to Query the data that need in a Graph Like data structure. 
`We do not want to Graph Concepts and bore you with details on it. If you know good, if not you can find lots of good material in internet.'

Concept of Graph QL was evolved to support some of the complexities around data optimization when supporting low bandwidth networks, to support mobile interfaces to skim off unnceccasry data. However, this has evolved more on more usecase to support ever changing Micro Service way of building. GraphQL is very critical to speed up Micro Services way of building, lets list of the key capabilities that GraphQL provides.

- Tailored Responses
  * Apply transformation as needed or skim off unwanted; this will act as View Helper for those that are familiar Design Patterns.
- Consolidate Requests
  * Allows multiple API calls to be consolidated a single API Request, allows optimizing data fetching across different API needed for Screen Rendering.
- Strong Type Definition
  * GraphQL Supports Strong Typing for API, allowing Contract definition between consumer and provider, serves as documentation of API Schema.
- Streaming Data
  * Graph QL Support Data Subscription allowing realtime data updates as data changes.
- Faster UI Development 
  * GraphQL allows UI to work in decoupled fashion via established contracts, so UI does not have to wait for working backend, this support API virtualization and parallel development between UI and API.




## What is GraphQL Architecture and Design ?


GraphQL is client & server architecture; this involves a client, which is an UI or another API, which is interacting with GraphQL Server, which is middleware acting as a gateway to backend API, for fulfilling the request.




## What are the important terms and their usage ?


**Resolvers**

A Resolver, as the name indicates, its roles is to find the value of field, can be simple or complex with in a given Schema.  Resolver is responsible for populated data fetching from respective source - which could be another API or Database.

You can think of Resolvers as design construct, which allows you to write code to work with respective data source.


**Query**

A Query, is an operation for fetching data. GraphQL clients interact with Server via Query to fetch desired data. This is the fundamental concept with GrapQL. Query can be designed as per below.

- Matching exact data needs based on Client.
- Nested Queries.
- Introspection, so client can discover and understand schema for usage.
- String Typing.

Query's are built around;

- Fields.
- Arguments.
- Fragments.
- Variable.

**Schema** 

A Schema, define stricture of the data that need to be fecthed via GraphQL. It defines type of the data that can either be queried or mutaed. It contains relationship between different types and operations that can be performed.


**Mutation**

A Mutation is type of operation in GrapQL which allows data to be modified. This is mainly intended for Creating, Updaating and Deleting.

Key Characteristics of 




