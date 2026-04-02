# Ontologies

It's all about data. 

We are looking to get more information from the data we have, think about banks and other organizations, which want to optimize their spending, find ways to gain more and areas to grow. Companies are paying more to get insights into their data and processes, using consultancy companies or which will come even more common, AI. As consultancy companies, AI also needs context. Consultants know what to ask. With AI, it works the other way. First, we ask. We ask what to improve, what else we can change, make faster. AI normally asks us which path to follow between given ones. It might not doubt the data we have in a way a person would, more specifically, a data modeller.

A data modeller who's job is to model the data in a logical data model, will have do doubt everything. The constant question would be
"Is the model detailed enough, what am I missing?". But they know which question to ask to doubt this less and less. Which takes time.

How do we get to ontologies then? The process of data modelling is a long one. It could end or start with an ontology of the data. All the models would be created much more simply if ontology is created first, but there are many reasons against it. Especially now when we have to build fast. Less questions the better. "Ship first, ask later." Then we end up with a database which doesn't what it was required in the beginning. But once it grows, and we'll want to understand more about the business, we'll end up with the same question. "What is in our data?". Normally in a larger company, each department knows their par tof the data well. Often, there are only specific people in the department that know the data well. They've worked with it for years, they know what goes in, what are the rules, the business context of every attribute and enum, every table. They have the knowledge that could be stored in an ontology.

## Ontology Matching

In this repository we show how we can use Python to do basic ontology matching using test ontologies from [OAEI](https://oaei.ontologymatching.org/). Specifically, we show the use of Levenshtein distance, n-gram similarity, cosine similarity and path distance. 

The extension is in the works.