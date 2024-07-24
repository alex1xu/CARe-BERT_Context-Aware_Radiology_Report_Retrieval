# CARe-BERT: BERT-Powered Graph Augmentation for Context-Aware Radiology Report Retrieval

> A new approach to training radiology report retrieval large language models without the need for time-consuming, manual data annotation by radiologists.

[![Codacy Badge](https://app.codacy.com/project/badge/Grade/5b74f79ee2904bfc92cc90fdbfdd3421)](https://app.codacy.com/gh/alex1xu/VapeVeritas-Twitter_E-cig_Surveillance/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](https://jupyter.org/try)

## 🚩 TL;DR

<details>
  <summary><b>Spoiler</b></summary>
  
</details>

## Table of Contents

- [Why?](#why)
- [Novel Pipeline Overview](#novel-pipeline-overview)
- [Model Evaluation](#model-evaluation)
- [Conclusions](#conclusions)

## Why?

![Screenshot 2024-07-23 193745](https://github.com/user-attachments/assets/4f57d3ea-3fc3-4916-9eee-9b7936c20f42)

You may have had chest X-rays taken before, but what you might not know is that afterward, radiologists will detail their observations in a free-text document called a radiology report. This past summer I worked in a lab at a University Hospital, where researchers spend hundreds of hours reviewing databases of these reports for cohort studies.

![Screenshot 2024-07-23 193754](https://github.com/user-attachments/assets/90484424-8df6-48f8-96e9-82c16899fd0f)

In natural language processing research, developing software to automate this process is called the radiology report retrieval task. Given a free text query and a collection of radiology reports, how can we create some algorithm that returns the reports relevant to the query?

**Vector space** approaches are the major focus of current research.

https://github.com/user-attachments/assets/c0aeec47-f57a-4183-8401-9278081a097f

How does it work? Let’s imagine a high-dimensional space. Vector Space models represent each report and query as a vector, or embedding, in this space where the position stores the text’s meaning. Now, the cosine of the angle between the embeddings of reports to the query represents their similarity. Ideally, the related queries and reports are close while irrelated pairs are distant.

The **Sentence-BERT**, or SBERT, transformer model has achieved state-of-the-art performance in the vector space approach.

It pools learned word embeddings into a single sentence embedding. Then, sentence embeddings are computed for three sample documents: a query text, a positive, or relevant report, and a negative, or irrelevant report. Through the triplet objective loss function, the distance between query and positive sentence embeddings is minimized, while the distance between query and negative sentence embeddings is maximized.

However, datasets of these query, positive, and negative training triplets are **currently small and restricted as they require the expertise of radiologists to create**, which is hindering the advancement of better models.

Motivated by this challenge, researchers have developed **weakly supervised learning** approaches to **programmatically generate training triplets**. Current approaches can handle queries consisting of the presence or absence of a single radiological finding.

## Novel Pipeline Overview

![Screenshot 2024-07-23 185824](https://github.com/user-attachments/assets/46e8fa05-08fd-4d82-a01d-8b0a9f0f3acd)

I improved upon existing approaches by creating **CARe-BERT**, a program to generate training triplets that teach SBERT to handle queries containing **multiple radiological findings**, and **the context of findings**, including anatomical and descriptive modifiers.

### Named Entity and Relationship Extraction

![Screenshot 2024-07-23 184628](https://github.com/user-attachments/assets/c6e7af64-9cf5-494a-b667-69ce3ac8b161)

The first stage is Named Entity Recognition (NER) and Relation Extraction (RE), which identify the clinically relevant terms that radiologists may be interested in querying. I utilize the RadGraph CXR radiology report inference dataset labeled using a joint NER and RE DyGIE++ model, initialized with PubMedBERT weights, as developed by Jain et al. (2021). Mentioned findings and anatomy in a report are identified by the model. Findings are identified in three levels of radiologists’ confidence: present, uncertain, and absent. It also identifies the relationships between these entities, including suggestive of, located at, and described as.

Past methods to extract entities for training IR models have relied on lexicon or rule-based techniques. However, these methods are vulnerable to issues like spelling mistakes, form variability (e.g., "fluid in lungs" and "pulmonary edema" describe the same condition), and ambiguous abbreviations (e.g., does "LLQ" pertain to "left lower quadrant" or "left lower lobe," or neither). Furthermore, lexicon-based approaches often fall short of capturing the relationships between entities, which are crucial for understanding the semantics of a text, especially in cases where there are multiple radiological findings in the same sentence.

### Knowledge Graph Construction

![Screenshot 2024-07-23 184708](https://github.com/user-attachments/assets/40e2fbb9-1bb7-4e79-9a43-070b6aee0d04)

The goal of this step is to transform each document into a collection of knowledge graphs, where each graph represents a cluster of related entities (Figure 1B). This is a necessary step as knowledge graphs are a more simplified and versatile representation of the original text (Luan et al., 2018). Each entity is represented as a node, and each relation is represented as a directed edge. The RadGraph schema inherently forms directed acyclic graphs (DAGs), although inaccuracies in the model’s output can cause cycles. These DAGs do not necessarily adhere to a tree structure, as a child node can have multiple parent nodes.
Given the specific RadGraph schema, I renamed the "modify" relation to "described as," and reversed the edge direction for all such relations. This modification standardizes the directionality of relations, ensuring that "located at," "described as," and "suggestive of," which connect to additional information, all stem from, or direct outwards from, the radiological finding that is being modified. The resulting graph becomes more navigable, and in subsequent steps, it will facilitate the generation of more natural sentences.

### Subgraph Permutation for Synthetic Query Generation

https://github.com/user-attachments/assets/792a37c0-5841-4103-93d7-4663b9041153

To create a large training set, generating numerous queries, matched documents (i.e. relevant, positive), and unmatched documents (i.e. irrelevant, negative) from a single graph is ideal. There should also be queries of different lengths and complexities to create a robust model. Achieving both objectives doesn't necessitate the usage of the entire graph for the creation of each query; instead, subgraphs can be leveraged to address these requirements.

Most graphs created from the RadGraph inference dataset appear to have one or more roots, or nodes where the indegree is zero. These nodes are usually important findings of a radiology report, but radiologists may also be interested in searching for non-root terms. So, in my approach, I chose not to use a single node as the starting point for subgraph generation.

The bitmasking technique was employed to permute through every possible subgraph of a knowledge graph. Although bitmasking has exponential time complexity, it remains efficient in this task as the number of nodes is relatively small.

For each bitmask, multiple depth-first search traversals were conducted to create synthetic queries. Each node was used as a starting point for the DFS of a subgraph. At each step of the traversal, if the current node is present in the mask, the node’s token value is appended to the synthetic query. If the current node is not present in the mask then its descendants are not visited. When traversing along a directed edge, the label of the relation is also appended. In cases where a node has multiple children, "and" is inserted between each child token’s label in the synthetic query. The choice of DFS is grounded in the idea that the children, or descriptors, of a node, will be closer together in the resulting query, mirroring the ordering in natural language. After the traversal, if any of the nodes of the mask were not visited, the entire mask was discarded. This indicates that the subgraph is not continuous, and therefore, not all modifiers are interrelated. Without this step, the dataset could become dominated by short queries, particularly when dealing with numerous incomplete large subgraphs.

![Screenshot 2024-07-23 185444](https://github.com/user-attachments/assets/e37c6ac5-c76a-4e37-b0b9-165ef97ac9f7)

### Document Augmentation to Create Hard Negatives

![Screenshot 2024-07-23 185510](https://github.com/user-attachments/assets/8c09d23a-1393-4af5-8a76-05a8ca946334)

For a given synthetic query, the original sentence that it was sourced from can be used as its positive sample. However, finding unmatched documents to serve as negative examples poses a harder challenge. Randomly assigning non-positive documents may be too easily distinguishable from positive ones, limiting the effectiveness of the triplet objective function. To provide a more refined training signal, and because the key entities and relations described in a query are known, I chose to augment matched documents to produce a set of challenging negative examples (i.e. sentences that are similar to the positive sample but are not relevant to the query).

One technique for this purpose is Label-wise Token Replacement (LwTR). The LwTR of a given document is implemented as follows: For each entity present in both the matched document and the synthetic query, a binomial distribution is used to determine whether it should be replaced. If an entity is indicated for replacement, it is substituted with another token drawn from a vocabulary in the RadGraph corpus. To ensure the replaced vocabulary is semantically similar to the original term, only tokens that share the same entity and outgoing relation type are considered, expanding upon previous approaches presented, which only took into account the entity type. Additionally, replacement tokens that share the same root as the current token are not considered.

By implementing this method, multiple distinct negative samples are generated for each positive document and query. This technique maximizes the lexical overlap between the query and negative samples while remaining incorrect to the query, making these hard negative training samples. Finally, these queries and associated positive and negative samples are assembled into triplets to serve as weakly-labeled training data.

### PubMedBERT Transfer Learning

![Screenshot 2024-07-23 185804](https://github.com/user-attachments/assets/c76c8f01-c6a8-4778-85b4-f61f76b212d6)

SBERT was employed as the retrieval model. The model was trained using the triplet objective function, where, given a query q, a matched sentence m, and an unmatched sentence u, the network is fine-tuned to ensure that the distance between the embeddings of q and m is smaller than the distance between the embeddings of q and u by a margin ϵ. In other words, the function is defined as max(|| eq − em || − || eq − eu || + ϵ, 0) with eq, em, and eu representing the sentence embeddings for q, m, and u. Per Shi et al. (2022), ||·|| was cosine distance and ϵ = 0.5. At inference, the cosine similarity between the query embedding and the report sentence embedding is used to determine the level of relevance. The SBERT model was initialized with the weights of the Hugging Face PubMedBERT model, pre-trained on a corpus of all PubMed abstracts. Subsequently, it underwent fine-tuning over 7 epochs using the training data, comprising 289,158 triplets, generated by our proposed pipeline. The training process employed the AdamW optimizer with an initial learning rate of 2e-5.

## Model Evaluation

![Screenshot 2024-07-23 185611](https://github.com/user-attachments/assets/b3e362cc-6c9d-4571-a8ce-66bea52329cf)

### Embedding Space Separation

![Screenshot 2024-07-23 185743](https://github.com/user-attachments/assets/3c24e12f-999a-4802-bd7d-26417ac53600)

To assess the quality of the vector space model, I conducted an embedding space separation analysis. An ideal vector space model would demonstrate clear separation in the embedding space for queries with opposite or different meanings (opposite-modifier queries). To make queries of opposite meanings, I concatenate opposite modifiers with 13 common radiological findings. Modifiers fall into 3 categories, locational (e.g. “left” vs. “right”), descriptive (e.g. “streaky” vs. “patchy”), or observational (e.g. “presence of fluid” vs. “possible fluid”). To evaluate the separation, I first find cosine similarity, which is the dot product of two vectors divided by the product of their magnitudes, resulting in a value between -1 and 1 (Figure 3C). Then, the cosine distance measure is simply one minus the cosine similarity. The distance metric produces values between 0 and 2, where 0 signifies identical vectors and 2 indicates entirely dissimilar vectors.  

![Screenshot 2024-07-23 185704](https://github.com/user-attachments/assets/49150c3e-d66c-46ef-ae34-e894efaaaa14)

The distance between opposite-modifier vectors is expected to be very large. As a control, I ensure that the distance between pairs of synonymous vectors (e.g. “evidence of fluid” vs. “there is fluid”) has negligible separation.

### Retrieval Performance

![Screenshot 2024-07-23 185627](https://github.com/user-attachments/assets/d01b8ef4-13ba-4b0e-8f26-1960a57855e4)

One of the primary objectives of this pipeline is to enable the handling of more complex free-text queries. To evaluate the model’s performance across different levels of “complexity,” a measure of query complexity in web IR based on the number of variables (or terms) in a query by Jansen (2000) can be adapted. For this study, I define the metric complexity at N, or C@N, where N is the number of variables in an evaluation query. Examples of queries include “atelectasis” (C1), “bilateral deformities” (C2), “​​calcified upper lobe” (C3), “calcified lung granuloma and thoracic deformity” (C5), and “thoracic atherosclerosis and lower lung cicatrix and pulmonary emphysema and calcified granuloma” (C10).

![Screenshot 2024-07-23 185729](https://github.com/user-attachments/assets/9f74a0a7-30b2-4e05-9e9b-d284e65b7d41)

## Conclusions
