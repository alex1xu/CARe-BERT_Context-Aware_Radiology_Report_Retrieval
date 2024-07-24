# CARe-BERT: BERT-Powered Graph Augmentation for Context-Aware Radiology Report Retrieval

> A new approach to training radiology report retrieval large language models without the need for time-consuming, manual data annotation by radiologists.

[![Codacy Badge](https://app.codacy.com/project/badge/Grade/5b74f79ee2904bfc92cc90fdbfdd3421)](https://app.codacy.com/gh/alex1xu/VapeVeritas-Twitter_E-cig_Surveillance/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](https://jupyter.org/try)

## 🚩 TL;DR

<details>
  <summary><b>Spoiler</b></summary>
  In this study, I introduce a <b>new approach to the weakly-supervised learning based radiology report retrieval task</b>. Weakly-supervised approaches, like CARe-BERT, are important as they <b>circumvent the resource-intensive nature of traditional supervised learning</b>. This novel pipeline leverages named entity and relationship extraction techniques to construct comprehensive knowledge graph representations of radiology reports. Using bitmasking, subgraphs are systematically permuted to synthesize complex queries. Subsequently, the application of Label-wise Token Replacement augments original reports to create challenging negative samples for training. The triplet objective function is then used to train an SBERT model on this generated data. <b>CARe-BERT surpasses established benchmark approaches and is comparable with recent models proposed in recent literature in terms of mean average precision and mean recall</b>. Embedding space separation analysis demonstrates <b>advancement in the semantic comprehension of anatomical positions, negation, and condition descriptions</b>. It is also the <b>first model to be explicitly trained for various levels of query complexity</b>, showing greater improvement over benchmarks as query complexity increases. While CARe-BERT does not yet possess the level of robustness and accuracy required for direct deployment in healthcare settings, it serves as a blueprint for the development of more sophisticated, weakly-supervised approaches.
</details>

## Table of Contents

- [Why?](#why)
- [Novel Pipeline Overview](#novel-pipeline-overview)
- [Model Evaluation](#model-evaluation)
- [Embedding Space Separation](#embedding-space-separation)
- [Retrieval Performance](#retrieval-performance)

## Why?

You may have had chest X-rays taken before, but what you might not know is that afterward, radiologists will detail their observations in a free-text document called a radiology report. These reports are subsequently queried for research, educational, and medical use.

In my work at a university hospital, researchers often spend hundreds of hours reviewing report databases for cohort studies. In my work as an emergency medical technician, quick access to relevant patient reports is equally critical, so contributing to research in this field is important to me.

![Screenshot 2024-07-23 193745](https://github.com/user-attachments/assets/4f57d3ea-3fc3-4916-9eee-9b7936c20f42)

In natural language processing research, developing software to more accurately automate this query process is called the radiology report retrieval task.

_Given a free text query and a collection of radiology reports, how can we create some algorithm that returns the reports relevant to the query?_

![Screenshot 2024-07-23 193754](https://github.com/user-attachments/assets/90484424-8df6-48f8-96e9-82c16899fd0f)

**Vector space** approaches are a major focus of current research.

Let’s imagine a high-dimensional space. Vector Space models represent each report and query as a vector, or embedding, in this space where the position stores the text’s meaning. The cosine of the angle between the embeddings of reports to the query represents their similarity. Ideally, the related queries and reports are close while irrelated pairs are distant.

https://github.com/user-attachments/assets/c0aeec47-f57a-4183-8401-9278081a097f

The **Sentence-BERT**, or SBERT, transformer model has achieved state-of-the-art performance in the vector space approach.

It pools learned word embeddings into a single sentence embedding. Then, sentence embeddings are computed for three sample documents: a query text, a positive, or relevant report, and a negative, or irrelevant report. Through the triplet objective loss function, the distance between query and positive sentence embeddings is minimized, while the distance between query and negative sentence embeddings is maximized.

However, datasets of these query, positive, and negative training triplets are **currently small and restricted as they require the expertise of radiologists to create**, which is hindering the advancement of better models.

Motivated by this challenge, researchers have developed **weakly supervised learning** approaches to **programmatically generate training triplets**. Current approaches can handle queries consisting of the presence or absence of a single radiological finding.

## Novel Pipeline Overview

I improved upon existing approaches by creating **CARe-BERT**, a program to generate training triplets that teach SBERT to handle queries containing **multiple radiological findings**, and **the context of findings**, including anatomical and descriptive modifiers.

![Screenshot 2024-07-23 185824](https://github.com/user-attachments/assets/46e8fa05-08fd-4d82-a01d-8b0a9f0f3acd)

### Named Entity and Relationship Extraction

The first stage is Named Entity Recognition (NER) and Relation Extraction (RE), which **identify the clinically relevant terms** that radiologists may be interested in querying. I utilize the RadGraph CXR radiology report inference dataset labeled using a **joint NER and RE DyGIE++ model**, initialized with PubMedBERT weights, as developed by Jain et al. (2021). Mentioned findings and anatomy in a report are identified by the model. Findings are identified in three levels of radiologists’ confidence: present, uncertain, and absent. It also identifies the relationships between these entities, including suggestive of, located at, and described as.

![Screenshot 2024-07-23 184628](https://github.com/user-attachments/assets/c6e7af64-9cf5-494a-b667-69ce3ac8b161)

Past methods to extract entities for training IR models have relied on lexicon or rule-based techniques. However, these methods are vulnerable to issues like spelling mistakes, form variability (e.g., "fluid in lungs" and "pulmonary edema" describe the same condition), and ambiguous abbreviations (e.g., does "LLQ" pertain to "left lower quadrant" or "left lower lobe," or neither). Furthermore, lexicon-based approaches often fall short of capturing the relationships between entities, which are crucial for understanding the semantics of a text, especially in cases where there are multiple radiological findings in the same sentence.

### Knowledge Graph Construction

The goal of this step is to transform each document into a collection of knowledge graphs, where each graph represents a cluster of related entities. This is a necessary step as knowledge graphs are a more simplified and versatile representation of the original text. Each entity is represented as a node, and each relation is represented as a directed edge.

Given the specific RadGraph schema, I renamed the "modify" relation to "described as," and reversed the edge direction for all such relations. This modification standardizes the directionality of relations, ensuring that "located at," "described as," and "suggestive of," which connect to additional information, all stem from, or direct outwards from, the radiological finding that is being modified. The resulting graph becomes more navigable, and in subsequent steps, it will facilitate the generation of more natural sentences.

![Screenshot 2024-07-23 184708](https://github.com/user-attachments/assets/40e2fbb9-1bb7-4e79-9a43-070b6aee0d04)

### Subgraph Permutation for Synthetic Query Generation

To create a large training set, generating numerous queries, matched documents (i.e. relevant, positive), and unmatched documents (i.e. irrelevant, negative) from a single graph is ideal. There should also be queries of different lengths and complexities to create a robust model. Achieving both objectives doesn't necessitate the usage of the entire graph for the creation of each query; instead, **subgraphs can be leveraged** to address these requirements. Most graphs created from the RadGraph inference dataset appear to have one or more roots, or nodes where the indegree is zero. These nodes are usually important findings of a radiology report, but radiologists may also be interested in searching for non-root terms. So, in my approach, I chose not to use a single node as the starting point for subgraph generation.

The **bitmasking** technique was employed to permute through every possible subgraph of a knowledge graph. Although bitmasking has exponential time complexity, it remains efficient in this task as the number of nodes is relatively small.

https://github.com/user-attachments/assets/792a37c0-5841-4103-93d7-4663b9041153

For each bitmask, multiple **depth-first search traversals were conducted to create synthetic queries**. Each node was used as a starting point for the DFS of a subgraph. At each step of the traversal, if the current node is present in the mask, the node’s token value is appended to the synthetic query. If the current node is not present in the mask then its descendants are not visited. When traversing along a directed edge, the label of the relation is also appended. In cases where a node has multiple children, "and" is inserted between each child token’s label in the synthetic query. The choice of DFS is grounded in the idea that the children, or descriptors, of a node, will be closer together in the resulting query, mirroring the ordering in natural language. After the traversal, if any of the nodes of the mask were not visited, the entire mask was discarded. This indicates that the subgraph is not continuous, and therefore, not all modifiers are interrelated. Without this step, the dataset could become dominated by short queries, particularly when dealing with numerous incomplete large subgraphs.

![Screenshot 2024-07-23 185444](https://github.com/user-attachments/assets/e37c6ac5-c76a-4e37-b0b9-165ef97ac9f7)

### Document Augmentation to Create Hard Negatives

For a given synthetic query, the original sentence that it was sourced from can be used as its positive sample. However, finding unmatched documents to serve as negative examples poses a harder challenge. Randomly assigning non-positive documents may be too easily distinguishable from positive ones, limiting the effectiveness of the triplet objective function. To provide a more refined training signal, and because the key entities and relations described in a query are known, I chose to augment matched documents to produce a set of challenging negative examples (i.e. sentences that are similar to the positive sample but are not relevant to the query).

![Screenshot 2024-07-23 185510](https://github.com/user-attachments/assets/8c09d23a-1393-4af5-8a76-05a8ca946334)

One technique for this purpose is **Label-wise Token Replacement** (LwTR). The LwTR of a given document is implemented as follows: For each entity present in both the matched document and the synthetic query, a binomial distribution is used to determine whether it should be replaced. If an entity is indicated for replacement, it is substituted with another token drawn from a vocabulary in the RadGraph corpus. To ensure the replaced vocabulary is semantically similar to the original term, only tokens that share the same entity and outgoing relation type are considered, expanding upon previous approaches presented, which only took into account the entity type. Additionally, replacement tokens that share the same root as the current token are not considered.

By implementing this method, **multiple distinct negative samples are generated for each positive document and query**. This technique maximizes the lexical overlap between the query and negative samples while remaining incorrect to the query, making these hard negative training samples. Finally, these queries and associated positive and negative samples are assembled into triplets to serve as weakly-labeled training data.

![Screenshot 2024-07-23 185804](https://github.com/user-attachments/assets/c76c8f01-c6a8-4778-85b4-f61f76b212d6)

### PubMedBERT Transfer Learning

SBERT was employed as the retrieval model. The model was trained using the triplet objective function, where, given a query q, a matched sentence m, and an unmatched sentence u, the network is fine-tuned to ensure that the distance between the embeddings of q and m is smaller than the distance between the embeddings of q and u by a margin ϵ. In other words, the function is defined as max(|| eq − em || − || eq − eu || + ϵ, 0) with e<sub>q</sub>, e<sub>m</sub>, and e<sub>u</sub> representing the sentence embeddings for q, m, and u. Per Shi et al. (2022), ||·|| was cosine distance and ϵ = 0.5. At inference, the cosine similarity between the query embedding and the report sentence embedding is used to determine the level of relevance. The SBERT model was initialized with the weights of the Hugging Face PubMedBERT model, pre-trained on a corpus of all PubMed abstracts. Subsequently, it underwent fine-tuning over 7 epochs using the triplets generated by the proposed pipeline. The training process employed the AdamW optimizer with an initial learning rate of 2e<sup>-5</sup>.

## Model Evaluation

![Screenshot 2024-07-23 185611](https://github.com/user-attachments/assets/b3e362cc-6c9d-4571-a8ce-66bea52329cf)

Model evaluation was conducted using a publicly available collection of de-identified radiology reports obtained through the Open-I API, sourced from Indiana University (IND) and provided by the National Institutes of Health (NIH).

For comparison, I implemented the benchmark models:
 - BM25, or Okapi BM25, the standard ranking function used by many search engines
 - SBERT fine-tuned with the MS MARCO dataset
 - BioClinicalBERT
 - PubMedBERT, which serves as a baseline (no training on the novel pipeline data)

## Embedding Space Separation

To assess the quality of the vector space model, I conducted an embedding space separation analysis. An ideal vector space model would demonstrate clear separation in the embedding space for queries with opposite or different meanings (opposite-modifier queries). To make queries of opposite meanings, I concatenate opposite modifiers with 13 common radiological findings. Modifiers fall into 3 categories, locational (e.g. “left” vs. “right”), descriptive (e.g. “streaky” vs. “patchy”), or observational (e.g. “presence of fluid” vs. “possible fluid”). To evaluate the separation, I use the cosine distance, or one minus the cosine similarity. The distance metric produces values between 0 and 2, where 0 signifies identical vectors and 2 indicates entirely dissimilar vectors.

![Screenshot 2024-07-23 185743](https://github.com/user-attachments/assets/3c24e12f-999a-4802-bd7d-26417ac53600)

As a control, I ensure that the distance between pairs of synonymous vectors (e.g. “evidence of fluid” vs. “there is fluid”) has negligible separation.

![Screenshot 2024-07-23 185704](https://github.com/user-attachments/assets/49150c3e-d66c-46ef-ae34-e894efaaaa14)

### Stronger Semantic Understanding of Locational Terms but Weaker Understanding of Descriptive Terms

CARe-BERT shows considerable **strength in handling anatomical/spatial terms** but **falls short in comprehending descriptive/severity terms**, as indicated by the 95% confidence intervals overlapping with zero. SBERT (MS MARCO) surpasses our model in this regard, suggesting that these descriptive terms may be used more commonly in non-medical texts. Otherwise, CARe-BERT exhibits consistent improvements over the benchmark models in embedding space separation, highlighting the success of this novel pipeline approach in enhancing the semantic understanding of the model.

Banerjee et al. (2017) introduced a two-stage approach that combines semantic dictionary mapping with word2vec embeddings to capture both domain-specific terminology and general semantic context. Their approach achieved state-of-the-art cosine distances in the range of 1.000 to 1.300 in distinguishing severity-related terms. This **outperforms CARe-BERT’s ability to distinguish between severity-related modifiers by a wide margin**. However, CARe-BERT was trained and evaluated on a wider variety of terms, including anatomical positions, orientations, quantity, and other descriptions of anatomical findings. In general, this model operates within a more extensive context, encompassing a broader range of terminology beyond a predefined vocabulary, resulting in a "sparser" vector space with higher dimensionality, a characteristic associated with decreased performance.

## Retrieval Performance

One of the primary objectives of this pipeline is to enable the handling of more complex free-text queries. To evaluate the model’s performance across different levels of “complexity,” a measure of query complexity in web IR based on the number of variables (or terms) in a query by Jansen (2000) can be adapted. For this study, I define the metric complexity at N, or C@N, where N is the number of variables in an evaluation query.

![Screenshot 2024-07-23 185627](https://github.com/user-attachments/assets/d01b8ef4-13ba-4b0e-8f26-1960a57855e4)

My approach, CARe-BERT, was evaluated on Mean Average Precision (mAP), the model's ability to distinguish between relevant and irrelevant documents and Mean Recall (mR), the proportion of relevant documents that are retrieved. For all metrics, a score of 1 is ideal and a score of 0 indicates poor model performance. 

![Screenshot 2024-07-23 185729](https://github.com/user-attachments/assets/9f74a0a7-30b2-4e05-9e9b-d284e65b7d41)

### CARe-BERT Outperforms BM25, SBERT (MS MARCO), PubMedBERT, and BioClinicalBERT at All Levels of Query Complexity

CARe-BERT exhibits **consistent improvements over the benchmark models at all levels of complexity in retrieval metrics**, showing its enhanced semantic understanding of the model likely improved its radiology report retrieval ability. Across all models, there is a marked decrease in mAP and mR from C1 to C2. Nevertheless, **CARe-BERT showed the most gradual decline in mAP and mR with increasing complexity**. CARe-BERT’s performance at greater levels of complexity was even on par with benchmark models’ performances at lower levels of complexity, further evidence that the proposed pipeline enhances the model’s ability to understand the context of radiological findings and handle more complex queries.

These findings suggest that while there is still ample opportunity for improvement, the proposed pipeline enhances model retrieval quality beyond the benchmarks at the presented complexity levels. This is currently the **only IR study that explicitly trains a model to handle queries of varying levels of complexity** and provides empirical evidence of its performance across these complexity levels.

​​The performance of CARe-BERT is also balanced, indicated by the similarity in mAP and mR scores at all complexity levels. Since CARe-BERT is primarily designed as a retrieval model, greater emphasis should be placed on mR. To do this, it may be beneficial to consider a higher negative-to-positive ratio in the training data. Some studies have indicated that employing a batch configuration with only one positive sentence can improve the model’s ability to discern the most important terms in a positive sample, leading to improved recall in weakly-supervised learning, as observed in the work of Zhao et al. (2019).

### Comparable Retrieval Performance and Improved Negation Comprehension Compared to Recent Approaches at C1

Several studies target the C1 objective, where the query simply entails the presence or absence of a single radiological finding. Shi et al. (2022) proposed a weakly-supervised training pipeline for the retrieval of CXR radiology reports. Their approach used a CXR lexicon to identify entities (observations or conditions) within sentences. It incorporated the longest common subfix (LCF) algorithm and employed a two-step language structuring, vocabulary-based negation detection method to assess the positivity or negativity of medical findings. These identified entities subsequently served as synthetic queries, with the sentences containing them considered as positive examples, while all other sentences potentially acted as negative examples. Shi et al. further employed SBERT with the triplet objective training function, hard sampling, and mega batching to generate embeddings. 

**CARe-BERT has slightly lower mAP**, showing a 6% increase over BM25, compared to Shi et al.’s reported 8% increase. Our models both show a **comparable 7% increase in mR** over BM25. These results indicate that the CARe-BERT pipeline approach is **comparable to other weakly-supervised approaches** in training a model to discern the presence or absence of a single radiological finding within a text. When analyzing the embedding space separation between opposite-negation queries (e.g. “Observed calcifications” v.s. “No calcifications”), CARe-BERT achieved a notable 0.317 increase (equivalent to a 75.5% improvement) in cosine distance over Shi et al.’s approach, indicating that the **CARe-BERT pipeline enhances semantic understanding**. It is worth noting that CARe-BERT was also explicitly trained to recognize uncertain or suggested findings (i.e. three levels of observations instead of two), making determining the presence or absence of a finding an even more nuanced and challenging task. 
