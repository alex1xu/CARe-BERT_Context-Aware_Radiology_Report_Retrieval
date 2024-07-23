# CARe-BERT: BERT-Powered Graph Augmentation for Context-Aware Radiology Report Retrieval

> How are factors like marketing strategies, appealing flavors, and the lack of centralized authorities contributing to the positive perception of electronic cigarettes on Twitter?

[![Codacy Badge](https://app.codacy.com/project/badge/Grade/5b74f79ee2904bfc92cc90fdbfdd3421)](https://app.codacy.com/gh/alex1xu/VapeVeritas-Twitter_E-cig_Surveillance/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade) ![Lines of code](https://img.shields.io/tokei/lines/github/alex1xu/VapeVeritas-Twitter_E-cig_Surveillance)
 ![Static Badge](https://img.shields.io/badge/%20-%20?style=flat&label=what%20is%20vaping%3F&labelColor=%23C841F9&color=%23555555&link=https%3A%2F%2Fwww.dshs.texas.gov%2Fvaping%2Fwhat-is-vaping)

[![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](https://jupyter.org/try)

## 🚩 TL;DR

<details>
  <summary><b>Spoiler</b></summary>
  This study is the most comprehensive surveillance of e-cigarette perception on social media to date (8-year analysis, 18x previous largest study size). Natural Language Processing analysis of Twitter data indicates the success of e-cigarette brands in creating a positive image of their products among Twitter users. Factors potentially contributing to this phenomenon include marketing strategies, flavors, social appeal, the presence of echo chambers, the absence of central authorities, and the lack of implementation of Tobacco-21 legislation. There were also observed significant changes in tweet patterns during headline events, such as the E-cigarette Use Associated Lung Injury outbreak. This is also the first study to identify the long-term, ​​growing polarization of e-cig opinions and to quantify the dynamics of e-cig information dissemination. This understanding of the dynamics surrounding e-cig conversations will guide policymakers and health organizations in implementing more effective preventive and cessation strategies to address the e-cig epidemic.
</details>

## Table of Contents

- [Why?](#why)
- [Overview](#overview)
- [Sentiment Analysis](#sentiment-analysis)
- [Topic Modeling](#topic-modeling)
- [Location and Emotion Analysis](#location-and-emotion-analysis)
- [Information Dissemination](#information-dissemination)
- [Chrome Extension](#chrome-extension)
- [Conclusions](#conclusions)

## Why?

Alarming statistics show that 7.7% of American high schoolers vape, prompting the CDC to label e-cigarettes (ECs) an epidemic. This usage may be due to social media, which often portrays JUUL positively, fueled by misinformation from social bots, paid influencers, and peers. Over half of youth vapers are unaware of e-cigs containing nicotine. Studies link exposure to e-cig ads on platforms like Facebook to increased usage. 

Understanding people’s perception of e-cigs, what information is spread, and how it is spread is crucial for policymakers and health organizations to inform intervention strategies. Twitter, which was rebranded to X after the completion of my study, offers an extensive, real-time, corpus of user-generated content. Its user base also aligns with the demographics of vapers. These features are advantageous over traditional surveys.

To understand this large corpus of text, researchers turned to natural language processing, which is a field of AI that understands textual data. Over the past 5 years, numerous studies, including extensive work by USC Professor Jon-Patrick Allem, focus on specific health or legislative events and usually involve a single NLP technique or manual review. Important studies guiding recent legislation, like Jackler et al.'s of Stanford Tobacco Research Group claim that flavors and peer-to-peer marketing on Twitter drove JUUL’s dominance have not yet been quantitatively validated.

## Overview

I scraped tweets to find matching e-cig-related keywords and specific criteria to avoid limits set by the Twitter API. After obtaining basic tweet information, tweets were rehydrated to obtain their latent features such as follower lists, retweets, and location.

![Tweet extraction flowchart](images/vvfig1.png)

This process resulted in 9 million tweets. After filtering out bots and duplicates, I focused on 8 million tweets from 2015 to 2023, a dataset 8 times longer and 18 times larger than the prior largest study in 2021.

![Tweet volume histogram](images/vvfig2.png)
_Histogram of scraped tweets over time_

Right away we notice several peaks, corresponding to major events. There have been studies specifically dedicated to the EVALI outbreak which is the largest peak. The volume of recent tweets is less but increasing.

## Sentiment Analysis

The first analysis I performed was the sentiment analysis task. I used the Valence Aware Dictionary for Sentiment Reasoning (VADER), a rule-based model that categorizes tweets into positive, negative, or neutral sentiments and gauges sentiment intensity. VADER is attuned to social media analysis and understands emojis, slang, and abbreviations.

![Sentiment over time histogram](images/vvfig3.png)
_Histogram of scraped tweets over time stratified by sentiment class_

Overall there were 60% more tweets expressing positive than negative sentiment, which is consistent with existing literature, and suggests that e-cigarette brands ​​may have effectively created a positive image of their products among Twitter users. During the EVALI outbreak, however, the proportion of tweets expressing negative sentiment rose by 128%, likely due to increased media attention drawing more e-cigarette opponents into Twitter discussions.

In this same analysis, we can observe several trends suggesting that there is increased polarization of discussions on Twitter. First, the proportion of neutral tweets, which are more likely to be factual and non opinionated, has been markedly less in recent years.

![Sentiment over time scatterplot](images/vvfig6.png)
_B) shows mean tweet sentiment over time stratified by sentiment class. Bar chart compares mean tweet sentiment of positive and negative sentiment class tweets_

Now instead of just the volume, we can look at the mean sentiment intensity on a day-by-day basis. Let's add a trendline. The R-squared values of the trend lines are low, meaning the date alone does not explain the variation in mean sentiment well, but this is expected. Overall, tweets expressing positive sentiment intensify by about 0.2% each year, and those expressing negative intensify by about .9% each year. Together with the p-value, the data suggests that there is weak and small, but still statistically significant polarization in sentiment over time. 

During EVALI, there was even more polarization in negative sentiment, which is expected.

Additionally, analyzing total mean intensity by sentiment we see that positive sentiment tweets had a small, but significantly greater sentiment intensity scores than negative sentiment tweets suggesting that those with positive sentiments are more vocal about their opinions on Twitter.

## Topic Modeling

Next, I used Latent Dirichlet Allocation (LDA) to identify the topics among JUUL tweets. LDA identified 20 topics in the full corpus and then I manually vetted them for relevance and generalizability and named each with an identifying phrase.

![Topic bar chart](images/vvfig7.png)
_Frequency of top 5 tweet topics stratified by sentiment class_

For tweets expressing positive sentiment, the top topics were vaping products, ​​social and community appeal, marketing, marijuana, and smoking cessation. Continuous exposure to tweets about new vaping products and marketing may contribute to user addition and catalyze future e-cig usage, which is a huge concern. For tweets expressing negative sentiment, the top two topics by far were health risks and anti-e-cigarette legislative action. 

![Topic histogram](images/vvfig8.png)
_Histogram of tweet topics over time stratified by sentiment class_

There is heavy marketing and talks about flavors from 2015 to 2016. The prominence of these topics during JUUL's launch years for the first time quantitatively validates Jackler's 2019 hypothesis, which proposed that JUUL's rapid growth was primarily driven by its youth appeal through social media influencer promotions and enticing flavor options. More recently, there have been increasing talks of marijuana in e-cigarette-related conversations. 

For tweets expressing negative sentiment, banning and health risks completely dominate, especially whenever there is mainstream media coverage, like the EVALI outbreak and JUUL’s MDO. This is a sign of productive conversation, and bringing more of it to the positive sentiment population would be beneficial.

## Location and Emotion Analysis

![US state sentiment map](images/vvfig9.png)
_Mean tweet sentiment by state_

Next, I analyzed sentiment by region. Only four states had a negative mean sentiment score: North Dakota, Louisiana, Alabama, and Maine. Three out of these four states with negative mean sentiment intensity scores have passed Tobacco-21 (T21) legislation, a campaign aimed to raise the legal age of purchase of nicotine products to 21.

![T21 sentiment bar chart](images/vvfig10.png)

In general, states that have acted upon T21 show significantly lower mean sentiment intensity scores compared to states that have not. This indicates a potential heightened awareness of the product's health risks among adolescents when T21 legislation is on the table.

I fine-tuned a BERT transformer model on an annotated emotion dataset to classify each tweet into the emotion categories of joy, anger, fear, and neutral. 

![T21 emotion bar chart](images/vvfig11.png)

In states that have acted upon T21, the proportion of tweets that express neutral emotions is 20% less, and the proportion of tweets that express anger and fear is 20% greater. The association between emotions and T21 status is statistically significant (chi-square=128.19, p<0.001). Again, this is a further indication of polarization. It may also reflect a growing awareness of the risks associated with e-cig use among individuals who were previously neutral or uninformed. 

## Information Dissemination

To quantify the spread and structure of information dissemination I looked at diffusion trees. These are graphs where each node is a user, and an edge indicates a retweet. However, on Twitter, retweet references point to the original tweet and not the tweet that was directly responded to. So, I adopted the approach developed by Liang et al. (2019) by inferring the retweet relationships using users’ follower lists and retweet datetimes.

![Example diffusion trees](images/vvfig12.png)

In total, I generated 245 diffusion trees from the last two months of November 2022. The structure of a tree can be classified as broadcast or peer-to-peer based on its structural virality, which is the average pairwise distance between nodes. Broadcast indicates that there is a central source disseminating information to many users, while peer-to-peer indicates that information is spreading virally through many connections. Normalized structural virality had a mean of 0.104, indicating broadcast. Yet, this value is still twice as viral as Liang et al’s recorded value of 0.05 in the context of Ebola. This suggests that information about EC is disseminated more extensively through peer-to-peer channels compared to other epidemics, rather than through coverage by health organizations like the CDC, which tend to disseminate information in a broadcast fashion.

![Diffusion metrics bar chart](images/vvfig13.png)

_Diffusion tree metrics_

It is more nuanced when broken down by sentiment intensity. Negative sentiment diffusion trees spread with higher structural virality, network size, and network height compared to positive sentiment trees. So overall, negative sentiment diffusion trees reached a larger audience and tended to spread more virally compared to positive sentiment diffusion trees. Additionally, further analysis showed that 90% of all direct retweets shared the sentiment of the original tweet. Each tweet further removed from the original discussion had a 1.2 times increase in the likelihood of disagreement with the original tweet. These factors all indicate the presence of echo chambers within EC-related discussions. These structures may reinforce users' existing beliefs and make them less receptive to alternative perspectives. 

## Chrome Extension

Finally, I created an application to demonstrate how social media platforms can implement preventative strategies. The Chrome extension monitors the user’s Twitter feed for EC-related information. Once it is detected, information is sent to a backend Python Flask application. VADER and precomputed LDA topics are used to determine whether a post should be hidden and what sources of credible information should be provided instead. 

[AntiJUULTwitterDemo (1).webm](https://github.com/alex1xu/hsr23-analysis/assets/65417426/4d883624-f0f9-420a-8fde-fca1e2506730)

## Conclusions

As the most comprehensive study to date, this study marks a significant milestone in understanding the landscape of the EC epidemic through social media surveillance. Analysis indicates the success of brands in creating a positive image of ECs on Twitter. Factors potentially contributing to this phenomenon include marketing strategies, flavors, social appeal, the presence of echo chambers, the absence of central authorities, and the lack of implementation of T21 legislation. There were also observed significant changes in tweet patterns during headline events, such as the EVALI. 

This is also the first study to identify the long-term, ​​growing polarization of EC opinions and to quantify the dynamics of EC information dissemination. This understanding of the dynamics surrounding EC conversations will guide policymakers and health organizations in implementing more effective preventive and cessation strategies to address the EC epidemic. In addition to these insights, I developed a plugin that can be integrated into social media platforms to facilitate corrective actions. Beyond this study, this research incorporated many methods to paint a complete picture of EC discourse and serves as a comprehensive framework for future social media surveillance studies.
