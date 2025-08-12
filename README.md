# UltraFastFoldIn

The code for the paper "Ultra Fast Warm Start Solution for Graph Recommendations", which provide ultra fast updates of recommendations for the UltraGCN graph-based recommender system. 

In this work, we present a fast and effective Linear approach for updating recommendations in a scalable graph-based recommender system UltraGCN. Solving this task is extremely important to maintain the relevance of the recommendations under the conditions of a large amount of new data and changing user preferences. To address this issue, we adapt the simple yet effective low-rank approximation approach to the graph-based model. Our method delivers instantaneous recommendations that are up to $30$ times faster than conventional methods, with gains in recommendation quality, and demonstrates high scalability even on the large catalogue datasets.

We utilize the following versions of libraries:

numpy==1.26.4

pandas==2.0.3

torch==2.5.1+cu124

sklearn==1.6.0

scipy==1.11.4
