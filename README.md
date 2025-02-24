# GC - VF: Generative and Contrastive self - supervised learning framework for virulence factor identification

## Introduction



GC - VF is a cutting - edge framework designed for the identification of virulence factors, leveraging the power of generative and contrastive self - supervised learning. This repository contains the source code and corresponding datasets necessary to reproduce and build upon the research presented in our work.

## Framework Highlights



GC - VF consists of two fundamental modules that work in tandem to enhance the accuracy of virulence factor identification:

### 1. Generative Attribute Reconstruction Module



This module focuses on learning representations within the attribute space by reconstructing protein features. By doing so, it can effectively capture the intrinsic patterns of the data and mitigate the impact of noise. This is crucial for extracting meaningful information from complex biological data, as it helps in distilling the essential features that contribute to the identification of virulence factors.

### 2. Local Contrastive Learning Module



The local contrastive learning module employs node - level contrastive learning techniques. It aims to capture local features and contextual information with high precision. During the process of aggregating global information, there is often a risk of losing important local details. This module addresses this issue, ensuring that the node representations accurately reflect the characteristics of the individual nodes. This fine - grained approach significantly improves the discriminatory power of the framework for identifying virulence factors.

## Code and Data Availability



The codebase provided in this repository is written in Python and is structured to facilitate easy understanding and modification. The datasets used in our experiments are also included, allowing for direct replication of our results and further exploration of the proposed framework.

## Environment Requirements



To run the code in this repository, please refer to the following Python libraries installed with the specified versions:

| Library Name   | Version      |
| -------------- | ------------ |
| Python         | 3.10.8       |
| dgl            | 2.2.1+cu121  |
| networkx       | 3.2.1        |
| numpy          | 1.26.3       |
| scikit - learn | 1.5.0        |
| torch          | 2.1.2+cu121  |
| torchvision    | 0.16.2+cu121 |