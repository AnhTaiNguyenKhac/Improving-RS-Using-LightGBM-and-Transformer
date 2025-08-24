# Improving Fashion Recommendation Systems Using LightGBM and Transformer: Practical Implementation and Evaluation.

## Authors

**Nguyen Khac Anh Tai**  
**Le Tuan Thanh**  
- Faculty of Information Technology, Ton Duc Thang University  

Scientific Supervisor: **Dr. Tran Trung Tin**

---

## Thesis Overview

This thesis investigates and implements two **independent approaches** aimed at improving the accuracy and scalability of fashion recommendation systems:

- **The first approach leverages LightGBM**, a highly efficient gradient boosting framework. It is applied to model the probability of user–item interactions using metadata and behavioral history. LightGBM demonstrates strong performance on sparse data, offering fast training, scalability to large datasets, and competitive predictive accuracy.

- **The second approach explores Transformer-based Graph Neural Networks**, implemented through the proposed architecture **Visual-TransGNN**. This model combines GNNs with Transformer attention mechanisms, incorporating **Positional Encoding** to enrich graph structural information and **visual features** extracted via CNNs to enhance personalization. By applying Transformer attention to graphs, the model flexibly expands the receptive field and identifies influential nodes more effectively.

Together, these models represent two modern paradigms: **boosted decision tree learning (LightGBM)** and **deep learning with attention mechanisms (Transformer)**. They are compared and evaluated to highlight their respective advantages and limitations in the context of recommendation systems.

---

## Key Results

- On the **Vibrant Clothes Rental dataset**:  
  - **Visual-TransGNN** achieved:  
    - **Recall@20 = 0.12773**  
    - **HitRate@20 = 0.51301**  
  - Demonstrated significant improvements over LightGBM, particularly in Top-K recommendation metrics.

- On the **H&M Fashion dataset**:  
  - **LightGBM** showed robust generalization to large-scale data, with fast processing and stable performance.  
  - **Visual-TransGNN** consistently outperformed in metrics such as **MRR** and **NDCG**, especially in exploiting structural and visual information.

---

## Conclusion

The thesis successfully implemented **two state-of-the-art recommendation models**:  
- **LightGBM**, which proved highly effective on structured data, suitable for systems prioritizing speed in training and prediction.  
- **Visual-TransGNN**, which effectively combined graph structure and visual features through Transformer attention, yielding superior performance in personalized fashion recommendations.

The project provided practical experience in building pipelines, handling data, designing models, and optimizing/evaluating performance, while deepening knowledge of **LightGBM and Transformer architectures**—two essential tools in modern machine learning.  

The results highlight the importance of selecting models aligned with dataset characteristics, and demonstrate strong potential for deployment in **large-scale e-commerce recommendation systems** tailored to user preferences and behaviors.

---
