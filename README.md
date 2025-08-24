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

- On the **Vibrent Clothes Rental dataset**:  
  - **Visual-TransGNN** achieved:  
    - **Recall@20 = 0.12773**  
    - **HitRate@20 = 0.51301**  
  - Demonstrated significant improvements over LightGBM, particularly in Top-K recommendation metrics.

- On the **H&M Fashion dataset**:  
  - **LightGBM** showed robust generalization to large-scale data, with fast processing and stable performance.  
  - **Visual-TransGNN** consistently outperformed in metrics such as **MRR** and **NDCG**, especially in exploiting structural and visual information.

---

## Experimental Results

### Top-5 Results

| Model           | Recall (Vibrent) | NDCG (Vibrent) | MRR (Vibrent) | Hit Rate (Vibrent) | Recall (H&M) | NDCG (H&M) | MRR (H&M) | Hit Rate (H&M) |
|-----------------|------------------|----------------|---------------|---------------------|--------------|------------|-----------|----------------|
| BPR-MF          | 0.00137          | 0.01175        | 0.00891       | 0.02174             | 0.00302      | 0.02675    | 0.02444   | 0.04592        |
| NGCF            | 0.00082          | 0.01373        | 0.01330       | 0.02174             | 0.00010      | 0.00012    | 0.00010   | 0.00015        |
| LightGCN        | 0.00125          | 0.00371        | 0.00011       | 0.00023             | 0.00162      | 0.00473    | 0.00010   | 0.00018        |
| Image-embed     | 0.00731          | 0.01020        | 0.01430       | 0.04541             | 0.04193      | 0.03712    | 0.02730   | 0.06777        |
| Visual-GCN      | 0.01269          | 0.01738        | 0.02369       | 0.06605             | 0.02837      | 0.04146    | 0.01616   | 0.08327        |
| LightGBM        | 0.05101          | 0.02984        | 0.06981       | 0.09448             | 0.05668      | 0.03872    | 0.05480   | 0.09044        |
| Visual-TransGNN | **0.07213**      | **0.09331**    | 0.03856       | **0.20875**         | 0.04951      | 0.05103    | 0.02225   | 0.09392        |

---

### Top-20 Results

| Model           | Recall (Vibrent) | NDCG (Vibrent) | MRR (Vibrent) | Hit Rate (Vibrent) | Recall (H&M) | NDCG (H&M) | MRR (H&M) | Hit Rate (H&M) |
|-----------------|------------------|----------------|---------------|---------------------|--------------|------------|-----------|----------------|
| BPR-MF          | 0.00417          | 0.01902        | 0.01145       | 0.06957             | 0.00879      | 0.04802    | 0.03079   | 0.11291        |
| NGCF            | 0.00516          | 0.01333        | 0.00686       | 0.05021             | 0.00001      | 0.00018    | 0.00012   | 0.00038        |
| LightGCN        | 0.00593          | 0.00537        | 0.00016       | 0.00085             | 0.00449      | 0.00460    | 0.00013   | 0.00048        |
| Image-embed     | 0.04591          | 0.01719        | 0.02382       | 0.23982             | 0.05298      | 0.05864    | 0.03155   | 0.08447        |
| Visual-GCN      | 0.07151          | 0.02556        | 0.03492       | 0.30823             | 0.03950      | 0.03883    | 0.01864   | 0.09570        |
| LightGBM        | 0.06927          | 0.03064        | 0.07686       | 0.14173             | 0.06990      | 0.04465    | 0.06017   | 0.14410        |
| Visual-TransGNN | **0.12773**      | **0.11068**    | 0.04235       | **0.51301**         | **0.07221**  | **0.07064**| 0.03168   | **0.14350**    |


## Conclusion

The thesis successfully implemented **two state-of-the-art recommendation models**:  
- **LightGBM**, which proved highly effective on structured data, suitable for systems prioritizing speed in training and prediction.  
- **Visual-TransGNN**, which effectively combined graph structure and visual features through Transformer attention, yielding superior performance in personalized fashion recommendations.

The project provided practical experience in building pipelines, handling data, designing models, and optimizing/evaluating performance, while deepening knowledge of **LightGBM and Transformer architectures**—two essential tools in modern machine learning.  

The results highlight the importance of selecting models aligned with dataset characteristics, and demonstrate strong potential for deployment in **large-scale e-commerce recommendation systems** tailored to user preferences and behaviors.

---


