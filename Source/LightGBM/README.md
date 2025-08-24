# H&M Personalized Fashion Recommendations

Liên kết các Google Drive chứa các Embeddings được tạo sẵn.
- [dssm_item_embd.npy](https://drive.google.com/file/d/13rGRbevjcd0yZdwuOTPmNyMOIx9WOLb9/view?usp=sharing)
- [dssm_user_embd.npy](https://drive.google.com/file/d/13nkDc7Dt6QtXx91i3sjnotQNGX2JpSk_/view?usp=sharing)
- [yt_item_embd.npy](https://drive.google.com/file/d/11Q8nWxOlSTspQwH9OGmR9vGoAqJ2wWbS/view?usp=sharing)
- [yt_user_embd.npy](https://drive.google.com/file/d/11OX9vuHmCrCk8Mcl6XA1TF0l0nBL___j/view?usp=sharing)
- [w2v_item_embd.npy](https://drive.google.com/file/d/1-8spKOVtb0jr5xYT8oMKMC5z3BPpCOU-/view?usp=sharing)
- [w2v_user_embd.npy](https://drive.google.com/file/d/1-6CAnA2_pHXrhCyplV-WsI9lreSf6Rm-/view?usp=sharing)
- [w2v_product_embd.npy](https://drive.google.com/file/d/1-R8Rww7QqHZOIcyIhZxEMiXRW1hzJ5wI/view?usp=sharing)
- [w2v_skipgram_item_embd.npy](https://drive.google.com/file/d/1-AmzbyCHx9i0CddZIdbqNJPAMXw3Kg34/view?usp=sharing)
- [w2v_skipgram_user_embd.npy](https://drive.google.com/file/d/1-8BpDQUn310Vns72t1up3uIOOnV_nR4h/view?usp=sharing)
- [w2v_skipgram_product_embd.npy](https://drive.google.com/file/d/1-QhHbFr16koCBL5OIMHxJX9ZAQJAhbHF/view?usp=sharing)
- [image_embd.npy](https://drive.google.com/file/d/1-WkIeInVvHJz4ScA3n-CRyVLQjW51gDH/view?usp=sharing)

Nếu đường dẫn không tồn tại thì chạy mã Embeddings.ipynb để tạo ra các file bị thiếu

Giải pháp bao gồm 2 chiến lược thu hồi và đã đào tạo 3 mô hình xếp hạng khác nhau (xếp hạng LGB, phân loại LGB, DNN) cho mỗi chiến lược.
## Cách sử dụng:
1. Tạo các thư mục dữ liệu theo cấu trúc hiển thị bên dưới và sao chép bốn tệp .csv từ tập dữ liệu cuộc thi Kaggle gốc học trong folder `source/data` vào `data/raw/` (thay đổi cấu trúc đường dẫn cho phù hợp với máy tính của bạn).
2. Các nhúng được đào tạo trước có thể được tạo bằng hoặc bạn có thể tải xuống trực tiếp thông qua các liên kết bên dưới và đặt chúng vào `data/external/`.
3. Chạy Jupyter Notebooks trong `notebooks/`. Lưu ý rằng các tính năng được sử dụng bởi tất cả các mô hình đều được tạo trong phần `Feature Engineering` trong `LGB Recall 1.ipynb`, vì vậy hãy đảm bảo bạn chạy phần đó trước, sau đó là DNN 1 và sau đó là mô hình LightGBM 2 -> DNN 2.
4. Chạy gen_submit.ipynb vào mô hình colab hai và nhận kết quả

Project Organization
------------
   Recommend_system/
    ├── LICENSE
    ├── README.md
    ├── data
    │   ├── external 
    │   ├── index_id_map
    │   ├── interim    
    │   ├── processed      
    │   └── raw       
    ├── docs             
    ├── models            
    ├── notebooks         
    └── src              
        ├── __init__.py    
        ├── data          
        │   ├── datahelper.py
        │   └── metrics.py
        ├── features       
        │   └── base_features.py
        └── retrieval     
            ├── collector.py
            └── rules.py
