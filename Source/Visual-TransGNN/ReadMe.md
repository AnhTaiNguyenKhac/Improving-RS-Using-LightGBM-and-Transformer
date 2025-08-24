# Visual-TransGNN: Mô hình Khuyến nghị Đa phương thức dựa trên GNN và Transformer

## Mô tả tổng quan

Visual-TransGNN là một mô hình khuyến nghị hiện đại kết hợp giữa **Graph Neural Network (GNN)** và **Transformer**, có khả năng xử lý dữ liệu thị giác và tương tác người dùng – sản phẩm trong hệ thống gợi ý thời trang.

Mô hình được thiết kế để:
- Khai thác thông tin cấu trúc từ đồ thị tương tác người dùng - sản phẩm.
- Kết hợp embedding hình ảnh (trích xuất từ CNN).
- Áp dụng Attention Sampling để giảm nhiễu và tăng hiệu quả lan truyền thông tin.
- Bổ sung Positional Encoding để tận dụng thông tin cấu trúc đồ thị.

---

## Tập dữ liệu sử dụng

Mô hình đã được huấn luyện và đánh giá trên **hai tập dữ liệu**:

1. **Vibrent Clothes Rental**: Dữ liệu người dùng thuê trang phục, chứa lịch sử tương tác và thông tin sản phẩm.
2. **H&M Fashion Dataset**: Bộ dữ liệu khuyến nghị thời trang từ H&M, bao gồm lịch sử mua hàng, mô tả sản phẩm và hình ảnh.

> **Lưu ý**: Vui lòng tải và giải nén các tệp dữ liệu trước khi thực hiện huấn luyện. Dữ liệu được đặt trong thư mục `Data/` với tên `vibrent/` và `HM/`.

---

## Chuẩn bị

Ngoài ra, cần tạo sẵn **các thư mục sau ở cấp thư mục cha (cùng cấp với thư mục chứa file `Main.py`)** để đảm bảo mô hình lưu được checkpoint và log huấn luyện:

- `Models/`: dùng để lưu lại checkpoint của mô hình trong quá trình huấn luyện.
- `History/`: dùng để ghi lại lịch sử huấn luyện và kết quả đánh giá mô hình.

### Thiết lập tham số mô hình

Toàn bộ tham số huấn luyện được định nghĩa trong tệp `Params.py`, bao gồm các thông số nổi bật như:

- `--epoch`: số epoch huấn luyện
- `--batch`: batch size
- `--lr`: learning rate
- `--gnn_layer`: số tầng GNN
- `--att_head`: số lượng đầu attention
- `--dropout`: tỷ lệ dropout trong Transformer
- `--data`: tên tập dữ liệu (`vibrent` hoặc `hm`)
- `--gpu`: lựa chọn GPU để huấn luyện (ví dụ: `'0'` hoặc `'1'`)

Các giá trị mặc định đã được tinh chỉnh và ổn định qua thực nghiệm, do đó người dùng có thể sử dụng trực tiếp mà không cần thay đổi nếu không cần tùy chỉnh thêm.

### Huấn luyện mô hình

Sau khi chuẩn bị dữ liệu và cấu trúc thư mục, có thể bắt đầu huấn luyện và đánh giá mô hình bằng các lệnh sau:

# Với tập dữ liệu Vibrent Clothes Rental
python Main.py --data vibrent

# Với tập dữ liệu H&M Fashion Dataset
python Main.py --data HM

# Có thể ghi đè các tham số từ dòng lệnh, ví dụ:
python Main.py --data vibrent --lr 0.0005 --epoch 100 --att_head 4 


