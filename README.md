# Ứng dụng Chatbot hỏi đáp với website tin tức

## Giới thiệu
Chào mừng bạn đến với dự án Chatbot hỏi đáp với bất kỳ website! Dự án này cung cấp một giải pháp linh hoạt cho việc trích xuất thông tin từ các trang web và cung cấp câu trả lời dưới dạng chatbot. Sử dụng sức mạnh của LangChain và mô hình ngôn ngữ lớn, chatbot này có khả năng tương tác và xử lý thông tin từ nhiều nguồn trực tuyến, hỗ trợ người dùng trong việc tìm kiếm và giải đáp thắc mắc một cách hiệu quả.

## Tính năng
- **Trích xuất thông tin**: Crawl thông tin từ các trang web lớn như VnExpress (và DanTri nếu có thời gian).
- **Tích hợp mô hình ngôn ngữ lớn**: Sử dụng các model Ngôn ngữ lớn để hiểu và trả lời các câu hỏi của người dùng dựa trên thông tin được trích xuất, đảm bảo câu trả lời chính xác và tự nhiên.
- **Giao diện thân thiện**: Giao diện người dùng được xây dựng bằng Streamlit, cung cấp trải nghiệm trực quan, dễ sử dụng, phù hợp cho người dùng ở mọi cấp độ kỹ thuật.
- **Ngôn ngữ Python**: Dự án được phát triển hoàn toàn bằng Python, dễ dàng để tùy chỉnh và mở rộng.

## Yêu cầu hệ thống
- Python 3.7 trở lên
- GPU có hỗ trợ CUDA

## Hướng dẫn cài đặt

### 1. Clone repository
Clone repository này về máy tính của mình:

```bash
git clone [repository-link]
cd [repository-directory]
```

### 2. Cài đặt các thư viện cần thiết

```bash
pip install -r requirements.txt
```
## # 3. Cách sử dụng

Sử dụng file `base_pipeline.ipynb` để chạy