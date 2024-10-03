# Ứng dụng chatbot hỏi đáp với bất kỳ website

Source: `https://github.com/alejandro-ao/chat-with-websites`

## Giới thiệu
Chào mừng bạn đến với dự án Chatbot hỏi đáp với bất kỳ website! Dự án này cung cấp một giải pháp linh hoạt cho việc trích xuất thông tin từ các trang web và cung cấp câu trả lời dưới dạng chatbot. Sử dụng sức mạnh của LangChain và mô hình ngôn ngữ lớn GPT-4, chatbot này có khả năng tương tác và xử lý thông tin từ nhiều nguồn trực tuyến, hỗ trợ người dùng trong việc tìm kiếm và giải đáp thắc mắc một cách hiệu quả.

## Tính năng
- **Trích xuất thông tin**: Sử dụng phiên bản mới nhất của LangChain để tương tác và trích xuất thông tin từ các trang web khác nhau, cho phép bạn nhanh chóng tìm thấy câu trả lời cần thiết.
- **Tích hợp mô hình ngôn ngữ lớn**: Sử dụng GPT-4 để hiểu và trả lời các câu hỏi của người dùng dựa trên thông tin được trích xuất, đảm bảo câu trả lời chính xác và tự nhiên.
- **Giao diện thân thiện**: Giao diện người dùng được xây dựng bằng Streamlit, cung cấp trải nghiệm trực quan, dễ sử dụng, phù hợp cho người dùng ở mọi cấp độ kỹ thuật.
- **Ngôn ngữ Python**: Dự án được phát triển hoàn toàn bằng Python, dễ dàng để tùy chỉnh và mở rộng.

## Yêu cầu hệ thống
Trước khi bắt đầu, hãy đảm bảo rằng hệ thống của bạn đã cài đặt:
- Python 3.7 trở lên
- Các thư viện Python cần thiết được liệt kê trong file `requirements.txt`

## Hướng dẫn cài đặt

### 1. Clone repository
Đầu tiên, bạn cần clone repository này về máy tính của mình:

```bash
git clone [repository-link]
cd [repository-directory]
```

### 2. Cài đặt các thư viện cần thiết
Sử dụng pip để cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

### 3. Cấu hình API Key
Tạo một file `.env` trong thư mục gốc của dự án và thêm khóa API OpenAI của bạn:

```bash
OPENAI_API_KEY=[your-openai-api-key]
```

**Lưu ý**: Bạn cần có tài khoản OpenAI và API key để có thể sử dụng GPT-4.

## Cách sử dụng

### Chạy ứng dụng Streamlit
Sau khi hoàn thành các bước cài đặt, bạn có thể chạy ứng dụng Streamlit bằng lệnh:

```bash
streamlit run src/app.py
```

Ứng dụng sẽ mở ra trong trình duyệt của bạn, và bạn có thể bắt đầu sử dụng chatbot để hỏi đáp thông tin từ bất kỳ trang web nào.


## Giấy phép
Dự án này được cấp phép theo giấy phép MIT. Vui lòng tham khảo file `LICENSE` để biết thêm chi tiết.

Cảm ơn bạn đã quan tâm đến dự án của chúng tôi! Chúng tôi hy vọng bạn sẽ tìm thấy nó hữu ích và thú vị.
```
