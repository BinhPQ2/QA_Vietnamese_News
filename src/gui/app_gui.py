import time
import streamlit as st
import requests
from fastapi import HTTPException

st.set_page_config(page_title="Q&A Vietnamese News", page_icon="📚")
st.title("Q&A Vietnamese News")

with st.sidebar:
    st.button("Nâng cấp gói Plus", icon="🗝️", use_container_width=True)

st.sidebar.markdown("---")

with st.sidebar:
    st.title("Nhóm: SoDeep")
    with st.expander("Thông tin nhóm", expanded=True):
        st.write("23C11018 - Phạm Quốc Bình")
        st.write("23C11054 - Nguyễn Khắc Toàn")
        st.write("23C15027 - Trần Tuyết Huê")
        st.write("23C15030 - Nguyễn Vũ Linh")
        st.write("23C15037 - Bùi Trọng Quý")

st.sidebar.markdown("---")
with st.sidebar:
    st.title("Feedback")
    sentiment_mapping = ["one", "two", "three", "four", "five"]
    selected = st.feedback("stars")
    if selected is not None:
        st.info(f"Bạn vừa đánh giá **{selected + 1}** sao. Cảm ơn bạn đã phản hồi!")


if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Tôi có thể giúp được gì cho bạn?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Nhập nội dung bất kỳ..."):
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message('user'):
        st.markdown(prompt)

    try:
        response = requests.post(
            "http://localhost:8000/qa-vn-news",  
            json={"question": prompt}
        )
        response.raise_for_status() 
        data = response.json()

        answer = data.get("answer", "Không tìm thấy câu trả lời.")
        urls = data.get("url", [])

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": prompt
            }
        )

        with st.chat_message('assistant'):
            full_res = ""
            holder = st.empty()

        for word in answer.split():
            full_res += word + " "
            time.sleep(0.05)
            holder.markdown(full_res + "▌")

        if len(urls) > 0:
            full_res = full_res + (f"\n\n :bookmark: Xem chi tiết tại: {urls[0]}")
        
        holder.markdown(full_res)

    except HTTPException as e:
        st.error(f"Error: {e.detail}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
