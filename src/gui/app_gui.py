# import time
# import streamlit as st
# # st.markdown(link,unsafe_allow_html=True)
# st.set_page_config(page_title="Q&A Vietnamese News", page_icon="📚")
# st.title("Q&A Vietnamese News")

# st.sidebar.title("Nhóm: SoDeep")
# st.sidebar.write("23C11018 - Phạm Quốc Bình")
# st.sidebar.write("23C11054 - Nguyễn Khắc Toàn")
# st.sidebar.write("23C15027 - Trần Tuyết Huê")
# st.sidebar.write("23C15030 - Nguyễn Vũ Linh")
# st.sidebar.write("23C15037 - Bùi Trọng Quý")

# # with st.expander("Expand for more details"):
# #     st.write("This content is hidden until expanded.")
# #     st.write("You can place various widgets inside an expander.")
# #     checkbox = st.checkbox("Check me!")
# #     if checkbox:
# #         st.write("Checkbox is checked.")

# method = st.sidebar.selectbox(
#     "Chọn trang báo",
#     ("VNExpress", "DanTri")
# )

# if "messages" not in  st.session_state:
#     st.session_state.messages = []

# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])


# if prompt := st.chat_input("Tôi có thể giúp gì được cho bạn?"):
#     st.session_state.messages.append(
#         {
#             "role": "user",
#             "content": prompt
#         }
#     )

#     with st.chat_message('user'):
#         st.markdown(prompt)

#     with st.chat_message('assistant'):
#         full_res = ""
#         holder = st.empty()

#         for word in prompt.split():
#             full_res += word + " "
#             time.sleep(0.05)
#             holder.markdown(full_res + "▌")
#         holder.markdown(full_res)


#     st.session_state.messages.append(
#         {
#             "role": "assistant",
#             "content": full_res
#         }
#     )


import time
import streamlit as st
import requests  # Import requests to make API calls
from fastapi import HTTPException

st.set_page_config(page_title="Q&A Vietnamese News", page_icon="📚")
st.title("Q&A Vietnamese News")

st.sidebar.title("Nhóm: SoDeep")
st.sidebar.write("23C11018 - Phạm Quốc Bình")
st.sidebar.write("23C11054 - Nguyễn Khắc Toàn")
st.sidebar.write("23C15027 - Trần Tuyết Huê")
st.sidebar.write("23C15030 - Nguyễn Vũ Linh")
st.sidebar.write("23C15037 - Bùi Trọng Quý")

method = st.sidebar.selectbox(
    "Chọn trang báo",
    ("VNExpress", "DanTri")
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Tôi có thể giúp gì được cho bạn?"):
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message('user'):
        st.markdown(prompt)

    # with st.chat_message('assistant'):
    #     full_res = ""
    #     holder = st.empty()

    #     # Simulate typing effect
    #     for word in prompt.split():
    #         full_res += word + " "
    #         time.sleep(0.05)
    #         holder.markdown(full_res + "▌")
    #     holder.markdown(full_res)

    # Call the FastAPI backend
    try:
        response = requests.post(
            "http://localhost:8000/qa-vn-news",  # Replace with your FastAPI URL
            json={"question": prompt}
        )
        response.raise_for_status()  # Raise an error for bad responses
        data = response.json()  # Parse JSON response

        # Extracting answer and URLs
        answer = data.get("answer", "No answer found.")
        urls = data.get("url", [])

        # Add the response to session messages
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        with st.chat_message('assistant'):
            full_res = ""
            holder = st.empty()

        # Simulate typing effect
        for word in answer.split():
            full_res += word + " "
            time.sleep(0.05)
            holder.markdown(full_res + "▌")

        holder.markdown(full_res)

        # Display the assistant's response and URLs
        # st.markdown(answer)
        # if urls:
        #     st.write("Related URLs:")
        #     for url in urls:
        #         st.markdown(f"- [{url}]({url})")

    except HTTPException as e:
        st.error(f"Error: {e.detail}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
