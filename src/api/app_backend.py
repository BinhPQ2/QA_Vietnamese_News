from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
from typing import List
from src.api.controllers.controller import pipeline

app = FastAPI()

class UserPromptRequest(BaseModel):
    question: str

class AssistantResponse(BaseModel):
    answer: str
    url: List[str]


@app.post("/qa-vn-news", response_model=AssistantResponse)
def get_response(data: UserPromptRequest):
    try:
        print(f"Received question: {data.question}")
        answer,url = pipeline(data.question)
        print(f"Response answer: {answer}")
        print(f"Response url: {answer}")
        #answer = "Giá căn hộ mới tại Hà Nội tăng 6% trong quý III lên trung bình 69 triệu đồng một m2, trong khi TP HCM đạt 68 triệu đồng, theo Savills. Trong báo cáo thị trường quý III, đơn vị tư vấn dịch vụ bất động sản Savills cho biết giá chung cư tại Hà Nội tiếp tục leo thang ở cả dự án mới và cũ. Theo đó, mặt bằng giá dự án mới đã tăng 6% theo quý và 28% theo năm, đạt 69 triệu đồng một m2. Dự án sơ cấp tăng cao cũng kéo giá bán chung cư cũ leo thang, với mức 41% theo năm, lên 51 triệu đồng một m2. Số căn hộ bán được trong quý này tại Hà Nội đạt hơn 6.800 căn, tăng 35% theo quý và 226% theo năm. Căn hộ trung cấp đến cao cấp dẫn đầu, đóng góp 98% lượng giao dịch. Tính chung 9 tháng, 70% số căn được giao dịch có giá trên 4 tỷ đồng, tăng mạnh từ mức 2% năm 2020. Còn phân khúc từ 2 đến 4 tỷ đồng chiếm 29%."
        #url= ["https://vnexpress.net/gia-can-ho-moi-tai-ha-noi-tang-6-trong-quy-iii-4385797.html"]
        #print(f"Response answer: {answer}")
        #print(f"Response url: {url}")
        return AssistantResponse(
            answer=answer,
            url=url
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    