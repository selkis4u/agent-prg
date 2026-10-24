# Step2. 임포트
import os
import pandas as pd
import gradio as gr
from datetime import datetime

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        ".env 파일에 OPENAI_API_KEY가 설정되어 있지 않습니다."
    )



# Step1. 상품 데이터
products_data = [
    {"상품ID": "P001", "상품명": "프리미엄 강아지 간식 세트", "카테고리": "간식", "재고수량": 36, "가격": 32417, "제조사": "펫스토리", "태그": "간식,강아지,프리미엄", "판매량": 890},
    {"상품ID": "P002", "상품명": "고양이 캣타워 XL", "카테고리": "가구", "재고수량": 91, "가격": 47747, "제조사": "캣플레이", "태그": "고양이,캣타워,대형", "판매량": 245},
    {"상품ID": "P003", "상품명": "자동 급식기", "카테고리": "전자제품", "재고수량": 10, "가격": 89900, "제조사": "스마트펫", "태그": "자동급식기,스마트", "판매량": 567},
    {"상품ID": "P004", "상품명": "이동가방 대형", "카테고리": "외출용품", "재고수량": 99, "가격": 91542, "제조사": "펫캐리", "태그": "이동가방,여행,대형견", "판매량": 120},
    {"상품ID": "P005", "상품명": "펫 샴푸", "카테고리": "목욕용품", "재고수량": 49, "가격": 41276, "제조사": "펫케어", "태그": "샴푸,목욕,미용", "판매량": 678},
    {"상품ID": "P006", "상품명": "LED 목줄", "카테고리": "외출용품", "재고수량": 75, "가격": 18900, "제조사": "세이프펫", "태그": "목줄,LED,야간산책", "판매량": 445},
    {"상품ID": "P007", "상품명": "칫솔 치약 세트", "카테고리": "구강관리", "재고수량": 120, "가격": 15500, "제조사": "펫덴탈", "태그": "칫솔,치약,구강관리", "판매량": 234},
    {"상품ID": "P008", "상품명": "스크래처", "카테고리": "가구", "재고수량": 45, "가격": 35900, "제조사": "캣홈", "태그": "스크래처,고양이,발톱", "판매량": 189}
    ]


# Step2. 고객 데이터
customers_data = [
    {"고객ID": "C001", "고객명": "박예준", "전화번호": "010-1234-5678", "이메일": "park@example.com", "가입일자": "2020-08-10", "등급": "VIP", "포인트": 5010, "총구매액": 892540, "구매횟수": 12},
    {"고객ID": "C002", "고객명": "김미숙", "전화번호": "010-2345-6789", "이메일": "kim@example.com", "가입일자": "2021-01-23", "등급": "VIP", "포인트": 5771, "총구매액": 456780, "구매횟수": 5},
    {"고객ID": "C003", "고객명": "이준호", "전화번호": "010-3456-7890", "이메일": "lee@example.com", "가입일자": "2022-07-10", "등급": "일반", "포인트": 2119, "총구매액": 235680, "구매횟수": 3},
]


# Step3. 주문 데이터
orders_data = [
    {"주문ID": "O0001", "고객ID": "C001", "상품ID": "P001", "수량": 2, "주문일자": "2025-03-25", "결제상태": "결제완료", "배송ID": "D0001", "최종결제액": 53346},
    {"주문ID": "O0002", "고객ID": "C001", "상품ID": "P006", "수량": 1, "주문일자": "2025-03-20", "결제상태": "결제완료", "배송ID": "D0002", "최종결제액": 18900},
    {"주문ID": "O0003", "고객ID": "C001", "상품ID": "P003", "수량": 1, "주문일자": "2025-02-28", "결제상태": "결제완료", "배송ID": "D0003", "최종결제액": 89900},
    {"주문ID": "O0004", "고객ID": "C001", "상품ID": "P002", "수량": 1, "주문일자": "2025-01-15", "결제상태": "취소", "배송ID": "D0004", "최종결제액": 47747},
    {"주문ID": "O0005", "고객ID": "C002", "상품ID": "P005", "수량": 2, "주문일자": "2025-03-18", "결제상태": "결제완료", "배송ID": "D0005", "최종결제액": 82552},
    {"주문ID": "O0006", "고객ID": "C003", "상품ID": "P007", "수량": 3, "주문일자": "2025-03-10", "결제상태": "결제완료", "배송ID": "D0006", "최종결제액": 46500},
]


# Step4. 배송 데이터
deliveries_data = [
    {"배송ID": "D0001", "배송상태": "배송중", "출고일자": "2025-03-26", "도착예정일": "2025-03-28", "배송사": "CJ대한통운", "송장번호": "123456789"},
    {"배송ID": "D0002", "배송상태": "배송완료", "출고일자": "2025-03-21", "도착예정일": "2025-03-23", "배송사": "한진택배", "송장번호": "987654321"},
    {"배송ID": "D0003", "배송상태": "배송완료", "출고일자": "2025-03-01", "도착예정일": "2025-03-03", "배송사": "우체국택배", "송장번호": "555666777"},
    {"배송ID": "D0004", "배송상태": "취소", "출고일자": "-", "도착예정일": "-", "배송사": "-", "송장번호": "-"},
    {"배송ID": "D0005", "배송상태": "상품준비중", "출고일자": "-", "도착예정일": "2025-03-30", "배송사": "롯데택배", "송장번호": "-"},
    {"배송ID": "D0006", "배송상태": "배송중", "출고일자": "2025-03-11", "도착예정일": "2025-03-13", "배송사": "CJ대한통운", "송장번호": "111222333"},
]



# Step5. 장바구니 데이터
cart_data = [
    {"장바구니ID": "CART001", "고객ID": "C001", "상품ID": "P008", "수량": 1, "추가일자": "2025-03-26"},
    {"장바구니ID": "CART002", "고객ID": "C001", "상품ID": "P007", "수량": 2, "추가일자": "2025-03-25"},
    {"장바구니ID": "CART003", "고객ID": "C002", "상품ID": "P004", "수량": 1, "추가일자": "2025-03-24"},
]



# Step6. 리뷰 데이터
reviews_data = [
    {"리뷰ID": "R001", "고객ID": "C001", "상품ID": "P001", "평점": 5, "리뷰내용": "강아지가 정말 좋아해요. 재구매 의사 있습니다!", "작성일자": "2025-03-27"},
    {"리뷰ID": "R002", "고객ID": "C001", "상품ID": "P006", "평점": 4, "리뷰내용": "LED가 밝아서 밤 산책때 안전해요. 배터리도 오래갑니다.", "작성일자": "2025-03-25"},
    {"리뷰ID": "R003", "고객ID": "C001", "상품ID": "P003", "평점": 5, "리뷰내용": "자동급식기 정말 편리해요. 외출할 때 걱정이 덜어요.", "작성일자": "2025-03-10"},
    {"리뷰ID": "R004", "고객ID": "C002", "상품ID": "P005", "평점": 3, "리뷰내용": "향은 좋은데 거품이 잘 안나요", "작성일자": "2025-03-20"},
    {"리뷰ID": "R005", "고객ID": "C003", "상품ID": "P007", "평점": 5, "리뷰내용": "치약 향도 좋고 강아지가 양치질 싫어하지 않아요", "작성일자": "2025-03-15"},
]


# Step7. 포인트 히스토리 데이터
point_history_data = [
    {"포인트ID": "PT001", "고객ID": "C001", "구분": "적립", "포인트": 533, "내용": "O0001 구매 적립", "일자": "2025-03-25"},
    {"포인트ID": "PT002", "고객ID": "C001", "구분": "사용", "포인트": -2000, "내용": "O0003 구매시 사용", "일자": "2025-02-28"},
    {"포인트ID": "PT003", "고객ID": "C001", "구분": "적립", "포인트": 1000, "내용": "리뷰 작성 보너스", "일자": "2025-03-27"},
    {"포인트ID": "PT004", "고객ID": "C002", "구분": "적립", "포인트": 825, "내용": "O0005 구매 적립", "일자": "2025-03-18"},
]


# Step8. 프로모션 데이터
promotions_data = [
    {"프로모션ID": "E001", "제목": "봄맞이 펫용품 대전", "내용": "전 상품 10% 할인 + 5만 원 이상 구매시 사료샘플 증정", "시작일": "2025-03-01", "종료일": "2025-03-31", "대상상품": "전체"},
    {"프로모션ID": "E002", "제목": "신규회원 웰컴 혜택", "내용": "첫 구매시 15% 할인쿠폰 + 적립금 2배", "시작일": "2025-01-01", "종료일": "2025-12-31", "대상상품": "전체"},
    {"프로모션ID": "E003", "제목": "고양이 용품 특가전", "내용": "고양이 관련 상품 20% 할인", "시작일": "2025-03-15", "종료일": "2025-04-15", "대상상품": "고양이"},
]


# Step9. 데이터프레임 생성
products_df = pd.DataFrame(products_data)
customers_df = pd.DataFrame(customers_data)
orders_df = pd.DataFrame(orders_data)
deliveries_df = pd.DataFrame(deliveries_data)
cart_df = pd.DataFrame(cart_data)
reviews_df = pd.DataFrame(reviews_data)
point_history_df = pd.DataFrame(point_history_data)
promotions_df = pd.DataFrame(promotions_data)


# Step1. 고객 프로필 조회 함수
@tool
def get_customer_profile(customer_id: str) -> str:
    """고객 ID로 고객 프로필 정보를 조회합니다."""
    customer = customers_df[customers_df['고객ID'] == customer_id]
    if customer.empty:
        return f"고객 ID {customer_id}를 찾을 수 없습니다."

    c = customer.iloc[0]
    result = []
    result.append(f"고객ID: {c['고객ID']}")
    result.append(f"고객명: {c['고객명']}")
    result.append(f"전화번호: {c['전화번호']}")
    result.append(f"이메일: {c['이메일']}")
    result.append(f"가입일자: {c['가입일자']}")
    result.append(f"고객등급: {c['등급']}")
    result.append(f"보유포인트: {c['포인트']:,}점")
    result.append(f"총구매액: {c['총구매액']:,}원")
    result.append(f"구매횟수: {c['구매횟수']}회")

    return "\n".join(result)


# Step2. 상품 검색 함수
@tool
def search_products(keyword: str = None, category: str = None,
                   price_min: int = None, price_max: int = None) -> str:
    """키워드로 상품을 검색합니다."""
    df = products_df.copy()

    if keyword:
        mask = (
            df['상품명'].str.contains(keyword, case=False, na=False) |
            df['태그'].str.contains(keyword, case=False, na=False) |
            df['제조사'].str.contains(keyword, case=False, na=False)
        )
        df = df[mask]

    if category:
        df = df[df['카테고리'].str.contains(category, case=False, na=False)]

    if price_min:
        df = df[df['가격'] >= price_min]

    if price_max:
        df = df[df['가격'] <= price_max]

    if df.empty:
        return "조건에 맞는 상품이 없습니다."

    result = [f"검색 결과: {len(df)}개 상품\n"]

    for _, p in df.iterrows():
        result.append(f"상품ID: {p['상품ID']}")
        result.append(f"상품명: {p['상품명']}")
        result.append(f"가격: {p['가격']:,}원")
        result.append(f"재고: {p['재고수량']}개")
        result.append(f"카테고리: {p['카테고리']}")
        result.append("-------------------")

    return "\n".join(result)


# Step3. 고객 주문 내역 조회 함수
@tool
def get_customer_orders(customer_id: str, start_date: str = None, end_date: str = None) -> str:
    """고객의 주문 내역을 조회합니다."""
    orders = orders_df[orders_df['고객ID'] == customer_id].copy()

    if orders.empty:
        return f"고객 ID {customer_id}의 주문 내역이 없습니다."

    if start_date:
        orders = orders[orders['주문일자'] >= start_date]
    if end_date:
        orders = orders[orders['주문일자'] <= end_date]

    if orders.empty:
        return "해당 기간의 주문 내역이 없습니다."

    result = [f"주문 내역 ({len(orders)}건):\n"]

    for _, order in orders.iterrows():
        product = products_df[products_df['상품ID'] == order['상품ID']]
        product_name = product.iloc[0]['상품명'] if not product.empty else "상품정보없음"

        result.append(f"주문ID: {order['주문ID']}")
        result.append(f"주문일자: {order['주문일자']}")
        result.append(f"상품: {product_name}")
        result.append(f"수량: {order['수량']}개")
        result.append(f"결제금액: {order['최종결제액']:,}원")
        result.append(f"결제상태: {order['결제상태']}")
        result.append("-------------------")

    return "\n".join(result)


# Step4. 배송 상태 확인 함수
@tool
def get_delivery_status(customer_id: str = None, order_id: str = None) -> str:
    """배송 상태를 확인합니다."""
    if order_id:
        order = orders_df[orders_df['주문ID'] == order_id]
        if order.empty:
            return f"주문번호 {order_id}를 찾을 수 없습니다."

        delivery_id = order.iloc[0]['배송ID']
        delivery = deliveries_df[deliveries_df['배송ID'] == delivery_id]

        if delivery.empty:
            return "배송 정보를 찾을 수 없습니다."

        d = delivery.iloc[0]
        product = products_df[products_df['상품ID'] == order.iloc[0]['상품ID']]
        product_name = product.iloc[0]['상품명'] if not product.empty else "상품정보없음"

        result = []
        result.append(f"주문번호: {order_id}")
        result.append(f"상품: {product_name}")
        result.append(f"배송상태: {d['배송상태']}")
        if d['배송상태'] != '취소':
            result.append(f"배송사: {d['배송사']}")
            result.append(f"송장번호: {d['송장번호']}")
            result.append(f"출고일자: {d['출고일자']}")
            result.append(f"도착예정일: {d['도착예정일']}")

        return "\n".join(result)

    elif customer_id:
        orders = orders_df[orders_df['고객ID'] == customer_id]
        if orders.empty:
            return f"고객 ID {customer_id}의 주문이 없습니다."

        result = ["배송 현황:\n"]

        for _, order in orders.iterrows():
            if order['결제상태'] == '결제완료':
                delivery = deliveries_df[deliveries_df['배송ID'] == order['배송ID']]
                if not delivery.empty:
                    d = delivery.iloc[0]
                    product = products_df[products_df['상품ID'] == order['상품ID']]
                    product_name = product.iloc[0]['상품명'] if not product.empty else "상품정보없음"

                    result.append(f"주문번호: {order['주문ID']}")
                    result.append(f"상품: {product_name}")
                    result.append(f"배송상태: {d['배송상태']}")
                    result.append("-------------------")

        return "\n".join(result)

    return "고객ID나 주문ID를 입력해주세요."    


# Step5. 리뷰 검색 함수
@tool
def search_reviews(keyword: str = None, product_name: str = None,
                  customer_id: str = None, rating: int = None) -> str:
    """리뷰를 검색합니다."""
    df = reviews_df.copy()

    if customer_id:
        df = df[df['고객ID'] == customer_id]

    if product_name:
        matching_products = products_df[
            products_df['상품명'].str.contains(product_name, case=False, na=False)
        ]['상품ID'].tolist()
        if matching_products:
            df = df[df['상품ID'].isin(matching_products)]

    if keyword:
        df = df[df['리뷰내용'].str.contains(keyword, case=False, na=False)]

    if rating:
        df = df[df['평점'] == rating]

    if df.empty:
        return "조건에 맞는 리뷰가 없습니다."

    result = [f"리뷰 검색 결과 ({len(df)}개):\n"]

    for _, review in df.iterrows():
        product = products_df[products_df['상품ID'] == review['상품ID']]
        product_name = product.iloc[0]['상품명'] if not product.empty else "상품정보없음"

        customer = customers_df[customers_df['고객ID'] == review['고객ID']]
        customer_name = customer.iloc[0]['고객명'] if not customer.empty else "고객정보없음"

        result.append(f"상품: {product_name}")
        result.append(f"평점: {'★' * review['평점']}")
        result.append(f"리뷰: {review['리뷰내용']}")
        result.append(f"작성자: {customer_name}")
        result.append(f"작성일: {review['작성일자']}")
        result.append("-------------------")

    return "\n".join(result)


# Step6. 장바구니 조회 함수
@tool
def get_customer_cart(customer_id: str) -> str:
    """고객의 장바구니를 조회합니다."""
    cart_items = cart_df[cart_df['고객ID'] == customer_id]

    if cart_items.empty:
        return f"고객 ID {customer_id}의 장바구니가 비어 있습니다."

    result = [f"장바구니 ({len(cart_items)}개 상품):\n"]
    total = 0

    for _, item in cart_items.iterrows():
        product = products_df[products_df['상품ID'] == item['상품ID']]
        if not product.empty:
            p = product.iloc[0]
            subtotal = p['가격'] * item['수량']
            total += subtotal

            result.append(f"상품: {p['상품명']}")
            result.append(f"수량: {item['수량']}개")
            result.append(f"단가: {p['가격']:,}원")
            result.append(f"소계: {subtotal:,}원")
            result.append("-------------------")

    result.append(f"\n총액: {total:,}원")
    result.append(f"배송비: {0 if total >= 50000 else 3000:,}원")
    result.append(f"결제예상액: {total + (0 if total >= 50000 else 3000):,}원")

    return "\n".join(result)


# Step7. 포인트 이력 조회 함수
@tool
def get_point_history(customer_id: str) -> str:
    """포인트 내역을 조회합니다."""
    customer = customers_df[customers_df['고객ID'] == customer_id]
    if customer.empty:
        return f"고객 ID {customer_id}를 찾을 수 없습니다."

    current_points = customer.iloc[0]['포인트']
    history = point_history_df[point_history_df['고객ID'] == customer_id]

    result = [f"현재 포인트: {current_points:,}점\n"]

    if not history.empty:
        result.append("포인트 내역:")
        for _, record in history.iterrows():
            result.append(f"{record['일자']} {record['구분']} {record['포인트']:,}점")
            result.append(f"내용: {record['내용']}")
            result.append("-------------------")

    return "\n".join(result)


# Step8. 현재 프로모션 조회 함수
@tool
def get_current_promotions() -> str:
    """현재 진행중인 프로모션을 조회합니다."""
    today = datetime.now().strftime("%Y-%m-%d")
    active = promotions_df[
        (promotions_df['시작일'] <= today) &
        (promotions_df['종료일'] >= today)
    ]

    if active.empty:
        return "현재 진행중인 프로모션이 없습니다."

    result = ["현재 진행중인 프로모션:\n"]

    for _, promo in active.iterrows():
        result.append(f"[{promo['제목']}]")
        result.append(f"내용: {promo['내용']}")
        result.append(f"기간: {promo['시작일']} ~ {promo['종료일']}")
        result.append("-------------------")

    return "\n".join(result)



# Step9. 인기 상품 조회 함수
@tool
def get_popular_products(category: str = None, period: str = "month") -> str:
    """인기상품을 조회합니다."""
    df = products_df.copy()

    if category:
        df = df[df['카테고리'].str.contains(category, case=False, na=False)]

    # 판매량 기준 정렬
    df = df.nlargest(5, '판매량')

    if df.empty:
        return "조건에 맞는 상품이 없습니다."

    result = ["인기상품 TOP 5:\n"]

    for i, (_, product) in enumerate(df.iterrows(), 1):
        result.append(f"{i}. {product['상품명']}")
        result.append(f"   가격: {product['가격']:,}원")
        result.append(f"   판매량: {product['판매량']}개")
        result.append("-------------------")

    return "\n".join(result)



# Step1. 도구 리스트 생성
tools = [
    get_customer_profile,
    search_products,
    get_customer_orders,
    get_delivery_status,
    search_reviews,
    get_customer_cart,
    get_point_history,
    get_current_promotions,
    get_popular_products,
]



# Step2. LLM 설정
llm = ChatOpenAI(model="gpt-4o", temperature=0)



# Step3. 시스템 프롬프트
SYSTEM_PROMPT = """당신은 쇼핑몰 고객 지원 도우미입니다. 사용자의 질문에 최선을 다해 답변하세요.

1. 고객 프로필, 주문 내역, 배송 상태, 포인트, 결제 정보 등에 관한 질문은 반드시 도구를 호출하여 답변합니다.
2. 이전 대화 맥락을 참고하여 일관된 답변을 제공합니다.
3. 고객의 문의에 친절하고 전문적으로 응대합니다.
4. ID 형식을 정확히 사용하십시오.
5. 현재 로그인한 사용자의 ID는 C001로 가정합니다.
"""


# Step4. 에이전트 생성
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT
)


# Step1. 대화 관리 클래스
class ConversationManager:
    def __init__(self):
        self.agent = agent
        self.chat_history = []

    def process_message(self, message: str):
        # 사용자 메시지를 히스토리에 추가
        self.chat_history.append({"role": "user", "content": message})

        # 에이전트 실행
        response = self.agent.invoke({"messages": self.chat_history})

        # 응답에서 마지막 메시지 추출
        last_message = response["messages"][-1]
        answer = last_message.content if hasattr(last_message, 'content') else str(last_message)

        # 어시스턴트 응답을 히스토리에 추가
        self.chat_history.append({"role": "assistant", "content": answer})

        # 실행 로그 생성
        log_content = [f"[User Input] {message}"]

        # tool_calls 정보 추출
        for msg in response["messages"]:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    log_content.append(f"[Function Call] {tool_call['name']}")
                    log_content.append(f"[Parameters] {tool_call['args']}")
            if hasattr(msg, 'name') and msg.name:  # ToolMessage
                log_content.append(f"[Output] {msg.content}")

        log_content.append(f"[AI Response] {answer}")
        execution_log = "\n".join(log_content)

        return answer, execution_log

    def clear_history(self):
        self.chat_history = []
        return []


# 대화 관리자 인스턴스 생성
conversation_manager = ConversationManager()


# Step5. 멀티턴 대화 테스트
# 예제 1: 고객 프로필 조회
print("=== 예제 1: 고객 프로필 조회 ===")
query1 = "고객 ID C001의 프로필 정보를 알려주세요"
result1, log1 = conversation_manager.process_message(query1)
print("\n최종 답변:")
print(result1)
print("\n실행 로그:")
print(log1)

# 예제 2: 주문 내역 조회 (멀티턴 테스트)
print("\n=== 예제 2: 주문 내역 조회 (멀티턴) ===")
query2 = "그 고객의 주문 내역을 조회해주세요"
result2, log2 = conversation_manager.process_message(query2)
print("\n최종 답변:")
print(result2)
print("\n실행 로그:")
print(log2)

# 예제 3: 배송 상태 조회 (멀티턴 테스트)
print("\n=== 예제 3: 배송 상태 조회 (멀티턴) ===")
query3 = "배송 상태도 확인해주세요"
result3, log3 = conversation_manager.process_message(query3)
print("\n최종 답변:")
print(result3)
print("\n실행 로그:")
print(log3)