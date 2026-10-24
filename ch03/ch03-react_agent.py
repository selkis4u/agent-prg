import os
import re
import io
import sys
import json
import requests
from typing import Dict, List, Optional, Tuple
from contextlib import redirect_stdout

# LangChain 1.0 계열 임포트
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_chroma import Chroma
from langchain_core.tools import create_retriever_tool
from langchain_core.prompts import PromptTemplate

import gradio as gr

from dotenv import load_dotenv

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        ".env 파일에 OPENAI_API_KEY가 설정되어 있지 않습니다."
    )

urls = [
   "https://raw.githubusercontent.com/ai-agent-kr/agent-tutorial/main/Ch02/ict_japan_2024.pdf",
   "https://raw.githubusercontent.com/ai-agent-kr/agent-tutorial/main/Ch02/ict_usa_2024.pdf",
   "https://raw.githubusercontent.com/ai-agent-kr/agent-tutorial/main/Ch02/blockchain_usa_2025.pdf"
]

# 각 파일 다운로드
for url in urls:
    filename = url.split("/")[-1] # URL에서 파일명 추출
    if not os.path.exists(filename):
        response = requests.get(url)
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"{filename} 다운로드 완료")

embd = OpenAIEmbeddings(model="text-embedding-3-small")


def create_pdf_retriever(
    pdf_path: str,  # PDF 파일 경로
    persist_directory: str,  # 벡터 스토어 저장 경로
    embedding_model: OpenAIEmbeddings,  # OpenAIEmbeddings 임베딩 모델
    chunk_size: int = 512,  # 청크 크기 기본값 512
    chunk_overlap: int = 0  # 청크 오버랩 크기 기본값 0
):
        
    # 이미 로컬에 저장된 벡터 DB가 있다면 API 호출 없이 바로 로드
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model
        )
        return vectorstore.as_retriever()
    
    loader = PyMuPDFLoader(pdf_path)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    doc_splits = text_splitter.split_documents(data)

    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        embedding=embedding_model,
        persist_directory=persist_directory,
    )
    return vectorstore.as_retriever()

# 일본 ICT 정책 데이터베이스 생성
retriever_ict_japan = create_pdf_retriever(
    pdf_path="ict_japan_2024.pdf",
    persist_directory="db_ict_policy_japan_2024",
    embedding_model=embd
)

# 미국 ICT 정책 데이터베이스 생성
retriever_ict_usa = create_pdf_retriever(
    pdf_path="ict_usa_2024.pdf",
    persist_directory="db_ict_policy_usa_2024",
    embedding_model=embd
)

# 미국 블록체인 동향 데이터베이스 생성
retriever_blockchain_usa = create_pdf_retriever(
    pdf_path="blockchain_usa_2025.pdf",
    persist_directory="db_blockchain_usa_2025",
    embedding_model=embd
)


ict_japan_engine = create_retriever_tool(
    retriever=retriever_ict_japan,
    name="japan_ict_trend_searcher",
    description="일본의 ICT 산업의 시장동향 정보를 제공합니다."
)

ict_usa_engine = create_retriever_tool(
    retriever=retriever_ict_usa,
    name="usa_ict_trend_searcher",
    description="미국의 ICT 산업의 시장동향 정보를 제공합니다."
)

blockchain_usa_engine = create_retriever_tool(
    retriever=retriever_blockchain_usa,
    name="usa_blockchain_trend_searcher",
    description="미국의 블록체인 산업의 동향 정보를 제공합니다."
)


tools = [ict_japan_engine, ict_usa_engine, blockchain_usa_engine]


tool_map: Dict[str, object] = {t.name: t for t in tools}


react_template = '''다음 질문에 최선을 다해 답변하세요. 당신은 다음 도구들에 접근할 수 있습니다:

{tools}

다음 형식을 사용하세요:

Question: 답변해야 하는 입력 질문
Thought: 무엇을 할지 항상 생각하세요
Action: 취해야 할 행동, [{tool_names}] 중 하나여야 합니다. 리스트에 있는 도구 중 1개를 택하십시오.
Action Input: 행동에 대한 입력값
Observation: 행동의 결과
... (이 Thought/Action/Action Input/Observation의 과정이 N번 반복될 수 있습니다)
Thought: 이제 최종 답변을 알겠습니다
Final Answer: 원래 입력된 질문에 대한 최종 답변 (한글로 작성하십시오.)

## 추가적인 주의사항
- 반드시 [Thought -> Action -> Action Input -> Observation] 순서를 준수하십시오. 항상 Action 전에는 Thought가 먼저 나와야 합니다.
- 필요한 정보를 충분히 얻었다면 절대 'Action: 없음'이나 임의의 도구를 작성하지 마십시오.
- 정보를 확보한 즉시 'Thought: 이제 최종 답변을 알겠습니다'에 이어 바로 'Final Answer:'를 작성하여 종료하십시오.
- 동일한 키워드로 도구를 반복 호출하지 마십시오. 원하는 정보가 없으면 검색어를 영어(예: "market size", "market volume")로 바꾸거나 다른 도구를 시도하십시오.
- 최종 답변에는 최대한 많은 내용을 포함하십시오.
- 한 번의 검색으로 해결되지 않을 것 같다면 문제를 분할하여 푸는 것도 고려하십시오.
- 정보가 취합되었다면 불필요하게 사이클을 반복하지 마십시오.
- 묻지 않은 정보를 찾으려고 도구를 사용하지 마십시오.

시작하세요!

Question: {input}
{agent_scratchpad}'''


prompt = PromptTemplate.from_template(react_template)


def _format_tools_for_prompt(ts: List[object]) -> Tuple[str, str]:
    lines, names = [], []
    for t in ts:
        names.append(t.name)
        desc = getattr(t, "description", "")
        lines.append(f"{t.name}: {desc}")
    return "\n".join(lines), ", ".join(names)

def _render_prompt(user_input: str, scratchpad: str) -> str:
    tools_str, tool_names = _format_tools_for_prompt(tools)
    return prompt.format(
        tools=tools_str,
        tool_names=tool_names,
        input=user_input,
        agent_scratchpad=scratchpad,
    )


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# =========================
# ReAct 파서 및 실행 루프
# =========================
ACTION_RE = re.compile(r"^Action\s*:\s*(?P<tool>.+?)\s*$", re.MULTILINE)
ACTION_INPUT_RE = re.compile(r"^Action Input\s*:\s*(?P<input>.+?)\s*$", re.MULTILINE)
FINAL_ANSWER_RE = re.compile(r"Final Answer\s*:\s*(?P<final>[\s\S]+)$", re.IGNORECASE)


def _parse_action_and_input(text: str) -> Tuple[Optional[str], Optional[str]]:
    # 1. 텍스트 안에 'Final Answer:'가 있으면 무조건 최우선으로 즉시 종료 처리!
    m_final = FINAL_ANSWER_RE.search(text)
    if m_final:
        return "__FINAL__", m_final.group("final").strip()

    # 2. Final Answer가 없을 때만 Action 및 Action Input 탐색
    m_act = ACTION_RE.search(text)
    m_in = ACTION_INPUT_RE.search(text)
    if m_act and m_in:
        tool = m_act.group("tool").strip()
        action_input = (
            m_in.group("input").strip().splitlines()[0].strip().strip('"\'')
        )

        # 3. 모델이 'Action: 없음', 'none', '완료' 등을 뱉었을 때 에러 내지 않고 강제 종료
        if tool.lower() in [
            "없음",
            "none",
            "n/a",
            "finish",
            "종료",
            "완료",
        ]:
            return (
                "__FINAL__",
                text,
            )  # 파싱 실패로 되돌리지 않고 종료로 넘김

        return tool, action_input    

def _observation_to_text(observation_obj) -> str:
    if isinstance(observation_obj, list):
        # Document 리스트일 수 있음
        def doc_to_str(d):
            try:
                meta = getattr(d, "metadata", {}) or {}
                src = meta.get("source") or meta.get("file_path") or ""
                txt = getattr(d, "page_content", "")
                if len(txt) > 500:
                    txt = txt[:500] + "..."
                return f"[source={src}] {txt}"
            except Exception:
                return str(d)
        return "\n".join(doc_to_str(d) for d in observation_obj[:5])
    return str(observation_obj)

STOP_SEQ = ["\nObservation:"]  # LLM이 Observation 내용을 쓰기 직전에 출력이 끊기게 함

def run_react(user_input: str, max_iters: int = 8) -> Dict[str, str]:
    scratchpad = ""
    for _ in range(max_iters):
        rendered = _render_prompt(user_input, scratchpad)
        resp = llm.invoke(rendered, stop=STOP_SEQ)
        text = resp.content if hasattr(resp, "content") else str(resp)

        tool, action_input = _parse_action_and_input(text)
        if tool is None:
            hint = "\n[파싱안내] 형식을 엄격히 따르세요. 반드시 'Action:'와 'Action Input:'을 한 줄씩 제공하십시오.\n"
            scratchpad += f"{text}\n{hint}"
            continue

        if tool == "__FINAL__":
            final_answer = action_input
            return {"output": final_answer, "log": scratchpad + "\n" + text}

        if tool not in tool_map:
            observation = f"[에러] 존재하지 않는 도구입니다: {tool}"
            scratchpad += f"{text}\nObservation: {observation}\n"
            continue

        try:
            observation_obj = tool_map[tool].invoke(action_input)
            observation = _observation_to_text(observation_obj)
            scratchpad += f"{text}\nObservation: {observation}\n"
        except Exception as e:
            scratchpad += f"{text}\nObservation: [도구실행오류] {e}\n"

    return {
        "output": "반복 한도를 초과했습니다. 질문을 더 구체화해 주세요.",
        "log": scratchpad,
    }
     


# =========================
# 단건 질의
# =========================
try:
    q1 = "미국의 블록체인 시장 규모는 어떤가요?"
    out1 = run_react(q1, max_iters=8)
    print("최종 답변:", out1["output"])
    print("\n=== 실행 로그 (Thought / Action / Action Input / Observation) ===\n")
    print(out1["log"])
except Exception as e:
    print(f"단건 질의 중 오류: {e}")