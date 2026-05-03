import streamlit as st
from backend.core import run_llm
from typing import List, Dict, Any

def _format_sources(context_docs: List[Any]) -> List[str]:
    return [
        str((meta.get("source") or "Unknown"))
        for doc in (context_docs or [])
        if (meta := (getattr(doc, "metadata", None) or {})) is not None
    ]

st.set_page_config(page_title="LangChain Documentation Helper", page_icon=":book:", layout="centered")
st.title("LangChain Documentation Helper")

with st.sidebar:
    st.subheader("Session")
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.pop("messages", None)
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I'm your LangChain documentation assistant. How can I help you today?",
            "sources": []
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if (msg.get("sources") and len(msg["sources"]) > 0):
            with st.expander("Sources"):
                for src in msg["sources"]:
                    st.markdown(f"**{src}**")

prompt = st.chat_input("Ask me anything about LangChain....")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt,"sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving relevant documentation..."):
                result: Dict[str, Any] = run_llm(prompt)
                answer = str(result.get("answer","")).strip() or "Sorry, I don't know the answer to that question."
                sources = _format_sources(result.get("context",[]))
            
            st.markdown(answer)
            if sources:
                with st.expander("Sources"):
                    for src in sources:
                        st.markdown(f"**{src}**")
            
            st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.session_state.messages[-1]["content"] = "Sorry, I encountered an error. Please try again."
            st.session_state.messages[-1]["sources"] = []
        finally:
            st.rerun()
