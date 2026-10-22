"""
Phase 2: Streamlit Frontend
This is the chat interface where users interact with the RAG bot.
"""

import streamlit as st
import requests

# ============================================================================
# 1. PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Symposium RAG Bot",
    page_icon="🤖",
    layout="centered"
)


# ============================================================================
# 2. BACKEND API CONFIGURATION
# ============================================================================

BACKEND_URL = "http://localhost:8000/chat"


# ============================================================================
# 3. INITIALIZE CHAT HISTORY
# ============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================================
# 4. UI HEADER
# ============================================================================

st.title("🤖 Symposium RAG Bot")
st.markdown("Ask me anything about the symposium document!")
st.divider()


# ============================================================================
# 5. DISPLAY CHAT HISTORY
# ============================================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Display cache badge + sources for assistant messages
        if message["role"] == "assistant":
            if message.get("cache_hit"):
                sim = message.get("similarity")
                st.caption(f"⚡ Cached  ·  similarity {sim:.4f}" if sim else "⚡ Cached")
            if message.get("sources"):
                with st.expander("📚 Sources"):
                    for source in message["sources"]:
                        st.caption(source)


# ============================================================================
# 6. USER INPUT HANDLING
# ============================================================================

if user_question := st.chat_input("Type your question here..."):

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_question)

    # Add user message to chat history
    st.session_state.messages.append({
        "role": "user",
        "content": user_question
    })


    # ========================================================================
    # 7. CALL BACKEND API
    # ========================================================================

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            # Send request to backend
            response = requests.post(
                BACKEND_URL,
                json={"question": user_question}
            )

            result = response.json()
            answer = result["answer"]
            sources = result.get("sources", [])
            cache_hit = result.get("cache_hit", False)
            similarity = result.get("similarity")


            # ====================================================================
            # 8. DISPLAY ASSISTANT RESPONSE
            # ====================================================================

            if cache_hit:
                st.caption(f"⚡ Cached  ·  similarity {similarity:.4f}" if similarity else "⚡ Cached")

            st.markdown(answer)

            # Show sources in an expander (only on cache misses)
            if sources:
                with st.expander("📚 Sources"):
                    for source in sources:
                        st.caption(source)


    # Add assistant message to chat history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "cache_hit": cache_hit,
        "similarity": similarity,
    })
