# app.py
import os
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

# ─── LangChain 0.2+ Community Imports ─────────────────────────────────
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# ─── Google GenAI SDK for Grounding ─────────────────────────────────
from google import genai
from google.genai import types

# ─── Configuration ───────────────────────────────────────────────────
TOP_K = 5
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-2.5-flash"
API_KEY_ENV = "GEMINI_API_KEY"

load_dotenv()

@st.cache_resource(show_spinner=False)
def build_retriever():
   
    txt_dir = Path(__file__).parent / "documents"
    if not txt_dir.exists():
        st.error(f"Missing folder: {txt_dir}")
        return None

    loader = DirectoryLoader(
        str(txt_dir),
        glob="**/*.txt",
        loader_cls=TextLoader
    )
    raw_docs = loader.load()
    if not raw_docs:
        st.warning("No `.txt` found in `documents/`. Add files and press R.")
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=200)
    chunks = splitter.split_documents(raw_docs)

    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectordb = FAISS.from_documents(
        documents=chunks,
        embedding=embedder,
    )
    return vectordb.as_retriever(search_kwargs={"k": TOP_K})

@st.cache_resource(show_spinner=False)
def build_qa_chain():
    
    retriever = build_retriever()
    if retriever is None:
        return None

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=os.getenv('GEMINI_API_KEY'),
        disable_streaming=False,
        temperature=0.0,
    )

    # Create the system prompt using the modern ChatPromptTemplate
    system_prompt = (
        "You are Buddy AI, a helpful assistant. Use the given context to answer the question. "
        "You can also reference information from our previous conversation. "
        "If you don't know the answer based on the context or conversation, say 'Sorry, I don't know based on the company data.' "
        "Keep the answer concise and relevant. "
        "If someone asks who you are, introduce yourself as Buddy AI. "
        "If someone tells you their name, remember it and use it in future responses to personalize the conversation.\n\n"
        "Context: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # Create the question-answer chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    
    # Create the retrieval chain
    return create_retrieval_chain(retriever, question_answer_chain)

@st.cache_resource(show_spinner=False)
def build_grounded_chain():
    """Build a grounded chain using Google GenAI SDK with native Google Search"""
    
    retriever = build_retriever()
    if retriever is None:
        return None, None
    
    # Configure the Google GenAI client
    client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
    
    # Define the grounding tool
    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    
    # Configure generation settings
    config = types.GenerateContentConfig(tools=[grounding_tool])
    
    return client, config, retriever

def query_grounded_chain(client, config, retriever, question):
    """Query the grounded chain with both RAG context and web search - returns streaming response"""
    
    # Get relevant documents from RAG
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Create enhanced prompt with both RAG context and web search capability
    enhanced_prompt = f"""
    You are Buddy AI, a helpful assistant. You have access to company documents and web search.
    You can also reference information from our previous conversation.
    
    Company Documents Context:
    {context}
    
    Question: {question}
    
    Instructions:
    1. First, check if the company documents contain relevant information
    2. If the company documents have sufficient information, use that primarily
    3. Use web search to supplement or verify information when needed
    4. If the answer requires recent information not in company documents, rely more on web search
    5. Always be clear about which sources you're using
    6. If you can't find relevant information in either source, say so clearly
    7. If someone asks who you are, introduce yourself as Buddy AI
    8. If someone tells you their name, remember it and use it in future responses to personalize the conversation
    9. Reference previous conversation context when relevant
    """
    
    # Make the streaming request with grounding
    response_stream = client.models.generate_content_stream(
        model="gemini-2.5-flash",
        contents=enhanced_prompt,
        config=config,
    )
    
    return response_stream, docs

def add_citations_to_text(response):
    """Add inline citations to the response text using grounding metadata"""
    
    if not hasattr(response, 'candidates') or not response.candidates:
        return response.text if hasattr(response, 'text') else str(response)
    
    candidate = response.candidates[0]
    if not hasattr(candidate, 'grounding_metadata') or not candidate.grounding_metadata:
        return response.text if hasattr(response, 'text') else str(response)
    
    text = response.text
    grounding_metadata = candidate.grounding_metadata
    
    supports = getattr(grounding_metadata, 'grounding_supports', [])
    chunks = getattr(grounding_metadata, 'grounding_chunks', [])
    
    if not supports or not chunks:
        return text
    
    # Sort supports by end_index in descending order to avoid shifting issues when inserting
    sorted_supports = sorted(supports, key=lambda s: getattr(s.segment, 'end_index', 0), reverse=True)
    
    for support in sorted_supports:
        segment = getattr(support, 'segment', None)
        if not segment:
            continue
            
        end_index = getattr(segment, 'end_index', None)
        chunk_indices = getattr(support, 'grounding_chunk_indices', [])
        
        if end_index is not None and chunk_indices:
            # Create citation string like [1](link1), [2](link2)
            citation_links = []
            for i in chunk_indices:
                if i < len(chunks):
                    chunk = chunks[i]
                    web_info = getattr(chunk, 'web', None)
                    if web_info:
                        uri = getattr(web_info, 'uri', '')
                        title = getattr(web_info, 'title', f'Source {i+1}')
                        citation_links.append(f"[{title}]({uri})")
            
            if citation_links:
                citation_string = " " + ", ".join(citation_links)
                text = text[:end_index] + citation_string + text[end_index:]
    
    return text

def main():
    st.set_page_config(page_title="Company-Doc Chatbot", page_icon="🤖")
    st.title("Buddy AI")

    # Sidebar for settings
    with st.sidebar:
        # st.header("⚙️ Settings")
        
        # Web search toggle
        use_web_search = st.checkbox(
            "🌐 Enable Web Search", 
            value=False,
            help="Enable Google Search grounding for real-time information"
        )
        
        if use_web_search:
            st.info("🔍 Web search enabled - responses will include real-time information and citations")
        else:
            st.info("📄 Document-only mode - responses based on your uploaded documents")
    
    # Initialize the appropriate chain based on settings
    if use_web_search:
        # Try to build grounded chain
        try:
            grounded_components = build_grounded_chain()
            if grounded_components[0] is None:
                st.error("Failed to initialize grounded search. Falling back to document-only mode.")
                use_web_search = False
                qa = build_qa_chain()
            else:
                client, config, retriever = grounded_components
                qa = None  # We'll handle grounded queries differently
        except Exception as e:
            st.error(f"Error initializing web search: {e}")
            st.error("Falling back to document-only mode.")
            use_web_search = False
            qa = build_qa_chain()
    else:
        # Standard RAG-only mode
        qa = build_qa_chain()
        if qa is None:
            return

    if "history" not in st.session_state:
        st.session_state.history = []
        
        # Add welcome message as the first message
        welcome_message = {
            "role": "assistant",
            "content": "Hi there! 👋 Welcome to the team! \n\nI'm Buddy, your AI assistant. I'm here to help you with any questions or doubts you might have about our company, policies, procedures, or anything work-related. \n\nFeel free to ask me anything - I have access to all our company documents and can also search the web for recent information when needed. Let's make your onboarding journey smooth and productive! 🚀\n\nWhat would you like to know? (Feel free to tell me your name so I can personalize our conversation!)"
        }
        st.session_state.history.append(welcome_message)

    # Display chat history
    for message in st.session_state.history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display stored sources if they exist (preserves sources across mode switches)
            if isinstance(message, dict) and message.get("sources"):
                sources = message["sources"]
                with st.expander(f"📄 Retrieved {len(sources)} source doc(s)"):
                    for i, doc in enumerate(sources, 1):
                        if hasattr(doc, 'page_content'):
                            snip = doc.page_content.strip().replace("\n", " ")
                            st.write(f"**#{i}**: _{snip[:250]}…_")
                        else:
                            st.write(f"**#{i}**: {str(doc)[:250]}...")

    if query := st.chat_input("Ask a question about your documents..."):
        # Add user message to history and display it
        st.session_state.history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
            
        with st.chat_message("assistant"):
            try:
                if use_web_search and 'client' in locals():
                    # Use grounded search with streaming
                    with st.spinner("Searching documents and web..."):
                        response_stream, rag_docs = query_grounded_chain(client, config, retriever, query)
                    
                    # Stream the response in real-time
                    def grounded_stream_generator():
                        full_text = ""
                        final_response = None
                        
                        for chunk in response_stream:
                            if hasattr(chunk, 'text') and chunk.text:
                                full_text += chunk.text
                                yield chunk.text
                            final_response = chunk
                        
                        # Store final response for metadata processing
                        st.session_state.last_grounded_response = final_response
                        st.session_state.last_rag_docs = rag_docs
                    
                    # Display streamed response
                    answer = st.write_stream(grounded_stream_generator())
                    
                    # Process and display grounding metadata after streaming is complete
                    if hasattr(st.session_state, 'last_grounded_response'):
                        final_response = st.session_state.last_grounded_response
                        
                        # Add citations to the answer
                        answer_with_citations = add_citations_to_text(final_response)
                        if answer_with_citations != answer:
                            st.markdown("**With citations:**")
                            st.markdown(answer_with_citations)
                        
                        # Display grounding info
                        if hasattr(final_response, 'candidates') and final_response.candidates:
                            candidate = final_response.candidates[0]
                            if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                                grounding_metadata = candidate.grounding_metadata
                                
                                # Show search queries used
                                search_queries = getattr(grounding_metadata, 'web_search_queries', [])
                                if search_queries:
                                    with st.expander(f"🔍 Web search queries used ({len(search_queries)})"):
                                        for i, query_text in enumerate(search_queries, 1):
                                            st.write(f"**{i}.** {query_text}")
                                
                                # Show grounding sources
                                grounding_chunks = getattr(grounding_metadata, 'grounding_chunks', [])
                                if grounding_chunks:
                                    with st.expander(f"🌐 Web sources ({len(grounding_chunks)})"):
                                        for i, chunk in enumerate(grounding_chunks, 1):
                                            web_info = getattr(chunk, 'web', None)
                                            if web_info:
                                                uri = getattr(web_info, 'uri', '')
                                                title = getattr(web_info, 'title', f'Source {i}')
                                                st.write(f"**{i}.** [{title}]({uri})")
                        
                        # Show RAG documents
                        if hasattr(st.session_state, 'last_rag_docs') and st.session_state.last_rag_docs:
                            rag_docs = st.session_state.last_rag_docs
                            with st.expander(f"📄 Company documents ({len(rag_docs)})"):
                                for i, doc in enumerate(rag_docs, 1):
                                    snip = doc.page_content.strip().replace("\n", " ")
                                    st.write(f"**#{i}**: _{snip[:250]}…_")
                
                else:
                    # Use standard RAG-only chain with streaming
                    def rag_stream_generator():
                        for chunk in qa.stream({"input": query}):
                            if "answer" in chunk:
                                yield chunk["answer"]
                    
                    # Stream the RAG response
                    answer = st.write_stream(rag_stream_generator())
                    
                    # Get the full response for source documents
                    full_output = qa.invoke({"input": query})
                    docs = full_output.get("context", [])
                    if docs:
                        with st.expander(f"📄 Retrieved {len(docs)} source doc(s)"):
                            for i, doc in enumerate(docs, 1):
                                snip = doc.page_content.strip().replace("\n", " ")
                                st.write(f"**#{i}**: _{snip[:250]}…_")
                
            except Exception as e:
                answer = f"Error during generation: {e}"
                st.markdown(answer)

            # Store message with sources in history (simple approach)
            message_data = {"role": "assistant", "content": answer}
            
            # Add sources if they exist
            if use_web_search and 'rag_docs' in locals():
                message_data["sources"] = rag_docs
            elif 'docs' in locals():
                message_data["sources"] = docs
            
            st.session_state.history.append(message_data)


if __name__ == "__main__":
    main()
