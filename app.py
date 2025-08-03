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
        "You are Jerry AI, the company's AI assistant. You have direct knowledge of all company information including policies, procedures, people, and data. "
        "You can also reference information from our previous conversation. "
        "Answer questions directly and naturally without mentioning sources like 'according to company documents' or 'based on our data'. "
        "Respond as if you naturally know this information as part of being the company's AI assistant. "
        "When someone asks about specific people or their details, provide ALL the relevant information you have about them including full names, contact details, fun facts, and any other specifics in a friendly, personal way. "
        "Use first person pronouns when appropriate (e.g., 'Your buddy is Sneha' rather than 'Sneha is mentioned as a buddy'). "
        "Avoid generic responses - be specific and use the actual details available. "
        "If you don't have the information, simply say 'Sorry, I don't have that information.' "
        "Keep the answer concise and relevant. "
        "If someone asks who you are, introduce yourself as Jerry AI. "
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

def query_grounded_chain(client, config, retriever, question, conversation_history=None):
    """Query the grounded chain with both RAG context and web search - returns streaming response"""
    
    # Get relevant documents from RAG
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Create enhanced prompt with both RAG context and web search capability
    conversation_context = ""
    if conversation_history:
        conversation_context = f"""
Previous conversation:
{conversation_history}
"""
    
    enhanced_prompt = f"""
    You are Jerry AI, the company's AI assistant. You have direct knowledge of company information and access to web search.
    You can also reference information from our previous conversation.
    
    Company Information Available:
    {context}
    {conversation_context}
    Question: {question}
    
    Instructions:
    1. You have direct knowledge of company information - answer naturally without mentioning sources
    2. Do not say things like "according to company documents" or "based on our data" - just answer directly
    3. Respond as if you naturally know this information (e.g., 'Your buddy is Sneha' not 'Sneha is mentioned as a buddy')
    4. When someone asks about specific people, provide ALL relevant details you have about them including full names, contact details, fun facts, and any other specifics in a friendly, personal way
    5. Avoid generic responses - be specific and use the actual details available
    6. For company-related questions, use your company knowledge first
    7. Use web search to supplement when you need current/external information
    8. For recent information not in company data, rely on web search
    9. If you can't find relevant information anywhere, simply say you don't have that information
    10. If someone asks who you are, introduce yourself as Jerry AI
    11. If someone tells you their name, remember it and use it in future responses to personalize the conversation
    12. Reference previous conversation context when relevant
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
    st.set_page_config(page_title="Jerry AI", page_icon="🤖")
    # st.title("Jerry AI")

    # Sidebar for settings
    with st.sidebar:
        # Jerry AI branding
        st.markdown(
            """
            <div style='display: flex; flex-direction: column; align-items: center; text-align: center;'>
                <h1 style='color: #FF6B35; margin: 0; font-size: 2.5rem;'>Jerry AI</h1>
                <p style='color: #888; font-style: italic; margin: 5px 0 20px 0;'>Your AI Buddy</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Sample questions with better styling
        st.markdown("---")
        st.markdown("### 💭 **Sample Questions**")
        
        # Create expandable sections for better organization
        with st.container():

            st.markdown("""
            <div style='background-color: #2E3440; color: white; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #1E88E5;'>
                <strong>🚀 Getting Started:</strong><br>
                • What should I do in my first week?<br>
                • Tell me about the onboarding process
            </div>
            """, unsafe_allow_html=True)


            
            st.markdown("""
            <div style='background-color: #2E3440; color: white; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #FF6B35;'>
                <strong>🏢 Company Info:</strong><br>
                • What are the company values?<br>
                • Tell me about the company culture
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style='background-color: #2E3440; color: white; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #4C9A2A;'>
                <strong>👥 Contacts & Support:</strong><br>
                • Who is my HR contact?<br>
                • How do I contact IT support?
            </div>
            """, unsafe_allow_html=True)
            

        st.divider()
        
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
            st.info("📄 Document-only mode - responses are based on Company Policies, Procedures, People, and Data")
    
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
            "content": "Hi there! 👋 Welcome to the team! \n\nI'm Jerry, your AI assistant. I'm here to help you with any questions or doubts you might have about our company, policies, procedures, or anything work-related. \n\nFeel free to ask me anything . Let's make your onboarding journey smooth and productive! 🚀\n\nWhat would you like to know? (Feel free to tell me your name so I can personalize our conversation!)"
        }
        st.session_state.history.append(welcome_message)

    # Display chat history
    for message in st.session_state.history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Sources display removed per user request

    if query := st.chat_input("Ask a question ..."):
        # Add user message to history and display it
        st.session_state.history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
            
        with st.chat_message("assistant"):
            try:
                if use_web_search and 'client' in locals():
                    # Use grounded search with streaming
                    # Include conversation history
                    conversation_context = ""
                    if len(st.session_state.history) > 1:  # More than just the welcome message
                        recent_history = st.session_state.history[-6:]  # Last 6 messages (3 exchanges)
                        for msg in recent_history:
                            if msg["role"] == "user":
                                conversation_context += f"User: {msg['content']}\n"
                            elif msg["role"] == "assistant":
                                conversation_context += f"Assistant: {msg['content']}\n"
                    
                    with st.spinner("Searching documents and web..."):
                        response_stream, rag_docs = query_grounded_chain(client, config, retriever, query, conversation_context)
                    
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
                        
                        # Add citations to the answer only if there are actual web sources
                        answer_with_citations = add_citations_to_text(final_response)
                        
                        # Check if there are actual web sources before showing citations
                        has_web_sources = False
                        if hasattr(final_response, 'candidates') and final_response.candidates:
                            candidate = final_response.candidates[0]
                            if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                                grounding_metadata = candidate.grounding_metadata
                                grounding_chunks = getattr(grounding_metadata, 'grounding_chunks', [])
                                if grounding_chunks:  # Check if chunks exist
                                    for chunk in grounding_chunks:
                                        web_info = getattr(chunk, 'web', None)
                                        if web_info:
                                            has_web_sources = True
                                            break
                        
                        # Only show citations if there are actual web sources and the text has changed
                        if answer_with_citations != answer and has_web_sources:
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
                        
                        # RAG documents display removed per user request
                
                else:
                    # Use standard RAG-only chain with streaming
                    # Include conversation history in the input
                    conversation_context = ""
                    if len(st.session_state.history) > 1:  # More than just the welcome message
                        recent_history = st.session_state.history[-6:]  # Last 6 messages (3 exchanges)
                        for msg in recent_history:
                            if msg["role"] == "user":
                                conversation_context += f"User: {msg['content']}\n"
                            elif msg["role"] == "assistant":
                                conversation_context += f"Assistant: {msg['content']}\n"
                    
                    # Combine current query with conversation context
                    enhanced_query = f"""Previous conversation:
{conversation_context}

Current question: {query}"""
                    
                    def rag_stream_generator():
                        for chunk in qa.stream({"input": enhanced_query}):
                            if "answer" in chunk:
                                yield chunk["answer"]
                    
                    # Stream the RAG response
                    answer = st.write_stream(rag_stream_generator())
                    
                    # Source documents display removed per user request
                
            except Exception as e:
                answer = f"Error during generation: {e}"
                st.markdown(answer)

            # Store message in history (include citations if they exist)
            final_content = answer
            if use_web_search and hasattr(st.session_state, 'last_grounded_response'):
                final_response = st.session_state.last_grounded_response
                answer_with_citations = add_citations_to_text(final_response)
                
                # Check if there are actual web sources
                has_web_sources = False
                if hasattr(final_response, 'candidates') and final_response.candidates:
                    candidate = final_response.candidates[0]
                    if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                        grounding_metadata = candidate.grounding_metadata
                        grounding_chunks = getattr(grounding_metadata, 'grounding_chunks', [])
                        if grounding_chunks:
                            for chunk in grounding_chunks:
                                web_info = getattr(chunk, 'web', None)
                                if web_info:
                                    has_web_sources = True
                                    break
                
                # If there are web sources and citations, include them in the stored content
                if answer_with_citations != answer and has_web_sources:
                    final_content = f"{answer}\n\n**With citations:**\n{answer_with_citations}"
            
            message_data = {"role": "assistant", "content": final_content}
            st.session_state.history.append(message_data)


if __name__ == "__main__":
    main()
