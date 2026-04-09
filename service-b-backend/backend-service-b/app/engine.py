from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- NEW IMPORTS FOR RAG ---
# UPDATED: Using the new langchain_text_splitters package
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# =================================================================
# 1. YOUR ORIGINAL ENGINE (TF-IDF & Cosine Similarity)
# =================================================================
def run_intelligent_analysis(target_text, corpus):
    # 1. Setup Vectorizer
    # Converts text to high-dimensional sparse semantic vectors
    vectorizer = TfidfVectorizer(stop_words='english')
    all_content = [target_text] + corpus
    
    try:
        tfidf_matrix = vectorizer.fit_transform(all_content)
    except ValueError:
        # Failsafe if the text is empty or only contains stop words
        return [{"score": 0.0, "keywords": []} for _ in corpus]
    
    # 2. Compute Cosine Similarity
    # Evaluates angle orientation between the target (idx 0) and the corpus, ignoring absolute length magnitude (unlike Euclidean distance)
    scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    feature_names = vectorizer.get_feature_names_out()
    
    # 3. Compile Raw Data Output
    results = []
    for i, score in enumerate(scores):
        doc_vec = tfidf_matrix[i+1].toarray().flatten()
        
        # Get top 5 highest ranking TF-IDF keywords for this document
        top_indices = doc_vec.argsort()[-5:][::-1]
        keywords = [feature_names[idx] for idx in top_indices if doc_vec[idx] > 0]
        
        results.append({
            "score": round(float(score) * 100, 2),
            "keywords": keywords
        })
        
    return results

# =================================================================
# 2. NEW RAG ENGINE (LangChain + Vector Retrieval)
# =================================================================
def run_rag_retrieval(query, texts):
    """
    Chunks documents, stores them in an in-memory vector database, 
    and retrieves only the paragraphs relevant to the user's query.
    """
    if not query or not texts:
        return []
    
    # 1. Combine all text into one giant string
    full_text = " \n\n ".join(texts)
    
    # 2. Chunking: Break the document into readable 1000-character blocks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(full_text)
    
    if not chunks:
        return []

    # 3. Embeddings: Use HuggingFace (Free, Local, Fast) to understand semantic meaning
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 4. Vector Database: Create a temporary FAISS database in memory
    vectorstore = FAISS.from_texts(chunks, embeddings)
    
    # 5. Retrieval: Fetch the top 3 most relevant chunks based on the user's prompt
    docs = vectorstore.similarity_search(query, k=3)
    
    # Return just the text content of those top chunks
    return [doc.page_content for doc in docs]