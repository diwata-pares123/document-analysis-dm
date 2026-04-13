from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def run_intelligent_analysis(target_text, corpus):
   
    vectorizer = TfidfVectorizer(stop_words='english')
    all_content = [target_text] + corpus
    
    try:
        tfidf_matrix = vectorizer.fit_transform(all_content)
    except ValueError:
       
        return [{"score": 0.0, "keywords": []} for _ in corpus]
    
    
    scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    feature_names = vectorizer.get_feature_names_out()
    
    results = []
    for i, score in enumerate(scores):
        doc_vec = tfidf_matrix[i+1].toarray().flatten()
        
        top_indices = doc_vec.argsort()[-5:][::-1]
        keywords = [feature_names[idx] for idx in top_indices if doc_vec[idx] > 0]
        
        results.append({
            "score": round(float(score) * 100, 2),
            "keywords": keywords
        })
        
    return results


def run_rag_retrieval(query, texts):
    """
    Chunks documents, stores them in an in-memory vector database, 
    and retrieves only the paragraphs relevant to the user's query.
    """
    if not query or not texts:
        return []
    
    full_text = " \n\n ".join(texts)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(full_text)
    
    if not chunks:
        return []

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = FAISS.from_texts(chunks, embeddings)
    
    docs = vectorstore.similarity_search(query, k=3)
    
    return [doc.page_content for doc in docs]