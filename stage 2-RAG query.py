
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os


def setup_working_rag():
    """Working RAG setup with local pipeline"""
    # 1. Load FAISS (we know this works)
    embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local("faiss_index_folder", embeddings)
    print("✅ FAISS loaded")

    # 2. Initialize LLM using transformers pipeline (more reliable)
    model_name = "google/flan-t5-base"

    try:
        # Create a local pipeline instead of using HuggingFaceHub
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=200,
            temperature=0.3,
            repetition_penalty=1.1
        )

        llm = HuggingFacePipeline(pipeline=pipe)
        print("✅ LLM initialized locally")

    except Exception as e:
        print(f"❌ Failed to load local LLM: {e}")
        return None

    # 3. Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 1}),
        return_source_documents=True
    )

    return qa_chain


def test_rag():
    """Test the RAG system"""
    qa_chain = setup_working_rag()

    if not qa_chain:
        print("❌ RAG setup failed")
        return

    test_queries = [
        "Find reviews from disgusted customers",
        "Show me reviews from sad users",
        "What are people's opinions?"
    ]

    for query in test_queries:
        print(f"\n" + "=" * 50)
        print(f"QUERY: {query}")
        print("=" * 50)

        try:
            result = qa_chain({"query": query})
            print(f"ANSWER: {result['result']}")

            print(f"\nSOURCE DOCUMENTS ({len(result['source_documents'])}):")
            for i, doc in enumerate(result['source_documents']):
                print(f"{i + 1}. {doc.page_content}")

        except Exception as e:
            print(f"ERROR: {e}")


if __name__ == "__main__":
    test_rag()