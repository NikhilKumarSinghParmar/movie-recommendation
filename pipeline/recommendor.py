import pandas as pd
import numpy as  np
import os
from dotenv import load_dotenv

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


from utils.save_and_load_file import (
    save_pkl_file,
    load_pkl_file
) 


load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH")
BOOK_EMBEDDINGS_PATH = os.getenv("BOOK_EMBEDDINGS_PATH")
EMBEDDING_IDS_PATH = os.getenv("EMBEDDING_IDS_PATH")
BOOK_LOOKUP_PATH = os.getenv("BOOK_LOOKUP_PATH")


# BUILD_AND_SAVE_FLAG = True
BUILD_AND_SAVE_FLAG = False



def build_and_save_model(
    df
):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    df["meta_data"] = df["meta_data"].fillna("").astype(str)

    book_embeddings = model.encode(
        df["meta_data"].tolist(),
        show_progress_bar = True
    )

    embedding_ids = df["isbn10"].astype(str).tolist()

    # book_lookup = df.set_index('isbn10').to_dict("index")

    book_lookup = (
        df.assign(isbn10=df["isbn10"].astype(str))
        .set_index("isbn10")
        .to_dict("index")
    )

    model.save(
        MODEL_PATH  
    )

    np.save(BOOK_EMBEDDINGS_PATH, book_embeddings)

    save_pkl_file(
        book_lookup,
        BOOK_LOOKUP_PATH
    )

    save_pkl_file(
        embedding_ids,
        EMBEDDING_IDS_PATH
    )




def load_model():
    model = SentenceTransformer(MODEL_PATH)
 
    book_embeddings = np.load(BOOK_EMBEDDINGS_PATH)
 
    embedding_ids = load_pkl_file(EMBEDDING_IDS_PATH)
 
    book_lookup = load_pkl_file(BOOK_LOOKUP_PATH)


    return model, book_embeddings, embedding_ids, book_lookup





def recommend_book(
    query,
    model,
    book_embeddings,
    embedding_ids,
    book_lookup,
    top_k = 10,
    threshold = 0.3
):
    query_embedding = model.encode([query])

    scores = cosine_similarity(
        query_embedding,
        book_embeddings
    ).flatten()

    top_indices = scores.argsort()[-top_k : ][ : : -1]

    results = []

    for idx in top_indices:
        score = scores[idx]

        if score >= threshold:
            isbn = embedding_ids[idx]
            metadata = book_lookup[isbn]

            results.append({
                "title": metadata["title"],
                "subtitle": metadata["subtitle"],
                "authors": metadata["authors"],
                "categories": metadata["categories"],
                "description": metadata["description"],
                "isbn10": isbn,
                "isbn13": metadata["isbn13"],
                "published_year": metadata.get("published_year"),
                "average_rating": metadata.get("average_rating"),
                "num_pages": metadata.get("num_pages"),
                "ratings_count": metadata.get("ratings_count"),
                "similarity_score": round(float(score), 4)
            })

    return pd.DataFrame(results)




def recommend_loop(
    model,
    book_embeddings,
    embedding_ids,
    book_lookup
):

    query = input("Enter a book title/genre/description: ").strip()

    if not query:
        print("Empty query.")
        return pd.DataFrame()

    recommendations = recommend_book(
        query=query,
        model=model,
        book_embeddings=book_embeddings,
        embedding_ids=embedding_ids,
        book_lookup=book_lookup,
        top_k=10,
        threshold=0.30
    )

    return recommendations





def show_recommendations(recommendations):
    # print("\nRecommended Books:\n")
    # print(recommendations)

    
    if recommendations.empty:
        print("No matching books found.")
        return

    for rank, (_, row) in enumerate(recommendations.iterrows(), start=1):
        print(f"\nRank: {rank}")
        print(f"Title: {row['title']}")
        print(f"Author: {row['authors']}")
        print(f"Category: {row['categories']}")
        print(f"ISBN10: {row['isbn10']}")
        print(f"ISBN13: {row['isbn13']}")
        print(f"Similarity Score: {row['similarity_score']}")
        print(f"Description: {row['description'][:300]}...")
        print(f"\n ----- \n")






def recommend_books(df):
    if BUILD_AND_SAVE_FLAG :
        build_and_save_model(df)
    
    model, book_embeddings, embedding_ids, book_lookup = load_model()


    while True:
        flag = input("Do you want recommendation? (1/0): ").strip()
        
        if flag == "0":
            break
            
        recommendations = recommend_loop(
            model,
            book_embeddings,
            embedding_ids,
            book_lookup
        )
            
        show_recommendations(recommendations)    