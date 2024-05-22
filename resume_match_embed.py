# %%
from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.embeddings import OllamaEmbeddings
from sentence_transformers import SentenceTransformer, util
from scipy.spatial.distance import cosine
import os
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tqdm import tqdm

# %%
# embeddings = OllamaEmbeddings()
embeddings =  SentenceTransformer('all-MiniLM-L6-v2')

# %%
job_description_loader = PyPDFLoader("python-job-description.pdf")
job_description = job_description_loader.load()
job_description_text = job_description[0].page_content

# %%
candidate_resumes = []
resume_dir = "path/to/your/resumes/directory"
for filename in os.listdir(resume_dir):
    if filename.endswith(".pdf"):
        file_path = os.path.join(resume_dir, filename)
        resume_loader = PyPDFLoader(file_path)
        resume = resume_loader.load()
        resume_text = resume[0].page_content
        candidate_name = os.path.splitext(filename)[0]  # Extract candidate name from file name
        candidate_resumes.append((candidate_name, resume_text))


# %%
# job_description_embedding = embeddings.embed_query(job_description_text)
# resume_embeddings = [embeddings.embed_documents([resume_text])[0] for _, resume_text in candidate_resumes]

job_description_embedding = embeddings.encode(job_description_text)
resume_embeddings = [embeddings.encode(resume_text) for _, resume_text in candidate_resumes]

# %%
relevancy_scores = [1 - cosine(resume_embedding, job_description_embedding) for resume_embedding in resume_embeddings]

# %%
relevancy_percentages = [f"{round((score * 100),2)}%" for score in relevancy_scores]

# %%
# ranked_candidates = sorted(zip(candidate_resumes, relevancy_percentages), key=lambda x: x[1], reverse=True)
ranked_candidates = sorted(zip(candidate_resumes, relevancy_scores), key=lambda x: x[1], reverse=True)


# %%
ranked_candidates

# %%
llm = Ollama(model="llama3:8b",  temperature=0)

# %%
def generate_explanation(job_description_text,resume_text,relevancy_scores):
    prompt = PromptTemplate(
        template="Given the following job description:\n\n{job_description_text}\n\nAnd the following resume:\n\n{resume_text}\n\nWith a relevancy score of {relevancy_scores:.2f}, provide a rationale or justification for why this resume is relevant and what not matches for the job description.",
        input_variables=["job_description_text", "resume_text", "relevancy_scores"]
    )
    chain  = prompt | llm | StrOutputParser()
    response = chain.invoke({"job_description_text": job_description_text, "resume_text": resume_text, "relevancy_scores": relevancy_scores})
    return response

# %%
output_file = "relevancy_explanations1.txt"
with open(output_file, "w") as file:
    for candidate, score in tqdm(ranked_candidates):
        candidate_name, resume_text = candidate
        explanation = generate_explanation(job_description_text, resume_text, score)
        file.write(f"Candidate: {candidate_name}, Relevancy Score: {score:.2f}")
        file.write(f"Explanation: {explanation}")
        print(f"Candidate: {candidate_name}, Relevancy Score: {score:.2f}")
        print(f"Explanation: {explanation}")
        print()

# %%
# ranked_candidates = sorted(zip(candidate_resumes, relevancy_scores, relevancy_rationale), key=lambda x: x[1], reverse=True)


