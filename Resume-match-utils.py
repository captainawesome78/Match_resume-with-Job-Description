# %%
import langchain
from sentence_transformers import SentenceTransformer, util
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# %%
model = SentenceTransformer('all-MiniLM-L6-v2')

# %%
job_description_loader = PyPDFLoader("python-job-description.pdf")
job_description = job_description_loader.load()
job_description_text = job_description[0].page_content

# %%
candidate_resumes = []
resume_dir = "path/to/resumes/directory"
for filename in os.listdir(resume_dir):
    if filename.endswith(".pdf"):
        file_path = os.path.join(resume_dir, filename)
        resume_loader = PyPDFLoader(file_path)
        resume = resume_loader.load()
        resume_text = resume[0].page_content
        candidate_name = os.path.splitext(filename)[0]
        candidate_resumes.append((candidate_name, resume_text))

# %%
job_description_embedding = model.encode(job_description_text, convert_to_tensor=True)
resume_embeddings = model.encode([resume_text for _, resume_text in candidate_resumes], convert_to_tensor=True)

# %%
cosine_scores = util.cos_sim(job_description_embedding, resume_embeddings)

# %%
cosine_scores = cosine_scores.squeeze().tolist()

# %%
ranked_candidates = sorted(zip(candidate_resumes, cosine_scores), key=lambda x: x[1], reverse=True)

# %%
for candidate, score in ranked_candidates:
    print(f"Candidate: {candidate[0]}, Relevancy Score: ({round((score*100),2)}%)")

# %%
llm = Ollama(model="llama3:8b", temperature = 0)

# %%
def generate_explanation(job_description, resume, score):
    prompt = PromptTemplate(
    template="Given the following job description:\n\n{job_description}\n\nAnd the following resume:\n\n{resume}\n\nWith a relevancy score of {score:.2f}, provide a rationale or justification for why this resume is relevant or not relevant for the job description.",
    input_variables=["job_description", "resume", "score"]
    )
    chain  = prompt | llm | StrOutputParser()
    response = chain.invoke({"job_description": job_description, "resume": resume, "score": score})
    return response

# %%
output_file = "relevancy_explanations.txt"
with open(output_file, 'w') as file:
    for candidate, score in ranked_candidates:
        explanation = generate_explanation(job_description_text, candidate[1], score)
        file.write(f"Candidate: {candidate[0]}, Relevancy Score: {score:.2f}")
        file.write(f"Explanation: {explanation}")
        print(f"Candidate: {candidate[0]}, Relevancy Score: {score:.2f}")
        print(f"Explanation: {explanation}")
        print()


