import os, torch 
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM # SentenceTransformer 모델 불러오기
import chromadb 
from tqdm import tqdm 

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# Arrange GPU devices starting from 0 os.environ["CUDA_VISIBLE_DEVICES"]= "0,1,4,5"
# Set the GPUs 2 and 3 to use 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', device) 
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count()) 


model_st = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS') # LDCC/LDCC-SOLAR-10.7B 모델과 토크나이저 불러오기
tokenizer = AutoTokenizer.from_pretrained("kfkas/Legal-Llama-2-ko-7b-Chat", device_map="auto") 
model_ldcc = AutoModelForCausalLM.from_pretrained("kfkas/Legal-Llama-2-ko-7b-Chat", device_map="auto", rope_scaling={"type": "dynamic", "factor": 2}) 


chroma_client = chromadb.Client() # Initialize ChromaDB client and create collection
# client = chromadb.PersistentClient() 
LSA = chroma_client.create_collection(name="LSA")
OSHA = chroma_client.create_collection(name="OSHA") 
IACI = chroma_client.create_collection(name="IACI")

# Load the model for embeddings
model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS') 

# Directory containing text files
directory_path_LSA = "./근로기준법" 
directory_path_OSHA = "./산업안전보건법" 
directory_path_IACI = "./산업재해보상보험법"


# Process each text file in the directory 
def process_and_insert_text_file_OSHA(file_path):
  """ 텍스트 파일을 읽어 내용을 두 개의 줄바꿈으로 분할하고, 각 섹션을 별도로 처리하여 ChromaDB에 저장합니다. """ 
  with open(file_path, 'r', encoding='utf-8') as file: content = file.read() 
  sections = content.split('\n\n') 
  # 두 개의 줄바꿈으로 섹션 분할 
  for index, section in enumerate(tqdm(sections)): 
    # 각 섹션을 하나의 'query-answer' 쌍으로 가정 
    metadata = { "query": f"Section {index + 1} of {os.path.basename(file_path)}", "answer": section.strip() }
    embedding = model.encode(section.strip(), normalize_embeddings=True).tolist() # ChromaDB에 데이터 삽입 
    OSHA.add(embeddings=[embedding], ids=[f"{file_path}-{index}"], metadatas=[metadata])
    
    
def process_and_insert_text_file_LSA(file_path): 
	""" 텍스트 파일을 읽어 내용을 두 개의 줄바꿈으로 분할하고, 각 섹션을 별도로 처리하여 ChromaDB에 저장합니다. """ 
	with open(file_path, 'r', encoding='utf-8') as file: content = file.read()
	sections = content.split('\n\n') 
	# 두 개의 줄바꿈으로 섹션 분할 
	for index, section in enumerate(tqdm(sections)): 
		# 각 섹션을 하나의 'query-answer' 쌍으로 가정 
		metadata = { "query": f"Section {index + 1} of {os.path.basename(file_path)}", "answer": section.strip() } 
		embedding = model.encode(section.strip(), normalize_embeddings=True).tolist() # ChromaDB에 데이터 삽입 
		LSA.add(embeddings=[embedding], ids=[f"{file_path}-{index}"], metadatas=[metadata])


def process_and_insert_text_file_IACI(file_path):
	""" 텍스트 파일을 읽어 내용을 두 개의 줄바꿈으로 분할하고, 각 섹션을 별도로 처리하여 ChromaDB에 저장합니다. """ 
	with open(file_path, 'r', encoding='utf-8') as file: content = file.read() 
	sections = content.split('\n\n') # 두 개의 줄바꿈으로 섹션 분할 
	for index, section in enumerate(tqdm(sections)):
		# 각 섹션을 하나의 'query-answer' 쌍으로 가정 
		metadata = { "query": f"Section {index + 1} of {os.path.basename(file_path)}", "answer": section.strip() }
		embedding = model.encode(section.strip(), normalize_embeddings=True).tolist()
		# ChromaDB에 데이터 삽입 
		IACI.add(embeddings=[embedding], ids=[f"{file_path}-{index}"], metadatas=[metadata])
  
  
def insert_data_to_chromadb_with_section_split_LSA(directory):
	files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.txt')]
	for file_path in tqdm(files):
		process_and_insert_text_file_LSA(file_path)

def insert_data_to_chromadb_with_section_split_OSHA(directory):
	files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.txt')]
	for file_path in tqdm(files): process_and_insert_text_file_OSHA(file_path)

def insert_data_to_chromadb_with_section_split_IACI(directory):
	files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.txt')] 
	for file_path in tqdm(files):
		process_and_insert_text_file_IACI(file_path)
		insert_data_to_chromadb_with_section_split_LSA(directory_path_LSA) 
		insert_data_to_chromadb_with_section_split_OSHA(directory_path_OSHA) 
		insert_data_to_chromadb_with_section_split_IACI(directory_path_IACI) 

# ChromaDB 클라이언트 초기화 
client = chromadb.Client()
answers = chroma_client.create_collection(name="answers")
answers = client.get_collection(name="LSA")

### 근로기준법 
def get_related_answers_LSA(question, n_results=3):
	query_embedding = model_st.encode(question, normalize_embeddings=True).tolist()
	result = LSA.query(query_embeddings=query_embedding, n_results=n_results)
	answers_only = [meta['answer'] for meta in result['metadatas'][0]]
	print("get_related_answers_LSA") 
	return answers_only 

### 산업재해보상보험
def get_related_answers_IACI(question, n_results=3):
	query_embedding = model_st.encode(question, normalize_embeddings=True).tolist() 
	result = IACI.query(query_embeddings=query_embedding, n_results=n_results)
	answers_only = [meta['answer'] for meta in result['metadatas'][0]] 
	print("get_related_answers_IACI") 
	return answers_only

### 산업안전보건법 
def get_related_answers_OSHA(question, n_results=3):
	query_embedding = model_st.encode(question, normalize_embeddings=True).tolist()
	result = OSHA.query(query_embeddings=query_embedding, n_results=n_results)
	answers_only = [meta['answer'] for meta in result['metadatas'][0]]
	print("get_related_answers_OSHA")
	return answers_only

def generate_response(context, question, max_tokens=524):
	prompt = f""" 너는 변호사야. 아래의 질문에 대해 해결방안을 제시해줘.이때 관련법령을 참고해서, 구체적으로 답변해주면 좋아 질문 : {question} 관련 법령: {context} """ 
	inputs = tokenizer(prompt, return_tensors="pt").to(model_ldcc.device)
	generated = model_ldcc.generate( **inputs, max_new_tokens=max_tokens, early_stopping=True, do_sample=True, top_k=20, top_p=0.92, no_repeat_ngram_size=3, eos_token_id=2, repetition_penalty=1.2, num_beams=3 )
	response = tokenizer.decode(generated[0], skip_special_tokens=True) 
	return response

### 근로기준법
def process_user_question_LSA(user_question):
	# ChromaDB에서 관련된 답변 검색
	related_answers = get_related_answers_LSA(user_question)
	# 검색된 답변을 컨텍스트로 사용하여 최종 답변 생성 
	context = " ".join(related_answers)
	final_response = generate_response(context, user_question) 
	print("process_user_question_LSA") 
	return final_response

### 산업재해보상보험 
def process_user_question_IACA(user_question):
	# ChromaDB에서 관련된 답변 검색 
	related_answers = get_related_answers_IACI(user_question)
	# 검색된 답변을 컨텍스트로 사용하여 최종 답변 생성 
	context = " ".join(related_answers)
	final_response = generate_response(context, user_question) 
	print("process_user_question_IACA")
	return final_response

### 산업안전보건법 
def process_user_question_OSHA(user_question):
	# ChromaDB에서 관련된 답변 검색 
	related_answers = get_related_answers_OSHA(user_question) 
	# 검색된 답변을 컨텍스트로 사용하여 최종 답변 생성
	context = " ".join(related_answers) 
	final_response = generate_response(context, user_question) 
	print("process_user_question_OSHA") 
	return final_response

# # # # 사용자의 질문을 받아 처리하는 예시
# user_question = "동료가 업무 중 사망했습니다. 유족은 어떤 보상을 받을 수 있나요?"
# final_answer = process_user_question_LSA(user_question) 
# print(final_answer)

user_question = "저는 한국에서 일하던 외국인 근로자입니다. 업무상 재해로 인해 요양 중인데 곧 귀국을 해야 합니다. 이럴 경우 보험급여를 일시적으로 받을 수 있나요?" 
final_answer = process_user_question_IACA(user_question) 
print(final_answer)

# user_question = "제조 공장에서 일하던 중 기계의 결함으로 손가락을 다쳤습니다. 이 상황에서 제 부상을 산업재해로 볼 수 있나요?
# final_answer = process_user_question_OSHA(user_question)
# r_answer = final_answer.split( sep = 'Answer:') 
# print(real_answer)
# print(final_answer)