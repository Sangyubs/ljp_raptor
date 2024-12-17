'''
사용법 : 
이미 tree가 있다면 load_path를 지정해주고, 
없다면 add_documents()를 통해 문서를 추가하고 save_path를 지정해준다.
그 후, answer_question을 통해 질문에 대한 답을 얻을 수 있다.
'''
# from json import load
from qamodels import GPT4oModel
from retrieveraugmentation import RetrievalAugmentation, RetrievalAugmentationConfig

# config 설정
custom_qa = GPT4oModel() # or None
load_path = '../data/20241217_OpenAItree' # or None
custom_summarizer = None # or model class
custom_embedding = None # or model class

# tree load : load_path = None 이면 생성 후 load
if load_path is None:
    # tree가 없다면 생성 후 저장
    RA = RetrievalAugmentation()
    # tree 생성
    RA.add_documents()
    # tree 저장
    save_path = "../data/tree"
    RA.save(save_path)

# Create a config with your custom models
custom_config = RetrievalAugmentationConfig(
    summarization_model=custom_summarizer,
    qa_model=custom_qa,
    embedding_model=custom_embedding
)

# Initialize RAPTOR with your custom config
RA = RetrievalAugmentation(config=custom_config, tree = load_path)

question = "다른 사람이 카페에 놓아둔 지갑을 가져갔다. 해당하는 형법 법조항은?"
answer = RA.answer_question(question)
print('answer : \n', answer)
