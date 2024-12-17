'''
사용법 : 
tree_path, custom_qa, custom_summarizer, custom_embedding를 설정
tree_path : None이라면 생성 
custom_qa : None 이라면 gpt4-turbo
custom_summarizer : None이라면 gpt4-turbo
custom_embedding : None 이면 text-embedding-ada-002
embedding는 tree build, retrive에 모두 사용되므로 해당 embedding 모델 tree가 있는지 확인후 
'''

# import
from qamodels import GPT4oModel
from retrieveraugmentation import RetrievalAugmentation, RetrievalAugmentationConfig

# config 설정 
tree_path = '../data/20241217_OpenAItree' # or None
custom_qa = GPT4oModel() # or None
custom_summarizer = None # or model class
custom_embedding = None # or model class

custom_config = RetrievalAugmentationConfig(
    summarization_model=custom_summarizer,
    qa_model=custom_qa,
    embedding_model=custom_embedding
)

# tree build or load : tree_path = None 이면 custom_config에 따라 build 후 load
if tree_path is None:
    # tree가 없다면 생성 후 저장
    RA = RetrievalAugmentation(config=custom_config)
    # tree 생성
    RA.add_documents()
    # tree 저장
    save_path = "../data/tree"
    RA.save(save_path)

# Initialize RAPTOR with your custom config
RA = RetrievalAugmentation(config=custom_config, tree = tree_path)

question = "다른 사람이 카페에 놓아둔 지갑을 가져갔다. 해당하는 형법 법조항은?"
answer = RA.answer_question(question=question, collapse_tree = False)
# def answer_question(
#     self,
#     question,
#     top_k: int = 10,
#     start_layer: int = None,
#     num_layers: int = None,
#     max_tokens: int = 3500,
#     collapse_tree: bool = True,
#     return_layer_information: bool = True,
# ):
print('answer : \n', answer)
