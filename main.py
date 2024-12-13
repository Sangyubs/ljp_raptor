'''
사용법 : 
이미 tree가 있다면 load_path를 지정해주고, 
없다면 add_documents()를 통해 문서를 추가하고 save_path를 지정해준다.
그 후, answer_question을 통해 질문에 대한 답을 얻을 수 있다.
'''
# from json import load
from retrieveraugmentation import RetrievalAugmentation

load_path = None

if load_path is None:
    # tree가 없다면 생성 후 저장
    RA = RetrievalAugmentation()
    # tree 생성
    RA.add_documents()
    # tree 저장
    save_path = "./data/tree"
    RA.save(save_path)
else:
    # tree가 있다면 로드
    RA = RetrievalAugmentation(tree = load_path)
 
question = "다른 사람이 카페에 놓아둔 지갑을 가져갔다. 해당하는 형법 법조항은?"
answer = RA.answer_question(question)
