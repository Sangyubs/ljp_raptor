from retrieveraugmentation import RetrievalAugmentation

RA = RetrievalAugmentation()
save_path = "./data/tree"
RA.add_documents()
question = "다른 사람이 카페에 놓아둔 지갑을 가져갔다. 해당하는 형법 법조항은?"
answer = RA.answer_question(question)
