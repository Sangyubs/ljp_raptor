from retrieveraugmentation import RetrievalAugmentation

RA = RetrievalAugmentation()
RA.add_documents()
question = "다른 사람이 카페에 놓아둔 지갑을 가져갔다."
answer = RA.answer_question(question)