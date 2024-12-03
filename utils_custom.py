import logging
from typing import Dict, List, Set
from scipy import spatial
import numpy as np
import json
import pandas as pd
from structure_tree import Node
import re

# text -> chunks로 변경 : 나는 법령 전체를 받아서 chunks로 나누는 함수 수정 필요
def split_text(file_path: str = r"..\data\criminal_law_specific.txt" ):
    '''
    형법각칙의 파일을 읽어와서 조문별로 나누는 함수. 
    조문별 구분방법
    -\n\n으로 조문split
    - 조문번호 :'제'로 시작해 '('로 끝남, '('는 불포함
    - 조문 이름 : '('로 시작해 ')'로 끝남, 
    - 조문 내용 : 조문번호, 조문 이름 이후 부분 모두
        - <>로 묶인 부분은 제외
        -[]로 묶인 부분은 제외하되 안에 '헌법'이 들어간 경우는 제외하지 않음
    - 횡령, 배임은 355조에 각각 있는 관계로 355조1항(횡령), 355조2항(배임)으로 나눔
    
    형법조문 분류 csv와 매칭시키는 방법은 
    1. -, _가 없다면 f'{조문숫자}조'로 표현
    2. '-'가 있다면 '-' split -> f'제{split[0]}조의{split[1]}'로 표현
    3. '_'가 있다면 '_' split -> f'제{split[0]}조제{split[1]}항'로 표현
    
    Args:
        text_path (str): The path of the text file to be split.
    Returns:
        json: A list of dict(조문). -> 저장
        csv로 저장
    '''
    # 출력 JSON 파일 경로
    output_json_path = r'..\data\parsed_laws.json'

    # CSV 저장 경로
    output_csv_path = r'..\data\parsed_laws.csv'

    # 파일 읽기
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # 조문별로 분리
    provisions = content.split('\n\n')
    
    # 결과를 저장할 리스트
    parsed_laws = []
    
    for provision in provisions:
               
        # 조문번호 추출
        num_match = re.search(r'^제.+?\(', provision)
        if num_match:
            article_id = num_match.group().rstrip('(').strip()
        
        # 조문 이름 추출
        name_match = re.search(r'\(.+?\)', provision)
        if name_match:
            title = name_match.group().strip('()')
        
        # 조문 내용 추출
        content_start = name_match.end() if name_match else len(num_match.group()) if num_match else 0
        content = provision[content_start:].strip()
        
        # <> 및 [] 제거 (단, '헌법'이 포함된 []는 유지)
        content = re.sub(r'<.*?>', '', content)
        content = re.sub(r'\[(?!.*?헌법).*?\]', '', content)
        content = content.strip()
        
        # '삭제'가 포함된 경우 pass
        if '삭제' in content:
            continue  # 해당 조문을 건너뜀
        
        # raw_text 생성 (제목 + 본문)
        raw_text = f"{title} : {content}"
        
        # character_count 계산
        character_count = len(raw_text)
        
        # 파싱된 데이터를 딕셔너리 형태로 저장
        parsed_laws.append({
            "article_id": article_id,
            "title": title,
            "content": content,
            "raw_text": raw_text,
            "character_count": character_count
        })

    # JSON 파일로 저장
    with open(output_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(parsed_laws, json_file, ensure_ascii=False, indent=4)

    print(f"JSON 파일 저장 완료: {output_json_path}")

    # JSON 데이터를 DataFrame으로 변환
    df = pd.DataFrame(parsed_laws)

    # CSV 파일로 저장
    df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"CSV 파일 저장 완료: {output_csv_path}")

    # CSV 내용 확인
    print("\nCSV 데이터 일부 미리보기:")
    print(df.head())

# def split_text(
#     text: str, tokenizer: tiktoken.get_encoding("cl100k_base"), max_tokens: int, overlap: int = 0
# ):
#     """
#     Splits the input text into smaller chunks based on the tokenizer and maximum allowed tokens.
    
#     Args:
#         text (str): The text to be split.
#         tokenizer (CustomTokenizer): The tokenizer to be used for splitting the text.
#         max_tokens (int): The maximum allowed tokens.
#         overlap (int, optional): The number of overlapping tokens between chunks. Defaults to 0.
    
#     Returns:
#         List[str]: A list of text chunks.
#     """
#     # Split the text into sentences using multiple delimiters
#     delimiters = [".", "!", "?", "\n"]
#     regex_pattern = "|".join(map(re.escape, delimiters))
#     sentences = re.split(regex_pattern, text)
    
#     # Calculate the number of tokens for each sentence
#     n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]
    
#     chunks = []
#     current_chunk = []
#     current_length = 0
    
#     for sentence, token_count in zip(sentences, n_tokens):
#         # If the sentence is empty or consists only of whitespace, skip it
#         if not sentence.strip():
#             continue
        
#         # If the sentence is too long, split it into smaller parts
#         if token_count > max_tokens:
#             sub_sentences = re.split(r"[,;:]", sentence)
            
#             # there is no need to keep empty os only-spaced strings
#             # since spaces will be inserted in the beginning of the full string
#             # and in between the string in the sub_chuk list
#             filtered_sub_sentences = [sub.strip() for sub in sub_sentences if sub.strip() != ""]
#             sub_token_counts = [len(tokenizer.encode(" " + sub_sentence)) for sub_sentence in filtered_sub_sentences]
            
#             sub_chunk = []
#             sub_length = 0
            
#             for sub_sentence, sub_token_count in zip(filtered_sub_sentences, sub_token_counts):
#                 if sub_length + sub_token_count > max_tokens:
                    
#                     # if the phrase does not have sub_sentences, it would create an empty chunk
#                     # this big phrase would be added anyways in the next chunk append
#                     if sub_chunk:
#                         chunks.append(" ".join(sub_chunk))
#                         sub_chunk = sub_chunk[-overlap:] if overlap > 0 else []
#                         sub_length = sum(sub_token_counts[max(0, len(sub_chunk) - overlap):len(sub_chunk)])
                
#                 sub_chunk.append(sub_sentence)
#                 sub_length += sub_token_count
            
#             if sub_chunk:
#                 chunks.append(" ".join(sub_chunk))
        
#         # If adding the sentence to the current chunk exceeds the max tokens, start a new chunk
#         elif current_length + token_count > max_tokens:
#             chunks.append(" ".join(current_chunk))
#             current_chunk = current_chunk[-overlap:] if overlap > 0 else []
#             current_length = sum(n_tokens[max(0, len(current_chunk) - overlap):len(current_chunk)])
#             current_chunk.append(sentence)
#             current_length += token_count
        
#         # Otherwise, add the sentence to the current chunk
#         else:
#             current_chunk.append(sentence)
#             current_length += token_count
    
#     # Add the last chunk if it's not empty
#     if current_chunk:
#         chunks.append(" ".join(current_chunk))
    
#     return chunks



def distances_from_embeddings(
    query_embedding: List[float],
    embeddings: List[List[float]],
    distance_metric: str = "cosine",
) -> List[float]:
    """
    Calculates the distances between a query embedding and a list of embeddings.

    Args:
        query_embedding (List[float]): The query embedding.
        embeddings (List[List[float]]): A list of embeddings to compare against the query embedding.
        distance_metric (str, optional): The distance metric to use for calculation. Defaults to 'cosine'.

    Returns:
        List[float]: The calculated distances between the query embedding and the list of embeddings.
    """
    distance_metrics = {
        "cosine": spatial.distance.cosine,
        "L1": spatial.distance.cityblock,
        "L2": spatial.distance.euclidean,
        "Linf": spatial.distance.chebyshev,
    }

    if distance_metric not in distance_metrics:
        raise ValueError(
            f"Unsupported distance metric '{distance_metric}'. Supported metrics are: {list(distance_metrics.keys())}"
        )

    distances = [
        distance_metrics[distance_metric](query_embedding, embedding)
        for embedding in embeddings
    ]

    return distances


def get_embeddings(node_list: List[Node], embedding_model: str) -> List:
    """
    Extracts the embeddings of nodes from a list of nodes.

    Args:
        node_list (List[Node]): List of nodes.
        embedding_model (str): The name of the embedding model to be used.

    Returns:
        List: List of node embeddings.
    """
    return [node.embeddings[embedding_model] for node in node_list]



def indices_of_nearest_neighbors_from_distances(distances: List[float]) -> np.ndarray:
    """
    Returns the indices of nearest neighbors sorted in ascending order of distance.

    Args:
        distances (List[float]): A list of distances between embeddings.

    Returns:
        np.ndarray: An array of indices sorted by ascending distance.
    """
    return np.argsort(distances)