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
    
# 파싱된 법률에 대해 최종 중분류, 대분류가 포함된 csv 파일, json을 작성
def add_law_categories(
    file_path = r'..\data\parsed_laws.csv',
    cls_path = r'..\data\형법 분류.csv'    
):
    '''
    법률조문에 대해 최종 중분류, 대분류가 포함된 csv 파일, json 작성
    Args:
        file_path (str): The path of the parsed law file.
        cls_path (str): The path of the law classification file.
    Returns:
        json: A list of dict -> 저장
        csv로 저장
    
    '''
    # 출력 JSON, csv 파일 경로
    output_json_path = r'..\data\final_law.json'
    output_csv_path = r'..\data\final_law.csv'
    
    # df load
    law_df = pd.read_csv(file_path)
    cls_df = pd.read_csv(cls_path)
    
    def clean_article_id(article_id):
        if pd.isna(article_id):  # Handle NaN values gracefully
            return article_id
        article_id = article_id.replace('조의', '-').replace('조제', '_')
        article_id = re.sub(r'[가-힣]', '', article_id)  # Remove all Korean characters
        return article_id

    def exact_match(article_id, law_string):
        # '조문'을 쉼표로 분리하고 공백을 제거한 후, 정확한 일치를 검사
        law_list = [x.strip() for x in law_string.split(',')]
        return article_id in law_list

    # Apply the function to the 'article_id' column and save
    law_df['article_raw_id'] = law_df['article_id'].apply(clean_article_id)
        
    # Iterate through all rows in law_df
    for index, row in law_df.iterrows():
        article_id = str(row['article_raw_id'])  # Ensure article_id is a string for comparison
        # Find matching rows in classification_df
        match = cls_df[cls_df['조문'].apply(lambda x: exact_match(article_id, x))]
        if not match.empty:
            # Add '중분류' and '소분류' to the corresponding row in law_df
            law_df.loc[index, '중분류'] = match.iloc[0]['중분류']
            law_df.loc[index, '소분류'] = match.iloc[0]['소분류']
        else:
            print(f"No match found for article_id: {article_id}")

    # Create a 'full_text' column with the specified format
    law_df['full_text'] = law_df.apply(
        lambda row: f"조항번호 : {row['article_id']}\n조문명 : {row['title']}\n조문내용 : {row['content']}", axis=1
    )

    # JSON 파일로 저장
    law_df.to_json(output_json_path, orient='records', force_ascii=False, indent=4)
    print(f"JSON 파일 저장 완료: {output_json_path}")

    # CSV 파일로 저장
    law_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"CSV 파일 저장 완료: {output_csv_path}")

    # CSV 내용 확인
    print("\nCSV 데이터 일부 미리보기:")
    print(law_df.head())
    
    # embedding에 사용할 수 있도록 full_text를 리스트로 반환
    return law_df['소분류'].tolist(), law_df['full_text'].tolist()


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


def get_text(node_list: List[Node]) -> str:
    """
    Generates a single text string by concatenating the text from a list of nodes.

    Args:
        node_list (List[Node]): List of nodes.

    Returns:
        str: Concatenated text.
    """
    text = ""
    for node in node_list:
        text += f"{' '.join(node.text.splitlines())}"
        text += "\n\n"
    return text