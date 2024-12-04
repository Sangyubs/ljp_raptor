import tiktoken
from threading import Lock
from typing import Dict, List, Optional, Set, Tuple
import logging, copy
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

from .summarizationmodels import BaseSummarizationModel, GPTSummarizationModel
from .embeddingmodels import BaseEmbeddingModel, OpenAIEmbeddingModel
from .structure_tree import Node, Tree
from .utils_custom import (distances_from_embeddings, get_embeddings, 
                           indices_of_nearest_neighbors_from_distances, 
                           split_text, add_law_categories, get_text)
class TreeBuilderConfig:
    def __init__(
        self,
        tokenizer=None,
        max_tokens=None,
        num_layers=None,
        threshold=None,
        top_k=None,
        selection_mode=None,
        summarization_length=None,
        summarization_model=None,
        embedding_models=None,
        cluster_embedding_model=None,
    ):

# 원 파일과 달리 max_tokens(chunk의 max_tokens 수)를 None으로 설정
        # if max_tokens is None:
        #     max_tokens = 100
        # if not isinstance(max_tokens, int) or max_tokens < 1:
        #     raise ValueError("max_tokens must be an integer and at least 1")
        self.max_tokens = max_tokens

        if num_layers is None:
            num_layers = 3
        if not isinstance(num_layers, int) or num_layers < 1:
            raise ValueError("num_layers must be an integer and at least 1")
        self.num_layers = num_layers

        if threshold is None:
            threshold = 0.5
        if not isinstance(threshold, (int, float)) or not (0 <= threshold <= 1):
            raise ValueError("threshold must be a number between 0 and 1")
        self.threshold = threshold

        if top_k is None:
            top_k = 5
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be an integer and at least 1")
        self.top_k = top_k

        if selection_mode is None:
            selection_mode = "top_k"
        if selection_mode not in ["top_k", "threshold"]:
            raise ValueError("selection_mode must be either 'top_k' or 'threshold'")
        self.selection_mode = selection_mode

        if summarization_length is None:
            summarization_length = 300
        self.summarization_length = summarization_length

        if summarization_model is None:
            summarization_model = GPTSummarizationModel()
        if not isinstance(summarization_model, BaseSummarizationModel):
            raise ValueError(
                "summarization_model must be an instance of BaseSummarizationModel"
            )
        self.summarization_model = summarization_model

        if embedding_models is None:
            embedding_models = {"OpenAI": OpenAIEmbeddingModel()}
        if not isinstance(embedding_models, dict):
            raise ValueError(
                "embedding_models must be a dictionary of model_name: instance pairs"
            )
        for model in embedding_models.values():
            if not isinstance(model, BaseEmbeddingModel):
                raise ValueError(
                    "All embedding models must be an instance of BaseEmbeddingModel"
                )
        self.embedding_models = embedding_models

        if tokenizer is None:
            tokenizer = tiktoken.get_encoding("cl100k_base")
            tokenizer = tiktoken.encoding_for_model(self.embedding_models['OpenAI'].model)
        self.tokenizer = tokenizer
        
        if cluster_embedding_model is None:
            cluster_embedding_model = "OpenAI"
        if cluster_embedding_model not in self.embedding_models:
            raise ValueError(
                "cluster_embedding_model must be a key in the embedding_models dictionary"
            )
        self.cluster_embedding_model = cluster_embedding_model

    def log_config(self):
        config_log = """
        TreeBuilderConfig:
            Tokenizer: {tokenizer}
            Max Tokens: {max_tokens}
            Num Layers: {num_layers}
            Threshold: {threshold}
            Top K: {top_k}
            Selection Mode: {selection_mode}
            Summarization Length: {summarization_length}
            Summarization Model: {summarization_model}
            Embedding Models: {embedding_models}
            Cluster Embedding Model: {cluster_embedding_model}
        """.format(
            tokenizer=self.tokenizer,
            max_tokens=self.max_tokens,
            num_layers=self.num_layers,
            threshold=self.threshold,
            top_k=self.top_k,
            selection_mode=self.selection_mode,
            summarization_length=self.summarization_length,
            summarization_model=self.summarization_model,
            embedding_models=self.embedding_models,
            cluster_embedding_model=self.cluster_embedding_model,
        )
        return config_log




class TreeBuilder:
    '''
    The TreeBuilder class is responsible for building a hierarchical text abstraction
    structure, known as a "tree," using summarization models and
    embedding models.
    leaf node로만 이루어진 tree build
    전체 트리는 tree_cluster_builder.py의 ClusterTreeBuilder에서 구현
    '''

    def __init__(self, config) -> None:
        """Initializes the tokenizer, maximum tokens, number of layers, top-k value, threshold, and selection mode."""

        self.tokenizer = config.tokenizer
        self.max_tokens = config.max_tokens
        self.num_layers = config.num_layers
        self.top_k = config.top_k
        self.threshold = config.threshold
        self.selection_mode = config.selection_mode
        self.summarization_length = config.summarization_length
        self.summarization_model = config.summarization_model
        self.embedding_models = config.embedding_models
        self.cluster_embedding_model = config.cluster_embedding_model

        logging.info(
            f"Successfully initialized TreeBuilder with Config {config.log_config()}"
        )
    lock = Lock()
    # text, index -> (index, Node)    
    def create_node(
            self, index: int, text: str, children_indices: Optional[Set[int]] = None, title: str = None
        ) -> Tuple[int, Node]:
            """Creates a new node with the given index, text, and (optionally) children indices.

            Args:
                index (int): The index of the new node.
                text (str): The text associated with the new node.
                children_indices (Optional[Set[int]]): A set of indices representing the children of the new node.
                    If not provided, an empty set will be used.

            Returns:
                Tuple[int, Node]: A tuple containing the index and the newly created node.
            """
            if children_indices is None:
                children_indices = set()

            embeddings = {
                model_name: model.create_embedding(text)
                for model_name, model in self.embedding_models.items()
            }
            return (index, Node(text, index, children_indices, embeddings, title))

    # text -> embedding(cluster_embedding_model)
    def create_embedding(self, text) -> List[float]:
            """
            Generates embeddings for the given text using the specified embedding model.

            Args:
                text (str): The text for which to generate embeddings.

            Returns:
                List[float]: The generated embeddings.
            """
            return self.embedding_models[self.cluster_embedding_model].create_embedding(
                text
            )
    
    # context -> summary(summarization_model)
    # 원 파일과 달리 max_tokens를 150->200으로 수정
    def summarize(self, context, max_tokens=200) -> str:
        """
        Generates a summary of the input context using the specified summarization model.

        Args:
            context (str, optional): The context to summarize.
            max_tokens (int, optional): The maximum number of tokens in the generated summary. Defaults to 150.o

        Returns:
            str: The generated summary.
        """
        return self.summarization_model.summarize(context, max_tokens)
    
    
    # 원본 : current_node 기준 관련되는 node retrieval
    # 수정 : 해당 분류 컬럼을 기준으로 text를 모두 가져온 뒤, 
    # 분류컬럼의 기준 : fulltext인 dict의 list를 반환
    def get_relevant_fulltext(self, file_path, search_column, text_column) -> List[Dict[str, str]]:
        """
        Reads a CSV file and concatenates text from specified columns based on a grouping column.

        Args:
            filepath (str): The path to the CSV file.
            search_column (str): The column name to group by.
            text_column (str): The column name containing text to concatenate.

        Returns:
            List[Dict[str, str]]: A list of dictionaries with the grouped column and concatenated text.
        """
        df = pd.read_csv(file_path = file_path)
        result = (
            df.groupby(search_column)[text_column]
            .apply(lambda texts: " ".join(texts.dropna()))
            .reset_index()
            .rename(columns={text_column : 'concatenated_text'})
        )
        relevant_fulltext_list = result.to_dict(orient='records')
        return relevant_fulltext_list

    def multithreaded_create_leaf_nodes(self, chunks: List[str]) -> Dict[int, Node]:
        """Creates leaf nodes using multithreading from the given list of text chunks.

        Args:
            chunks (List[str]): A list of text chunks to be turned into leaf nodes.

        Returns:
            Dict[int, Node]: A dictionary mapping node indices to the corresponding leaf nodes.
        """
        with ThreadPoolExecutor() as executor:
            future_nodes = {
                executor.submit(self.create_node, index, text): (index, text)
                for index, text in enumerate(chunks)
            }

            leaf_nodes = {}
            for future in as_completed(future_nodes):
                index, node = future.result()
                leaf_nodes[index] = node

        return leaf_nodes
 
    # text -> Tree / utils의 split_text 수정 필요
    def build_from_text(self, lock=lock, use_multithreading: bool = True) -> Tree:
        """Builds a golden tree from the input text, optionally using multithreading.

        Args:
            text (str): The input text.
            use_multithreading (bool, optional): Whether to use multithreading when creating leaf nodes.
                Default: True.

        Returns:
            Tree: The golden tree structure.
        """
        # Building All Leaf Nodes 
        chunks = add_law_categories() # law_df['소분류'].tolist(), law_df['full_text'].tolist()

        logging.info("Creating Leaf Nodes")

        if use_multithreading:
            leaf_nodes = self.multithreaded_create_leaf_nodes(chunks)
        else:
            leaf_nodes = {}
            for index, chunk in enumerate(zip(*chunks)):
                __, node = self.create_node(index = index, 
                                            text = chunk[1], title = chunk[0])
                leaf_nodes[index] = node

        layer_to_nodes = {0: list(leaf_nodes.values())}

        logging.info(f"Created {len(leaf_nodes)} Leaf Embeddings")

        logging.info("Building 2nd level Nodes")

        all_nodes = copy.deepcopy(leaf_nodes)
        
        # Building 2nd layer tree
        file_path = r"..\data\final_law.csv"
        relevant_fulltext_list = self.get_relevant_fulltext(file_path, '소분류', 'full_text')
        next_node_index = len(all_nodes)
        second_level_nodes = {}
        # 소분류마다 노드 생성
        for fulltext in relevant_fulltext_list:
            text = fulltext['concatenated_text']
            title = fulltext['소분류'],
            summarized_text = self.summarize(
            context=text,
            max_tokens=self.summarization_length,
            )
            indices_with_title = [node.index for node in leaf_nodes if node.title==title]
            
            next_node_index, new_parent_node = self.create_node(next_node_index, summarized_text, indices_with_title, title)
            all_nodes[next_node_index] = new_parent_node
                    
            with lock:
                second_level_nodes[next_node_index] = new_parent_node
            next_node_index += 1
        layer_to_nodes.update({1: list(second_level_nodes.values())})
        
        # Building 3rd layer tree   
        # 중분류 기준 소분류 모음
        df = pd.read_csv(file_path)                
        result = (
            df.groupby('중분류')['소분류']
            .apply(lambda texts: list(set(texts.dropna()))) # 텍스트들을 리스트로 변환
            .reset_index()
            .rename(columns={'소분류': 'text_list'})   # 결과 컬럼 이름 변경
        )
        relevant_fulltext_list = result.to_dict(orient='records') 
        
        third_level_nodes = {}  
        relevant_node_list = defaultdict(list)
        next_node_index = len(all_nodes)
        for key, values in relevant_fulltext_list:
            for node in layer_to_nodes[1]:
                if node.title in values:
                    relevant_node_list[key].append(node)
                else:
                    raise ValueError("2nd node title이 3rd node text_list에 없음")
            text = " ".join([node.text for node in relevant_node_list[key]])
            title = key
            summarized_text = self.summarize(
            context=text,
            max_tokens=self.summarization_length,
            )
            indices_with_title = [node.index for node in relevant_node_list[key]]
            next_node_index, new_parent_node = self.create_node(next_node_index, summarized_text, indices_with_title, title)
            all_nodes[next_node_index] = new_parent_node
            with lock:
                third_level_nodes[next_node_index] = new_parent_node
            next_node_index += 1
        layer_to_nodes.update({2: list(third_level_nodes.values())})
                
        root_nodes = copy.deepcopy(third_level_nodes)
        
        tree = Tree(all_nodes, root_nodes, leaf_nodes, self.num_layers, layer_to_nodes)

        return tree
    
    # # 아래는 원본에 있는 내용이나 필ㅇ요 없을 것 같아서 주석처리 
    # def construct_tree(
    #     self,
    #     current_level_nodes: Dict[int, Node],
    #     all_tree_nodes: Dict[int, Node],
    #     layer_to_nodes: Dict[int, List[Node]],
    #     use_multithreading: bool = True,
    # ) -> Dict[int, Node]:
    #     """
    #     Constructs the hierarchical tree structure layer by layer by iteratively summarizing groups
    #     of relevant nodes and updating the current_level_nodes and all_tree_nodes dictionaries at each step.

    #     Args:
    #         current_level_nodes (Dict[int, Node]): The current set of nodes.
    #         all_tree_nodes (Dict[int, Node]): The dictionary of all nodes.
    #         use_multithreading (bool): Whether to use multithreading to speed up the process.

    #     Returns:
    #         Dict[int, Node]: The final set of root nodes.
    #     """
    #     pass

    #     logging.info("Using Transformer-like TreeBuilder")

    #     def process_node(idx, current_level_nodes, new_level_nodes, all_tree_nodes, next_node_index, lock):
    #         relevant_nodes_chunk = self.get_relevant_nodes(
    #             current_level_nodes[idx], current_level_nodes
    #         )

    #         node_texts = get_text(relevant_nodes_chunk)

    #         summarized_text = self.summarize(
    #             context=node_texts,
    #             max_tokens=self.summarization_length,
    #         )

    #         logging.info(
    #             f"Node Texts Length: {len(self.tokenizer.encode(node_texts))}, Summarized Text Length: {len(self.tokenizer.encode(summarized_text))}"
    #         )

    #         next_node_index, new_parent_node = self.create_node(
    #             next_node_index,
    #             summarized_text,
    #             {node.index for node in relevant_nodes_chunk}
    #         )

    #         with lock:
    #             new_level_nodes[next_node_index] = new_parent_node

    #     for layer in range(self.num_layers):
    #         logging.info(f"Constructing Layer {layer}: ")

    #         node_list_current_layer = get_node_list(current_level_nodes)
    #         next_node_index = len(all_tree_nodes)

    #         new_level_nodes = {}
    #         lock = Lock()

    #         if use_multithreading:
    #             with ThreadPoolExecutor() as executor:
    #                 for idx in range(0, len(node_list_current_layer)):
    #                     executor.submit(process_node, idx, node_list_current_layer, new_level_nodes, all_tree_nodes, next_node_index, lock)
    #                     next_node_index += 1
    #                 executor.shutdown(wait=True)
    #         else:
    #             for idx in range(0, len(node_list_current_layer)):
    #                 process_node(idx, node_list_current_layer, new_level_nodes, all_tree_nodes, next_node_index, lock)

    #         layer_to_nodes[layer + 1] = list(new_level_nodes.values())
    #         current_level_nodes = new_level_nodes
    #         all_tree_nodes.update(new_level_nodes)

    #     return new_level_nodes


