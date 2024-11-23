# Import required libraries
import logging
import pickle

# Import custom modules for tree building, embedding, QA, and summarization functionality
from .cluster_tree_builder import ClusterTreeBuilder, ClusterTreeConfig
from .EmbeddingModels import BaseEmbeddingModel
from .QAModels import BaseQAModel, GPT3TurboQAModel
from .SummarizationModels import BaseSummarizationModel
from .tree_builder import TreeBuilder, TreeBuilderConfig
from .tree_retriever import TreeRetriever, TreeRetrieverConfig
from .tree_structures import Node, Tree

# Dictionary mapping supported tree builder types to their respective builder classes and config classes
supported_tree_builders = {"cluster": (ClusterTreeBuilder, ClusterTreeConfig)}

# Configure basic logging with timestamp and INFO level
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class RetrievalAugmentationConfig:
    """
    Configuration class for RetrievalAugmentation.
    Handles settings for tree building, retrieval, QA model, embedding model, and summarization.
    """
    def __init__(
        self,
        tree_builder_config=None,      # Custom configuration for tree builder
        tree_retriever_config=None,    # Custom configuration for tree retriever
        qa_model=None,                 # Model for question answering
        embedding_model=None,          # Model for text embeddings
        summarization_model=None,      # Model for text summarization
        tree_builder_type="cluster",   # Type of tree builder to use
        # TreeRetriever specific parameters
        tr_tokenizer=None,            # Tokenizer for tree retriever
        tr_threshold=0.5,             # Similarity threshold for retrieval
        tr_top_k=5,                   # Number of top results to retrieve
        tr_selection_mode="top_k",    # Mode for selecting results
        tr_context_embedding_model="OpenAI",  # Embedding model for context
        tr_embedding_model=None,       # Custom embedding model for retriever
        tr_num_layers=None,           # Number of layers to traverse
        tr_start_layer=None,          # Starting layer for traversal
        # TreeBuilder specific parameters
        tb_tokenizer=None,            # Tokenizer for tree builder
        tb_max_tokens=100,            # Maximum tokens per node
        tb_num_layers=5,              # Number of layers in tree
        tb_threshold=0.5,             # Similarity threshold for clustering
        tb_top_k=5,                   # Number of top clusters
        tb_selection_mode="top_k",    # Mode for cluster selection
        tb_summarization_length=100,  # Maximum length of summaries
        tb_summarization_model=None,  # Model for node summarization
        tb_embedding_models=None,     # Dictionary of embedding models
        tb_cluster_embedding_model="OpenAI",  # Model for cluster embeddings
    ):
        # Validate that the specified tree builder type is supported
        if tree_builder_type not in supported_tree_builders:
            raise ValueError(
                f"tree_builder_type must be one of {list(supported_tree_builders.keys())}"
            )

        # Ensure QA model is of correct type if provided
        if qa_model is not None and not isinstance(qa_model, BaseQAModel):
            raise ValueError("qa_model must be an instance of BaseQAModel")

        # Validate embedding model and configure related settings
        if embedding_model is not None and not isinstance(
            embedding_model, BaseEmbeddingModel
        ):
            raise ValueError(
                "embedding_model must be an instance of BaseEmbeddingModel"
            )
        elif embedding_model is not None:
            # Prevent duplicate embedding model specifications
            if tb_embedding_models is not None:
                raise ValueError(
                    "Only one of 'tb_embedding_models' or 'embedding_model' should be provided, not both."
                )
            # Configure embedding models for both tree builder and retriever
            tb_embedding_models = {"EMB": embedding_model}
            tr_embedding_model = embedding_model
            tb_cluster_embedding_model = "EMB"
            tr_context_embedding_model = "EMB"

        # Validate summarization model and configure related settings
        if summarization_model is not None and not isinstance(
            summarization_model, BaseSummarizationModel
        ):
            raise ValueError(
                "summarization_model must be an instance of BaseSummarizationModel"
            )
        elif summarization_model is not None:
            # Prevent duplicate summarization model specifications
            if tb_summarization_model is not None:
                raise ValueError(
                    "Only one of 'tb_summarization_model' or 'summarization_model' should be provided, not both."
                )
            tb_summarization_model = summarization_model

        # Initialize TreeBuilder configuration
        tree_builder_class, tree_builder_config_class = supported_tree_builders[
            tree_builder_type
        ]
        if tree_builder_config is None:
            tree_builder_config = tree_builder_config_class(
                tokenizer=tb_tokenizer,
                max_tokens=tb_max_tokens,
                num_layers=tb_num_layers,
                threshold=tb_threshold,
                top_k=tb_top_k,
                selection_mode=tb_selection_mode,
                summarization_length=tb_summarization_length,
                summarization_model=tb_summarization_model,
                embedding_models=tb_embedding_models,
                cluster_embedding_model=tb_cluster_embedding_model,
            )
        elif not isinstance(tree_builder_config, tree_builder_config_class):
            raise ValueError(
                f"tree_builder_config must be a direct instance of {tree_builder_config_class} for tree_builder_type '{tree_builder_type}'"
            )

        # Initialize TreeRetriever configuration
        if tree_retriever_config is None:
            tree_retriever_config = TreeRetrieverConfig(
                tokenizer=tr_tokenizer,
                threshold=tr_threshold,
                top_k=tr_top_k,
                selection_mode=tr_selection_mode,
                context_embedding_model=tr_context_embedding_model,
                embedding_model=tr_embedding_model,
                num_layers=tr_num_layers,
                start_layer=tr_start_layer,
            )
        elif not isinstance(tree_retriever_config, TreeRetrieverConfig):
            raise ValueError(
                "tree_retriever_config must be an instance of TreeRetrieverConfig"
            )

        # Store configurations as instance attributes
        self.tree_builder_config = tree_builder_config
        self.tree_retriever_config = tree_retriever_config
        self.qa_model = qa_model or GPT3TurboQAModel()  # Use default QA model if none provided
        self.tree_builder_type = tree_builder_type

    def log_config(self):
        """
        Generates a formatted string representation of the current configuration.
        """
        config_summary = """
        RetrievalAugmentationConfig:
            {tree_builder_config}
            
            {tree_retriever_config}
            
            QA Model: {qa_model}
            Tree Builder Type: {tree_builder_type}
        """.format(
            tree_builder_config=self.tree_builder_config.log_config(),
            tree_retriever_config=self.tree_retriever_config.log_config(),
            qa_model=self.qa_model,
            tree_builder_type=self.tree_builder_type,
        )
        return config_summary


class RetrievalAugmentation:
    """
    Main class for performing retrieval augmented generation.
    Combines tree building, information retrieval, and question answering capabilities.
    """

    def __init__(self, config=None, tree=None):
        """
        Initialize RetrievalAugmentation with configuration and optional existing tree.
        
        Args:
            config (RetrievalAugmentationConfig): Configuration object
            tree (Tree or str): Existing tree instance or path to pickled tree file
        """
        # Initialize with default config if none provided
        if config is None:
            config = RetrievalAugmentationConfig()
        if not isinstance(config, RetrievalAugmentationConfig):
            raise ValueError(
                "config must be an instance of RetrievalAugmentationConfig"
            )

        # Load tree from file if path provided
        if isinstance(tree, str):
            try:
                with open(tree, "rb") as file:
                    self.tree = pickle.load(file)
                if not isinstance(self.tree, Tree):
                    raise ValueError("The loaded object is not an instance of Tree")
            except Exception as e:
                raise ValueError(f"Failed to load tree from {tree}: {e}")
        elif isinstance(tree, Tree) or tree is None:
            self.tree = tree
        else:
            raise ValueError(
                "tree must be an instance of Tree, a path to a pickled Tree, or None"
            )

        # Initialize tree builder and retriever
        tree_builder_class = supported_tree_builders[config.tree_builder_type][0]
        self.tree_builder = tree_builder_class(config.tree_builder_config)
        
        self.tree_retriever_config = config.tree_retriever_config
        self.qa_model = config.qa_model

        # Initialize retriever if tree exists
        if self.tree is not None:
            self.retriever = TreeRetriever(self.tree_retriever_config, self.tree)
        else:
            self.retriever = None

        logging.info(
            f"Successfully initialized RetrievalAugmentation with Config {config.log_config()}"
        )

    def add_documents(self, docs):
        """
        Add new documents to create a tree structure.
        
        Args:
            docs (str): Text content to be added to the tree
        """
        # Confirm overwrite if tree already exists
        if self.tree is not None:
            user_input = input(
                "Warning: Overwriting existing tree. Did you mean to call 'add_to_existing' instead? (y/n): "
            )
            if user_input.lower() == "y":
                return

        # Build new tree and initialize retriever
        self.tree = self.tree_builder.build_from_text(text=docs)
        self.retriever = TreeRetriever(self.tree_retriever_config, self.tree)

    def retrieve(
        self,
        question,
        start_layer: int = None,
        num_layers: int = None,
        top_k: int = 10,
        max_tokens: int = 3500,
        collapse_tree: bool = True,
        return_layer_information: bool = True,
    ):
        """
        Retrieve relevant information from the tree based on a question.
        
        Args:
            question (str): Query to retrieve information for
            start_layer (int): Starting layer in tree
            num_layers (int): Number of layers to traverse
            top_k (int): Number of top results to retrieve
            max_tokens (int): Maximum tokens in retrieved context
            collapse_tree (bool): Whether to collapse tree structure
            return_layer_information (bool): Whether to return layer details
            
        Returns:
            Retrieved context and optionally layer information
        """
        if self.retriever is None:
            raise ValueError(
                "The TreeRetriever instance has not been initialized. Call 'add_documents' first."
            )

        return self.retriever.retrieve(
            question,
            start_layer,
            num_layers,
            top_k,
            max_tokens,
            collapse_tree,
            return_layer_information,
        )

    def answer_question(
        self,
        question,
        top_k: int = 10,
        start_layer: int = None,
        num_layers: int = None,
        max_tokens: int = 3500,
        collapse_tree: bool = True,
        return_layer_information: bool = False,
    ):
        """
        Answer a question using retrieved context.
        
        Args:
            question (str): Question to answer
            top_k (int): Number of top results to consider
            start_layer (int): Starting layer in tree
            num_layers (int): Number of layers to traverse
            max_tokens (int): Maximum tokens in context
            collapse_tree (bool): Whether to collapse tree structure
            return_layer_information (bool): Whether to return layer details
            
        Returns:
            Answer and optionally layer information
        """
        # Retrieve relevant context
        context, layer_information = self.retrieve(
            question, start_layer, num_layers, top_k, max_tokens, collapse_tree, True
        )

        # Generate answer using QA model
        answer = self.qa_model.answer_question(context, question)

        if return_layer_information:
            return answer, layer_information

        return answer

    def save(self, path):
        """
        Save the current tree structure to a file.
        
        Args:
            path (str): File path to save the tree
        """
        if self.tree is None:
            raise ValueError("There is no tree to save.")
        with open(path, "wb") as file:
            pickle.dump(self.tree, file)
        logging.info(f"Tree successfully saved to {path}")