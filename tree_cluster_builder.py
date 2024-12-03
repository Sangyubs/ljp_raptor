import logging

logging.basicConfig(foramt = "%(asctime)s - %(message)s", level=logging.INFO)

from .build_tree import TreeBuilder, TreeBuilderConfig

class ClusterTreeConfig(TreeBuilderConfig):
    