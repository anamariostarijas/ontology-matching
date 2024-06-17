import rdflib
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, OWL, XSD
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ontology_matching.src.helpers import Calculator, Processor

class MatchingTechnique:
    """Uses different simple ontology matching techniques and produces matchen between entities from left and right ontology.
    """

    def __init__(self) -> None:
        pass
    
    @staticmethod
    def match_with_levenshtein(left_info: dict, right_info: dict, threshold: float = 90) -> dict:
        """Takes lists of labels of entities as an input and returns best match for each label from the left ontology, if the similarity is larger than threshold.

        Args:
            left_info (dict): labels of entities from the left ontology
            right_info (dict): labels of entities from the right ontology
            threshold (int, optional): The minimal similarity score acceptable. Defaults to 90.

        Returns:
            dict: {left label: [matching right label, score]}
        """
        matches = {}
        for left_uri, left_label in left_info.items():
            match, score = process.extractOne(left_label, list(right_info.values()), scorer=fuzz.token_sort_ratio)
            if score >= threshold:
                match_uri = [s for s,v in right_info.items() if v == match][0]
                matches[left_uri] = [match_uri, score]
        return matches
    
    @staticmethod
    def match_with_n_gram(left_info: dict, right_info: dict, threshold: float = 0.8, n: int = 3) -> dict:
        """Matches labels based on n-gram similarity.

        Args:
            left_info (dict): labels and uris of entities from the left ontology
            right_info (dict): labels and uris of entities from the right ontology
            threshold (float): The minimal similarity score acceptable. Defaults to 90.

        Returns:
            dict: {left label: [matching right label, score]}
        """
        # Ensure nltk data is downloaded
        nltk.download('punkt')
        matches = {}
        for left_uri, left_label in left_info.items():
            for right_uri, right_label in right_info.items():
                similarity = Calculator.ngram_similarity(left_label, right_label, n) * 100
                if left_uri not in matches and similarity >= threshold:
                    matches[left_uri] = [right_uri, similarity]
                if left_uri in matches and similarity >= matches[left_uri][1]:
                    matches[left_uri] = [right_uri, similarity]
        return matches

    @staticmethod
    def match_with_cosine(left_info: dict, right_info: dict, threshold: float = 0.9) -> dict:
        """Matches labels based on cosine similarity.

            left_info (dict): labels and uris of entities from the left ontology
            right_info (dict): labels and uris of entities from the right ontology
            threshold (float): The minimal similarity score acceptable. Defaults to 90.

        Returns:
            dict: {left label: [matching right label, score]}
        """
        # Combine texts for vectorization
        all_texts = list(left_info.values()) + list(right_info.values())
        # Compute vectors using CountVectorizer
        vectorizer = CountVectorizer().fit(all_texts)
        vector_matrix = vectorizer.transform(all_texts)
        # Split vector matrix into two parts
        vector_matrix1 = vector_matrix[:len(left_info.values())]
        vector_matrix2 = vector_matrix[len(left_info.values()):]
        # Compute cosine similarity
        similarity_matrix = cosine_similarity(vector_matrix1, vector_matrix2)

        # Create correspondences based on similarity scores
        matches = {}
        for i, uri1 in enumerate(left_info.keys()):
            for j, uri2 in enumerate(right_info.keys()):
                confidence = similarity_matrix[i, j] * 100 
                if uri1 not in matches and confidence >= threshold:
                    matches[uri1] = [uri2, confidence]
                if uri1 in matches and confidence >= matches[uri1][1]:
                    matches[uri1] = [uri2, confidence]
        return matches
    
    @staticmethod
    def match_with_path(left_graph: Graph, right_graph: Graph, threshold: float = 0.9, l: float = 0.7) -> dict:
        """Matches labels based on the paths (class hierarchies).

        Args:
            left_graph (Graph): left ontology graph
            right_graph (Graph): right ontology graph
            threshold (float, optional): The minimal similarity score acceptable. Defaults to 0.9.
            l (float, optional): Lambda factor. Defaults to 0.7.

        Returns:
            dict: {left label: [matching right label, score]}
        """
        left_paths = Processor.find_paths(left_graph)
        right_paths = Processor.find_paths(right_graph)
        matches = {}
        for left_uri, left_label in left_paths.items():
            for right_uri, right_label in right_paths.items():
                similarity = 1 - Calculator.path_distance(left_label, right_label, l)
                if left_uri not in matches and similarity >= threshold:
                    matches[left_uri] = [right_uri, similarity]
                if left_uri in matches and similarity >= matches[left_uri][1]:
                    matches[left_uri] = [right_uri, similarity]
        return matches
        