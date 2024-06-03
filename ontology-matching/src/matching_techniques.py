import rdflib
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, OWL, XSD
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.helpers import Calculator

class MatchingTechnique:
    """Uses different simple ontology matching techniques and produces matchen between entities from left and right ontology.
    """

    def __init__(self, left_info: dict, right_info: dict) -> None:
        self.left_info = left_info
        self.right_info = right_info

    def match_with_levenshtein(self, left_info: dict, right_info: dict, threshold: float = 90) -> dict:
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
            match, score = process.extractOne(left_label, right_info.values(), scorer=fuzz.token_sort_ratio)
            if score >= threshold:
                match_uri = [s for s,v in right_info.items() if v == match][0]
                matches[left_uri] = [match_uri, score]
        return matches
    
    def match_with_n_gram(self, left_info: dict, right_info: dict, threshold: float = 0.9) -> dict:
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
                similarity = Calculator.ngram_similarity(left_label, right_label)
                if similarity >= threshold:
                    matches[left_uri] = [right_uri, similarity]

    def match_with_cosine(self, left_info: dict, right_info: dict, threshold: float = 0.9) -> dict:
        """Matches labels based on cosine similarity.

            left_info (dict): labels and uris of entities from the left ontology
            right_info (dict): labels and uris of entities from the right ontology
            threshold (float): The minimal similarity score acceptable. Defaults to 90.

        Returns:
            dict: {left label: [matching right label, score]}
        """
        # Combine texts for vectorization
        all_texts = left_info.values() + right_info.values()
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
                confidence = similarity_matrix[i, j]
                if confidence >= threshold:
                    matches[uri1] = [uri2, confidence]
        return matches