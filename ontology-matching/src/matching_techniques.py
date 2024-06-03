import rdflib
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, OWL, XSD
import nltk
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
    
    def ngram_similarity(self, str1: str, str2: str, n: int = 3) -> float:
        """Calculates n-gram similarity between str1 and str2 for given n.

        Args:
            str1 (str): left string to compare
            str2 (str): right string to compare
            n (int, optional): length of the substrings. Defaults to 3.

        Returns:
            float: similarity score calculated
        """
        ngrams1 = set(ngrams(str1, n))
        ngrams2 = set(ngrams(str2, n))
        intersection = ngrams1.intersection(ngrams2)
        denom = min(len(str1), len(str2)) - n + 1
        return len(intersection) / denom

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
                similarity = self.ngram_similarity(left_label, right_label)
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
    
    def get_children(self, graph: Graph, entity: rdflib.term.URIRef) -> list:
        """Retrieves children of a class in a graph.

        Args:
            graph (Graph): ontology graph
            entity (rdflib.term.URIRef): en entity of an ontology

        Returns:
            list: list of children
        """
        returnlist = [s for s,v,o in graph.triples((None, RDFS.subClassOf, entity))]
        return returnlist

    def extract_class_children(self, graph: Graph) -> dict:
        """Returns a dictionary, where keys are entities and their values are their children.

        Args:
            graph (Graph): ontology graph

        Returns:
            dict: {uri: uris of children}
        """
        hierarchy = {}
        for s, p, o in graph:
            if p == RDF.type and o == OWL.Class:
                hier = self.get_children(graph, s)
                hierarchy[s] = hier
            elif p == rdflib.RDFS.label or p == rdflib.namespace.SKOS.prefLabel:
                hierarchy[s] = []
        return hierarchy

    def extract_class_parents(self, children_dict: dict) -> dict:
        """Transforms a dictionary of children into a dictionary of parents.

        Args:
            children_dict (dict): dictionary of children retrieved from method extract_class_children

        Returns:
            dict: {uri: uri of the parent}
        """
        hierarchy = {}
        for k, v in children_dict.items():
            for child in v:
                hierarchy[child] = k
        return hierarchy

    def find_paths(self, graph: Graph) -> dict:
        """Returns the paths of class hierarchy, starting with the root and going until the entity.

        Args:
            graph (Graph): ontology graph

        Returns:
            dict: {uri: <root name>:<child of root>:...:<parent of entity>:<entity>}
        """
        first_level_hier = self.extract_class_children(graph)
        parents = self.extract_class_parents(first_level_hier)
        complete = {}
        for s, p, o in graph:
            if p == RDF.type and o == OWL.Class:
                parent_id = parents[s] if s in parents else ""
                parent = parents[s].replace("#", "/").split("/")[-1] if s in parents else ""
                #print(f"{s} has parent {parent}")
                path = str(parent)
                while parent != "":
                    parent = parents[parent_id].replace("#", "/").split("/")[-1] if parent_id in parents else ""
                    parent_id = parents[parent_id] if parent_id in parents else ""
                    path = str(parent) + ":" + path
                    #print(f"----which has parent {parent}")
                complete[s] = path + ":" + s.replace("#", "/").split("/")[-1]
            elif p == rdflib.RDFS.label or p == rdflib.namespace.SKOS.prefLabel:
                complete[s] = s.replace("#", "/").split("/")[-1]
        return complete

    def path_distance(self, ss: list, tt: list, l: float = 0.7) -> float:
        """Calculates path distance between string sequences ss and tt with lambda l.
        The formula uses levenshtein distance as a base distance.

        Args:
            ss (list): first string sequence
            tt (list): second string sequence
            l (float, optional): _description_. Defaults to 0.7.

        Returns:
            float: _description_
        """
        n = len(ss)
        m = len(tt)
        if n == 0 or m == 0:
            k = m if n == 0 else n
            #print(f"k = {k}")
            return k
        else:
            left_label = ss[n-1]
            right_label = tt[m-1]
            #print(f"left_label = {left_label}, right_label = {right_label}, n={n}, m = {m}")
            match, score = process.extractOne(left_label, [right_label], scorer=fuzz.token_sort_ratio)
            #print("distance: " + str(1 - score/100))
            return ((l * (1 - score/100)) + (1 - l) * self.path_distance(ss[:n-1], tt[:m-1], l))