from nltk.util import ngrams
from fuzzywuzzy import process
from fuzzywuzzy import fuzz
import rdflib
from rdflib import Graph
from rdflib.namespace import RDF, RDFS, OWL

class Calculator:

    def __init__(self) -> None:
        pass

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
    
    def path_distance(self, ss: list, tt: list, l: float = 0.7) -> float:
        """Calculates path distance between string sequences ss and tt with lambda l.
        The formula uses levenshtein distance as a base distance.

        Args:
            ss (list): first string sequence
            tt (list): second string sequence
            l (float, optional): Lambda factor. Defaults to 0.7.

        Returns:
            float: path distance
        """
        n = len(ss)
        m = len(tt)
        if n == 0 or m == 0:
            k = m if n == 0 else n
            return k
        else:
            left_label = ss[n-1]
            right_label = tt[m-1]
            match, score = process.extractOne(left_label, [right_label], scorer=fuzz.token_sort_ratio)
            return ((l * (1 - score/100)) + (1 - l) * self.path_distance(ss[:n-1], tt[:m-1], l))
        

class Processor:

    def __init__(self) -> None:
        pass

    def get_parent(self, graph: Graph, entity: str) -> list:
        """Retrieves children of a class in a graph.

        Args:
            graph (Graph): ontology graph
            entity (str): uri of the entity of an ontology

        Returns:
            list: list of children
        """
        parents = [o for s,v,o in graph.triples((rdflib.term.URIRef(entity), RDFS.subClassOf, None)) if ("#" in o or "/" in o)]
        parent = parents[0] if len(parents) != 0 else ""
        return parent

    def extract_class_parents(self, graph):
        """Transforms a dictionary of children into a dictionary of parents.

        Args:
            graph (Graph): ontology graph

        Returns:
            dict: {uri: uri of the parent}
        """
        hierarchy = {}
        for s, p, o in graph:
            if o == OWL.Class:
                hier = self.get_parent(graph, s)
                hierarchy[s] = hier
        return hierarchy

    def find_paths(self, graph: Graph) -> dict:
        """Returns the paths of class hierarchy, starting with the root and going until the entity.

        Args:
            graph (Graph): ontology graph

        Returns:
            dict: {uri: <root name>:<child of root>:...:<parent of entity>:<entity>}
        """
        parents = self.extract_class_parents(graph)
        complete = {}
        for s, p, o in graph:
            if o == OWL.Class:
                parent_id = parents[s] if s in parents else ""
                parent = parents[s].replace("#", "/").split("/")[-1] if s in parents else None
                path = []
                while parent != "":
                    path = [parent] + path
                    parent = parents[parent_id].replace("#", "/").split("/")[-1] if parent_id in parents else ""
                    parent_id = parents[parent_id] if parent_id in parents else ""
                    #print(f"----which has parent {parent}")
                complete[s] = path + [s.replace("#", "/").split("/")[-1]]
            elif s not in complete and (p == rdflib.RDFS.label or p == rdflib.namespace.SKOS.prefLabel):
                complete[s] = [s.replace("#", "/").split("/")[-1]]
        return complete