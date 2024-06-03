import rdflib
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, OWL, XSD
import nltk
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class OntologyMatcher:
    """Uses different simple ontology matching techniques and produces an alignment between the two input ontologies.
    """

    def __init__(self, left_ontology: str, right_ontology: str) -> None:
        self.left_ontology = left_ontology
        self.right_ontology = right_ontology

    def import_ontologies(self) -> dict:
        """Imports ontologies into Graph instances.

        Returns:
            dict: {<left_ontology name>: <Graph of ontology>, 
                    <left_ontology name>: <Graph of ontology>}
        """
        g1 = rdflib.Graph()
        g1.parse(self.left_ontology, format="xml")
        g2 = rdflib.Graph()
        g2.parse(self.right_ontology, format="xml")
        return {self.left_ontology.split(".")[0]: g1, self.right_ontology.split(".")[0]: g2}
    
    def extract_labels(self, graph: Graph, include_comments: bool = False) -> dict:
        texts = {}
        uris = set()
        for s, p, o in graph:
            if p == rdflib.RDFS.label or p == rdflib.namespace.SKOS.prefLabel:
                if s not in texts:
                    texts[s] = {"label": "", "comment": ""}
                    uris.add(s)
                texts[s]["label"] += " " + str(o)
            elif p == RDFS.comment:
                if s not in texts:
                    texts[s] = {"label": "", "comment": ""}
                    uris.add(s)
                texts[s]["comment"] += " " + str(o)
            elif p == rdflib.RDF.type and o == rdflib.OWL.Class:
                if s not in texts:
                    texts[s] = {"label": "", "comment": ""}
                    uris.add(s)
                texts[s]["label"] += " " + str(s.split('#')[-1])  # Extract class name from URI
        if include_comments:
            info = {s: e["label"] + " " + e["comment"] for s, e in texts.items()}
        else:
            info = {s: e["label"] for s, e in texts.items()}
        return info
    
    def match_ontologies(self, technique: str, threshold: float) -> dict:
        """Uses the technique specified and generates one-to-one matches between entities from the ontologies.

        Args:
            technique (str): Levenshtein, n-gram, cosine, path
            threshold (float): the minimum level of similarity

        Returns:
            dict: {entity1: [entity2, similarity score]}
                    where entity1 is from left_ontology and entity2 is from right_ontology
        """
        # Extract labels
        graphs = self.import_ontologies()
        g1 = graphs.values()[0]
        g2 = graphs.values()[1]
        info1 = self.extract_labels(g1)
        info2 = self.extract_labels(g2)
        matches = match_labels(left_labels, right_labels)
        # Print matches
        for match in matches:
            print(f"Match: {match[0]} <-> {match[1]} with score {match[2]}")

    def create_alignment_ontology(self, matches):
        # Create alignment ontology
        ALIGN = Namespace("http://example.org/alignment#")

        alignment_graph = Graph()
        alignment_graph.bind("align", ALIGN)

        alignment_graph.add((ALIGN.Correspondence, RDF.type, OWL.Class))
        alignment_graph.add((ALIGN.entity1, RDF.type, OWL.ObjectProperty))
        alignment_graph.add((ALIGN.entity2, RDF.type, OWL.ObjectProperty))
        alignment_graph.add((ALIGN.confidence, RDF.type, OWL.DatatypeProperty))
        alignment_graph.add((ALIGN.confidence, RDFS.range, XSD.float))

        def add_correspondence(alignment_graph, entity1, entity2, confidence):
            correspondence = URIRef(f"http://example.org/alignment#{entity1.split('#')[-1]}_{entity2.split('#')[-1]}")
            alignment_graph.add((correspondence, RDF.type, ALIGN.Correspondence))
            alignment_graph.add((correspondence, ALIGN.entity1, URIRef(entity1)))
            alignment_graph.add((correspondence, ALIGN.entity2, URIRef(entity2)))
            alignment_graph.add((correspondence, ALIGN.confidence, Literal(confidence, datatype=XSD.float)))

        for match in matches:
            entity1 = f"http://example.org/left_ontology#{match[0]}"
            entity2 = f"http://example.org/right_ontology#{match[1]}"
            confidence = match[2] / 100  # Normalize confidence to [0, 1]
            add_correspondence(alignment_graph, entity1, entity2, confidence)

        # Serialize the alignment ontology
        alignment_graph.serialize("alignment_ontology.owl", format="xml")