import rdflib
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, OWL, XSD

from ontology_matching.src.matching_techniques import MatchingTechnique

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
            if type(s) != rdflib.term.BNode:
                if o == rdflib.OWL.ObjectProperty:
                    if s not in texts:
                        texts[s] = {"label": "", "comment": ""}
                        uris.add(s)
                    label = str(s.replace("#", "/").split('/')[-1]) if "/" in s else s
                    texts[s]["label"] = label
                    if include_comments:
                        try:
                            comm = list(o2 for s2, p2, o2 in graph.triples((s, RDFS.comment, None)))[0]
                            texts[s]["comment"] = str(comm)
                        except:
                            print(f"No comment for {s} {p} {o}")
                elif o == rdflib.OWL.Class:
                    if s not in texts:
                        texts[s] = {"label": "", "comment": ""}
                        uris.add(s)
                    label = str(s.replace("#", "/").split('/')[-1]) if "/" in s else s
                    texts[s]["label"] = label  # Extract class name from URI
                    try:
                        comm = list(o2 for s2, p2, o2 in graph.triples((s, RDFS.comment, None)))[0]
                        texts[s]["comment"] = str(comm)
                    except:
                        print(f"No comment for {s} {p} {o}")
        if include_comments:
            info = {s: e["label"] + " " + e["comment"] for s, e in texts.items()}
        else:
            info = {s: e["label"] for s, e in texts.items()}
        return info
    
    def match_ontologies(self, technique: str, threshold: float, include_comments: bool = False, l: float = 0.7) -> dict:
        """Uses the technique specified and generates one-to-zero or one matches between entities from the ontologies, if the similarity score is greater than the threshold.

        Args:
            technique (str): levenshtein, ngram, cosine, path
            threshold (float): the minimum level of similarity

        Returns:
            dict: {entity1: [entity2, similarity score]}
                    where entity1 is from left_ontology and entity2 is from right_ontology
        """
        # Extract labels
        graphs = self.import_ontologies()
        left_g = graphs[self.left_ontology.split(".")[0]]
        right_g = graphs[self.right_ontology.split(".")[0]]
        left_info = self.extract_labels(left_g, include_comments)
        right_info = self.extract_labels(right_g, include_comments)
        # generate matches based on the technique
        if technique == "levenshtein":
            matches = MatchingTechnique.match_with_levenshtein(left_info, right_info, threshold*100)
        elif technique == "ngram":
            matches = MatchingTechnique.match_with_n_gram(left_info, right_info, threshold)
        elif technique == "cosine":
            matches = MatchingTechnique.match_with_cosine(left_info, right_info, threshold)
        elif technique == "path":
            matches = MatchingTechnique.match_with_path(left_g, right_g, threshold, l)
        else:
            raise ValueError(f"technique must be one of these: levenshtein, ngram, cosine, path!")
        # Print matches
        for match_l, match_r in matches.items():
            print(f"Match: {match_l} <-> {match_r[0]} with score {match_r[1]}")
        return matches
    
    def add_correspondence(self, alignment_graph: Graph, entity1: str, entity2: str, confidence: float, ALIGN: Namespace, name: str) -> None:
        """Adds a correspondence of type alignmentCell to alignment_graph with specific confidence measure.

        Args:
            alignment_graph (Graph): rdf alignment graph to update
            entity1 (str): left entity of the correspondence
            entity2 (str): right entity of the correspondence
            confidence (float): confidence measure score
            ALIGN (Namespace): Namespace of the graph
            name (str): name of the graph
        """
        correspondence = URIRef(f"http://example.org/{name}#{entity1.split('#')[-1]}_{entity2.split('#')[-1]}")
        alignment_graph.add((correspondence, RDF.type, ALIGN.alignmentCell))
        alignment_graph.add((correspondence, ALIGN.entity1, URIRef(entity1)))
        alignment_graph.add((correspondence, ALIGN.entity2, URIRef(entity2)))
        alignment_graph.add((correspondence, ALIGN.measure, Literal(confidence, datatype=XSD.float)))


    def create_alignment_ontology(self, matches: dict, file_name: str) -> None:
        """Generates an alignment rdf graph from the given matches and saves the graph under the name.

        Args:
            matches (dict): a dictionary of matches between the ontologies
            file_name (str): file path where to save the file
        """
        # Create alignment ontology
        ont_name = file_name.split(".rdf")[0]
        ALIGN = Namespace(f"http://example.org/{ont_name}#")
        alignment_graph = Graph()
        alignment_graph.bind("align", ALIGN)
        alignment_graph.add((ALIGN.alignmentCell, RDF.type, OWL.Class))
        alignment_graph.add((ALIGN.entity1, RDF.type, OWL.ObjectProperty))
        alignment_graph.add((ALIGN.entity2, RDF.type, OWL.ObjectProperty))
        alignment_graph.add((ALIGN.measure, RDF.type, OWL.DatatypeProperty))
        alignment_graph.add((ALIGN.measure, RDFS.range, XSD.float))
        for left_match, right_match in matches.items():
            entity1 = left_match
            entity2 = right_match[0]
            confidence = right_match[1]
            self.add_correspondence(alignment_graph, entity1, entity2, confidence, ALIGN, ont_name)

        # Serialize the alignment ontology
        alignment_graph.serialize(file_name, format="xml")

    
    