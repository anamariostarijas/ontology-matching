import rdflib
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, OWL, XSD

class MatcherAssessment:
    """Compares the generated alignment graph with the expected alignment graph.
    """

    def __init__(self, alignment_path: str, expected_alignment_path: str) -> None:
        """_summary_

        Args:
            alignment (str): generated alignment graph path
            expected_alignment (str): expected alignment graph path

        """           
        self.alignment_path = alignment_path
        self.expected_alignment_path = expected_alignment_path

    def import_alignments(self) -> dict:
        """Imports the alignment graphs.

        Returns:
            dict: {"alignment": graph, "expected_alignment": graph}
        """
        g1 = rdflib.Graph()
        g1.parse(self.alignment_path, format="xml")
        g2 = rdflib.Graph()
        g2.parse(self.expected_alignment_path, format="xml")
        return {"alignment": g1, "expected_alignment": g2}
    
    def transform_alignments(self, alignment_graph: Graph, expected_alignment_graph: Graph) -> dict:
        """Transforms alignment graphs into dictionary of matches.

        Args:
            alignment_graph (Graph): generated alignment graph
            expected_alignment_graph (Graph): expected alignment graph

        Returns:
            dict: {"alignment": {<left entity>: [<right entity>, similarity]}, 
                   "expected_alignment": [<right entity>, similarity]}}
        """
        matches = {"alignment": {}, "expected_alignment": {}}
        name_alignment = self.alignment_path.split(".rdf")[0]
        ALIGN1 = Namespace(f"http://example.org/{name_alignment}#")
        for s, p, o in alignment_graph.triples((None, RDF.type, ALIGN1.alignmentCell)):
            ent2 = list(o2 for s2, p2, o2 in alignment_graph.triples((s, ALIGN1.entity2, None)))[0]
            ent1 = list(o2 for s2, p2, o2 in alignment_graph.triples((s, ALIGN1.entity1, None)))[0]
            confidence = list(o2 for s2, p2, o2 in alignment_graph.triples((s, ALIGN1.measure, None)))[0]
            matches["alignment"][ent1.fragment] = [ent2.fragment, confidence.value]
        ALIGN2 = Namespace("http://knowledgeweb.semanticweb.org/heterogeneity/alignment")
        for s, p, o in expected_alignment_graph.triples((None, RDF.type, ALIGN2.Cell)):
            ent2 = list(o2 for s2, p2, o2 in expected_alignment_graph.triples((s, ALIGN2.entity2, None)))[0]
            ent1 = list(o2 for s2, p2, o2 in expected_alignment_graph.triples((s, ALIGN2.entity1, None)))[0]
            confidence = list(o2 for s2, p2, o2 in expected_alignment_graph.triples((s, ALIGN2.measure, None)))[0]
            matches["expected_alignment"][ent1.fragment] = [ent2.fragment, confidence.value]
            print(f"Added {s}")
        return matches