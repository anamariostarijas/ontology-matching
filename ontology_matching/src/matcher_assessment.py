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
    
    def transform_alignments(self, alignment_dictionary: dict) -> dict:
        """Transforms alignment graphs into dictionary of matches.

        Args:
            alignment_dictionary (dict): dictionary of the two alignment graphs, {"alignment": graph, "expected_alignment": graph} 

        Returns:
            dict: dictionary of matches for both alignments in the form:
                {"alignment": {<left entity>: [<right entity>, similarity]}, 
                 "expected_alignment": [<right entity>, similarity]}}
        """
        alignment_graph = alignment_dictionary["alignment"]
        expected_alignment_graph = alignment_dictionary["expected_alignment"]
        matches = {"alignment": {}, "expected_alignment": {}}
        name_alignment = self.alignment_path.split(".rdf")[0]
        ALIGN1 = Namespace(f"http://example.org/{name_alignment}#")
        for s, p, o in alignment_graph.triples((None, RDF.type, ALIGN1.alignmentCell)):
            ent2 = list(o2 for s2, p2, o2 in alignment_graph.triples((s, ALIGN1.entity2, None)))[0]
            ent1 = list(o2 for s2, p2, o2 in alignment_graph.triples((s, ALIGN1.entity1, None)))[0]
            confidence = list(o2 for s2, p2, o2 in alignment_graph.triples((s, ALIGN1.measure, None)))[0]
            matches["alignment"][ent1.fragment] = [ent2.fragment, confidence.value/100]
        ALIGN2 = Namespace("http://knowledgeweb.semanticweb.org/heterogeneity/alignment")
        for s, p, o in expected_alignment_graph.triples((None, RDF.type, ALIGN2.Cell)):
            ent2 = list(o2 for s2, p2, o2 in expected_alignment_graph.triples((s, ALIGN2.entity2, None)))[0]
            ent1 = list(o2 for s2, p2, o2 in expected_alignment_graph.triples((s, ALIGN2.entity1, None)))[0]
            confidence = list(o2 for s2, p2, o2 in expected_alignment_graph.triples((s, ALIGN2.measure, None)))[0]
            matches["expected_alignment"][ent1.fragment] = [ent2.fragment, confidence.value]
        return matches
    
    def compare_alignments(self) -> dict:
        """Reads the alignments and creates a summary dictionary of comparison between the alignments.

        Returns:
            dict: {"missing": {}, "incorrect": {}, "extras": {}, "correct": {}}
        """
        dict_alignments = self.import_alignments()
        matches = self.transform_alignments(dict_alignments)
        # find matches that are missing, incorrect matches and correct matches
        results = {"missing": {}, "incorrect": {}, "extras": {}, "correct": {}}
        for ent1, match in matches["expected_alignment"].items():
            # for ent1 find the match from generated alignment
            if ent1 in matches["alignment"] and matches["alignment"][ent1][0] == matches["expected_alignment"][ent1][0]:
                # match was found and is correct
                results["correct"][ent1] = match
            elif ent1 in matches["alignment"] and matches["alignment"][ent1][0] != matches["expected_alignment"][ent1][0]:
                # match was found and is not correct
                results["incorrect"][ent1] = match
            elif ent1 not in matches["alignment"]:
                # match was not found
                results["missing"][ent1] = match
        # matches made that shouldn't be there
        extras = set(matches["alignment"].keys()).difference(matches["expected_alignment"].keys())
        for extra in extras:
            results["extras"][extra] = matches["alignment"][extra]
        return results
