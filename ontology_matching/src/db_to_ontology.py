import psycopg2
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, OWL, XSD


# Maps PostgreSQL data types to XSD types
_PG_TO_XSD = {
    "integer":            XSD.integer,
    "bigint":             XSD.integer,
    "smallint":           XSD.integer,
    "serial":             XSD.integer,
    "bigserial":          XSD.integer,
    "numeric":            XSD.decimal,
    "decimal":            XSD.decimal,
    "real":               XSD.float,
    "double precision":   XSD.double,
    "character varying":  XSD.string,
    "varchar":            XSD.string,
    "text":               XSD.string,
    "boolean":            XSD.boolean,
    "date":               XSD.date,
    "timestamp without time zone": XSD.dateTime,
    "timestamp with time zone":    XSD.dateTime,
}


class DBToOntology:
    """Generates an OWL ontology from a PostgreSQL database schema.

    Tables become OWL classes. A table whose primary key is also a foreign key
    to another table is declared a subclass of that table (table-per-type
    inheritance pattern). Regular columns become datatype properties; non-PK
    foreign-key columns become object properties.

    Usage::

        gen = DBToOntology(host="localhost", port=5432, dbname="bookstore",
                           user="postgres", password="secret")
        gen.generate("bookstore_from_db.rdf")
    """

    def __init__(
        self,
        host: str,
        port: int,
        dbname: str,
        user: str,
        password: str,
        base_iri: str = "http://example.org/bookstore#",
    ) -> None:
        self.conn_params = dict(host=host, port=port, dbname=dbname,
                                user=user, password=password)
        self.base_iri = base_iri if base_iri.endswith("#") else base_iri + "#"
        self.NS = Namespace(self.base_iri)

    # ------------------------------------------------------------------
    # Schema introspection helpers
    # ------------------------------------------------------------------

    def _get_tables(self, cur) -> list[str]:
        cur.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
            ORDER BY table_name
        """)
        return [row[0] for row in cur.fetchall()]

    def _get_primary_keys(self, cur, table: str) -> set[str]:
        cur.execute("""
            SELECT kcu.column_name
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
                ON tc.constraint_name = kcu.constraint_name
               AND tc.table_schema    = kcu.table_schema
            WHERE tc.constraint_type = 'PRIMARY KEY'
              AND tc.table_schema    = 'public'
              AND tc.table_name      = %s
        """, (table,))
        return {row[0] for row in cur.fetchall()}

    def _get_foreign_keys(self, cur, table: str) -> list[tuple[str, str]]:
        """Returns [(column_name, referenced_table_name), ...]."""
        cur.execute("""
            SELECT kcu.column_name, ccu.table_name AS foreign_table
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
                ON tc.constraint_name = kcu.constraint_name
               AND tc.table_schema    = kcu.table_schema
            JOIN information_schema.constraint_column_usage AS ccu
                ON ccu.constraint_name = tc.constraint_name
               AND ccu.table_schema    = tc.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY'
              AND tc.table_schema    = 'public'
              AND tc.table_name      = %s
        """, (table,))
        return cur.fetchall()

    def _get_columns(self, cur, table: str) -> list[tuple[str, str]]:
        """Returns [(column_name, data_type), ...] ordered by position."""
        cur.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = %s
            ORDER BY ordinal_position
        """, (table,))
        return cur.fetchall()

    # ------------------------------------------------------------------
    # OWL generation
    # ------------------------------------------------------------------

    def _class_uri(self, table: str) -> URIRef:
        return self.NS[table.capitalize()]

    def _prop_uri(self, column: str) -> URIRef:
        return self.NS[column]

    def generate(self, output_path: str) -> Graph:
        """Introspects the database and writes an OWL/RDF file.

        Args:
            output_path: path where the .rdf file will be saved.

        Returns:
            The populated rdflib Graph.
        """
        g = Graph()
        NS = self.NS
        g.bind("",     NS)
        g.bind("owl",  OWL)
        g.bind("rdfs", RDFS)
        g.bind("xsd",  XSD)

        ont_uri = URIRef(self.base_iri.rstrip("#"))
        g.add((ont_uri, RDF.type, OWL.Ontology))

        conn = psycopg2.connect(**self.conn_params)
        cur  = conn.cursor()

        tables = self._get_tables(cur)

        # Collect FK info for all tables upfront
        fk_map  = {t: self._get_foreign_keys(cur, t) for t in tables}
        pk_map  = {t: self._get_primary_keys(cur, t)  for t in tables}

        # Identify inheritance: table whose entire PK is a FK to another table
        parent_map: dict[str, str] = {}
        for table, fks in fk_map.items():
            fk_cols = {col for col, _ in fks}
            if pk_map[table] and pk_map[table].issubset(fk_cols):
                # All PK columns are FKs → this is a subtype
                # Take the referenced table of the first PK FK as parent
                pk_fk_targets = [ft for col, ft in fks if col in pk_map[table]]
                if pk_fk_targets:
                    parent_map[table] = pk_fk_targets[0]

        for table in tables:
            class_uri = self._class_uri(table)
            g.add((class_uri, RDF.type,      OWL.Class))
            g.add((class_uri, RDFS.label,    Literal(table.capitalize())))

            if table in parent_map:
                g.add((class_uri, RDFS.subClassOf, self._class_uri(parent_map[table])))

            pk_cols = pk_map[table]
            # Map FK column -> referenced table for quick lookup
            fk_col_to_table = {col: ft for col, ft in fk_map[table]}

            for col_name, data_type in self._get_columns(cur, table):
                if col_name in pk_cols:
                    continue  # skip surrogate/inherited PKs

                prop_uri = self._prop_uri(col_name)

                if col_name in fk_col_to_table:
                    # Non-PK FK → object property
                    g.add((prop_uri, RDF.type,      OWL.ObjectProperty))
                    g.add((prop_uri, RDFS.label,    Literal(col_name)))
                    g.add((prop_uri, RDFS.domain,   class_uri))
                    g.add((prop_uri, RDFS.range,    self._class_uri(fk_col_to_table[col_name])))
                else:
                    # Plain column → datatype property
                    xsd_type = _PG_TO_XSD.get(data_type.lower(), XSD.string)
                    g.add((prop_uri, RDF.type,      OWL.DatatypeProperty))
                    g.add((prop_uri, RDFS.label,    Literal(col_name)))
                    g.add((prop_uri, RDFS.domain,   class_uri))
                    g.add((prop_uri, RDFS.range,    xsd_type))

        cur.close()
        conn.close()

        g.serialize(output_path, format="xml")
        print(f"Ontology written to {output_path}")
        return g
