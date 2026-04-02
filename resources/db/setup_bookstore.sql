-- Bookstore database schema
-- Tables mirror the Slovenian ontology (ontologija knjigarna.owx):
--   product  -> Izdelek   (title->Naslov, price->Cena, year->Leto)
--   book     -> Knjiga    (author->Avtor)
--   magazine -> Revija
--   notebook -> Zvezek    (size->Velikost)

DROP TABLE IF EXISTS notebook;
DROP TABLE IF EXISTS magazine;
DROP TABLE IF EXISTS book;
DROP TABLE IF EXISTS product;

CREATE TABLE product (
    id     SERIAL PRIMARY KEY,
    title  VARCHAR(255) NOT NULL,
    price  NUMERIC(10, 2),
    year   INTEGER
);

CREATE TABLE book (
    id      INTEGER PRIMARY KEY REFERENCES product(id) ON DELETE CASCADE,
    author  VARCHAR(255),
    isbn    VARCHAR(20)
);

CREATE TABLE magazine (
    id           INTEGER PRIMARY KEY REFERENCES product(id) ON DELETE CASCADE,
    publisher    VARCHAR(255),
    issue_number INTEGER
);

CREATE TABLE notebook (
    id    INTEGER PRIMARY KEY REFERENCES product(id) ON DELETE CASCADE,
    size  VARCHAR(50),
    pages INTEGER
);

-- Sample data
INSERT INTO product (title, price, year) VALUES
    ('The Great Gatsby',        12.99, 1925),
    ('Dune',                    15.50, 1965),
    ('National Geographic',      8.00, 2023),
    ('Scientific American',      7.50, 2023),
    ('Moleskine Classic',        14.99, 2022),
    ('Leuchtturm1917',           16.99, 2023);

INSERT INTO book (id, author, isbn) VALUES
    (1, 'F. Scott Fitzgerald', '978-0743273565'),
    (2, 'Frank Herbert',       '978-0441013593');

INSERT INTO magazine (id, publisher, issue_number) VALUES
    (3, 'National Geographic Society', 201),
    (4, 'Springer Nature',             315);

INSERT INTO notebook (id, size, pages) VALUES
    (5, 'A5', 192),
    (6, 'B5', 249);
