"""RFC003 D3 Observation Sidecar (RFC001 §37, RFC003 §24–§25).

Maps Canonical AST nodes to rendered-page geometry extracted from the
LibreOffice PDF text layer (`pdftotext -bbox-layout`), producing the
Observation Sidecar. Geometry is integer ppm of the page; the sidecar
never enters the AST or its hash.
"""
