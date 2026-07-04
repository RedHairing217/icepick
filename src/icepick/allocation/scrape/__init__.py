"""In-house acquisition scrapers.

Each scraper turns an external source into raw candidate rows (upstream
shape ``link / question / answer / tier / truth`` plus ``metadata``) for an
allocation adapter to normalise into canonical handoff records. Scrapers
live here — behind ``allocation`` — never inside ``processing``.

Current scrapers: ``realmath`` (arXiv, in-house).
"""
