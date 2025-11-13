"""
Quick inspection utility to verify which sources (CSV, TXT, PDF) are retrieved
for a set of queries. This does not modify the vector store; it only queries
the existing retriever from `vector.py`.
"""

import os
import sys

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

from backend.vector import retriever

TEST_QUERIES = [
	#only in TXT
	"HackRonyX 2025",
	"GitHub rn-dev",
	"SVPCET Nagpur",
	#in PDF
	"ukulele",
	#CSV
	"Smart India Hackathon 2024",
	"AI Health and Wellness coach",
	#Cross-source concepts
	"hockey at the national level",
]


def run_queries(queries):
	for q in queries:
		print("\n=== QUERY ===\n", q)
		docs = retriever.invoke(q)
		print(f"Retrieved: {len(docs)} docs")
		for i, d in enumerate(docs, start=1):
			src = d.metadata.get("source", "unknown")
			topic = d.metadata.get("topic")
			preview = (d.page_content or "")[:240].replace("\n", " ")
			print(f"-- {i}. Source: {src} | Topic: {topic or 'N/A'}\n   {preview}")


if __name__ == "__main__":
	run_queries(TEST_QUERIES)
	try:
		from pathlib import Path
		import PyPDF2
		pdf_dir = Path("data/pdfs")
		for pdf_path in pdf_dir.glob("*.pdf"):
			reader = PyPDF2.PdfReader(str(pdf_path))
			text = "\n".join([p.extract_text() or "" for p in reader.pages])
			print(f"\n[PDF CHECK] {pdf_path.name}: chars={len(text)} pages={len(reader.pages)}")
			print((text[:400] or "<no text extracted>").replace("\n"," "))
	except Exception as e:
		print("[PDF CHECK] Error:", e)
