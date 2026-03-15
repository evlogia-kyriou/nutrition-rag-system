"""
Document loader for PDF files using PyMuPDF.
Extracts text and metadata from nutrition books.
"""
import os
from typing import List, Dict, Any
from pathlib import Path
import fitz # PyMyPDF
from llama_index.core import Document
from llama_index.core.schema import TextNode


class PDFDocumentLoader:
    """
    Loads PDF documents and converts to LlamaIndex format.
    Uses PyMyPDF for robust PDF parsing.
    """

    def __init__(self, pdf_directory: str= "./data/raw"):
        """
        Initialize PDF loader.
        
        Args:
            pdf_directory: Directory containing PDF files
        """
        self.pdf_directory = Path(pdf_directory)

        if not self.pdf_directory.exists():
            raise FileNotFoundError(f"Directory not found: {pdf_directory}")
    
    def load_pdf(self, pdf_path: Path) -> List[Document]:
        """
        Load single PDF file.
        Args: 
            pdf_path: Path to PDF file
        
        Returns:
            List of Document objects (one per page)
        """
        documents = []

        try: 
            # Open PDF
            pdf_doc = fitz.open(pdf_path)
            book_name = pdf_path.stem

            print(f" Processing: {book_name}")
            print(f" Pages: {len(pdf_doc)}")

            # Extract text from each page
            for page_num in range(len(pdf_doc)):
                page = pdf_doc[page_num]
                text = page.get_text()

                # Skip empty pages
                if not text.strip():
                    continue

                # Create LlamaIndex Document
                doc = Document(
                    text = text,
                    metadata = {
                        "book_name": book_name,
                        "page_number": page_num + 1,
                        "total_pages": len(pdf_doc),
                        "source_file": str(pdf_path)
                    }
                )
                documents.append(doc)
            
            pdf_doc.close()

            print(f" Extracted {len(documents)} pages")
            return documents
    
        except Exception as e:
            print(f" Error processing {pdf_path.name}: {e}")
            return []
        
    def load_all_pdf(self) -> List[Document]:
        """
        Load all PDF files from directory
        Returns:
            List of all documents frol all PDFs
        """
        print(f"\n Loading PDFs from: {self.pdf_directory}")

        # Find all PDF file
        pdf_files = list(self.pdf_directory.glob("*.pdf"))

        if not pdf_files:
            raise FileNotFoundError(
                f"No PDF files found in {self.pdf_directory}\n"
                f"Please place your Nutrition books in data/raw/"
            )
        
        print(f" Found{len(pdf_files)} PDF files\n")

        # Load all PDFs
        all_documents = []
        for pdf_path in pdf_files:
            docs= self.load_pdf(pdf_path)
            all_documents.extend(docs)
        
        print(f" Total documents load: {len(all_documents)}")
        print(f" Total text length: {sum(len(doc.text) for doc in all_documents):,} characters")

        return all_documents
    
    def get_document_stats(self, documents: List[Document]) -> Dict[str, any]:
        """
        Get statistics about loaded documents.
        Args: 
            documents: List of documents
            
        Returns:
            Dictionary with statistics
        """

        books = {}
        total_chars = 0

        for doc in documents:
            book_name = doc.metadata["book_name"]
            if book_name not in books:
                books[book_name] = {
                    "pages": 0,
                    "chars": 0
                }
            books[book_name]["pages"] += 1
            books[book_name]["chars"] += len(doc.text)
            total_chars += len(doc.text)
        
        return {
            "total_documents": len(documents),
            "total_books": len(books),
            "total_chars": total_chars,
            "books": books
        }
