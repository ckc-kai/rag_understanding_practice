import logging
from llama_index.core.schema import TextNode
from llama_index.core.node_parser import SentenceSplitter
import fitz
from config import setup_logger

setup_logger()
logger = logging.getLogger(__name__)

def get_chapter_nodes(pdf_path):
    '''
    This is a function that extract the level 1 title and page number from the pdf file.
    '''
    doc = fitz.open(pdf_path)
    # Get toc
    toc = doc.get_toc(simple=True)
    
    if not toc:
        logger.warning("No TOC found! Falling back to full text.")
        full_text = ""
        for page in doc:
            full_text += page.get_text() + "\n"
        return [TextNode(text=full_text, metadata={"title": "Full Document"})]

    # Convert to 0-based indices and prepare list
    chapters = []
    for entry in toc:
        level, title, page_num = entry
        if level == 1:
            start_idx = page_num - 1 
            chapters.append({
                "title": title,
                "start_idx": start_idx
            })

    chapters.sort(key=lambda x: x['start_idx'])
    
    nodes = []
    for i, chapter in enumerate(chapters):
        start_idx = chapter['start_idx']
        
        # End page is the start of next chapter, or last page
        if i < len(chapters) - 1:
            end_idx = chapters[i+1]['start_idx']
        else:
            end_idx = len(doc)
            
        # Extract text
        chapter_text = ""
        # fitz pages are 0-indexed
        for p in range(start_idx, end_idx):
            chapter_text += doc[p].get_text() + "\n"
            
        # Sub-chunk the chapter text to fit embedding model (512 tokens)
        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        chunks = splitter.split_text(chapter_text)
        
        for chunk_text in chunks:
            node = TextNode(
                text=chunk_text,
                metadata={
                    "title": chapter['title'],
                    "start_page_idx": start_idx,
                    "end_page_idx": end_idx
                }
            )
            nodes.append(node)
        
    logger.info(f"Created {len(nodes)} nodes from chapters using PyMuPDF (chunked to 512 tokens).")
    return nodes