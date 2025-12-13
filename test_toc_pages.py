import fitz  # PyMuPDF

def get_clean_toc(pdf_path):
    doc = fitz.open(pdf_path)
    
    # get_toc() returns a list of lists: [level, title, page_num]
    # simple=True removes complex link details, leaving just the core info
    toc = doc.get_toc(simple=True)
    
    if not toc:
        print("No embedded Table of Contents found.")
        return []

    print(f"Found {len(toc)} TOC entries.\n")
    
    cleaned_data = []
    
    for entry in toc:
        lvl, title, page_num = entry
        
        # INDUSTRIAL TIP: PyMuPDF returns 'page_num' as 1-based (Physical Page 1).
        # Python lists/arrays are 0-based. You MUST subtract 1 to use it as an index.
        real_page_index = page_num - 1
        
        print(f"Level: {lvl} | Title: '{title}' | jump_to_index: {real_page_index}")
        
        cleaned_data.append({
            "level": lvl,
            "title": title,
            "start_page_idx": real_page_index
        })
        
    return cleaned_data

# Usage
#toc_data = get_clean_toc("./files/BuildTrap.pdf")
doc = fitz.open("./files/BuildTrap.pdf")
page = doc[136]
print(page.get_text())