import os
import requests
from pdf2image import convert_from_path
import tempfile

PAPERS_API = "https://paperswithcode.com/api/v1/papers/"

def find_arxiv_pdf_url(topic: str) -> str:
    """
    Searches Papers With Code for a topic and returns the ArXiv PDF URL of the top result.
    """
    search_url = f"https://paperswithcode.com/search?q={topic.replace(' ', '+')}"
    response = requests.get(search_url)
    
    if response.status_code != 200:
        raise Exception(f"Failed to search PapersWithCode: {response.status_code}")
    
    # Dirty extract of first result’s slug
    start_index = response.text.find("/paper/")
    if start_index == -1:
        raise Exception("No paper found.")
    
    end_index = response.text.find('"', start_index)
    slug = response.text[start_index:end_index].split("/paper/")[-1]
    
    api_url = PAPERS_API + slug
    paper_info = requests.get(api_url).json()
    
    if "paper_url" not in paper_info or not paper_info["paper_url"].startswith("https://arxiv.org"):
        raise Exception("No ArXiv paper found.")
    
    # Convert ArXiv page to PDF link
    arxiv_id = paper_info["paper_url"].split("/")[-1]
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    return pdf_url

def extract_diagrams_from_arxiv(topic: str, save_dir="arxiv_images") -> list:
    """
    Downloads the ArXiv paper PDF and extracts images using pdf2image.
    Returns a list of saved image file paths.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    pdf_url = find_arxiv_pdf_url(topic)
    print(f"Found PDF: {pdf_url}")
    
    pdf_response = requests.get(pdf_url)
    if pdf_response.status_code != 200:
        raise Exception("Failed to download PDF.")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
        temp_pdf.write(pdf_response.content)
        temp_pdf_path = temp_pdf.name

    images = convert_from_path(temp_pdf_path, dpi=200)

    saved_paths = []
    for i, image in enumerate(images[:5]):  # Limit to first 5 pages
        path = os.path.join(save_dir, f"{topic.replace(' ', '_')}_page_{i + 1}.png")
        image.save(path, "PNG")
        saved_paths.append(path)

    return saved_paths
