import os
import fitz  # PyMuPDF


def save_uploaded_file(uploaded_file):
    """
    Save uploaded PDF to the uploads folder.
    Returns the saved file path.
    """
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path


def extract_text_from_pdf(file_path):
    """
    Extract text from a PDF stored on disk.
    """
    pdf = fitz.open(file_path)

    full_text = ""

    for page_num, page in enumerate(pdf):
        text = page.get_text()
        full_text += f"\n--- Page {page_num + 1} ---\n"
        full_text += text

    pdf.close()

    return full_text