import docx2txt
import PyPDF2
import io

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def extract_text_from_docx(file):
    # docx2txt expects a path or a file-like object
    if hasattr(file, "read"):
        # If file is a file-like object, save to BytesIO and pass to docx2txt
        file.seek(0)
        temp_file = io.BytesIO(file.read())
        temp_file.seek(0)
        text = docx2txt.process(temp_file)
        temp_file.close()
        return text
    else:
        return docx2txt.process(file)