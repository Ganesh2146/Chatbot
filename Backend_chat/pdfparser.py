from PyPDF2 import PdfReader
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = []
    for page in reader.pages:
        text.append(page.extract_text())
    return "\n".join(text)
# pdf = extract_text_from_pdf("pdf.pdf")
# print(pdf)