from pypdf import PdfReader
from docx import Document


def load_pdf(file_path):

    text = ""

    try:

        reader = PdfReader(file_path)

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:

                text += page_text + "\n"

    except Exception as e:

        print(
            f"PDF Error: {e}"
        )

    return text


def load_docx(file_path):

    text = ""

    try:

        document = Document(
            file_path
        )

        for paragraph in document.paragraphs:

            text += (
                paragraph.text + "\n"
            )

    except Exception as e:

        print(
            f"DOCX Error: {e}"
        )

    return text


def load_txt(file_path):

    text = ""

    try:

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            text = file.read()

    except Exception as e:

        print(
            f"TXT Error: {e}"
        )

    return text


def load_document(
    file_path,
    file_type
):

    if file_type == "pdf":

        return load_pdf(
            file_path
        )

    elif file_type == "docx":

        return load_docx(
            file_path
        )

    elif file_type == "txt":

        return load_txt(
            file_path
        )

    else:

        return ""