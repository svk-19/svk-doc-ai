from pypdf import PdfReader


def load_pdf(pdf_path):

    text = ""

    try:

        reader = PdfReader(pdf_path)

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:

                text += page_text + "\n"

    except Exception as e:

        print(
            f"PDF Loading Error: {e}"
        )

    return text