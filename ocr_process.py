import ocrmypdf
import concurrent.futures
from pathlib import Path

BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(exist_ok=True)

def process_pdf(pdf_path : Path):
    output_pdf = OUTPUT_DIR / f"{pdf_path.stem}_ocr.pdf"
    print(f"Processing {pdf_path.name}")
    ocrmypdf.ocr(
        str(pdf_path),
        str(output_pdf),
        deskew=True,
        language='deu+eng',
        optimize=1,
        jobs=1,
    )

    print(f"Done {pdf_path.name}")

def process_all():
    pdf_files = list(INPUT_DIR.glob("*.pdf"))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(process_pdf, pdf_files)

if __name__ == '__main__':
    process_all()