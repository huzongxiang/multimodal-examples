from pdf2image import convert_from_path
import os
import shutil
import sys
from pathlib import Path


def _resolve_poppler_path():
    configured_path = os.getenv("POPPLER_PATH")
    if configured_path and Path(configured_path).exists():
        return configured_path

    env_bin = Path(sys.executable).resolve().parent
    if (env_bin / "pdfinfo").exists() and (env_bin / "pdftoppm").exists():
        return str(env_bin)

    return None

class PdfManager:
    def __init__(self):
        pass
        
    def clear_and_recreate_dir(self, output_folder):
        print(f"Clearing output folder {output_folder}")

        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)

        os.makedirs(output_folder)

    def save_images(self, id, pdf_path, max_pages, pages: list[int] = None) -> list[str]:
        output_folder = f"pages/{id}/"
        images = convert_from_path(pdf_path, poppler_path=_resolve_poppler_path())

        print(f"Saving images from {pdf_path} to {output_folder}. Max pages: {max_pages}")

        self.clear_and_recreate_dir(output_folder)

        num_page_processed = 0

        for i, image in enumerate(images):
            if max_pages and num_page_processed >= max_pages:
                break

            if pages and i not in pages:
                continue

            full_save_path = f"{output_folder}/page_{i + 1}.png"

            #print(f"Saving image to {full_save_path}")

            image.save(full_save_path, "PNG")

            num_page_processed += 1

        return [f"{output_folder}/page_{i + 1}.png" for i in range(num_page_processed)]
