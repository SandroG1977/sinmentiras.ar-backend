from io import BytesIO
from pathlib import Path


class DocumentTextService:
    def _is_probably_text(self, data: bytes) -> bool:
        if not data:
            return False
        sample = data[:2048]
        if b"\x00" in sample:
            return False
        try:
            sample.decode("utf-8")
            return True
        except UnicodeDecodeError:
            return False

    def _detect_file_type(self, data: bytes, source_path: str) -> str:
        extension = Path(source_path.lower()).suffix
        if data.startswith(b"%PDF") or extension == ".pdf":
            return "pdf"
        if data.startswith(b"\x89PNG") or extension == ".png":
            return "image"
        if data.startswith(b"\xff\xd8\xff") or extension in {".jpg", ".jpeg"}:
            return "image"
        if extension in {".txt", ".md", ".json", ".csv"}:
            return "text"
        if self._is_probably_text(data):
            return "text"
        return "binary"

    def _extract_pdf_text(self, data: bytes) -> str:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("pypdf is required for PDF text extraction") from exc

        reader = PdfReader(BytesIO(data))
        parts: list[str] = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                parts.append(page_text)
        return "\n\n".join(parts).strip()

    def identify_if_needs_ocr(
        self, data: bytes, source_path: str
    ) -> tuple[bool, str, str]:
        file_type = self._detect_file_type(data, source_path)

        if file_type == "image":
            return True, file_type, "Image source requires OCR"

        if file_type == "text":
            return False, file_type, "Text source can be decoded directly"

        if file_type == "pdf":
            extracted = self._extract_pdf_text(data)
            needs_ocr = len(extracted.strip()) < 80
            reason = (
                "PDF has little/no extractable text"
                if needs_ocr
                else "PDF text extraction succeeded"
            )
            return needs_ocr, file_type, reason

        return True, file_type, "Unknown/binary source requires OCR"

    def perform_ocr_and_return_pure_text(self, data: bytes, file_type: str) -> str:
        try:
            import pytesseract
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("pytesseract and Pillow are required for OCR") from exc

        if file_type == "image":
            image = Image.open(BytesIO(data))
            return pytesseract.image_to_string(image, lang="spa+eng").strip()

        if file_type == "pdf":
            try:
                import pypdfium2 as pdfium
            except ImportError as exc:
                raise RuntimeError("pypdfium2 is required for PDF OCR") from exc

            pdf = pdfium.PdfDocument(data)
            pages_text: list[str] = []
            for index in range(len(pdf)):
                page = pdf[index]
                bitmap = page.render(scale=2).to_pil()
                page_text = pytesseract.image_to_string(bitmap, lang="spa+eng").strip()
                if page_text:
                    pages_text.append(page_text)
            return "\n\n".join(pages_text).strip()

        raise RuntimeError("OCR is not supported for this file type")

    def extract_pure_text(
        self, data: bytes, source_path: str
    ) -> tuple[str, bool, str, str]:
        needs_ocr, file_type, reason = self.identify_if_needs_ocr(data, source_path)

        if not needs_ocr:
            if file_type == "pdf":
                text = self._extract_pdf_text(data)
            else:
                text = data.decode("utf-8", errors="ignore")
            return text.strip(), False, file_type, reason

        text = self.perform_ocr_and_return_pure_text(data, file_type)
        return text.strip(), True, file_type, reason


document_text_service = DocumentTextService()
