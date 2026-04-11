from app.services.document_text_service import DocumentTextService


def test_identify_if_needs_ocr_for_plain_text() -> None:
    service = DocumentTextService()

    needs_ocr, file_type, reason = service.identify_if_needs_ocr(
        b"Articulo 1. Texto legal de prueba", "doc.txt"
    )

    assert needs_ocr is False
    assert file_type == "text"
    assert "decoded" in reason.lower()


def test_identify_if_needs_ocr_for_pdf_without_extractable_text(monkeypatch) -> None:
    service = DocumentTextService()

    monkeypatch.setattr(service, "_extract_pdf_text", lambda _data: "")

    needs_ocr, file_type, reason = service.identify_if_needs_ocr(
        b"%PDF-1.4 fake", "ley.pdf"
    )

    assert needs_ocr is True
    assert file_type == "pdf"
    assert "little/no extractable text" in reason


def test_extract_pure_text_uses_ocr_path(monkeypatch) -> None:
    service = DocumentTextService()

    monkeypatch.setattr(
        service,
        "identify_if_needs_ocr",
        lambda _data, _path: (True, "image", "Image source requires OCR"),
    )
    monkeypatch.setattr(
        service,
        "perform_ocr_and_return_pure_text",
        lambda _data, _file_type: "texto ocr limpio",
    )

    text, used_ocr, file_type, reason = service.extract_pure_text(b"binary", "scan.png")

    assert text == "texto ocr limpio"
    assert used_ocr is True
    assert file_type == "image"
    assert "requires OCR" in reason
