from app.services.legislative_metadata_service import LegislativeMetadataService


def test_infer_legislative_metadata_from_text() -> None:
    service = LegislativeMetadataService()
    text = (
        "Iniciado en: Camara de Diputados\n"
        "Expediente Diputados: 1234-D-2026\n"
        "Expediente Senado: 5678-S-2026\n"
        "Publicado en: Boletin Oficial de la Republica Argentina\n"
        "Fecha: 2026-04-10\n"
        "LEY 27804\n"
    )

    metadata = service.infer_from_text(text)

    assert metadata["iniciado_en"] == "Camara de Diputados"
    assert metadata["expediente_diputados"] == "1234-D-2026"
    assert metadata["expediente_senado"] == "5678-S-2026"
    assert "Boletin Oficial" in str(metadata["publicado_en"])
    assert metadata["fecha"] == "2026-04-10"
    assert metadata["ley_numero"] == "27804"
