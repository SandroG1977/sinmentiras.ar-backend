import re
from typing import TypedDict


class LegislativeMetadata(TypedDict):
    iniciado_en: str | None
    expediente_diputados: str | None
    expediente_senado: str | None
    publicado_en: str | None
    fecha: str | None
    ley_numero: str | None


class LegislativeMetadataService:
    def infer_from_text(self, text: str) -> LegislativeMetadata:
        content = text or ""

        ley_numero = self._match_first(
            content,
            [
                r"\bley\s*(?:n(?:u|u\.)?m(?:ero)?\s*)?([0-9]{2}\.?[0-9]{3})\b",
                r"\bley\s+([0-9]{4,6})\b",
            ],
        )

        expediente_diputados = self._match_first(
            content,
            [r"expediente\s*(?:de)?\s*diputados\s*[:\-]?\s*([A-Za-z0-9\-./]+)"],
        )
        expediente_senado = self._match_first(
            content,
            [r"expediente\s*(?:de)?\s*senado\s*[:\-]?\s*([A-Za-z0-9\-./]+)"],
        )

        iniciado_en = self._match_first(
            content,
            [
                r"iniciado\s+en\s*[:\-]?\s*(camara\s+de\s+diputados|camara\s+de\s+senadores|senado)",
            ],
        )

        publicado_en = self._match_first(
            content,
            [
                r"publicad[oa]\s+en\s*[:\-]?\s*(boletin\s+oficial[^\n\.]{0,80})",
            ],
        )

        fecha = self._match_first(
            content,
            [
                r"\b([0-3]?\d/[01]?\d/[12]\d{3})\b",
                r"\b([12]\d{3}-[01]\d-[0-3]\d)\b",
            ],
        )

        return {
            "iniciado_en": self._normalize(iniciado_en),
            "expediente_diputados": self._normalize(expediente_diputados),
            "expediente_senado": self._normalize(expediente_senado),
            "publicado_en": self._normalize(publicado_en),
            "fecha": self._normalize(fecha),
            "ley_numero": self._normalize(ley_numero),
        }

    @staticmethod
    def _match_first(text: str, patterns: list[str]) -> str | None:
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    @staticmethod
    def _normalize(value: str | None) -> str | None:
        if not value:
            return None
        return re.sub(r"\s+", " ", value).strip()


legislative_metadata_service = LegislativeMetadataService()
