from urllib.parse import urlparse

from fastapi import HTTPException

from app.core.config import settings


class MinIOService:
    @staticmethod
    def parse_minio_path(minio_path: str) -> tuple[str, str]:
        # Supports both `minio://bucket/object` and `bucket/object`.
        if minio_path.startswith("minio://"):
            parsed = urlparse(minio_path)
            bucket = parsed.netloc.strip()
            object_path = parsed.path.lstrip("/").strip()
        else:
            parts = minio_path.split("/", 1)
            if len(parts) != 2:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid minio_path. Use minio://bucket/object or bucket/object",
                )
            bucket, object_path = parts[0].strip(), parts[1].strip()

        if not bucket or not object_path:
            raise HTTPException(
                status_code=400,
                detail="Invalid minio_path. Bucket and object are required",
            )
        return bucket, object_path

    def _build_client(self):
        if not settings.MINIO_ENDPOINT:
            raise HTTPException(
                status_code=500, detail="MINIO_ENDPOINT is not configured"
            )
        if not settings.MINIO_ACCESS_KEY or not settings.MINIO_SECRET_KEY:
            raise HTTPException(
                status_code=500,
                detail="MINIO_ACCESS_KEY and MINIO_SECRET_KEY are required",
            )

        try:
            from minio import Minio
        except ImportError as exc:
            raise HTTPException(
                status_code=500,
                detail="Package 'minio' is not installed in the backend environment",
            ) from exc

        return Minio(
            endpoint=settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE,
        )

    def read_bytes(self, minio_path: str) -> bytes:
        bucket, object_path = self.parse_minio_path(minio_path)
        client = self._build_client()

        try:
            response = client.get_object(bucket, object_path)
            data = response.read()
            response.close()
            response.release_conn()
        except Exception as exc:
            raise HTTPException(
                status_code=404,
                detail=f"Cannot read object '{object_path}' from bucket '{bucket}': {exc}",
            ) from exc

        if not data:
            raise HTTPException(status_code=400, detail="The MinIO object is empty")

        return data

    def read_text(self, minio_path: str) -> tuple[str, int]:
        data = self.read_bytes(minio_path)

        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HTTPException(
                status_code=400,
                detail="The MinIO object is not valid UTF-8 text",
            ) from exc

        return text, len(data)


minio_service = MinIOService()
