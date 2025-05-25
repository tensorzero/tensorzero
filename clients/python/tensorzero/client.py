# This imports from the native module
from .tensorzero import (
    AsyncTensorZeroGateway as AsyncTensorZeroGateway,
)
from .tensorzero import (
    BaseTensorZeroGateway as BaseTensorZeroGateway,
)
from .tensorzero import (
    TensorZeroGateway as TensorZeroGateway,
)
from .types import InferenceInput, PDFInput
import base64

class TensorZeroGateway:
    async def inference(self, input: InferenceInput) -> dict:
        payload = input.dict(exclude_none=True)
        if input.pdf and input.pdf.file_path:
            with open(input.pdf.file_path, "rb") as f:
                payload["pdf"]["base64_content"] = base64.b64encode(f.read()).decode("utf-8")
                payload["pdf"].pop("file_path")
        response = await self.http_client.post("/v1/inference", json=payload)
        return response.json()
